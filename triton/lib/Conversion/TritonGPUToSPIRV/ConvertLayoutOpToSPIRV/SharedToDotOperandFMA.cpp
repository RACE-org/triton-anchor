#include "../ConvertLayoutOpToSPIRV.h"
#include "../Utility.h"

using ValueTable = std::map<std::pair<int, int>, Value>;
using ::mlir::spirv::getSharedMemoryObjectFromStruct;
using ::mlir::spirv::getStridesFromShapeAndOrder;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;
namespace SharedToDotOperandFMA {
// delinerize thread id into multiple dim thread id
SmallVector<Value> getThreadIds(Value threadId,
                                ArrayRef<unsigned int> shapePerCTATile,
                                ArrayRef<unsigned int> sizePerThread,
                                ArrayRef<unsigned int> order,
                                ConversionPatternRewriter &rewriter,
                                Location loc, bool singleBatch) {
  int dim = order.size();
  SmallVector<Value> threadIds(dim);
  for (unsigned k = 0; k < dim - 1; k++) {
    Value dimK = i32_val(shapePerCTATile[order[k]] / sizePerThread[order[k]]);
    Value rem = urem(threadId, dimK);
    threadId = udiv(threadId, dimK);
    threadIds[order[k]] = rem;
  }
  Value dimK = i32_val(shapePerCTATile[order[dim - 1]]);
  if (singleBatch) {
    dimK = i32_val(1);
  }
  threadIds[order[dim - 1]] = urem(threadId, dimK);
  return threadIds;
}

// Get shapePerCTATile for M or N axis.
int getShapePerCTATileForMN(BlockedEncodingAttr layout, bool isM,
                            int orderOffset) {
  auto order = layout.getOrder();
  auto shapePerCTATile = getShapePerCTATile(layout);

  int mShapePerCTATile = order[0] == (1 + orderOffset)
                             ? shapePerCTATile[order[1]]
                             : shapePerCTATile[order[0]];
  int nShapePerCTATile = order[0] == (0 + orderOffset)
                             ? shapePerCTATile[order[1]]
                             : shapePerCTATile[order[0]];
  return isM ? mShapePerCTATile : nShapePerCTATile;
}

// Get sizePerThread for M or N axis.
int getSizePerThreadForMN(BlockedEncodingAttr layout, bool isM,
                          int orderOffset) {
  auto order = layout.getOrder();
  auto sizePerThread = getSizePerThread(layout);

  int mSizePerThread = order[0] == (1 + orderOffset) ? sizePerThread[order[1]]
                                                     : sizePerThread[order[0]];
  int nSizePerThread = order[0] == (0 + orderOffset) ? sizePerThread[order[1]]
                                                     : sizePerThread[order[0]];
  return isM ? mSizePerThread : nSizePerThread;
}
} // namespace SharedToDotOperandFMA

Value getStructFromValueTable(ArrayRef<Value> vals,
                              ConversionPatternRewriter &rewriter, Location loc,
                              TritonGPUToSPIRVTypeConverter *typeConverter,
                              Type elemTy) {
  SmallVector<Type> elemTypes(vals.size(), elemTy);
  SmallVector<Value> elems;
  elems.reserve(vals.size());
  for (auto &val : vals) {
    elems.push_back(val);
  }
  MLIRContext *ctx = elemTy.getContext();
  Type structTy = struct_ty(elemTypes);
  return typeConverter->packLLElements(loc, elems, rewriter, structTy);
}

ValueTable getValueTableFromStruct(Value val, int K, int n0, int shapePerCTA,
                                   int sizePerThread,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc,
                                   TritonGPUToSPIRVTypeConverter *typeConverter,
                                   Type type) {
  ValueTable res;
  auto elems = typeConverter->unpackLLElements(loc, val, rewriter, type);
  int index = 0;
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned m = 0; m < n0; m += shapePerCTA)
      for (unsigned mm = 0; mm < sizePerThread; ++mm) {
        res[{m + mm, k}] = elems[index++];
      }
  }
  return res;
}

Value loadAFMA(Value A, Value llA, BlockedEncodingAttr dLayout, Value thread,
               Location loc, TritonGPUToSPIRVTypeConverter *typeConverter,
               ConversionPatternRewriter &rewriter) {
  auto aTensorTy = cast<MemDescType>(A.getType());
  auto aLayout = mlir::cast<SharedEncodingAttr>(aTensorTy.getEncoding());
  auto aShapePerCTA = getShapePerCTA(aTensorTy);
  ArrayRef<int64_t> aShape = aTensorTy.getShape();

  auto aOrder = aLayout.getOrder();
  auto order = dLayout.getOrder();
  bool is3D = aOrder.size() == 3;
  assert((!is3D || aOrder[2] == 0) &&
         "Unexpected rank of loadAFMA(shared->dotOp)");
  int orderOffset = aOrder.size() == 2 ? 0 : 1;

  bool isARow = aOrder[0] == (1 + orderOffset);

  auto aSmem = getSharedMemoryObjectFromStruct(loc, llA, rewriter);
  Value strideAM = aSmem.strides[0 + orderOffset];
  Value strideAK = aSmem.strides[1 + orderOffset];
  Value strideA0 = isARow ? strideAK : strideAM;
  Value strideA1 = isARow ? strideAM : strideAK;
  int aNumPtr = 8;
  int K = aShapePerCTA[1 + orderOffset];
  int M = aShapePerCTA[0 + orderOffset];

  auto shapePerCTATile = getShapePerCTATile(dLayout);
  auto sizePerThread = getSizePerThread(dLayout);

  Value _0 = i32_val(0);

  Value mContig = i32_val(sizePerThread[order[1]]);

  // threadId in blocked layout
  auto threadIds = SharedToDotOperandFMA::getThreadIds(
      thread, shapePerCTATile, sizePerThread, order, rewriter, loc,
      (is3D && aShape[0] == 1));
  Value threadIdB = is3D ? threadIds[0] : i32_val(0);
  Value threadIdM = threadIds[0 + orderOffset];

  // aOff is shared memory offset, it depends on offA0 and offA1
  // Make sure threadIdM isn't out of boundary
  threadIdM = urem(threadIdM, i32_val(M));

  Value offA0 = isARow ? _0 : mul(threadIdM, mContig);
  Value offA1 = isARow ? mul(threadIdM, mContig) : _0;
  SmallVector<Value> aOff(aNumPtr);
  for (int i = 0; i < aNumPtr; ++i) {
    aOff[i] = add(mul(offA0, strideA0), mul(offA1, strideA1));
  }
  auto elemTy = mlir::cast<MemDescType>(A.getType()).getElementType();

  Type ptrTy = ptr_ty(elemTy, spirv::StorageClass::Workgroup);
  SmallVector<Value> aPtrs(aNumPtr);
  for (int i = 0; i < aNumPtr; ++i)
    aPtrs[i] = gep(ptrTy, aSmem.base, aOff[i]);

  SmallVector<Value> vas;

  int mShapePerCTATile = SharedToDotOperandFMA::getShapePerCTATileForMN(
      dLayout, true /*isM*/, orderOffset);
  int mSizePerThread = SharedToDotOperandFMA::getSizePerThreadForMN(
      dLayout, true /*isM*/, orderOffset);

  auto warpsPerCTA = dLayout.getWarpsPerCTA();
  auto threadsPerWarp = dLayout.getThreadsPerWarp();
  int warpsPerBatch = is3D ? std::min<unsigned>(aShape[0], warpsPerCTA[0]) : 1;
  int batchPerThread =
      is3D
          ? std::max<int64_t>(1, aShape[0] / warpsPerCTA[0] / threadsPerWarp[0])
          : 1;
  // unsigned iWarpSize = triton::gpu::getWarpSize(dLayout);
  // assert(iWarpSize == 128);
  // Value waveSize = i32_val(iWarpSize);
  // Value linearWarpId = udiv(thread, waveSize);

  // Value warpIdInBatch = urem(linearWarpId, i32_val(warpsPerBatch));
  int operandSize = K * M;
  for (unsigned b = 0; b < batchPerThread; ++b) {
    Value batchOffset =
        mul(i32_val(operandSize),
            add(threadIdB, i32_val(b * warpsPerBatch * threadsPerWarp[0])));
    for (unsigned k = 0; k < K; ++k)
      for (unsigned m = 0; m < M; m += mShapePerCTATile)
        for (unsigned mm = 0; mm < mSizePerThread; ++mm) {
          Value offset =
              add(mul(i32_val(m + mm), strideAM), mul(i32_val(k), strideAK));
          if (is3D) {
            offset = add(offset, batchOffset);
          }
          Value pa = gep(ptrTy, aPtrs[0], offset);
          Value va = load(pa);
          vas.emplace_back(va);
        }
  }

  return getStructFromValueTable(vas, rewriter, loc, typeConverter, elemTy);
}

Value loadBFMA(Value B, Value llB, BlockedEncodingAttr dLayout, Value thread,
               Location loc, TritonGPUToSPIRVTypeConverter *typeConverter,
               ConversionPatternRewriter &rewriter) {
  auto bTensorTy = mlir::cast<MemDescType>(B.getType());
  auto bLayout = mlir::cast<SharedEncodingAttr>(bTensorTy.getEncoding());
  auto bShapePerCTA = getShapePerCTA(bTensorTy);
  ArrayRef<int64_t> bShape = bTensorTy.getShape();

  auto bOrder = bLayout.getOrder();
  auto order = dLayout.getOrder();
  bool is3D = bOrder.size() == 3;
  assert((!is3D || bOrder[2] == 0) &&
         "Unexpected rank of loadBFMA(shared->dotOp)");
  int orderOffset = bOrder.size() == 2 ? 0 : 1;

  bool isBRow = bOrder[0] == (1 + orderOffset);

  auto bSmem = getSharedMemoryObjectFromStruct(loc, llB, rewriter);
  Value strideBN = bSmem.strides[1 + orderOffset];
  Value strideBK = bSmem.strides[0 + orderOffset];
  Value strideB0 = isBRow ? strideBN : strideBK;
  Value strideB1 = isBRow ? strideBK : strideBN;
  int bNumPtr = 8;
  int K = bShapePerCTA[0 + orderOffset];
  int N = bShapePerCTA[1 + orderOffset];

  auto shapePerCTATile = getShapePerCTATile(dLayout);
  auto sizePerThread = getSizePerThread(dLayout);

  Value _0 = i32_val(0);

  Value nContig = i32_val(sizePerThread[order[0]]);

  // threadId in blocked layout
  auto threadIds = SharedToDotOperandFMA::getThreadIds(
      thread, shapePerCTATile, sizePerThread, order, rewriter, loc,
      (is3D && bShape[0] == 1));
  Value threadIdB = is3D ? threadIds[0] : i32_val(0);
  Value threadIdN = threadIds[1 + orderOffset];

  // bOff is shared memory offset, it depends on offB0 and offB1
  // Make sure threadIdN isn't out of boundary
  threadIdN = urem(threadIdN, i32_val(N));

  Value offB0 = isBRow ? mul(threadIdN, nContig) : _0;
  Value offB1 = isBRow ? _0 : mul(threadIdN, nContig);
  SmallVector<Value> bOff(bNumPtr);
  for (int i = 0; i < bNumPtr; ++i) {
    bOff[i] = add(mul(offB0, strideB0), mul(offB1, strideB1));
  }
  auto elemTy = mlir::cast<MemDescType>(B.getType()).getElementType();

  Type ptrTy = ptr_ty(elemTy, spirv::StorageClass::Workgroup);
  SmallVector<Value> bPtrs(bNumPtr);
  for (int i = 0; i < bNumPtr; ++i)
    bPtrs[i] = gep(ptrTy, bSmem.base, bOff[i]);

  SmallVector<Value> vbs;

  int nShapePerCTATile = SharedToDotOperandFMA::getShapePerCTATileForMN(
      dLayout, false /*isM*/, orderOffset);
  int nSizePerThread = SharedToDotOperandFMA::getSizePerThreadForMN(
      dLayout, false /*isM*/, orderOffset);

  auto warpsPerCTA = dLayout.getWarpsPerCTA();
  auto threadsPerWarp = dLayout.getThreadsPerWarp();
  int warpsPerBatch = is3D ? std::min<unsigned>(bShape[0], warpsPerCTA[0]) : 1;
  int batchPerThread =
      is3D
          ? std::max<int64_t>(1, bShape[0] / warpsPerCTA[0] / threadsPerWarp[0])
          : 1;
  // unsigned iWarpSize = triton::gpu::getWarpSize(dLayout);
  // assert(iWarpSize == 128);
  // Value waveSize = i32_val(iWarpSize);
  // Value linearWarpId = udiv(thread, waveSize);
  // Value warpIdInBatch = urem(linearWarpId, i32_val(warpsPerBatch));
  int operandSize = K * N;
  for (unsigned b = 0; b < batchPerThread; ++b) {
    Value batchOffset =
        mul(i32_val(operandSize),
            add(threadIdB, i32_val(b * warpsPerBatch * threadsPerWarp[0])));
    for (unsigned k = 0; k < K; ++k)
      for (unsigned n = 0; n < N; n += nShapePerCTATile)
        for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
          Value offset =
              add(mul(i32_val(n + nn), strideBN), mul(i32_val(k), strideBK));
          if (is3D) {
            offset = add(offset, batchOffset);
          }
          Value pb = gep(ptrTy, bPtrs[0], offset);
          Value vb = load(pb);
          vbs.emplace_back(vb);
        }
  }

  return getStructFromValueTable(vbs, rewriter, loc, typeConverter, elemTy);
}

namespace SharedToDotOperandFMA {
Value convertLayout(int opIdx, Value val, Value llVal,
                    BlockedEncodingAttr dLayout, Value thread, Location loc,
                    TritonGPUToSPIRVTypeConverter *typeConverter,
                    ConversionPatternRewriter &rewriter) {
  if (opIdx == 0)
    return loadAFMA(val, llVal, dLayout, thread, loc, typeConverter, rewriter);
  else
    return loadBFMA(val, llVal, dLayout, thread, loc, typeConverter, rewriter);
}
} // namespace SharedToDotOperandFMA
