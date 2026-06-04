#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#ifndef NO_TTGIR
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#endif // NO_TTGIR

using ValueTable = std::map<std::pair<int, int>, Value>;
using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::LLVM::getStridesFromShapeAndOrder;
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
#ifndef NO_TTGIR
using ::mlir::triton::gpu::expandMatrixOrderWithBatch;
using ::mlir::triton::gpu::expandMatrixShapeWithBatch;
#endif // NO_TTGIR
using ::mlir::triton::gpu::getContigPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::isaDistributedLayout;
using ::mlir::triton::gpu::SharedEncodingAttr;

#ifdef NO_TTGIR
SmallVector<Value>
getThreadIds(Value threadId, ArrayRef<unsigned int> shapePerCTATile,
             ArrayRef<unsigned int> sizePerThread, ArrayRef<unsigned int> order,
             ConversionPatternRewriter &rewriter, Location loc) {
  int dim = order.size();
  SmallVector<Value> threadIds(dim);
  for (unsigned k = 0; k < dim - 1; k++) {
    Value dimK = i32_val(shapePerCTATile[order[k]] / sizePerThread[order[k]]);
    Value rem = urem(threadId, dimK);
    threadId = udiv(threadId, dimK);
    threadIds[order[k]] = rem;
  }
  Value dimK = i32_val(shapePerCTATile[order[dim - 1]]);
  threadIds[order[dim - 1]] = urem(threadId, dimK);
  return threadIds;
}

// Get shapePerCTATile for M or N axis.
int getShapePerCTATileForMN(BlockedEncodingAttr layout, bool isM) {
  auto order = layout.getOrder();
  auto shapePerCTATile = getShapePerCTATile(layout);

  int mShapePerCTATile =
      order[0] == 1 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  int nShapePerCTATile =
      order[0] == 0 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  return isM ? mShapePerCTATile : nShapePerCTATile;
}

// Get sizePerThread for M or N axis.
int getSizePerThreadForMN(BlockedEncodingAttr layout, bool isM) {
  auto order = layout.getOrder();
  auto sizePerThread = getSizePerThread(layout);

  int mSizePerThread =
      order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  int nSizePerThread =
      order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  return isM ? mSizePerThread : nSizePerThread;
}

#endif // NO_TTGIR
Value getStructFromValueTable(ArrayRef<Value> vals,
                              ConversionPatternRewriter &rewriter, Location loc,
                              const LLVMTypeConverter *typeConverter,
                              Type elemTy) {
  SmallVector<Type> elemTypes(vals.size(), elemTy);
  SmallVector<Value> elems;
  elems.reserve(vals.size());
  for (auto &val : vals) {
    elems.push_back(val);
  }
  MLIRContext *ctx = elemTy.getContext();
  Type structTy = struct_ty(elemTypes);
  return packLLElements(loc, typeConverter, elems, rewriter, structTy);
}

#ifndef NO_TTGIR
bool isSwizzled(SharedEncodingAttr layout) { return layout.getMaxPhase() != 1; }

SmallVector<Value> swizzleIndices(ConversionPatternRewriter &rewriter,
                                  Location loc, SmallVector<Value> rawIndices,
                                  SharedEncodingAttr layout) {
  const auto &order = layout.getOrder();
  auto rank = order.size();

  if (!isSwizzled(layout))
    return rawIndices;

  auto vec = i32_val(layout.getVec());
  auto perPhase = i32_val(layout.getPerPhase());
  auto maxPhase = i32_val(layout.getMaxPhase());

  auto fastIdx = rawIndices[order[0]];
  auto secondIdx = rawIndices[order[1]];
  // Original algorithm taken from getSwizzledSharedPtrs function
  // (TritonGPUToLLVMBase.h)
  //
  // phase = (secondIdx // perPhase) % maxPhase
  // swizzledGroup = ((fastIdx // vec) ^ phase) * vec
  // groupRemainder = fastIdx % vec
  // colOff = swizzledGroup + groupRemainder
  auto phase = urem(udiv(secondIdx, perPhase), maxPhase);
  auto swizzledGroup = mul(xor_(udiv(fastIdx, vec), phase), vec);
  auto groupRemainder = urem(fastIdx, vec);
  auto colOff = add(swizzledGroup, groupRemainder);

  SmallVector<Value> swizzledIndices = rawIndices;
  swizzledIndices[order[0]] = colOff;

  return swizzledIndices;
#else
ValueTable getValueTableFromStruct(Value val, int K, int n0, int shapePerCTA,
                                   int sizePerThread,
                                   ConversionPatternRewriter &rewriter,
                                   Location loc,
                                   const LLVMTypeConverter *typeConverter,
                                   Type type) {
  ValueTable res;
  auto elems = unpackLLElements(loc, val, rewriter);
  int index = 0;
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned m = 0; m < n0; m += shapePerCTA)
      for (unsigned mm = 0; mm < sizePerThread; ++mm) {
        res[{m + mm, k}] = elems[index++];
      }
  }
  return res;
#endif // NO_TTGIR
}

#ifndef NO_TTGIR
struct DimIdx {
  unsigned batch;
  unsigned k;
  unsigned nonK;
};
#else
Value loadAFMA(Value A, Value llA, BlockedEncodingAttr dLayout, Value thread,
               Location loc, const LLVMTypeConverter *typeConverter,
               ConversionPatternRewriter &rewriter) {
  auto aTensorTy = cast<MemDescType>(A.getType());
  auto aLayout = cast<SharedEncodingAttr>(aTensorTy.getEncoding());
  auto aShapePerCTA = getShapePerCTA(aTensorTy);
#endif // NO_TTGIR

#ifndef NO_TTGIR
/// Put elements from Value vec to appropriate indexes in opValues array.
///
/// This function maps elements of 3d sub-tensor in linear array.
/// Axes are arranged in an order provided "opOrder" argument
void storeValuesInLinearVector(PatternRewriter &rewriter, Location loc,
                               SmallVector<Value> &opValues, Value vec,
                               ArrayRef<unsigned> perThreadTileShape,
                               unsigned kIdx, unsigned nonKIdx, unsigned bIdx,
                               const DimIdx &dim, int vecDim,
                               ArrayRef<unsigned> opOrder) {
  auto vecTy = cast<VectorType>(vec.getType());
  auto vectorSize = vecTy.getNumElements();
  auto elemTy = vecTy.getElementType();
  for (int elem = 0; elem < vectorSize; ++elem) {
    unsigned spatialIdx[3] = {};
    spatialIdx[dim.batch] = bIdx;
    spatialIdx[dim.k] = kIdx;
    spatialIdx[dim.nonK] = nonKIdx;
    spatialIdx[vecDim] += elem;
#else
  auto aOrder = aLayout.getOrder();
  auto order = dLayout.getOrder();
#endif // NO_TTGIR

#ifndef NO_TTGIR
    unsigned linearIdx = linearize(spatialIdx, perThreadTileShape, opOrder);
    opValues[linearIdx] = extract_element(elemTy, vec, i32_val(elem));
#else
  bool isARow = aOrder[0] == 1;

  auto aSmem = getSharedMemoryObjectFromStruct(
      loc, llA, typeConverter->convertType(aTensorTy.getElementType()),
      rewriter);
  Value strideAM = aSmem.strides[0];
  Value strideAK = aSmem.strides[1];
  Value strideA0 = isARow ? strideAK : strideAM;
  Value strideA1 = isARow ? strideAM : strideAK;
  int aNumPtr = 8;
  int K = aShapePerCTA[1];
  int M = aShapePerCTA[0];

  auto shapePerCTATile = getShapePerCTATile(dLayout);
  auto sizePerThread = getSizePerThread(dLayout);

  Value _0 = i32_val(0);

  Value mContig = i32_val(sizePerThread[order[1]]);

  // threadId in blocked layout
  auto threadIds = getThreadIds(thread, shapePerCTATile, sizePerThread, order,
                                rewriter, loc);
  Value threadIdM = threadIds[0];

  Value offA0 = isARow ? _0 : mul(threadIdM, mContig);
  Value offA1 = isARow ? mul(threadIdM, mContig) : _0;
  SmallVector<Value> aOff(aNumPtr);
  for (int i = 0; i < aNumPtr; ++i) {
    aOff[i] = add(mul(offA0, strideA0), mul(offA1, strideA1));
#endif // NO_TTGIR
  }
#ifdef NO_TTGIR
  auto elemTy = typeConverter->convertType(aTensorTy.getElementType());

  Type ptrTy = ptr_ty(rewriter.getContext(), 3);
  SmallVector<Value> aPtrs(aNumPtr);
  for (int i = 0; i < aNumPtr; ++i)
    aPtrs[i] = gep(ptrTy, elemTy, aSmem.base, aOff[i]);

  SmallVector<Value> vas;

  int mShapePerCTATile = getShapePerCTATileForMN(dLayout, true /*isM*/);
  int mSizePerThread = getSizePerThreadForMN(dLayout, true /*isM*/);

  for (unsigned k = 0; k < K; ++k)
    for (unsigned m = 0; m < M; m += mShapePerCTATile)
      for (unsigned mm = 0; mm < mSizePerThread; ++mm) {
        Value offset =
            add(mul(i32_val(m + mm), strideAM), mul(i32_val(k), strideAK));
        Value pa = gep(ptrTy, elemTy, aPtrs[0], offset);
        Value va = load(elemTy, pa);
        vas.emplace_back(va);
      }

  return getStructFromValueTable(vas, rewriter, loc, typeConverter, elemTy);
#endif // NO_TTGIR
}

#ifndef NO_TTGIR
void verifyCTALayout(CTALayoutAttr ctaLayout) {
  auto ctaSplit = ctaLayout.getCTASplitNum();
  for (auto split : ctaSplit) {
    if (split != 1)
      llvm::report_fatal_error("tensors splited in CGA(thread group clusters) "
                               "are not supported in FMA dot yet.");
#else
Value loadBFMA(Value B, Value llB, BlockedEncodingAttr dLayout, Value thread,
               Location loc, const LLVMTypeConverter *typeConverter,
               ConversionPatternRewriter &rewriter) {
  auto bTensorTy = cast<MemDescType>(B.getType());
  auto bLayout = cast<SharedEncodingAttr>(bTensorTy.getEncoding());
  auto bShapePerCTA = getShapePerCTA(bTensorTy);

  auto bOrder = bLayout.getOrder();
  auto order = dLayout.getOrder();

  bool isBRow = bOrder[0] == 1;

  auto bSmem = getSharedMemoryObjectFromStruct(
      loc, llB, typeConverter->convertType(bTensorTy.getElementType()),
      rewriter);
  Value strideBN = bSmem.strides[1];
  Value strideBK = bSmem.strides[0];
  Value strideB0 = isBRow ? strideBN : strideBK;
  Value strideB1 = isBRow ? strideBK : strideBN;
  int bNumPtr = 8;
  int K = bShapePerCTA[0];
  int N = bShapePerCTA[1];

  auto shapePerCTATile = getShapePerCTATile(dLayout);
  auto sizePerThread = getSizePerThread(dLayout);

  Value _0 = i32_val(0);

  Value nContig = i32_val(sizePerThread[order[0]]);

  // threadId in blocked layout
  auto threadIds = getThreadIds(thread, shapePerCTATile, sizePerThread, order,
                                rewriter, loc);
  Value threadIdN = threadIds[1];

  Value offB0 = isBRow ? mul(threadIdN, nContig) : _0;
  Value offB1 = isBRow ? _0 : mul(threadIdN, nContig);
  SmallVector<Value> bOff(bNumPtr);
  for (int i = 0; i < bNumPtr; ++i) {
    bOff[i] = add(mul(offB0, strideB0), mul(offB1, strideB1));
#endif // NO_TTGIR
  }
#ifndef NO_TTGIR
}
#else
  auto elemTy = typeConverter->convertType(bTensorTy.getElementType());
#endif // NO_TTGIR

#ifndef NO_TTGIR
/// Get a linear offset of first element loaded by thread.
///
/// In unswizzled case offset of any element computed with formula:
/// smem.base + first_element_offset + constant_offset.
///
/// first_element_offset depends on lane Id and warp Id
/// constant_offset depends on value number, which is same for all threads.
/// \returns first_element_offset
Value getUnswizzledFirstElemOffset(ConversionPatternRewriter &rewriter,
                                   Location loc, unsigned B, unsigned NonK,
                                   Value bTileOffset, Value nonKTileOffset,
                                   Value bStride, Value nonKStride) {
  auto bOffset = mul(urem(bTileOffset, i32_val(B)), bStride);
  auto nonKOffset = mul(urem(nonKTileOffset, i32_val(NonK)), nonKStride);
  Value threadIdDependantOffset = add(bOffset, nonKOffset);
  return threadIdDependantOffset;
}
#else
  Type ptrTy = ptr_ty(rewriter.getContext(), 3);
  SmallVector<Value> bPtrs(bNumPtr);
  for (int i = 0; i < bNumPtr; ++i)
    bPtrs[i] = gep(ptrTy, elemTy, bSmem.base, bOff[i]);
#endif // NO_TTGIR

#ifndef NO_TTGIR
/// \returns number of elements stored by one thread across each dimension
SmallVector<unsigned> getElemsPerThreadInOp(ArrayRef<int64_t> opTensorShape,
                                            ArrayRef<unsigned> shapePerCTATile,
                                            ArrayRef<unsigned> sizePerThread) {
  int rank = opTensorShape.size();
  SmallVector<unsigned> elemsPerThread(rank);
  for (int d = 0; d < rank; ++d) {
    auto numReps =
        ceil(static_cast<unsigned>(opTensorShape[d]), shapePerCTATile[d]);
    elemsPerThread[d] = numReps * sizePerThread[d];
  }
  return elemsPerThread;
}
#else
  SmallVector<Value> vbs;
#endif // NO_TTGIR

#ifndef NO_TTGIR
struct Indexes {
  unsigned bTile;
  unsigned b;
  unsigned k;
  unsigned nonKTile;
  unsigned nonK;
};
#else
  int nShapePerCTATile = getShapePerCTATileForMN(dLayout, false /*isM*/);
  int nSizePerThread = getSizePerThreadForMN(dLayout, false /*isM*/);
#endif // NO_TTGIR

#ifndef NO_TTGIR
/// Computes a linear memory offset of a given element relative to
/// beginning of shared memory object.
Value computeSwizzledOffset(ConversionPatternRewriter &rewriter, Location loc,
                            const Indexes &i, const DimIdx &dim,
                            Value bTileOffset, Value nonKTileOffset,
                            unsigned shapePerCTABTile,
                            unsigned shapePerCTANonKTile,
                            SharedEncodingAttr sharedLayout,
                            ArrayRef<int64_t> opTensorShape,
                            ArrayRef<Value> strides) {
  Value offset = i32_val(0);
  // Compute unswizzled multi dim coordinates in shared memmory object
  SmallVector<Value> elemMultiDimIndices(3);
  elemMultiDimIndices[dim.batch] =
      add(bTileOffset, i32_val(i.bTile * shapePerCTABTile + i.b));
  elemMultiDimIndices[dim.nonK] =
      add(nonKTileOffset, i32_val(i.nonKTile * shapePerCTANonKTile + i.nonK));
  elemMultiDimIndices[dim.k] = i32_val(i.k);
#else
  for (unsigned k = 0; k < K; ++k)
    for (unsigned n = 0; n < N; n += nShapePerCTATile)
      for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
        Value offset =
            add(mul(i32_val(n + nn), strideBN), mul(i32_val(k), strideBK));
        Value pb = gep(ptrTy, elemTy, bPtrs[0], offset);
        Value vb = load(elemTy, pb);
        vbs.emplace_back(vb);
      }
#endif // NO_TTGIR

#ifndef NO_TTGIR
  // Apply swizzling pattern to fastest dimension
  SmallVector<Value> swizzledIndices =
      swizzleIndices(rewriter, loc, elemMultiDimIndices, sharedLayout);

  // Linearize shared mem object dimensions into flat offset
  for (int d = 0; d < 3; ++d) {
    // wrap index if it is larger than tensor
    auto wrappedDimIndex = urem(swizzledIndices[d], i32_val(opTensorShape[d]));
    auto dimOffset = mul(wrappedDimIndex, strides[d]);
    offset = add(offset, dimOffset);
  }
  return offset;
}

/// Computes memory offset of a given element relative to the
/// first element loaded by a thread.
Value computeNonSwizzledOffset(ConversionPatternRewriter &rewriter,
                               Location loc, const Indexes &i,
                               const DimIdx &dim, ArrayRef<int64_t> tensorShape,
                               unsigned shapePerCTABTile,
                               unsigned shapePerCTANonKTile,
                               ArrayRef<Value> strides) {
  SmallVector<Value> offsetIndices(3);
  offsetIndices[dim.batch] =
      i32_val((i.bTile * shapePerCTABTile + i.b) % tensorShape[dim.batch]);
  offsetIndices[dim.nonK] = i32_val(
      (i.nonKTile * shapePerCTANonKTile + i.nonK) % tensorShape[dim.nonK]);
  offsetIndices[dim.k] = i32_val(i.k);

  Value offset = i32_val(0);
  for (int d = 0; d < 3; ++d)
    offset = add(offset, mul(offsetIndices[d], strides[d]));
  return offset;
}

/// Generates llvm IR for loading FMA dot operand from shared memory.
///
/// \param srcVal triton_gpu MemDescType value
/// \param llVal llvm IR values corresponding to srcVal
/// \param dLayout parent dot operand layout
/// \param thread thread id
/// \param loc
/// \param typeConverter
/// \param rewriter
/// \param dotOpNo
/// \returns llvm value with loaded elements
Value loadFMAOp(Value srcVal, Value llVal, BlockedEncodingAttr dLayout,
                Value thread, Location loc,
                const LLVMTypeConverter *typeConverter,
                ConversionPatternRewriter &rewriter, const int dotOpNo) {
  verifyCTALayout(dLayout.getCTALayout());

  DimIdx dim;
  dim.batch = 0;
  dim.k = dotOpNo == 0 ? 2 : 1;
  dim.nonK = dotOpNo == 0 ? 1 : 2;
  auto opTensorTy = cast<MemDescType>(srcVal.getType());
  auto opTensorShape = expandMatrixShapeWithBatch(opTensorTy.getShape());
  auto sharedLayout = cast<SharedEncodingAttr>(opTensorTy.getEncoding());

  auto opOrder = expandMatrixOrderWithBatch(dLayout.getOrder());

  auto origSmem = getSharedMemoryObjectFromStruct(
      loc, llVal, typeConverter->convertType(opTensorTy.getElementType()),
      rewriter);
  auto smem = getExpandedSharedMemoryObject(rewriter, loc, origSmem,
                                            opTensorTy.getShape());
  auto strides = smem.strides;
  int B = opTensorShape[dim.batch];
  int K = opTensorShape[dim.k];
  int NonK = opTensorShape[dim.nonK];

  auto shapePerCTATile =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTATile(dLayout)));
  shapePerCTATile[dim.k] = K;
  auto sizePerThread =
      expandMatrixShapeWithBatch(ArrayRef(getSizePerThread(dLayout)));
  sizePerThread[dim.k] = K;
  auto threadsPerWarp =
      expandMatrixShapeWithBatch(ArrayRef(dLayout.getThreadsPerWarp()));
  auto warpsPerCTA =
      expandMatrixShapeWithBatch(ArrayRef(dLayout.getWarpsPerCTA()));

  auto warpSize = i32_val(triton::gpu::getWarpSize(dLayout));
  auto laneId = urem(thread, warpSize);
  auto warpId = udiv(thread, warpSize);
  auto laneIds =
      mlir::LLVM::delinearize(rewriter, loc, laneId, threadsPerWarp, opOrder);
  auto warpIds =
      mlir::LLVM::delinearize(rewriter, loc, warpId, warpsPerCTA, opOrder);
  auto sizePerWarpB = sizePerThread[dim.batch] * threadsPerWarp[dim.batch];
  auto sizePerWarpNonK = sizePerThread[dim.nonK] * threadsPerWarp[dim.nonK];

  Value bTileOffset =
      mul(laneIds[dim.batch], i32_val(sizePerThread[dim.batch]));
  bTileOffset =
      add(bTileOffset, mul(warpIds[dim.batch], i32_val(sizePerWarpB)));
  Value nonKTileOffset =
      mul(laneIds[dim.nonK], i32_val(sizePerThread[dim.nonK]));
  nonKTileOffset =
      add(nonKTileOffset, mul(warpIds[dim.nonK], i32_val(sizePerWarpNonK)));

  auto elemTy = typeConverter->convertType(opTensorTy.getElementType());
  Type ptrTy = smem.base.getType();

  auto sharedOrder = expandMatrixOrderWithBatch(sharedLayout.getOrder());
  // compute contiguity of fastest dimension in shared layout.
  unsigned vectorSize = sizePerThread[sharedOrder[0]];
  vectorSize = std::min(vectorSize, 128 / elemTy.getIntOrFloatBitWidth());

  bool swizzlePath = isSwizzled(sharedLayout);

  if (swizzlePath)
    vectorSize = std::min(vectorSize, sharedLayout.getVec());
  auto vecTy = vec_ty(elemTy, vectorSize);
  // loop increments depend on fastest dim
  unsigned dimStep[3] = {1, 1, 1};
  dimStep[sharedOrder[0]] = vectorSize;

  auto shapePerCTABTile = shapePerCTATile[dim.batch];
  auto shapePerCTANonKTile = shapePerCTATile[dim.nonK];
  auto sizeBPerThread = sizePerThread[dim.batch];
  auto sizeNonKPerThread = sizePerThread[dim.nonK];
  auto numBTiles = std::max(1u, B / shapePerCTABTile);
  auto numNonKTiles = std::max(1u, NonK / shapePerCTANonKTile);

  auto perThreadShape =
      getElemsPerThreadInOp(opTensorShape, shapePerCTATile, sizePerThread);

  SmallVector<Value> opValues(numBTiles * sizeBPerThread * K * numNonKTiles *
                              sizeNonKPerThread);

  // In swizzled memory case basePtr stores pointer to the beginning of shared
  // memmory object.
  //
  // If memory is not swizzled, algorithm breaks element offset pointer into
  // constant and non-constant part. Non-constant (depends on thread id) part is
  // the offset of the first element of the thread, which is same for all
  // elements of the thread. It is computed only once. basePtr stores this
  // non-constant part
  Value basePtr;
  if (swizzlePath) {
    basePtr = smem.base;
  } else {
    auto laneOffset = getUnswizzledFirstElemOffset(
        rewriter, loc, B, NonK, bTileOffset, nonKTileOffset, strides[dim.batch],
        strides[dim.nonK]);
    basePtr = gep(ptrTy, elemTy, smem.base, laneOffset);
  }
  // This loop nest iterates over all values loaded in one thread across batch,
  // k and nonK dimensions. Blocked dot operand layout allocates data in tiles
  // of size <sizePerThread>*<threadsPerWarp>*<numberWarps> for batch and nonK
  // dimensions. If tensor shape is larger than tile, pattern repeats. To take
  // these repeats into account iterations for batch and nonK are split into
  // "intra tile" + "inter tile" indexes: b + bTile, nonK + nonKTile
  for (unsigned bTile = 0; bTile < numBTiles; ++bTile)
    for (unsigned b = 0; b < sizeBPerThread; b += dimStep[dim.batch])
      for (unsigned k = 0; k < K; k += dimStep[dim.k])
        for (unsigned nonKTile = 0; nonKTile < numNonKTiles; ++nonKTile)
          for (unsigned nonK = 0; nonK < sizeNonKPerThread;
               nonK += dimStep[dim.nonK]) {
            Value offset = i32_val(0);
            Indexes idx = {bTile, b, k, nonKTile, nonK};

            // swizzled variant is more general, but it limits optimization of
            // address computation,
            if (swizzlePath) {
              offset = computeSwizzledOffset(
                  rewriter, loc, idx, dim, bTileOffset, nonKTileOffset,
                  shapePerCTABTile, shapePerCTANonKTile, sharedLayout,
                  opTensorShape, strides);
            } else {
              offset = computeNonSwizzledOffset(rewriter, loc, idx, dim,
                                                opTensorShape, shapePerCTABTile,
                                                shapePerCTANonKTile, strides);
            }

            Value elemAddr = gep(ptrTy, elemTy, basePtr, offset);
            Value vec = load(vecTy, elemAddr);
            storeValuesInLinearVector(
                rewriter, loc, opValues, vec, perThreadShape, /*kIdx*/ k,
                /*nonKIdx*/ nonKTile * sizeNonKPerThread + nonK,
                /*bIdx*/ bTile * sizeBPerThread + b, dim, sharedOrder[0],
                opOrder);
          }

  return getStructFromValueTable(opValues, rewriter, loc, typeConverter,
                                 elemTy);
#else
  return getStructFromValueTable(vbs, rewriter, loc, typeConverter, elemTy);
#endif // NO_TTGIR
}

namespace SharedToDotOperandFMA {
Value convertLayout(int opIdx, Value val, Value llVal,
                    BlockedEncodingAttr dLayout, Value thread, Location loc,
                    const LLVMTypeConverter *typeConverter,
                    ConversionPatternRewriter &rewriter) {
#ifndef NO_TTGIR
  return loadFMAOp(val, llVal, dLayout, thread, loc, typeConverter, rewriter,
                   opIdx);
#else
  if (opIdx == 0)
    return loadAFMA(val, llVal, dLayout, thread, loc, typeConverter, rewriter);
  else
    return loadBFMA(val, llVal, dLayout, thread, loc, typeConverter, rewriter);
#endif // NO_TTGIR
}
} // namespace SharedToDotOperandFMA
