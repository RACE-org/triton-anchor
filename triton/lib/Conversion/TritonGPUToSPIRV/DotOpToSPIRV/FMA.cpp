#include "../DotOpToSPIRV.h"
#include "../Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;

// using ValueTableFMA = std::map<std::pair<int, int>, Value>;
using ValueTableFMA = std::map<std::array<int, 3>, Value>;

static ValueTableFMA getValueTableFromStructFMA(
    Value val, int batchPerThread, int K, int n0, int shapePerCTATile,
    int sizePerThread, ConversionPatternRewriter &rewriter, Location loc,
    TritonGPUToSPIRVTypeConverter *typeConverter, Type type) {
  ValueTableFMA res;
  auto elems = typeConverter->unpackLLElements(loc, val, rewriter, type);
  int index = 0;
  for (int b = 0; b < batchPerThread; ++b) {
    for (int k = 0; k < K; ++k) {
      for (int m = 0; m < n0; m += shapePerCTATile)
        for (int mm = 0; mm < sizePerThread; ++mm) {
          res[{b, m + mm, k}] = elems[index++];
        }
    }
  }
  return res;
}

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToSPIRVTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto B = op.getB();
  auto C = op.getC();
  auto D = op.getResult();

  auto aTensorTy = mlir::cast<RankedTensorType>(A.getType());
  auto bTensorTy = mlir::cast<RankedTensorType>(B.getType());
  auto dTensorTy = mlir::cast<RankedTensorType>(D.getType());

  auto aShapePerCTA = getShapePerCTA(aTensorTy);
  auto bShapePerCTA = getShapePerCTA(bTensorTy);
  bool is3D = aShapePerCTA.size() == 3;

  BlockedEncodingAttr dLayout =
      mlir::cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
  auto order = dLayout.getOrder();
  auto cc =
      typeConverter->unpackLLElements(loc, adaptor.getC(), rewriter, dTensorTy);

  assert((!is3D || order[2] == 0) && "Unexpected rank of convertFMADot");
  int orderOffset = order.size() == 2 ? 0 : 1;

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread = getSizePerThread(dLayout);
  auto shapePerCTATile = getShapePerCTATile(dLayout);

  ArrayRef<int64_t> shape = dTensorTy.getShape();
  auto warpsPerCTA = dLayout.getWarpsPerCTA();
  auto threadsPerWarp = dLayout.getThreadsPerWarp();
  int batchPerThread =
      is3D ? std::max<int64_t>(1, shape[0] / warpsPerCTA[0] / threadsPerWarp[0])
           : 1;
  int K = aShapePerCTA[1 + orderOffset];
  int M = aShapePerCTA[0 + orderOffset];
  int N = bShapePerCTA[1 + orderOffset];

  int mShapePerCTATile = order[0] == (1 + orderOffset)
                             ? shapePerCTATile[order[1]]
                             : shapePerCTATile[order[0]];
  int mSizePerThread = order[0] == (1 + orderOffset) ? sizePerThread[order[1]]
                                                     : sizePerThread[order[0]];
  int nShapePerCTATile = order[0] == (0 + orderOffset)
                             ? shapePerCTATile[order[1]]
                             : shapePerCTATile[order[0]];
  int nSizePerThread = order[0] == (0 + orderOffset) ? sizePerThread[order[1]]
                                                     : sizePerThread[order[0]];

  auto has = getValueTableFromStructFMA(
      llA, batchPerThread, K, M, mShapePerCTATile, mSizePerThread, rewriter,
      loc, typeConverter, aTensorTy);
  auto hbs = getValueTableFromStructFMA(
      llB, batchPerThread, K, N, nShapePerCTATile, nSizePerThread, rewriter,
      loc, typeConverter, bTensorTy);

  SmallVector<Value> ret = cc;
  bool isCRow = order[0] == (1 + orderOffset);
  int batchPerOffset = M / mShapePerCTATile * N / nShapePerCTATile *
                       mSizePerThread * nSizePerThread;
  for (int b = 0; b < batchPerThread; b++) {
    for (int k = 0; k < K; k++) {
      for (int m = 0; m < M; m += mShapePerCTATile)
        for (int n = 0; n < N; n += nShapePerCTATile)
          for (int mm = 0; mm < mSizePerThread; ++mm)
            for (int nn = 0; nn < nSizePerThread; ++nn) {
              int mIdx = m / mShapePerCTATile * mSizePerThread + mm;
              int nIdx = n / nShapePerCTATile * nSizePerThread + nn;
              int z = isCRow
                          ? mIdx * N / nShapePerCTATile * mSizePerThread + nIdx
                          : nIdx * M / mShapePerCTATile * nSizePerThread + mIdx;
              z += b * batchPerOffset;
              auto valueA = has[{b, m + mm, k}];
              auto valueB = hbs[{b, n + nn, k}];
              if (valueA.getType() != f32_ty) {
                valueA = fpext(f32_ty, valueA);
              }
              if (valueB.getType() != f32_ty) {
                valueB = fpext(f32_ty, valueB);
              }
              ret[z] =
                  rewriter.create<spirv::CLFmaOp>(loc, valueA, valueB, ret[z]);
            }
    }
  }
  auto res = typeConverter->packLLElements(loc, ret, rewriter, dTensorTy);

  rewriter.replaceOp(op, res);

  return success();
}
