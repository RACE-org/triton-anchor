#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#ifndef NO_TTGIR
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#endif // NO_TTGIR

using namespace mlir;
using namespace mlir::triton;
#ifndef NO_TTGIR
using namespace ::mlir::triton::gpu;
#endif // NO_TTGIR

#ifndef NO_TTGIR
using ::mlir::LLVM::linearize;
using ::mlir::triton::gpu::expandMatrixOrderWithBatch;
using ::mlir::triton::gpu::expandMatrixShapeWithBatch;
#else
using ::mlir::triton::gpu::DotOperandEncodingAttr;
#endif // NO_TTGIR
using ::mlir::triton::gpu::getShapePerCTA;
#ifndef NO_TTGIR
using ::mlir::triton::gpu::getSizePerThread;
#else
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;

using ValueTableFMA = std::map<std::pair<int, int>, Value>;
#endif // NO_TTGIR

#ifndef NO_TTGIR
using ValueTableFMA = std::map<std::tuple<int, int, int>, Value>;
#endif // NO_TTGIR
static ValueTableFMA
#ifndef NO_TTGIR
getValueTableFromStructFMA(Value val, ArrayRef<unsigned> perTileShape,
                           unsigned kDim, unsigned nonKDim,
#else
getValueTableFromStructFMA(Value val, int K, int n0, int shapePerCTATile,
                           int sizePerThread,
#endif // NO_TTGIR
                           ConversionPatternRewriter &rewriter, Location loc,
#ifndef NO_TTGIR
                           ArrayRef<unsigned> order) {
#else
                           const LLVMTypeConverter *typeConverter, Type type) {
#endif // NO_TTGIR
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
#ifndef NO_TTGIR
  assert(perTileShape.size() == 3);
  assert(elems.size() == product(perTileShape));
  assert(kDim == 1 || kDim == 2);
  assert(nonKDim == 1 || nonKDim == 2);
  const unsigned bDim = 0;

  for (unsigned idx = 0; idx < elems.size(); ++idx) {
    auto spatialIdx = mlir::LLVM::delinearize(idx, perTileShape, order);
    res[{spatialIdx[bDim], spatialIdx[nonKDim], spatialIdx[kDim]}] = elems[idx];
#else
  int index = 0;
  for (unsigned k = 0; k < K; ++k) {
    for (unsigned m = 0; m < n0; m += shapePerCTATile)
      for (unsigned mm = 0; mm < sizePerThread; ++mm) {
        res[{m + mm, k}] = elems[index++];
      }
#endif // NO_TTGIR
  }
  return res;
}

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
#ifdef NO_TTGIR
  auto B = op.getB();
  auto C = op.getC();
#endif // NO_TTGIR
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
#ifdef NO_TTGIR
  auto bTensorTy = cast<RankedTensorType>(B.getType());
#endif // NO_TTGIR
  auto dTensorTy = cast<RankedTensorType>(D.getType());

#ifndef NO_TTGIR
  SmallVector<int64_t> aShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(aTensorTy)));
  auto dShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(dTensorTy)));
#else
  auto aShapePerCTA = getShapePerCTA(aTensorTy);
  auto bShapePerCTA = getShapePerCTA(bTensorTy);
#endif // NO_TTGIR

  BlockedEncodingAttr dLayout =
      cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
#ifndef NO_TTGIR
  auto order = expandMatrixOrderWithBatch(dLayout.getOrder());
#else
  auto order = dLayout.getOrder();
#endif // NO_TTGIR
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

#ifndef NO_TTGIR
  auto sizePerThread =
      expandMatrixShapeWithBatch(ArrayRef(getSizePerThread(dLayout)));
  auto shapePerCTATile =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTATile(dLayout)));
#else
  auto sizePerThread = getSizePerThread(dLayout);
  auto shapePerCTATile = getShapePerCTATile(dLayout);
#endif // NO_TTGIR

#ifndef NO_TTGIR
  unsigned K = aShapePerCTA[2];
#else
  int K = aShapePerCTA[1];
  int M = aShapePerCTA[0];
  int N = bShapePerCTA[1];
#endif // NO_TTGIR

#ifndef NO_TTGIR
  unsigned perThreadShape[3];
  for (int i = 0; i < 3; ++i) {
    unsigned numRep = dShapePerCTA[i] / shapePerCTATile[i];
    numRep = std::max(static_cast<unsigned>(1), numRep);
    perThreadShape[i] = numRep * sizePerThread[i];
#else
  int mShapePerCTATile =
      order[0] == 1 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  int mSizePerThread =
      order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
  int nShapePerCTATile =
      order[0] == 0 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
  int nSizePerThread =
      order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];

  auto has =
      getValueTableFromStructFMA(llA, K, M, mShapePerCTATile, mSizePerThread,
                                 rewriter, loc, typeConverter, aTensorTy);
  auto hbs =
      getValueTableFromStructFMA(llB, K, N, nShapePerCTATile, nSizePerThread,
                                 rewriter, loc, typeConverter, bTensorTy);

  SmallVector<Value> ret = cc;
  bool isCRow = order[0] == 1;

  for (unsigned k = 0; k < K; k++) {
    for (unsigned m = 0; m < M; m += mShapePerCTATile)
      for (unsigned n = 0; n < N; n += nShapePerCTATile)
        for (unsigned mm = 0; mm < mSizePerThread; ++mm)
          for (unsigned nn = 0; nn < nSizePerThread; ++nn) {
            int mIdx = m / mShapePerCTATile * mSizePerThread + mm;
            int nIdx = n / nShapePerCTATile * nSizePerThread + nn;

            int z = isCRow
                        ? mIdx * N / nShapePerCTATile * mSizePerThread + nIdx
                        : nIdx * M / mShapePerCTATile * nSizePerThread + mIdx;
            ret[z] = rewriter.create<LLVM::FMulAddOp>(loc, has[{m + mm, k}],
                                                      hbs[{n + nn, k}], ret[z]);
          }
#endif // NO_TTGIR
  }

#ifndef NO_TTGIR
  auto has = getValueTableFromStructFMA(
      llA, {perThreadShape[0], perThreadShape[1], K},
      /*kDim*/ 2, /*nonKDim*/ 1, rewriter, loc, order);
  auto hbs = getValueTableFromStructFMA(
      llB, {perThreadShape[0], K, perThreadShape[2]},
      /*kDim*/ 1, /*nonKDim*/ 2, rewriter, loc, order);

  SmallVector<Value> acc = cc;

  for (unsigned b = 0; b < perThreadShape[0]; ++b)
    for (unsigned m = 0; m < perThreadShape[1]; ++m)
      for (unsigned n = 0; n < perThreadShape[2]; ++n) {
        SmallVector<unsigned> multiDimAccumIdx = {b, m, n};
        unsigned linearAccumIdx =
            linearize(multiDimAccumIdx, perThreadShape, order);
        for (unsigned k = 0; k < K; ++k) {
          auto valueA = has[{b, m, k}];
          auto valueB = hbs[{b, n, k}];
          if (valueA.getType() != f32_ty) {
            valueA = fpext(f32_ty, valueA);
          }
          if (valueB.getType() != f32_ty) {
            valueB = fpext(f32_ty, valueB);
          }
          acc[linearAccumIdx] = rewriter.create<LLVM::FMulAddOp>(
              loc, valueA, valueB, acc[linearAccumIdx]);
        }
      }

  auto res = packLLElements(loc, typeConverter, acc, rewriter, dTensorTy);
#else
  auto res = packLLElements(loc, typeConverter, ret, rewriter, dTensorTy);
#endif // NO_TTGIR
  rewriter.replaceOp(op, res);

  return success();
}
