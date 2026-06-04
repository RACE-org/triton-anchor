#include "ReshapeOpToSPIRV.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getTotalElemsPerThread;

struct SplatOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::SplatOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::SplatOp>::ConvertTritonGPUOpToSPIRVPattern;

  // Convert SplatOp or arith::ConstantOp with SplatElementsAttr to a
  // spirv::StructType value.
  //
  // @elemType: the element type in operand.
  // @resType: the return type of the Splat-like op.
  // @constVal: a spirv::ConstantOp or other scalar value.
  static Value convertSplatLikeOp(Type elemType, Type resType, Value constVal,
                                  TritonGPUToSPIRVTypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
    auto tensorTy = cast<RankedTensorType>(resType);
    auto srcType = typeConverter->convertType(elemType);
    auto spirvSrc = bitcast(constVal, srcType);
    size_t elemsPerThread = getTotalElemsPerThread(tensorTy);
    llvm::SmallVector<Value> elems(elemsPerThread, spirvSrc);
    return typeConverter->packLLElements(loc, elems, rewriter, resType);
  }

  LogicalResult matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto src = adaptor.getSrc();
    auto spirvStruct = convertSplatLikeOp(src.getType(), op.getType(), src,
                                          getTypeConverter(), rewriter, loc);
    rewriter.replaceOp(op, {spirvStruct});
    return success();
  }
};

// This pattern helps to convert arith::ConstantOp(with SplatElementsAttr),
// the logic is the same as triton::SplatOp, so the underlying implementation
// is reused.
struct ArithConstantSplatOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<arith::ConstantOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      arith::ConstantOp>::ConvertTritonGPUOpToSPIRVPattern;

  explicit ArithConstantSplatOpSPIRVConversion(
      TritonGPUToSPIRVTypeConverter &converter, MLIRContext *context,
      PatternBenefit benefit = 1, bool use_INTELConvertFToBF16Op = false)
      : ConvertTritonGPUOpToSPIRVPattern<arith::ConstantOp>(converter, context,
                                                            benefit),
        use_INTELConvertFToBF16Op(use_INTELConvertFToBF16Op) {}

  bool use_INTELConvertFToBF16Op = false;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();
    if (!dyn_cast<SplatElementsAttr>(value))
      return failure();

    auto loc = op->getLoc();

    auto values = dyn_cast<SplatElementsAttr>(op.getValue());
    auto elemType = values.getElementType();

    Attribute val;
    if (elemType.isBF16()) {
      // spirv::ConstantOp does not support bf16, Thus it needs special
      // treatment first.
      auto v = values.getValues<FloatAttr>()[0];
      auto lit_v = v.getValue();
      val = rewriter.getF32FloatAttr(lit_v.convertToFloat());
    } else if (type::isFloat(elemType)) {
      val = values.getValues<FloatAttr>()[0];
    } else if (type::isInt(elemType)) {
      val = values.getValues<IntegerAttr>()[0];
    } else {
      llvm::errs()
          << "ArithConstantSplatOpSPIRVConversion get unsupported type: "
          << value.getType() << "\n";
      return failure();
    }

    Value constOp;
    if (elemType.isBF16()) {
      // spirv::ConstantOp does not support bf16.
      constOp = rewriter.create<spirv::ConstantOp>(loc, f32_ty, val);
    } else {
      constOp = rewriter.create<spirv::ConstantOp>(loc, elemType, val);
    }

    if (elemType.isBF16()) {
      constOp = mlir::spirv::convertFp32ToBf16(loc, rewriter, constOp,
                                               use_INTELConvertFToBF16Op);
    }
    auto llStruct = SplatOpSPIRVConversion::convertSplatLikeOp(
        elemType, op.getType(), constOp, getTypeConverter(), rewriter, loc);
    rewriter.replaceOp(op, llStruct);

    return success();
  }
};

struct CatOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::CatOp> {
  using OpAdaptor = typename CatOp::Adaptor;

  explicit CatOpSPIRVConversion(TritonGPUToSPIRVTypeConverter &typeConverter,
                                MLIRContext *context,
                                PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToSPIRVPattern<triton::CatOp>(typeConverter, context,
                                                        benefit) {}

  LogicalResult
  matchAndRewrite(CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = cast<RankedTensorType>(op.getType());
    // unpack input values
    auto lhsVals = getTypeConverter()->unpackLLElements(
        loc, adaptor.getLhs(), rewriter, op.getOperand(0).getType());
    auto rhsVals = getTypeConverter()->unpackLLElements(
        loc, adaptor.getRhs(), rewriter, op.getOperand(1).getType());
    // concatenate (and potentially reorder) values
    SmallVector<Value> retVals;
    for (Value v : lhsVals)
      retVals.push_back(v);
    for (Value v : rhsVals)
      retVals.push_back(v);
    // pack and replace
    Value ret =
        getTypeConverter()->packLLElements(loc, retVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct SplitOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::SplitOp> {
  using OpAdaptor = typename SplitOp::Adaptor;

  explicit SplitOpSPIRVConversion(TritonGPUToSPIRVTypeConverter &typeConverter,
                                  MLIRContext *context,
                                  PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToSPIRVPattern<triton::SplitOp>(typeConverter,
                                                          context, benefit) {}

  LogicalResult
  matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We rely on the following invariants of this op (which are checked by its
    // verifier):
    //
    // - The op has a blocked encoding.
    // - The last dimension (the one we're spliting) is also the most minor
    //   dimension, and has sizePerThread=2.
    //   Pass TritonSplitOpPattern can make sure sizePerThread=2
    //
    // With these invariants, split is trivial: Every other value goes into
    // return value 0, and every other goes into return value 1.
    Location loc = op->getLoc();
    auto typeConverter = getTypeConverter();
    SmallVector<Value> srcVals = typeConverter->unpackLLElements(
        loc, adaptor.getSrc(), rewriter, op.getOperand().getType());
    assert(srcVals.size() % 2 == 0);
    SmallVector<Value> outLhsVals;
    SmallVector<Value> outRhsVals;
    for (int i = 0; i < srcVals.size(); i += 2) {
      outLhsVals.push_back(srcVals[i]);
      outRhsVals.push_back(srcVals[i + 1]);
    }
    auto resultTy = cast<RankedTensorType>(op.getResult(0).getType());
    Value retLhs =
        typeConverter->packLLElements(loc, outLhsVals, rewriter, resultTy);
    Value retRhs =
        typeConverter->packLLElements(loc, outRhsVals, rewriter, resultTy);
    rewriter.replaceOp(op, {retLhs, retRhs});
    return success();
  }
};

struct ReshapeOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<ReshapeOp> {
  using OpAdaptor = typename ReshapeOp::Adaptor;
  using ConvertTritonGPUOpToSPIRVPattern<
      ReshapeOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    if (!op.getAllowReorder()) {
      if (triton::gpu::isExpensiveView(op.getSrc().getType(), op.getType())) {
        return emitOptionalError(loc,
                                 "expensive view not supported on reshape op");
      }

      // Only support trivial block layouts for now.
      // reshape's src and dst layout cannot make sure it's equal to default
      // layout, because memory coaleasc pass will modify it
      // auto mod = op->getParentOfType<ModuleOp>();
      // int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
      // int threadsPerWarp =
      //     triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      // int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);

      // assert(resultTy.getEncoding() ==
      // triton::gpu::getDefaultBlockedEncoding(
      //                                      op.getContext(),
      //                                      resultTy.getShape(), numWarps,
      //                                      threadsPerWarp, numCTAs) &&
      //        "ReshapeOp lowering only support block encoding right now.");
      // assert(srcTy.getEncoding() == triton::gpu::getDefaultBlockedEncoding(
      //                                   op.getContext(), srcTy.getShape(),
      //                                   numWarps, threadsPerWarp, numCTAs) &&
      //        "ReshapeOp lowering only support block encoding right now.");
    }

    auto vals = this->getTypeConverter()->unpackLLElements(
        loc, adaptor.getSrc(), rewriter, op.getOperand().getType());
    Value ret =
        this->getTypeConverter()->packLLElements(loc, vals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct ExpandDimsOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<ExpandDimsOp> {
  using OpAdaptor = typename ExpandDimsOp::Adaptor;
  using ConvertTritonGPUOpToSPIRVPattern<
      ExpandDimsOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto srcVals = this->getTypeConverter()->unpackLLElements(
        loc, adaptor.getSrc(), rewriter, op.getOperand().getType());

    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    auto resultTy = cast<RankedTensorType>(op.getType());

    assert(isa<SliceEncodingAttr>(srcTy.getEncoding()) &&
           "ExpandDimsOp only support SliceEncodingAttr");
    auto srcLayout = dyn_cast<SliceEncodingAttr>(srcTy.getEncoding());
    auto resultLayout = resultTy.getEncoding();

    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    DenseMap<SmallVector<unsigned>, Value, SmallVectorKeyInfo> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }

    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      offset.erase(offset.begin() + srcLayout.getDim());
      resultVals.push_back(srcValues.lookup(offset));
    }
    Value ret = this->getTypeConverter()->packLLElements(loc, resultVals,
                                                         rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

struct TransOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::TransOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::TransOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = cast<TensorOrMemDesc>(op.getType());
    if (auto enc = dyn_cast<SharedEncodingAttr>(resultTy.getEncoding())) {
      auto srcSmemObj =
          getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(), rewriter);
      SmallVector<Value> dstStrides = {srcSmemObj.strides[1],
                                       srcSmemObj.strides[0]};
      SmallVector<Value> dstOffsets = {srcSmemObj.offsets[1],
                                       srcSmemObj.offsets[0]};
      auto dstSmemObj =
          SharedMemoryObject(srcSmemObj.base, dstStrides, dstOffsets);
      auto retVal = getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
      rewriter.replaceOp(op, retVal);
      return success();
    } else if (auto enc = mlir::dyn_cast<BlockedEncodingAttr>(
                   resultTy.getEncoding())) {
      // If the dst encoding is blocked, then TransOp::inferReturnTypes
      // ensures that:
      //  - the src encoding is also blocked, and
      //  - the translation from src to dst is just a "renaming" of the
      //    registers, i.e. each thread has exactly the same values.
      // Thus the transpose op simply returns the same values it got.
      auto vals = this->getTypeConverter()->unpackLLElements(
          loc, adaptor.getSrc(), rewriter, op.getOperand().getType());
      Value ret = this->getTypeConverter()->packLLElements(loc, vals, rewriter,
                                                           resultTy);
      rewriter.replaceOp(op, ret);
      return success();
    }
    return emitOptionalError(loc, "unsupported encoding for TransOp");
  }
};

struct JoinOpConversion : public ConvertTritonGPUOpToSPIRVPattern<JoinOp> {
  using OpAdaptor = typename JoinOp::Adaptor;
  explicit JoinOpConversion(TritonGPUToSPIRVTypeConverter &typeConverter,
                            MLIRContext *context, PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToSPIRVPattern<JoinOp>(typeConverter, context,
                                                 benefit) {}
  LogicalResult
  matchAndRewrite(JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We rely on the following invariants of this op (which are checked by its
    // verifier):
    //
    // - The op has a blocked encoding.
    // - The last dimension (the one we're joining) is also the most minor
    //   dimension.
    // - The input and output encodings are the same, except the output has
    //   2 elements per thread in the last dim.
    //
    // With these invariants, join is trivial: We just return the i'th element
    // from lhs, followed by the i'th elem from rhs.
    Location loc = op->getLoc();
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto typeConverter = getTypeConverter();
    SmallVector<Value> lhsVals = typeConverter->unpackLLElements(
        loc, adaptor.getLhs(), rewriter, op.getOperand(0).getType());
    SmallVector<Value> rhsVals = typeConverter->unpackLLElements(
        loc, adaptor.getRhs(), rewriter, op.getOperand(1).getType());
    assert(lhsVals.size() == rhsVals.size());
    SmallVector<Value> joinedVals;
    for (int i = 0; i < lhsVals.size(); i++) {
      joinedVals.push_back(lhsVals[i]);
      joinedVals.push_back(rhsVals[i]);
    }
    Value ret =
        typeConverter->packLLElements(loc, joinedVals, rewriter, resultTy);
    rewriter.replaceOp(op, ret);
    return success();
  }
};

void populateViewOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    mlir::RewritePatternSet &patterns, int numWarps,
    mlir::triton::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    mlir::ModuleAllocation *allocation, mlir::Value smem,
    mlir::PatternBenefit benefit,
    std::map<std::string, int> &computeCapability) {

  patterns.add<ReshapeOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<ExpandDimsOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<SplatOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<ArithConstantSplatOpSPIRVConversion>(
      typeConverter, context, benefit,
      mlir::spirv::checkOpSupported(computeCapability, "FantGPU"));
  patterns.add<CatOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<SplitOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<TransOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<JoinOpConversion>(typeConverter, context, benefit);
}
