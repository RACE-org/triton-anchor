#include "DotOpToSPIRV.h"
#include "Utility.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::DotOperandEncodingAttr;

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            TritonGPUToSPIRVTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);

struct DotOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::DotOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::DotOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    // or in the register directly (refer to case test_dot_without_load)
    auto AShape = mlir::cast<RankedTensorType>(A.getType()).getShape();
    size_t reduceAxis = 1;
    unsigned K = AShape[reduceAxis];
    bool isOuter = K == 1;
    // isOuter is for later use in case of TC in fantgpu

    // in the first phase, FantGpu only support normal FMA
    // so D must be Blocked layout, instead of MMA layout
    if (mlir::isa<BlockedEncodingAttr>(
            mlir::cast<RankedTensorType>(D.getType()).getEncoding()))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to SPIRV.");
  }
};

void populateDotOpToSPIRVPatterns(TritonGPUToSPIRVTypeConverter &typeConverter,
                                  mlir::MLIRContext *context,
                                  RewritePatternSet &patterns,
                                  ModuleAllocation &allocation,
                                  PatternBenefit benefit) {
  patterns.add<DotOpSPIRVConversion>(typeConverter, context, allocation,
                                     benefit);
}
