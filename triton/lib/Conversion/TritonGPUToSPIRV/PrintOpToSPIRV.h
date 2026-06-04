#ifndef TRITON_CONVERSION_TRITONGPU_TO_SPIRV_PRINT_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_SPIRV_PRINT_OP_H

#include "TritonGPUToSPIRVBase.h"

using namespace mlir;
using namespace mlir::triton;

void populatePrintOpToSPIRVPattern(TritonGPUToSPIRVTypeConverter &typeConverter,
                                   mlir::MLIRContext *context,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit);

#endif