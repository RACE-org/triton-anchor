#ifndef TRITON_TRITONGPUTOSPIRV_H
#define TRITON_TRITONGPUTOSPIRV_H

#include "TritonGPUToSPIRVBase.h"
#include "triton/Analysis/AxisInfo.h"

void populateTritonGPUToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    mlir::triton::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    mlir::ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    mlir::PatternBenefit benefit);

#endif
