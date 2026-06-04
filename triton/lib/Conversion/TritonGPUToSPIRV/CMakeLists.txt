add_mlir_conversion_library(TritonGPUToSPIRV
    TritonGPUToSPIRV.cpp
    ElementwiseOpToSPIRV.cpp
    TritonGPUToSPIRVPass.cpp
    LoadStoreOpToSPIRV.cpp
    ConvertLayoutOpToSPIRV.cpp
    PrintOpToSPIRV.cpp
    ReshapeOpToSPIRV.cpp
    ReduceOpToSPIRV.cpp
    ScanOpToSPIRV.cpp
    Utility.cpp
    TypeConverter.cpp
    HistogramOpToSPIRV.cpp
    ConvertLayoutOpToSPIRV/SharedToDotOperandFMA.cpp
    DotOpToSPIRV/FMA.cpp
    DotOpToSPIRV.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonGPUToSPIRV

    DEPENDS
    TritonGPUToSPIRVConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
)
