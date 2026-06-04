add_triton_library(TritonToTritonGPU
    TritonGPUConversion.cpp
    TritonToTritonGPUPass.cpp

    DEPENDS
    TritonConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    $<$<NOT:$<BOOL:${NO_TTGIR}>>:MLIRMathTransforms>
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
)
