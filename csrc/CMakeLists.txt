set(TRITON_ANCHOR_CSRC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "triton-anchor csrc source dir")
set(TRITON_ANCHOR_CSRC_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "triton-anchor csrc binary dir")

include_directories(${TRITON_ANCHOR_CSRC_SOURCE_DIR}/include)
include_directories(${TRITON_ANCHOR_CSRC_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools/triton-shared-opt)

function(triton_anchor_add_triton_shared_plugin)
  add_triton_plugin(TritonSharedPlugin
    ${TRITON_ANCHOR_CSRC_SOURCE_DIR}/triton_shared.cc
    ${TRITON_ANCHOR_CSRC_SOURCE_DIR}/bindings/triton_anchor_ext.cc
    ${TRITON_ANCHOR_CSRC_SOURCE_DIR}/bindings/triton_anchor_validator.cc
    ${TRITON_ANCHOR_CSRC_SOURCE_DIR}/lib/Validation/AnchorIRValidator.cpp

    LINK_LIBS
    TritonSharedAnalysis
    TritonToLinalg
    TritonTilingExtIR
    XSMTIR
    TLEIR
    TLEToLinalg
    ProtonIR
    MLIRPtrDialect
    MLIRAffineDialect
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRGPUDialect
    MLIRIndexDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRNVGPUDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRVectorDialect
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTransforms
  )

  target_include_directories(TritonSharedPlugin PRIVATE
    ${TRITON_ANCHOR_CSRC_SOURCE_DIR}/include
    ${TRITON_ANCHOR_CSRC_BINARY_DIR}/include
  )

  target_compile_definitions(TritonSharedPlugin PRIVATE
    TRITON_ANCHOR_VALIDATION_EXPORTS
  )

  target_link_libraries(TritonSharedPlugin PRIVATE
    Python3::Module
    pybind11::headers
    ${Python3_LIBRARIES}
  )

  add_dependencies(TritonSharedPlugin
    XSMTTableGen
    XSMTAsyncTableGen
    TritonStructuredTableGen
    TritonTilingExtOpsIncGen
    TritonTilingExtInterfacesIncGen
    TLETableGen
    TLEToLinalgConversionPassIncGen
  )
endfunction()
