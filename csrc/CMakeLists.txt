set(TRITON_ANCHOR_CSRC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" CACHE INTERNAL "triton-anchor csrc source dir")
set(TRITON_ANCHOR_CSRC_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" CACHE INTERNAL "triton-anchor csrc binary dir")

include_directories(${TRITON_ANCHOR_CSRC_SOURCE_DIR}/include)
include_directories(${TRITON_ANCHOR_CSRC_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools/triton-shared-opt)

function(triton_anchor_add_spine_plugin)
  add_triton_plugin(SpineTriton
    ${TRITON_ANCHOR_CSRC_SOURCE_DIR}/triton_shared.cc

    LINK_LIBS
    TritonSharedAnalysis
    TritonToLinalg
    TritonTilingExtIR
    XSMTIR
    TLEIR
    TLEToLinalg
    ProtonIR
    MLIRPtrDialect
  )

  target_include_directories(SpineTriton PRIVATE
    ${TRITON_ANCHOR_CSRC_SOURCE_DIR}/include
    ${TRITON_ANCHOR_CSRC_BINARY_DIR}/include
  )

  target_link_libraries(SpineTriton PRIVATE
    Python3::Module
    pybind11::headers
    ${Python3_LIBRARIES}
  )

  add_dependencies(SpineTriton
    XSMTTableGen
    XSMTAsyncTableGen
    TritonStructuredTableGen
    TritonTilingExtOpsIncGen
    TritonTilingExtInterfacesIncGen
    TLETableGen
    TLEToLinalgConversionPassIncGen
  )
endfunction()
