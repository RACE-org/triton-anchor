cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(triton_shared_core CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(TRITON_BUILD_PYTHON_MODULE "Build Python Triton bindings" ON)
option(TRITON_BUILD_WITH_CCACHE "Build with ccache (if available)" ON)
set(TRITON_PLUGIN_DIRS "" CACHE STRING
    "Optional semicolon-separated external plugin source directories")

if(TRITON_BUILD_WITH_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}"
        CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}"
        CACHE STRING "CXX compiler launcher")
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS -fPIC -std=gnu++17")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(NOT MLIR_DIR)
  set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()

find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

function(add_triton_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name}
    PRIVATE ${ARG_UNPARSED_ARGUMENTS}
    INTERFACE $<TARGET_OBJECTS:${name}>
  )
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction()

set_property(GLOBAL PROPERTY TRITON_LIBS "")
function(add_triton_library name)
  set_property(GLOBAL APPEND PROPERTY TRITON_LIBS ${name})
  add_triton_object(${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()

set_property(GLOBAL PROPERTY TRITON_PLUGINS "")
function(add_triton_plugin name)
  set_property(GLOBAL APPEND PROPERTY TRITON_PLUGINS ${name})
  add_triton_object(${name} ${ARGN})
endfunction()

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default -fvisibility=hidden")
else()
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} /wd4244 /wd4624 /wd4715 /wd4530")
endif()

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party)
include_directories(${PROJECT_BINARY_DIR}/third_party)
set(TRITON_ANCHOR_CSRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../csrc")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(third_party/f2reduce)
add_subdirectory(third_party/proton/Dialect)
add_subdirectory(${TRITON_ANCHOR_CSRC_DIR} ${CMAKE_CURRENT_BINARY_DIR}/csrc)

if(NOT WIN32 AND NOT APPLE AND NOT BSD)
  link_libraries(stdc++fs)
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
  include_directories(${PYTHON_SRC_PATH})

  find_package(Python3 REQUIRED COMPONENTS Development.Module Interpreter)
  find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")
  include_directories(${Python3_INCLUDE_DIRS})

  list(APPEND TRITON_PLUGIN_NAMES "triton_shared")

  triton_anchor_add_triton_shared_plugin()

  if(TRITON_PLUGIN_DIRS)
    foreach(PLUGIN_DIR ${TRITON_PLUGIN_DIRS})
      cmake_path(APPEND PLUGIN_DIR "backend" "name.conf"
                 OUTPUT_VARIABLE PLUGIN_NAME_PATH)
      file(READ ${PLUGIN_NAME_PATH} PLUGIN_NAME)
      string(STRIP ${PLUGIN_NAME} PLUGIN_NAME)

      list(APPEND TRITON_PLUGIN_NAMES ${PLUGIN_NAME})

      cmake_path(APPEND PROJECT_BINARY_DIR "third_party" ${PLUGIN_NAME}
                 OUTPUT_VARIABLE PLUGIN_DIR_BUILD_OUTPUT)
      add_subdirectory(${PLUGIN_DIR} ${PLUGIN_DIR_BUILD_OUTPUT})
    endforeach()
  endif()

  get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
  get_property(triton_plugins GLOBAL PROPERTY TRITON_PLUGINS)

  set(TRITON_LIBRARIES
    ${triton_libs}
    ${triton_plugins}
    MLIRAnalysis
    MLIRAMDGPUDialect
    MLIRArithDialect
    MLIRBuiltinToLLVMIRTranslation
    MLIRBytecodeWriter
    MLIRControlFlowDialect
    MLIRControlFlowToLLVM
    MLIRGPUDialect
    MLIRGPUToLLVMIRTranslation
    MLIRGPUToGPURuntimeTransforms
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    MLIRIR
    MLIRIndexToLLVM
    MLIRLLVMDialect
    MLIRLLVMIRTransforms
    MLIRLLVMToLLVMIRTranslation
    MLIRMathDialect
    MLIRMathToLLVM
    MLIRNVVMDialect
    MLIRNVVMToLLVMIRTranslation
    MLIRParser
    MLIRPass
    MLIRPtrDialect
    MLIRPtrToLLVMIRTranslation
    MLIRROCDLToLLVMIRTranslation
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIRUBDialect
    MLIRUBToLLVM
    LLVMPasses
    Python3::Module
    pybind11::headers
  )

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR
     CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR
     CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    list(APPEND TRITON_LIBRARIES
      LLVMAArch64AsmParser
      LLVMAArch64CodeGen
    )
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR
         CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    list(APPEND TRITON_LIBRARIES
      LLVMX86AsmParser
      LLVMX86CodeGen
    )
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le")
    list(APPEND TRITON_LIBRARIES
      LLVMPowerPCAsmParser
      LLVMPowerPCCodeGen
    )
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    list(APPEND TRITON_LIBRARIES
      LLVMRISCVAsmParser
      LLVMRISCVCodeGen
    )
  else()
    message(FATAL_ERROR
      "LLVM codegen/ASM parser libs: unsupported host architecture (${CMAKE_SYSTEM_PROCESSOR}).")
  endif()

  if(TRITON_PLUGIN_NAMES)
    string(JOIN "," TRITON_BACKENDS_TUPLE ${TRITON_PLUGIN_NAMES})
    set(TRITON_BACKENDS_TUPLE "(${TRITON_BACKENDS_TUPLE})")
    add_compile_definitions(
      TRITON_HAS_BACKENDS=1
      TRITON_BACKENDS_TUPLE=${TRITON_BACKENDS_TUPLE}
    )
  endif()

  add_library(triton SHARED
    ${PYTHON_SRC_PATH}/main.cc
    ${PYTHON_SRC_PATH}/ir.cc
    ${PYTHON_SRC_PATH}/gluon_ir.cc
    ${PYTHON_SRC_PATH}/linear_layout.cc
    ${PYTHON_SRC_PATH}/passes.cc
    ${PYTHON_SRC_PATH}/interpreter.cc
    ${PYTHON_SRC_PATH}/llvm.cc
    ${PYTHON_SRC_PATH}/specialize.cc
  )

  target_link_libraries(triton PRIVATE ${TRITON_LIBRARIES})
  if(WIN32)
    target_link_libraries(triton PRIVATE ${CMAKE_DL_LIBS})
    set_target_properties(triton PROPERTIES SUFFIX ".pyd")
    set_target_properties(triton PROPERTIES PREFIX "lib")
  else()
    target_link_libraries(triton PRIVATE z)
  endif()
  target_link_options(triton PRIVATE ${LLVM_LDFLAGS})
endif()
