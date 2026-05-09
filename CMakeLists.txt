cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(triton-anchor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# ═══════════════════════════════════════════════════════════════════════
# LLVM / MLIR 查找
# ═══════════════════════════════════════════════════════════════════════
if(NOT MLIR_DIR)
  if(LLVM_LIBRARY_DIR)
    set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
  endif()
endif()

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Pybind11
find_package(pybind11 REQUIRED)

# ═══════════════════════════════════════════════════════════════════════
# 编译器标志（与上游 Triton 保持一致）
# ═══════════════════════════════════════════════════════════════════════
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__STDC_FORMAT_MACROS -fPIC -std=gnu++17")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default -fvisibility=hidden")

set(CMAKE_C_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")
set(CMAKE_CXX_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT WIN32 AND NOT APPLE)
  link_libraries(stdc++fs)
endif()

if(UNIX AND NOT APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
endif()

# ═══════════════════════════════════════════════════════════════════════
# 路径定义
# ═══════════════════════════════════════════════════════════════════════
set(UPSTREAM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/upstream")
set(TRITON_SOURCE_DIR "${UPSTREAM_DIR}")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/upstream")

# ═══════════════════════════════════════════════════════════════════════
# include 路径
# ═══════════════════════════════════════════════════════════════════════
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
# 上游 Triton 头文件
include_directories(${UPSTREAM_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/upstream/include)  # TableGen 生成
include_directories(${UPSTREAM_DIR})                                # third_party 引用
include_directories(${CMAKE_CURRENT_BINARY_DIR}/upstream)           # third_party TableGen 生成
# triton-anchor 自有头文件
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/csrc/include)       # triton-anchor TableGen 生成

# ═══════════════════════════════════════════════════════════════════════
# Triton 核心构建宏（从上游摘取并简化）
# ═══════════════════════════════════════════════════════════════════════
set_property(GLOBAL PROPERTY TRITON_LIBS "")

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

function(add_triton_library name)
  set_property(GLOBAL APPEND PROPERTY TRITON_LIBS ${name})
  add_triton_object(${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()

# ═══════════════════════════════════════════════════════════════════════
# Part 1: 编译上游 Triton 核心 C++ 库
# ═══════════════════════════════════════════════════════════════════════
# TableGen 生成的头文件（Dialect 定义）
add_subdirectory(upstream/include)
# C++ 库实现
add_subdirectory(upstream/lib)
# f2reduce 辅助库
add_subdirectory(upstream/third_party/f2reduce)

# 在 csrc 加入之前捕获上游库列表，防止 triton-anchor 的 Pass 被混入 libtriton.so
get_property(TRITON_UPSTREAM_LIBS GLOBAL PROPERTY TRITON_LIBS)

# ═══════════════════════════════════════════════════════════════════════
# Part 2: 编译 triton-anchor 自有 C++ Pass（triton-linalg 等）
# ═══════════════════════════════════════════════════════════════════════
add_subdirectory(csrc)

# ═══════════════════════════════════════════════════════════════════════
# Part 3: 构建 libtriton.so（上游 Triton 的 Python C++ 绑定）
# ═══════════════════════════════════════════════════════════════════════
if(TRITON_BUILD_PYTHON_MODULE)
  message(STATUS "Adding Python module (libtriton.so)")

  set(PYTHON_SRC_PATH ${UPSTREAM_DIR}/python/src)
  include_directories(${PYTHON_SRC_PATH})

  if(PYTHON_INCLUDE_DIRS)
    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${PYBIND11_INCLUDE_DIR})
  else()
    find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
    include_directories(${Python3_INCLUDE_DIRS})
    include_directories(${pybind11_INCLUDE_DIR})
    link_directories(${Python3_LIBRARY_DIRS})
    link_libraries(${Python3_LIBRARIES})
    add_link_options(${Python3_LINK_OPTIONS})
  endif()

  # 后端标识（空 = 无硬件后端，由 triton-anchor 插件机制代替）
  set(TRITON_BACKENDS_TUPLE "()")
  add_compile_definitions(TRITON_BACKENDS_TUPLE=${TRITON_BACKENDS_TUPLE})

  add_library(triton SHARED
    ${PYTHON_SRC_PATH}/main.cc
    ${PYTHON_SRC_PATH}/ir.cc
    ${PYTHON_SRC_PATH}/passes.cc
    ${PYTHON_SRC_PATH}/interpreter.cc
    ${PYTHON_SRC_PATH}/llvm.cc
  )

  set(TRITON_LIBRARIES
    ${TRITON_UPSTREAM_LIBS}

    # MLIR 核心
    MLIRAMDGPUDialect
    MLIRNVVMDialect
    MLIRNVVMToLLVMIRTranslation
    MLIRGPUToNVVMTransforms
    MLIRGPUToGPURuntimeTransforms
    MLIRGPUTransforms
    MLIRIR
    MLIRControlFlowToLLVM
    MLIRBytecodeWriter
    MLIRPass
    MLIRTransforms
    MLIRLLVMDialect
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRMathToLLVM
    MLIRROCDLToLLVMIRTranslation
    MLIRGPUDialect
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRIndexToLLVM
    MLIRGPUToROCDLTransforms

    # LLVM
    LLVMPasses
    LLVMNVPTXCodeGen
    LLVMAMDGPUCodeGen
    LLVMAMDGPUAsmParser
  )

  # 按架构选择指令后端
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    list(APPEND TRITON_LIBRARIES LLVMAArch64CodeGen LLVMAArch64AsmParser)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    list(APPEND TRITON_LIBRARIES LLVMX86CodeGen LLVMX86AsmParser)
  else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  target_link_libraries(triton PUBLIC ${TRITON_LIBRARIES})

  if(WIN32)
    target_link_libraries(triton PRIVATE ${CMAKE_DL_LIBS})
  else()
    target_link_libraries(triton PRIVATE z)
  endif()

  target_link_options(triton PRIVATE ${LLVM_LDFLAGS})

  if(NOT WIN32)
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
    if(APPLE)
      target_link_libraries(triton PRIVATE "-undefined dynamic_lookup -flto")
    endif()
  endif()
  
  # 确保 libtriton.so 输出到 triton/_C 目录
  set_target_properties(triton PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/triton/_C)
endif()

# ═══════════════════════════════════════════════════════════════════════
# Part 4: 构建 _C.so（triton-anchor 的 Python 绑定）
# ═══════════════════════════════════════════════════════════════════════
pybind11_add_module(_C csrc/bindings/triton_anchor_passes.cc)

target_link_libraries(_C PRIVATE
    TritonLinalgAnalysis
    TritonLinalgPipelines
    TritonLinalgUtils
    ArithToLinalg
    MathToLinalg
    TritonToLinalg
    LinalgExtDialect
    LinalgExtInterface
    LinalgExtTransforms
    MathExtDialect
    AuxiliaryDialect
    TritonTransformsExtend
)

# 确保 _C.so 输出到 triton_anchor 目录
set_target_properties(_C PROPERTIES 
    OUTPUT_NAME "_C"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/triton_anchor
)
install(TARGETS _C DESTINATION triton_anchor)
