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

# Pybind11: 通过 Python3 的 site-packages 路径自动发现 pip 安装的 pybind11
find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# 全局添加 Python 和 pybind11 的 include 路径（与 sophgo 保持一致）
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})

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

# 注意：--exclude-libs,ALL 已移至 libtriton target 级别,
# 避免全局影响下游插件需要导出的方言符号。
# if(UNIX AND NOT APPLE)
#   set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
# endif()

# ═══════════════════════════════════════════════════════════════════════
# 路径定义
# ═══════════════════════════════════════════════════════════════════════
set(UPSTREAM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/triton")
set(TRITON_SOURCE_DIR "${UPSTREAM_DIR}")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/triton")

# ═══════════════════════════════════════════════════════════════════════
# include 路径
# ═══════════════════════════════════════════════════════════════════════
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
# 上游 Triton 头文件
include_directories(${UPSTREAM_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/triton/include)  # TableGen 生成
include_directories(${UPSTREAM_DIR})                                # third_party 引用
include_directories(${CMAKE_CURRENT_BINARY_DIR}/triton)           # third_party TableGen 生成
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
add_subdirectory(triton/include)
# C++ 库实现
add_subdirectory(triton/lib)
# f2reduce 辅助库
add_subdirectory(triton/third_party/f2reduce)

# 在 csrc 加入之前捕获上游库列表，防止 triton-anchor 的 Pass 被混入 libtriton.so
get_property(TRITON_UPSTREAM_LIBS GLOBAL PROPERTY TRITON_LIBS)

# ═══════════════════════════════════════════════════════════════════════
# Part 2: 编译 triton-anchor 自有 C++ Pass（triton-linalg 等）
# ═══════════════════════════════════════════════════════════════════════
add_subdirectory(csrc)

# ═══════════════════════════════════════════════════════════════════════
# 导出下游硬件后端插件需要的方言符号
# ═══════════════════════════════════════════════════════════════════════
# 默认全局 -fvisibility=hidden 会导致所有符号在 libtriton.so 中不可见。
# 但下游的硬件后端插件（如 triton-sophgo-backend）需要引用这些方言的
# TypeID 和 Op 方法。这里显式将它们的可见性恢复为 default。
set(TRITON_ANCHOR_EXPORTED_TARGETS
    LinalgExtDialect
    LinalgExtInterface
    MathExtDialect
    AuxiliaryDialect
    TritonIR
)
foreach(target ${TRITON_ANCHOR_EXPORTED_TARGETS})
    if(TARGET ${target})
        target_compile_options(${target} PRIVATE -fvisibility=default)
    endif()
endforeach()

# ═══════════════════════════════════════════════════════════════════════
# Part 3: 构建 libtriton.so（上游 Triton 的 Python C++ 绑定）
# ═══════════════════════════════════════════════════════════════════════
if(TRITON_BUILD_PYTHON_MODULE)
  message(STATUS "Adding Python module (libtriton.so)")

  set(PYTHON_SRC_PATH ${UPSTREAM_DIR}/python/src)
  include_directories(${PYTHON_SRC_PATH})

  # 如果 setup.py 传入了 PYTHON_INCLUDE_DIRS，用它覆盖 find_package 的结果
  if(PYTHON_INCLUDE_DIRS)
    include_directories(${PYTHON_INCLUDE_DIRS})
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
    csrc/bindings/triton_anchor_passes.cc
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

    # 为 triton-anchor 补充的 Linalg 依赖
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRMemRefDialect
    MLIRTensorDialect
    MLIRAffineDialect

    # triton-anchor 本身的内部库
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
    AuxiliaryTransforms
    TritonTransformsExtend
    TritonInterfaceExtend
    TritonDialectUtils
    DialectUtils
    LinalgExtDialectUtils

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
  # 确保 pybind11 头文件路径对 triton 可见（pybind11 3.x 通过 target 传递 include）
  target_link_libraries(triton PRIVATE pybind11::headers)

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

