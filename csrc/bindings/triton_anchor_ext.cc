// triton-anchor 的 Python 绑定入口
//
// 此文件注册了 triton-anchor 提供的 MLIR 方言和通用 triton-shared Pass，
// 通过 pybind11 暴露给 Python 层（以 triton._C.libtriton.anchor 形式访问）。
//
// 架构：
//   triton-anchor 只包含原生的、硬件无关的通用 Pass：
//     - TritonToLinalg
//     - TritonToStructured / TritonToUnstructured
//     - UnstructuredToMemref / TritonPtrToMemref
//   以及基础方言：TritonStructured, TritonTilingExt
//
//   硬件专有 Pass 由各硬件后端插件负责注册。

#include "AnchorIRValidatorBindings.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// triton-shared 方言（flir 原生）
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

// triton-shared Pass（flir 原生，硬件无关）
#include "triton-shared/Conversion/TritonToLinalg/TritonToLinalg.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"

// Python 绑定所需的 passes.h（来自 triton/python/src/）
#include "passes.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mlir;

// 注册通用 triton-shared Pass（供 triton-anchor 本身或下游后端调用）
void init_triton_anchor_passes(py::module &&m) {
  // flir 原生通用 Pass
  ADD_PASS_WRAPPER_0("add_triton_to_linalg",
                     triton::createTritonToLinalgPass);
  ADD_PASS_WRAPPER_0("add_triton_to_structured",
                     triton::createTritonToStructuredPass);
  ADD_PASS_WRAPPER_0("add_triton_to_unstructured",
                     triton::createTritonToUnstructuredPass);
  ADD_PASS_WRAPPER_0("add_unstructured_to_memref",
                     triton::createUnstructuredToMemrefPass);
  ADD_PASS_WRAPPER_0("add_triton_ptr_to_memref",
                     triton::createTritonPtrToMemrefPass);

  // 通用优化 Pass（便于后端直接使用）
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_canonicalize", createCanonicalizerPass);
}

// triton-anchor 模块入口（对应 triton._C.libtriton.anchor）
void init_triton_anchor(py::module &&m) {
  // 注册 triton-shared 方言到给定的 MLIR Context
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    DialectRegistry registry;
    // triton-shared 基础方言（flir 原生）
    registry.insert<affine::AffineDialect, arith::ArithDialect,
                    bufferization::BufferizationDialect,
                    cf::ControlFlowDialect, func::FuncDialect,
                    gpu::GPUDialect, index::IndexDialect,
                    linalg::LinalgDialect, math::MathDialect,
                    memref::MemRefDialect, nvgpu::NVGPUDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    vector::VectorDialect, triton::TritonDialect,
                    triton::gpu::TritonGPUDialect,
                    tts::TritonStructuredDialect,
                    ttx::TritonTilingExtDialect>();
    func::registerInlinerExtension(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  // Structured AnchorIR validation and normalization share this same public
  // module with the 3.3 pass bindings.
  init_triton_anchor_validator(m);

  // Pass 子模块
  auto passes = m.def_submodule("passes", "triton-anchor generic passes");
  init_triton_anchor_passes(std::move(passes));
}
