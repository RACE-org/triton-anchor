#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "triton-linalg/Conversion/Passes.h"
#include "triton-linalg/Dialect/Triton/Transforms/Passes.h"
#include "triton-linalg/Pipelines/Pipelines.h"
#include "triton-linalg/RegisterTritonLinalgDialects.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define ADD_PASS_WRAPPER_0(name, builder)                                      \
  m.def(name, [](mlir::PassManager &pm) { pm.addPass(builder()); })

void init_triton_anchor_passes_triton_to_linalg(py::module_ &m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_0("add_triton_to_linalg", createTritonToLinalgPass);
  ADD_PASS_WRAPPER_0("add_arith_to_linalg", createArithToLinalgPass);
  ADD_PASS_WRAPPER_0("add_math_to_linalg", createMathToLinalgPass);
  ADD_PASS_WRAPPER_0("add_canonicalize_triton", createCanonicalizeTritonPass);
  ADD_PASS_WRAPPER_0("add_wrap_func_body_with_single_block",
                     createWrapFuncBodyWithSingleBlockPass);
  ADD_PASS_WRAPPER_0("add_pointer_strength_reduction",
                     createPointerStrengthReductionPass);
  m.def("add_extract_like_move_backward", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::func::FuncOp>(createExtractLikeMoveBackwardPass());
  });
}

void init_triton_anchor(py::module &&m) {
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registerTritonLinalgDialects(registry);
    mlir::func::registerInlinerExtension(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  auto passes = m.def_submodule("anchor_passes", "Triton Anchor Passes");
  auto tl = passes.def_submodule("triton_to_linalg", "Triton to Linalg passes");
  init_triton_anchor_passes_triton_to_linalg(tl);

  // Register passes into global registry so pipeline parsing works if needed
  registerTritonLinalgPasses();
}
