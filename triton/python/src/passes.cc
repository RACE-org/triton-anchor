#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Tools/PluginUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

void init_triton_analysis(py::module &&m) {
  py::class_<mlir::ModuleAllocation>(m, "allocation", py::module_local())
      .def(py::init<mlir::ModuleOp>());
  py::class_<mlir::ModuleMembarAnalysis>(m, "membar", py::module_local())
      .def(py::init<mlir::ModuleAllocation *>())
      .def("run", &mlir::ModuleMembarAnalysis::run);
}

void init_triton_passes_common(py::module &&m) {
  using namespace mlir;
  ADD_PASS_WRAPPER_0("add_sccp", createSCCPPass);
  ADD_PASS_WRAPPER_0("add_symbol_dce", createSymbolDCEPass);
  ADD_PASS_WRAPPER_0("add_inliner", createInlinerPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", createCanonicalizerPass);
  ADD_PASS_WRAPPER_0("add_cse", createCSEPass);
  ADD_PASS_WRAPPER_0("add_licm", createLoopInvariantCodeMotionPass);
  ADD_PASS_WRAPPER_0("print_ir", createPrintIRPass);
}

void init_triton_passes_ttir(py::module &&m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_0("add_combine", createTritonCombineOps);
  ADD_PASS_WRAPPER_0("add_reorder_broadcast", createTritonReorderBroadcast);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer",
                     createTritonRewriteTensorPointer);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_descriptor_to_pointer",
                     createTritonRewriteTensorDescriptorToPointer);
  ADD_PASS_WRAPPER_0("add_loop_unroll", createTritonLoopUnroll);
  ADD_PASS_WRAPPER_0("add_triton_licm", createTritonLoopInvariantCodeMotion);
  ADD_PASS_WRAPPER_0("add_loop_aware_cse", createTritonLoopAwareCSE);
}

void init_triton_passes_ttgpuir(py::module &&) {}

void init_plugin_passes(py::module &&m) {
  std::string filename =
      mlir::triton::tools::getStrEnv("TRITON_PASS_PLUGIN_PATH");
  if (filename.empty())
    return;

  TritonPlugin TP(filename);
  std::vector<const char *> passNames;
  if (auto result = TP.getPassHandles(passNames); !result)
    throw TP.err2exp(result.takeError());

  for (unsigned i = 0; i < passNames.size(); ++i) {
    const char *passName = passNames.data()[i];

    m.def(passName, [passName](mlir ::PassManager &pm) {
      std::string filename =
          mlir::triton::tools::getStrEnv("TRITON_PASS_PLUGIN_PATH");
      TritonPlugin TP(filename);
      if (auto result = TP.addPass(&pm, passName); !result)
        throw TP.err2exp(result.takeError());
    });
  }
}

void init_triton_passes_convert(py::module &&) {}

void init_triton_passes_llvmir(py::module &&) {}

void init_gluon_passes(py::module &&m) {
  using namespace mlir;
  namespace gluon = mlir::triton::gluon;
  ADD_PASS_WRAPPER_0("add_resolve_auto_encodings",
                     gluon::createGluonResolveAutoEncodingsPass);
  ADD_PASS_WRAPPER_0("add_canonicalizer", gluon::createGluonCanonicalize);
  ADD_PASS_WRAPPER_0("add_inliner", gluon::createGluonInline);
  ADD_PASS_WRAPPER_0("add_infer_coalesced_encodings",
                     gluon::createGluonInferCoalescedEncodingsPass);
}

void init_triton_passes(py::module &&m) {
  init_triton_analysis(m.def_submodule("analysis"));
  init_triton_passes_common(m.def_submodule("common"));
  init_triton_passes_convert(m.def_submodule("convert"));
  init_triton_passes_ttir(m.def_submodule("ttir"));
  init_triton_passes_ttgpuir(m.def_submodule("ttgpuir"));
  init_triton_passes_llvmir(m.def_submodule("llvmir"));
  init_gluon_passes(m.def_submodule("gluon"));
  init_plugin_passes(m.def_submodule("plugin"));
}
