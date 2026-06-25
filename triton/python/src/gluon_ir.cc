#include "ir.h"
#include "mlir/IR/MLIRContext.h"
#include "pybind11/pybind11.h"

#include <stdexcept>

namespace py = pybind11;
using namespace mlir;
using namespace ir;

namespace {

[[noreturn]] void raiseUnsupported() {
  throw std::runtime_error(
      "gluon_ir is not built in triton-shared-core minimal frontend");
}

struct GluonOpBuilder : public TritonOpBuilder {
  using TritonOpBuilder::TritonOpBuilder;
};

} // namespace

void init_gluon_ir(py::module &&m) {
  py::class_<GluonOpBuilder, TritonOpBuilder>(
      m, "GluonOpBuilder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>());

  m.def("compute_tmem_reg_layout",
        [](py::args, py::kwargs) -> py::object { raiseUnsupported(); });
  m.def("get_amd_mfma_scale_layout",
        [](py::args, py::kwargs) -> py::object { raiseUnsupported(); });
  m.def("get_amd_wmma_scale_layout",
        [](py::args, py::kwargs) -> py::object { raiseUnsupported(); });
}
