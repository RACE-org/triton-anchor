#pragma once

namespace pybind11 {
class module_;
}

/// Register the structured AnchorIR validation and normalization APIs on
/// ``triton._C.libtriton.anchor``.
///
/// Keep this declaration separate from the pass bindings so that the binding
/// entry point has one shared signature across translation units.
void init_triton_anchor_validator(pybind11::module_ &module);
