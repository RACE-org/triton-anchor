"""Integration tests for the mandatory Adapter/Backend AnchorIR boundary.

The real Linalg Adapter pass pipeline is covered here; whole-compiler corpus
execution and per-Pass capture belong to the T10.3 integration.
"""

import os
import tempfile
from typing import Any

import pytest
from triton._C.libtriton import anchor, ir

from triton_anchor import (
    ANCHOR_IR_NORMALIZATION_VERSION,
    ANCHOR_IR_SPEC_VERSION,
    AnchorIRNormalizer,
    AnchorIRPhase,
    AnchorIRTrack,
    AnchorIRValidationError,
    StructuredAnchorIRValidator,
    run_anchor_ir_compilation,
)
from triton_anchor.adapters.base import ITritonToLinalgAdapter
from triton_anchor.adapters.base import AdapterConversionError
from triton_anchor.adapters.triton_linalg_adapter import TritonLinalgAdapter

VALID_IR = """
module {
  func.func @kernel() {
    func.return
  }
}
"""

MINIMAL_TTIR = """
module {
  tt.func public @kernel() {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32}
        : tensor<16xi32>
    tt.return
  }
}
"""

CONTROL_FLOW_TTIR = """
module {
  tt.func public @kernel(%flag: i1) {
    cf.cond_br %flag, ^positive, ^negative
  ^positive:
    tt.return
  ^negative:
    tt.return
  }
}
"""

PROGRAM_ID_TTIR = """
module {
  tt.func public @kernel() -> i32 {
    %pid = tt.get_program_id x : i32
    tt.return %pid : i32
  }
}
"""


class RecordingTextAdapter(ITritonToLinalgAdapter):
    def __init__(self, output: str):
        self.output = output
        self.calls = 0

    def name(self) -> str:
        return "recording-text"

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> str:
        self.calls += 1
        metadata["adapter_called"] = True
        return self.output


class RecordingHook:
    def __init__(self, output: str):
        self.output = output
        self.calls = 0

    def on_anchor_ir_ready(self, anchor_ir: str) -> str:
        self.calls += 1
        return self.output


class RecordingModuleAdapter(ITritonToLinalgAdapter):
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def name(self) -> str:
        return "recording-module"

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None):
        self.calls += 1
        return self.output


def _parse_module(text: str, *, load_anchor: bool = True):
    context = ir.context()
    ir.load_dialects(context)
    if load_anchor:
        anchor.load_dialects(context)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", encoding="utf-8", delete=False
    ) as source:
        source.write(text)
        source_path = source.name
    try:
        return ir.parse_mlir_module(source_path, context), context
    finally:
        os.unlink(source_path)


def test_adapter_compile_pre_hook_failure_raises_before_hook_and_lowering():
    adapter = RecordingTextAdapter('module { "smt.adapter_failure"() : () -> () }')
    hook = RecordingHook(VALID_IR)
    lowered = []
    metadata = {}

    with pytest.raises(AnchorIRValidationError) as captured:
        adapter.compile(
            object(),
            metadata,
            hook=hook,
            backend_lowering=lowered.append,
            source_name="adapter-output.mlir",
        )

    error = captured.value
    assert adapter.calls == 1
    assert metadata["adapter_called"] is True
    assert hook.calls == 0
    assert lowered == []
    assert error.report.phase is AnchorIRPhase.PRE_HOOK
    assert error.report.diagnostics[0].code == "AIR-LINALG-001"
    rendered = str(error)
    assert "operation_path: builtin.module" in rendered
    assert "location: adapter-output.mlir:" in rendered
    assert "hint:" in rendered


def test_adapter_compile_post_hook_failure_raises_before_backend_lowering():
    adapter = RecordingTextAdapter(VALID_IR)
    hook = RecordingHook('module { "vendor.undeclared"() : () -> () }')
    lowered = []

    with pytest.raises(AnchorIRValidationError) as captured:
        adapter.compile(
            object(),
            {},
            hook=hook,
            backend_lowering=lowered.append,
        )

    error = captured.value
    assert adapter.calls == 1
    assert hook.calls == 1
    assert lowered == []
    assert error.report.phase is AnchorIRPhase.POST_HOOK
    assert error.report.diagnostics[0].code == "AIR-COMMON-001"
    assert "vendor.undeclared" in str(error)
    assert "hint:" in str(error)


def test_adapter_compile_success_runs_backend_lowering_and_returns_report():
    adapter = RecordingTextAdapter(VALID_IR)
    hook = RecordingHook(VALID_IR)
    lowered = []

    report = adapter.compile(
        object(),
        {},
        hook=hook,
        backend_lowering=lambda anchor_ir: lowered.append(anchor_ir) or "binary",
    )

    assert report.valid
    assert adapter.calls == 1
    assert hook.calls == 1
    assert report.lowering_executed
    assert report.lowered_output == "binary"
    assert lowered == [VALID_IR]


def test_public_compilation_entry_requires_explicit_track_contract():
    adapter = RecordingTextAdapter(VALID_IR)

    report = run_anchor_ir_compilation(
        adapter,
        object(),
        {},
        hook=None,
        backend_lowering=None,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
    )

    assert report.valid
    assert report.pre_hook.track is AnchorIRTrack.LINALG
    assert report.post_hook is not None
    assert report.post_hook.track is AnchorIRTrack.LINALG


def test_compilation_rejects_policy_request_before_running_adapter():
    adapter = RecordingTextAdapter(VALID_IR)

    with pytest.raises(AnchorIRValidationError) as captured:
        run_anchor_ir_compilation(
            adapter,
            object(),
            {},
            hook=None,
            backend_lowering=None,
            spec_version="anchor-ir/unsupported",
            track=AnchorIRTrack.LINALG,
        )

    assert adapter.calls == 0
    assert [item.code for item in captured.value.report.diagnostics] == [
        "AIR-REQUEST-002"
    ]


def test_linalg_adapter_cannot_be_misclassified_as_triton_gpu_output():
    adapter = RecordingTextAdapter(VALID_IR)

    with pytest.raises(ValueError, match="requires track='linalg'"):
        run_anchor_ir_compilation(
            adapter,
            object(),
            {},
            hook=None,
            backend_lowering=None,
            track=AnchorIRTrack.TRITON_GPU,
        )

    assert adapter.calls == 0


def test_adapter_compile_dispatches_real_module_through_both_validation_phases():
    module, context = _parse_module(VALID_IR)
    adapter = RecordingModuleAdapter(module)
    hook_calls = []
    lowered = []

    class ModuleHook:
        def on_anchor_ir_ready(self, value):
            hook_calls.append(value)
            return value

    report = adapter.compile(
        object(),
        {},
        hook=ModuleHook(),
        context=context,
        backend_lowering=lambda value: lowered.append(value) or "binary",
    )

    assert report.valid
    assert report.pre_hook.phase is AnchorIRPhase.PRE_HOOK
    assert report.post_hook is not None
    assert report.post_hook.phase is AnchorIRPhase.POST_HOOK
    assert report.post_hook_snapshot is not None
    assert report.post_hook_snapshot.acceptable
    assert adapter.calls == 1
    assert hook_calls == [module]
    assert len(lowered) == 1
    assert lowered[0] is not report.output
    assert report.lowered_output == "binary"
    assert context is not None


def test_real_triton_linalg_pass_pipeline_enters_strict_boundary_and_lowering():
    """Exercise the in-repository C++ Adapter passes, not a recording double."""

    module, context = _parse_module(MINIMAL_TTIR, load_anchor=False)
    lowered = []

    report = TritonLinalgAdapter().compile(
        module,
        {},
        hook=None,
        backend_lowering=lambda value: lowered.append(value) or "binary",
        context=context,
    )

    assert report.valid
    assert report.pre_hook.phase is AnchorIRPhase.PRE_HOOK
    assert report.post_hook is not None
    assert report.post_hook.phase is AnchorIRPhase.POST_HOOK
    assert report.post_hook_snapshot is not None
    assert report.post_hook_snapshot.acceptable
    assert report.lowering_executed
    assert report.lowered_output == "binary"
    assert len(lowered) == 1
    assert lowered[0] is not report.output
    assert "func.func @kernel" in str(report.output)
    assert "tt.func" not in str(report.output)


def test_real_linalg_adapter_residual_program_id_fails_closed():
    """Do not let an Adapter/contract mismatch silently enter a backend."""

    module, context = _parse_module(PROGRAM_ID_TTIR, load_anchor=False)
    hook = RecordingHook(VALID_IR)
    lowered = []

    with pytest.raises(AnchorIRValidationError) as captured:
        TritonLinalgAdapter().compile(
            module,
            {},
            hook=hook,
            backend_lowering=lowered.append,
            context=context,
        )

    assert hook.calls == 0
    assert lowered == []
    assert [
        (item.code, item.object_name)
        for item in captured.value.report.diagnostics
    ] == [("AIR-LINALG-001", "tt.get_program_id")]


def test_repeated_real_adapter_compilation_has_stable_normalized_hash():
    """Reparse and rerun the real C++ passes; do not only renormalize one Module."""

    hashes = []
    normalized_texts = []
    for _ in range(3):
        module, context = _parse_module(MINIMAL_TTIR, load_anchor=False)
        report = TritonLinalgAdapter().compile(
            module,
            {},
            hook=None,
            backend_lowering=lambda value: value,
            context=context,
        )
        assert report.valid
        normalized = AnchorIRNormalizer().normalize_module(
            report.output,
            normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            phase=AnchorIRPhase.POST_HOOK,
        )
        assert normalized.acceptable
        hashes.append(normalized.sha256)
        normalized_texts.append(normalized.normalized_text)

    assert len(set(hashes)) == 1
    assert len(set(normalized_texts)) == 1


def test_real_tritongpu_conversion_preserves_legal_cf_control_flow():
    """The pinned Triton conversion explicitly keeps the cf dialect legal."""

    from triton._C.libtriton import passes

    module, context = _parse_module(CONTROL_FLOW_TTIR, load_anchor=False)
    manager = ir.pass_manager(context)
    passes.ttir.add_convert_to_ttgpuir(
        manager,
        "cuda:80",
        4,
        32,
        1,
    )
    manager.run(module)
    anchor.load_dialects(context)

    assert "cf.cond_br" in str(module)
    report = StructuredAnchorIRValidator().validate_module(
        module,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.TRITON_GPU,
        phase=AnchorIRPhase.PRE_HOOK,
    )
    assert report.valid
    assert report.diagnostics == ()


def test_real_triton_linalg_adapter_requires_context_when_module_has_none():
    module, context = _parse_module(MINIMAL_TTIR)

    with pytest.raises(AdapterConversionError, match="MLIR context is required"):
        TritonLinalgAdapter().convert(module, {}, context=None)

    # Keep the owning Python context alive until after the failure path.
    assert context is not None


def test_real_adapter_wraps_pipeline_setup_failure_as_conversion_error(
    monkeypatch,
):
    module, context = _parse_module(MINIMAL_TTIR, load_anchor=False)
    adapter = TritonLinalgAdapter()

    def fail_setup(_pm, _passes):
        raise RuntimeError("missing required pass registration")

    monkeypatch.setattr(adapter, "_add_passes", fail_setup)
    with pytest.raises(
        AdapterConversionError,
        match="missing required pass registration",
    ) as captured:
        adapter.convert(module, {}, context=context)

    assert isinstance(captured.value.__cause__, RuntimeError)
