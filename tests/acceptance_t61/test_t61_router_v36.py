"""T6.1 acceptance tests for Router, fallback, metadata, and AnchorIR.

These tests target the 3.6 worktree only. They use fake adapters and fake
HWCapability objects so the Router can be verified without real hardware.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from triton_anchor.adapters.base import AdapterConversionError, ITritonToLinalgAdapter
from triton_anchor.adapters.hybrid_adapter import HybridAdapter
from triton_anchor.adapters.registry import AdapterNotFoundError, AdapterRegistry
from triton_anchor.anchor_ir import AnchorIRError, AnchorIRTrack, AnchorIRValidator
from triton_anchor.hw_capability import (
    ComputeParadigm,
    GPGPUCapability,
    HWCapability,
    TensorCapability,
)
from triton_anchor.pipeline import make_anchor_ir, resolve_adapter_route_for_compile


class RecordingAdapter(ITritonToLinalgAdapter):
    def __init__(self, adapter_name: str, result: object = None, error: Exception = None):
        self.adapter_name = adapter_name
        self.result = adapter_name if result is None else result
        self.error = error
        self.calls = 0

    def name(self) -> str:
        return self.adapter_name

    def convert(self, ttir_module, metadata: dict, context=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def isolated_registry(monkeypatch):
    AdapterRegistry.reset()
    monkeypatch.setattr(AdapterRegistry, "_discovered", True)
    yield
    AdapterRegistry.reset()


def _hw(
    ptr_model: str,
    *,
    track: AnchorIRTrack = AnchorIRTrack.LINALG,
    preferred_adapter: str | None = None,
    adapter_fallback_policy: str = "managed",
) -> HWCapability:
    if track == AnchorIRTrack.TRITON_GPU:
        return HWCapability(
            name="t61-gpu",
            arch_family="gpu",
            compute_paradigm=ComputeParadigm.GPGPU,
            anchor_ir_track=track,
            ptr_model=ptr_model,
            preferred_adapter=preferred_adapter,
            adapter_fallback_policy=adapter_fallback_policy,
            gpgpu_cap=GPGPUCapability(num_warps=4, warp_size=32),
        )
    return HWCapability(
        name="t61-linalg",
        arch_family="tpu",
        compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
        anchor_ir_track=track,
        ptr_model=ptr_model,
        preferred_adapter=preferred_adapter,
        adapter_fallback_policy=adapter_fallback_policy,
        tensor_cap=TensorCapability(num_cores=1),
    )


def _register_fakes() -> None:
    for name in ("triton-shared", "triton-linalg", "hybrid", "triton-gpu"):
        AdapterRegistry.register(RecordingAdapter(name))


def test_router_selects_triton_gpu_for_triton_gpu_track(isolated_registry):
    _register_fakes()
    metadata = {}

    adapter = AdapterRegistry.get_adapter(
        _hw("structured", track=AnchorIRTrack.TRITON_GPU), metadata=metadata
    )

    assert adapter.name() == "triton-gpu"
    assert metadata["anchor_adapter"] == "triton-gpu"
    assert metadata["anchor_adapter_route"]["anchor_ir_track"] == "triton_gpu"


def test_router_selects_shared_for_structured_linalg(isolated_registry):
    _register_fakes()
    metadata = {}

    adapter = AdapterRegistry.get_adapter(_hw("structured"), metadata=metadata)

    assert adapter.name() == "triton-shared"
    assert metadata["anchor_adapter"] == "triton-shared"
    assert metadata["anchor_adapter_route"]["selected_adapter"] == "triton-shared"


def test_router_selects_linalg_for_axis_info_linalg(isolated_registry):
    _register_fakes()
    metadata = {}

    adapter = AdapterRegistry.get_adapter(_hw("axis_info"), metadata=metadata)

    assert adapter.name() == "triton-linalg"
    assert metadata["anchor_adapter_route"]["fallback_chain"] == ["triton-linalg"]


def test_router_selects_hybrid_for_hybrid_linalg(isolated_registry):
    _register_fakes()
    metadata = {}

    adapter = AdapterRegistry.get_adapter(_hw("hybrid"), metadata=metadata)

    assert adapter.name() == "hybrid"
    assert metadata["anchor_adapter_route"]["selected_adapter"] == "hybrid"


def test_preferred_adapter_wins_when_available_and_compatible(isolated_registry):
    _register_fakes()
    metadata = {}

    adapter = AdapterRegistry.get_adapter(
        _hw("axis_info", preferred_adapter="triton-linalg"), metadata=metadata
    )

    assert adapter.name() == "triton-linalg"
    assert metadata["anchor_adapter_route"]["explicit_preference"] is True


def test_adapter_router_module_owns_final_selection():
    from triton_anchor.adapters.router import AdapterRouter

    assert hasattr(AdapterRouter, "resolve")


def test_preferred_adapter_missing_fails(isolated_registry):
    _register_fakes()

    with pytest.raises(AdapterNotFoundError, match="missing-adapter"):
        AdapterRegistry.get_adapter(_hw("structured", preferred_adapter="missing-adapter"))


def test_preferred_adapter_incompatible_fails(isolated_registry):
    _register_fakes()

    with pytest.raises(AdapterNotFoundError):
        AdapterRegistry.get_adapter(
            _hw("structured", preferred_adapter="triton-gpu")
        )


def test_no_implicit_registry_order_fallback(isolated_registry):
    AdapterRegistry.register(RecordingAdapter("triton-shared"))

    with pytest.raises(AdapterNotFoundError, match="triton-linalg"):
        AdapterRegistry.get_adapter(_hw("axis_info"))


def test_invalid_linalg_gpu_combo_fails(isolated_registry):
    _register_fakes()

    with pytest.raises(AdapterNotFoundError):
        AdapterRegistry.get_adapter(_hw("gpu", track=AnchorIRTrack.LINALG))


def test_selection_is_deterministic_across_repeated_runs(isolated_registry):
    _register_fakes()
    hw = _hw("hybrid")

    first = AdapterRegistry.get_route(hw)
    second = AdapterRegistry.get_route(hw)

    assert first.cache_key() == second.cache_key()
    assert first.to_cache_payload() == second.to_cache_payload()


def test_selection_is_deterministic_across_registration_order(isolated_registry):
    for name in ("triton-gpu", "hybrid", "triton-linalg", "triton-shared"):
        AdapterRegistry.register(RecordingAdapter(name))
    first = AdapterRegistry.get_route(_hw("axis_info")).cache_key()

    AdapterRegistry.reset()
    AdapterRegistry._discovered = True
    for name in ("triton-shared", "triton-linalg", "hybrid", "triton-gpu"):
        AdapterRegistry.register(RecordingAdapter(name))
    second = AdapterRegistry.get_route(_hw("axis_info")).cache_key()

    assert first == second


def test_strict_mode_blocks_fallback():
    structured = RecordingAdapter("triton-shared", error=AdapterConversionError("triton-shared", detail="structured rejected"))
    axis_info = RecordingAdapter("triton-linalg", result="axis-info-ir")
    adapter = HybridAdapter(structured_adapter=structured, axis_info_adapter=axis_info)
    metadata = {"anchor_adapter_fallback_policy": "strict"}

    with pytest.raises(AdapterConversionError):
        adapter.convert("ttir", metadata)

    assert structured.calls == 1
    assert axis_info.calls == 0
    assert metadata["anchor_adapter_fallback_blocked"] is True


def test_allowed_fallback_records_reason():
    structured = RecordingAdapter(
        "triton-shared",
        error=AdapterConversionError("triton-shared", detail="structured rejected"),
    )
    axis_info = RecordingAdapter("triton-linalg", result="axis-info-ir")
    adapter = HybridAdapter(structured_adapter=structured, axis_info_adapter=axis_info)
    metadata = {}

    result = adapter.convert("ttir", metadata)

    assert result == "axis-info-ir"
    assert metadata["anchor_adapter_fallback"] == "axis_info"
    assert metadata["anchor_adapter_hybrid_attempts"][0]["detail"] == (
        "Adapter 'triton-shared' failed to convert: structured rejected"
    )
    assert metadata["anchor_adapter_hybrid_attempts"][-1]["result"] == "succeeded"


def test_disallowed_fallback_fails_before_convert():
    structured = RecordingAdapter("triton-shared", error=RuntimeError("broken structured path"))
    axis_info = RecordingAdapter("triton-linalg", result="axis-info-ir")
    adapter = HybridAdapter(structured_adapter=structured, axis_info_adapter=axis_info)
    metadata = {}

    with pytest.raises(RuntimeError, match="broken structured path"):
        adapter.convert("ttir", metadata)

    assert structured.calls == 1
    assert axis_info.calls == 0
    assert "anchor_adapter_fallback" not in metadata


def test_hybrid_structured_success_does_not_call_linalg():
    structured = RecordingAdapter("triton-shared", result="structured-ir")
    axis_info = RecordingAdapter("triton-linalg", result="axis-info-ir")
    adapter = HybridAdapter(structured_adapter=structured, axis_info_adapter=axis_info)
    metadata = {}

    result = adapter.convert("ttir", metadata)

    assert result == "structured-ir"
    assert structured.calls == 1
    assert axis_info.calls == 0
    assert metadata["anchor_adapter_effective"] == "triton-shared"


def test_hybrid_structured_failure_allowed_fallback_calls_linalg():
    structured = RecordingAdapter(
        "triton-shared",
        error=AdapterConversionError("triton-shared", detail="structured rejected"),
    )
    axis_info = RecordingAdapter("triton-linalg", result="axis-info-ir")
    adapter = HybridAdapter(structured_adapter=structured, axis_info_adapter=axis_info)
    metadata = {}

    result = adapter.convert("ttir", metadata)

    assert result == "axis-info-ir"
    assert structured.calls == 1
    assert axis_info.calls == 1
    assert metadata["anchor_adapter_effective"] == "triton-linalg"
    assert metadata["anchor_adapter_fallback"] == "axis_info"


def test_hybrid_structured_failure_strict_mode_fails():
    structured = RecordingAdapter(
        "triton-shared",
        error=AdapterConversionError("triton-shared", detail="structured rejected"),
    )
    axis_info = RecordingAdapter("triton-linalg", result="axis-info-ir")
    adapter = HybridAdapter(structured_adapter=structured, axis_info_adapter=axis_info)
    metadata = {"anchor_adapter_fallback_policy": "strict"}

    with pytest.raises(AdapterConversionError, match="structured rejected"):
        adapter.convert("ttir", metadata)

    assert structured.calls == 1
    assert axis_info.calls == 0
    assert metadata["anchor_adapter_fallback_blocked"] is True


def test_decision_metadata_contains_selected_adapter_and_reasons(isolated_registry):
    _register_fakes()
    metadata = {}

    route = AdapterRegistry.get_route(_hw("structured"), metadata=metadata)

    assert metadata["anchor_adapter"] == "triton-shared"
    assert metadata["anchor_adapter_cache_key"] == route.cache_key()
    assert metadata["anchor_adapter_route"]["selected_adapter"] == "triton-shared"
    missing = [
        key
        for key in (
            "decision_policy_version",
            "rejected_candidates",
            "reject_reasons",
            "fallback_reason",
            "input_decision_key",
        )
        if key not in metadata["anchor_adapter_route"]
    ]
    assert not missing, f"missing metadata fields: {missing}"


def test_linalg_anchor_ir_rejects_forbidden_triton_gpu_dialect():
    validator = AnchorIRValidator(track=AnchorIRTrack.LINALG)
    ir_text = """
    module {
      func.func @kernel() {
        %0 = ttg.convert_layout %arg0 : tensor<16xf32> -> tensor<16xf32>
        return
      }
    }
    """

    violations = validator.validate_pre_hook(ir_text)
    assert any(violation.dialect == "ttg" for violation in violations)


def test_triton_gpu_anchor_ir_rejects_transition_dialects():
    validator = AnchorIRValidator(track=AnchorIRTrack.TRITON_GPU)
    ir_text = """
    module {
      func.func @kernel() {
        %0 = tts.transition %arg0 : tensor<16xf32>
        return
      }
    }
    """

    violations = validator.validate_pre_hook(ir_text)
    assert any(violation.dialect == "tts" for violation in violations)


def test_all_builtin_adapters_are_registered_or_have_clear_unavailable_reason(monkeypatch):
    AdapterRegistry.reset()
    AdapterRegistry.register_builtins()
    monkeypatch.setattr(AdapterRegistry, "_discovered", True)

    adapters = AdapterRegistry.list_adapters()

    assert set(adapters) >= {
        "triton-shared",
        "triton-linalg",
        "hybrid",
        "triton-gpu",
    }
    assert adapters["triton-shared"] == "TritonSharedAdapter"
    assert adapters["triton-linalg"] == "TritonLinalgAdapter"
    assert adapters["hybrid"] == "HybridAdapter"
    assert adapters["triton-gpu"] == "TritonGPUAdapter"


def test_adapter_output_is_anchorir_validated_before_return(monkeypatch):
    adapter = RecordingAdapter(
        "triton-shared",
        result="""
        module {
          func.func @kernel() {
            %0 = ttg.convert_layout %arg0 : tensor<16xf32> -> tensor<16xf32>
            return
          }
        }
        """,
    )
    route = SimpleNamespace(cache_key=lambda: "route", to_metadata=lambda: {})
    monkeypatch.setattr(AdapterRegistry, "resolve", lambda hw, metadata=None: (adapter, route))

    with pytest.raises(AnchorIRError):
        make_anchor_ir("ttir", {}, hw=_hw("structured"))


def test_compile_route_helper_records_metadata(isolated_registry):
    _register_fakes()
    metadata = {}

    route = resolve_adapter_route_for_compile(options=SimpleNamespace(hw_capability=_hw("hybrid")), metadata=metadata)

    assert route.selected_adapter == "hybrid"
    assert metadata["anchor_adapter"] == "hybrid"
    assert metadata["anchor_adapter_cache_key"] == route.cache_key()
