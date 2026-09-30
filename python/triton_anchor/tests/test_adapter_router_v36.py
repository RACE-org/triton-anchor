"""Triton 3.6 adapter Router and fallback tests."""

from __future__ import annotations

import sys
import types

import pytest

from triton_anchor.adapters.base import AdapterConversionError, ITritonToLinalgAdapter
from triton_anchor.adapters.hybrid_adapter import HybridAdapter
from triton_anchor.adapters.registry import AdapterNotFoundError, AdapterRegistry
from triton_anchor.adapters.triton_gpu_adapter import TritonGPUAdapter
from triton_anchor.adapters.triton_shared_adapter import TritonSharedAdapter
from triton_anchor.anchor_ir import AnchorIRTrack
from triton_anchor.hw_capability import (
    ComputeParadigm,
    GPGPUCapability,
    HWCapability,
    TensorCapability,
)
from triton_anchor.pipeline import resolve_adapter_route_for_compile


class DummyAdapter(ITritonToLinalgAdapter):
    def __init__(self, adapter_name: str, result: object = None, error: str = ""):
        self.adapter_name = adapter_name
        self.result = result if result is not None else adapter_name
        self.error = error
        self.calls = 0

    def name(self) -> str:
        return self.adapter_name

    def convert(self, ttir_module, metadata: dict, context=None):
        self.calls += 1
        if self.error:
            raise AdapterConversionError(self.name(), detail=self.error)
        return self.result


class FakeModule:
    def __init__(self, text: str = "module { tt.func @kernel() { return } }"):
        self.text = text
        self.context = object()
        self.ran = False

    def __str__(self):
        return self.text


class FakePassManager:
    def __init__(self, context):
        self.context = context
        self.added = []
        self.debug_enabled = False

    def enable_debug(self):
        self.debug_enabled = True

    def run(self, module):
        module.ran = True


def _tpu_hw(
    ptr_model: str,
    preferred_adapter: str | None = None,
    adapter_fallback_policy: str = "managed",
) -> HWCapability:
    return HWCapability(
        name="test-tpu",
        arch_family="tpu",
        compute_paradigm=ComputeParadigm.TENSOR_PROCESSOR,
        anchor_ir_track=AnchorIRTrack.LINALG,
        ptr_model=ptr_model,
        preferred_adapter=preferred_adapter,
        adapter_fallback_policy=adapter_fallback_policy,
        tensor_cap=TensorCapability(num_cores=1),
    )


def _gpu_hw() -> HWCapability:
    return HWCapability(
        name="test-gpu",
        arch_family="gpu",
        compute_paradigm=ComputeParadigm.GPGPU,
        anchor_ir_track=AnchorIRTrack.TRITON_GPU,
        ptr_model="gpu",
        gpgpu_cap=GPGPUCapability(num_warps=4, warp_size=32),
    )


@pytest.fixture
def isolated_registry(monkeypatch):
    AdapterRegistry.reset()
    monkeypatch.setattr(AdapterRegistry, "_discovered", True)
    yield
    AdapterRegistry.reset()


def _register_dummy_adapters():
    for name in ("triton-shared", "triton-linalg", "hybrid", "triton-gpu"):
        AdapterRegistry.register(DummyAdapter(name))


@pytest.mark.parametrize(
    ("hw", "expected", "fallback_chain"),
    [
        (_tpu_hw("structured"), "triton-shared", ["triton-shared"]),
        (_tpu_hw("axis_info"), "triton-linalg", ["triton-linalg"]),
        (_tpu_hw("hybrid"), "hybrid", ["triton-shared", "triton-linalg"]),
        (_gpu_hw(), "triton-gpu", ["triton-gpu"]),
    ],
)
def test_router_selects_distinct_v36_paths(
    isolated_registry, hw, expected, fallback_chain
):
    _register_dummy_adapters()
    metadata = {}

    adapter = AdapterRegistry.get_adapter(hw, metadata=metadata)

    assert adapter.name() == expected
    assert metadata["anchor_adapter"] == expected
    assert metadata["anchor_adapter_fallback_chain"] == fallback_chain
    assert metadata["anchor_adapter_route"]["selected_adapter"] == expected
    assert metadata["anchor_adapter_route"]["fallback_policy"] == "managed"
    assert metadata["anchor_adapter_cache_key"].startswith(
        "triton-anchor-adapter-route-v1:sha256:"
    )


def test_router_is_deterministic_across_registration_order(isolated_registry):
    for name in ("triton-gpu", "hybrid", "triton-linalg", "triton-shared"):
        AdapterRegistry.register(DummyAdapter(name))
    first = AdapterRegistry.get_route(_tpu_hw("axis_info")).cache_key()

    AdapterRegistry.reset()
    AdapterRegistry._discovered = True
    for name in ("triton-shared", "triton-linalg", "hybrid", "triton-gpu"):
        AdapterRegistry.register(DummyAdapter(name))
    second = AdapterRegistry.get_route(_tpu_hw("axis_info")).cache_key()

    assert first == second


def test_router_does_not_fallback_to_unrelated_adapter(isolated_registry):
    AdapterRegistry.register(DummyAdapter("triton-shared"))

    with pytest.raises(AdapterNotFoundError, match="triton-linalg"):
        AdapterRegistry.get_adapter(_tpu_hw("axis_info"))


def test_builtin_discovery_restores_all_v36_routes(monkeypatch):
    AdapterRegistry.reset()
    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda **kwargs: [],
    )

    adapters = AdapterRegistry.list_adapters()

    assert adapters["triton-shared"] == "TritonSharedAdapter"
    assert adapters["triton-linalg"] == "TritonLinalgAdapter"
    assert adapters["hybrid"] == "HybridAdapter"
    assert adapters["triton-gpu"] == "TritonGPUAdapter"


def test_router_records_preferred_adapter_in_route(isolated_registry):
    _register_dummy_adapters()
    metadata = {}

    adapter = AdapterRegistry.get_adapter(
        _tpu_hw("axis_info", preferred_adapter="triton-linalg"),
        metadata=metadata,
    )

    assert adapter.name() == "triton-linalg"
    assert metadata["anchor_adapter_route"]["explicit_preference"] is True
    assert metadata["anchor_adapter_route"]["fallback_chain"] == ["triton-linalg"]


def test_router_cache_key_includes_fallback_policy(isolated_registry):
    _register_dummy_adapters()

    managed = AdapterRegistry.get_route(_tpu_hw("hybrid")).cache_key()
    strict = AdapterRegistry.get_route(
        _tpu_hw("hybrid", adapter_fallback_policy="strict")
    ).cache_key()

    assert managed != strict


def test_compile_route_helper_records_metadata(isolated_registry):
    _register_dummy_adapters()

    class Options:
        hw_capability = _tpu_hw("hybrid")

    metadata = {}
    route = resolve_adapter_route_for_compile(options=Options(), metadata=metadata)

    assert route.selected_adapter == "hybrid"
    assert metadata["anchor_adapter"] == "hybrid"
    assert metadata["anchor_adapter_cache_key"] == route.cache_key()


def test_hybrid_adapter_uses_structured_first_success():
    structured = DummyAdapter("triton-shared", result="structured-ir")
    axis_info = DummyAdapter("triton-linalg", result="axis-info-ir")
    adapter = HybridAdapter(structured_adapter=structured, axis_info_adapter=axis_info)
    metadata = {}

    result = adapter.convert("ttir", metadata)

    assert result == "structured-ir"
    assert structured.calls == 1
    assert axis_info.calls == 0
    assert metadata["anchor_adapter_effective"] == "triton-shared"
    assert metadata["anchor_adapter_hybrid_attempts"][-1]["result"] == "succeeded"


def test_hybrid_adapter_falls_back_to_axis_info_after_structured_failure():
    structured = DummyAdapter("triton-shared", error="structured rejected")
    axis_info = DummyAdapter("triton-linalg", result="axis-info-ir")
    adapter = HybridAdapter(structured_adapter=structured, axis_info_adapter=axis_info)
    metadata = {}

    result = adapter.convert("ttir", metadata)

    assert result == "axis-info-ir"
    assert structured.calls == 1
    assert axis_info.calls == 1
    assert metadata["anchor_adapter_effective"] == "triton-linalg"
    assert metadata["anchor_adapter_fallback"] == "axis_info"


def test_hybrid_adapter_strict_fallback_blocks_axis_info():
    structured = DummyAdapter("triton-shared", error="structured rejected")
    axis_info = DummyAdapter("triton-linalg", result="axis-info-ir")
    adapter = HybridAdapter(structured_adapter=structured, axis_info_adapter=axis_info)
    metadata = {"anchor_adapter_fallback_policy": "strict"}

    with pytest.raises(AdapterConversionError, match="structured rejected"):
        adapter.convert("ttir", metadata)

    assert structured.calls == 1
    assert axis_info.calls == 0
    assert metadata["anchor_adapter_fallback_blocked"] is True


def _install_fake_libtriton(monkeypatch, passes):
    triton = types.ModuleType("triton")
    triton.__path__ = []
    c_ext = types.ModuleType("triton._C")
    c_ext.__path__ = []
    libtriton = types.ModuleType("triton._C.libtriton")
    libtriton.ir = types.SimpleNamespace(pass_manager=FakePassManager)
    libtriton.passes = passes
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton._C", c_ext)
    monkeypatch.setitem(sys.modules, "triton._C.libtriton", libtriton)


def test_gpu_adapter_uses_bound_convert_pass(monkeypatch):
    def add_convert(pm, *args):
        pm.added.append(("convert", args))

    passes = types.SimpleNamespace(
        ttir=types.SimpleNamespace(add_convert_to_ttgpuir=add_convert),
        ttgpuir=types.SimpleNamespace(),
        common=types.SimpleNamespace(add_cse=lambda pm: pm.added.append("cse")),
        gluon=types.SimpleNamespace(),
        plugin=types.SimpleNamespace(),
    )
    _install_fake_libtriton(monkeypatch, passes)
    module = FakeModule()
    metadata = {"hw_capability": _gpu_hw()}

    result = TritonGPUAdapter().convert(module, metadata)

    assert result is module
    assert module.ran is True
    assert metadata["anchor_gpu_convert_pass"] == "passes.ttir.add_convert_to_ttgpuir"
    assert metadata["anchor_gpu_post_passes"] == ["add_cse"]


def test_gpu_adapter_uses_plugin_convert_pass_without_legacy_binding(monkeypatch):
    def add_plugin_convert(pm):
        pm.added.append("plugin-convert")

    passes = types.SimpleNamespace(
        ttir=types.SimpleNamespace(),
        ttgpuir=types.SimpleNamespace(),
        common=types.SimpleNamespace(),
        gluon=types.SimpleNamespace(add_resolve_auto_encodings=lambda pm: None),
        plugin=types.SimpleNamespace(add_convert_to_ttgpuir=add_plugin_convert),
    )
    _install_fake_libtriton(monkeypatch, passes)
    metadata = {}

    TritonGPUAdapter().convert(FakeModule(), metadata)

    assert metadata["anchor_gpu_convert_pass"] == "passes.plugin.add_convert_to_ttgpuir"
    assert metadata["anchor_gpu_convert_pass_args"] == []


def test_gpu_adapter_reports_v36_plugin_gluon_diagnostic(monkeypatch):
    passes = types.SimpleNamespace(
        ttir=types.SimpleNamespace(),
        ttgpuir=types.SimpleNamespace(),
        common=types.SimpleNamespace(),
        gluon=types.SimpleNamespace(add_resolve_auto_encodings=lambda pm: None),
        plugin=types.SimpleNamespace(),
    )
    _install_fake_libtriton(monkeypatch, passes)

    with pytest.raises(AdapterConversionError) as exc_info:
        TritonGPUAdapter().convert(FakeModule(), {})

    message = str(exc_info.value)
    assert "Triton 3.6 GPU lowering requires" in message
    assert "TRITON_PASS_PLUGIN_PATH" in message
    assert "passes.ttgpuir may be empty" in message
    assert "gluon" in message


def test_shared_adapter_missing_tool_diagnostic_mentions_package_and_runpath():
    adapter = TritonSharedAdapter(opt_path="/tmp/not-a-triton-shared-opt")

    with pytest.raises(AdapterConversionError) as exc_info:
        adapter.convert("module {}", {})

    message = str(exc_info.value)
    assert "constructor opt_path" in message
    assert "triton/bin/triton-shared-opt" in message
    assert "RUNPATH" in message
