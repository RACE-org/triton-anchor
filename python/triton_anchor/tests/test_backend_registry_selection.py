"""Registry-level W6-W8 selection tests under the production full gate."""

from dataclasses import replace

import pytest

from triton_anchor.backends import (
    BACKEND_SELECTOR_ENV,
    BackendPluginCapabilityError,
    BackendPluginCompatibilityError,
    BackendPluginConflictError,
    BackendPluginLifecycleError,
    BackendPluginProtocolError,
    BackendPluginRegistry,
    BackendPluginSelectionError,
    PluginLifecycleState,
    SelectionMethod,
)
from triton_anchor.tests.test_backend_registry import (
    FakeDistribution,
    RuntimePlugin,
    SUPPORTED_TAG,
    TEST_ABI_FINGERPRINT,
    core_environment,
    manifest,
)


def triton_plugin(
    entry_point,
    *,
    plugin_id=None,
    triton_version=">=3.0,<3.1",
    targets=("mock",),
    capabilities=(),
    requires_capabilities=(),
    priority=0,
):
    """Return a minimal Python-only fixture; optional constraints stay omitted."""
    return {
        "plugin_id": plugin_id or f"vendor.{entry_point}",
        "entry_point": entry_point,
        "backend_protocol": ">=1.0,<2.0",
        "requires_triton": {"version": triton_version},
        "targets": list(targets),
        "capabilities": list(capabilities),
        "requires_capabilities": list(requires_capabilities),
        "isolation_mode": "python_only",
        "priority": priority,
    }


def registry_for(
    distributions,
    *,
    core_capabilities=(),
    preflight_profile="full",
):
    return BackendPluginRegistry(
        distribution_provider=lambda: tuple(distributions),
        environment_provider=core_environment,
        supported_tags=(SUPPORTED_TAG,),
        core_capabilities=core_capabilities,
        preflight_profile=preflight_profile,
    )


def distribution_for(
    tmp_path,
    entry_point,
    *,
    name,
    declaration=None,
):
    return FakeDistribution(
        tmp_path / name,
        name=name,
        manifest=manifest(
            declaration or triton_plugin(entry_point)
        ),
        entry_points=((entry_point, RuntimePlugin()),),
    )


def test_registry_selects_unique_priority_and_loads_only_winner(tmp_path):
    low = distribution_for(
        tmp_path,
        "low",
        name="low-backend",
        declaration=triton_plugin("low", priority=1),
    )
    high = distribution_for(
        tmp_path,
        "high",
        name="high-backend",
        declaration=triton_plugin("high", priority=2),
    )
    registry = registry_for((low, high))

    decision = registry.select("mock", environment={})

    assert decision.plugin_id == "vendor.high"
    assert decision.method is SelectionMethod.MANIFEST_PRIORITY
    assert decision.record.state is PluginLifecycleState.SELECTED
    assert decision.record.selected_targets == ("mock",)
    assert [low.entry_points[0].load_calls, high.entry_points[0].load_calls] == [
        0,
        1,
    ]
    assert registry.get_selection("mock") == decision
    assert (
        registry.diagnostics()["selections"]["mock"]["plugin_id"]
        == "vendor.high"
    )


def test_triton_mismatch_is_rejected_before_selection_and_import(tmp_path):
    incompatible = distribution_for(
        tmp_path,
        "future",
        name="future-backend",
        declaration=triton_plugin("future", triton_version=">=9.0"),
    )
    compatible = distribution_for(
        tmp_path,
        "current",
        name="current-backend",
    )
    registry = registry_for((incompatible, compatible))

    decision = registry.select("mock", environment={})
    records = {
        record.plugin_id: record for record in registry.list()
    }

    assert decision.plugin_id == "vendor.current"
    assert (
        records["vendor.future"].state
        is PluginLifecycleState.REJECTED
    )
    assert [incompatible.entry_points[0].load_calls,
            compatible.entry_points[0].load_calls] == [0, 1]


def test_explicit_triton_mismatch_preserves_original_diagnostic(tmp_path):
    incompatible = distribution_for(
        tmp_path,
        "future",
        name="future-backend",
        declaration=triton_plugin("future", triton_version=">=9.0"),
    )
    registry = registry_for((incompatible,))

    with pytest.raises(BackendPluginCompatibilityError) as automatic:
        registry.select("mock", environment={})
    with pytest.raises(BackendPluginCompatibilityError) as explicit:
        registry.select(
            "mock",
            explicit_selector="vendor.future",
            environment={},
        )

    assert automatic.value.field == "Triton version"
    assert explicit.value.field == "Triton version"
    assert incompatible.entry_points[0].load_calls == 0


def test_default_registry_profile_checks_full_protocol_before_import(tmp_path):
    declaration = triton_plugin("staged")
    declaration.update(
        {
            "backend_protocol": ">=9.0",
            "requires_core": ">=9.0",
            "requires_llvm_version": ">=99.0",
            "requires_mlir_version": ">=99.0",
        }
    )
    distribution = FakeDistribution(
        tmp_path,
        name="staged-backend",
        manifest=manifest(declaration),
        entry_points=(("staged", RuntimePlugin()),),
        wheel_text="Wheel-Version: 1.0\nTag: py3-none-any\n",
    )
    registry = registry_for((distribution,))

    with pytest.raises(BackendPluginProtocolError) as caught:
        registry.select("mock", environment={})

    assert caught.value.field == "backend_protocol"
    assert distribution.entry_points[0].load_calls == 0


def test_explicit_triton_version_migration_profile_remains_frozen(tmp_path):
    declaration = triton_plugin("staged")
    declaration.update(
        {
            "backend_protocol": ">=9.0",
            "requires_core": ">=9.0",
            "requires_llvm_version": ">=99.0",
            "requires_mlir_version": ">=99.0",
        }
    )
    distribution = FakeDistribution(
        tmp_path,
        name="staged-backend",
        manifest=manifest(declaration),
        entry_points=(("staged", RuntimePlugin()),),
        wheel_text="Wheel-Version: 1.0\nTag: cp39-cp39-win_amd64\n",
    )
    registry = registry_for(
        (distribution,),
        preflight_profile="triton_version",
    )

    decision = registry.select("mock", environment={})

    assert [
        check.dimension
        for check in decision.record.compatibility_report.checks
    ] == ["Triton version"]
    assert distribution.entry_points[0].load_calls == 1


@pytest.mark.parametrize(
    ("isolation_mode", "extra"),
    (
        (
            "native_in_process",
            {
                "native_libraries": ["vendor_backend/libunchecked.so"],
                "abi_fingerprint": TEST_ABI_FINGERPRINT,
            },
        ),
        ("subprocess", {}),
    ),
)
def test_staged_profile_rejects_unchecked_non_python_isolation_before_import(
    tmp_path,
    isolation_mode,
    extra,
):
    declaration = triton_plugin("unchecked")
    declaration.update(
        {
            "isolation_mode": isolation_mode,
            **extra,
        }
    )
    distribution = distribution_for(
        tmp_path,
        "unchecked",
        name=f"{isolation_mode}-backend",
        declaration=declaration,
    )
    registry = registry_for(
        (distribution,),
        preflight_profile="triton_version",
    )

    with pytest.raises(BackendPluginCompatibilityError) as caught:
        registry.select("mock", environment={})

    assert caught.value.dimension == (
        "isolation mode for triton_version profile"
    )
    assert caught.value.expected == "python_only"
    assert caught.value.actual == isolation_mode
    assert (
        registry.list()[0].state
        is PluginLifecycleState.REJECTED
    )
    assert distribution.entry_points[0].load_calls == 0


def test_capability_filter_beats_priority_without_loading_loser(tmp_path):
    incapable = distribution_for(
        tmp_path,
        "incapable",
        name="incapable-backend",
        declaration=triton_plugin(
            "incapable",
            capabilities=("dtype.fp16",),
            priority=100,
        ),
    )
    capable = distribution_for(
        tmp_path,
        "capable",
        name="capable-backend",
        declaration=triton_plugin(
            "capable",
            capabilities=("runtime.launch",),
            requires_capabilities=("core.anchor_ir",),
            priority=1,
        ),
    )
    registry = registry_for(
        (incapable, capable),
        core_capabilities=("core.anchor_ir",),
    )

    decision = registry.select(
        "mock",
        kernel_required_capabilities=("runtime.launch",),
        environment={},
    )

    assert decision.plugin_id == "vendor.capable"
    assert decision.capability_report.compatible
    assert [incapable.entry_points[0].load_calls,
            capable.entry_points[0].load_calls] == [0, 1]


def test_kernel_capability_failure_keeps_record_validated_and_unloaded(
    tmp_path,
):
    distribution = distribution_for(
        tmp_path,
        "limited",
        name="limited-backend",
        declaration=triton_plugin(
            "limited",
            capabilities=("dtype.fp16",),
        ),
    )
    registry = registry_for((distribution,))

    with pytest.raises(BackendPluginCapabilityError) as caught:
        registry.select(
            "mock",
            kernel_required_capabilities=("runtime.launch",),
            environment={},
        )

    record = registry.list()[0]
    assert caught.value.field == "kernel_required_capabilities"
    assert caught.value.missing_kernel_capabilities == (
        "runtime.launch",
    )
    assert record.state is PluginLifecycleState.VALIDATED
    assert distribution.entry_points[0].load_calls == 0


def test_missing_plugin_requirement_fails_before_import(tmp_path):
    distribution = distribution_for(
        tmp_path,
        "needs_core",
        name="needs-core-backend",
        declaration=triton_plugin(
            "needs_core",
            requires_capabilities=("core.anchor_ir",),
        ),
    )
    registry = registry_for((distribution,))
    record = registry.discover()[0]

    with pytest.raises(BackendPluginCapabilityError) as caught:
        registry.load(record.record_id)

    assert caught.value.missing_capabilities == ("core.anchor_ir",)
    assert caught.value.field == "requires_capabilities"
    assert caught.value.missing_plugin_capabilities == (
        "core.anchor_ir",
    )
    assert (
        registry.inspect(record.record_id).state
        is PluginLifecycleState.REJECTED
    )
    assert distribution.entry_points[0].load_calls == 0


def test_fatal_identity_conflict_blocks_every_import(tmp_path):
    first = distribution_for(
        tmp_path,
        "first",
        name="first-backend",
        declaration=triton_plugin(
            "first",
            plugin_id="vendor.duplicate",
        ),
    )
    second = distribution_for(
        tmp_path,
        "second",
        name="second-backend",
        declaration=triton_plugin(
            "second",
            plugin_id="vendor.duplicate",
        ),
    )
    registry = registry_for((second, first))

    with pytest.raises(BackendPluginConflictError):
        registry.select(
            "mock",
            explicit_selector=registry.discover()[0].record_id,
            environment={},
        )

    assert [first.entry_points[0].load_calls,
            second.entry_points[0].load_calls] == [0, 0]


def test_equal_priority_is_deterministic_and_import_free(tmp_path):
    alpha = distribution_for(
        tmp_path,
        "alpha",
        name="alpha-backend",
    )
    beta = distribution_for(
        tmp_path,
        "beta",
        name="beta-backend",
    )
    registry = registry_for((beta, alpha))

    with pytest.raises(BackendPluginSelectionError) as caught:
        registry.select("mock", environment={})

    assert caught.value.field == "priority"
    assert caught.value.actual == "alpha-backend:alpha, beta-backend:beta"
    assert [alpha.entry_points[0].load_calls,
            beta.entry_points[0].load_calls] == [0, 0]


def test_python_then_environment_selection_switches_state_deterministically(
    tmp_path,
):
    low = distribution_for(
        tmp_path,
        "low",
        name="low-backend",
        declaration=triton_plugin("low", priority=1),
    )
    high = distribution_for(
        tmp_path,
        "high",
        name="high-backend",
        declaration=triton_plugin("high", priority=2),
    )
    registry = registry_for((high, low))

    explicit = registry.select(
        "mock",
        explicit_selector="vendor.low",
        environment={BACKEND_SELECTOR_ENV: "vendor.high"},
    )
    assert explicit.plugin_id == "vendor.low"
    assert explicit.method is SelectionMethod.PYTHON_EXPLICIT

    selected = registry.select(
        "mock",
        environment={BACKEND_SELECTOR_ENV: "vendor.high"},
    )
    records = {record.plugin_id: record for record in registry.list()}

    assert selected.plugin_id == "vendor.high"
    assert selected.method is SelectionMethod.ENVIRONMENT
    assert records["vendor.low"].state is PluginLifecycleState.REGISTERED
    assert records["vendor.low"].selected_targets == ()
    assert records["vendor.high"].state is PluginLifecycleState.SELECTED
    assert records["vendor.high"].selected_targets == ("mock",)
    assert [low.entry_points[0].load_calls, high.entry_points[0].load_calls] == [
        1,
        1,
    ]

    repeated = registry.select(
        "mock",
        environment={BACKEND_SELECTOR_ENV: "vendor.high"},
    )
    assert repeated.record_id == selected.record_id
    assert high.entry_points[0].load_calls == 1


def test_registry_uses_environment_variable_when_mapping_is_omitted(
    tmp_path,
    monkeypatch,
):
    alpha = distribution_for(
        tmp_path,
        "alpha",
        name="alpha-backend",
    )
    beta = distribution_for(
        tmp_path,
        "beta",
        name="beta-backend",
    )
    registry = registry_for((alpha, beta))
    monkeypatch.setenv(BACKEND_SELECTOR_ENV, "vendor.beta")

    decision = registry.select("mock")

    assert decision.plugin_id == "vendor.beta"
    assert decision.method is SelectionMethod.ENVIRONMENT
    assert [alpha.entry_points[0].load_calls,
            beta.entry_points[0].load_calls] == [0, 1]


def test_w9_active_target_cannot_switch_without_runtime_reset(tmp_path):
    low = distribution_for(
        tmp_path,
        "low",
        name="low-backend",
    )
    high = distribution_for(
        tmp_path,
        "high",
        name="high-backend",
    )
    registry = registry_for((low, high))
    first = registry.select(
        "mock",
        explicit_selector="vendor.low",
        environment={},
    )
    registry._replace(
        replace(first.record, state=PluginLifecycleState.ACTIVE)
    )

    with pytest.raises(BackendPluginLifecycleError) as caught:
        registry.select(
            "mock",
            explicit_selector="vendor.high",
            environment={},
        )

    previous = registry.inspect(first.record_id)
    assert caught.value.field == "active selection"
    assert previous.state is PluginLifecycleState.ACTIVE
    assert previous.selected_targets == ("mock",)
    assert high.entry_points[0].load_calls == 0


def test_one_plugin_can_be_selected_for_multiple_declared_targets(tmp_path):
    distribution = distribution_for(
        tmp_path,
        "multi",
        name="multi-backend",
        declaration=triton_plugin(
            "multi",
            targets=("target-a", "target-b"),
        ),
    )
    registry = registry_for((distribution,))

    registry.select("target-b", environment={})
    decision = registry.select("target-a", environment={})

    assert decision.record.selected_targets == ("target-a", "target-b")
    assert distribution.entry_points[0].load_calls == 1
    assert set(registry.diagnostics()["selections"]) == {
        "target-a",
        "target-b",
    }
