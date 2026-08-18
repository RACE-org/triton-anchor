"""Guardrails for behavior-preserving BackendPluginRegistry refactors."""

import json
from pathlib import Path

import pytest

from triton_anchor.backends import (
    PluginLifecycleState,
    PluginSource,
    can_transition,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_PATH = REPO_ROOT / "acceptance" / "refactor_baseline.json"


MANIFEST_TRANSITIONS = {
    PluginLifecycleState.DISCOVERED: {
        PluginLifecycleState.VALIDATED,
        PluginLifecycleState.REJECTED,
    },
    PluginLifecycleState.VALIDATED: {
        PluginLifecycleState.LOADED,
        PluginLifecycleState.REJECTED,
    },
    PluginLifecycleState.LOADED: {
        PluginLifecycleState.REGISTERED,
        PluginLifecycleState.REJECTED,
    },
    PluginLifecycleState.REGISTERED: {
        PluginLifecycleState.SELECTED,
        PluginLifecycleState.REJECTED,
    },
    PluginLifecycleState.SELECTED: {
        PluginLifecycleState.ACTIVE,
        PluginLifecycleState.REGISTERED,
        PluginLifecycleState.REJECTED,
    },
    PluginLifecycleState.ACTIVE: {
        PluginLifecycleState.REGISTERED,
        PluginLifecycleState.REJECTED,
    },
    PluginLifecycleState.REJECTED: set(),
}


@pytest.mark.parametrize("source", [PluginSource.MANIFEST, PluginSource.LEGACY])
def test_complete_lifecycle_transition_matrix_is_frozen(source):
    expected = {state: set(targets) for state, targets in MANIFEST_TRANSITIONS.items()}
    if source is PluginSource.LEGACY:
        expected[PluginLifecycleState.DISCOVERED].add(PluginLifecycleState.LOADED)

    for current in PluginLifecycleState:
        actual = {
            target
            for target in PluginLifecycleState
            if can_transition(current, target, source)
        }
        assert actual == expected[current]


def test_refactor_baseline_manifest_is_self_contained_and_t10_2_free():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    assert baseline["source_checkpoint"] == (
        "893761dc862f5e59f04d4e3b25db54ecff1125c1"
    )
    assert baseline["production_baseline"] == (
        "b85a506656bf58b70e6093e7d48adcd6a2908f64"
    )
    assert [item["id"] for item in baseline["scope"]["excluded"]] == [
        "T10.2",
        "protocol_field_evolution_matrix",
        "positive_subprocess_contract",
        "production_structure_changes",
    ]

    fast_gate = baseline["fast_gate"]
    assert fast_gate["environment"] == {
        "PYTHONPATH": "python",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    assert fast_gate["argv"][2:4] == ["pytest", "python/triton_anchor/tests"]
    for node_id in fast_gate["contract_tests"]:
        path, separator, test_name = node_id.partition("::")
        assert separator and test_name.startswith("test_")
        assert (REPO_ROOT / path).is_file()
        assert "t10.2" not in node_id.lower()

    assert {gate["id"] for gate in baseline["slow_gates"]} == {
        "ac1_rejection_matrix",
        "ac2_wheel_coexistence",
        "ac3_real_compile",
    }
    for gate in baseline["slow_gates"]:
        command_path = gate["argv"][1]
        assert (REPO_ROOT / command_path).is_file()
        assert "t10.2" not in " ".join(gate["argv"]).lower()

    results = baseline["recorded_results"]
    assert results["fast_gate"] == {"status": "passed", "tests_passed": 256}
    assert results["ac1_rejection_matrix"]["artifact_case_count"] == 33
    assert results["ac1_rejection_matrix"][
        "static_rejection_entry_point_load_calls"
    ] == 0
    assert results["ac2_wheel_coexistence"][
        "final_entry_point_load_calls_per_plugin"
    ] == 1
    assert results["ac3_real_compile"]["compile_success"] is True
    assert results["ac3_real_compile"]["entry_point_load_calls"] == 1
