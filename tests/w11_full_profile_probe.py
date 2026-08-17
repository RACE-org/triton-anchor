"""Fresh-process probe for the W11 full-profile real-wheel matrix."""

import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import tempfile


TRITON_COMMIT = "757b6a61e7df814ba806f498f8bb3160f84b120c"
TOOLCHAIN_COMMIT = "10dc3a8e916d73291269e5e2b82dd22681489aa1"

COMPATIBLE = {
    "distribution": "triton-anchor-w11-full-compatible-backend",
    "entry_point": "w11_full_compatible",
    "module": "w11_full_compatible_backend",
    "marker": "full_compatible.imported",
}

BAD_CASES = (
    {
        "case": "bad_protocol",
        "distribution": "triton-anchor-w11-bad-protocol-backend",
        "entry_point": "w11_bad_protocol",
        "module": "w11_bad_protocol_backend",
        "marker": "bad_protocol.imported",
        "error_code": "backend_plugin_protocol_error",
        "field": "backend_protocol",
        "expected": ">=9.0,<10.0",
        "actual_environment_field": "backend_protocol_version",
    },
    {
        "case": "bad_core",
        "distribution": "triton-anchor-w11-bad-core-backend",
        "entry_point": "w11_bad_core",
        "module": "w11_bad_core_backend",
        "marker": "bad_core.imported",
        "error_code": "backend_plugin_compatibility_error",
        "field": "triton-anchor Core version",
        "expected": ">=9.0,<10.0",
        "actual_environment_field": "core_version",
    },
    {
        "case": "bad_triton_commit",
        "distribution": "triton-anchor-w11-bad-triton-commit-backend",
        "entry_point": "w11_bad_triton_commit",
        "module": "w11_bad_triton_commit_backend",
        "marker": "bad_triton_commit.imported",
        "error_code": "backend_plugin_compatibility_error",
        "field": "vendored Triton commit",
        "expected": "0000000000000000000000000000000000000001",
        "actual_environment_field": "vendored_triton_commit",
    },
    {
        "case": "bad_llvm_version",
        "distribution": "triton-anchor-w11-bad-llvm-version-backend",
        "entry_point": "w11_bad_llvm_version",
        "module": "w11_bad_llvm_version_backend",
        "marker": "bad_llvm_version.imported",
        "error_code": "backend_plugin_compatibility_error",
        "field": "LLVM version",
        "expected": ">=20.0,<21.0",
        "actual_environment_field": "actual_llvm_version",
    },
    {
        "case": "bad_llvm_commit",
        "distribution": "triton-anchor-w11-bad-llvm-commit-backend",
        "entry_point": "w11_bad_llvm_commit",
        "module": "w11_bad_llvm_commit_backend",
        "marker": "bad_llvm_commit.imported",
        "error_code": "backend_plugin_compatibility_error",
        "field": "LLVM commit",
        "expected": "0000000000000000000000000000000000000002",
        "actual_environment_field": "actual_llvm_commit",
    },
    {
        "case": "bad_mlir_version",
        "distribution": "triton-anchor-w11-bad-mlir-version-backend",
        "entry_point": "w11_bad_mlir_version",
        "module": "w11_bad_mlir_version_backend",
        "marker": "bad_mlir_version.imported",
        "error_code": "backend_plugin_compatibility_error",
        "field": "MLIR version",
        "expected": ">=20.0,<21.0",
        "actual_environment_field": "actual_mlir_version",
    },
    {
        "case": "bad_mlir_commit",
        "distribution": "triton-anchor-w11-bad-mlir-commit-backend",
        "entry_point": "w11_bad_mlir_commit",
        "module": "w11_bad_mlir_commit_backend",
        "marker": "bad_mlir_commit.imported",
        "error_code": "backend_plugin_compatibility_error",
        "field": "MLIR commit",
        "expected": "0000000000000000000000000000000000000003",
        "actual_environment_field": "actual_mlir_commit",
    },
)


def _assert_installed_wheel(case):
    distribution = metadata.distribution(case["distribution"])
    files = tuple(distribution.files or ())
    manifests = tuple(
        item for item in files if item.name == "triton_anchor_backend.json"
    )
    assert len(manifests) == 1
    assert distribution.locate_file(manifests[0]).is_file()
    assert any(item.name == "entry_points.txt" for item in files)
    assert any(item.name == "WHEEL" for item in files)
    assert any(item.name == "RECORD" for item in files)
    backend_entry_points = tuple(
        entry_point
        for entry_point in distribution.entry_points
        if entry_point.group == "triton.backends"
    )
    assert len(backend_entry_points) == 1
    assert backend_entry_points[0].name == case["entry_point"]
    return backend_entry_points[0]


def _assert_module_origins(triton, triton_anchor):
    environment_root = Path(sys.prefix).resolve()
    for module in (triton, triton_anchor):
        origin = Path(module.__file__).resolve()
        origin.relative_to(environment_root)


def _record_by_distribution(records, distribution_name):
    matches = tuple(
        record
        for record in records
        if record.distribution_name == distribution_name
    )
    assert len(matches) == 1
    return matches[0]


def _install_entry_point_counter(entry_point_names):
    original_load = metadata.EntryPoint.load
    load_calls = {name: 0 for name in entry_point_names}

    def counted_load(entry_point):
        if entry_point.name in load_calls:
            load_calls[entry_point.name] += 1
        return original_load(entry_point)

    metadata.EntryPoint.load = counted_load
    return original_load, load_calls


def _assert_not_imported(case, marker_dir):
    assert case["module"] not in sys.modules
    assert not (marker_dir / case["marker"]).exists()


def _assert_environment(environment, triton, triton_anchor):
    assert environment.build_info_generated is True
    assert environment.core_version == triton_anchor.__version__ == "0.2.0"
    assert environment.backend_protocol_version == "1.0"
    assert environment.triton_version == triton.__version__ == "3.0.0"
    assert environment.vendored_triton_commit == TRITON_COMMIT
    assert environment.actual_llvm_version == "19.0.0"
    assert environment.actual_llvm_commit == TOOLCHAIN_COMMIT
    assert environment.actual_mlir_version == "19.0.0"
    assert environment.actual_mlir_commit == TOOLCHAIN_COMMIT


def _assert_compatible_report(record, environment):
    assert record.compatibility_report is not None
    report = record.compatibility_report
    checks = {check.dimension: check for check in report.checks}
    expected_actuals = {
        "Backend Plugin Protocol": environment.backend_protocol_version,
        "triton-anchor Core version": environment.core_version,
        "Triton version": environment.triton_version,
        "vendored Triton commit": environment.vendored_triton_commit,
        "LLVM version": environment.actual_llvm_version,
        "LLVM commit": environment.actual_llvm_commit,
        "MLIR version": environment.actual_mlir_version,
        "MLIR commit": environment.actual_mlir_commit,
        "core Python SOABI": environment.runtime_python_soabi,
        "core build platform": environment.runtime_platform,
    }
    assert set(expected_actuals).issubset(checks)
    for dimension, actual in expected_actuals.items():
        assert checks[dimension].actual == actual
    assert "wheel platform tag" in checks
    assert report.native_artifacts == ()


def _assert_rejected(case, record, environment):
    from triton_anchor.backends import PluginLifecycleState

    assert record.state is PluginLifecycleState.REJECTED
    assert record.error is not None
    error = record.error.to_dict()
    assert error["code"] == case["error_code"]
    assert error["field"] == case["field"]
    assert error["expected"] == case["expected"]
    actual = getattr(environment, case["actual_environment_field"])
    assert error["actual"] == actual
    assert error["remediation"]
    return error


def _run(marker_dir):
    cases = (COMPATIBLE,) + BAD_CASES
    entry_points = {
        case["entry_point"]: _assert_installed_wheel(case)
        for case in cases
    }
    original_load, load_calls = _install_entry_point_counter(
        tuple(entry_points)
    )

    try:
        for case in cases:
            _assert_not_imported(case, marker_dir)
        assert all(count == 0 for count in load_calls.values())

        triton = importlib.import_module("triton")
        triton_anchor = importlib.import_module("triton_anchor")
        _assert_module_origins(triton, triton_anchor)

        from triton_anchor.backends import (
            PluginLifecycleState,
            collect_core_environment,
            get_backend_plugin_registry,
        )

        environment = collect_core_environment()
        _assert_environment(environment, triton, triton_anchor)
        registry = get_backend_plugin_registry()
        assert registry.diagnostics()["preflight_profile"] == "full"
        assert all(count == 0 for count in load_calls.values())

        records = registry.validate()
        compatible = _record_by_distribution(
            records, COMPATIBLE["distribution"]
        )
        assert compatible.state is PluginLifecycleState.VALIDATED
        _assert_compatible_report(compatible, environment)

        rejected = {}
        for case in BAD_CASES:
            record = _record_by_distribution(
                records, case["distribution"]
            )
            rejected[case["case"]] = _assert_rejected(
                case, record, environment
            )
            assert load_calls[case["entry_point"]] == 0
            _assert_not_imported(case, marker_dir)

        assert all(count == 0 for count in load_calls.values())
        assert COMPATIBLE["module"] not in sys.modules
        assert not (marker_dir / COMPATIBLE["marker"]).exists()

        compatible = registry.load(compatible.record_id)
        assert compatible.state is PluginLifecycleState.LOADED
        assert load_calls[COMPATIBLE["entry_point"]] == 1
        assert COMPATIBLE["module"] in sys.modules
        assert (
            marker_dir / COMPATIBLE["marker"]
        ).read_text(encoding="utf-8") == "imported\n"

        for case in BAD_CASES:
            assert load_calls[case["entry_point"]] == 0
            _assert_not_imported(case, marker_dir)

        return {
            "mode": "full-profile-real-wheel-matrix",
            "preflight_profile": "full",
            "environment": {
                "core_version": environment.core_version,
                "backend_protocol_version": (
                    environment.backend_protocol_version
                ),
                "triton_version": environment.triton_version,
                "vendored_triton_commit": (
                    environment.vendored_triton_commit
                ),
                "llvm_version": environment.actual_llvm_version,
                "llvm_commit": environment.actual_llvm_commit,
                "mlir_version": environment.actual_mlir_version,
                "mlir_commit": environment.actual_mlir_commit,
            },
            "compatible_record": compatible.record_id,
            "compatible_dimensions": [
                check.dimension
                for check in compatible.compatibility_report.checks
            ],
            "rejected": rejected,
            "entry_point_load_calls": load_calls,
        }
    finally:
        metadata.EntryPoint.load = original_load


def main():
    assert not os.environ.get("PYTHONPATH")
    with tempfile.TemporaryDirectory(
        prefix="triton-anchor-w11-full-markers."
    ) as marker_dir:
        marker_path = Path(marker_dir)
        os.environ["TRITON_ANCHOR_W11_FULL_MARKER_DIR"] = str(marker_path)
        result = _run(marker_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
