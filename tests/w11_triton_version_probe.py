"""Fresh-process probe for the W11 Triton-only real-wheel matrix."""

import argparse
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys
import tempfile


COMPATIBLE_DISTRIBUTION = (
    "triton-anchor-w11-compatible-backend"
)
INCOMPATIBLE_DISTRIBUTION = (
    "triton-anchor-w11-incompatible-backend"
)
MALFORMED_DISTRIBUTION = "triton-anchor-w11-malformed-backend"

COMPATIBLE_MODULE = "w11_compatible_backend"
INCOMPATIBLE_MODULE = "w11_incompatible_backend"
MALFORMED_MODULE = "w11_malformed_backend"


def _assert_installed_wheel(distribution_name):
    distribution = metadata.distribution(distribution_name)
    files = tuple(distribution.files or ())
    manifest_files = tuple(
        item
        for item in files
        if item.name == "triton_anchor_backend.json"
    )
    assert len(manifest_files) == 1
    assert any(item.name == "entry_points.txt" for item in files)
    assert any(item.name == "RECORD" for item in files)
    backend_entry_points = tuple(
        entry_point
        for entry_point in distribution.entry_points
        if entry_point.group == "triton.backends"
    )
    assert len(backend_entry_points) == 1
    return distribution, backend_entry_points[0]


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


def _assert_bad_plugin_not_imported(module_name, marker):
    assert module_name not in sys.modules
    assert not marker.exists()


def _common_environment_checks(triton, triton_anchor):
    from triton_anchor.backends import collect_core_environment

    _assert_module_origins(triton, triton_anchor)
    environment = collect_core_environment()
    assert environment.build_info_generated is True
    assert environment.triton_version == triton.__version__ == "3.0.0"
    assert triton_anchor.__version__ == "0.2.0"
    assert environment.actual_llvm_version == "19.0.0"
    assert environment.actual_mlir_version == "19.0.0"
    assert environment.actual_llvm_commit
    assert environment.actual_mlir_commit
    assert environment.cxx11_abi in ("0", "1")
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        environment.core_abi_fingerprint or "",
    )
    return environment


def _run_mixed(marker_dir):
    distribution_names = (
        COMPATIBLE_DISTRIBUTION,
        INCOMPATIBLE_DISTRIBUTION,
        MALFORMED_DISTRIBUTION,
    )
    entry_points = {
        name: _assert_installed_wheel(name)[1]
        for name in distribution_names
    }
    original_load, load_calls = _install_entry_point_counter(
        tuple(entry_point.name for entry_point in entry_points.values())
    )
    compatible_marker = marker_dir / "compatible.imported"
    incompatible_marker = marker_dir / "incompatible.imported"
    malformed_marker = marker_dir / "malformed.imported"

    try:
        triton = importlib.import_module("triton")
        triton_anchor = importlib.import_module("triton_anchor")
        environment = _common_environment_checks(triton, triton_anchor)
        _assert_bad_plugin_not_imported(
            INCOMPATIBLE_MODULE,
            incompatible_marker,
        )
        _assert_bad_plugin_not_imported(
            MALFORMED_MODULE,
            malformed_marker,
        )
        assert COMPATIBLE_MODULE not in sys.modules
        assert not compatible_marker.exists()
        assert all(count == 0 for count in load_calls.values())

        from triton.compiler.compiler import make_backend
        from triton.backends.compiler import GPUTarget
        from triton.runtime.driver import _create_driver
        from triton_anchor.backends import (
            BackendPluginCompatibilityError,
            BackendPluginManifestError,
            PluginLifecycleState,
            get_backend_plugin_registry,
        )

        registry = get_backend_plugin_registry()
        records = registry.list()
        compatible = _record_by_distribution(
            records,
            COMPATIBLE_DISTRIBUTION,
        )
        incompatible = _record_by_distribution(
            records,
            INCOMPATIBLE_DISTRIBUTION,
        )
        malformed = _record_by_distribution(
            records,
            MALFORMED_DISTRIBUTION,
        )

        assert compatible.state is PluginLifecycleState.DISCOVERED
        assert incompatible.state is PluginLifecycleState.DISCOVERED
        assert malformed.state is PluginLifecycleState.REJECTED
        assert isinstance(malformed.error, BackendPluginManifestError)
        assert malformed.error.field == "requires_triton"
        assert all(count == 0 for count in load_calls.values())
        _assert_bad_plugin_not_imported(
            INCOMPATIBLE_MODULE,
            incompatible_marker,
        )
        _assert_bad_plugin_not_imported(
            MALFORMED_MODULE,
            malformed_marker,
        )

        target = GPUTarget("w11_mock", 1, 1)
        compiler = make_backend(target)
        records = registry.list()
        compatible = _record_by_distribution(
            records,
            COMPATIBLE_DISTRIBUTION,
        )
        incompatible = _record_by_distribution(
            records,
            INCOMPATIBLE_DISTRIBUTION,
        )
        malformed = _record_by_distribution(
            records,
            MALFORMED_DISTRIBUTION,
        )

        assert compatible.state is PluginLifecycleState.SELECTED
        assert [
            check.dimension
            for check in compatible.compatibility_report.checks
        ] == [
            "wheel platform tag",
            "Backend Plugin Protocol",
            "Triton version",
            "core Python SOABI",
            "core build platform",
        ]
        assert incompatible.state is PluginLifecycleState.REJECTED
        assert isinstance(
            incompatible.error,
            BackendPluginCompatibilityError,
        )
        assert incompatible.error.field == "Triton version"
        assert incompatible.error.expected == ">=9.0,<10.0"
        assert incompatible.error.actual == "3.0.0"
        assert incompatible.error.remediation
        assert malformed.state is PluginLifecycleState.REJECTED
        assert isinstance(malformed.error, BackendPluginManifestError)
        assert malformed.error.field == "requires_triton"

        assert load_calls["w11_compatible"] == 1
        assert load_calls["w11_incompatible"] == 0
        assert load_calls["w11_malformed"] == 0
        _assert_bad_plugin_not_imported(
            INCOMPATIBLE_MODULE,
            incompatible_marker,
        )
        _assert_bad_plugin_not_imported(
            MALFORMED_MODULE,
            malformed_marker,
        )

        runtime_driver = _create_driver()
        records = registry.list()
        compatible = _record_by_distribution(
            records,
            COMPATIBLE_DISTRIBUTION,
        )

        assert compatible.state is PluginLifecycleState.ACTIVE
        assert compiler.__class__ is compatible.compiler_cls
        assert runtime_driver.__class__ is compatible.driver_cls
        assert compiler.target == runtime_driver.get_current_target()
        assert compatible_marker.read_text(
            encoding="utf-8"
        ) == "imported\n"
        assert load_calls["w11_compatible"] == 1
        assert load_calls["w11_incompatible"] == 0
        assert load_calls["w11_malformed"] == 0
        _assert_bad_plugin_not_imported(
            INCOMPATIBLE_MODULE,
            incompatible_marker,
        )
        _assert_bad_plugin_not_imported(
            MALFORMED_MODULE,
            malformed_marker,
        )

        return {
            "mode": "mixed",
            "triton_version": environment.triton_version,
            "active_record": compatible.record_id,
            "compatibility_dimensions": [
                check.dimension
                for check in compatible.compatibility_report.checks
            ],
            "entry_point_load_calls": load_calls,
        }
    finally:
        metadata.EntryPoint.load = original_load


def _run_incompatible_only(marker_dir):
    _, entry_point = _assert_installed_wheel(
        INCOMPATIBLE_DISTRIBUTION
    )
    original_load, load_calls = _install_entry_point_counter(
        (entry_point.name,)
    )
    marker = marker_dir / "incompatible.imported"

    try:
        triton = importlib.import_module("triton")
        triton_anchor = importlib.import_module("triton_anchor")
        environment = _common_environment_checks(triton, triton_anchor)
        _assert_bad_plugin_not_imported(INCOMPATIBLE_MODULE, marker)

        from triton.compiler.compiler import make_backend
        from triton.backends.compiler import GPUTarget
        from triton.runtime.driver import _create_driver
        from triton_anchor.backends import (
            BackendPluginCompatibilityError,
            PluginLifecycleState,
            get_backend_plugin_registry,
        )

        registry = get_backend_plugin_registry()
        record = _record_by_distribution(
            registry.list(),
            INCOMPATIBLE_DISTRIBUTION,
        )
        assert record.state is PluginLifecycleState.DISCOVERED
        assert record.error is None
        assert load_calls["w11_incompatible"] == 0

        target = GPUTarget("w11_mock", 1, 1)
        compiler_error = None
        driver_error = None
        try:
            make_backend(target)
        except BackendPluginCompatibilityError as error:
            compiler_error = error

        record = _record_by_distribution(
            registry.list(),
            INCOMPATIBLE_DISTRIBUTION,
        )
        assert record.state is PluginLifecycleState.REJECTED
        assert isinstance(record.error, BackendPluginCompatibilityError)
        assert record.error.field == "Triton version"
        assert record.error.expected == ">=9.0,<10.0"
        assert record.error.actual == "3.0.0"

        try:
            _create_driver()
        except BackendPluginCompatibilityError as error:
            driver_error = error

        assert compiler_error is record.error
        assert driver_error is record.error
        assert load_calls["w11_incompatible"] == 0
        _assert_bad_plugin_not_imported(INCOMPATIBLE_MODULE, marker)
        return {
            "mode": "incompatible-only",
            "triton_version": environment.triton_version,
            "error": record.error.to_dict(),
            "entry_point_load_calls": load_calls,
        }
    finally:
        metadata.EntryPoint.load = original_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("mixed", "incompatible-only"),
    )
    arguments = parser.parse_args()
    assert not os.environ.get("PYTHONPATH")

    with tempfile.TemporaryDirectory(
        prefix="triton-anchor-w11-markers."
    ) as marker_dir:
        marker_path = Path(marker_dir)
        os.environ["TRITON_ANCHOR_W11_MARKER_DIR"] = str(marker_path)
        if arguments.mode == "mixed":
            result = _run_mixed(marker_path)
        else:
            result = _run_incompatible_only(marker_path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
