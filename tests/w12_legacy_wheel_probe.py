"""Fresh-process proof for installed, no-Manifest Legacy backend wheels."""

from __future__ import annotations

import argparse
from collections import Counter
from email.parser import Parser
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys


CASES = {
    "legacy_good": {
        "distribution": "triton-anchor-w12-legacy-good-backend",
        "module": "w12_legacy_good_backend",
        "entry_point": "w12_legacy_good",
    },
    "legacy_bad_interface": {
        "distribution": (
            "triton-anchor-w12-legacy-bad-interface-backend"
        ),
        "module": "w12_legacy_bad_interface_backend",
        "entry_point": "w12_legacy_bad_interface",
        "invalid_fields": ["driver_cls"],
        "missing_fields": [],
    },
    "legacy_missing_compiler": {
        "distribution": "triton-anchor-w12-legacy-missing-compiler-backend",
        "module": "w12_legacy_missing_compiler_backend",
        "entry_point": "w12_legacy_missing_compiler",
        "invalid_fields": [],
        "missing_fields": ["compiler_cls"],
    },
    "legacy_illegal_compiler": {
        "distribution": "triton-anchor-w12-legacy-illegal-compiler-backend",
        "module": "w12_legacy_illegal_compiler_backend",
        "entry_point": "w12_legacy_illegal_compiler",
        "invalid_fields": ["compiler_cls"],
        "missing_fields": [],
    },
    "legacy_missing_driver": {
        "distribution": "triton-anchor-w12-legacy-missing-driver-backend",
        "module": "w12_legacy_missing_driver_backend",
        "entry_point": "w12_legacy_missing_driver",
        "invalid_fields": [],
        "missing_fields": ["driver_cls"],
    },
}
MODES = (
    "discovery",
    "good-runtime",
    "bad-register",
    "bad-compiler",
    "bad-runtime",
)
MARKER_ENV = "TRITON_ANCHOR_W12_LEGACY_MARKER_DIR"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pip_freeze():
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(
        line for line in completed.stdout.splitlines() if line.strip()
    )


def _installed_wheel(case_name, wheel_path):
    case = CASES[case_name]
    distribution = metadata.distribution(case["distribution"])
    files = tuple(distribution.files or ())
    manifests = tuple(
        item
        for item in files
        if item.name == "triton_anchor_backend.json"
    )
    native_files = tuple(
        item
        for item in files
        if str(item).lower().endswith(
            (".so", ".dylib", ".dll", ".pyd")
        )
    )
    entry_points = tuple(
        item
        for item in distribution.entry_points
        if item.group == "triton.backends"
    )
    wheel = Parser().parsestr(distribution.read_text("WHEEL") or "")
    tags = tuple(wheel.get_all("Tag", ()))
    module_relative = Path(case["module"]) / "__init__.py"
    module_path = Path(
        distribution.locate_file(module_relative)
    ).resolve()
    direct_url = json.loads(
        distribution.read_text("direct_url.json") or "{}"
    )

    assert wheel_path.is_file()
    assert not manifests
    assert not native_files
    assert len(entry_points) == 1
    assert entry_points[0].name == case["entry_point"]
    assert entry_points[0].value == case["module"]
    assert any(item.name == "RECORD" for item in files)
    assert any(item.name == "entry_points.txt" for item in files)
    assert (wheel.get("Root-Is-Purelib") or "").lower() == "true"
    assert "py3-none-any" in tags
    assert module_path.is_file()

    return {
        "distribution": distribution,
        "entry_point": entry_points[0],
        "module_path": module_path,
        "wheel": {
            "path": str(wheel_path),
            "sha256": _sha256(wheel_path),
            "root_is_purelib": True,
            "tags": list(tags),
            "manifest_count": len(manifests),
            "native_file_count": len(native_files),
            "direct_url": direct_url,
        },
    }


def _install_load_counter(entry_point):
    original = metadata.EntryPoint.load
    calls = {entry_point.name: 0}

    def counted_load(current):
        if current.name in calls:
            calls[current.name] += 1
        return original(current)

    metadata.EntryPoint.load = counted_load
    return original, calls


def _events(path):
    if not path.exists():
        return ()
    return tuple(
        json.loads(line)["event"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _record_snapshot(record):
    return {
        "record_id": record.record_id,
        "registry_key": record.registry_key,
        "source": record.source.value,
        "state": record.state.value,
        "compatibility_status": record.compatibility_status.value,
        "manifest": record.manifest is not None,
        "compatibility_report": (
            record.compatibility_report is not None
        ),
        "plugin_object_loaded": record.plugin_object is not None,
        "compiler_class_available": record.compiler_cls is not None,
        "driver_class_available": record.driver_cls is not None,
        "initialized": record.initialized,
        "selected_targets": list(record.selected_targets),
    }


def _record_by_distribution(registry, distribution_name):
    matches = tuple(
        record
        for record in registry.list()
        if record.distribution_name == distribution_name
    )
    assert len(matches) == 1
    return matches[0]


def _structured_error(error):
    if error is None:
        return None
    result = (
        error.to_dict()
        if hasattr(error, "to_dict")
        else {"message": str(error)}
    )
    result["exception_type"] = type(error).__name__
    return result


def _common_result(
    *,
    case_name,
    installed,
    record,
    load_calls,
    marker_path,
):
    case = CASES[case_name]
    module = sys.modules.get(case["module"])
    module_origin = (
        str(Path(module.__file__).resolve())
        if module is not None and getattr(module, "__file__", None)
        else None
    )
    return {
        "case": case_name,
        "distribution": case["distribution"],
        "entry_point": case["entry_point"],
        "wheel": installed["wheel"],
        "installed_module_path": str(installed["module_path"]),
        "imported_module_origin": module_origin,
        "module_in_sys_modules": module is not None,
        "entry_point_load_calls": dict(load_calls),
        "events": list(_events(marker_path)),
        "event_counts": dict(Counter(_events(marker_path))),
        "record": _record_snapshot(record),
        "pip_freeze": list(_pip_freeze()),
        "python": str(Path(sys.executable).resolve()),
        "sys_prefix": str(Path(sys.prefix).resolve()),
    }


def _assert_legacy_identity(result, *, state):
    record = result["record"]
    assert result["wheel"]["manifest_count"] == 0
    assert result["wheel"]["native_file_count"] == 0
    assert record["source"] == "legacy"
    assert record["state"] == state
    assert record["compatibility_status"] == "legacy_unverified"
    assert record["manifest"] is False
    assert record["compatibility_report"] is False


def _run_discovery(case_name, installed, load_calls, marker_path):
    from triton_anchor.backends import BackendPluginRegistry

    registry = BackendPluginRegistry()
    discovered = _record_by_distribution(
        registry,
        CASES[case_name]["distribution"],
    )
    before_validate = _record_snapshot(discovered)
    validated = registry.validate(discovered.record_id)
    result = _common_result(
        case_name=case_name,
        installed=installed,
        record=validated,
        load_calls=load_calls,
        marker_path=marker_path,
    )
    result.update(
        {
            "mode": "discovery",
            "pid": os.getpid(),
            "before_validate": before_validate,
            "registry_diagnostics": registry.diagnostics(
                validated.record_id
            ),
        }
    )
    return result


def _assert_discovery(result):
    _assert_legacy_identity(result, state="discovered")
    assert result["before_validate"] == result["record"]
    assert all(
        count == 0
        for count in result["entry_point_load_calls"].values()
    )
    assert result["events"] == []
    assert result["module_in_sys_modules"] is False
    assert result["imported_module_origin"] is None
    assert result["record"]["plugin_object_loaded"] is False


def _run_good_runtime(installed, load_calls, marker_path):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import make_backend
    from triton.runtime import driver
    from triton_anchor.backends import (
        BackendPluginSelectionError,
        get_backend_plugin_registry,
    )

    case = CASES["legacy_good"]
    registry = get_backend_plugin_registry()
    record = _record_by_distribution(registry, case["distribution"])
    target = GPUTarget(case["entry_point"], 1, 1)

    automatic_error = None
    try:
        registry.select(target, environment={})
    except BackendPluginSelectionError as error:
        automatic_error = error
    automatic_selection_snapshot = {
        "entry_point_load_calls": dict(load_calls),
        "events": list(_events(marker_path)),
        "record_state": registry.inspect(record.record_id).state.value,
    }

    decision = registry.select(
        target,
        explicit_selector=record.registry_key,
        environment={},
    )
    selected = registry.inspect(record.record_id)
    compiler = make_backend(target)
    compiler_result = compiler.compile()
    runtime_target = driver.active.get_current_target()
    active = registry.inspect(record.record_id)
    active_records = tuple(
        item
        for item in registry.list()
        if item.state.value == "active"
    )
    driver_object = driver.default._obj
    before_reset = _common_result(
        case_name="legacy_good",
        installed=installed,
        record=active,
        load_calls=load_calls,
        marker_path=marker_path,
    )
    reset_errors = registry.reset()
    after_reset = _record_by_distribution(
        registry,
        case["distribution"],
    )

    before_reset.update(
        {
            "mode": "good-runtime",
            "pid": os.getpid(),
            "automatic_selection_error": _structured_error(
                automatic_error
            ),
            "automatic_selection_snapshot": automatic_selection_snapshot,
            "decision_record_id": decision.record_id,
            "selected_record_id": selected.record_id,
            "compiler_record_class_identity": (
                compiler.__class__ is selected.compiler_cls
            ),
            "driver_record_class_identity": (
                driver_object is not None
                and driver_object.__class__ is active.driver_cls
            ),
            "compiler_result": compiler_result,
            "runtime_target_backend": runtime_target.backend,
            "active_record_count": len(active_records),
            "reset_errors": [
                _structured_error(error) for error in reset_errors
            ],
            "after_reset": _record_snapshot(after_reset),
            "runtime_proxy_cleared": driver.default._obj is None,
            "events_after_reset": list(_events(marker_path)),
            "event_counts_after_reset": dict(
                Counter(_events(marker_path))
            ),
        }
    )
    return before_reset


def _assert_good_runtime(result):
    _assert_legacy_identity(result, state="active")
    assert result["automatic_selection_error"] is not None
    assert (
        result["automatic_selection_error"]["code"]
        == "backend_plugin_selection_error"
    )
    assert result["automatic_selection_snapshot"] == {
        "entry_point_load_calls": {
            CASES["legacy_good"]["entry_point"]: 0
        },
        "events": [],
        "record_state": "discovered",
    }
    assert result["entry_point_load_calls"] == {
        CASES["legacy_good"]["entry_point"]: 1
    }
    counts = result["event_counts_after_reset"]
    assert counts == {
        "import": 1,
        "compiler_construct": 1,
        "compiler_call": 1,
        "driver_is_active": 1,
        "driver_construct": 1,
        "runtime_call": 2,
    }
    assert result["record"]["initialized"] is False
    assert result["compiler_record_class_identity"] is True
    assert result["driver_record_class_identity"] is True
    assert result["decision_record_id"] == result["record"]["record_id"]
    assert result["selected_record_id"] == result["record"]["record_id"]
    assert result["compiler_result"] == {
        "backend": CASES["legacy_good"]["entry_point"]
    }
    assert (
        result["runtime_target_backend"]
        == CASES["legacy_good"]["entry_point"]
    )
    assert result["active_record_count"] == 1
    assert result["reset_errors"] == []
    assert result["after_reset"]["state"] == "discovered"
    assert (
        result["after_reset"]["compatibility_status"]
        == "legacy_unverified"
    )
    assert result["runtime_proxy_cleared"] is True
    assert (
        result["imported_module_origin"]
        == result["installed_module_path"]
    )


def _run_bad_interface(mode, installed, load_calls, marker_path, case_name):
    from triton.backends.compiler import GPUTarget
    from triton.runtime import driver
    from triton_anchor.backends import (
        BACKEND_SELECTOR_ENV,
        get_backend_plugin_registry,
    )

    case = CASES[case_name]
    registry = get_backend_plugin_registry()
    record = _record_by_distribution(registry, case["distribution"])
    os.environ[BACKEND_SELECTOR_ENV] = record.registry_key
    target = GPUTarget(case["entry_point"], 1, 1)
    error = None
    returned = None
    try:
        if mode == "bad-register":
            returned = registry.register(record.registry_key)
        elif mode == "bad-compiler":
            from triton.compiler import make_backend

            returned = make_backend(target)
        else:
            returned = driver.active.get_current_target()
    except Exception as caught:
        error = caught

    rejected = registry.inspect(record.record_id)
    result = _common_result(
        case_name=case_name,
        installed=installed,
        record=rejected,
        load_calls=load_calls,
        marker_path=marker_path,
    )
    result.update(
        {
            "mode": mode,
            "pid": os.getpid(),
            "returned": (
                None if returned is None else type(returned).__name__
            ),
            "error": _structured_error(error),
            "runtime_proxy_object_is_none": driver.default._obj is None,
            "registry_diagnostics": registry.diagnostics(
                rejected.record_id
            ),
        }
    )
    return result


def _assert_bad_interface(result):
    case = CASES[result["case"]]
    _assert_legacy_identity(result, state="rejected")
    assert result["returned"] is None
    assert result["entry_point_load_calls"] == {
        case["entry_point"]: 1
    }
    assert result["event_counts"] == {"import": 1}
    assert result["module_in_sys_modules"] is True
    assert (
        result["imported_module_origin"]
        == result["installed_module_path"]
    )
    error = result["error"]
    assert error is not None
    assert error["exception_type"] == "BackendPluginInterfaceError"
    assert error["code"] == "backend_plugin_interface_error"
    assert all(
        field in error["field"]
        for field in case["invalid_fields"] + case["missing_fields"]
    )
    assert error["expected"]
    assert error["actual"]
    assert error["remediation"]
    assert error["invalid_fields"] == case["invalid_fields"]
    assert error["missing_fields"] == case["missing_fields"]
    assert result["record"]["initialized"] is False
    assert result["record"]["selected_targets"] == []
    assert result["record"]["compiler_class_available"] is False
    assert result["record"]["driver_class_available"] is False
    assert result["runtime_proxy_object_is_none"] is True


def run(mode, wheel_path, evidence_dir, case_name=None):
    case_name = (
        "legacy_good"
        if mode in {"discovery", "good-runtime"}
        else (case_name or "legacy_bad_interface")
    )
    installed = _installed_wheel(case_name, wheel_path)
    marker_path = evidence_dir / f"{case_name}.events.jsonl"
    original_load, load_calls = _install_load_counter(
        installed["entry_point"]
    )
    try:
        if mode == "discovery":
            result = _run_discovery(
                case_name,
                installed,
                load_calls,
                marker_path,
            )
        elif mode == "good-runtime":
            result = _run_good_runtime(
                installed,
                load_calls,
                marker_path,
            )
        else:
            result = _run_bad_interface(
                mode,
                installed,
                load_calls,
                marker_path,
                case_name,
            )
    finally:
        metadata.EntryPoint.load = original_load
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--wheel-path", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=tuple(CASES),
        default="legacy_bad_interface",
    )
    arguments = parser.parse_args()
    assert not os.environ.get("PYTHONPATH")

    evidence_dir = arguments.evidence_dir.resolve()
    evidence_dir.mkdir(parents=True, exist_ok=False)
    os.environ[MARKER_ENV] = str(evidence_dir)
    result = run(
        arguments.mode,
        arguments.wheel_path.resolve(),
        evidence_dir,
        arguments.case,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload, flush=True)
    (evidence_dir / f"{arguments.mode}.result.json").write_text(
        payload + "\n",
        encoding="utf-8",
    )

    if arguments.mode == "discovery":
        _assert_discovery(result)
    elif arguments.mode == "good-runtime":
        _assert_good_runtime(result)
    else:
        _assert_bad_interface(result)


if __name__ == "__main__":
    main()
