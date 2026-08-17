"""Fresh-process W10 probe for native ABI and ELF conflict governance."""

import argparse
from email.parser import Parser
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys
import tempfile


FIXTURES = {
    "compatible": {
        "distribution": (
            "triton-anchor-w10-native-compatible-backend"
        ),
        "entry_point": "w10_native_compatible",
        "module": "w10_native_compatible",
        "marker": "compatible.imported",
    },
    "bad_abi": {
        "distribution": (
            "triton-anchor-w10-native-bad-abi-backend"
        ),
        "entry_point": "w10_native_bad_abi",
        "module": "w10_native_bad_abi",
        "marker": "bad_abi.imported",
    },
    "collision_a": {
        "distribution": (
            "triton-anchor-w10-native-collision-a-backend"
        ),
        "entry_point": "w10_native_collision_a",
        "module": "w10_native_collision_a",
        "marker": "collision_a.imported",
    },
    "collision_b": {
        "distribution": (
            "triton-anchor-w10-native-collision-b-backend"
        ),
        "entry_point": "w10_native_collision_b",
        "module": "w10_native_collision_b",
        "marker": "collision_b.imported",
    },
}
FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
COLLISION_SONAME = "libtriton_anchor_w10_native_collision.so"
COLLISION_SYMBOL = "triton_anchor_w10_native_collision_symbol"
FATAL_CONFLICT_MODES = (
    "load-first",
    "register-first",
    "select-first",
    "compiler-first",
    "runtime-first",
)


def _assert_installed_native_wheel(fixture):
    distribution = metadata.distribution(fixture["distribution"])
    files = tuple(distribution.files or ())
    manifests = tuple(
        item
        for item in files
        if item.name == "triton_anchor_backend.json"
    )
    native_files = tuple(
        item
        for item in files
        if str(item).endswith(".so")
    )
    assert len(manifests) == 1
    assert len(native_files) == 1
    assert native_files[0].hash is not None
    assert native_files[0].hash.mode == "sha256"
    assert native_files[0].hash.value
    assert any(item.name == "entry_points.txt" for item in files)
    assert any(item.name == "RECORD" for item in files)

    wheel = Parser().parsestr(distribution.read_text("WHEEL") or "")
    assert (wheel.get("Root-Is-Purelib") or "").lower() == "false"
    tags = tuple(wheel.get_all("Tag", ()))
    assert tags
    assert any(not tag.endswith("-any") for tag in tags)

    entry_points = tuple(
        entry_point
        for entry_point in distribution.entry_points
        if entry_point.group == "triton.backends"
    )
    assert len(entry_points) == 1
    assert entry_points[0].name == fixture["entry_point"]
    return distribution, entry_points[0]


def _record_by_distribution(records, distribution_name):
    matches = tuple(
        record
        for record in records
        if record.distribution_name == distribution_name
    )
    assert len(matches) == 1
    return matches[0]


def _assert_not_imported(fixture, marker_dir):
    assert fixture["module"] not in sys.modules
    assert not (marker_dir / fixture["marker"]).exists()


def _module_or_submodule_imported(module_name):
    return any(
        name == module_name or name.startswith(module_name + ".")
        for name in sys.modules
    )


def _installed_native_library_path(distribution):
    native_files = tuple(
        item
        for item in (distribution.files or ())
        if str(item).endswith(".so")
    )
    assert len(native_files) == 1
    return Path(distribution.locate_file(native_files[0])).resolve()


def _linux_mapped_file_paths():
    maps_path = Path("/proc/self/maps")
    assert maps_path.is_file()
    mapped = set()
    for line in maps_path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        raw_path = fields[5]
        if not raw_path.startswith("/"):
            continue
        if raw_path.endswith(" (deleted)"):
            raw_path = raw_path[: -len(" (deleted)")]
        raw_path = raw_path.replace("\\040", " ")
        mapped.add(str(Path(raw_path).resolve()))
    return mapped


def _install_load_counter(entry_points):
    original = metadata.EntryPoint.load
    counts = {
        entry_point.name: 0
        for entry_point in entry_points
    }

    def counted_load(entry_point):
        if entry_point.name in counts:
            counts[entry_point.name] += 1
        return original(entry_point)

    metadata.EntryPoint.load = counted_load
    return original, counts


def _assert_core_origin_and_environment():
    triton = importlib.import_module("triton")
    triton_anchor = importlib.import_module("triton_anchor")
    environment_root = Path(sys.prefix).resolve()
    for module in (triton, triton_anchor):
        Path(module.__file__).resolve().relative_to(environment_root)

    from triton_anchor.backends import collect_core_environment

    environment = collect_core_environment()
    assert environment.build_info_generated is True
    assert FINGERPRINT_PATTERN.fullmatch(
        environment.core_abi_fingerprint or ""
    )
    assert environment.core_version == "0.2.0"
    assert environment.backend_protocol_version == "1.0"
    assert environment.triton_version == "3.0.0"
    assert (
        environment.vendored_triton_commit
        == "757b6a61e7df814ba806f498f8bb3160f84b120c"
    )
    assert environment.actual_llvm_version == "19.0.0"
    assert environment.actual_mlir_version == "19.0.0"
    assert (
        environment.actual_llvm_commit
        == "10dc3a8e916d73291269e5e2b82dd22681489aa1"
    )
    assert environment.actual_mlir_commit == environment.actual_llvm_commit
    return environment


def _validate_compatible(registry, marker_dir, load_counts):
    from triton_anchor.backends import PluginLifecycleState

    fixture = FIXTURES["compatible"]
    record = _record_by_distribution(
        registry.list(), fixture["distribution"]
    )
    validated = registry.validate(record.record_id)
    assert validated.state is PluginLifecycleState.VALIDATED
    dimensions = tuple(
        check.dimension
        for check in validated.compatibility_report.checks
    )
    required_dimensions = {
        "wheel platform tag",
        "native library artifacts",
        "Backend Plugin Protocol",
        "triton-anchor Core version",
        "Triton version",
        "vendored Triton commit",
        "LLVM version",
        "LLVM commit",
        "MLIR version",
        "MLIR commit",
        "core Python SOABI",
        "core build platform",
        "Core ABI fingerprint",
    }
    assert required_dimensions.issubset(set(dimensions))

    artifacts = validated.compatibility_report.native_artifacts
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.identity == (
        "libtriton_anchor_w10_native_compatible.so"
    )
    assert artifact.exported_symbols == (
        "triton_anchor_w10_native_compatible_symbol",
    )
    assert artifact.binary_format == "ELF"
    assert artifact.sha256
    assert load_counts[fixture["entry_point"]] == 0
    _assert_not_imported(fixture, marker_dir)

    loaded = registry.load(record.record_id)
    assert loaded.state is PluginLifecycleState.LOADED
    assert load_counts[fixture["entry_point"]] == 1
    assert (
        marker_dir / fixture["marker"]
    ).read_text(encoding="utf-8") == "imported\n"
    return loaded, dimensions


def _validate_bad_abi(registry, marker_dir, load_counts):
    from triton_anchor.backends import (
        BackendPluginCompatibilityError,
        PluginLifecycleState,
    )

    fixture = FIXTURES["bad_abi"]
    record = _record_by_distribution(
        registry.list(), fixture["distribution"]
    )
    error = None
    try:
        registry.validate(record.record_id)
    except BackendPluginCompatibilityError as exc:
        error = exc
    assert error is not None
    assert error.field == "Core ABI fingerprint"
    assert error.expected != error.actual

    rejected = _record_by_distribution(
        registry.list(), fixture["distribution"]
    )
    assert rejected.state is PluginLifecycleState.REJECTED
    assert rejected.error is error
    assert load_counts[fixture["entry_point"]] == 0
    _assert_not_imported(fixture, marker_dir)
    return error


def _validate_collisions(registry, marker_dir, load_counts):
    from triton.backends.compiler import GPUTarget
    from triton_anchor.backends import (
        BackendPluginConflictError,
        ConflictKind,
        PluginLifecycleState,
    )

    records = {}
    for key in ("collision_a", "collision_b"):
        fixture = FIXTURES[key]
        record = _record_by_distribution(
            registry.list(), fixture["distribution"]
        )
        validated = registry.validate(record.record_id)
        assert validated.state is PluginLifecycleState.VALIDATED
        artifact = validated.compatibility_report.native_artifacts[0]
        assert artifact.identity == COLLISION_SONAME
        assert artifact.exported_symbols == (COLLISION_SYMBOL,)
        records[key] = validated
        assert load_counts[fixture["entry_point"]] == 0
        _assert_not_imported(fixture, marker_dir)

    report = registry.conflicts()
    fatal = {
        (conflict.kind, conflict.claim)
        for conflict in report.fatal_conflicts
    }
    assert (
        ConflictKind.DUPLICATE_NATIVE_IDENTITY,
        COLLISION_SONAME,
    ) in fatal
    assert (
        ConflictKind.DUPLICATE_EXPORTED_SYMBOL,
        COLLISION_SYMBOL,
    ) in fatal

    conflict_error = None
    try:
        registry.select(
            GPUTarget("w10_native_collision", 1, 1)
        )
    except BackendPluginConflictError as exc:
        conflict_error = exc
    assert conflict_error is not None
    assert conflict_error.field == "native_libraries.SONAME"

    for key in ("collision_a", "collision_b"):
        fixture = FIXTURES[key]
        assert load_counts[fixture["entry_point"]] == 0
        _assert_not_imported(fixture, marker_dir)
    return report, conflict_error


def _fatal_conflict_gate_snapshot(
    registry,
    marker_dir,
    load_counts,
    native_paths,
):
    mapped_paths = _linux_mapped_file_paths()
    records = registry.list()
    records_by_distribution = {
        record.distribution_name: record
        for record in records
    }
    diagnostics = registry.diagnostics()
    diagnostics_by_distribution = {
        item["distribution_name"]: item
        for item in diagnostics["plugins"]
    }
    plugins = {}
    plugin_diagnostics = {}
    for key in ("collision_a", "collision_b"):
        fixture = FIXTURES[key]
        marker = marker_dir / fixture["marker"]
        record = records_by_distribution[fixture["distribution"]]
        diagnostic = diagnostics_by_distribution[fixture["distribution"]]
        plugins[key] = {
            "distribution": fixture["distribution"],
            "entry_point": fixture["entry_point"],
            "module_or_submodule_imported": (
                _module_or_submodule_imported(fixture["module"])
            ),
            "python_imported_marker_exists": marker.exists(),
            "python_imported_marker_content": (
                marker.read_text(encoding="utf-8")
                if marker.exists()
                else None
            ),
            "native_library": str(native_paths[key]),
            "native_library_mapped": str(native_paths[key]) in mapped_paths,
            "state": record.state.value,
            "plugin_object_loaded": record.plugin_object is not None,
            "compiler_class_available": record.compiler_cls is not None,
            "driver_class_available": record.driver_cls is not None,
            "initialized": record.initialized,
        }
        plugin_diagnostics[key] = {
            "state": diagnostic["state"],
            "compatibility_status": diagnostic["compatibility_status"],
            "loaded": diagnostic["loaded"],
            "initialized": diagnostic["initialized"],
            "errors": diagnostic["errors"],
        }
    return {
        "entry_point_load_calls": dict(sorted(load_counts.items())),
        "plugins": plugins,
        "registry_diagnostics": {
            "registry_errors": diagnostics["registry_errors"],
            "conflicts": diagnostics["conflicts"],
            "plugins": plugin_diagnostics,
        },
    }


def _structured_error(error):
    if error is None:
        return None
    if hasattr(error, "to_dict"):
        result = error.to_dict()
    else:
        result = {
            "message": str(error),
        }
    result["exception_type"] = type(error).__name__
    return result


def run_fatal_conflict_gate_probe(mode, marker_dir):
    assert mode in {"load-first", "register-first", "select-first"}
    fixture_keys = ("collision_a", "collision_b")
    installed = {
        key: _assert_installed_native_wheel(FIXTURES[key])
        for key in fixture_keys
    }
    native_paths = {
        key: _installed_native_library_path(installed[key][0])
        for key in fixture_keys
    }
    original_load, load_counts = _install_load_counter(
        tuple(installed[key][1] for key in fixture_keys)
    )
    try:
        environment = _assert_core_origin_and_environment()
        from triton_anchor.backends import BackendPluginRegistry

        distributions = tuple(installed[key][0] for key in fixture_keys)
        registry = BackendPluginRegistry(
            distribution_provider=lambda: distributions
        )
        discovered = registry.discover()
        assert len(discovered) == 2
        before = _fatal_conflict_gate_snapshot(
            registry,
            marker_dir,
            load_counts,
            native_paths,
        )
        assert {
            plugin["state"]
            for plugin in before["plugins"].values()
        } == {"discovered"}
        assert all(
            count == 0
            for count in before["entry_point_load_calls"].values()
        )

        target_key = {
            "load-first": "collision_a",
            "register-first": "collision_b",
            "select-first": "collision_a",
        }[mode]
        target = _record_by_distribution(
            discovered,
            FIXTURES[target_key]["distribution"],
        )
        error = None
        returned_state = None
        try:
            if mode == "load-first":
                returned_state = registry.load(target.record_id).state.value
            else:
                if mode == "register-first":
                    returned_state = registry.register(
                        target.record_id
                    ).state.value
                else:
                    returned_state = registry.select(
                        "w10_native_collision",
                        environment={},
                    ).record.state.value
        except Exception as exc:
            error = exc

        after = _fatal_conflict_gate_snapshot(
            registry,
            marker_dir,
            load_counts,
            native_paths,
        )
        return {
            "mode": mode,
            "pid": os.getpid(),
            "first_loading_operation": {
                "load-first": "Registry.load()",
                "register-first": "Registry.register()",
                "select-first": "Registry.select()",
            }[mode],
            "core_abi_fingerprint": environment.core_abi_fingerprint,
            "target_record_id": target.record_id,
            "record_ids": sorted(
                record.record_id for record in discovered
            ),
            "stage_counters": {
                "compiler_constructed": 0,
                "driver_constructed": 0,
                "runtime_returned": 0,
            },
            "before": before,
            "after": after,
            "returned_state": returned_state,
            "error": _structured_error(error),
        }
    finally:
        metadata.EntryPoint.load = original_load


def run_triton_entry_fatal_conflict_gate_probe(mode, marker_dir):
    assert mode in {"compiler-first", "runtime-first"}
    fixture_keys = ("collision_a", "collision_b")
    installed = {
        key: _assert_installed_native_wheel(FIXTURES[key])
        for key in fixture_keys
    }
    native_paths = {
        key: _installed_native_library_path(installed[key][0])
        for key in fixture_keys
    }
    original_load, load_counts = _install_load_counter(
        tuple(installed[key][1] for key in fixture_keys)
    )
    try:
        environment = _assert_core_origin_and_environment()
        from triton_anchor.backends import get_backend_plugin_registry

        registry = get_backend_plugin_registry()
        discovered = registry.discover()
        before = _fatal_conflict_gate_snapshot(
            registry,
            marker_dir,
            load_counts,
            native_paths,
        )
        assert {
            plugin["state"]
            for plugin in before["plugins"].values()
        } == {"discovered"}
        assert all(
            count == 0
            for count in before["entry_point_load_calls"].values()
        )

        target = _record_by_distribution(
            discovered,
            FIXTURES["collision_a"]["distribution"],
        )
        stage_counters = {
            "compiler_constructed": 0,
            "driver_constructed": 0,
            "runtime_returned": 0,
        }
        error = None
        returned_state = None
        runtime_proxy_state = None
        try:
            if mode == "compiler-first":
                from triton.backends.compiler import GPUTarget
                from triton.compiler import make_backend

                compiler = make_backend(
                    GPUTarget("w10_native_collision", 1, 1)
                )
                stage_counters["compiler_constructed"] += 1
                returned_state = type(compiler).__name__
            else:
                from triton.runtime import driver

                runtime_proxy_state = {
                    "before_object_is_none": driver.default._obj is None,
                    "before_initializing": driver.default._initializing,
                }
                runtime_driver = driver.active.get_current_target()
                stage_counters["driver_constructed"] += 1
                stage_counters["runtime_returned"] += 1
                returned_state = type(runtime_driver).__name__
        except Exception as exc:
            error = exc

        if mode == "runtime-first":
            from triton.runtime import driver

            runtime_proxy_state.update(
                {
                    "after_object_is_none": driver.default._obj is None,
                    "after_initializing": driver.default._initializing,
                }
            )

        after = _fatal_conflict_gate_snapshot(
            registry,
            marker_dir,
            load_counts,
            native_paths,
        )
        return {
            "mode": mode,
            "pid": os.getpid(),
            "first_loading_operation": {
                "compiler-first": "triton.compiler.make_backend()",
                "runtime-first": (
                    "triton.runtime.driver.active.get_current_target()"
                ),
            }[mode],
            "core_abi_fingerprint": environment.core_abi_fingerprint,
            "target_record_id": target.record_id,
            "record_ids": sorted(
                record.record_id
                for record in discovered
                if record.distribution_name
                in {
                    FIXTURES["collision_a"]["distribution"],
                    FIXTURES["collision_b"]["distribution"],
                }
            ),
            "stage_counters": stage_counters,
            "runtime_proxy_state": runtime_proxy_state,
            "before": before,
            "after": after,
            "returned_state": returned_state,
            "error": _structured_error(error),
        }
    finally:
        metadata.EntryPoint.load = original_load


def _assert_fatal_conflict_gate(result):
    context = json.dumps(result, indent=2, sort_keys=True)
    after = result["after"]
    assert all(
        count == 0
        for count in after["entry_point_load_calls"].values()
    ), (
        "fatal-conflict gate was bypassed before EntryPoint.load():\n"
        + context
    )
    assert all(
        not plugin["python_imported_marker_exists"]
        for plugin in after["plugins"].values()
    ), "plugin import marker proves Python code executed:\n" + context
    assert all(
        not plugin["module_or_submodule_imported"]
        for plugin in after["plugins"].values()
    ), "plugin module reached sys.modules:\n" + context
    assert all(
        not plugin["native_library_mapped"]
        for plugin in after["plugins"].values()
    ), "plugin DSO reached /proc/self/maps:\n" + context
    assert all(
        not plugin["plugin_object_loaded"]
        and not plugin["compiler_class_available"]
        and not plugin["driver_class_available"]
        and not plugin["initialized"]
        for plugin in after["plugins"].values()
    ), "plugin runtime interface was materialized:\n" + context
    assert all(
        count == 0
        for count in result["stage_counters"].values()
    ), "compiler or runtime code executed:\n" + context

    error = result["error"]
    assert error is not None, "fatal conflict returned successfully:\n" + context
    assert (
        error["exception_type"] == "BackendPluginConflictError"
    ), "fatal conflict was replaced by another exception:\n" + context
    assert (
        error["code"] == "backend_plugin_conflict_error"
    ), "fatal conflict error code was not preserved:\n" + context
    required = {
        "code",
        "message",
        "field",
        "expected",
        "actual",
        "remediation",
    }
    assert required.issubset(error), (
        "structured conflict error is incomplete:\n" + context
    )
    assert all(error[field] for field in required), (
        "structured conflict error contains empty required values:\n"
        + context
    )
    assert error["field"] in {
        "native_libraries.SONAME",
        "native_libraries.exported_symbols",
    }, "native conflict type was not preserved:\n" + context
    if error["field"] == "native_libraries.SONAME":
        assert COLLISION_SONAME in error["message"], (
            "conflicting SONAME is absent from the error:\n" + context
        )
    else:
        assert COLLISION_SYMBOL in error["message"], (
            "conflicting exported symbol is absent from the error:\n"
            + context
        )
    assert error["code"] != "backend_plugin_load_error", (
        "BackendPluginLoadError replaced the conflict:\n" + context
    )
    assert result["returned_state"] is None, (
        "fatal-conflict operation returned a usable record:\n" + context
    )
    assert {
        plugin["state"] for plugin in after["plugins"].values()
    } == {"rejected"}, (
        "conflicted records did not enter REJECTED:\n" + context
    )
    if result["mode"] == "runtime-first":
        proxy = result["runtime_proxy_state"]
        assert proxy == {
            "before_object_is_none": True,
            "before_initializing": False,
            "after_object_is_none": True,
            "after_initializing": False,
        }, "runtime LazyProxy retained partial state:\n" + context


def run_probe(marker_dir):
    installed = {
        key: _assert_installed_native_wheel(fixture)
        for key, fixture in FIXTURES.items()
    }
    original_load, load_counts = _install_load_counter(
        tuple(item[1] for item in installed.values())
    )
    try:
        environment = _assert_core_origin_and_environment()
        for fixture in FIXTURES.values():
            _assert_not_imported(fixture, marker_dir)
        assert all(count == 0 for count in load_counts.values())

        from triton_anchor.backends import BackendPluginRegistry

        registry = BackendPluginRegistry()
        compatible, dimensions = _validate_compatible(
            registry, marker_dir, load_counts
        )
        bad_error = _validate_bad_abi(
            registry, marker_dir, load_counts
        )
        conflict_report, conflict_error = _validate_collisions(
            registry, marker_dir, load_counts
        )

        return {
            "core_abi_fingerprint": (
                environment.core_abi_fingerprint
            ),
            "compatible_record": compatible.record_id,
            "compatible_dimensions": list(dimensions),
            "bad_abi_error": bad_error.to_dict(),
            "conflicts": conflict_report.to_dict(),
            "selection_error": conflict_error.to_dict(),
            "entry_point_load_calls": load_counts,
        }
    finally:
        metadata.EntryPoint.load = original_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("full",) + FATAL_CONFLICT_MODES,
        default="full",
    )
    parser.add_argument("--evidence-dir")
    arguments = parser.parse_args()
    assert not os.environ.get("PYTHONPATH")

    def execute(marker_path):
        os.environ["TRITON_ANCHOR_W10_NATIVE_MARKER_DIR"] = str(
            marker_path
        )
        if arguments.mode == "full":
            result = run_probe(marker_path)
        elif arguments.mode in {"compiler-first", "runtime-first"}:
            result = run_triton_entry_fatal_conflict_gate_probe(
                arguments.mode,
                marker_path,
            )
        else:
            result = run_fatal_conflict_gate_probe(
                arguments.mode,
                marker_path,
            )
        payload = json.dumps(result, indent=2, sort_keys=True)
        print(payload, flush=True)
        (marker_path / (arguments.mode + ".result.json")).write_text(
            payload + "\n",
            encoding="utf-8",
        )
        if arguments.mode != "full":
            _assert_fatal_conflict_gate(result)

    if arguments.evidence_dir:
        marker_path = Path(arguments.evidence_dir).resolve()
        marker_path.mkdir(parents=True, exist_ok=False)
        execute(marker_path)
    else:
        with tempfile.TemporaryDirectory(
            prefix="triton-anchor-w10-native-markers."
        ) as marker_dir:
            execute(Path(marker_dir))


if __name__ == "__main__":
    main()
