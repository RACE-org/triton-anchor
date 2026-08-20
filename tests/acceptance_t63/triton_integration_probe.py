#!/usr/bin/env python3
"""Subprocess probe for the T6.3 Triton public integration paths.

This helper intentionally imports Triton's real Python sources while replacing
only the unavailable ``libtriton`` and unrelated compiler dependencies with
small modules.  Backend discovery, Registry selection, the public ``backends``
mapping, compiler ``make_backend``, and runtime driver/reset code are not
stubbed.

The ``legacy`` mode can execute either the checked-out port or the frozen
``upstream/triton_v3.3`` commit source from git.  The other modes execute the port.
Output is one JSON document so the pytest acceptance oracles can report a
complete minimal reproduction even when more than one public path is broken.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Callable


BASELINE_REF = "31b073040a6d11afa63c7a062d24d546e3dd887d"


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return repr(value)


def _attempt(callback: Callable[[], Any]) -> dict[str, Any]:
    try:
        value = callback()
    except BaseException as exc:  # the exception class is acceptance evidence
        diagnostic = None
        to_dict = getattr(exc, "to_dict", None)
        if callable(to_dict):
            try:
                diagnostic = to_dict()
            except BaseException:
                diagnostic = None
        return {
            "ok": False,
            "type": type(exc).__name__,
            "message": str(exc),
            "diagnostic": _json_value(diagnostic),
        }
    return {"ok": True, "value": _json_value(value)}


def _module(name: str, **attributes: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def _package(name: str, path: Path | None = None) -> types.ModuleType:
    module = _module(name)
    module.__package__ = name
    module.__path__ = [] if path is None else [str(path)]
    return module


def _prepare_imports(repository: Path, site: Path) -> None:
    python_root = repository / "python"
    triton_python_root = repository / "triton/python"
    for item in (site, python_root, triton_python_root):
        text = str(item)
        if text not in sys.path:
            sys.path.insert(0, text)

    # Entry-point discovery is intentionally isolated to the wheel under test.
    # Otherwise an unrelated globally installed Triton/backend can change the
    # candidate count and hide (or create) a public-path regression.
    metadata_distributions = importlib.metadata.distributions

    def isolated_distributions(**_kwargs):
        return metadata_distributions(path=[str(site)])

    def isolated_entry_points(**selection):
        discovered = tuple(
            entry_point
            for distribution in metadata_distributions(path=[str(site)])
            for entry_point in distribution.entry_points
        )
        if selection:
            discovered = tuple(
                entry_point
                for entry_point in discovered
                if all(
                    getattr(entry_point, key, None) == value
                    for key, value in selection.items()
                )
            )
        return discovered

    importlib.metadata.distributions = isolated_distributions
    importlib.metadata.entry_points = isolated_entry_points

    triton_dir = triton_python_root / "triton"
    triton = _package("triton", triton_dir)
    triton.__version__ = "3.3.0"

    # compiler.py cannot be imported from a source checkout without the C++
    # extension.  These names are import-only dependencies for make_backend().
    _package("triton._C", triton_dir / "_C")
    _module(
        "triton._C.libtriton",
        get_cache_invalidating_env_vars=lambda: {},
        ir=types.SimpleNamespace(),
    )

    runtime = _package("triton.runtime", triton_dir / "runtime")
    autotuner = _module(
        "triton.runtime.autotuner",
        OutOfResources=type("OutOfResources", (RuntimeError,), {}),
    )
    cache = _module(
        "triton.runtime.cache",
        get_cache_manager=lambda *_args, **_kwargs: None,
        get_dump_manager=lambda *_args, **_kwargs: None,
        get_override_manager=lambda *_args, **_kwargs: None,
    )
    runtime.autotuner = autotuner
    runtime.cache = cache

    tools = _package("triton.tools", triton_dir / "tools")
    disasm = _module("triton.tools.disasm", get_sass=lambda _value: "")
    tools.disasm = disasm

    compiler = _package("triton.compiler", triton_dir / "compiler")
    code_generator = _module(
        "triton.compiler.code_generator", ast_to_ttir=lambda *_a, **_k: None
    )
    compiler.code_generator = code_generator


def _git_source(repository: Path, relative_path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{BASELINE_REF}:{relative_path}"],
        cwd=repository,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"unable to read {relative_path} from {BASELINE_REF}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def _exec_package_source(
    name: str,
    source: str,
    filename: str,
    package_path: Path | None = None,
) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__package__ = name.rpartition(".")[0]
    if package_path is not None:
        module.__path__ = [str(package_path)]
        module.__package__ = name
    sys.modules[name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


def _load_backends(repository: Path, baseline: bool) -> types.ModuleType:
    if not baseline:
        return importlib.import_module("triton.backends")
    relative = "triton/python/triton/backends/__init__.py"
    return _exec_package_source(
        "triton.backends",
        _git_source(repository, relative),
        f"{BASELINE_REF}:{relative}",
        repository / "triton/python/triton/backends",
    )


def _load_compiler(repository: Path, baseline: bool) -> types.ModuleType:
    if not baseline:
        return importlib.import_module("triton.compiler.compiler")
    relative = "triton/python/triton/compiler/compiler.py"
    return _exec_package_source(
        "triton.compiler.compiler",
        _git_source(repository, relative),
        f"{BASELINE_REF}:{relative}",
    )


def _load_runtime_driver(repository: Path, baseline: bool) -> types.ModuleType:
    if not baseline:
        return importlib.import_module("triton.runtime.driver")
    relative = "triton/python/triton/runtime/driver.py"
    return _exec_package_source(
        "triton.runtime.driver",
        _git_source(repository, relative),
        f"{BASELINE_REF}:{relative}",
    )


def _class_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _active_driver_observation(driver_config: Any) -> dict[str, Any]:
    target = driver_config.active.get_current_target()
    active = driver_config.active
    concrete = getattr(active, "_obj", None) or active
    return {
        "class": _class_name(concrete),
        "target": target.backend,
    }


def _mapping(backends_module: types.ModuleType) -> dict[str, Any]:
    result = {}
    for name, backend in sorted(backends_module.backends.items()):
        result[name] = {
            "compiler": _class_name(backend.compiler),
            "driver": _class_name(backend.driver),
            "record_id": getattr(backend, "record_id", None),
            "plugin_id": getattr(backend, "plugin_id", None),
        }
    return result


def _legacy_probe(
    repository: Path, site: Path, marker: Path, baseline: bool
) -> dict[str, Any]:
    _prepare_imports(repository, site)
    os.environ.pop("TRITON_ANCHOR_BACKEND", None)
    os.environ["T63_PLUGIN_IMPORT_MARKER"] = str(marker)

    backends_module = _load_backends(repository, baseline)
    target_cls = importlib.import_module("triton.backends.compiler").GPUTarget
    target = target_cls("legacy_fixture", "fixture-arch", 32)
    runtime_driver = _load_runtime_driver(repository, baseline)
    compiler_module = _load_compiler(repository, baseline)

    import_after_discovery = marker.exists()
    mapping_after_discovery = _mapping(backends_module)

    compiler_result = _attempt(
        lambda: {
            "class": _class_name(compiler_module.make_backend(target)),
            "target": compiler_module.make_backend(target).target.backend,
        }
    )
    runtime_result = _attempt(
        lambda: _active_driver_observation(runtime_driver.driver)
    )
    mapping_after_implicit_use = _mapping(backends_module)

    if baseline:
        reset_result = _attempt(
            lambda: (
                runtime_driver.driver.reset_active(),
                runtime_driver.driver.active.get_current_target().backend,
            )[-1]
        )
        records = None
        explicit_selection = None
    else:
        import triton_anchor.backends as api

        registry = api.get_backend_plugin_registry()
        records = [record.to_dict() for record in registry.list()]
        reset_result = _attempt(
            lambda: (
                registry.reset(),
                runtime_driver.driver.active.get_current_target().backend,
            )[-1]
        )

        def explicitly_select_and_consume():
            legacy = tuple(
                record
                for record in registry.list()
                if record.entry_point_name == "legacy_fixture"
            )
            if len(legacy) != 1:
                raise AssertionError(
                    f"expected one Legacy fixture record, found {len(legacy)}"
                )
            decision = registry.select(
                target, explicit_selector=legacy[0].registry_key
            )
            compiler_instance = compiler_module.make_backend(target)
            runtime_observation = _active_driver_observation(
                runtime_driver.driver
            )
            return {
                "decision": decision.to_dict(),
                "compiler": _class_name(compiler_instance),
                "runtime_driver": runtime_observation,
                "mapping": _mapping(backends_module),
            }

        explicit_selection = _attempt(explicitly_select_and_consume)

    return {
        "mode": "baseline" if baseline else "port",
        "source_ref": BASELINE_REF if baseline else "HEAD working tree",
        "imported_during_discovery": import_after_discovery,
        "plugin_imported": marker.exists(),
        "mapping_after_discovery": mapping_after_discovery,
        "mapping_after_public_use": mapping_after_implicit_use,
        "compiler": compiler_result,
        "runtime_driver": runtime_result,
        "reset_then_runtime_driver": reset_result,
        "explicit_selection": explicit_selection,
        "records": records,
    }


def _new_source_registry(site: Path):
    import importlib.metadata
    import triton_anchor.backends as api

    registry_module = importlib.import_module("triton_anchor.backends.registry")
    registry = api.BackendPluginRegistry(
        distribution_provider=lambda: importlib.metadata.distributions(
            path=[str(site)]
        ),
        environment_provider=api.collect_core_environment,
        preflight_profile="triton_version",
    )
    registry_module.backend_plugin_registry = registry
    # Keep the public alias truthful for diagnostic consumers.  The getter is
    # defined in registry.py and already reads the replaced module global.
    api.backend_plugin_registry = registry
    return api, registry


def _manifest_probe(
    repository: Path, site: Path, marker: Path
) -> dict[str, Any]:
    _prepare_imports(repository, site)
    os.environ.pop("TRITON_ANCHOR_BACKEND", None)
    os.environ["T63_PLUGIN_IMPORT_MARKER"] = str(marker)
    api, registry = _new_source_registry(site)

    backends_module = _load_backends(repository, baseline=False)
    target_cls = importlib.import_module("triton.backends.compiler").GPUTarget
    target = target_cls("manifest_fixture", "fixture-arch", 32)
    runtime_driver = _load_runtime_driver(repository, baseline=False)
    compiler_module = _load_compiler(repository, baseline=False)

    mapping_before_selection = _mapping(backends_module)
    imported_before_selection = marker.exists()
    compiler_result = _attempt(
        lambda: {
            "class": _class_name(compiler_module.make_backend(target)),
            "target": compiler_module.make_backend(target).target.backend,
        }
    )
    mapping_after_compiler = _mapping(backends_module)
    runtime_result = _attempt(
        lambda: _active_driver_observation(runtime_driver.driver)
    )
    first_diagnostics = registry.diagnostics()

    reset_errors = registry.reset()
    mapping_after_reset = _mapping(backends_module)
    compiler_after_reset = _attempt(
        lambda: _class_name(compiler_module.make_backend(target))
    )
    runtime_after_reset = _attempt(
        lambda: runtime_driver.driver.active.get_current_target().backend
    )
    second_diagnostics = registry.diagnostics()

    return {
        "mode": "manifest",
        "imported_before_selection": imported_before_selection,
        "plugin_imported": marker.exists(),
        "mapping_before_selection": mapping_before_selection,
        "mapping_after_compiler": mapping_after_compiler,
        "mapping_after_reset": mapping_after_reset,
        "mapping_after_reselection": _mapping(backends_module),
        "compiler": compiler_result,
        "runtime_driver": runtime_result,
        "reset_errors": [error.to_dict() for error in reset_errors],
        "compiler_after_reset": compiler_after_reset,
        "runtime_after_reset": runtime_after_reset,
        "first_diagnostics": first_diagnostics,
        "second_diagnostics": second_diagnostics,
        "selector_environment": os.environ.get(api.BACKEND_SELECTOR_ENV),
    }


def _v30_probe(repository: Path, site: Path, marker: Path) -> dict[str, Any]:
    _prepare_imports(repository, site)
    os.environ.pop("TRITON_ANCHOR_BACKEND", None)
    os.environ["T63_PLUGIN_IMPORT_MARKER"] = str(marker)
    _api, registry = _new_source_registry(site)

    backends_module = _load_backends(repository, baseline=False)
    target_cls = importlib.import_module("triton.backends.compiler").GPUTarget
    target = target_cls("v30_fixture", "fixture-arch", 32)
    runtime_driver = _load_runtime_driver(repository, baseline=False)
    compiler_module = _load_compiler(repository, baseline=False)

    selection = _attempt(lambda: registry.select(target).to_dict())
    selected_record = registry.get_selection(target.backend)
    abstract_methods = None
    if selected_record is not None:
        record = registry.inspect(selected_record.record_id)
        abstract_methods = {
            "compiler": sorted(
                getattr(record.compiler_cls, "__abstractmethods__", ())
            ),
            "driver": sorted(
                getattr(record.driver_cls, "__abstractmethods__", ())
            ),
        }

    compiler_result = _attempt(lambda: compiler_module.make_backend(target))
    driver_result = _attempt(
        lambda: runtime_driver.driver.active.get_current_target().backend
    )
    diagnostics = registry.diagnostics()

    return {
        "mode": "v30_api_gap",
        "plugin_imported": marker.exists(),
        "selection": selection,
        "abstract_methods": abstract_methods,
        "compiler": compiler_result,
        "runtime_driver": driver_result,
        "mapping_after_late_failures": _mapping(backends_module),
        "diagnostics": diagnostics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("legacy", "manifest", "v30"), required=True
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    repository = args.repository.resolve()
    site = args.site.resolve()
    marker = args.marker.resolve()
    if marker.exists():
        marker.unlink()

    if args.mode == "legacy":
        result = _legacy_probe(repository, site, marker, args.baseline)
    elif args.mode == "manifest":
        if args.baseline:
            parser.error("--baseline is only valid with --mode legacy")
        result = _manifest_probe(repository, site, marker)
    else:
        if args.baseline:
            parser.error("--baseline is only valid with --mode legacy")
        result = _v30_probe(repository, site, marker)

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
