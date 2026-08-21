#!/usr/bin/env python3
"""Isolated real-distribution probe for the governed v3.3 Legacy bridge.

The caller installs a standards-compliant wheel without a Manifest into
``site``.  This process uses the real ``importlib.metadata`` Distribution and
EntryPoint implementations from that directory.  Only Triton's unavailable
native extension and compiler-only imports are stubbed; Registry discovery,
the backend adapter, compiler ``make_backend`` and runtime driver are product
code from the checked-out tree.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable


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
    except BaseException as exc:
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
        value = str(item)
        if value not in sys.path:
            sys.path.insert(0, value)

    distributions = importlib.metadata.distributions

    def isolated_distributions(**_kwargs: Any):
        return distributions(path=[str(site)])

    def isolated_entry_points(**selection: Any):
        discovered = tuple(
            entry_point
            for distribution in distributions(path=[str(site)])
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

    _package("triton._C", triton_dir / "_C")
    _module(
        "triton._C.libtriton",
        get_cache_invalidating_env_vars=lambda: {},
        ir=types.SimpleNamespace(),
    )
    runtime = _package("triton.runtime", triton_dir / "runtime")
    runtime.autotuner = _module(
        "triton.runtime.autotuner",
        OutOfResources=type("OutOfResources", (RuntimeError,), {}),
    )
    runtime.cache = _module(
        "triton.runtime.cache",
        get_cache_manager=lambda *_args, **_kwargs: None,
        get_dump_manager=lambda *_args, **_kwargs: None,
        get_override_manager=lambda *_args, **_kwargs: None,
    )
    tools = _package("triton.tools", triton_dir / "tools")
    tools.disasm = _module(
        "triton.tools.disasm", get_sass=lambda _value: ""
    )
    compiler = _package("triton.compiler", triton_dir / "compiler")
    compiler.code_generator = _module(
        "triton.compiler.code_generator",
        ast_to_ttir=lambda *_args, **_kwargs: None,
    )


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mapping(backends_module: types.ModuleType) -> dict[str, Any]:
    return {
        name: {
            "compiler": backend.compiler.__module__
            + "."
            + backend.compiler.__qualname__,
            "driver": backend.driver.__module__
            + "."
            + backend.driver.__qualname__,
            "record_id": backend.record_id,
            "entry_point": backend.entry_point_name,
        }
        for name, backend in sorted(backends_module.backends.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--events", required=True, type=Path)
    args = parser.parse_args()

    repository = args.repository.resolve()
    site = args.site.resolve()
    event_path = args.events.resolve()
    if event_path.exists():
        event_path.unlink()
    os.environ["T63_LEGACY_EVENTS"] = str(event_path)
    os.environ.pop("TRITON_ANCHOR_BACKEND", None)
    _prepare_imports(repository, site)

    real_distributions = tuple(importlib.metadata.distributions())
    real_entry_points = tuple(
        entry_point
        for distribution in real_distributions
        for entry_point in distribution.entry_points
        if entry_point.group == "triton.backends"
    )
    metadata = {
        "distribution_types": sorted(
            {type(distribution).__name__ for distribution in real_distributions}
        ),
        "entry_point_types": sorted(
            {type(entry_point).__name__ for entry_point in real_entry_points}
        ),
        "entry_points": sorted(
            (entry_point.name, entry_point.value)
            for entry_point in real_entry_points
        ),
        "manifest_files": sorted(
            str(item)
            for distribution in real_distributions
            for item in (distribution.files or ())
            if str(item).endswith("triton_anchor_backend.json")
        ),
    }

    import triton_anchor.backends as api

    registry_module = importlib.import_module("triton_anchor.backends.registry")
    registry = api.BackendPluginRegistry(
        distribution_provider=lambda: importlib.metadata.distributions(),
        environment_provider=api.collect_core_environment,
        preflight_profile="triton_version",
    )
    registry_module.backend_plugin_registry = registry
    api.backend_plugin_registry = registry

    backends_module = importlib.import_module("triton.backends")
    target_cls = importlib.import_module("triton.backends.compiler").GPUTarget
    compiler_module = importlib.import_module("triton.compiler.compiler")
    runtime_driver = importlib.import_module("triton.runtime.driver")
    target = target_cls("wheel_legacy", "fixture-arch", 32)

    backends_module._discover_backends()
    records = registry.list()
    inspected = registry.inspect(records[0].record_id)
    diagnostics = registry.diagnostics()
    discovery = {
        "events": _events(event_path),
        "mapping": _mapping(backends_module),
        "record": inspected.to_dict(),
        "diagnostics": diagnostics,
    }

    compiler = _attempt(
        lambda: {
            "class": type(compiler_module.make_backend(target)).__qualname__,
            "target": compiler_module.make_backend(target).target.backend,
        }
    )
    after_compiler = {
        "events": _events(event_path),
        "mapping": _mapping(backends_module),
        "selection": _json_value(
            registry.get_selection(target.backend).to_dict()
            if registry.get_selection(target.backend) is not None
            else None
        ),
    }

    reset = _attempt(lambda: [error.to_dict() for error in registry.reset()])
    after_reset = {
        "events": _events(event_path),
        "mapping": _mapping(backends_module),
        "selection": registry.get_selection(target.backend) is not None,
        "lazy_driver_materialized": (
            runtime_driver.driver.default._obj is not None
        ),
    }
    runtime = _attempt(
        lambda: runtime_driver.driver.active.get_current_target().backend
    )
    after_runtime = {
        "events": _events(event_path),
        "mapping": _mapping(backends_module),
        "selection": _json_value(
            registry.get_selection(target.backend).to_dict()
            if registry.get_selection(target.backend) is not None
            else None
        ),
        "active_driver_class": (
            type(runtime_driver.driver.default._obj).__qualname__
            if runtime_driver.driver.default._obj is not None
            else None
        ),
    }

    print(
        json.dumps(
            {
                "metadata": metadata,
                "discovery": discovery,
                "compiler": compiler,
                "after_compiler": after_compiler,
                "reset": reset,
                "after_reset": after_reset,
                "runtime": runtime,
                "after_runtime": after_runtime,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
