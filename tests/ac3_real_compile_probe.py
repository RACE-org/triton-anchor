"""Real-wheel proof: a Registry-selected plugin enters the real triton.compile().

The probe deliberately resolves the compiler only through
BackendPluginRegistry (discover -> validate -> select) and then calls the real
``triton.compile()`` API.  It never imports the fixture compiler directly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import triton
import triton.compiler
import triton.language as tl

import w10_native_wheel_probe as w10_support


DISTRIBUTION = "triton-anchor-ac3-real-compile-backend"
ENTRY_POINT = "ac3_real_compile"
MODULE = "ac3_real_compile_backend"
TARGET = "ac3_mock"


@triton.jit
def _ac3_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Trivial kernel compiled through the Registry-selected backend."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def _assert_installed():
    from importlib import metadata

    distribution = metadata.distribution(DISTRIBUTION)
    entry_points = tuple(
        ep
        for ep in distribution.entry_points
        if ep.group == "triton.backends"
    )
    assert len(entry_points) == 1, entry_points
    entry_point = entry_points[0]
    assert entry_point.name == ENTRY_POINT
    assert entry_point.value == MODULE
    Path(distribution.locate_file("")).resolve().relative_to(
        Path(sys.prefix).resolve()
    )
    return distribution, entry_point


def _compile_source():
    return triton.compiler.ASTSource(
        fn=_ac3_add_kernel,
        signature={0: "*fp32", 1: "*fp32", 2: "*fp32", 3: "i32"},
        constants={4: 256},
    )


def run_probe():
    from triton.backends.compiler import GPUTarget
    from triton_anchor.backends import (
        PluginLifecycleState,
        get_backend_plugin_registry,
    )

    # The acceptance proof must always run the real compile pipeline; never
    # accept a previously cached metadata/stage result.
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    distribution, entry_point = _assert_installed()
    original_load, load_counts = w10_support._install_load_counter(
        (entry_point,)
    )
    try:
        # Use the same singleton registry that triton.backends (the W9 bridge)
        # consumes, so triton.compile() resolves the probe's selection instead
        # of re-selecting (and re-loading) in a second registry instance.
        registry = get_backend_plugin_registry()
        records = registry.discover()
        record = w10_support._record_by_distribution(records, DISTRIBUTION)
        assert record.entry_point_name == ENTRY_POINT
        assert load_counts[ENTRY_POINT] == 0

        record = registry.validate(record.record_id)
        assert record.state is PluginLifecycleState.VALIDATED
        assert load_counts[ENTRY_POINT] == 0

        target = GPUTarget(TARGET, 1, 1)
        decision = registry.select(
            target,
            explicit_selector=record.registry_key,
            environment={},
        )
        assert decision.method.value == "python_explicit"
        assert load_counts[ENTRY_POINT] == 1

        selected = registry.inspect(decision.record_id)
        assert selected.state is PluginLifecycleState.SELECTED
        assert selected.compiler_cls is not None
        assert selected.driver_cls is not None
        compiler_cls = selected.compiler_cls
        assert compiler_cls.__module__ == MODULE
        assert selected.driver_cls.__module__ == MODULE
        assert registry.get_selection(TARGET).record_id == selected.record_id

        # Create a compiler instance from the plugin's compiler_cls.
        compiler = compiler_cls(target)
        assert compiler.target == target

        # Real Triton compile.  triton.compile() -> make_backend() resolves the
        # same Registry selection and instantiates the plugin's compiler_cls.
        result = triton.compile(
            _compile_source(),
            target=target,
            options={"num_warps": 1, "num_stages": 1},
        )
        assert isinstance(result, triton.compiler.CompiledKernel)
        assert "ttir" in compiler_cls.stage_calls
        assert "asm" in compiler_cls.stage_calls
        # triton.compile() consumed the Registry selection; it must not have
        # triggered a second plugin load.
        assert load_counts[ENTRY_POINT] == 1

        from triton.backends import backends as triton_backends

        # Manifest bindings are transient projections of RegistryState.  The
        # public mapping remains a Legacy compatibility catalog.
        assert ENTRY_POINT not in triton_backends

        return {
            "plugin": DISTRIBUTION,
            "plugin_id": selected.plugin_id,
            "record_id": selected.record_id,
            "state": selected.state.value,
            "registry_state": selected.state.value,
            "selection_method": decision.method.value,
            "compiler_created": True,
            "compiler_module": compiler_cls.__module__,
            "driver_module": selected.driver_cls.__module__,
            "compile_called": True,
            "compile_success": True,
            "compiled_kernel_hash": result.hash,
            "stage_calls": list(compiler_cls.stage_calls),
            "entry_point_load_calls": dict(load_counts),
        }
    finally:
        w10_support.metadata.EntryPoint.load = original_load


def run_fail_compile():
    """Injected compile failure must surface as a real error and exit nonzero."""
    from triton.backends.compiler import GPUTarget
    from triton_anchor.backends import get_backend_plugin_registry

    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    distribution, _ = _assert_installed()
    registry = get_backend_plugin_registry()
    records = registry.discover()
    record = w10_support._record_by_distribution(records, DISTRIBUTION)
    registry.select(
        GPUTarget(TARGET, 1, 1),
        explicit_selector=record.registry_key,
        environment={},
    )
    try:
        triton.compile(
            _compile_source(),
            target=GPUTarget(TARGET, 1, 1),
            options={"break_parse_options": True},
        )
    except RuntimeError as error:
        if "AC3 fixture injected compile failure" not in str(error):
            raise
        return {
            "plugin": DISTRIBUTION,
            "state": "selected",
            "compile_called": True,
            "compile_success": False,
            "error": str(error),
        }
    raise AssertionError("injected compile failure did not fail triton.compile()")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("real-compile", "fail-compile"),
        default="real-compile",
    )
    args = parser.parse_args()
    assert not os.environ.get("PYTHONPATH")
    if args.mode == "fail-compile":
        result = run_fail_compile()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    result = run_probe()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
