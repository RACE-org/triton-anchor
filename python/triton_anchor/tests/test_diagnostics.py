"""Tests for pass-level diagnostics."""

from __future__ import annotations

import sys
import types

import pytest

from triton_anchor.diagnose import main
from triton_anchor.diagnostics import PassDescriptor, PassDiagnostic


class FakeModule:
    def __init__(self) -> None:
        self.context = object()
        self.ir = "module { func.func @kernel() { return } }"

    def __str__(self) -> str:
        return self.ir


class FakePassManager:
    def __init__(self, context) -> None:
        self.context = context
        self.passes = []
        self.debug_enabled = False

    def enable_debug(self) -> None:
        self.debug_enabled = True

    def run(self, mod: FakeModule) -> None:
        for pass_fn in self.passes:
            pass_fn(mod)


def _install_fake_libtriton(monkeypatch: pytest.MonkeyPatch) -> None:
    ir_module = types.SimpleNamespace(
        pass_manager=lambda context: FakePassManager(context),
    )
    libtriton_module = types.SimpleNamespace(ir=ir_module)

    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    monkeypatch.setitem(sys.modules, "triton._C", types.ModuleType("triton._C"))
    monkeypatch.setitem(sys.modules, "triton._C.libtriton", libtriton_module)


def test_pass_diagnostic_reports_first_failed_pass(tmp_path, monkeypatch):
    _install_fake_libtriton(monkeypatch)

    def pass_ok(mod: FakeModule) -> None:
        mod.ir = mod.ir.replace("return", "// pass ok\n    return")

    def pass_fail(mod: FakeModule) -> None:
        raise RuntimeError("synthetic pass failure")

    descriptors = [
        PassDescriptor("common.inliner", lambda pm: pm.passes.append(pass_ok)),
        PassDescriptor("ttir.combine", lambda pm: pm.passes.append(pass_fail)),
        PassDescriptor("common.canonicalizer", lambda pm: pm.passes.append(pass_ok)),
    ]

    monkeypatch.setattr(
        "triton_anchor.diagnostics.build_ttir_pass_descriptors",
        lambda hw=None: descriptors,
    )

    result = PassDiagnostic(output_dir=tmp_path).diagnose_ttir(FakeModule())

    assert not result.ok
    assert result.failed_pass == "ttir.combine"
    assert result.failed_index == 2
    assert result.executed_passes == 2
    assert "synthetic pass failure" in result.error
    assert (tmp_path / "01-common.inliner.before.mlir").exists()
    assert (tmp_path / "01-common.inliner.after.mlir").exists()
    assert (tmp_path / "02-ttir.combine.before.mlir").exists()
    assert (tmp_path / "summary.json").exists()


def test_pass_diagnostic_extracts_mlir_location_and_operation(tmp_path, monkeypatch):
    _install_fake_libtriton(monkeypatch)

    class LocatedModule(FakeModule):
        def __init__(self) -> None:
            self.context = object()
            self.ir = "\n".join(
                [
                    "module {",
                    '  %0 = "tt.load"() : () -> f32',
                    "  return",
                    "}",
                ]
            )

    def pass_fail(mod: FakeModule) -> None:
        raise RuntimeError(
            "<stdin>:2:8: error: failed to legalize operation 'tt.load'"
        )

    descriptors = [
        PassDescriptor("ttir.combine", lambda pm: pm.passes.append(pass_fail)),
    ]

    monkeypatch.setattr(
        "triton_anchor.diagnostics.build_ttir_pass_descriptors",
        lambda hw=None: descriptors,
    )

    result = PassDiagnostic(output_dir=tmp_path).diagnose_ttir(LocatedModule())

    assert not result.ok
    assert result.mlir_location is not None
    assert result.mlir_location.line == 2
    assert result.mlir_location.column == 8
    assert result.mlir_location.operation == "tt.load"
    assert result.mlir_location.ir_line == 2
    assert '"tt.load"' in result.mlir_location.ir_snippet
    assert (tmp_path / "01-ttir.combine.diagnostic.txt").exists()


def test_pass_diagnostic_success_writes_summary(tmp_path, monkeypatch):
    _install_fake_libtriton(monkeypatch)

    def pass_ok(mod: FakeModule) -> None:
        mod.ir += "\n// ok"

    descriptors = [
        PassDescriptor("common.inliner", lambda pm: pm.passes.append(pass_ok)),
        PassDescriptor("ttir.combine", lambda pm: pm.passes.append(pass_ok)),
    ]

    monkeypatch.setattr(
        "triton_anchor.diagnostics.build_ttir_pass_descriptors",
        lambda hw=None: descriptors,
    )

    result = PassDiagnostic(output_dir=tmp_path).diagnose_ttir(FakeModule())

    assert result.ok
    assert result.failed_pass is None
    assert result.executed_passes == 2
    assert (tmp_path / "02-ttir.combine.after.mlir").exists()
    assert result.summary_path == tmp_path / "summary.json"


def test_triton_linalg_pipeline_can_be_diagnosed(tmp_path, monkeypatch):
    _install_fake_libtriton(monkeypatch)

    def pass_ok(mod: FakeModule) -> None:
        mod.ir += "\n// linalg ok"

    def pass_fail(mod: FakeModule) -> None:
        raise RuntimeError("operation 'tt.store' failed in linalg lowering")

    descriptors = [
        PassDescriptor(
            "triton_linalg.canonicalize_triton",
            lambda pm: pm.passes.append(pass_ok),
        ),
        PassDescriptor(
            "triton_linalg.triton_to_linalg",
            lambda pm: pm.passes.append(pass_fail),
        ),
    ]

    monkeypatch.setattr(
        "triton_anchor.diagnostics.build_triton_linalg_pass_descriptors",
        lambda: descriptors,
    )

    result = PassDiagnostic(output_dir=tmp_path).diagnose_triton_linalg(FakeModule())

    assert not result.ok
    assert result.pipeline == "triton-linalg"
    assert result.failed_pass == "triton_linalg.triton_to_linalg"
    assert result.failed_index == 2
    assert result.mlir_location is not None
    assert result.mlir_location.operation == "tt.store"


def test_cli_python_mode_diagnoses_in_memory_module(tmp_path, monkeypatch):
    _install_fake_libtriton(monkeypatch)

    def pass_ok(mod: FakeModule) -> None:
        mod.ir += "\n// diagnosed"

    descriptors = [
        PassDescriptor("common.inliner", lambda pm: pm.passes.append(pass_ok)),
    ]

    monkeypatch.setattr(
        "triton_anchor.diagnostics.build_ttir_pass_descriptors",
        lambda hw=None: descriptors,
    )
    monkeypatch.setattr(
        "triton_anchor.diagnose._make_ttir_from_python",
        lambda python_target, signature, constants: FakeModule(),
    )

    exit_code = main(
        [
            "--python",
            "tests.test_smoke:_smoke_add_kernel",
            "--signature",
            "*fp32,*fp32,*fp32,i32",
            "--constant",
            "BLOCK=256",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "01-common.inliner.before.mlir").exists()
    assert (tmp_path / "01-common.inliner.after.mlir").exists()


def test_cli_help_loads_without_libtriton(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "triton-anchor-diagnose" in captured.out
    assert "triton-linalg" in captured.out
    assert "--python" in captured.out
