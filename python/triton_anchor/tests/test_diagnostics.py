"""Tests for pass-level diagnostics."""

from __future__ import annotations

import sys
import types

import pytest

from triton_anchor.diagnose import InputDiagnosticError, main
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
        raise RuntimeError("<stdin>:2:8: error: failed to legalize operation 'tt.load'")

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


def test_cli_reports_input_parse_diagnostic(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "bad.mlir"
    input_path.write_text(
        "\n".join(
            [
                "module {",
                '  "tt.load"() : () -> f32',
                "}",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "diag"

    def fail_parse(path):
        original = RuntimeError(f"{path}:2:3: error: failed to parse operation 'tt.load'")
        raise InputDiagnosticError(
            "input-parse",
            str(original),
            diagnostic_text=str(original),
            input_path=path,
            original=original,
        )

    monkeypatch.setattr("triton_anchor.diagnose._load_mlir_module", fail_parse)

    exit_code = main([str(input_path), "--output-dir", str(output_dir)])

    assert exit_code == 2
    assert (output_dir / "input-parse.diagnostic.txt").exists()
    assert (output_dir / "summary.json").exists()
    diagnostic_text = (output_dir / "input-parse.diagnostic.txt").read_text(
        encoding="utf-8"
    )
    assert "stage: input-parse" in diagnostic_text
    assert "operation: tt.load" in diagnostic_text
    captured = capsys.readouterr()
    assert "FAILED: input-parse failed before pass diagnostics." in captured.out
    assert "location:" in captured.out


def test_cli_reports_python_frontend_diagnostic(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "diag"
    python_target = "tests.test_smoke:_smoke_add_kernel"

    def fail_frontend(python_target, signature, constants):
        original = RuntimeError(
            'loc("kernel.py":4:5): error: operation \'tl.load\' failed before TTIR'
        )
        raise InputDiagnosticError(
            "python-frontend",
            str(original),
            diagnostic_text=str(original),
            python_target=python_target,
            original=original,
        )

    monkeypatch.setattr(
        "triton_anchor.diagnose._make_ttir_from_python",
        fail_frontend,
    )

    exit_code = main(
        [
            "--python",
            python_target,
            "--signature",
            "*fp32,*fp32,*fp32,i32",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 2
    assert (output_dir / "python-frontend.diagnostic.txt").exists()
    assert (output_dir / "summary.json").exists()
    diagnostic_text = (output_dir / "python-frontend.diagnostic.txt").read_text(
        encoding="utf-8"
    )
    assert "stage: python-frontend" in diagnostic_text
    assert "python_target: tests.test_smoke:_smoke_add_kernel" in diagnostic_text
    assert "operation: tl.load" in diagnostic_text
    captured = capsys.readouterr()
    assert "FAILED: python-frontend failed before TTIR generation." in captured.out
    assert "operation: tl.load" in captured.out


def test_cli_help_loads_without_libtriton(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "triton-anchor-diagnose" in captured.out
    assert "triton-linalg" in captured.out
    assert "--python" in captured.out


# ── T2.5 observability metrics tests ─────────────────────────────────────────


def test_pass_diagnostic_records_timing_and_ir_sizes(tmp_path, monkeypatch):
    """T2.5: verify per-pass timing, IR size tracking, and aggregate metrics."""
    _install_fake_libtriton(monkeypatch)

    def pass_a(mod):
        pass

    def pass_b(mod):
        pass

    descriptors = [
        PassDescriptor("pass.a", lambda pm: pm.passes.append(pass_a)),
        PassDescriptor("pass.b", lambda pm: pm.passes.append(pass_b)),
    ]

    monkeypatch.setattr(
        "triton_anchor.diagnostics.build_ttir_pass_descriptors",
        lambda hw=None: descriptors,
    )

    result = PassDiagnostic(output_dir=tmp_path).diagnose_ttir(FakeModule())

    assert result.ok
    assert result.executed_passes == 2

    # T2.5 aggregate metrics
    assert result.total_duration_ms >= 0.0
    assert result.input_ir_bytes > 0
    assert result.output_ir_bytes > 0
    assert result.peak_rss_bytes >= 0  # may be 0 on some platforms

    # T2.5 per-pass metrics
    for rec in result.records:
        assert rec.duration_ms >= 0.0
        assert rec.before_ir_bytes > 0
        assert rec.after_ir_bytes > 0
        assert rec.ir_delta_bytes == rec.after_ir_bytes - rec.before_ir_bytes
        assert rec.peak_rss_bytes >= 0

    # T2.5 slowest_pass property
    slowest = result.slowest_pass
    assert slowest is not None
    assert slowest.duration_ms == max(r.duration_ms for r in result.records)


def test_pass_diagnostic_metrics_on_failure(tmp_path, monkeypatch):
    """T2.5: verify metrics are captured even when a pass fails."""
    _install_fake_libtriton(monkeypatch)

    def pass_ok(mod):
        pass

    def pass_fail(mod):
        raise RuntimeError("deliberate failure")

    descriptors = [
        PassDescriptor("pass.ok", lambda pm: pm.passes.append(pass_ok)),
        PassDescriptor("pass.fail", lambda pm: pm.passes.append(pass_fail)),
    ]

    monkeypatch.setattr(
        "triton_anchor.diagnostics.build_ttir_pass_descriptors",
        lambda hw=None: descriptors,
    )

    result = PassDiagnostic(output_dir=tmp_path).diagnose_ttir(FakeModule())

    assert not result.ok
    assert result.executed_passes == 2

    # T2.5: metrics captured up to failure point
    assert result.total_duration_ms >= 0.0
    assert result.input_ir_bytes > 0
    # output_ir_bytes is last known good (before the failed pass)
    assert result.output_ir_bytes > 0

    # The failed pass should have timing even though it failed
    failed = result.failed_record
    assert failed is not None
    assert failed.duration_ms >= 0.0
    assert failed.before_ir_bytes > 0
    assert failed.after_ir_bytes == 0  # pass failed, no output
    assert failed.ir_delta_bytes == 0
