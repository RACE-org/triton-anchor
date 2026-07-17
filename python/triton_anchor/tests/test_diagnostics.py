"""Tests for pass-level diagnostics."""

from __future__ import annotations

import sys
import types

import pytest

from triton_anchor.diagnose import InputDiagnosticError, main
from triton_anchor.diagnostics import (
    PassDescriptor,
    PassDiagnostic,
    StageDiagnostic,
    extract_mlir_location,
)


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
        self.verifier_enabled = True

    def enable_debug(self) -> None:
        self.debug_enabled = True

    def enable_verifier(self, value: bool) -> None:
        self.verifier_enabled = value

    def run(self, mod: FakeModule) -> None:
        for pass_fn in self.passes:
            pass_fn(mod)


def _install_fake_libtriton(monkeypatch: pytest.MonkeyPatch) -> None:
    ir_module = types.SimpleNamespace(
        pass_manager=lambda context: FakePassManager(context),
    )
    common_module = types.ModuleType("triton._C.libtriton.passes.common")
    common_module.add_cse = lambda pm: pm.passes.append(lambda mod: None)
    common_module.add_canonicalizer = lambda pm: pm.passes.append(lambda mod: None)
    passes_module = types.ModuleType("triton._C.libtriton.passes")
    passes_module.common = common_module
    libtriton_module = types.ModuleType("triton._C.libtriton")
    libtriton_module.ir = ir_module
    libtriton_module.passes = passes_module

    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    monkeypatch.setitem(sys.modules, "triton._C", types.ModuleType("triton._C"))
    monkeypatch.setitem(sys.modules, "triton._C.libtriton", libtriton_module)
    monkeypatch.setitem(sys.modules, "triton._C.libtriton.passes", passes_module)
    monkeypatch.setitem(
        sys.modules,
        "triton._C.libtriton.passes.common",
        common_module,
    )


def _install_fake_sophgo_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    def add_triton_to_ppl(pm: FakePassManager) -> None:
        pm.passes.append(lambda mod: setattr(mod, "ir", str(mod) + "\n// triton->ppl"))

    def add_linalg_to_ppl(pm: FakePassManager) -> None:
        def fail(mod):
            raise RuntimeError("operation 'linalg.generic' failed in ppl lowering")

        pm.passes.append(fail)

    passes_module = types.SimpleNamespace(
        add_triton_to_ppl=add_triton_to_ppl,
        add_linalg_to_ppl=add_linalg_to_ppl,
    )
    c_module = types.ModuleType("triton_sophgo._C")
    c_module.passes = passes_module

    monkeypatch.setitem(sys.modules, "triton_sophgo", types.ModuleType("triton_sophgo"))
    monkeypatch.setitem(sys.modules, "triton_sophgo._C", c_module)


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


def test_extract_mlir_location_maps_pplir_broadcast_failure(tmp_path):
    before_ir = "\n".join(
        [
            '#loc = loc("/workspace/kernel.py":1:0)',
            "module {",
            "  %0 = tensor.empty() : tensor<4x8x16xf32> loc(#loc2)",
            "  %broadcasted = linalg.broadcast ins(%collapsed : tensor<4x16xf32>) outs(%0 : tensor<4x8x16xf32>) dimensions = [1]  loc(#loc2)",
            "} loc(#loc)",
            '#loc2 = loc("/workspace/kernel.py":28:24)',
        ]
    )
    diagnostic_text = "\n".join(
        [
            "C dimension affected, case not supported yet.",
            'loc("/workspace/kernel.py":28:24): error: '
            "'tensor.collapse_shape' op operand #0 must be tensor of any type values",
        ]
    )

    location = extract_mlir_location(
        "PassManager::run failed",
        diagnostic_text,
        before_ir,
        tmp_path / "before.mlir",
    )

    assert location is not None
    assert location.file == "/workspace/kernel.py"
    assert location.line == 28
    assert location.column == 24
    assert location.operation == "linalg.broadcast"
    assert location.ir_line == 4
    assert "linalg.broadcast" in location.ir_snippet


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


def test_sophgo_pplir_pipeline_can_be_diagnosed(tmp_path, monkeypatch):
    _install_fake_libtriton(monkeypatch)
    _install_fake_sophgo_passes(monkeypatch)

    common = types.SimpleNamespace(
        add_cse=lambda pm: pm.passes.append(lambda mod: None),
        add_canonicalizer=lambda pm: pm.passes.append(lambda mod: None),
    )
    monkeypatch.setattr(
        sys.modules["triton._C.libtriton"],
        "passes",
        types.SimpleNamespace(common=common),
    )

    result = PassDiagnostic(output_dir=tmp_path).diagnose_sophgo_pplir(FakeModule())

    assert not result.ok
    assert result.pipeline == "sophgo-pplir"
    assert result.failed_pass == "sophgo.linalg_to_ppl"
    assert result.failed_index == 2
    assert result.mlir_location is not None
    assert result.mlir_location.operation == "linalg.generic"
    assert (tmp_path / "02-sophgo.linalg_to_ppl.diagnostic.txt").exists()


def test_stage_diagnostic_writes_external_stage_failure(tmp_path):
    input_path = tmp_path / "kernel.mlir"
    input_path.write_text(
        "\n".join(
            [
                "module {",
                '  %0 = "ppl.copy"() : () -> ()',
                "}",
            ]
        ),
        encoding="utf-8",
    )

    result = StageDiagnostic(output_dir=tmp_path / "diag").record_failure(
        "ppl-compile",
        f"{input_path}:2:4: error: operation 'ppl.copy' failed",
        diagnostic_text="ppl-compile failed",
        command=["ppl-compile", input_path],
        returncode=1,
        input_path=input_path,
        artifacts={"work_dir": tmp_path},
    )

    assert not result.ok
    assert result.stage == "ppl-compile"
    assert result.mlir_location is not None
    assert result.mlir_location.operation == "ppl.copy"
    assert result.mlir_location.ir_line == 2
    assert result.summary_path == tmp_path / "diag" / "summary.json"
    assert (tmp_path / "diag" / "ppl-compile.diagnostic.txt").exists()


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
