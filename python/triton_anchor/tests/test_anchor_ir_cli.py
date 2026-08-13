"""Acceptance tests for the unified AnchorIR Python API and CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest

from triton_anchor import (
    ANCHOR_IR_SPEC_VERSION,
    AnchorIRPhase,
    AnchorIRTrack,
    AnchorIRValidationError,
    AnchorIRValidationReport,
    StructuredAnchorIRValidator,
    format_anchor_ir_validation_report,
)
from triton_anchor import _anchor_ir_text_isolation as text_isolation
from triton_anchor import anchor_ir_cli
from triton_anchor.anchor_ir import AnchorIRError, AnchorIRValidator

VALID_IR = """
module {
  func.func @kernel() {
    func.return
  }
}
"""

INVALID_IR = """
module {
  func.func @kernel() {
    "smt.injected"() : () -> ()
    func.return
  }
}
"""

UNSAFE_GPU_VERIFIER_IR = """
#layout = affine_map<(d0) -> (d0)>
module attributes {
  "ttg.num-warps" = 1 : i32,
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.num-ctas" = 1 : i32
} {
  func.func @invalid(%arg: tensor<4xf32, #layout>)
      -> tensor<2x2xf32, #layout> {
    %result = "tt.reshape"(%arg) :
      (tensor<4xf32, #layout>) -> tensor<2x2xf32, #layout>
    func.return %result : tensor<2x2xf32, #layout>
  }
}
"""


def _command(
    path: Path,
    output_format: str,
    *,
    spec_version: str = ANCHOR_IR_SPEC_VERSION,
    track: AnchorIRTrack = AnchorIRTrack.LINALG,
    phase: AnchorIRPhase = AnchorIRPhase.PRE_HOOK,
    extension_dialects: tuple[str, ...] = (),
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "triton_anchor.anchor_ir_cli",
        str(path),
        "--spec-version",
        spec_version,
        "--track",
        track.value,
        "--phase",
        phase.value,
        "--format",
        output_format,
    ]
    for dialect in extension_dialects:
        command.extend(["--allow-extension", dialect])
    return command


def _python_report(
    path: Path,
    *,
    spec_version: str = ANCHOR_IR_SPEC_VERSION,
    track: AnchorIRTrack = AnchorIRTrack.LINALG,
    phase: AnchorIRPhase = AnchorIRPhase.PRE_HOOK,
    extension_dialects: tuple[str, ...] = (),
) -> AnchorIRValidationReport:
    return StructuredAnchorIRValidator().validate_text(
        path.read_text(encoding="utf-8"),
        spec_version=spec_version,
        track=track,
        phase=phase,
        source_name=str(path),
        extension_dialects=extension_dialects,
    )


def test_published_console_script_is_installed_and_has_formal_help():
    console_scripts = {
        entry.name: entry.value
        for entry in distribution("triton-anchor").entry_points
        if entry.group == "console_scripts"
    }
    assert console_scripts["triton-anchor-validate"] == (
        "triton_anchor.anchor_ir_cli:main"
    )

    executable = Path(sys.executable).with_name("triton-anchor-validate")
    assert executable.is_file()
    completed = subprocess.run(
        [str(executable), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.startswith("usage: triton-anchor-validate")
    assert "--allow-extension DIALECT" in completed.stdout


def test_python_api_and_cli_json_are_the_same_report(tmp_path):
    source = tmp_path / "invalid.mlir"
    source.write_text(INVALID_IR, encoding="utf-8")

    python_report = _python_report(source)
    completed = subprocess.run(
        _command(source, "json"),
        check=False,
        capture_output=True,
        text=True,
    )
    cli_report = json.loads(completed.stdout)

    assert completed.returncode == 1
    assert completed.stderr == ""
    assert cli_report == python_report.to_dict()
    assert cli_report["track"] == "linalg"
    assert cli_report["phase"] == "pre_hook"
    diagnostic = cli_report["diagnostics"][0]
    assert diagnostic["code"] == "AIR-LINALG-001"
    assert diagnostic["object_name"] == "smt.injected"
    assert diagnostic["operation_path"].endswith("/smt.injected#0")
    assert diagnostic["location"]["file"] == str(source)
    assert diagnostic["location"]["line"] == 4
    assert diagnostic["location"]["column"] > 0
    assert diagnostic["track"] == "linalg"
    assert diagnostic["phase"] == "pre_hook"
    assert diagnostic["hint"]


def test_cli_and_python_api_share_the_text_input_limit_report(
    tmp_path,
    monkeypatch,
    capfd,
):
    monkeypatch.setattr(text_isolation, "MAX_ANCHOR_IR_TEXT_BYTES", 32)
    source = tmp_path / "too-large.mlir"
    source.write_bytes(b"x" * 33)
    python_report = _python_report(source)

    exit_code = anchor_ir_cli.main(
        [
            str(source),
            "--spec-version",
            ANCHOR_IR_SPEC_VERSION,
            "--track",
            AnchorIRTrack.LINALG.value,
            "--phase",
            AnchorIRPhase.PRE_HOOK.value,
            "--format",
            "json",
        ]
    )
    captured = capfd.readouterr()

    assert exit_code == 1
    assert captured.err == ""
    assert json.loads(captured.out) == python_report.to_dict()
    assert [item.code for item in python_report.diagnostics] == ["AIR-PARSE-001"]
    assert "32-byte input limit" in python_report.diagnostics[0].message


@pytest.mark.parametrize("pythonioencoding", ["ascii", "latin-1"])
def test_cli_stdin_stdout_are_utf8_independent_of_pythonioencoding(
    pythonioencoding,
):
    text = 'module { "smt.中文"() : () -> () }\n'
    python_report = StructuredAnchorIRValidator().validate_text(
        text,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        source_name="<stdin>",
    )
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = pythonioencoding
    completed = subprocess.run(
        _command(Path("-"), "json"),
        input=text.encode("utf-8"),
        check=False,
        capture_output=True,
        env=environment,
    )

    assert completed.returncode == 1
    assert completed.stderr == b""
    assert json.loads(completed.stdout.decode("utf-8")) == python_report.to_dict()
    assert "中文" in completed.stdout.decode("utf-8")


def test_fail_closed_exception_uses_the_same_actionable_text_renderer(tmp_path):
    source = tmp_path / "exception.mlir"
    source.write_text(INVALID_IR, encoding="utf-8")
    report = _python_report(source)

    error = AnchorIRValidationError(report)

    assert error.report is report
    assert str(error) == format_anchor_ir_validation_report(report)
    for value in (
        report.diagnostics[0].code,
        report.diagnostics[0].operation_path,
        str(source),
        report.diagnostics[0].hint,
    ):
        assert value in str(error)

    valid_source = tmp_path / "valid-exception.mlir"
    valid_source.write_text(VALID_IR, encoding="utf-8")
    with pytest.raises(ValueError, match="valid report"):
        AnchorIRValidationError(_python_report(valid_source))


@pytest.mark.parametrize("track", list(AnchorIRTrack))
@pytest.mark.parametrize("phase", list(AnchorIRPhase))
def test_python_cli_equivalence_for_both_tracks_and_phases(
    tmp_path,
    track,
    phase,
):
    source = tmp_path / ("%s-%s.mlir" % (track.value, phase.value))
    source.write_text(INVALID_IR, encoding="utf-8")
    python_report = _python_report(source, track=track, phase=phase)

    completed = subprocess.run(
        _command(source, "json", track=track, phase=phase),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == python_report.to_dict()
    assert python_report.track == track
    assert python_report.phase == phase
    expected_code = "AIR-LINALG-001" if track is AnchorIRTrack.LINALG else "AIR-GPU-001"
    assert [item.code for item in python_report.diagnostics] == [expected_code]


@pytest.mark.parametrize("track", list(AnchorIRTrack))
@pytest.mark.parametrize(
    "invalid_ir, expected_common_code",
    [
        ('module { ".tt.hidden"() : () -> () }', "AIR-COMMON-001"),
        (
            "module attributes {func.container = {vendor.marker}} {}",
            "AIR-COMMON-003",
        ),
    ],
    ids=("empty-operation-namespace", "nested-dictionary-attribute-name"),
)
def test_python_cli_equivalence_for_namespace_and_dictionary_regressions(
    tmp_path,
    track,
    invalid_ir,
    expected_common_code,
):
    source = tmp_path / ("%s-regression.mlir" % track.value)
    source.write_text(invalid_ir, encoding="utf-8")
    python_report = _python_report(source, track=track)

    completed = subprocess.run(
        _command(source, "json", track=track),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == python_report.to_dict()
    assert [item.code for item in python_report.diagnostics] == [
        expected_common_code
    ]


def test_python_cli_equivalence_for_policy_preflight_of_unsafe_verifier_ir(
    tmp_path,
):
    source = tmp_path / "unsafe-verifier-preflight.mlir"
    source.write_text(UNSAFE_GPU_VERIFIER_IR, encoding="utf-8")
    python_report = _python_report(source, track=AnchorIRTrack.TRITON_GPU)

    completed = subprocess.run(
        _command(source, "json", track=AnchorIRTrack.TRITON_GPU),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == python_report.to_dict()
    assert {item.code for item in python_report.diagnostics} == {
        "AIR-GPU-010",
        "AIR-GPU-016",
    }


def test_python_cli_equivalence_for_declared_post_hook_extensions(tmp_path):
    source = tmp_path / "declared-extension.mlir"
    source.write_text(
        'module { "vendor_ext.accepted"() : () -> () }',
        encoding="utf-8",
    )
    extensions = ("vendor_ext",)
    python_report = _python_report(
        source,
        phase=AnchorIRPhase.POST_HOOK,
        extension_dialects=extensions,
    )

    completed = subprocess.run(
        _command(
            source,
            "json",
            phase=AnchorIRPhase.POST_HOOK,
            extension_dialects=extensions,
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert python_report.valid
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == python_report.to_dict()


def test_cli_extension_cannot_override_core_forbidden(tmp_path):
    source = tmp_path / "forbidden-extension.mlir"
    source.write_text('module { "smt.still_forbidden"() : () -> () }', encoding="utf-8")

    completed = subprocess.run(
        _command(
            source,
            "json",
            phase=AnchorIRPhase.POST_HOOK,
            extension_dialects=("smt",),
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "core-forbidden dialect(s): smt" in completed.stderr


def test_cli_extension_properties_cannot_hide_forbidden_attribute(tmp_path):
    source = tmp_path / "forbidden-property.mlir"
    source.write_text(
        'module { "vendor.op"() <{payload = #smt.marker}> : () -> () }',
        encoding="utf-8",
    )

    completed = subprocess.run(
        _command(
            source,
            "json",
            phase=AnchorIRPhase.POST_HOOK,
            extension_dialects=("vendor",),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert completed.returncode == 1
    assert completed.stderr == ""
    assert any(
        diagnostic["code"] == "AIR-LINALG-003"
        and diagnostic["object_path"] == "properties.entry[payload]"
        for diagnostic in report["diagnostics"]
    )


def test_cli_rejects_extension_declaration_during_pre_hook(tmp_path):
    source = tmp_path / "pre-hook-extension.mlir"
    source.write_text(VALID_IR, encoding="utf-8")

    completed = subprocess.run(
        _command(
            source,
            "json",
            extension_dialects=("vendor_ext",),
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "only valid for post_hook" in completed.stderr


@pytest.mark.parametrize(
    "unsupported",
    [
        "anchor-ir/1.0.0",
        "anchor-ir/1.1.0",
        "anchor-ir/2.0.0",
        "anchor-ir/9.9.9",
    ],
)
def test_unsupported_spec_version_is_a_structured_nonzero_report(
    tmp_path, unsupported
):
    source = tmp_path / "unsupported-version.mlir"
    source.write_text(VALID_IR, encoding="utf-8")
    python_report = _python_report(source, spec_version=unsupported)

    completed = subprocess.run(
        _command(source, "json", spec_version=unsupported),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == python_report.to_dict()
    assert [item.code for item in python_report.diagnostics] == ["AIR-REQUEST-002"]


@pytest.mark.parametrize(
    "option, value, expected_code",
    [
        ("--track", "not_a_track", "AIR-REQUEST-004"),
        ("--phase", "not_a_phase", "AIR-REQUEST-006"),
    ],
)
def test_cli_request_errors_share_python_structured_report(
    tmp_path, option, value, expected_code
):
    source = tmp_path / "request-error.mlir"
    source.write_text(VALID_IR, encoding="utf-8")
    track = value if option == "--track" else AnchorIRTrack.LINALG
    phase = value if option == "--phase" else AnchorIRPhase.PRE_HOOK
    python_report = StructuredAnchorIRValidator().validate_text(
        VALID_IR,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=track,
        phase=phase,
        source_name=str(source),
    )
    command = _command(source, "json")
    command[command.index(option) + 1] = value

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == python_report.to_dict()
    assert [item.code for item in python_report.diagnostics] == [expected_code]


def test_cli_text_contains_the_same_actionable_fields(tmp_path):
    source = tmp_path / "invalid-text.mlir"
    source.write_text(INVALID_IR, encoding="utf-8")
    report = _python_report(source)
    diagnostic = report.diagnostics[0]

    completed = subprocess.run(
        _command(source, "text"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    assert "AnchorIR validation: FAIL" in completed.stdout
    assert "track: linalg" in completed.stdout
    assert "phase: pre_hook" in completed.stdout
    for value in (
        diagnostic.code,
        diagnostic.object_name,
        diagnostic.operation_path,
        diagnostic.hint,
    ):
        assert value in completed.stdout


@pytest.mark.parametrize("output_format", ["text", "json"])
def test_valid_cli_input_returns_zero(tmp_path, output_format):
    source = tmp_path / "valid.mlir"
    source.write_text(VALID_IR, encoding="utf-8")

    completed = subprocess.run(
        _command(source, output_format),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    if output_format == "json":
        assert json.loads(completed.stdout)["valid"] is True
    else:
        assert "AnchorIR validation: PASS" in completed.stdout


def test_malformed_ir_and_missing_file_return_nonzero(tmp_path):
    malformed = tmp_path / "malformed.mlir"
    malformed.write_text("module { func.func @broken( { }", encoding="utf-8")
    malformed_result = subprocess.run(
        _command(malformed, "json"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert malformed_result.returncode == 1
    assert (
        json.loads(malformed_result.stdout)["diagnostics"][0]["code"] == "AIR-PARSE-001"
    )

    missing_result = subprocess.run(
        _command(tmp_path / "missing.mlir", "json"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_result.returncode == 2
    assert missing_result.stdout == ""
    assert "cannot read input" in missing_result.stderr


def test_native_dense_parser_abort_is_contained_and_api_matches_cli(tmp_path):
    source = tmp_path / "unsafe-dense.mlir"
    source.write_text(
        "module attributes {func.payload = "
        "dense<0> : tensor<1x!tt.ptr<i32>>} {}",
        encoding="utf-8",
    )

    python_report = _python_report(source)
    completed = subprocess.run(
        _command(source, "json"),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == python_report.to_dict()
    assert [item.code for item in python_report.diagnostics] == ["AIR-PARSE-001"]
    assert python_report.diagnostics[0].hint


@pytest.mark.parametrize(
    "payload",
    [
        "#bad = #ttg.slice<{}>\nmodule {}",
        (
            "#bad = #ttg.amd_mfma<{version = 1, "
            "warpsPerCTA = [1, 1], instrShape = [], "
            "isTransposed = false}>\nmodule {}"
        ),
        "#bad = #ttg.nvmma_shared<{transposed = false}>\nmodule {}",
        (
            "#bad = #ttg.blocked<{sizePerThread = [1], "
            "threadsPerWarp = [32], warpsPerCTA = [1], order = [0], "
            "CGALayout = [[2]]}>\n"
            "module {}"
        ),
    ],
)
def test_other_pinned_parser_aborts_are_contained_and_api_matches_cli(
    tmp_path,
    payload,
):
    source = tmp_path / "unsafe-triton-parser.mlir"
    source.write_text(payload, encoding="utf-8")

    python_report = _python_report(source, track=AnchorIRTrack.TRITON_GPU)
    completed = subprocess.run(
        _command(source, "json", track=AnchorIRTrack.TRITON_GPU),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == python_report.to_dict()
    assert [item.code for item in python_report.diagnostics] == ["AIR-PARSE-001"]
    assert python_report.diagnostics[0].hint


def test_wide_gpu_configuration_returns_report_instead_of_aborting(tmp_path):
    source = tmp_path / "wide-gpu-config.mlir"
    source.write_text(
        """
module attributes {
  "ttg.num-warps" = 18446744073709551615 : i65,
  "ttg.threads-per-warp" = 32 : i32,
  "ttg.num-ctas" = 1 : i32
} {
  tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
}
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        _command(
            source,
            "json",
            track=AnchorIRTrack.TRITON_GPU,
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    configuration = [
        item for item in payload["diagnostics"] if item["code"] == "AIR-GPU-011"
    ]
    assert len(configuration) == 1
    assert configuration[0]["object_name"] == "ttg.num-warps"
    assert "int32-range" in configuration[0]["hint"]


def test_cli_json_is_byte_stable_for_repeated_validation(tmp_path):
    source = tmp_path / "stable.mlir"
    source.write_text(INVALID_IR, encoding="utf-8")

    results = [
        subprocess.run(
            _command(source, "json"),
            check=False,
            capture_output=True,
            text=True,
        )
        for _ in range(3)
    ]

    assert {item.returncode for item in results} == {1}
    assert len({item.stdout for item in results}) == 1
    assert {item.stderr for item in results} == {""}


def test_legacy_boolean_and_raise_wrappers_remain_compatible():
    validator = AnchorIRValidator(track=AnchorIRTrack.LINALG)

    assert validator.is_valid(VALID_IR)
    assert not validator.is_valid(INVALID_IR)
    with pytest.raises(AnchorIRError, match="AnchorIR validation failed"):
        validator.validate_and_raise(INVALID_IR, context="cli-compat")
