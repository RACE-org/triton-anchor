"""Acceptance tests for the C++ structured AnchorIR operation validator."""

import os
import tempfile
import threading
import time

import pytest
from triton._C.libtriton import anchor, ir

from triton_anchor import (
    ANCHOR_IR_SPEC_VERSION,
    AnchorIRPhase,
    AnchorIRTrack,
    StructuredAnchorIRValidator,
    resolve_anchor_ir_policy,
)
from triton_anchor._anchor_ir_text_isolation import requires_parser_isolation
from triton_anchor import _anchor_ir_text_isolation as text_isolation

VALID_NESTED_IR = """
module {
  func.func @nested(%cond: i1) {
    scf.if %cond {
      scf.if %cond {
        %c0 = arith.constant 0 : i32
      }
    }
    func.return
  }
}
"""

DEEP_UNKNOWN_IR = """
module {
  func.func @nested(%cond: i1) {
    scf.if %cond {
      scf.if %cond {
        "vendor.deep"() : () -> ()
      }
    }
    func.return
  }
}
"""


def _validate_text(
    text,
    *,
    track=AnchorIRTrack.LINALG,
    phase=AnchorIRPhase.PRE_HOOK,
    source_name="case.mlir",
    extension_dialects=(),
    context=None,
):
    return StructuredAnchorIRValidator().validate_text(
        text,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=track,
        phase=phase,
        context=context,
        source_name=source_name,
        extension_dialects=extension_dialects,
    )


def test_accepts_real_module_op():
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", encoding="utf-8", delete=False
    ) as source:
        source.write(VALID_NESTED_IR)
        source_path = source.name
    try:
        module = ir.parse_mlir_module(source_path, context)
        report = StructuredAnchorIRValidator().validate_module(
            module,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            phase=AnchorIRPhase.PRE_HOOK,
        )
    finally:
        os.unlink(source_path)

    assert report.valid
    assert report.diagnostics == ()


def test_explicit_fresh_context_loads_required_dialects():
    report = StructuredAnchorIRValidator().validate_text(
        VALID_NESTED_IR,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        context=ir.context(),
        source_name="explicit-context.mlir",
    )

    assert report.valid
    assert report.diagnostics == ()


def test_explicit_context_retains_registered_parser_path_for_safe_slice_text(
    monkeypatch,
):
    text = """
#parent = #triton_gpu.blocked<{
  sizePerThread = [1, 1],
  threadsPerWarp = [1, 32],
  warpsPerCTA = [1, 1],
  order = [1, 0]
}>
#slice = #triton_gpu.slice<{dim = 0, parent = #parent}>
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @kernel(%arg: tensor<4xf32, #slice>)
      -> tensor<1x4xf32, #parent> {
    %result = "tt.expand_dims"(%arg) {axis = 0 : i32} :
      (tensor<4xf32, #slice>) -> tensor<1x4xf32, #parent>
    func.return %result : tensor<1x4xf32, #parent>
  }
}
"""
    assert requires_parser_isolation(text)

    def worker_must_not_run(*_args, **_kwargs):
        raise AssertionError("explicit MLIR Context was replaced by worker")

    monkeypatch.setattr(text_isolation, "run_isolated_native_text", worker_must_not_run)
    report = StructuredAnchorIRValidator().validate_text(
        text,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.TRITON_GPU,
        phase=AnchorIRPhase.PRE_HOOK,
        context=ir.context(),
        source_name="explicit-slice-context.mlir",
    )

    assert report.valid


def test_top_level_forbidden_operation_is_detected():
    report = _validate_text(
        'module {\n  "smt.top"() : () -> ()\n}\n',
        source_name="top.mlir",
    )

    assert not report.valid
    assert len(report.diagnostics) == 1
    diagnostic = report.diagnostics[0]
    assert report.spec_version == ANCHOR_IR_SPEC_VERSION
    assert report.track == AnchorIRTrack.LINALG
    assert report.phase == AnchorIRPhase.PRE_HOOK
    assert diagnostic.code == "AIR-LINALG-001"
    assert diagnostic.spec_version == report.spec_version
    assert diagnostic.track == report.track
    assert diagnostic.phase == report.phase
    assert diagnostic.object_name == "smt.top"
    assert diagnostic.operation_path == ("builtin.module/region[0]/block[0]/smt.top#0")
    assert diagnostic.location is not None
    assert diagnostic.location.file == "top.mlir"
    assert diagnostic.location.line == 2
    assert diagnostic.location.column > 0


def test_unknown_and_forbidden_dialects_have_different_codes():
    unknown = _validate_text('module { "vendor.op"() : () -> () }')
    forbidden = _validate_text('module { "smt.op"() : () -> () }')
    gpu_forbidden = _validate_text(
        'module { "smt.op"() : () -> () }',
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert [item.code for item in unknown.diagnostics] == ["AIR-COMMON-001"]
    assert [item.code for item in forbidden.diagnostics] == ["AIR-LINALG-001"]
    assert [item.code for item in gpu_forbidden.diagnostics] == ["AIR-GPU-001"]


def test_unregistered_op_inside_allowed_dialect_is_rejected():
    linalg = _validate_text('module { "linalg.fake"() : () -> () }')
    gpu = _validate_text(
        """
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  "tt.fake"() : () -> ()
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )

    for report in (linalg, gpu):
        assert not report.valid
        assert [item.code for item in report.diagnostics] == ["AIR-VERIFY-001"]


@pytest.mark.parametrize("track", list(AnchorIRTrack))
@pytest.mark.parametrize(
    "text, expected_code, expected_name, expected_object_path",
    [
        (
            'module { ".tt.hidden"() : () -> () }',
            "AIR-COMMON-001",
            ".tt.hidden",
            "",
        ),
        (
            'module attributes {".smt.marker"} {}',
            "AIR-COMMON-003",
            ".smt.marker",
            "attribute[.smt.marker]",
        ),
    ],
    ids=("empty-operation-namespace", "empty-attribute-namespace"),
)
def test_empty_or_malformed_namespace_cannot_bypass_track_policy(
    track,
    text,
    expected_code,
    expected_name,
    expected_object_path,
):
    report = _validate_text(text, track=track, source_name="namespace.mlir")

    assert not report.valid
    assert [item.code for item in report.diagnostics] == [expected_code]
    diagnostic = report.diagnostics[0]
    assert diagnostic.object_name == expected_name
    assert diagnostic.object_path == expected_object_path
    assert diagnostic.location is not None
    assert diagnostic.location.file == "namespace.mlir"
    assert "<empty>" in diagnostic.message


@pytest.mark.parametrize(
    "track, key, expected_code",
    [
        (AnchorIRTrack.LINALG, "smt.marker", "AIR-LINALG-003"),
        (AnchorIRTrack.TRITON_GPU, "smt.marker", "AIR-GPU-003"),
        (AnchorIRTrack.LINALG, "vendor.marker", "AIR-COMMON-003"),
        (AnchorIRTrack.TRITON_GPU, "vendor.marker", "AIR-COMMON-003"),
    ],
)
def test_dictionary_attribute_names_are_checked_recursively(
    track,
    key,
    expected_code,
):
    report = _validate_text(
        "module attributes {func.container = {%s}} {}" % key,
        track=track,
        source_name="dictionary-name.mlir",
    )

    assert not report.valid
    assert [item.code for item in report.diagnostics] == [expected_code]
    diagnostic = report.diagnostics[0]
    assert diagnostic.object_kind.value == "attribute"
    assert diagnostic.object_name == key
    assert diagnostic.operation_path == "builtin.module"
    assert diagnostic.object_path == "attribute[func.container].entry[%s]" % key


def test_arbitrarily_deep_region_operation_is_detected_with_stable_path():
    reports = [
        _validate_text(DEEP_UNKNOWN_IR, source_name="deep.mlir") for _ in range(10)
    ]
    expected = reports[0].to_dict()

    assert all(report.to_dict() == expected for report in reports)
    assert len(reports[0].diagnostics) == 1
    diagnostic = reports[0].diagnostics[0]
    assert diagnostic.code == "AIR-COMMON-001"
    assert diagnostic.operation_path == (
        "builtin.module/region[0]/block[0]/func.func@nested#0"
        "/region[0]/block[0]/scf.if#0"
        "/region[0]/block[0]/scf.if#0"
        "/region[0]/block[0]/vendor.deep#0"
    )
    assert diagnostic.location is not None
    assert diagnostic.location.file == "deep.mlir"
    assert diagnostic.location.line == 6


def test_legal_nested_regions_do_not_report_false_positive():
    report = _validate_text(VALID_NESTED_IR, source_name="legal.mlir")
    assert report.valid
    assert report.diagnostics == ()


def test_syntax_failure_has_independent_parse_diagnostic():
    report = _validate_text(
        "module { func.func @broken( { }",
        source_name="syntax.mlir",
    )

    assert not report.valid
    assert len(report.diagnostics) == 1
    diagnostic = report.diagnostics[0]
    assert diagnostic.code == "AIR-PARSE-001"
    assert diagnostic.object_name == "builtin.module"
    assert diagnostic.operation_path == ""
    assert diagnostic.hint
    assert diagnostic.location is not None
    assert diagnostic.location.file == "syntax.mlir"


def test_native_text_entry_contains_dense_parser_assertion():
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    )
    unsafe = "module attributes {func.payload = dense<0> : tensor<1x!tt.ptr<i32>>} {}"

    report = anchor.validate_anchor_ir_text(
        unsafe,
        context,
        policy.to_dict(),
        "native-unsafe-dense.mlir",
    )

    assert report["valid"] is False
    diagnostic = report["diagnostics"][0]
    assert diagnostic["code"] == "AIR-PARSE-001"
    assert diagnostic["location"]["file"] == "native-unsafe-dense.mlir"
    assert "dense literal" in diagnostic["message"]


def test_dense_string_literal_with_custom_type_is_not_preflight_rejected():
    report = _validate_text(
        'module attributes {func.payload = dense<"value"> : tensor<1x!tt.ptr<i32>>} {}'
    )

    assert [item.code for item in report.diagnostics] == ["AIR-LINALG-002"]


@pytest.mark.parametrize(
    "text",
    [
        'module attributes {func.note = "dense<0> : tensor<1x!smt.type>"} {}',
        "module {\n  // dense<0> : tensor<1x!smt.type>\n}",
    ],
)
def test_dense_preflight_ignores_string_data_and_line_comments(text):
    assert not requires_parser_isolation(text)

    report = _validate_text(text)

    assert report.valid
    assert report.diagnostics == ()


@pytest.mark.parametrize("action", ["validate", "normalize"])
@pytest.mark.parametrize(
    ("timeout_seconds", "expected_timeout"),
    ((None, 60.0), (0.25, 0.25)),
)
def test_isolated_parser_timeout_fails_closed_with_structured_report(
    action,
    timeout_seconds,
    expected_timeout,
    monkeypatch,
):
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    ).to_dict()

    def time_out(*args, **kwargs):
        raise text_isolation.subprocess.TimeoutExpired(
            args[0],
            kwargs["timeout"],
        )

    monkeypatch.setattr(text_isolation.subprocess, "run", time_out)
    result = text_isolation.run_isolated_native_text(
        action,
        "module {}",
        policy,
        "timeout.mlir",
        timeout_seconds=timeout_seconds,
    )
    report = result if action == "validate" else result["validation_report"]

    assert report["valid"] is False
    assert report["diagnostics"][0]["code"] == "AIR-PARSE-001"
    assert "safety timeout" in report["diagnostics"][0]["message"]
    assert str(expected_timeout).rstrip("0").rstrip(".") in (
        report["diagnostics"][0]["message"]
    )
    if action == "normalize":
        assert result["normalized_text"] is None


@pytest.mark.parametrize("action", ["validate", "normalize"])
def test_isolated_parser_enforces_text_input_limit_before_worker_spawn(
    action,
    monkeypatch,
):
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    ).to_dict()
    monkeypatch.setattr(text_isolation, "MAX_ANCHOR_IR_TEXT_BYTES", 16)

    def worker_must_not_run(*_args, **_kwargs):
        raise AssertionError("oversized text reached the parser worker")

    monkeypatch.setattr(text_isolation.subprocess, "run", worker_must_not_run)
    result = text_isolation.run_isolated_native_text(
        action,
        "测" * 6,
        policy,
        "too-large.mlir",
    )
    report = result if action == "validate" else result["validation_report"]

    assert report["valid"] is False
    assert report["diagnostics"][0]["code"] == "AIR-PARSE-001"
    assert "16-byte input limit" in report["diagnostics"][0]["message"]
    if action == "normalize":
        assert result["normalized_text"] is None


def test_public_text_validator_enforces_utf8_byte_limit_before_any_parser(
    monkeypatch,
):
    monkeypatch.setattr(text_isolation, "MAX_ANCHOR_IR_TEXT_BYTES", 16)

    def worker_must_not_run(*_args, **_kwargs):
        raise AssertionError("oversized text reached the parser worker")

    monkeypatch.setattr(text_isolation, "run_isolated_native_text", worker_must_not_run)
    kwargs = {
        "spec_version": ANCHOR_IR_SPEC_VERSION,
        "track": AnchorIRTrack.LINALG,
        "phase": AnchorIRPhase.PRE_HOOK,
        "source_name": "too-large.mlir",
    }
    default_report = StructuredAnchorIRValidator().validate_text("测" * 6, **kwargs)
    explicit_report = StructuredAnchorIRValidator().validate_text(
        "测" * 6,
        context=object(),
        **kwargs,
    )

    assert default_report.to_dict() == explicit_report.to_dict()
    assert default_report.valid is False
    assert [item.code for item in default_report.diagnostics] == ["AIR-PARSE-001"]
    assert "16-byte input limit" in default_report.diagnostics[0].message


def test_public_text_validator_bounds_source_name_before_any_parser(monkeypatch):
    monkeypatch.setattr(
        text_isolation,
        "MAX_ANCHOR_IR_SOURCE_NAME_BYTES",
        8,
        raising=False,
    )

    def worker_must_not_run(*_args, **_kwargs):
        raise AssertionError("oversized source_name reached the parser worker")

    monkeypatch.setattr(text_isolation, "run_isolated_native_text", worker_must_not_run)
    kwargs = {
        "spec_version": ANCHOR_IR_SPEC_VERSION,
        "track": AnchorIRTrack.LINALG,
        "phase": AnchorIRPhase.PRE_HOOK,
        "source_name": "012345678",
    }
    validator = StructuredAnchorIRValidator()

    with pytest.raises(ValueError, match="source_name exceeds the 8-byte"):
        validator.validate_text(VALID_NESTED_IR, **kwargs)
    with pytest.raises(ValueError, match="source_name exceeds the 8-byte"):
        validator.validate_text(VALID_NESTED_IR, context=object(), **kwargs)


@pytest.mark.parametrize("track", list(AnchorIRTrack))
def test_nested_object_depth_limit_fails_closed_before_recursive_scans(track):
    value = "0 : i32"
    for _ in range(300):
        value = "[%s]" % value
    text = "module attributes {func.container = %s} {}\n" % value

    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", encoding="utf-8", delete=False
    ) as source:
        source.write(text)
        source_path = source.name
    try:
        module = ir.parse_mlir_module(source_path, context)
        report = StructuredAnchorIRValidator().validate_module(
            module,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=track,
            phase=AnchorIRPhase.PRE_HOOK,
        )
    finally:
        os.unlink(source_path)

    assert not report.valid
    assert [item.code for item in report.diagnostics] == ["AIR-COMMON-004"]
    diagnostic = report.diagnostics[0]
    assert diagnostic.object_name == "nested object depth"
    assert diagnostic.operation_path == "builtin.module"
    assert diagnostic.location is not None
    assert diagnostic.location.file == source_path


@pytest.mark.parametrize("action", ["validate", "normalize"])
def test_isolated_parser_signal_fails_closed_with_structured_report(
    action,
    monkeypatch,
):
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    ).to_dict()
    completed = text_isolation.subprocess.CompletedProcess(
        args=["worker"],
        returncode=-11,
        stdout="",
        stderr="native crash",
    )
    monkeypatch.setattr(
        text_isolation.subprocess,
        "run",
        lambda *args, **kwargs: completed,
    )

    result = text_isolation.run_isolated_native_text(
        action,
        "module {}",
        policy,
        "signal.mlir",
    )
    report = result if action == "validate" else result["validation_report"]

    assert report["valid"] is False
    assert report["diagnostics"][0]["code"] == "AIR-PARSE-001"
    assert "signal 11" in report["diagnostics"][0]["message"]
    if action == "normalize":
        assert result["normalized_text"] is None


@pytest.mark.parametrize("action", ["validate", "normalize"])
def test_isolated_parser_positive_exit_is_infrastructure_error(action, monkeypatch):
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    ).to_dict()
    completed = text_isolation.subprocess.CompletedProcess(
        args=["worker"],
        returncode=1,
        stdout="",
        stderr="ModuleNotFoundError: triton_anchor\n" + ("x" * 1000),
    )
    monkeypatch.setattr(
        text_isolation.subprocess,
        "run",
        lambda *args, **kwargs: completed,
    )

    with pytest.raises(RuntimeError, match="status 1.*ModuleNotFoundError"):
        text_isolation.run_isolated_native_text(
            action,
            "module {}",
            policy,
            "worker.mlir",
        )


def test_isolated_parser_process_creation_failure_is_infrastructure_error(
    monkeypatch,
):
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    ).to_dict()

    def fail_to_start(*args, **kwargs):
        raise OSError("resource unavailable")

    monkeypatch.setattr(text_isolation.subprocess, "run", fail_to_start)
    with pytest.raises(RuntimeError, match="failed to start isolated"):
        text_isolation.run_isolated_native_text(
            "validate",
            "module {}",
            policy,
            "spawn.mlir",
        )


@pytest.mark.parametrize(
    "stdout, expected",
    [
        (
            '{"protocol_version": "anchor-ir-text-worker/1.0.0", '
            '"worker_error": "boom"}',
            "parser failed",
        ),
        ("not json", "invalid JSON"),
        (
            '{"protocol_version": "anchor-ir-text-worker/1.0.0", "unexpected": {}}',
            "invalid result",
        ),
    ],
)
def test_isolated_parser_malformed_worker_response_is_infrastructure_error(
    stdout,
    expected,
    monkeypatch,
):
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    ).to_dict()
    completed = text_isolation.subprocess.CompletedProcess(
        args=["worker"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )
    monkeypatch.setattr(
        text_isolation.subprocess,
        "run",
        lambda *args, **kwargs: completed,
    )

    with pytest.raises((RuntimeError, TypeError), match=expected):
        text_isolation.run_isolated_native_text(
            "validate",
            "module {}",
            policy,
            "worker-response.mlir",
        )


def test_isolated_parser_uses_a_fixed_utf8_versioned_protocol(monkeypatch):
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    ).to_dict()
    observed = {}

    def time_out(*args, **kwargs):
        observed.update(kwargs)
        raise text_isolation.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(text_isolation.subprocess, "run", time_out)
    result = text_isolation.run_isolated_native_text(
        "validate",
        'module attributes {func.note = "UTF-8: 测试"} {}',
        policy,
        "中文-source.mlir",
    )

    assert result["diagnostics"][0]["code"] == "AIR-PARSE-001"
    assert observed["encoding"] == "utf-8"
    assert observed["errors"] == "strict"
    assert '"protocol_version": "anchor-ir-text-worker/1.0.0"' in observed["input"]


def test_isolated_parser_rejects_non_utf8_python_strings_before_spawn():
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    ).to_dict()

    with pytest.raises(ValueError, match="valid Unicode encodable as UTF-8"):
        text_isolation.run_isolated_native_text(
            "validate",
            "module {}\udcff",
            policy,
            "surrogate.mlir",
        )


@pytest.mark.parametrize("context", [None, pytest.param(ir.context(), id="explicit")])
def test_public_text_validator_rejects_non_utf8_input_before_native_binding(context):
    with pytest.raises(ValueError, match="ir_text must be valid Unicode"):
        StructuredAnchorIRValidator().validate_text(
            "module {}\udcff",
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            phase=AnchorIRPhase.PRE_HOOK,
            context=context,
        )


@pytest.mark.parametrize("explicit_context", [False, True])
def test_non_utf8_mlir_location_filename_is_sanitized_in_structured_report(
    explicit_context,
):
    context = ir.context() if explicit_context else None
    report = _validate_text(
        'module { "smt.bad"() : () -> () loc("\\FF":2:3) }',
        context=context,
        source_name="location.mlir",
    )

    diagnostic = next(
        item for item in report.diagnostics if item.code == "AIR-LINALG-001"
    )
    assert diagnostic.location is not None
    assert diagnostic.location.file == "%FF"
    assert diagnostic.location.line == 2
    assert diagnostic.location.column == 3


def test_verifier_failure_has_independent_verify_diagnostic():
    report = _validate_text(
        "module { func.func @bad() -> i32 { func.return } }",
        source_name="verify.mlir",
    )

    assert not report.valid
    assert len(report.diagnostics) == 1
    diagnostic = report.diagnostics[0]
    assert diagnostic.code == "AIR-VERIFY-001"
    assert diagnostic.object_name == "builtin.module"
    assert diagnostic.operation_path == "builtin.module"
    assert diagnostic.hint
    assert diagnostic.location is not None
    assert diagnostic.location.file == "verify.mlir"


def test_sibling_ordinals_and_diagnostic_order_are_stable():
    text = """
module {
  "vendor.same"() : () -> ()
  "vendor.same"() : () -> ()
}
"""
    report = _validate_text(text, source_name="siblings.mlir")

    assert [item.operation_path for item in report.diagnostics] == [
        "builtin.module/region[0]/block[0]/vendor.same#0",
        "builtin.module/region[0]/block[0]/vendor.same#1",
    ]


def test_function_signature_and_block_argument_types_are_checked():
    report = _validate_text(
        """
module {
  func.func @bad(%arg0: !tt.ptr<f32>) {
    func.return
  }
}
""",
        source_name="signature.mlir",
    )

    assert [item.code for item in report.diagnostics] == [
        "AIR-LINALG-002",
        "AIR-LINALG-002",
    ]
    assert {item.object_path for item in report.diagnostics} == {
        "attribute[function_type].value.input[0]",
        "region[0].block[0].argument[0].type",
    }
    assert {item.operation_path for item in report.diagnostics} == {
        "builtin.module/region[0]/block[0]/func.func@bad#0"
    }
    assert all(item.object_kind.value == "type" for item in report.diagnostics)
    assert all(
        item.location is not None
        and item.location.file == "signature.mlir"
        and item.location.line == 3
        for item in report.diagnostics
    )


def test_nested_result_and_operand_element_types_are_checked():
    report = _validate_text(
        """
module {
  func.func @bad() -> tensor<4x!tt.ptr<f32>> {
    %0 = tensor.empty() : tensor<4x!tt.ptr<f32>>
    func.return %0 : tensor<4x!tt.ptr<f32>>
  }
}
""",
        source_name="nested-types.mlir",
    )

    assert [item.code for item in report.diagnostics] == [
        "AIR-LINALG-002",
        "AIR-LINALG-002",
        "AIR-LINALG-002",
    ]
    assert {item.object_path for item in report.diagnostics} == {
        "attribute[function_type].value.result[0].element_type",
        "result[0].type.element_type",
        "operand[0].type.element_type",
    }
    assert any(
        item.operation_path.endswith("/tensor.empty#0") for item in report.diagnostics
    )
    assert any(
        item.operation_path.endswith("/func.return#0") for item in report.diagnostics
    )


def test_memref_memory_space_attribute_is_checked_recursively():
    text = """
module {
  func.func @bad(%arg0: memref<4xf32, #gpu.address_space<workgroup>>) {
    func.return
  }
}
"""
    report = _validate_text(
        text,
        source_name="memref.mlir",
    )
    gpu_report = _validate_text(text, track=AnchorIRTrack.TRITON_GPU)

    assert [item.code for item in report.diagnostics] == [
        "AIR-COMMON-003",
        "AIR-COMMON-003",
    ]
    assert {item.object_path for item in report.diagnostics} == {
        "attribute[function_type].value.input[0].memory_space",
        "region[0].block[0].argument[0].type.memory_space",
    }
    assert gpu_report.valid
    assert gpu_report.diagnostics == ()


def test_module_array_dictionary_typeattr_and_encoding_are_recursive():
    report = _validate_text(
        """
module attributes {
  linalg.payload = [{nested = tensor<4xf32, #smt.encoding>}]
} {
}
""",
        source_name="containers.mlir",
    )

    assert {item.code for item in report.diagnostics} == {"AIR-LINALG-003"}
    diagnostic = next(
        item for item in report.diagnostics if item.code == "AIR-LINALG-003"
    )
    assert diagnostic.code == "AIR-LINALG-003"
    assert diagnostic.object_kind.value == "attribute"
    assert diagnostic.object_name == "#smt.encoding"
    assert diagnostic.operation_path == "builtin.module"
    assert diagnostic.object_path == (
        "attribute[linalg.payload].element[0].entry[nested].value.encoding"
    )
    assert diagnostic.location is not None
    assert diagnostic.location.file == "containers.mlir"


def test_operation_attribute_is_checked():
    report = _validate_text(
        """
module {
  func.func @bad() attributes {func.payload = #smt.bad} {
    func.return
  }
}
""",
        source_name="operation-attribute.mlir",
    )

    assert len(report.diagnostics) == 1
    diagnostic = report.diagnostics[0]
    assert diagnostic.code == "AIR-LINALG-003"
    assert diagnostic.operation_path.endswith("/func.func@bad#0")
    assert diagnostic.object_path == "attribute[func.payload]"


def test_opaque_type_and_attribute_preserve_original_dialect_namespace():
    report = _validate_text(
        """
module {
  func.func @bad(%arg0: !vendor<"type">)
      attributes {func.payload = #vendor<"attr">} {
    func.return
  }
}
""",
        source_name="opaque.mlir",
    )

    assert [item.code for item in report.diagnostics] == [
        "AIR-COMMON-003",
        "AIR-COMMON-002",
        "AIR-COMMON-002",
    ]
    assert {
        (item.object_kind.value, item.object_name, item.object_path)
        for item in report.diagnostics
    } == {
        ("attribute", '#vendor<"attr">', "attribute[func.payload]"),
        (
            "type",
            '!vendor<"type">',
            "attribute[function_type].value.input[0]",
        ),
        (
            "type",
            '!vendor<"type">',
            "region[0].block[0].argument[0].type",
        ),
    }


def test_generic_type_subelements_are_walked_beyond_builtin_containers():
    report = _validate_text(
        """
module {
  func.func @bad(%arg0: !llvm.struct<(!smt.bad)>) {
    func.return
  }
}
""",
        source_name="generic-type-walk.mlir",
    )

    assert [item.code for item in report.diagnostics] == [
        "AIR-COMMON-002",
        "AIR-LINALG-002",
        "AIR-COMMON-002",
        "AIR-LINALG-002",
    ]
    assert {
        item.object_path for item in report.diagnostics if item.code == "AIR-LINALG-002"
    } == {
        "attribute[function_type].value.input[0].type[0]",
        "region[0].block[0].argument[0].type.type[0]",
    }


def test_dialect_qualified_attribute_name_is_checked():
    text = "module attributes {smt.marker = true} { }"
    report = _validate_text(
        text,
        source_name="named-attribute.mlir",
    )
    gpu_report = _validate_text(text, track=AnchorIRTrack.TRITON_GPU)

    assert len(report.diagnostics) == 1
    diagnostic = report.diagnostics[0]
    assert diagnostic.code == "AIR-LINALG-003"
    assert diagnostic.object_name == "smt.marker"
    assert diagnostic.operation_path == "builtin.module"
    assert diagnostic.object_path == "attribute[smt.marker]"
    assert [item.code for item in gpu_report.diagnostics] == ["AIR-GPU-003"]


LINALG_GENERIC_IR = """
module {
  func.func @generic(%input: tensor<4xf32>, %output: tensor<4xf32>) -> tensor<4xf32> {
    %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%input : tensor<4xf32>) outs(%output : tensor<4xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
    } -> tensor<4xf32>
    func.return %result : tensor<4xf32>
  }
}
"""

GPU_CONFIG = (
    'attributes {"triton_gpu.num-warps" = 1 : i32, '
    '"triton_gpu.threads-per-warp" = 32 : i32, '
    '"triton_gpu.num-ctas" = 1 : i32}'
)
GPU_BLOCKED_1D = (
    "#triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], "
    "warpsPerCTA = [1], order = [0]}>"
)
GPU_BLOCKED_2D = (
    "#triton_gpu.blocked<{sizePerThread = [1, 1], "
    "threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>"
)


def test_linalg_semantic_rules_have_valid_baseline_and_do_not_apply_to_gpu():
    valid = _validate_text(LINALG_GENERIC_IR)
    gpu = _validate_text(LINALG_GENERIC_IR, track=AnchorIRTrack.TRITON_GPU)

    assert valid.valid
    assert not any(item.code.startswith("AIR-LINALG-01") for item in gpu.diagnostics)


def test_linalg_rejects_unfinished_conversion_cast_with_stable_code():
    report = _validate_text(
        'module { "builtin.unrealized_conversion_cast"() : () -> () }'
    )

    assert "AIR-LINALG-010" in [item.code for item in report.diagnostics]
    assert not any(
        item.code == "AIR-LINALG-010"
        for item in _validate_text(
            "module { func.func @ok() { func.return } }",
            track=AnchorIRTrack.LINALG,
        ).diagnostics
    )


def test_linalg_rejects_unranked_values_and_incomplete_generic_region():
    unranked = _validate_text(
        """
module {
  func.func @unranked(%input: tensor<*xf32>) {
    "linalg.fake"(%input) : (tensor<*xf32>) -> ()
    func.return
  }
}
"""
    )
    incomplete = _validate_text('module { "linalg.generic"() : () -> () }')

    assert "AIR-LINALG-011" in [item.code for item in unranked.diagnostics]
    assert "AIR-LINALG-012" in [item.code for item in incomplete.diagnostics]


def test_malformed_linalg_generic_block_returns_diagnostic_without_asserting():
    # Validation deliberately precedes MLIR verification.  An empty block must
    # therefore become a normal semantic diagnostic even with an explicit
    # Context (where a native signal cannot be hidden by the text worker).
    malformed = """
module {
  "linalg.generic"() <{indexing_maps = [], iterator_types = []}> ({
  ^bb0:
  }) : () -> ()
}
"""
    report = _validate_text(malformed, context=ir.context())

    assert not report.valid
    assert any(item.code == "AIR-LINALG-012" for item in report.diagnostics)


def test_linalg_accepts_legal_buffer_semantics_generic_yield():
    report = _validate_text(
        """
module {
  func.func @copy(%input: memref<4xf32>, %output: memref<4xf32>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    } ins(%input : memref<4xf32>) outs(%output : memref<4xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    }
    return
  }
}
"""
    )

    assert report.valid
    assert report.diagnostics == ()


def test_gpu_tensor_encoding_and_module_configuration_rules():
    encoded = _validate_text(
        """
module %s {
  tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, %s>
}
"""
        % (GPU_CONFIG, GPU_BLOCKED_1D),
        track=AnchorIRTrack.TRITON_GPU,
    )
    missing_encoding = _validate_text(
        """
module %s {
  tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
    )
    missing_config = _validate_text(
        """
module {
  tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, %s>
}
"""
        % GPU_BLOCKED_1D,
        track=AnchorIRTrack.TRITON_GPU,
    )
    linalg_view = _validate_text(
        """
module %s {
  tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.LINALG,
    )

    assert encoded.valid
    assert "AIR-GPU-010" in [item.code for item in missing_encoding.diagnostics]
    assert [item.code for item in missing_config.diagnostics] == [
        "AIR-GPU-011",
        "AIR-GPU-011",
        "AIR-GPU-011",
    ]
    assert {item.object_name for item in missing_config.diagnostics} == {
        "triton_gpu.num-warps",
        "triton_gpu.threads-per-warp",
        "triton_gpu.num-ctas",
    }
    assert "AIR-VERIFY-001" not in [item.code for item in missing_config.diagnostics]
    assert not any(item.code.startswith("AIR-GPU-") for item in linalg_view.diagnostics)


@pytest.mark.parametrize(
    "shaped_type",
    [
        "tensor<4xf32, affine_map<(d0) -> (d0)>>",
        "!tt.memdesc<4xf32>",
        '!tt.memdesc<4xf32, "not-a-layout">',
    ],
)
def test_gpu_rejects_missing_or_builtin_shaped_encoding(shaped_type):
    report = _validate_text(
        """
module %s {
  func.func @invalid(%%value: %s) {
    func.return
  }
}
"""
        % (GPU_CONFIG, shaped_type),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-010" in [item.code for item in report.diagnostics]


def test_gpu_memdesc_cannot_hide_forbidden_encoding():
    report = _validate_text(
        """
module %s {
  func.func @invalid(%%value: !tt.memdesc<4xf32, #smt.bad>) {
    func.return
  }
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
    )

    diagnostic = next(item for item in report.diagnostics if item.code == "AIR-GPU-003")
    assert diagnostic.object_path.endswith(".encoding")


def test_gpu_policy_preflight_blocks_unsafe_registered_op_verifier_paths():
    report = _validate_text(
        """
#layout = affine_map<(d0) -> (d0)>
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @invalid(%arg: tensor<4xf32, #layout>)
      -> tensor<2x2xf32, #layout> {
    %result = "tt.reshape"(%arg) <{allow_reorder = false}> :
      (tensor<4xf32, #layout>) -> tensor<2x2xf32, #layout>
    func.return %result : tensor<2x2xf32, #layout>
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
        source_name="unsafe-verifier-preflight.mlir",
    )

    assert not report.valid
    assert {item.code for item in report.diagnostics} == {
        "AIR-GPU-010",
        "AIR-GPU-016",
    }
    assert "AIR-VERIFY-001" not in [item.code for item in report.diagnostics]


def test_native_reduced_policy_cannot_fail_open_unsafe_dot_verifier_preflight():
    text = """
#layout = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @invalid(
      %%a: tensor<4x4xf32, #layout>,
      %%b: tensor<4x4xf32, #layout>,
      %%acc: tensor<4x4xf32, #layout>) -> tensor<4x4xf32, #layout> {
    %%result = tt.dot %%a, %%b, %%acc :
      tensor<4x4xf32, #layout> * tensor<4x4xf32, #layout>
      -> tensor<4x4xf32, #layout>
    func.return %%result : tensor<4x4xf32, #layout>
  }
}
""" % ()
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.TRITON_GPU,
        phase=AnchorIRPhase.PRE_HOOK,
    ).to_dict()
    policy["allowed_dialects"] = sorted(set(policy["allowed_dialects"]) | {"affine"})
    policy["enabled_invariants"] = []
    policy["semantic_diagnostics"] = {}

    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    raw = anchor.validate_anchor_ir_text(
        text,
        context,
        policy,
        "reduced-policy.mlir",
    )

    assert raw["valid"] is False
    assert raw["diagnostics"][0]["code"] == "AIR-VERIFY-001"
    assert "safety preflight" in raw["diagnostics"][0]["message"]


def test_native_reduced_policy_cannot_fail_open_malformed_reshape_preflight():
    text = """
module {
  func.func @invalid() -> tensor<2x2xf32> {
    %result = "tt.reshape"() <{allow_reorder = false}> :
      () -> tensor<2x2xf32>
    func.return %result : tensor<2x2xf32>
  }
}
"""
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.TRITON_GPU,
        phase=AnchorIRPhase.PRE_HOOK,
    ).to_dict()
    policy["enabled_invariants"] = []
    policy["semantic_diagnostics"] = {}

    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    raw = anchor.validate_anchor_ir_text(
        text,
        context,
        policy,
        "reduced-policy-reshape.mlir",
    )

    assert raw["valid"] is False
    assert raw["diagnostics"][0]["code"] == "AIR-VERIFY-001"
    assert "tt.reshape cardinality" in raw["diagnostics"][0]["message"]


def test_gpu_allowed_core_attribute_namespace_cannot_be_a_tensor_encoding():
    report = _validate_text(
        """
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @invalid(%arg: tensor<1xf32, #gpu.address_space<workgroup>>) {
    func.return
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert not report.valid
    assert "AIR-GPU-010" in [item.code for item in report.diagnostics]


def test_gpu_out_of_range_custom_integer_is_rejected_before_truncating_parser():
    report = _validate_text(
        """
#layout = #triton_gpu.blocked<{
  sizePerThread = [4294967297 : i64],
  threadsPerWarp = [1],
  warpsPerCTA = [1],
  order = [0]
}>
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 1 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @invalid(%arg: tensor<1xf32, #layout>) {
    func.return
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert not report.valid
    assert any(item.code == "AIR-PARSE-001" for item in report.diagnostics)


def test_gpu_custom_integer_preflight_ignores_line_comments():
    report = _validate_text(
        """
#layout = #triton_gpu.blocked<{
  sizePerThread = [1], // 4294967297 is documentation, not an attribute value.
  threadsPerWarp = [32],
  warpsPerCTA = [1],
  order = [0]
}>
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @valid(%arg: tensor<32xf32, #layout>) {
    func.return
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert report.valid


@pytest.mark.parametrize(
    "text",
    (
        """
#parent = #triton_gpu.blocked<{
  sizePerThread = [1, 1],
  threadsPerWarp = [1, 32],
  warpsPerCTA = [1, 1],
  order = [1, 0]
}>
#layout = #triton_gpu.slice<{
  dim = 0, // a legal custom-attribute dictionary comment
  parent = #parent
}>
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @valid(%arg: tensor<4xf32, #layout>) {
    func.return
  }
}
""",
        """
#layout = #triton_gpu.amd_mfma<{
  versionMajor = 1, // a legal custom-attribute dictionary comment
  versionMinor = 0,
  warpsPerCTA = [1, 1],
  instrShape = [16, 16],
  isTransposed = false
}>
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 64 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @valid(%arg: tensor<16x16xf32, #layout>) {
    func.return
  }
}
""",
    ),
)
def test_gpu_custom_attribute_preflight_accepts_legal_line_comments(text):
    report = _validate_text(text, track=AnchorIRTrack.TRITON_GPU)

    assert report.valid
    assert report.diagnostics == ()


def test_gpu_cta_preflight_comments_cannot_hide_an_invalid_layout():
    report = _validate_text(
        """
#layout = #triton_gpu.blocked<{
  sizePerThread = [1],
  threadsPerWarp = [32],
  warpsPerCTA = [1],
  order = [0], // this comment must not bypass the CTA safety preflight
  CTAsPerCGA = [1],
  CTASplitNum = [0],
  CTAOrder = [0]
}>
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @invalid(%arg: tensor<32xf32, #layout>) {
    func.return
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert [item.code for item in report.diagnostics] == ["AIR-PARSE-001"]
    assert "invalid explicit CTA layout" in report.diagnostics[0].message


@pytest.mark.parametrize(
    "value",
    (
        "2147483648 : i64",
        "2147483648 : i32",
        "4294967295 : i32",
        "18446744073709551615 : i65",
        "9223372036854775808 : i128",
    ),
)
def test_gpu_configuration_rejects_out_of_int32_range_without_crashing(value):
    report = _validate_text(
        """
module attributes {
  "triton_gpu.num-warps" = %s,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
}
"""
        % value,
        track=AnchorIRTrack.TRITON_GPU,
        source_name="wide-gpu-config.mlir",
    )

    assert not report.valid
    configuration = [item for item in report.diagnostics if item.code == "AIR-GPU-011"]
    assert len(configuration) == 1
    assert configuration[0].object_name == "triton_gpu.num-warps"
    assert configuration[0].object_path == "attribute[triton_gpu.num-warps]"
    assert "int32-range" in configuration[0].hint
    assert "AIR-VERIFY-001" not in [item.code for item in report.diagnostics]


def test_gpu_layout_product_comparison_cannot_wrap_to_a_false_match():
    report = _validate_text(
        """
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 65536 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @overflow(
      %arg: tensor<4xf32, #triton_gpu.blocked<{
        sizePerThread = [65536],
        threadsPerWarp = [65536],
        warpsPerCTA = [1],
        order = [0]
      }>>) {
    func.return
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
        source_name="gpu-product-overflow.mlir",
    )

    assert "AIR-GPU-011" not in [item.code for item in report.diagnostics]
    assert "AIR-GPU-014" in [item.code for item in report.diagnostics]


def test_gpu_checks_unused_signature_tensor_and_layout_configuration():
    unused_signature = _validate_text(
        """
module {
  func.func @unused(%input: tensor<16xf32>) {
    func.return
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )
    mismatched_layout = _validate_text(
        """
module attributes {
  "triton_gpu.num-warps" = 2 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, %s>
}
"""
        % GPU_BLOCKED_1D,
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-010" in [item.code for item in unused_signature.diagnostics]
    configuration = [
        item for item in mismatched_layout.diagnostics if item.code == "AIR-GPU-011"
    ]
    assert configuration
    assert all("Distributed Encoding" in item.hint for item in configuration)


def test_gpu_configuration_covers_encoding_only_in_function_signature():
    missing_configuration = _validate_text(
        """
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module {
  func.func @signature(%input: tensor<16xf32, #blocked>) {
    func.return
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )
    mismatched_configuration = _validate_text(
        """
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {
  "triton_gpu.num-warps" = 2 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @signature(%input: tensor<16xf32, #blocked>) {
    func.return
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )
    valid_configuration = _validate_text(
        """
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @signature(%input: tensor<16xf32, #blocked>) {
    func.return
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert [item.code for item in missing_configuration.diagnostics] == [
        "AIR-GPU-011",
        "AIR-GPU-011",
        "AIR-GPU-011",
    ]
    assert {item.object_name for item in missing_configuration.diagnostics} == {
        "triton_gpu.num-warps",
        "triton_gpu.threads-per-warp",
        "triton_gpu.num-ctas",
    }
    assert [item.code for item in mismatched_configuration.diagnostics] == [
        "AIR-GPU-011",
        "AIR-GPU-011",
    ]
    assert all(
        item.object_path.endswith(".encoding")
        for item in mismatched_configuration.diagnostics
    )
    assert valid_configuration.valid


def test_gpu_encoding_rank_and_dot_contract_rules():
    rank_mismatch = _validate_text(
        """
module %s {
  %%scalar = arith.constant 1 : i32
  %%tensor = tt.splat %%scalar : i32 -> tensor<4x4xi32, %s>
}
"""
        % (GPU_CONFIG, GPU_BLOCKED_1D),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid_dot = _validate_text(
        """
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module %s {
  func.func @dot(%%a: tensor<16x16xf32, #blocked>, %%b: tensor<16x16xf32, #blocked>, %%c: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %%result = tt.dot %%a, %%b, %%c : tensor<16x16xf32, #blocked> * tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked>
    func.return %%result : tensor<16x16xf32, #blocked>
  }
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
    )
    valid_dot = _validate_text(
        """
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#a = #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>
#b = #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>
module %s {
  func.func @dot(%%a: tensor<16x16xf32, #a>, %%b: tensor<16x16xf32, #b>, %%c: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %%result = tt.dot %%a, %%b, %%c : tensor<16x16xf32, #a> * tensor<16x16xf32, #b> -> tensor<16x16xf32, #blocked>
    func.return %%result : tensor<16x16xf32, #blocked>
  }
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-012" in [item.code for item in rank_mismatch.diagnostics]
    assert "AIR-GPU-013" in [item.code for item in invalid_dot.diagnostics]
    assert valid_dot.valid


def test_gpu_rejects_out_of_range_slice_dimension():
    report = _validate_text(
        """
#parent = #triton_gpu.blocked<{
  sizePerThread = [1, 1],
  threadsPerWarp = [16, 2],
  warpsPerCTA = [1, 1],
  order = [1, 0]
}>
#slice = #triton_gpu.slice<{dim = 2, parent = #parent}>
module %s {
  func.func @invalid(%%value: tensor<4xf32, #slice>) {
    func.return
  }
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-012" in [item.code for item in report.diagnostics]


def test_gpu_dot_rejects_incompatible_matrix_shapes():
    report = _validate_text(
        """
#parent = #triton_gpu.blocked<{
  sizePerThread = [1, 1],
  threadsPerWarp = [8, 4],
  warpsPerCTA = [1, 1],
  order = [1, 0]
}>
#a = #triton_gpu.dot_op<{opIdx = 0, parent = #parent}>
#b = #triton_gpu.dot_op<{opIdx = 1, parent = #parent}>
module %s {
  func.func @invalid(
      %%a: tensor<8x16xf32, #a>,
      %%b: tensor<8x16xf32, #b>,
      %%acc: tensor<16x16xf32, #parent>)
      -> tensor<16x16xf32, #parent> {
    %%result = tt.dot %%a, %%b, %%acc :
      tensor<8x16xf32, #a> * tensor<8x16xf32, #b>
      -> tensor<16x16xf32, #parent>
    func.return %%result : tensor<16x16xf32, #parent>
  }
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-013" in [item.code for item in report.diagnostics]


def test_gpu_dot_with_builtin_encoding_returns_report_instead_of_aborting():
    report = _validate_text(
        """
#layout = affine_map<(d0, d1) -> (d0, d1)>
module %s {
  func.func @invalid(
      %%a: tensor<4x4xf32, #layout>,
      %%b: tensor<4x4xf32, #layout>,
      %%acc: tensor<4x4xf32, #layout>)
      -> tensor<4x4xf32, #layout> {
    %%result = tt.dot %%a, %%b, %%acc :
      tensor<4x4xf32, #layout> * tensor<4x4xf32, #layout>
      -> tensor<4x4xf32, #layout>
    func.return %%result : tensor<4x4xf32, #layout>
  }
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert not report.valid
    assert "AIR-GPU-010" in [item.code for item in report.diagnostics]
    assert "AIR-GPU-013" in [item.code for item in report.diagnostics]


def test_gpu_dot_with_shared_a_and_dot_operand_b_returns_report_not_segfault():
    report = _validate_text(
        """
#blocked = #triton_gpu.blocked<{
  sizePerThread = [1, 1],
  threadsPerWarp = [8, 4],
  warpsPerCTA = [1, 1],
  order = [1, 0]
}>
#shared = #triton_gpu.shared<{
  vec = 1,
  perPhase = 1,
  maxPhase = 1,
  order = [1, 0]
}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @invalid(
      %a: !tt.memdesc<16x16xf16, #shared>,
      %b: tensor<16x16xf16, #dot1>,
      %c: tensor<16x16xf32, #blocked>)
      -> tensor<16x16xf32, #blocked> {
    %result = tt.dot %a, %b, %c :
      !tt.memdesc<16x16xf16, #shared> *
      tensor<16x16xf16, #dot1> ->
      tensor<16x16xf32, #blocked>
    func.return %result : tensor<16x16xf32, #blocked>
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
        source_name="dot-mixed-layout.mlir",
    )

    assert not report.valid
    assert "AIR-GPU-013" in [item.code for item in report.diagnostics]
    assert "AIR-VERIFY-001" not in [item.code for item in report.diagnostics]


def test_gpu_dot_accepts_hopper_shared_memdesc_operands():
    report = _validate_text(
        """
#shared = #triton_gpu.shared<{
  vec = 1,
  perPhase = 1,
  maxPhase = 1,
  order = [1, 0]
}>
#mma = #triton_gpu.nvidia_mma<{
  versionMajor = 3,
  versionMinor = 0,
  warpsPerCTA = [4, 1],
  instrShape = [16, 16, 16]
}>
module attributes {
  "triton_gpu.num-warps" = 4 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @hopper(
      %a: !tt.memdesc<16x16xf16, #shared>,
      %b: !tt.memdesc<16x16xf16, #shared>,
      %acc: tensor<16x16xf32, #mma>)
      -> tensor<16x16xf32, #mma> {
    %result = tt.dot %a, %b, %acc :
      !tt.memdesc<16x16xf16, #shared> *
      !tt.memdesc<16x16xf16, #shared>
      -> tensor<16x16xf32, #mma>
    func.return %result : tensor<16x16xf32, #mma>
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert report.valid
    assert report.diagnostics == ()


def test_gpu_configuration_checks_topology_behind_slice_encoding():
    text = """
#parent = #triton_gpu.blocked<{
  sizePerThread = [1, 1],
  threadsPerWarp = [16, 2],
  warpsPerCTA = [2, 1],
  order = [1, 0]
}>
#slice = #triton_gpu.slice<{dim = 0, parent = #parent}>
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  "vendor.value"() : () -> tensor<4xf32, #slice>
}
"""
    report = _validate_text(
        text,
        track=AnchorIRTrack.TRITON_GPU,
        phase=AnchorIRPhase.POST_HOOK,
        extension_dialects={"vendor"},
    )

    assert any(
        diagnostic.code == "AIR-GPU-011"
        and diagnostic.object_path == "result[0].type.encoding"
        for diagnostic in report.diagnostics
    )


def test_gpu_dot_requires_operand_indices_parent_and_accumulator_encoding():
    wrong_relationship = _validate_text(
        """
#result = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#other = #triton_gpu.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#a = #triton_gpu.dot_op<{opIdx = 1, parent = #other}>
#b = #triton_gpu.dot_op<{opIdx = 0, parent = #other}>
module %s {
  func.func @dot(%%lhs: tensor<16x16xf32, #a>, %%rhs: tensor<16x16xf32, #b>, %%acc: tensor<16x16xf32, #result>) -> tensor<16x16xf32, #result> {
    %%value = tt.dot %%lhs, %%rhs, %%acc : tensor<16x16xf32, #a> * tensor<16x16xf32, #b> -> tensor<16x16xf32, #result>
    func.return %%value : tensor<16x16xf32, #result>
  }
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-013" in [item.code for item in wrong_relationship.diagnostics]


def test_nested_type_and_attribute_diagnostics_are_deterministic():
    text = """
module attributes {
  linalg.payload = [{nested = tensor<4xf32, #smt.encoding>}]
} {
}
"""
    reports = [
        _validate_text(text, source_name="deterministic-objects.mlir")
        for _ in range(10)
    ]

    expected = reports[0].to_dict()
    assert all(report.to_dict() == expected for report in reports)


def test_unknown_type_and_attribute_have_distinct_codes():
    type_report = _validate_text(
        "module { func.func @bad(%arg0: !vendor.type) { func.return } }"
    )
    attribute_report = _validate_text(
        "module attributes {func.payload = #vendor.attr} { }"
    )

    assert {item.code for item in type_report.diagnostics} == {"AIR-COMMON-002"}
    assert [
        item.code
        for item in attribute_report.diagnostics
        if item.code == "AIR-COMMON-003"
    ] == ["AIR-COMMON-003"]


def test_type_policy_is_track_specific():
    text = "module { func.func @ok(%arg0: !tt.ptr<f32>) { func.return } }"
    linalg = _validate_text(text, track=AnchorIRTrack.LINALG)
    gpu = _validate_text(text, track=AnchorIRTrack.TRITON_GPU)

    assert {item.code for item in linalg.diagnostics} == {"AIR-LINALG-002"}
    assert gpu.valid
    assert gpu.diagnostics == ()


@pytest.mark.parametrize(
    "property_value, expected_code, expected_kind, expected_path",
    [
        (
            "#smt.marker",
            "AIR-LINALG-003",
            "attribute",
            "properties.entry[payload]",
        ),
        (
            "!smt.secret",
            "AIR-LINALG-002",
            "type",
            "properties.entry[payload].value",
        ),
        (
            "{smt.marker = true}",
            "AIR-LINALG-003",
            "attribute",
            "properties.entry[payload].entry[smt.marker]",
        ),
    ],
)
def test_extension_properties_cannot_hide_forbidden_objects(
    property_value,
    expected_code,
    expected_kind,
    expected_path,
):
    report = _validate_text(
        'module { "vendor.op"() <{payload = %s}> : () -> () }' % property_value,
        phase=AnchorIRPhase.POST_HOOK,
        extension_dialects={"vendor"},
    )

    assert any(
        diagnostic.code == expected_code
        and diagnostic.object_kind.value == expected_kind
        and diagnostic.object_path == expected_path
        for diagnostic in report.diagnostics
    )


def test_registered_native_property_collision_does_not_hide_property_value():
    """A same-named raw attr must not mask a registered native property."""

    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    text = """
module {
  func.func @f(%arg0: i32) -> i32 {
    %0 = "llvm.add"(%arg0, %arg0)
        <{overflowFlags = #llvm.overflow<nsw>}>
        {overflowFlags = 7 : i32} : (i32, i32) -> i32
    func.return %0 : i32
  }
}
"""
    # This deliberately reduced native-core policy makes the registered LLVM
    # property observable as forbidden while keeping its operation parseable.
    # The public policy loader never produces an allowed/forbidden overlap;
    # this is a white-box regression for the storage de-duplication helper.
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    ).to_dict()
    policy["allowed_dialects"] = sorted(set(policy["allowed_dialects"]) | {"llvm"})
    policy["forbidden_dialects"] = sorted(set(policy["forbidden_dialects"]) | {"llvm"})

    raw_report = anchor.validate_anchor_ir_text(
        text,
        context,
        policy,
        "registered-property-collision.mlir",
    )

    assert any(
        diagnostic["code"] == "AIR-LINALG-003"
        and diagnostic["object_name"] == "#llvm.overflow<nsw>"
        and diagnostic["object_path"] == "properties.entry[overflowFlags]"
        for diagnostic in raw_report["diagnostics"]
    )


def test_registered_inherent_property_is_not_reported_twice():
    report = _validate_text(
        """
module {
  "func.func"() <{
    function_type = (!smt.secret) -> (),
    sym_name = "f"
  }> ({
    "func.return"() : () -> ()
  }) : () -> ()
}
"""
    )

    forbidden_type = [
        diagnostic
        for diagnostic in report.diagnostics
        if diagnostic.code == "AIR-LINALG-002"
        and diagnostic.object_name == "!smt.secret"
    ]
    assert len(forbidden_type) == 1
    assert forbidden_type[0].object_path == "attribute[function_type].value.input[0]"


@pytest.mark.parametrize(
    "text, expected_path",
    [
        (
            "module attributes {func.payload = "
            "dense<0> : tensor<1xi32, #smt.encoding>} {}",
            "attribute[func.payload].type.encoding",
        ),
        (
            'module { "vendor.op"() <{payload = '
            "dense<0> : tensor<1xi32, #smt.encoding>}> : () -> () }",
            "properties.entry[payload].type.encoding",
        ),
    ],
)
def test_dense_typed_attributes_cannot_hide_forbidden_encoding(
    text,
    expected_path,
):
    report = _validate_text(
        text,
        phase=AnchorIRPhase.POST_HOOK,
        extension_dialects={"vendor"},
    )

    assert any(
        diagnostic.code == "AIR-LINALG-003"
        and diagnostic.object_kind.value == "attribute"
        and diagnostic.object_name == "#smt.encoding"
        and diagnostic.object_path == expected_path
        for diagnostic in report.diagnostics
    )


def test_dense_typed_attribute_gpu_encoding_requires_module_configuration():
    report = _validate_text(
        """
#blocked = #triton_gpu.blocked<{
  sizePerThread = [1],
  threadsPerWarp = [32],
  warpsPerCTA = [1],
  order = [0]
}>
module attributes {
  func.payload = dense<0.0> : tensor<1xf32, #blocked>
} {}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )

    configuration = [item for item in report.diagnostics if item.code == "AIR-GPU-011"]
    assert {item.object_name for item in configuration} == {
        "triton_gpu.num-warps",
        "triton_gpu.threads-per-warp",
        "triton_gpu.num-ctas",
    }


def test_operation_path_percent_encodes_quoted_symbol_delimiters():
    report = _validate_text(
        """
module {
  func.func @"kernel/part#1"() attributes {smt.marker = true} {
    func.return
  }
}
"""
    )

    diagnostic = next(
        item
        for item in report.diagnostics
        if item.code == "AIR-LINALG-003" and item.object_path == "attribute[smt.marker]"
    )
    assert "/func.func@kernel%2Fpart%231#0" in diagnostic.operation_path


def test_diagnostic_fields_are_bounded_without_losing_leaf_operation():
    symbol = "kernel" + "a" * 5000
    source_name = "source" + "b" * 2048
    report = _validate_text(
        'module { func.func @%s() { "vendor.bad"() : () -> () } }' % symbol,
        source_name=source_name,
    )

    diagnostic = next(
        item for item in report.diagnostics if item.code == "AIR-COMMON-001"
    )
    fields = (
        diagnostic.message,
        diagnostic.hint,
        diagnostic.object_name,
        diagnostic.operation_path,
        diagnostic.object_path,
        diagnostic.location.file,
    )
    assert all(len(field.encode("utf-8")) <= 1024 for field in fields)
    assert "vendor.bad#0" in diagnostic.operation_path


def test_captured_parser_detail_respects_public_diagnostic_field_bound():
    long_unregistered_operation = "x" * 5000
    report = _validate_text("module { %s }" % long_unregistered_operation)

    diagnostic = report.diagnostics[0]
    assert diagnostic.code == "AIR-PARSE-001"
    assert len(diagnostic.message.encode("utf-8")) <= 1024
    assert "...<truncated>..." in diagnostic.message


def test_gpu_configuration_detects_encoding_inside_extension_properties():
    report = _validate_text(
        """
#blocked = #triton_gpu.blocked<{
  sizePerThread = [1],
  threadsPerWarp = [32],
  warpsPerCTA = [1],
  order = [0]
}>
module {
  "vendor.op"() <{payload = tensor<4xf32, #blocked>}> : () -> ()
}
""",
        track=AnchorIRTrack.TRITON_GPU,
        phase=AnchorIRPhase.POST_HOOK,
        extension_dialects={"vendor"},
    )

    configuration = [item for item in report.diagnostics if item.code == "AIR-GPU-011"]
    assert {item.object_name for item in configuration} == {
        "triton_gpu.num-warps",
        "triton_gpu.threads-per-warp",
        "triton_gpu.num-ctas",
    }


@pytest.mark.parametrize(
    "layout",
    [
        (
            "#triton_gpu.blocked<{sizePerThread = [0], "
            "threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>"
        ),
        (
            "#triton_gpu.blocked<{sizePerThread = [1], "
            "threadsPerWarp = [32], warpsPerCTA = [1], order = [0], "
            "CTAsPerCGA = [1], CTASplitNum = [0], CTAOrder = [0]}>"
        ),
        ("#triton_gpu.shared<{vec = 0, perPhase = 0, maxPhase = 0, order = [0]}>"),
        (
            "#triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 2, "
            "order = [0], hasLeadingOffset = true}>"
        ),
    ],
)
def test_gpu_rejects_unsafe_layout_components_without_native_abort(layout):
    report = _validate_text(
        """
#layout = %s
module %s {
  func.func @invalid(%%arg: tensor<4xf32, #layout>) {
    func.return
  }
}
"""
        % (layout, GPU_CONFIG),
        track=AnchorIRTrack.TRITON_GPU,
    )

    expected_code = (
        "AIR-PARSE-001" if "CTAsPerCGA" in layout else "AIR-GPU-014"
    )
    assert expected_code in [item.code for item in report.diagnostics]
    assert "AIR-VERIFY-001" not in [item.code for item in report.diagnostics]


@pytest.mark.parametrize(
    "layout, shape",
    [
        (
            "#triton_gpu.nvidia_mma<{versionMajor = 999, versionMinor = 0, "
            "warpsPerCTA = [1, 1], instrShape = [16, 8]}>",
            "4x4",
        ),
        (
            "#triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, "
            "warpsPerCTA = [1], instrShape = [16, 8]}>",
            "4",
        ),
        (
            "#triton_gpu.amd_mfma<{versionMajor = 1, versionMinor = 0, "
            "warpsPerCTA = [1], instrShape = [16, 16], "
            "isTransposed = false}>",
            "4",
        ),
        (
            "#triton_gpu.amd_wmma<{warpsPerCTA = [1]}>",
            "4",
        ),
        (
            "#triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, "
            "warpsPerCTA = [4, 1, 1], instrShape = [16, 16, 16]}>",
            "2x4x4",
        ),
        (
            "#triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, "
            "warpsPerCTA = [4, 1], instrShape = []}>",
            "4x4",
        ),
    ],
)
def test_gpu_rejects_unsupported_mma_support_domains(layout, shape):
    report = _validate_text(
        """
#layout = %s
module attributes {
  "triton_gpu.num-warps" = 4 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @invalid(%%arg: tensor<%sxf32, #layout>) {
    func.return
  }
}
"""
        % (layout, shape),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-014" in [item.code for item in report.diagnostics]
    assert "AIR-VERIFY-001" not in [item.code for item in report.diagnostics]


@pytest.mark.parametrize(
    "layout, shape",
    [
        (
            "#triton_gpu.nvidia_mma<{versionMajor = 999, versionMinor = 0, "
            "warpsPerCTA = [1, 1], instrShape = [16, 8]}>",
            "4x4",
        ),
        (
            "#triton_gpu.amd_mfma<{versionMajor = 1, versionMinor = 0, "
            "warpsPerCTA = [1], instrShape = [16, 16], "
            "isTransposed = false}>",
            "4",
        ),
        (
            "#triton_gpu.amd_wmma<{warpsPerCTA = [1]}>",
            "4",
        ),
    ],
)
def test_native_text_validator_diagnoses_unsafe_mma_before_trait_getters(
    layout,
    shape,
):
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.TRITON_GPU,
        phase=AnchorIRPhase.PRE_HOOK,
    )
    text = """
#layout = %s
module attributes {
  "triton_gpu.num-warps" = 4 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @invalid(%%arg: tensor<%sxf32, #layout>) {
    func.return
  }
}
""" % (layout, shape)

    report = anchor.validate_anchor_ir_text(
        text,
        context,
        policy.to_dict(),
        "native-mma-support-domain.mlir",
    )

    assert report["valid"] is False
    assert "AIR-GPU-014" in [item["code"] for item in report["diagnostics"]]
    assert "AIR-VERIFY-001" not in [item["code"] for item in report["diagnostics"]]


@pytest.mark.parametrize(
    "element_type",
    [
        "complex<f32>",
        "vector<2xf32>",
        "index",
        "i7",
        "si32",
    ],
)
def test_gpu_rejects_unsupported_ranked_tensor_element_types(element_type):
    report = _validate_text(
        """
#layout = %s
module %s {
  func.func @invalid(%%arg: tensor<4x%s, #layout>) {
    func.return
  }
}
"""
        % (GPU_BLOCKED_1D, GPU_CONFIG, element_type),
        track=AnchorIRTrack.TRITON_GPU,
    )

    diagnostic = next(item for item in report.diagnostics if item.code == "AIR-GPU-015")
    assert diagnostic.object_path.endswith(".element_type")
    assert diagnostic.hint


def test_gpu_rejects_pointer_element_in_memdesc():
    report = _validate_text(
        """
#layout = #triton_gpu.shared<{
  vec = 1, perPhase = 1, maxPhase = 1, order = [0]
}>
module %s {
  func.func @invalid(%%arg: !tt.memdesc<4x!tt.ptr<f32>, #layout>) {
    func.return
  }
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-015" in [item.code for item in report.diagnostics]


def test_gpu_shared_cta_topology_must_match_module_configuration():
    report = _validate_text(
        """
#layout = #triton_gpu.shared<{
  vec = 1,
  perPhase = 1,
  maxPhase = 1,
  order = [0],
  CTAsPerCGA = [2],
  CTASplitNum = [1],
  CTAOrder = [0]
}>
module %s {
  func.func @invalid(%%arg: !tt.memdesc<4xf32, #layout>) {
    func.return
  }
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
    )

    topology = [item for item in report.diagnostics if item.code == "AIR-GPU-011"]
    assert len(topology) == 2
    assert all(item.object_path.endswith(".encoding") for item in topology)


@pytest.mark.parametrize(
    "shape, order",
    [
        ("3x16x16", "[1, 0]"),
        ("3x16", "[1, 0]"),
        ("3", "[0]"),
    ],
)
def test_gpu_accepts_pipeline_shared_memdesc_shapes(shape, order):
    report = _validate_text(
        """
#layout = #triton_gpu.shared<{
  vec = 1, perPhase = 1, maxPhase = 1, order = %s
}>
module %s {
  func.func @staged(%%arg: !tt.memdesc<%sxf16, #layout>) {
    func.return
  }
}
"""
        % (order, GPU_CONFIG, shape),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert report.valid
    assert report.diagnostics == ()


def test_gpu_accepts_non_power_of_two_amd_wmma_warp_topology():
    report = _validate_text(
        """
#layout = #triton_gpu.amd_wmma<{warpsPerCTA = [2, 3]}>
module attributes {
  "triton_gpu.num-warps" = 6 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  func.func @wmma(%arg: tensor<32x64xf32, #layout>) {
    func.return
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert report.valid
    assert report.diagnostics == ()


@pytest.mark.parametrize(
    "type_text",
    [
        "tensor<16x16xf32, #triton_gpu.shared<{"
        "vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>>",
        "!tt.memdesc<16x16xf32, #triton_gpu.blocked<{"
        "sizePerThread = [1, 1], threadsPerWarp = [8, 4], "
        "warpsPerCTA = [1, 1], order = [1, 0]}>>",
    ],
)
def test_gpu_rejects_core_encoding_on_wrong_shaped_type_kind(type_text):
    report = _validate_text(
        """
module %s {
  func.func @invalid(%%arg: %s) {
    func.return
  }
}
"""
        % (GPU_CONFIG, type_text),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-014" in [item.code for item in report.diagnostics]


@pytest.mark.parametrize("shape", ["3x4", "2097152"])
def test_gpu_rejects_signature_only_tensor_element_contract_violations(shape):
    report = _validate_text(
        """
#layout = %s
module %s {
  func.func @invalid(%%arg: tensor<%sxf32, #layout>) {
    func.return
  }
}
"""
        % (GPU_BLOCKED_1D, GPU_CONFIG, shape),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-014" in [item.code for item in report.diagnostics]


def test_gpu_slice_rejects_non_distributed_shared_parent():
    report = _validate_text(
        """
#parent = #triton_gpu.shared<{
  vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]
}>
#slice = #triton_gpu.slice<{dim = 0, parent = #parent}>
module %s {
  func.func @invalid(%%arg: tensor<4xf32, #slice>) {
    func.return
  }
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-014" in [item.code for item in report.diagnostics]
    assert "AIR-VERIFY-001" not in [item.code for item in report.diagnostics]


def test_gpu_slice_rejects_removing_a_multi_cta_dimension():
    report = _validate_text(
        """
#parent = #triton_gpu.blocked<{
  sizePerThread = [1, 1],
  threadsPerWarp = [1, 32],
  warpsPerCTA = [1, 1],
  order = [1, 0],
  CTAsPerCGA = [2, 1],
  CTASplitNum = [1, 1],
  CTAOrder = [1, 0]
}>
#slice = #triton_gpu.slice<{dim = 0, parent = #parent}>
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 2 : i32
} {
  func.func @invalid(%arg: tensor<4xf32, #slice>) {
    func.return
  }
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-014" in [item.code for item in report.diagnostics]
    assert "AIR-VERIFY-001" not in [item.code for item in report.diagnostics]


def test_gpu_rank_four_dot_is_rejected_before_backend_lowering():
    report = _validate_text(
        """
#blocked = #triton_gpu.blocked<{
  sizePerThread = [1, 1, 1, 1],
  threadsPerWarp = [1, 1, 8, 4],
  warpsPerCTA = [1, 1, 1, 1],
  order = [3, 2, 1, 0]
}>
#a = #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>
#b = #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>
module %s {
  func.func @invalid(
      %%a: tensor<2x2x8x16xf32, #a>,
      %%b: tensor<2x2x16x8xf32, #b>,
      %%acc: tensor<2x2x8x8xf32, #blocked>)
      -> tensor<2x2x8x8xf32, #blocked> {
    %%result = tt.dot %%a, %%b, %%acc :
      tensor<2x2x8x16xf32, #a> * tensor<2x2x16x8xf32, #b>
      -> tensor<2x2x8x8xf32, #blocked>
    func.return %%result : tensor<2x2x8x8xf32, #blocked>
  }
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert "AIR-GPU-013" in [item.code for item in report.diagnostics]
    assert "AIR-VERIFY-001" not in [item.code for item in report.diagnostics]


def test_gpu_inline_asm_zero_packed_element_is_diagnosed_before_modulo():
    report = _validate_text(
        """
#layout = %s
module %s {
  func.func @invalid(%%arg: tensor<4xf32, #layout>)
      -> tensor<4xf32, #layout> {
    %%result = "tt.elementwise_inline_asm"(%%arg) <{
      asm_string = "mov.b32 $0, $1;",
      constraints = "=r,r",
      pure = true,
      packed_element = 0 : i32
    }> : (tensor<4xf32, #layout>) -> tensor<4xf32, #layout>
    func.return %%result : tensor<4xf32, #layout>
  }
}
"""
        % (GPU_BLOCKED_1D, GPU_CONFIG),
        track=AnchorIRTrack.TRITON_GPU,
    )

    diagnostic = next(item for item in report.diagnostics if item.code == "AIR-GPU-016")
    assert diagnostic.object_name == "tt.elementwise_inline_asm"
    assert diagnostic.object_path == "attribute[packed_element]"
    assert "AIR-VERIFY-001" not in [item.code for item in report.diagnostics]


def test_gpu_reshape_requires_registered_layout_interface():
    report = _validate_text(
        """
#layout = #vendor.layout
module %s {
  func.func @invalid(%%arg: tensor<4xf32, #layout>)
      -> tensor<2x2xf32, #layout> {
    %%result = "tt.reshape"(%%arg) <{allow_reorder = false}> :
      (tensor<4xf32, #layout>) -> tensor<2x2xf32, #layout>
    func.return %%result : tensor<2x2xf32, #layout>
  }
}
"""
        % GPU_CONFIG,
        track=AnchorIRTrack.TRITON_GPU,
        phase=AnchorIRPhase.POST_HOOK,
        extension_dialects={"vendor"},
    )

    diagnostic = next(item for item in report.diagnostics if item.code == "AIR-GPU-016")
    assert diagnostic.object_name == "tt.reshape"
    assert diagnostic.object_path == "operand[0].type.encoding"
    assert "AIR-VERIFY-001" not in [item.code for item in report.diagnostics]


@pytest.mark.parametrize(
    "operation_text, result_type",
    [
        (
            """
%result = "tt.reduce"(%arg) ({
^bb0(%lhs: f32, %rhs: f32):
  %sum = arith.addf %lhs, %rhs : f32
  "tt.reduce.return"(%sum) : (f32) -> ()
}) {axis = __AXIS__ : i32} : (tensor<4xf32, #layout>) -> f32
""",
            "f32",
        ),
        (
            """
%result = "tt.scan"(%arg) ({
^bb0(%lhs: f32, %rhs: f32):
  %sum = arith.addf %lhs, %rhs : f32
  "tt.scan.return"(%sum) : (f32) -> ()
}) {axis = __AXIS__ : i32, reverse = false} :
  (tensor<4xf32, #layout>) -> tensor<4xf32, #layout>
""",
            "tensor<4xf32, #layout>",
        ),
    ],
)
def test_gpu_reduce_scan_axis_contract_has_positive_and_negative_cases(
    operation_text,
    result_type,
):
    template = """
#layout = %s
module %s {
  func.func @kernel(%%arg: tensor<4xf32, #layout>) -> %s {
    %s
    func.return %%result : %s
  }
}
"""
    valid = _validate_text(
        template
        % (
            GPU_BLOCKED_1D,
            GPU_CONFIG,
            result_type,
            operation_text.replace("__AXIS__", "0"),
            result_type,
        ),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid = _validate_text(
        template
        % (
            GPU_BLOCKED_1D,
            GPU_CONFIG,
            result_type,
            operation_text.replace("__AXIS__", "99"),
            result_type,
        ),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    diagnostic = next(
        item for item in invalid.diagnostics if item.code == "AIR-GPU-016"
    )
    assert diagnostic.object_path == "attribute[axis]"
    assert "AIR-VERIFY-001" not in [item.code for item in invalid.diagnostics]


def test_gpu_expand_dims_axis_contract_has_positive_and_negative_cases():
    template = """
#parent = #triton_gpu.blocked<{
  sizePerThread = [1, 1],
  threadsPerWarp = [1, 32],
  warpsPerCTA = [1, 1],
  order = [1, 0]
}>
#slice = #triton_gpu.slice<{dim = 0, parent = #parent}>
module %s {
  func.func @kernel(%%arg: tensor<4xf32, #slice>)
      -> tensor<1x4xf32, #parent> {
    %%result = "tt.expand_dims"(%%arg) {axis = %s : i32} :
      (tensor<4xf32, #slice>) -> tensor<1x4xf32, #parent>
    func.return %%result : tensor<1x4xf32, #parent>
  }
}
"""
    valid = _validate_text(
        template % (GPU_CONFIG, 0),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid = _validate_text(
        template % (GPU_CONFIG, 99),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    diagnostic = next(
        item for item in invalid.diagnostics if item.code == "AIR-GPU-016"
    )
    assert diagnostic.object_path == "attribute[axis]"
    assert "AIR-VERIFY-001" not in [item.code for item in invalid.diagnostics]


def test_gpu_reduce_requires_all_input_shapes_to_match():
    template = """
#layout = %s
module %s {
  func.func @kernel(
      %%a: tensor<4xf32, #layout>,
      %%b: tensor<%sxf32, #layout>) -> (f32, f32) {
    %%result:2 = "tt.reduce"(%%a, %%b) ({
    ^bb0(%%a0: f32, %%b0: f32, %%a1: f32, %%b1: f32):
      "tt.reduce.return"(%%a0, %%b0) : (f32, f32) -> ()
    }) {axis = 0 : i32} :
      (tensor<4xf32, #layout>, tensor<%sxf32, #layout>) -> (f32, f32)
    func.return %%result#0, %%result#1 : f32, f32
  }
}
"""
    valid = _validate_text(
        template % (GPU_BLOCKED_1D, GPU_CONFIG, 4, 4),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid = _validate_text(
        template % (GPU_BLOCKED_1D, GPU_CONFIG, 8, 8),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    assert "AIR-GPU-016" in [item.code for item in invalid.diagnostics]
    assert "AIR-VERIFY-001" not in [item.code for item in invalid.diagnostics]


def test_gpu_broadcast_only_expands_singleton_dimensions():
    template = """
#layout = %s
module %s {
  func.func @kernel(%%arg: tensor<%sxf32, #layout>)
      -> tensor<4xf32, #layout> {
    %%result = tt.broadcast %%arg :
      tensor<%sxf32, #layout> -> tensor<4xf32, #layout>
    func.return %%result : tensor<4xf32, #layout>
  }
}
"""
    valid = _validate_text(
        template % (GPU_BLOCKED_1D, GPU_CONFIG, 1, 1),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid = _validate_text(
        template % (GPU_BLOCKED_1D, GPU_CONFIG, 2, 2),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    diagnostic = next(
        item for item in invalid.diagnostics if item.code == "AIR-GPU-016"
    )
    assert diagnostic.object_name == "tt.broadcast"
    assert diagnostic.object_path == "result[0].type"


def test_gpu_cat_requires_rank_one_and_exact_concatenated_shape():
    rank_one_template = """
#layout = %s
module %s {
  func.func @kernel(
      %%lhs: tensor<4xf32, #layout>,
      %%rhs: tensor<4xf32, #layout>) -> tensor<%sxf32, #layout> {
    %%result = tt.cat %%lhs, %%rhs :
      tensor<4xf32, #layout> -> tensor<%sxf32, #layout>
    func.return %%result : tensor<%sxf32, #layout>
  }
}
"""
    valid = _validate_text(
        rank_one_template % (GPU_BLOCKED_1D, GPU_CONFIG, 8, 8, 8),
        track=AnchorIRTrack.TRITON_GPU,
    )
    wrong_size = _validate_text(
        rank_one_template % (GPU_BLOCKED_1D, GPU_CONFIG, 16, 16, 16),
        track=AnchorIRTrack.TRITON_GPU,
    )
    wrong_rank = _validate_text(
        """
#layout = %s
module %s {
  func.func @kernel(
      %%lhs: tensor<4x4xf32, #layout>,
      %%rhs: tensor<4x4xf32, #layout>) -> tensor<8x4xf32, #layout> {
    %%result = tt.cat %%lhs, %%rhs :
      tensor<4x4xf32, #layout> -> tensor<8x4xf32, #layout>
    func.return %%result : tensor<8x4xf32, #layout>
  }
}
"""
        % (GPU_BLOCKED_2D, GPU_CONFIG),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    assert all(
        "AIR-GPU-016" in [item.code for item in report.diagnostics]
        for report in (wrong_size, wrong_rank)
    )


def test_gpu_histogram_requires_one_dimensional_input_and_i32_result():
    valid = _validate_text(
        """
#layout = %s
module %s {
  func.func @kernel(%%arg: tensor<4xi32, #layout>)
      -> tensor<8xi32, #layout> {
    %%result = tt.histogram %%arg :
      tensor<4xi32, #layout> -> tensor<8xi32, #layout>
    func.return %%result : tensor<8xi32, #layout>
  }
}
"""
        % (GPU_BLOCKED_1D, GPU_CONFIG),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid = _validate_text(
        """
#input = %s
#output = %s
module %s {
  func.func @kernel(%%arg: tensor<4x4xi32, #input>)
      -> tensor<8xi32, #output> {
    %%result = tt.histogram %%arg :
      tensor<4x4xi32, #input> -> tensor<8xi32, #output>
    func.return %%result : tensor<8xi32, #output>
  }
}
"""
        % (GPU_BLOCKED_2D, GPU_BLOCKED_1D, GPU_CONFIG),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    assert "AIR-GPU-016" in [item.code for item in invalid.diagnostics]


def test_gpu_make_tensor_pointer_metadata_matches_rank_and_order_is_permutation():
    template = """
#layout = %s
module %s {
  func.func @kernel(
      %%base: !tt.ptr<f32>,
      %%s0: i64, %%s1: i64,
      %%st0: i64, %%st1: i64,
      %%o0: i32, %%o1: i32)
      -> !tt.ptr<tensor<32x32xf32, #layout>> {
    %%result = tt.make_tensor_ptr
      %%base, [%%s0, %%s1], [%%st0, %%st1], [%%o0, %%o1]
      {order = array<i32: %s>} :
      !tt.ptr<tensor<32x32xf32, #layout>>
    func.return %%result : !tt.ptr<tensor<32x32xf32, #layout>>
  }
}
"""
    valid = _validate_text(
        template % (GPU_BLOCKED_2D, GPU_CONFIG, "1, 0"),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid = _validate_text(
        template % (GPU_BLOCKED_2D, GPU_CONFIG, "0, 0"),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    diagnostic = next(
        item for item in invalid.diagnostics if item.code == "AIR-GPU-016"
    )
    assert diagnostic.object_name == "tt.make_tensor_ptr"
    assert diagnostic.object_path == "attribute[order]"


def test_gpu_advance_offset_count_matches_block_pointer_rank():
    template = """
#layout = %s
module %s {
  func.func @kernel(
      %%ptr: !tt.ptr<tensor<32x32xf32, #layout>>,
      %%o0: i32, %%o1: i32)
      -> !tt.ptr<tensor<32x32xf32, #layout>> {
    %%result = tt.advance %%ptr, [%s] :
      !tt.ptr<tensor<32x32xf32, #layout>>
    func.return %%result : !tt.ptr<tensor<32x32xf32, #layout>>
  }
}
"""
    valid = _validate_text(
        template % (GPU_BLOCKED_2D, GPU_CONFIG, "%o0, %o1"),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid = _validate_text(
        template % (GPU_BLOCKED_2D, GPU_CONFIG, "%o0"),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    assert "AIR-GPU-016" in [item.code for item in invalid.diagnostics]
    assert "AIR-VERIFY-001" not in [item.code for item in invalid.diagnostics]


@pytest.mark.parametrize(
    "boundary",
    ("2", "0, 0"),
    ids=("out-of-range", "duplicate"),
)
def test_gpu_block_pointer_load_rejects_invalid_boundary_dimensions(boundary):
    template = """
#layout = %s
module %s {
  func.func @kernel(%%ptr: !tt.ptr<tensor<32x32xf32, #layout>>)
      -> tensor<32x32xf32, #layout> {
    %%result = tt.load %%ptr {boundaryCheck = array<i32: %s>} :
      !tt.ptr<tensor<32x32xf32, #layout>>
    func.return %%result : tensor<32x32xf32, #layout>
  }
}
"""
    valid = _validate_text(
        template % (GPU_BLOCKED_2D, GPU_CONFIG, "0, 1"),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid = _validate_text(
        template % (GPU_BLOCKED_2D, GPU_CONFIG, boundary),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    assert "AIR-GPU-016" in [item.code for item in invalid.diagnostics]


def test_gpu_block_pointer_load_rejects_nan_padding_for_integer_elements():
    template = """
#layout = %s
module %s {
  func.func @kernel(%%ptr: !tt.ptr<tensor<32x32x%s, #layout>>)
      -> tensor<32x32x%s, #layout> {
    %%result = tt.load %%ptr {
      boundaryCheck = array<i32: 0>,
      padding = 2 : i32
    } : !tt.ptr<tensor<32x32x%s, #layout>>
    func.return %%result : tensor<32x32x%s, #layout>
  }
}
"""
    valid = _validate_text(
        template % (GPU_BLOCKED_2D, GPU_CONFIG, "f32", "f32", "f32", "f32"),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid = _validate_text(
        template % (GPU_BLOCKED_2D, GPU_CONFIG, "i32", "i32", "i32", "i32"),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    assert "AIR-GPU-016" in [item.code for item in invalid.diagnostics]


def test_gpu_block_pointer_store_checks_boundary_dimensions():
    template = """
#layout = %s
module %s {
  func.func @kernel(
      %%ptr: !tt.ptr<tensor<32x32xf32, #layout>>,
      %%value: tensor<32x32xf32, #layout>) {
    tt.store %%ptr, %%value {boundaryCheck = array<i32: %s>} :
      !tt.ptr<tensor<32x32xf32, #layout>>
    func.return
  }
}
"""
    valid = _validate_text(
        template % (GPU_BLOCKED_2D, GPU_CONFIG, "0"),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid = _validate_text(
        template % (GPU_BLOCKED_2D, GPU_CONFIG, "2"),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    assert "AIR-GPU-016" in [item.code for item in invalid.diagnostics]


def test_gpu_async_wait_requires_nonnegative_count():
    template = """
module %s {
  func.func @kernel() {
    %%token = "triton_gpu.async_wait"() <{num = %s : i32}> :
      () -> !triton_gpu.async.token
    func.return
  }
}
"""
    valid = _validate_text(
        template % (GPU_CONFIG, 0),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid = _validate_text(
        template % (GPU_CONFIG, -1),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    assert "AIR-GPU-016" in [item.code for item in invalid.diagnostics]


def test_gpu_dot_requires_nonnegative_imprecise_accumulator_limit():
    template = """
#blocked = #triton_gpu.blocked<{
  sizePerThread = [1, 1],
  threadsPerWarp = [8, 4],
  warpsPerCTA = [1, 1],
  order = [1, 0]
}>
#a = #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>
#b = #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>
module %s {
  func.func @kernel(
      %%a: tensor<16x16xf32, #a>,
      %%b: tensor<16x16xf32, #b>,
      %%c: tensor<16x16xf32, #blocked>)
      -> tensor<16x16xf32, #blocked> {
    %%result = tt.dot %%a, %%b, %%c
      {maxNumImpreciseAcc = %s : i32} :
      tensor<16x16xf32, #a> * tensor<16x16xf32, #b>
      -> tensor<16x16xf32, #blocked>
    func.return %%result : tensor<16x16xf32, #blocked>
  }
}
"""
    valid = _validate_text(
        template % (GPU_CONFIG, 0),
        track=AnchorIRTrack.TRITON_GPU,
    )
    invalid = _validate_text(
        template % (GPU_CONFIG, -1),
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert valid.valid
    diagnostic = next(
        item for item in invalid.diagnostics if item.code == "AIR-GPU-016"
    )
    assert diagnostic.object_name == "tt.dot"
    assert diagnostic.object_path == "attribute[maxNumImpreciseAcc]"


@pytest.mark.parametrize(
    "text",
    [
        "#bad = #triton_gpu.slice<{}>\nmodule {}",
        (
            "#bad = #triton_gpu.amd_mfma<{versionMajor = 1, "
            "versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [], "
            "isTransposed = false}>\nmodule {}"
        ),
    ],
)
def test_default_text_api_contains_pinned_triton_parser_aborts(text):
    report = _validate_text(text, track=AnchorIRTrack.TRITON_GPU)

    assert [item.code for item in report.diagnostics] == ["AIR-PARSE-001"]
    assert report.diagnostics[0].hint


@pytest.mark.parametrize(
    "text",
    [
        "#bad = #triton_gpu.slice<{}>\nmodule {}",
        (
            '#bad = #triton_gpu.slice<{dim = "not-an-integer", '
            "parent = #triton_gpu.blocked<{sizePerThread = [1], "
            "threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>}>\n"
            "module {}"
        ),
        (
            "#bad = #triton_gpu.amd_mfma<{versionMajor = 1, "
            "versionMinor = 0, warpsPerCTA = [1, 1], "
            "instrShape = [16, 16]}>\nmodule {}"
        ),
        (
            "#bad = #triton_gpu.amd_mfma<{versionMajor = 1, "
            "versionMinor = 0, warpsPerCTA = [1, 1], "
            "instrShape = [16, 16, 16], isTransposed = false}>\nmodule {}"
        ),
        (
            "#bad = #triton_gpu.blocked<{sizePerThread = [4294967297 : i64], "
            "threadsPerWarp = [1], warpsPerCTA = [1], order = [0]}>\n"
            "module {}"
        ),
        (
            "#bad = #triton_gpu.blocked<{sizePerThread = [1], "
            "threadsPerWarp = [32], warpsPerCTA = [1], order = [0], "
            "CTAsPerCGA = [1], CTASplitNum = [0], CTAOrder = [0]}>\n"
            "module {}"
        ),
    ],
)
def test_native_text_entry_preflights_unsafe_triton_gpu_custom_parsers(text):
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.TRITON_GPU,
        phase=AnchorIRPhase.PRE_HOOK,
    )

    report = anchor.validate_anchor_ir_text(
        text,
        context,
        policy.to_dict(),
        "native-custom-parser.mlir",
    )

    assert report["valid"] is False
    assert [item["code"] for item in report["diagnostics"]] == ["AIR-PARSE-001"]
    assert report["diagnostics"][0]["location"]["file"] == ("native-custom-parser.mlir")


def test_native_text_calls_restore_shared_explicit_context_state(tmp_path):
    def module_text(operation_count):
        operations = "\n".join(
            "    %%%d = arith.constant %d : i32" % (index, index)
            for index in range(operation_count)
        )
        return (
            "module {\n  func.func @f() {\n"
            + operations
            + "\n    func.return\n  }\n}\n"
        )

    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    ).to_dict()
    fast_text = module_text(1500)
    slow_text = module_text(9000)
    worker_errors = []

    def run_fast(context):
        try:
            anchor.validate_anchor_ir_text(
                fast_text,
                context,
                policy,
                "concurrent-fast.mlir",
            )
        except BaseException as error:
            worker_errors.append(error)

    def run_slow(context):
        try:
            anchor.normalize_anchor_ir_text(
                slow_text,
                context,
                policy,
                "concurrent-slow.mlir",
            )
        except BaseException as error:
            worker_errors.append(error)

    unknown_source = tmp_path / "unknown-dialect.mlir"
    unknown_source.write_text(
        'module { "vendor.unknown"() : () -> () }\n',
        encoding="utf-8",
    )
    for _ in range(8):
        context = ir.context()
        ir.load_dialects(context)
        anchor.load_dialects(context)
        fast = threading.Thread(target=run_fast, args=(context,))
        slow = threading.Thread(target=run_slow, args=(context,))
        fast.start()
        time.sleep(0.01)
        slow.start()
        fast.join()
        slow.join()

        assert not worker_errors
        with pytest.raises(RuntimeError, match="Parse MLIR file failed"):
            ir.parse_mlir_module(str(unknown_source), context)
