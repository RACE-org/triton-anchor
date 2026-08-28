"""Tests for versioned AnchorIR policy and stable diagnostic schema."""

import copy
import json
import pkgutil
from dataclasses import FrozenInstanceError

import pytest

import triton_anchor.anchor_ir_rules as anchor_ir_rules
from triton_anchor.anchor_ir import (
    LINALG_TRACK_ALLOWED,
    LINALG_TRACK_FORBIDDEN,
    TRITON_GPU_TRACK_ALLOWED,
    TRITON_GPU_TRACK_FORBIDDEN,
)
from triton_anchor.anchor_ir_rules import (
    ANCHOR_IR_SPEC_VERSION,
    SUPPORTED_SPEC_VERSIONS,
    AnchorIRPolicyError,
    resolve_policy,
    validate_policy_request,
)
from triton_anchor.anchor_ir_schema import (
    AnchorIRDiagnostic,
    AnchorIRLocation,
    AnchorIRObjectKind,
    AnchorIRPhase,
    AnchorIRTrack,
    AnchorIRValidationReport,
    format_anchor_ir_validation_report,
)

EXPECTED_LINALG_ALLOWED = {
    "affine",
    "arith",
    "aux",
    "bufferization",
    "cf",
    "func",
    "index",
    "linalg",
    "linalg_ext",
    "math",
    "math_ext",
    "memref",
    "scf",
    "tensor",
    "vector",
}
EXPECTED_LINALG_FORBIDDEN = {
    "smt",
    "tptr",
    "triton",
    "triton_gpu",
    "triton_nvidia_gpu",
    "tt",
    "tts",
}
EXPECTED_GPU_ALLOWED = {
    "arith",
    "cf",
    "func",
    "gpu",
    "math",
    "nvgpu",
    "scf",
    "triton_gpu",
    "tt",
}
EXPECTED_GPU_FORBIDDEN = {"smt", "tptr", "tts"}


def _resolve(track, phase=AnchorIRPhase.PRE_HOOK):
    return resolve_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=track,
        phase=phase,
    )


def test_supported_version_and_policy_identity():
    assert SUPPORTED_SPEC_VERSIONS == ("anchor-ir/1.0.0", "anchor-ir/1.1.0")
    policy = _resolve(AnchorIRTrack.LINALG)
    assert policy.spec_version == ANCHOR_IR_SPEC_VERSION
    assert policy.track == AnchorIRTrack.LINALG
    assert policy.phase == AnchorIRPhase.PRE_HOOK


def test_v1_1_appends_cf_without_rewriting_v1_0_contract():
    legacy = resolve_policy(
        spec_version="anchor-ir/1.0.0",
        track=AnchorIRTrack.TRITON_GPU,
        phase=AnchorIRPhase.PRE_HOOK,
    )
    current = _resolve(AnchorIRTrack.TRITON_GPU)

    assert "cf" not in legacy.allowed_dialects
    assert current.allowed_dialects == legacy.allowed_dialects | {"cf"}
    assert current.forbidden_dialects == legacy.forbidden_dialects
    legacy_document = legacy.to_dict()
    current_document = current.to_dict()
    legacy_document.pop("spec_version")
    current_document.pop("spec_version")
    current_document["allowed_dialects"].remove("cf")
    current_document["core_allowed_dialects"].remove("cf")
    assert current_document == legacy_document


def test_versioned_rule_source_matches_frozen_legacy_sets():
    linalg = _resolve(AnchorIRTrack.LINALG)
    gpu = _resolve(AnchorIRTrack.TRITON_GPU)

    assert set(linalg.allowed_dialects) == EXPECTED_LINALG_ALLOWED
    assert set(linalg.forbidden_dialects) == EXPECTED_LINALG_FORBIDDEN
    assert set(gpu.allowed_dialects) == EXPECTED_GPU_ALLOWED
    assert set(gpu.forbidden_dialects) == EXPECTED_GPU_FORBIDDEN

    assert LINALG_TRACK_ALLOWED == EXPECTED_LINALG_ALLOWED
    assert LINALG_TRACK_FORBIDDEN == EXPECTED_LINALG_FORBIDDEN
    assert TRITON_GPU_TRACK_ALLOWED == EXPECTED_GPU_ALLOWED
    assert TRITON_GPU_TRACK_FORBIDDEN == EXPECTED_GPU_FORBIDDEN


@pytest.mark.parametrize("spec_version", SUPPORTED_SPEC_VERSIONS)
def test_linalg_forbids_the_registered_nvidia_dialect_namespace(spec_version):
    policy = resolve_policy(
        spec_version=spec_version,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.POST_HOOK,
    )

    assert "triton_nvidia_gpu" in policy.forbidden_dialects
    assert "nvidia_gpu" not in policy.forbidden_dialects


def test_policy_is_immutable_and_phase_is_explicit():
    policy = _resolve(AnchorIRTrack.TRITON_GPU, AnchorIRPhase.POST_HOOK)
    assert policy.phase == AnchorIRPhase.POST_HOOK
    with pytest.raises((AttributeError, FrozenInstanceError)):
        policy.track = AnchorIRTrack.LINALG
    with pytest.raises(AttributeError):
        policy.allowed_dialects.add("custom")
    with pytest.raises(TypeError):
        policy.semantic_diagnostics["gpu.tensor_encoding"] = None


def test_policy_core_and_extension_dialect_sets_are_explicit():
    policy = _resolve(AnchorIRTrack.TRITON_GPU, AnchorIRPhase.POST_HOOK)
    assert policy.extension_dialects == frozenset()
    assert "gpu" in policy.allowed_dialects
    assert "triton_gpu" in policy.allowed_dialects


def test_policy_exposes_track_scoped_semantic_invariants_and_diagnostics():
    linalg = _resolve(AnchorIRTrack.LINALG)
    gpu = _resolve(AnchorIRTrack.TRITON_GPU)

    assert set(linalg.enabled_invariants) == {
        "linalg.no_unrealized_conversion_cast",
        "linalg.ranked_shaped_values",
        "linalg.generic_region_contract",
    }
    assert set(gpu.enabled_invariants) == {
        "gpu.tensor_encoding",
        "gpu.module_configuration",
        "gpu.encoding_rank",
        "gpu.encoding_components",
        "gpu.shaped_element_type",
        "gpu.operation_contract",
        "gpu.dot_encoding_contract",
    }
    assert (
        linalg.semantic_diagnostics["linalg.generic_region_contract"].code
        == "AIR-LINALG-012"
    )
    assert gpu.semantic_diagnostics["gpu.dot_encoding_contract"].code == "AIR-GPU-013"
    assert gpu.semantic_diagnostics["gpu.encoding_components"].code == "AIR-GPU-014"
    assert gpu.semantic_diagnostics["gpu.shaped_element_type"].code == "AIR-GPU-015"
    assert gpu.semantic_diagnostics["gpu.operation_contract"].code == "AIR-GPU-016"
    assert set(linalg.enabled_invariants) == set(linalg.semantic_diagnostics)
    assert set(gpu.enabled_invariants) == set(gpu.semantic_diagnostics)


@pytest.mark.parametrize(
    "spec_version, track, phase, expected_code, expected_name",
    [
        (None, "linalg", "pre_hook", "AIR-REQUEST-001", "spec_version"),
        (
            "anchor-ir/9.9.9",
            "linalg",
            "pre_hook",
            "AIR-REQUEST-002",
            "anchor-ir/9.9.9",
        ),
        (
            ANCHOR_IR_SPEC_VERSION,
            None,
            "pre_hook",
            "AIR-REQUEST-003",
            "track",
        ),
        (
            ANCHOR_IR_SPEC_VERSION,
            "invalid",
            "pre_hook",
            "AIR-REQUEST-004",
            "invalid",
        ),
        (
            ANCHOR_IR_SPEC_VERSION,
            "linalg",
            None,
            "AIR-REQUEST-005",
            "phase",
        ),
        (
            ANCHOR_IR_SPEC_VERSION,
            "linalg",
            "invalid",
            "AIR-REQUEST-006",
            "invalid",
        ),
    ],
)
def test_each_basic_request_failure_has_stable_code_and_hint(
    spec_version, track, phase, expected_code, expected_name
):
    report = validate_policy_request(
        spec_version=spec_version,
        track=track,
        phase=phase,
    )

    assert not report.valid
    assert len(report.diagnostics) == 1
    diagnostic = report.diagnostics[0]
    assert diagnostic.code == expected_code
    assert diagnostic.object_kind == AnchorIRObjectKind.REQUEST
    assert diagnostic.object_name == expected_name
    assert diagnostic.message
    assert diagnostic.hint


def test_multiple_request_diagnostics_are_deterministic():
    reports = [
        validate_policy_request(spec_version=None, track=None, phase=None)
        for _ in range(10)
    ]
    expected = reports[0].to_dict()

    assert [item.code for item in reports[0].diagnostics] == [
        "AIR-REQUEST-001",
        "AIR-REQUEST-003",
        "AIR-REQUEST-005",
    ]
    assert all(report.to_dict() == expected for report in reports)


def test_unsupported_version_resolve_raises_structured_report():
    with pytest.raises(AnchorIRPolicyError) as caught:
        resolve_policy(
            spec_version="anchor-ir/9.9.9",
            track=AnchorIRTrack.LINALG,
            phase=AnchorIRPhase.PRE_HOOK,
        )

    assert caught.value.report.diagnostics[0].code == "AIR-REQUEST-002"
    assert caught.value.report.diagnostics[0].hint


def test_valid_request_has_no_diagnostics():
    report = validate_policy_request(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track="triton_gpu",
        phase="post_hook",
    )
    assert report.valid
    assert report.track == AnchorIRTrack.TRITON_GPU
    assert report.phase == AnchorIRPhase.POST_HOOK
    assert report.diagnostics == ()


def test_full_diagnostic_round_trip_preserves_location_and_path():
    diagnostic = AnchorIRDiagnostic(
        code="AIR-COMMON-001",
        severity="error",
        message="Unknown operation dialect 'bad'",
        hint="Lower the operation or declare an allowed post-hook extension.",
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        object_kind=AnchorIRObjectKind.OPERATION,
        object_name="bad.op",
        operation_path="builtin.module/func.func@kernel/bad.op#0",
        object_path="result[0].type.element_type",
        location=AnchorIRLocation(file="kernel.mlir", line=7, column=3),
    )
    report = AnchorIRValidationReport.build(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        diagnostics=[diagnostic],
    )

    encoded = report.to_dict()
    decoded = AnchorIRValidationReport.from_dict(encoded)

    assert decoded == report
    assert encoded["valid"] is False
    assert encoded["diagnostics"][0]["operation_path"].endswith("bad.op#0")
    assert encoded["diagnostics"][0]["object_path"] == "result[0].type.element_type"
    assert encoded["diagnostics"][0]["location"] == {
        "file": "kernel.mlir",
        "line": 7,
        "column": 3,
    }


def test_report_sorts_diagnostics_independent_of_input_order():
    def diagnostic(code, path):
        return AnchorIRDiagnostic(
            code=code,
            severity="error",
            message="failure " + code,
            hint="fix " + code,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            phase=AnchorIRPhase.PRE_HOOK,
            object_kind=AnchorIRObjectKind.OPERATION,
            object_name="bad.op",
            operation_path=path,
        )

    first = diagnostic("AIR-COMMON-002", "module/func@a/bad.op#0")
    second = diagnostic("AIR-COMMON-001", "module/func@b/bad.op#0")
    forward = AnchorIRValidationReport.build(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        diagnostics=[first, second],
    )
    reverse = AnchorIRValidationReport.build(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        diagnostics=[second, first],
    )

    assert forward == reverse
    assert [item.operation_path for item in forward.diagnostics] == [
        "module/func@a/bad.op#0",
        "module/func@b/bad.op#0",
    ]


def test_report_rejects_mismatched_identity_and_malformed_valid_field():
    diagnostic = AnchorIRDiagnostic(
        code="AIR-COMMON-001",
        severity="error",
        message="Unknown operation dialect 'bad'",
        hint="Lower the operation.",
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        object_kind=AnchorIRObjectKind.OPERATION,
        object_name="bad.op",
    )

    with pytest.raises(ValueError, match="identity"):
        AnchorIRValidationReport.build(
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.TRITON_GPU,
            phase=AnchorIRPhase.PRE_HOOK,
            diagnostics=[diagnostic],
        )

    encoded = AnchorIRValidationReport.build(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        diagnostics=[diagnostic],
    ).to_dict()
    encoded["valid"] = "false"
    with pytest.raises(TypeError, match="boolean"):
        AnchorIRValidationReport.from_dict(encoded)

    encoded["valid"] = False
    del encoded["diagnostics"][0]["hint"]
    with pytest.raises(ValueError, match="diagnostic dictionary"):
        AnchorIRValidationReport.from_dict(encoded)


def test_report_schema_rejects_unknown_fields_instead_of_silently_dropping_them():
    diagnostic = AnchorIRDiagnostic(
        code="AIR-COMMON-001",
        severity="error",
        message="Unknown operation dialect 'bad'",
        hint="Lower the operation.",
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        object_kind=AnchorIRObjectKind.OPERATION,
        object_name="bad.op",
        location=AnchorIRLocation(file="kernel.mlir", line=1, column=1),
    )
    encoded = AnchorIRValidationReport.build(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        diagnostics=[diagnostic],
    ).to_dict()

    report_extra = copy.deepcopy(encoded)
    report_extra["future_field"] = "unexpected"
    with pytest.raises(ValueError, match="report dictionary must contain exactly"):
        AnchorIRValidationReport.from_dict(report_extra)

    diagnostic_extra = copy.deepcopy(encoded)
    diagnostic_extra["diagnostics"][0]["future_field"] = "unexpected"
    with pytest.raises(
        ValueError,
        match="diagnostic dictionary must contain exactly",
    ):
        AnchorIRValidationReport.from_dict(diagnostic_extra)

    location_extra = copy.deepcopy(encoded)
    location_extra["diagnostics"][0]["location"]["future_field"] = "unexpected"
    with pytest.raises(ValueError, match="location dictionary must contain exactly"):
        AnchorIRValidationReport.from_dict(location_extra)


def test_rule_document_schema_rejects_ambiguous_or_corrupt_policy(
    monkeypatch,
):
    raw = pkgutil.get_data("triton_anchor", "spec/anchor-ir-1.1.0.json")
    assert raw is not None
    baseline = json.loads(raw.decode("utf-8"))

    cases = []

    not_an_array = copy.deepcopy(baseline)
    not_an_array["tracks"]["linalg"]["allowed_dialects"] = "arith"
    cases.append((not_an_array, "JSON array"))

    unexpected_top_level = copy.deepcopy(baseline)
    unexpected_top_level["allow_everything"] = True
    cases.append((unexpected_top_level, "unexpected: allow_everything"))

    unexpected_track_field = copy.deepcopy(baseline)
    unexpected_track_field["tracks"]["linalg"]["allowed_ops_typo"] = ["tt.load"]
    cases.append((unexpected_track_field, "unexpected: allowed_ops_typo"))

    unexpected_template_field = copy.deepcopy(baseline)
    unexpected_template_field["unknown_dialect_diagnostic"]["severity"] = "warning"
    cases.append((unexpected_template_field, "unexpected: severity"))

    duplicate_namespace = copy.deepcopy(baseline)
    duplicate_namespace["tracks"]["linalg"]["allowed_dialects"].append("arith")
    cases.append((duplicate_namespace, "duplicates"))

    invalid_namespace = copy.deepcopy(baseline)
    invalid_namespace["tracks"]["linalg"]["allowed_dialects"].append("vendor.bad")
    cases.append((invalid_namespace, "invalid name"))

    missing_semantic_diagnostic = copy.deepcopy(baseline)
    missing_semantic_diagnostic["tracks"]["linalg"]["semantic_diagnostics"].pop(
        "linalg.generic_region_contract"
    )
    cases.append((missing_semantic_diagnostic, "exactly one diagnostic"))

    duplicate_code = copy.deepcopy(baseline)
    duplicate_code["tracks"]["linalg"]["forbidden_dialect_diagnostic"]["code"] = (
        "AIR-COMMON-001"
    )
    cases.append((duplicate_code, "globally unique"))

    unsupported_placeholder = copy.deepcopy(baseline)
    unsupported_placeholder["unknown_dialect_diagnostic"]["message"] = (
        "Unknown {unsupported}"
    )
    cases.append((unsupported_placeholder, "unsupported placeholder"))

    unsupported_format_spec = copy.deepcopy(baseline)
    unsupported_format_spec["unknown_dialect_diagnostic"]["message"] = (
        "Unknown {dialect!r}"
    )
    cases.append((unsupported_format_spec, "unsupported placeholder"))

    lone_surrogate = copy.deepcopy(baseline)
    lone_surrogate["unknown_dialect_diagnostic"]["message"] = "bad\ud800"
    cases.append((lone_surrogate, "valid UTF-8"))

    unimplemented_invariant = copy.deepcopy(baseline)
    linalg_rules = unimplemented_invariant["tracks"]["linalg"]
    old_name = "linalg.generic_region_contract"
    new_name = "linalg.not_implemented"
    linalg_rules["enabled_invariants"][
        linalg_rules["enabled_invariants"].index(old_name)
    ] = new_name
    linalg_rules["semantic_diagnostics"][new_name] = linalg_rules[
        "semantic_diagnostics"
    ].pop(old_name)
    cases.append((unimplemented_invariant, "not implemented"))

    for document, expected in cases:
        encoded = json.dumps(document).encode("utf-8")
        monkeypatch.setattr(
            anchor_ir_rules.pkgutil,
            "get_data",
            lambda _package, _resource, encoded=encoded: encoded,
        )
        with pytest.raises(RuntimeError, match=expected):
            anchor_ir_rules._load_rule_document(ANCHOR_IR_SPEC_VERSION)


def test_rule_document_rejects_duplicate_json_object_keys(monkeypatch):
    raw = pkgutil.get_data("triton_anchor", "spec/anchor-ir-1.1.0.json")
    assert raw is not None
    duplicated = raw.decode("utf-8").replace(
        '"schema_version": 1,',
        '"schema_version": 1, "schema_version": 1,',
        1,
    )
    monkeypatch.setattr(
        anchor_ir_rules.pkgutil,
        "get_data",
        lambda _package, _resource: duplicated.encode("utf-8"),
    )

    with pytest.raises(RuntimeError, match="duplicate JSON key 'schema_version'"):
        anchor_ir_rules._load_rule_document(ANCHOR_IR_SPEC_VERSION)


@pytest.mark.parametrize(
    "track, template_name, expected_code",
    [
        (
            AnchorIRTrack.LINALG,
            "forbidden_dialect_diagnostic",
            "AIR-LINALG-001",
        ),
        (
            AnchorIRTrack.TRITON_GPU,
            "forbidden_dialect_diagnostic",
            "AIR-GPU-001",
        ),
        (
            AnchorIRTrack.LINALG,
            "unknown_dialect_diagnostic",
            "AIR-COMMON-001",
        ),
        (
            AnchorIRTrack.TRITON_GPU,
            "unknown_dialect_diagnostic",
            "AIR-COMMON-001",
        ),
        (
            AnchorIRTrack.LINALG,
            "forbidden_type_diagnostic",
            "AIR-LINALG-002",
        ),
        (
            AnchorIRTrack.TRITON_GPU,
            "forbidden_type_diagnostic",
            "AIR-GPU-002",
        ),
        (
            AnchorIRTrack.LINALG,
            "unknown_type_diagnostic",
            "AIR-COMMON-002",
        ),
        (
            AnchorIRTrack.LINALG,
            "forbidden_attribute_diagnostic",
            "AIR-LINALG-003",
        ),
        (
            AnchorIRTrack.TRITON_GPU,
            "forbidden_attribute_diagnostic",
            "AIR-GPU-003",
        ),
        (
            AnchorIRTrack.TRITON_GPU,
            "unknown_attribute_diagnostic",
            "AIR-COMMON-003",
        ),
    ],
)
def test_policy_exposes_versioned_dialect_diagnostic_templates(
    track, template_name, expected_code
):
    template = getattr(_resolve(track), template_name)
    message, hint = template.render(
        dialect="vendor",
        operation="vendor.op",
        object_name="!vendor.type",
    )

    assert template.code == expected_code
    assert message
    assert "vendor.op" in hint or "!vendor.type" in hint


def test_diagnostic_schema_rejects_surrogates_and_text_renderer_escapes_controls():
    with pytest.raises(ValueError, match="UTF-8 encodable"):
        AnchorIRLocation(file="bad\udcff")

    diagnostic = AnchorIRDiagnostic(
        code="AIR-TEST-001",
        severity="error",
        message="bad\nmessage\x1b[31m",
        hint="fix\tthis",
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        object_kind=AnchorIRObjectKind.OPERATION,
        object_name="evil\\name",
        operation_path="builtin.module\n/op",
        location=AnchorIRLocation(file="source\r\n.mlir", line=1, column=2),
    )
    report = AnchorIRValidationReport.build(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        diagnostics=(diagnostic,),
    )
    rendered = format_anchor_ir_validation_report(report)
    assert "bad\\nmessage\\x1B[31m" in rendered
    assert "fix\\tthis" in rendered
    assert "source\\r\\n.mlir:1:2" in rendered
    assert "builtin.module\\n/op" in rendered
    assert "evil\\\\name" in rendered
    assert "\x1b" not in rendered
