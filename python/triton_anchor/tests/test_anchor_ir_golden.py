"""Acceptance tests for ordered AnchorIR Golden Stage comparison."""

import copy
import hashlib
import json

import pytest

from triton_anchor import (
    ANCHOR_IR_GOLDEN_MANIFEST_VERSION,
    ANCHOR_IR_NORMALIZATION_VERSION,
    ANCHOR_IR_SPEC_VERSION,
    AnchorIRGoldenBuilder,
    AnchorIRGoldenError,
    AnchorIRGoldenManifest,
    AnchorIRGoldenValidationError,
    AnchorIRNormalizer,
    AnchorIRStageId,
    AnchorIRTrack,
    compare_anchor_ir_golden,
)
from triton_anchor import anchor_ir_golden

BASE_IR = """
module {
  func.func @kernel(%arg: i32) -> i32 {
    %result = arith.addi %arg, %arg : i32
    func.return %result : i32
  }
}
"""

PASS_CHANGED_IR = BASE_IR.replace("arith.addi", "arith.subi")

HOOK_CHANGED_IR = BASE_IR.replace(
    "func.func @kernel(%arg: i32) -> i32 {",
    'func.func @kernel(%arg: i32) -> i32 attributes {func.note = "hook-v2"} {',
)

PASS_STAGE = AnchorIRStageId.after_pass("canonicalize")
HOOK_STAGE = AnchorIRStageId.after_hook("vendor")


def _manifest(
    *,
    adapter_ir=BASE_IR,
    pass_ir=BASE_IR,
    hook_ir=BASE_IR,
    boundary_ir=BASE_IR,
    case_id="linalg/add",
):
    builder = AnchorIRGoldenBuilder(
        case_id=case_id,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
        track=AnchorIRTrack.LINALG,
    )
    builder.add_text(AnchorIRStageId.adapter_output(), adapter_ir)
    builder.add_text(PASS_STAGE, pass_ir)
    builder.add_text(HOOK_STAGE, hook_ir)
    builder.add_text(
        AnchorIRStageId.post_hook_boundary(),
        boundary_ir,
    )
    return builder.build()


def test_stable_stage_ids_have_fixed_identity_phase_and_order_group():
    stages = (
        AnchorIRStageId.adapter_output(),
        AnchorIRStageId.after_pass("canonicalize-cse"),
        AnchorIRStageId.after_hook("vendor.backend"),
        AnchorIRStageId.post_hook_boundary(),
    )

    assert [stage.value for stage in stages] == [
        "adapter.output",
        "pass.canonicalize-cse.after",
        "hook.vendor.backend.after",
        "boundary.post_hook",
    ]
    assert [stage.phase.value for stage in stages] == [
        "pre_hook",
        "pre_hook",
        "post_hook",
        "post_hook",
    ]
    assert [stage.order_group for stage in stages] == [0, 1, 2, 3]

    for invalid_stage_id in (
        "parser",
        "normalize",
        "hook.vendor.before",
        "pass.after",
        "pass...after",
    ):
        with pytest.raises(AnchorIRGoldenError, match="invalid AnchorIR Stage ID"):
            AnchorIRStageId(invalid_stage_id)

    with pytest.raises(AnchorIRGoldenError, match="invalid AnchorIR Stage ID"):
        AnchorIRStageId.after_pass("../unstable")


def test_manifest_is_versioned_self_contained_and_json_round_trips():
    manifest = _manifest()
    encoded = manifest.to_json()
    decoded = AnchorIRGoldenManifest.from_json(encoded)

    assert decoded == manifest
    assert decoded.to_json() == encoded
    assert encoded.endswith("\n")
    assert not encoded.endswith("\n\n")
    assert json.loads(encoded)["manifest_version"] == (
        ANCHOR_IR_GOLDEN_MANIFEST_VERSION
    )
    assert json.loads(encoded)["spec_version"] == ANCHOR_IR_SPEC_VERSION
    assert json.loads(encoded)["normalization_version"] == (
        ANCHOR_IR_NORMALIZATION_VERSION
    )
    assert [item["stage_id"] for item in json.loads(encoded)["stages"]] == [
        "adapter.output",
        "pass.canonicalize.after",
        "hook.vendor.after",
        "boundary.post_hook",
    ]
    for stage in decoded.stages:
        assert (
            stage.sha256
            == hashlib.sha256(stage.normalized_ir.encode("utf-8")).hexdigest()
        )


def test_manifest_json_rejects_duplicate_keys_at_every_depth():
    encoded = _manifest().to_json()
    duplicate_root = encoded.replace(
        '"case_id": "linalg/add",',
        '"case_id": "shadow",\n  "case_id": "linalg/add",',
        1,
    )
    duplicate_stage = encoded.replace(
        '"stage_id": "adapter.output"',
        '"stage_id": "hook.shadow.after",\n      "stage_id": "adapter.output"',
        1,
    )
    assert duplicate_root != encoded
    assert duplicate_stage != encoded

    for invalid in (duplicate_root, duplicate_stage):
        with pytest.raises(
            AnchorIRGoldenError,
            match=r"invalid Golden manifest JSON: duplicate key",
        ):
            AnchorIRGoldenManifest.from_json(invalid)


def test_manifest_json_utf8_input_limit_fails_before_json_decode(monkeypatch):
    monkeypatch.setattr(
        anchor_ir_golden,
        "MAX_ANCHOR_IR_GOLDEN_MANIFEST_BYTES",
        16,
    )

    def decoder_must_not_run(*_args, **_kwargs):
        raise AssertionError("oversized Golden JSON reached json.loads")

    monkeypatch.setattr(anchor_ir_golden.json, "loads", decoder_must_not_run)
    with pytest.raises(AnchorIRGoldenError, match="16-byte UTF-8 input limit"):
        AnchorIRGoldenManifest.from_json("测" * 6)


def test_manifest_json_nesting_limit_fails_before_json_decode(monkeypatch):
    deeply_nested = "[" * 257 + "0" + "]" * 257

    def decoder_must_not_run(*_args, **_kwargs):
        raise AssertionError("deep Golden JSON reached json.loads")

    monkeypatch.setattr(anchor_ir_golden.json, "loads", decoder_must_not_run)
    with pytest.raises(AnchorIRGoldenError, match="256-level nesting limit"):
        AnchorIRGoldenManifest.from_json(deeply_nested)


def test_manifest_stage_count_limit_precedes_per_stage_validation():
    payload = _manifest().to_dict()
    stage = copy.deepcopy(payload["stages"][0])
    payload["stages"] = [copy.deepcopy(stage) for _ in range(257)]

    with pytest.raises(AnchorIRGoldenError, match="256-Stage limit"):
        AnchorIRGoldenManifest.from_dict(payload)


def test_manifest_direct_dict_enforces_total_normalized_ir_budget(monkeypatch):
    payload = _manifest().to_dict()
    monkeypatch.setattr(anchor_ir_golden, "MAX_ANCHOR_IR_GOLDEN_PAYLOAD_BYTES", 1)

    def normalizer_must_not_run(*_args, **_kwargs):
        raise AssertionError("oversized manifest reached per-Stage normalization")

    monkeypatch.setattr(
        anchor_ir_golden.AnchorIRNormalizer,
        "normalize_text",
        normalizer_must_not_run,
    )

    with pytest.raises(AnchorIRGoldenError, match="normalized IR exceeds the 1-byte"):
        AnchorIRGoldenManifest.from_dict(payload)


def test_manifest_reuses_identical_payload_validation_across_stages(monkeypatch):
    payload = _manifest().to_dict()
    original = anchor_ir_golden.AnchorIRNormalizer.normalize_text
    calls = []

    def counting_normalizer(self, ir_text, **kwargs):
        calls.append((kwargs["phase"], kwargs["extension_dialects"], ir_text))
        return original(self, ir_text, **kwargs)

    monkeypatch.setattr(
        anchor_ir_golden.AnchorIRNormalizer,
        "normalize_text",
        counting_normalizer,
    )

    AnchorIRGoldenManifest.from_dict(payload)

    # The four sample stages contain one pre-hook and one post-hook policy for
    # the same canonical payload. Stage identity itself is not validation
    # input, so only those two distinct contracts require native work.
    assert len(calls) == 2


def test_manifest_default_verification_has_one_total_worker_deadline(monkeypatch):
    payload = _manifest(
        pass_ir=PASS_CHANGED_IR,
        hook_ir=PASS_CHANGED_IR,
        boundary_ir=PASS_CHANGED_IR,
    ).to_dict()
    normalizer = AnchorIRNormalizer()
    outcomes = {
        text: normalizer.normalize_text(
            text,
            normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            phase=phase,
        )
        for text, phase in (
            (payload["stages"][0]["normalized_ir"], "pre_hook"),
            (payload["stages"][1]["normalized_ir"], "pre_hook"),
        )
    }

    class FakeClock:
        values = iter((0.0, 0.0, 61.0))

        @classmethod
        def monotonic(cls):
            return next(cls.values)

    monkeypatch.setattr(anchor_ir_golden, "time", FakeClock, raising=False)
    monkeypatch.setattr(
        anchor_ir_golden.AnchorIRNormalizer,
        "normalize_text",
        lambda _self, ir_text, **_kwargs: outcomes[ir_text],
    )

    with pytest.raises(
        AnchorIRGoldenError,
        match="payload verification exceeded the 60-second total limit",
    ):
        AnchorIRGoldenManifest.from_dict(payload)


def test_manifest_json_large_integer_stays_in_golden_error_domain():
    invalid = '{"manifest_version":' + "9" * 5000 + "}"

    with pytest.raises(
        AnchorIRGoldenError,
        match="invalid Golden manifest JSON",
    ):
        AnchorIRGoldenManifest.from_json(invalid)


def test_manifest_json_nesting_scanner_ignores_escaped_quotes():
    encoded = (
        _manifest()
        .to_json()
        .replace(
            '"linalg/add"',
            '"linalg/\\"quoted\\"/[not-a-container]"',
            1,
        )
    )

    decoded = AnchorIRGoldenManifest.from_json(encoded)

    assert decoded.case_id == 'linalg/"quoted"/[not-a-container]'


def test_unmodified_pipeline_matches_every_stage():
    expected = _manifest()
    actual = _manifest()

    report = compare_anchor_ir_golden(expected, actual)

    assert report.matched
    assert report.first_divergence is None
    assert report.matched_stages == 4
    assert report.expected_stage_count == 4
    assert report.actual_stage_count == 4
    assert report.to_dict()["matched"] is True


def test_modified_hook_is_first_reported_at_corresponding_hook_stage():
    expected = _manifest()
    actual = _manifest(
        hook_ir=HOOK_CHANGED_IR,
        boundary_ir=HOOK_CHANGED_IR,
    )

    report = compare_anchor_ir_golden(expected, actual)

    assert not report.matched
    assert report.matched_stages == 2
    difference = report.first_divergence
    assert difference is not None
    assert difference.reason == "hash_mismatch"
    assert difference.stage_id == "hook.vendor.after"
    assert difference.expected_stage_id == "hook.vendor.after"
    assert difference.actual_stage_id == "hook.vendor.after"
    assert difference.old_hash == expected.stages[2].sha256
    assert difference.new_hash == actual.stages[2].sha256
    assert difference.old_hash != difference.new_hash
    assert report.first_changed_stage == "hook.vendor.after"
    assert report.old_hash == difference.old_hash
    assert report.new_hash == difference.new_hash
    assert report.normalized_ir_diff == difference.normalized_ir_diff
    assert "--- golden:hook.vendor.after" in difference.normalized_ir_diff
    assert "+++ current:hook.vendor.after" in difference.normalized_ir_diff
    assert "-  }) : () -> ()" in difference.normalized_ir_diff
    assert '+  }) {func.note = "hook-v2"} : () -> ()' in (difference.normalized_ir_diff)


def test_modified_pass_is_first_reported_after_that_pass():
    expected = _manifest()
    actual = _manifest(
        pass_ir=PASS_CHANGED_IR,
        hook_ir=PASS_CHANGED_IR,
        boundary_ir=PASS_CHANGED_IR,
    )

    report = compare_anchor_ir_golden(expected, actual)

    assert not report.matched
    assert report.matched_stages == 1
    difference = report.first_divergence
    assert difference is not None
    assert difference.reason == "hash_mismatch"
    assert difference.stage_id == "pass.canonicalize.after"
    assert difference.old_hash == expected.stages[1].sha256
    assert difference.new_hash == actual.stages[1].sha256
    assert '"arith.addi"' in difference.normalized_ir_diff
    assert '"arith.subi"' in difference.normalized_ir_diff


def test_stage_sequence_change_is_reported_at_first_changed_position():
    expected = _manifest()
    builder = AnchorIRGoldenBuilder(case_id="linalg/add")
    builder.add_text(AnchorIRStageId.adapter_output(), BASE_IR)
    builder.add_text(HOOK_STAGE, BASE_IR)
    builder.add_text(AnchorIRStageId.post_hook_boundary(), BASE_IR)
    actual = builder.build()

    report = compare_anchor_ir_golden(expected, actual)

    assert not report.matched
    assert report.matched_stages == 1
    difference = report.first_divergence
    assert difference is not None
    assert difference.reason == "stage_sequence_mismatch"
    assert difference.expected_stage_id == "pass.canonicalize.after"
    assert difference.actual_stage_id == "hook.vendor.after"
    assert difference.old_hash is not None
    assert difference.new_hash is not None
    assert difference.normalized_ir_diff == ""


def test_manifest_rejects_invalid_order_duplicate_stage_and_tampered_hash():
    manifest = _manifest()

    wrong_order = manifest.to_dict()
    wrong_order["stages"][1], wrong_order["stages"][2] = (
        wrong_order["stages"][2],
        wrong_order["stages"][1],
    )
    with pytest.raises(AnchorIRGoldenError, match="must follow"):
        AnchorIRGoldenManifest.from_dict(wrong_order)

    duplicate = manifest.to_dict()
    duplicate["stages"][2]["stage_id"] = duplicate["stages"][1]["stage_id"]
    duplicate["stages"][2]["phase"] = duplicate["stages"][1]["phase"]
    with pytest.raises(AnchorIRGoldenError, match="must be unique"):
        AnchorIRGoldenManifest.from_dict(duplicate)

    tampered = manifest.to_dict()
    tampered["stages"][1]["sha256"] = "0" * 64
    with pytest.raises(AnchorIRGoldenError, match="does not match"):
        AnchorIRGoldenManifest.from_dict(tampered)


def test_manifest_revalidates_stage_semantics_and_canonical_text():
    manifest = _manifest()
    invalid = manifest.to_dict()
    forbidden = 'module { "smt.forged"() : () -> () }\n'
    for stage in invalid["stages"]:
        stage["normalized_ir"] = forbidden
        stage["sha256"] = hashlib.sha256(forbidden.encode("utf-8")).hexdigest()

    with pytest.raises(AnchorIRGoldenValidationError, match="AIR-LINALG-001"):
        AnchorIRGoldenManifest.from_dict(invalid)

    noncanonical = manifest.to_dict()
    for stage in noncanonical["stages"]:
        stage["normalized_ir"] = BASE_IR
        stage["sha256"] = hashlib.sha256(BASE_IR.encode("utf-8")).hexdigest()

    with pytest.raises(AnchorIRGoldenError, match="not canonical"):
        AnchorIRGoldenManifest.from_dict(noncanonical)


def test_builder_is_poisoned_after_a_failed_stage_capture():
    builder = AnchorIRGoldenBuilder(case_id="poisoned/case")
    builder.add_text(AnchorIRStageId.adapter_output(), BASE_IR)
    with pytest.raises(AnchorIRGoldenValidationError):
        builder.add_text(
            PASS_STAGE,
            'module { "smt.invalid"() : () -> () }',
        )

    with pytest.raises(AnchorIRGoldenError, match="builder is poisoned"):
        builder.add_text(AnchorIRStageId.post_hook_boundary(), BASE_IR)
    with pytest.raises(AnchorIRGoldenError, match="builder is poisoned"):
        builder.build()

    fresh = AnchorIRGoldenBuilder(case_id="poisoned/fresh")
    fresh.add_text(AnchorIRStageId.adapter_output(), BASE_IR)
    fresh.add_text(AnchorIRStageId.post_hook_boundary(), BASE_IR)
    assert fresh.build().stages[0].stage_id == AnchorIRStageId.adapter_output()


def test_hook_stage_must_match_post_hook_boundary_and_policy():
    ir_mismatch = AnchorIRGoldenBuilder(case_id="hook-boundary/ir")
    ir_mismatch.add_text(AnchorIRStageId.adapter_output(), BASE_IR)
    ir_mismatch.add_text(HOOK_STAGE, BASE_IR)
    ir_mismatch.add_text(AnchorIRStageId.post_hook_boundary(), HOOK_CHANGED_IR)
    with pytest.raises(AnchorIRGoldenError, match="exactly match"):
        ir_mismatch.build()

    policy_mismatch = AnchorIRGoldenBuilder(case_id="hook-boundary/policy")
    policy_mismatch.add_text(AnchorIRStageId.adapter_output(), BASE_IR)
    policy_mismatch.add_text(
        HOOK_STAGE,
        BASE_IR,
        extension_dialects={"vendor_ext"},
    )
    policy_mismatch.add_text(AnchorIRStageId.post_hook_boundary(), BASE_IR)
    with pytest.raises(AnchorIRGoldenError, match="same extension"):
        policy_mismatch.build()


def test_no_hook_boundary_must_match_immediately_preceding_stage():
    builder = AnchorIRGoldenBuilder(case_id="no-hook-boundary/unrecorded-change")
    builder.add_text(AnchorIRStageId.adapter_output(), BASE_IR)
    builder.add_text(PASS_STAGE, BASE_IR)
    builder.add_text(AnchorIRStageId.post_hook_boundary(), HOOK_CHANGED_IR)

    with pytest.raises(AnchorIRGoldenError, match="exactly match"):
        builder.build()


def test_boundary_extension_requires_a_hook_stage():
    builder = AnchorIRGoldenBuilder(case_id="hook-boundary/missing-hook")
    builder.add_text(AnchorIRStageId.adapter_output(), BASE_IR)
    builder.add_text(
        AnchorIRStageId.post_hook_boundary(),
        'module { "vendor_ext.keep"() : () -> () }',
        extension_dialects={"vendor_ext"},
    )

    with pytest.raises(AnchorIRGoldenError, match="without a Hook Stage"):
        builder.build()


def test_to_json_refuses_output_larger_than_public_input_limit(monkeypatch):
    manifest = _manifest()
    encoded = manifest.to_json()
    monkeypatch.setattr(
        anchor_ir_golden,
        "MAX_ANCHOR_IR_GOLDEN_MANIFEST_BYTES",
        len(encoded.encode("utf-8")) - 1,
    )

    with pytest.raises(AnchorIRGoldenError, match="UTF-8 output limit"):
        manifest.to_json()


def test_builder_does_not_accept_a_caller_supplied_normalizer():
    with pytest.raises(TypeError, match="normalizer"):
        AnchorIRGoldenBuilder(case_id="fake-normalizer", normalizer=object())


@pytest.mark.parametrize(
    "case_id",
    [
        "bad" + "\ud800" + "case",
        "bad" + chr(0x7F) + "case",
        "bad" + chr(0x85) + "case",
    ],
)
def test_manifest_rejects_non_utf8_and_non_printable_case_ids(case_id):
    with pytest.raises(AnchorIRGoldenError, match="case_id"):
        AnchorIRGoldenBuilder(case_id=case_id)


def test_manifest_rejects_non_utf8_normalized_ir_with_domain_error():
    payload = copy.deepcopy(_manifest().to_dict())
    payload["stages"][0]["normalized_ir"] = "module {}" + "\ud800" + "\n"
    payload["stages"][0]["sha256"] = "0" * 64

    with pytest.raises(AnchorIRGoldenError, match="normalized_ir.*UTF-8"):
        AnchorIRGoldenManifest.from_dict(payload)


def test_manifest_rejects_unknown_track_and_phase_with_domain_errors():
    payload = _manifest().to_dict()
    payload["track"] = "bogus"
    with pytest.raises(AnchorIRGoldenError, match="invalid Golden track"):
        AnchorIRGoldenManifest.from_dict(payload)

    payload = _manifest().to_dict()
    payload["stages"][0]["phase"] = "bogus"
    with pytest.raises(AnchorIRGoldenError, match="invalid Golden Stage phase"):
        AnchorIRGoldenManifest.from_dict(payload)


def test_incompatible_manifest_metadata_is_not_compared():
    expected = _manifest()
    actual_dict = copy.deepcopy(expected.to_dict())
    actual_dict["case_id"] = "linalg/other"
    actual = AnchorIRGoldenManifest.from_dict(actual_dict)

    with pytest.raises(AnchorIRGoldenError, match="case_id"):
        compare_anchor_ir_golden(expected, actual)


def test_invalid_anchor_ir_cannot_be_recorded_as_golden():
    builder = AnchorIRGoldenBuilder(case_id="invalid/case")

    with pytest.raises(AnchorIRGoldenValidationError) as captured:
        builder.add_text(
            AnchorIRStageId.adapter_output(),
            'module { "smt.invalid"() : () -> () }',
        )

    error = captured.value
    assert error.stage_id == AnchorIRStageId.adapter_output()
    assert not error.report.valid
    assert error.report.diagnostics[0].code == "AIR-LINALG-001"


@pytest.mark.parametrize("track", list(AnchorIRTrack))
@pytest.mark.parametrize(
    "invalid",
    [
        'module { ".tt.hidden"() : () -> () }',
        "module attributes {func.container = {smt.marker}} {}",
    ],
    ids=("empty-operation-namespace", "nested-dictionary-attribute-name"),
)
def test_namespace_and_dictionary_regressions_cannot_enter_golden(track, invalid):
    builder = AnchorIRGoldenBuilder(
        case_id="invalid/%s" % track.value,
        track=track,
    )

    with pytest.raises(AnchorIRGoldenValidationError) as captured:
        builder.add_text(AnchorIRStageId.adapter_output(), invalid)

    assert captured.value.stage_id == AnchorIRStageId.adapter_output()
    assert not captured.value.report.valid


def test_wide_gpu_configuration_cannot_enter_golden_or_abort():
    builder = AnchorIRGoldenBuilder(
        case_id="invalid/wide-gpu-config",
        track=AnchorIRTrack.TRITON_GPU,
    )

    with pytest.raises(AnchorIRGoldenValidationError) as captured:
        builder.add_text(
            AnchorIRStageId.adapter_output(),
            """
module attributes {
  "triton_gpu.num-warps" = 18446744073709551615 : i65,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
}
""",
        )

    assert "AIR-GPU-011" in [item.code for item in captured.value.report.diagnostics]


def test_declared_post_hook_extension_is_part_of_manifest_policy():
    extension_ir = 'module { "vendor_ext.keep"() : () -> () }'
    builder = AnchorIRGoldenBuilder(case_id="extension/case")
    builder.add_text(AnchorIRStageId.adapter_output(), BASE_IR)
    builder.add_text(
        AnchorIRStageId.after_hook("vendor"),
        extension_ir,
        extension_dialects={"vendor_ext"},
    )
    builder.add_text(
        AnchorIRStageId.post_hook_boundary(),
        extension_ir,
        extension_dialects={"vendor_ext"},
    )
    manifest = builder.build()

    assert manifest.stages[1].extension_dialects == ("vendor_ext",)
    assert '"vendor_ext.keep"' in manifest.stages[1].normalized_ir
    assert AnchorIRGoldenManifest.from_json(manifest.to_json()) == manifest

    with pytest.raises(AnchorIRGoldenError, match="only valid"):
        AnchorIRGoldenBuilder(case_id="bad/extensions").add_text(
            AnchorIRStageId.adapter_output(),
            BASE_IR,
            extension_dialects={"vendor_ext"},
        )

    forbidden = AnchorIRGoldenBuilder(case_id="bad/forbidden-extension")
    forbidden.add_text(AnchorIRStageId.adapter_output(), BASE_IR)
    with pytest.raises(ValueError, match="core-forbidden.*smt"):
        forbidden.add_text(
            AnchorIRStageId.after_hook("vendor"),
            BASE_IR,
            extension_dialects={"smt"},
        )

    core = AnchorIRGoldenBuilder(case_id="bad/core-extension")
    core.add_text(AnchorIRStageId.adapter_output(), BASE_IR)
    with pytest.raises(ValueError, match="redeclare core dialect.*func"):
        core.add_text(
            AnchorIRStageId.after_hook("vendor"),
            BASE_IR,
            extension_dialects={"func"},
        )
