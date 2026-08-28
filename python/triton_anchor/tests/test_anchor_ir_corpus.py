"""T6.5 component acceptance over the committed dual-Track corpus.

T10.3 remains responsible for driving these cases through complete compiler
pipelines and supplying real per-Pass Stage observations.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from triton_anchor import (
    ANCHOR_IR_NORMALIZATION_VERSION,
    ANCHOR_IR_SPEC_VERSION,
    AnchorIRGoldenBuilder,
    AnchorIRGoldenManifest,
    AnchorIRNormalizer,
    AnchorIRPhase,
    AnchorIRTrack,
    StructuredAnchorIRValidator,
    compare_anchor_ir_golden,
)

CORPUS_ROOT = Path(__file__).parent / "data" / "anchor_ir"
CORPUS_INDEX_PATH = CORPUS_ROOT / "corpus.json"
CORPUS_INDEX = json.loads(CORPUS_INDEX_PATH.read_text(encoding="utf-8"))
POSITIVE_CASES = CORPUS_INDEX["positive_cases"]
NEGATIVE_CASES = CORPUS_INDEX["negative_cases"]
GOLDEN_CASES = CORPUS_INDEX["golden_cases"]
REQUIRED_NEGATIVE_CATEGORIES = {
    "nested_op",
    "unknown_dialect",
    "illegal_type",
    "illegal_attribute",
    "illegal_encoding",
    "semantic_missing",
}


def _case_id(case):
    return case["case_id"]


def _read(relative_path):
    return (CORPUS_ROOT / relative_path).read_text(encoding="utf-8")


def _validate(case):
    source_path = (CORPUS_ROOT / case["path"]).resolve()
    return StructuredAnchorIRValidator().validate_text(
        source_path.read_text(encoding="utf-8"),
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=case["track"],
        phase=AnchorIRPhase.PRE_HOOK,
        source_name=str(source_path),
    )


def _cli_report(case):
    source_path = (CORPUS_ROOT / case["path"]).resolve()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "triton_anchor.anchor_ir_cli",
            str(source_path),
            "--spec-version",
            ANCHOR_IR_SPEC_VERSION,
            "--track",
            case["track"],
            "--phase",
            AnchorIRPhase.PRE_HOOK.value,
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json.loads(completed.stdout)


def _build_golden(case, *, replacements=None):
    replacements = replacements or {}
    builder = AnchorIRGoldenBuilder(
        case_id=case["case_id"],
        spec_version=ANCHOR_IR_SPEC_VERSION,
        normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
        track=case["track"],
    )
    for stage in case["stages"]:
        text = replacements.get(stage["stage_id"], _read(stage["path"]))
        builder.add_text(
            stage["stage_id"],
            text,
            extension_dialects=stage.get("extension_dialects", ()),
            source_name=str((CORPUS_ROOT / stage["path"]).resolve()),
        )
    return builder.build()


def test_corpus_index_is_complete_unique_and_has_no_unindexed_mlir():
    assert CORPUS_INDEX["schema_version"] == 1
    all_cases = POSITIVE_CASES + NEGATIVE_CASES + GOLDEN_CASES
    case_ids = [case["case_id"] for case in all_cases]
    assert len(case_ids) == len(set(case_ids))

    for track in AnchorIRTrack:
        assert {
            case["category"] for case in NEGATIVE_CASES if case["track"] == track.value
        } == REQUIRED_NEGATIVE_CATEGORIES
        assert sum(case["track"] == track.value for case in POSITIVE_CASES) >= 1
        assert sum(case["track"] == track.value for case in GOLDEN_CASES) >= 1

    indexed_mlir = {case["path"] for case in POSITIVE_CASES + NEGATIVE_CASES}
    for case in GOLDEN_CASES:
        indexed_mlir.update(stage["path"] for stage in case["stages"])
    actual_mlir = {
        str(path.relative_to(CORPUS_ROOT)) for path in CORPUS_ROOT.rglob("*.mlir")
    }
    assert indexed_mlir == actual_mlir
    assert {case["manifest"] for case in GOLDEN_CASES} == {
        str(path.relative_to(CORPUS_ROOT))
        for path in CORPUS_ROOT.glob("golden/*/*.json")
    }


@pytest.mark.parametrize("case", POSITIVE_CASES, ids=_case_id)
def test_dual_track_positive_samples_pass_python_and_cli(case):
    python_report = _validate(case)
    completed, cli_report = _cli_report(case)

    assert python_report.valid
    assert python_report.diagnostics == ()
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert cli_report == python_report.to_dict()


@pytest.mark.parametrize("case", NEGATIVE_CASES, ids=_case_id)
def test_dual_track_negative_samples_have_exact_actionable_diagnostics(case):
    python_report = _validate(case)
    completed, cli_report = _cli_report(case)

    assert not python_report.valid
    matching = [
        diagnostic
        for diagnostic in python_report.diagnostics
        if diagnostic.code == case["expected_code"]
        and diagnostic.object_kind.value == case["expected_object_kind"]
    ]
    assert matching
    if "operation_path_contains" in case:
        assert any(
            case["operation_path_contains"] in diagnostic.operation_path
            for diagnostic in matching
        )
    assert all(diagnostic.hint for diagnostic in matching)
    assert all(
        diagnostic.location is not None
        and diagnostic.location.file == str((CORPUS_ROOT / case["path"]).resolve())
        for diagnostic in matching
    )

    assert completed.returncode == 1
    assert completed.stderr == ""
    assert cli_report == python_report.to_dict()


@pytest.mark.parametrize("case", POSITIVE_CASES, ids=_case_id)
def test_dual_track_positive_hashes_repeat_exactly(case):
    text = _read(case["path"])
    results = [
        AnchorIRNormalizer().normalize_text(
            text,
            normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=case["track"],
            phase=AnchorIRPhase.PRE_HOOK,
            source_name=str((CORPUS_ROOT / case["path"]).resolve()),
        )
        for _ in range(5)
    ]

    assert all(result.acceptable for result in results)
    assert len({(result.normalized_text, result.sha256) for result in results}) == 1


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=_case_id)
def test_committed_dual_track_golden_matches_all_source_stages(case):
    golden_path = CORPUS_ROOT / case["manifest"]
    expected = AnchorIRGoldenManifest.from_json(golden_path.read_text(encoding="utf-8"))
    actual_manifests = [_build_golden(case) for _ in range(3)]
    comparisons = [
        compare_anchor_ir_golden(expected, actual) for actual in actual_manifests
    ]
    comparison = comparisons[0]

    assert expected.to_json() == golden_path.read_text(encoding="utf-8")
    assert all(result.matched for result in comparisons)
    assert len({actual.to_json() for actual in actual_manifests}) == 1
    assert comparison.matched_stages == 4
    assert [stage.stage_id.value for stage in expected.stages] == [
        "adapter.output",
        "pass.canonicalize.after",
        "hook.vendor.after",
        "boundary.post_hook",
    ]
    assert expected.stages[0].sha256 != expected.stages[1].sha256
    assert expected.stages[1].sha256 != expected.stages[2].sha256
    assert expected.stages[2].sha256 == expected.stages[3].sha256


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=_case_id)
def test_dual_track_hook_change_is_first_located_at_hook_stage(case):
    expected = AnchorIRGoldenManifest.from_json(_read(case["manifest"]))
    hook_stage = next(
        stage for stage in case["stages"] if stage["stage_id"] == "hook.vendor.after"
    )
    changed_hook = _read(hook_stage["path"]).replace(
        "vendor.marker",
        "vendor.changed",
    )
    actual = _build_golden(
        case,
        replacements={
            "hook.vendor.after": changed_hook,
            "boundary.post_hook": changed_hook,
        },
    )

    report = compare_anchor_ir_golden(expected, actual)

    assert not report.matched
    assert report.first_changed_stage == "hook.vendor.after"
    assert report.old_hash is not None
    assert report.new_hash is not None
    assert report.old_hash != report.new_hash
    assert "vendor.marker" in report.normalized_ir_diff
    assert "vendor.changed" in report.normalized_ir_diff


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=_case_id)
def test_dual_track_pass_change_is_first_located_after_pass(case):
    expected = AnchorIRGoldenManifest.from_json(_read(case["manifest"]))
    pass_stage = next(
        stage
        for stage in case["stages"]
        if stage["stage_id"] == "pass.canonicalize.after"
    )
    pass_text = _read(pass_stage["path"])
    if case["track"] == AnchorIRTrack.LINALG.value:
        changed_pass = pass_text.replace("arith.muli", "arith.subi")
    else:
        changed_pass = pass_text.replace("end = 32", "end = 64").replace(
            "tensor<32xi32",
            "tensor<64xi32",
        )
    actual = _build_golden(
        case,
        replacements={
            "pass.canonicalize.after": changed_pass,
        },
    )

    report = compare_anchor_ir_golden(expected, actual)

    assert not report.matched
    assert report.first_changed_stage == "pass.canonicalize.after"
    assert report.old_hash is not None
    assert report.new_hash is not None
    assert report.old_hash != report.new_hash
    assert report.normalized_ir_diff
