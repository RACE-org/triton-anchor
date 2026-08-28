#!/usr/bin/env python3
"""Human-readable component and in-repository Adapter acceptance for T6.5."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from triton._C.libtriton import anchor, ir, passes
from triton_anchor import (
    ANCHOR_IR_NORMALIZATION_VERSION,
    ANCHOR_IR_SPEC_VERSION,
    AnchorIRGoldenBuilder,
    AnchorIRGoldenManifest,
    AnchorIRNormalizer,
    AnchorIRPhase,
    AnchorIRTrack,
    AnchorIRValidationError,
    StructuredAnchorIRValidator,
    compare_anchor_ir_golden,
    resolve_anchor_ir_policy,
)
from triton_anchor.adapters.base import ITritonToLinalgAdapter
from triton_anchor.adapters.triton_linalg_adapter import TritonLinalgAdapter

REPOSITORY_ROOT = Path(
    os.environ.get("T65_REPOSITORY_ROOT", Path(__file__).resolve().parents[1])
).resolve()
CORPUS_ROOT = (
    REPOSITORY_ROOT / "python" / "triton_anchor" / "tests" / "data" / "anchor_ir"
)
CORPUS = json.loads((CORPUS_ROOT / "corpus.json").read_text(encoding="utf-8"))
MIN_VISIBLE_SAMPLES_PER_ACCEPTANCE = 2
NEGATIVE_CATEGORY_ORDER = (
    "nested_op",
    "unknown_dialect",
    "illegal_type",
    "illegal_attribute",
    "illegal_encoding",
    "semantic_missing",
)


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _paint(text: str, color: str) -> str:
    if not _supports_color():
        return text
    codes = {
        "green": "32",
        "red": "31",
        "cyan": "36",
        "yellow": "33",
        "bold": "1",
    }
    return "\033[%sm%s\033[0m" % (codes[color], text)


def _heading(title: str) -> None:
    print()
    print(_paint("=" * 78, "cyan"))
    print(_paint(title, "bold"))
    print(_paint("=" * 78, "cyan"))


def _ok(message: str) -> None:
    print("%s %s" % (_paint("[PASS]", "green"), message))


def _skip(message: str) -> None:
    print("%s %s" % (_paint("[SKIP]", "yellow"), message))


def _fail(message: str) -> None:
    raise AssertionError(message)


def _read(relative_path: str) -> str:
    return (CORPUS_ROOT / relative_path).read_text(encoding="utf-8")


def _validate_case(case):
    source = (CORPUS_ROOT / case["path"]).resolve()
    return StructuredAnchorIRValidator().validate_text(
        source.read_text(encoding="utf-8"),
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=case["track"],
        phase=AnchorIRPhase.PRE_HOOK,
        source_name=str(source),
    )


def _cli_command() -> list[str]:
    """Run the CLI through this process' isolated Python environment.

    A console-script wrapper may have a shebang pointing at another worktree's
    interpreter.  The module invocation is the same entry point declared by
    setup.py and preserves the caller's version-specific import path.
    """

    # The aggregate runner supplies a launcher whose interpreter is already
    # started with ``-S``; direct invocations intentionally use the caller's
    # normal environment.  In both cases this is the same published module
    # entry point, rather than a possibly stale console-script shebang.
    return [sys.executable, "-m", "triton_anchor.anchor_ir_cli"]


def _build_golden(case, replacements=None):
    replacements = replacements or {}
    builder = AnchorIRGoldenBuilder(
        case_id=case["case_id"],
        track=case["track"],
    )
    for stage in case["stages"]:
        builder.add_text(
            stage["stage_id"],
            replacements.get(stage["stage_id"], _read(stage["path"])),
            extension_dialects=stage.get("extension_dialects", ()),
        )
    return builder.build()


def _show_policy_contract() -> None:
    _heading("Additional work evidence — versioned dual-Track contract")
    for track in AnchorIRTrack:
        policy = resolve_anchor_ir_policy(
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=track,
            phase=AnchorIRPhase.PRE_HOOK,
        )
        print(
            "%-12s allowed=%2d  forbidden=%d  invariants=%d"
            % (
                track.value,
                len(policy.allowed_dialects),
                len(policy.forbidden_dialects),
                len(policy.enabled_invariants),
            )
        )
        print("  allowed   :", ", ".join(sorted(policy.allowed_dialects)))
        print("  forbidden :", ", ".join(sorted(policy.forbidden_dialects)))
    _ok("rules are loaded from %s" % ANCHOR_IR_SPEC_VERSION)


def _show_structured_validation() -> int:
    _heading(
        "Acceptance 1/4 — accurately reject nested Op, unknown dialect, "
        "illegal Type/Attribute and missing semantics"
    )
    for case in CORPUS["positive_cases"]:
        report = _validate_case(case)
        if not report.valid:
            _fail("%s unexpectedly failed" % case["case_id"])
        _ok("positive prerequisite %-37s valid" % case["case_id"])

    visible_samples = 0
    required_tracks = {track.value for track in AnchorIRTrack}
    for category in NEGATIVE_CATEGORY_ORDER:
        cases = [
            case for case in CORPUS["negative_cases"] if case["category"] == category
        ]
        if len(cases) < MIN_VISIBLE_SAMPLES_PER_ACCEPTANCE:
            _fail("%s has fewer than two committed negative samples" % category)
        if {case["track"] for case in cases} != required_tracks:
            _fail("%s does not cover both AnchorIR Tracks" % category)
        print("  category=%s independent_samples=%d" % (category, len(cases)))
        for case in cases:
            report = _validate_case(case)
            matching = [
                diagnostic
                for diagnostic in report.diagnostics
                if diagnostic.code == case["expected_code"]
                and diagnostic.object_kind.value == case["expected_object_kind"]
            ]
            if report.valid or not matching:
                _fail(
                    "%s did not produce %s" % (case["case_id"], case["expected_code"])
                )
            diagnostic = matching[0]
            if diagnostic.location is None or not diagnostic.hint:
                _fail("%s lacks an actionable location or hint" % case["case_id"])
            print(
                "%s [A1 sample] %-38s %-16s kind=%-9s line=%s"
                % (
                    _paint("[PASS]", "green"),
                    case["case_id"],
                    diagnostic.code,
                    diagnostic.object_kind.value,
                    diagnostic.location.line,
                )
            )
            if case["category"] == "nested_op":
                print("       operation_path:", diagnostic.operation_path)
            visible_samples += 1
    return visible_samples


def _show_python_cli_equivalence() -> int:
    _heading(
        "Acceptance 2/4 — Python API and published CLI return identical "
        "code, location and hint"
    )
    cli = _cli_command()
    print("  CLI entry point:", " ".join(cli))
    selectors = (
        ("linalg", "nested_op"),
        ("triton_gpu", "illegal_attribute"),
    )
    visible_samples = 0
    for track, category in selectors:
        case = next(
            item
            for item in CORPUS["negative_cases"]
            if item["track"] == track and item["category"] == category
        )
        source = (CORPUS_ROOT / case["path"]).resolve()
        python_report = _validate_case(case).to_dict()
        completed = subprocess.run(
            [
                *_cli_command(),
                str(source),
                "--spec-version",
                ANCHOR_IR_SPEC_VERSION,
                "--track",
                track,
                "--phase",
                AnchorIRPhase.PRE_HOOK.value,
                "--format",
                "json",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        cli_report = json.loads(completed.stdout)
        if completed.returncode != 1 or cli_report != python_report:
            _fail("%s Python/CLI report mismatch" % case["case_id"])
        diagnostic = next(
            item
            for item in cli_report["diagnostics"]
            if item["code"] == case["expected_code"]
        )
        _ok(
            "[A2 sample] %-38s code=%s line=%s"
            % (
                case["case_id"],
                diagnostic["code"],
                diagnostic["location"]["line"],
            )
        )
        print("       operation_path:", diagnostic["operation_path"])
        print("       hint:", diagnostic["hint"])
        visible_samples += 1
    return visible_samples


def _show_lifecycle() -> None:
    _heading(
        "Additional work evidence — production fail-closed "
        "Adapter -> pre-hook -> Hook -> post-hook -> lowering"
    )
    base = "module { func.func @kernel() { func.return } }"

    class Hook:
        def __init__(self, output, allowed=()):
            self.output = output
            self.allowed = allowed
            self.calls = 0

        def get_allowed_dialects(self):
            return self.allowed

        def on_anchor_ir_ready(self, _anchor_ir):
            self.calls += 1
            return self.output

    class Adapter(ITritonToLinalgAdapter):
        def __init__(self, output):
            self.output = output
            self.calls = 0

        def name(self):
            return "visual-adapter"

        def convert(self, _ttir_module, metadata, context=None):
            self.calls += 1
            metadata["adapter_called"] = True
            return self.output

    invalid_adapter = Adapter('module { "smt.invalid"() : () -> () }')
    skipped = Hook(base)
    try:
        invalid_adapter.compile(
            object(),
            {},
            hook=skipped,
            backend_lowering=None,
            source_name="visual-pre-invalid.mlir",
        )
    except AnchorIRValidationError as error:
        if invalid_adapter.calls != 1 or skipped.calls:
            _fail("pre-hook failure did not skip the Hook")
        diagnostic = error.report.diagnostics[0]
        _ok("pre-hook automatically raises %s and skips Backend Hook" % diagnostic.code)
        print("       operation_path:", diagnostic.operation_path)
        location = diagnostic.location
        if location is not None:
            print(
                "       location: %s:%s:%s"
                % (location.file, location.line, location.column)
            )
        print("       hint:", diagnostic.hint)
    else:
        _fail("invalid pre-hook output did not raise AnchorIRValidationError")

    extension = Hook(
        'module { "backend_ext.accepted"() : () -> () }',
        allowed={"backend_ext"},
    )
    accepted = Adapter(base).compile(
        object(),
        {},
        hook=extension,
        backend_lowering=None,
    )
    if not accepted.valid:
        _fail("declared post-hook extension was rejected")
    _ok("declared post-hook extension is accepted")

    forbidden = Hook(
        "module attributes {smt.marker = true} {}",
        allowed={"smt"},
    )
    lowered = []
    try:
        Adapter(base).compile(
            object(),
            {},
            hook=forbidden,
            source_name="visual-post-invalid.mlir",
            backend_lowering=lowered.append,
        )
    except ValueError as error:
        if forbidden.calls or lowered:
            _fail("forbidden extension declaration executed Hook or lowering")
        _ok("core-Forbidden extension declaration is rejected before Hook")
        print("       reason:", error)
    else:
        _fail("core-Forbidden extension declaration was accepted")


def _show_real_linalg_adapter_pipeline() -> None:
    _heading(
        "Integration evidence — real TritonLinalgAdapter C++ passes enter the "
        "strict boundary"
    )
    ttir = """
module {
  tt.func public @kernel() {
    %range = tt.make_range {start = 0 : i32, end = 16 : i32}
        : tensor<16xi32>
    tt.return
  }
}
"""
    hashes = []
    normalized_texts = []
    for _ in range(3):
        context = ir.context()
        # Deliberately load only the normal upstream registry.  The Adapter must
        # install its own Anchor/Linalg dialects and inliner extension.
        ir.load_dialects(context)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", encoding="utf-8"
        ) as source:
            source.write(ttir)
            source.flush()
            module = ir.parse_mlir_module(source.name, context)

        lowered = []
        report = TritonLinalgAdapter().compile(
            module,
            {},
            hook=None,
            backend_lowering=(
                lambda value, lowered=lowered: lowered.append(value) or "binary"
            ),
            context=context,
        )
        if (
            not report.valid
            or not report.lowering_executed
            or report.lowered_output != "binary"
            or len(lowered) != 1
            or lowered[0] is report.output
            or "tt.func" in str(report.output)
            or "func.func @kernel" not in str(report.output)
        ):
            _fail(
                "real TritonLinalgAdapter pipeline did not cross the strict "
                "boundary"
            )
        normalized = AnchorIRNormalizer().normalize_module(
            report.output,
            normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            phase=AnchorIRPhase.POST_HOOK,
        )
        if not normalized.acceptable:
            _fail("real Adapter output could not be normalized")
        hashes.append(normalized.sha256)
        normalized_texts.append(normalized.normalized_text)
    if len(set(hashes)) != 1 or len(set(normalized_texts)) != 1:
        _fail("repeated real Adapter compilation changed normalized AnchorIR")
    _ok(
        "3 repeated real C++ Adapter compilations produced identical normalized "
        "AnchorIR, passed pre/post validation and reached lowering"
    )
    print("  normalized_sha256:", hashes[0])
    print(
        "  scope note: this is the in-repository Adapter boundary; T10.3 must "
        "still drive full compiler corpus and per-Pass Stage capture"
    )


def _show_real_tritongpu_control_flow_pipeline() -> bool:
    _heading(
        "Integration evidence — real TTIR→TritonGPU conversion preserves "
        "versioned cf control flow"
    )
    converter = getattr(getattr(passes, "ttir", None), "add_convert_to_ttgpuir", None)
    if converter is None:
        _skip(
            "TTIR→TritonGPU converter is not exposed by this Triton branch; "
            "validator/GPU policy evidence remains covered, and T10.3 or the "
            "external backend owns this conversion boundary"
        )
        return False
    ttir = """
module {
  tt.func public @kernel(%flag: i1) {
    cf.cond_br %flag, ^positive, ^negative
  ^positive:
    tt.return
  ^negative:
    tt.return
  }
}
"""
    hashes = []
    for _ in range(3):
        context = ir.context()
        ir.load_dialects(context)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", encoding="utf-8"
        ) as source:
            source.write(ttir)
            source.flush()
            module = ir.parse_mlir_module(source.name, context)

        manager = ir.pass_manager(context)
        converter(
            manager,
            "cuda:80",
            4,
            32,
            1,
        )
        manager.run(module)
        anchor.load_dialects(context)
        report = StructuredAnchorIRValidator().validate_module(
            module,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.TRITON_GPU,
            phase=AnchorIRPhase.PRE_HOOK,
        )
        normalized = AnchorIRNormalizer().normalize_module(
            module,
            normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.TRITON_GPU,
            phase=AnchorIRPhase.PRE_HOOK,
        )
        if (
            "cf.cond_br" not in str(module)
            or not report.valid
            or not normalized.acceptable
        ):
            _fail("real TritonGPU control-flow output violated the current policy")
        hashes.append(normalized.sha256)

    if len(set(hashes)) != 1:
        _fail("repeated TritonGPU control-flow conversion changed its hash")
    _ok(
        "3 real TritonGPU conversions retained cf.cond_br, passed "
        "%s and produced one stable hash" % ANCHOR_IR_SPEC_VERSION
    )
    print("  normalized_sha256:", hashes[0])
    return True


def _show_regression_hardening() -> None:
    _heading(
        "Additional hardening evidence — namespace, Dictionary and wide GPU "
        "configuration regressions"
    )
    validator = StructuredAnchorIRValidator()
    regressions = (
        ('module { ".tt.hidden"() : () -> () }', "AIR-COMMON-001"),
        (
            "module attributes {func.container = {smt.marker}} {}",
            None,
        ),
    )
    for track in AnchorIRTrack:
        for text, common_code in regressions:
            report = validator.validate_text(
                text,
                spec_version=ANCHOR_IR_SPEC_VERSION,
                track=track,
                phase=AnchorIRPhase.PRE_HOOK,
                source_name="visual-hardening.mlir",
            )
            expected = common_code or (
                "AIR-LINALG-003"
                if track is AnchorIRTrack.LINALG
                else "AIR-GPU-003"
            )
            if report.valid or [item.code for item in report.diagnostics] != [
                expected
            ]:
                _fail(
                    "%s hardening case did not produce %s"
                    % (track.value, expected)
                )
        _ok(
            "%s rejects empty namespace and nested Dictionary attribute name"
            % track.value
        )

    wide_gpu = """
module attributes {
  "triton_gpu.num-warps" = 18446744073709551615 : i65,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
}
"""
    completed = subprocess.run(
        [
            *_cli_command(),
            "--spec-version",
            ANCHOR_IR_SPEC_VERSION,
            "--track",
            AnchorIRTrack.TRITON_GPU.value,
            "--phase",
            AnchorIRPhase.PRE_HOOK.value,
            "--format",
            "json",
        ],
        input=wide_gpu,
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if (
        completed.returncode != 1
        or completed.stderr
        or "AIR-GPU-011"
        not in [item["code"] for item in payload["diagnostics"]]
    ):
        _fail("wide GPU configuration did not return a structured CLI report")
    _ok("wide GPU IntegerAttr returns AIR-GPU-011 instead of aborting")

    hidden_property = validator.validate_text(
        'module { "vendor.op"() <{payload = #smt.marker}> : () -> () }',
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.POST_HOOK,
        extension_dialects={"vendor"},
        source_name="visual-property-hardening.mlir",
    )
    if not any(
        item.code == "AIR-LINALG-003"
        and item.object_path == "properties.entry[payload]"
        for item in hidden_property.diagnostics
    ):
        _fail("unregistered Operation properties bypassed Attribute validation")
    _ok("unregistered Operation properties cannot hide forbidden Attributes")

    hidden_dense = validator.validate_text(
        "module attributes {func.payload = "
        "dense<0> : tensor<1xi32, #smt.encoding>} {}",
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        source_name="visual-dense-hardening.mlir",
    )
    if not any(
        item.code == "AIR-LINALG-003"
        and item.object_path == "attribute[func.payload].type.encoding"
        for item in hidden_dense.diagnostics
    ):
        _fail("DenseElementsAttr type bypassed recursive Attribute validation")
    _ok("DenseElementsAttr cannot hide a forbidden Type/Encoding")

    unsafe_dense = (
        "module attributes {func.payload = "
        "dense<0> : tensor<1x!tt.ptr<i32>>} {}"
    )
    completed = subprocess.run(
        [
            *_cli_command(),
            "--spec-version",
            ANCHOR_IR_SPEC_VERSION,
            "--track",
            AnchorIRTrack.LINALG.value,
            "--phase",
            AnchorIRPhase.PRE_HOOK.value,
            "--format",
            "json",
        ],
        input=unsafe_dense,
        check=False,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if (
        completed.returncode != 1
        or completed.stderr
        or [item["code"] for item in payload["diagnostics"]]
        != ["AIR-PARSE-001"]
    ):
        _fail("unsafe dense parser input was not converted to a CLI Report")
    _ok("native dense parser assertion is contained as AIR-PARSE-001")

    normalized = AnchorIRNormalizer().normalize_text(
        'module { ".tt.hidden"() : () -> () }',
        normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    )
    if normalized.acceptable or normalized.normalized_text or normalized.sha256:
        _fail("invalid namespace produced a normalization or Golden artifact")
    _ok("hardening failures cannot produce normalized IR or SHA-256")


def _show_stable_hashes_and_golden() -> int:
    _heading(
        "Acceptance 3/4 — repeated input has stable normalized hash and "
        "Golden comparison"
    )
    visible_samples = 0
    for track in AnchorIRTrack:
        case = next(
            case for case in CORPUS["positive_cases"] if case["track"] == track.value
        )
        results = [
            AnchorIRNormalizer().normalize_text(
                _read(case["path"]),
                normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
                spec_version=ANCHOR_IR_SPEC_VERSION,
                track=case["track"],
                phase=AnchorIRPhase.PRE_HOOK,
            )
            for _ in range(5)
        ]
        values = {(result.normalized_text, result.sha256) for result in results}
        if len(values) != 1 or not all(result.acceptable for result in results):
            _fail("%s normalization is unstable" % case["case_id"])
        golden_case = next(
            golden
            for golden in CORPUS["golden_cases"]
            if golden["track"] == track.value
        )
        expected = AnchorIRGoldenManifest.from_json(_read(golden_case["manifest"]))
        current_manifests = [_build_golden(golden_case) for _ in range(3)]
        comparisons = [
            compare_anchor_ir_golden(expected, current) for current in current_manifests
        ]
        if (
            not all(comparison.matched for comparison in comparisons)
            or len({current.to_json() for current in current_manifests}) != 1
        ):
            _fail("%s committed Golden does not match" % golden_case["case_id"])
        unchanged = comparisons[0]
        _ok(
            "[A3 sample] %-38s repeats=5 hash=%s" % (case["case_id"], results[0].sha256)
        )
        print(
            "       Golden %-31s rebuilds=3 matched_stages=%d"
            % (
                golden_case["case_id"],
                unchanged.matched_stages,
            )
        )
        print(
            "       Stage IDs:",
            " -> ".join(stage.stage_id.value for stage in expected.stages),
        )
        visible_samples += 1
    return visible_samples


def _changed_pass_text(case, pass_text: str) -> str:
    if case["track"] == AnchorIRTrack.LINALG.value:
        changed = pass_text.replace("arith.muli", "arith.subi")
    else:
        changed = pass_text.replace("end = 32", "end = 64").replace(
            "tensor<32xi32",
            "tensor<64xi32",
        )
    if changed == pass_text:
        _fail("%s Pass mutation did not change its IR" % case["case_id"])
    return changed


def _print_divergence(case, mutation_kind, expected_stage, report) -> None:
    if report.first_changed_stage != expected_stage:
        _fail(
            "%s %s drift expected %s, got %s"
            % (
                case["track"],
                mutation_kind,
                expected_stage,
                report.first_changed_stage,
            )
        )
    if (
        report.old_hash is None
        or report.new_hash is None
        or report.old_hash == report.new_hash
        or not report.normalized_ir_diff
    ):
        _fail(
            "%s %s drift lacks hashes or normalized diff"
            % (case["track"], mutation_kind)
        )
    print(
        "%s [A4 sample] %-12s mutation=%-4s first_changed_stage=%s"
        % (
            _paint("[PASS]", "green"),
            case["track"],
            mutation_kind,
            _paint(report.first_changed_stage, "yellow"),
        )
    )
    print("       old_hash:", report.old_hash)
    print("       new_hash:", report.new_hash)
    print("       normalized IR diff:")
    for line in report.normalized_ir_diff.splitlines()[:8]:
        print("         " + line)


def _show_first_divergence() -> int:
    _heading(
        "Acceptance 4/4 — Pass and Hook changes identify the first semantic "
        "divergence Stage"
    )
    visible_samples = 0
    for case in CORPUS["golden_cases"]:
        expected = AnchorIRGoldenManifest.from_json(_read(case["manifest"]))

        hook = next(
            stage
            for stage in case["stages"]
            if stage["stage_id"] == "hook.vendor.after"
        )
        changed_hook = _read(hook["path"]).replace(
            "vendor.marker",
            "vendor.changed",
        )
        changed = _build_golden(
            case,
            {
                "hook.vendor.after": changed_hook,
                "boundary.post_hook": changed_hook,
            },
        )
        _print_divergence(
            case,
            "Hook",
            "hook.vendor.after",
            compare_anchor_ir_golden(expected, changed),
        )
        visible_samples += 1

        pass_stage = next(
            stage
            for stage in case["stages"]
            if stage["stage_id"] == "pass.canonicalize.after"
        )
        changed_pass = _changed_pass_text(case, _read(pass_stage["path"]))
        changed = _build_golden(
            case,
            {"pass.canonicalize.after": changed_pass},
        )
        _print_divergence(
            case,
            "Pass",
            "pass.canonicalize.after",
            compare_anchor_ir_golden(expected, changed),
        )
        visible_samples += 1
    return visible_samples


def _verify_visible_coverage(sample_counts) -> None:
    _heading("VISIBLE ACCEPTANCE COVERAGE")
    for criterion in range(1, 5):
        count = sample_counts[criterion]
        if count < MIN_VISIBLE_SAMPLES_PER_ACCEPTANCE:
            _fail(
                "acceptance %d has %d visible samples; at least %d are required"
                % (criterion, count, MIN_VISIBLE_SAMPLES_PER_ACCEPTANCE)
            )
        _ok(
            "acceptance %d visible_samples=%d required>=%d"
            % (criterion, count, MIN_VISIBLE_SAMPLES_PER_ACCEPTANCE)
        )


def main() -> int:
    print(_paint("T6.5 AnchorIR implementation acceptance demo", "bold"))
    print("repository:", REPOSITORY_ROOT)
    sample_counts = {
        1: _show_structured_validation(),
        2: _show_python_cli_equivalence(),
        3: _show_stable_hashes_and_golden(),
        4: _show_first_divergence(),
    }
    _verify_visible_coverage(sample_counts)
    _show_policy_contract()
    _show_lifecycle()
    _show_real_linalg_adapter_pipeline()
    _show_real_tritongpu_control_flow_pipeline()
    _show_regression_hardening()
    _heading("FINAL RESULT")
    _ok("all visible T6.5 implementation checks passed")
    print(
        "External integration remains: T10.3 corpus runner and out-of-tree "
        "backends must invoke these interfaces in the full compiler pipeline."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, ValueError) as error:
        print("%s %s" % (_paint("[FAIL]", "red"), error), file=sys.stderr)
        raise SystemExit(1) from None
