"""Acceptance tests for versioned AnchorIR normalization and SHA-256."""

import hashlib
import json
import subprocess
import sys

import pytest
from triton._C.libtriton import anchor, ir

from triton_anchor import (
    ANCHOR_IR_NORMALIZATION_VERSION,
    ANCHOR_IR_SPEC_VERSION,
    AnchorIRNormalizer,
    AnchorIRPhase,
    AnchorIRTrack,
    resolve_anchor_ir_policy,
)
from triton_anchor import _anchor_ir_text_isolation as text_isolation

SCALAR_IR = """
module {
  func.func @compute(%input: i32) -> i32 {
    %result = arith.addi %input, %input : i32
    func.return %result : i32
  }
}
"""

RENAMED_LOCATED_IR = """
module {
  func.func @choose(%input: i32, %flag: i1) -> i32 {
    cf.cond_br %flag, ^positive, ^negative
  ^positive:
    %twice = arith.addi %input, %input : i32 loc("first.mlir":6:5)
    cf.br ^merge(%twice : i32)
  ^negative:
    %copy = arith.addi %input, %input : i32
    cf.br ^merge(%copy : i32)
  ^merge(%value: i32):
    func.return %value : i32
  }
}
"""

CRLF_RENAMED_LOCATED_IR = (
    "\r\nmodule {\r\n"
    " func.func @choose(%x : i32, %condition : i1) -> i32 {\r\n"
    " cf.cond_br %condition, ^alpha, ^beta\r\n"
    " ^alpha:\r\n"
    ' %first = arith.addi %x, %x : i32 loc("other.mlir":99:7)\r\n'
    " cf.br ^done(%first : i32)\r\n"
    " ^beta:\r\n"
    " %second = arith.addi %x, %x : i32\r\n"
    " cf.br ^done(%second : i32)\r\n"
    " ^done(%answer : i32):\r\n"
    " func.return %answer : i32\r\n"
    " }\r\n"
    "}\r\n"
)


def _normalize(
    text,
    *,
    track=AnchorIRTrack.LINALG,
    phase=AnchorIRPhase.PRE_HOOK,
    extension_dialects=None,
):
    return AnchorIRNormalizer().normalize_text(
        text,
        normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=track,
        phase=phase,
        source_name="normalization.mlir",
        extension_dialects=extension_dialects,
    )


def test_same_module_and_text_repeat_with_identical_text_and_hash(tmp_path):
    source = tmp_path / "stable.mlir"
    source.write_text(SCALAR_IR, encoding="utf-8")
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    normalizer = AnchorIRNormalizer()

    module_results = [
        normalizer.normalize_module(
            module,
            normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            phase=AnchorIRPhase.PRE_HOOK,
        )
        for _ in range(10)
    ]
    text_results = [_normalize(SCALAR_IR) for _ in range(10)]

    expected_text = module_results[0].normalized_text
    expected_hash = module_results[0].sha256
    assert expected_text is not None
    assert expected_hash is not None
    assert all(item.acceptable for item in module_results + text_results)
    assert {
        (item.normalized_text, item.sha256) for item in module_results + text_results
    } == {(expected_text, expected_hash)}
    renormalized = _normalize(expected_text)
    assert renormalized.acceptable
    assert renormalized.normalized_text == expected_text
    assert renormalized.sha256 == expected_hash


def test_same_text_is_byte_stable_across_independent_processes():
    script = """
import json
import sys
from triton_anchor import (
    ANCHOR_IR_NORMALIZATION_VERSION,
    ANCHOR_IR_SPEC_VERSION,
    AnchorIRNormalizer,
    AnchorIRPhase,
    AnchorIRTrack,
)

result = AnchorIRNormalizer().normalize_text(
    sys.stdin.read(),
    normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
    spec_version=ANCHOR_IR_SPEC_VERSION,
    track=AnchorIRTrack.LINALG,
    phase=AnchorIRPhase.PRE_HOOK,
)
print(json.dumps({
    "acceptable": result.acceptable,
    "normalized_text": result.normalized_text,
    "sha256": result.sha256,
}, ensure_ascii=False, sort_keys=True))
"""
    outputs = []
    for _ in range(3):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            input=SCALAR_IR,
            text=True,
            capture_output=True,
            check=True,
        )
        outputs.append(completed.stdout)

    assert len(set(outputs)) == 1
    payload = json.loads(outputs[0])
    assert payload["acceptable"] is True
    assert (
        payload["sha256"]
        == hashlib.sha256(payload["normalized_text"].encode("utf-8")).hexdigest()
    )


def test_dense_resource_payload_is_part_of_normalized_text_and_hash():
    def resource_ir(value):
        return """module attributes {func.payload = dense_resource<blob1> : tensor<1xi64>} {}
{-#
  dialect_resources: {
    builtin: {
      blob1: \"0x0800000000000000%s\"
    }
  }
#-}
""" % value

    first = _normalize(resource_ir("0100000000000000"))
    second = _normalize(resource_ir("0200000000000000"))

    assert first.acceptable and second.acceptable
    assert "dialect_resources" in first.normalized_text
    assert "0100000000000000" in first.normalized_text
    assert "0200000000000000" in second.normalized_text
    assert first.normalized_text != second.normalized_text
    assert first.sha256 != second.sha256
    # The canonical resource section must remain parseable and idempotent.
    rerun = _normalize(first.normalized_text)
    assert rerun.acceptable
    assert rerun.normalized_text == first.normalized_text
    assert rerun.sha256 == first.sha256


def test_resource_free_canonical_bytes_remain_legacy_compatible():
    result = _normalize(SCALAR_IR)

    assert result.acceptable
    # This is the pre-resource-fix local-scope byte contract.  Keep the
    # literal here so a future printer change cannot silently invalidate the
    # already committed resource-free Track Goldens.
    assert result.normalized_text == (
        '"builtin.module"() ({\n'
        '  "func.func"() <{function_type = (i32) -> i32, '
        'sym_name = "compute"}> ({\n'
        '  ^bb0(%arg0: i32):\n'
        '    %0 = "arith.addi"(%arg0, %arg0) '
        '<{overflowFlags = #arith.overflow<none>}> : '
        '(i32, i32) -> i32\n'
        '    "func.return"(%0) : (i32) -> ()\n'
        '  }) : () -> ()\n'
        '}) : () -> ()\n'
    )


def test_nested_resource_payload_is_not_elided_from_hash():
    def resource_ir(value):
        return """module attributes {func.payload = [{nested = dense_resource<blob1> : tensor<1xi64>}] } {}
{-#
  dialect_resources: {
    builtin: {
      blob1: \"0x0800000000000000%s\"
    }
  }
#-}
""" % value

    first = _normalize(resource_ir("0100000000000000"))
    second = _normalize(resource_ir("0200000000000000"))

    assert first.acceptable and second.acceptable
    assert first.sha256 != second.sha256
    assert "dialect_resources" in first.normalized_text
    assert "0100000000000000" in first.normalized_text
    assert "0200000000000000" in second.normalized_text


def test_normalize_module_does_not_mutate_callers_module(tmp_path):
    source = tmp_path / "unchanged.mlir"
    source.write_text(
        """
module attributes {
  func.debug = [loc("original.mlir":1:2),
                {nested = loc("original.mlir":3:4)}]
} {
  func.func @compute() {
    func.return loc("original.mlir":7:5)
  }
}
""",
        encoding="utf-8",
    )
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    before = str(module)

    result = AnchorIRNormalizer().normalize_module(
        module,
        normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    )

    assert result.acceptable
    assert str(module) == before
    assert "original.mlir" in before
    assert "original.mlir" not in result.normalized_text


def test_post_hook_extension_is_normalized_only_when_declared():
    extension_ir = 'module { "vendor_ext.keep"() : () -> () }'

    undeclared = _normalize(
        extension_ir,
        phase=AnchorIRPhase.POST_HOOK,
    )
    declared = _normalize(
        extension_ir,
        phase=AnchorIRPhase.POST_HOOK,
        extension_dialects={"vendor_ext"},
    )
    declared_again = _normalize(
        extension_ir,
        phase=AnchorIRPhase.POST_HOOK,
        extension_dialects={"vendor_ext"},
    )

    assert not undeclared.acceptable
    assert undeclared.normalized_text is None
    assert undeclared.sha256 is None
    assert declared.acceptable and declared_again.acceptable
    assert '"vendor_ext.keep"' in declared.normalized_text
    assert declared.normalized_text == declared_again.normalized_text
    assert declared.sha256 == declared_again.sha256


def test_whitespace_ssa_block_names_and_locations_do_not_change_hash():
    first = _normalize(RENAMED_LOCATED_IR)
    second = _normalize(CRLF_RENAMED_LOCATED_IR)

    assert first.acceptable and second.acceptable
    assert first.normalized_text == second.normalized_text
    assert first.sha256 == second.sha256
    assert "first.mlir" not in first.normalized_text
    assert "other.mlir" not in second.normalized_text
    assert "%twice" not in first.normalized_text
    assert "^positive" not in first.normalized_text
    assert "^bb0" in first.normalized_text


@pytest.mark.parametrize(
    "equivalent_ir",
    [
        (
            "\r\nmodule {\r\n"
            " func.func @compute(%input : i32) -> i32 {\r\n"
            " %result = arith.addi %input, %input : i32\r\n"
            " func.return %result : i32\r\n"
            " }\r\n}\r\n"
        ),
        """
module {
  func.func @compute(%renamed: i32) -> i32 {
    %renamed_result = arith.addi %renamed, %renamed : i32
    func.return %renamed_result : i32
  }
}
""",
        """
module {
  func.func @compute(%input: i32) -> i32 {
    %result = arith.addi %input, %input : i32 loc("different.mlir":40:9)
    func.return %result : i32 loc("different.mlir":41:3)
  } loc("different.mlir":39:1)
}
""",
    ],
    ids=("whitespace-and-crlf", "ssa-names", "source-locations"),
)
def test_each_nonsemantic_text_change_individually_keeps_hash(equivalent_ir):
    baseline = _normalize(SCALAR_IR)
    changed = _normalize(equivalent_ir)

    assert baseline.acceptable and changed.acceptable
    assert baseline.normalized_text == changed.normalized_text
    assert baseline.sha256 == changed.sha256


def test_location_attributes_are_recursively_canonicalized():
    first = _normalize(
        'module attributes {func.debug = [loc("a":1:2), {nested = loc("a":3:4)}]} {}'
    )
    second = _normalize(
        'module attributes {func.debug = [loc("b":90:8), {nested = loc("c":100:9)}]} {}'
    )

    assert first.acceptable and second.acceptable
    assert first.normalized_text == second.normalized_text
    assert first.sha256 == second.sha256
    assert "loc(unknown)" in first.normalized_text
    assert '"a"' not in first.normalized_text
    assert '"b"' not in second.normalized_text


@pytest.mark.parametrize(
    "first_ir, second_ir",
    [
        (SCALAR_IR, SCALAR_IR.replace("arith.addi", "arith.subi")),
        (SCALAR_IR, SCALAR_IR.replace("i32", "i64")),
        (
            'module attributes {func.semantic = "A"} { }',
            'module attributes {func.semantic = "B"} { }',
        ),
    ],
)
def test_operation_type_and_semantic_attribute_changes_change_hash(
    first_ir,
    second_ir,
):
    first = _normalize(first_ir)
    second = _normalize(second_ir)

    assert first.acceptable and second.acceptable
    assert first.normalized_text != second.normalized_text
    assert first.sha256 != second.sha256


def test_encoding_and_parallelism_attributes_are_preserved_and_hashed():
    def gpu_ir(size_per_thread):
        return (
            """
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  tt.make_range {start = 0 : i32, end = 16 : i32}
    : tensor<16xi32, #triton_gpu.blocked<{
        sizePerThread = [%d], threadsPerWarp = [32],
        warpsPerCTA = [1], order = [0]
      }>>
}
"""
            % size_per_thread
        )

    first = _normalize(gpu_ir(1), track=AnchorIRTrack.TRITON_GPU)
    second = _normalize(gpu_ir(2), track=AnchorIRTrack.TRITON_GPU)

    assert first.acceptable and second.acceptable
    assert first.sha256 != second.sha256
    for attribute in (
        "triton_gpu.num-warps",
        "triton_gpu.threads-per-warp",
        "triton_gpu.num-ctas",
        "triton_gpu.blocked",
    ):
        assert attribute in first.normalized_text


def test_output_is_utf8_lf_one_newline_and_sha256_of_exact_bytes():
    result = _normalize('module attributes {func.note = "中文"} { }\r\n')

    assert result.acceptable
    assert result.normalized_text is not None
    assert result.normalized_bytes is not None
    assert result.normalized_bytes.decode("utf-8") == result.normalized_text
    assert "\r" not in result.normalized_text
    assert result.normalized_text.endswith("\n")
    assert not result.normalized_text.endswith("\n\n")
    assert result.sha256 == hashlib.sha256(result.normalized_bytes).hexdigest()


@pytest.mark.parametrize("track", list(AnchorIRTrack))
@pytest.mark.parametrize(
    "invalid",
    [
        'module { ".tt.hidden"() : () -> () }',
        "module attributes {func.container = {smt.marker}} {}",
    ],
    ids=("empty-operation-namespace", "nested-dictionary-attribute-name"),
)
def test_namespace_and_dictionary_regressions_never_produce_hash(track, invalid):
    result = _normalize(invalid, track=track)

    assert not result.acceptable
    assert not result.validation_report.valid
    assert result.normalized_text is None
    assert result.normalized_bytes is None
    assert result.sha256 is None


def test_wide_gpu_configuration_never_produces_hash_or_aborts():
    result = _normalize(
        """
module attributes {
  "triton_gpu.num-warps" = 18446744073709551615 : i65,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
}
""",
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert not result.acceptable
    assert "AIR-GPU-011" in [
        item.code for item in result.validation_report.diagnostics
    ]
    assert result.normalized_text is None
    assert result.sha256 is None


@pytest.mark.parametrize(
    "invalid, expected_code",
    [
        ('module { "smt.invalid"() : () -> () }', "AIR-LINALG-001"),
        ("module { func.func @broken( { }", "AIR-PARSE-001"),
        (
            "module { func.func @bad() -> i32 { func.return } }",
            "AIR-VERIFY-001",
        ),
        (
            "module attributes {func.payload = "
            "dense<0> : tensor<1xi32, #smt.encoding>} {}",
            "AIR-LINALG-003",
        ),
        (
            "module attributes {func.payload = "
            "dense<0> : tensor<1x!tt.ptr<i32>>} {}",
            "AIR-PARSE-001",
        ),
    ],
    ids=(
        "policy",
        "parse",
        "verifier",
        "dense-typed-attribute",
        "native-parser-assertion",
    ),
)
def test_invalid_ir_has_no_public_or_native_normalized_artifact(
    invalid,
    expected_code,
):
    public = _normalize(invalid)

    assert not public.acceptable
    assert not public.validation_report.valid
    assert public.validation_report.diagnostics[0].code == expected_code
    assert public.normalized_text is None
    assert public.normalized_bytes is None
    assert public.sha256 is None
    assert public.to_dict()["acceptable"] is False

    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    policy = resolve_anchor_ir_policy(
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    )
    native = anchor.normalize_anchor_ir_text(
        invalid,
        context,
        policy.to_dict(),
        "native-invalid.mlir",
    )
    assert native["validation_report"]["valid"] is False
    assert native["validation_report"]["diagnostics"][0]["code"] == expected_code
    assert native["normalized_text"] is None


def test_native_dense_parser_abort_cannot_produce_a_golden_artifact():
    result = _normalize(
        "module attributes {func.payload = "
        "dense<0> : tensor<1x!tt.ptr<i32>>} {}"
    )

    assert not result.acceptable
    assert [item.code for item in result.validation_report.diagnostics] == [
        "AIR-PARSE-001"
    ]
    assert result.normalized_text is None
    assert result.normalized_bytes is None
    assert result.sha256 is None


def test_text_input_limit_cannot_produce_a_golden_artifact(monkeypatch):
    monkeypatch.setattr(text_isolation, "MAX_ANCHOR_IR_TEXT_BYTES", 16)
    kwargs = {
        "normalization_version": ANCHOR_IR_NORMALIZATION_VERSION,
        "spec_version": ANCHOR_IR_SPEC_VERSION,
        "track": AnchorIRTrack.LINALG,
        "phase": AnchorIRPhase.PRE_HOOK,
        "source_name": "too-large.mlir",
    }
    default_result = AnchorIRNormalizer().normalize_text("测" * 6, **kwargs)
    explicit_result = AnchorIRNormalizer().normalize_text(
        "测" * 6,
        context=object(),
        **kwargs,
    )

    assert default_result.to_dict() == explicit_result.to_dict()
    assert not default_result.acceptable
    assert [item.code for item in default_result.validation_report.diagnostics] == [
        "AIR-PARSE-001"
    ]
    assert default_result.normalized_text is None
    assert default_result.normalized_bytes is None
    assert default_result.sha256 is None


def test_source_name_limit_precedes_normalizer_worker_and_native_parser(monkeypatch):
    monkeypatch.setattr(text_isolation, "MAX_ANCHOR_IR_SOURCE_NAME_BYTES", 8)

    def worker_must_not_run(*_args, **_kwargs):
        raise AssertionError("oversized source_name reached normalization")

    monkeypatch.setattr(text_isolation, "run_isolated_native_text", worker_must_not_run)
    kwargs = {
        "normalization_version": ANCHOR_IR_NORMALIZATION_VERSION,
        "spec_version": ANCHOR_IR_SPEC_VERSION,
        "track": AnchorIRTrack.LINALG,
        "phase": AnchorIRPhase.PRE_HOOK,
        "source_name": "012345678",
    }
    normalizer = AnchorIRNormalizer()

    with pytest.raises(ValueError, match="source_name exceeds the 8-byte"):
        normalizer.normalize_text(SCALAR_IR, **kwargs)
    with pytest.raises(ValueError, match="source_name exceeds the 8-byte"):
        normalizer.normalize_text(SCALAR_IR, context=object(), **kwargs)


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
def test_other_pinned_parser_aborts_cannot_produce_golden_artifacts(text):
    result = _normalize(text, track=AnchorIRTrack.TRITON_GPU)

    assert not result.acceptable
    assert [item.code for item in result.validation_report.diagnostics] == [
        "AIR-PARSE-001"
    ]
    assert result.normalized_text is None
    assert result.normalized_bytes is None
    assert result.sha256 is None


def test_normalization_version_is_explicit_and_rejects_unknown_versions():
    with pytest.raises(ValueError, match="unsupported"):
        AnchorIRNormalizer().normalize_text(
            SCALAR_IR,
            normalization_version="anchor-ir-normalization/9.9.9",
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            phase=AnchorIRPhase.PRE_HOOK,
        )
