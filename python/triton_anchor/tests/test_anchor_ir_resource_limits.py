"""Regression coverage for bounded structured AnchorIR traversal."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from triton._C.libtriton import anchor, ir

from triton_anchor import (
    ANCHOR_IR_NORMALIZATION_VERSION,
    ANCHOR_IR_SPEC_VERSION,
    AnchorIRGoldenBuilder,
    AnchorIRGoldenValidationError,
    AnchorIRLifecycleOrchestrator,
    AnchorIRNormalizer,
    AnchorIRPhase,
    AnchorIRStageId,
    AnchorIRTrack,
    AnchorIRValidationError,
    StructuredAnchorIRValidator,
)


def _nested_scf_text(depth: int) -> str:
    return (
        "module {\n  func.func @nested(%cond: i1) {\n"
        + "    scf.if %cond {\n" * depth
        + "    }\n" * depth
        + "    func.return\n  }\n}\n"
    )


def _nested_unknown_text(depth: int) -> str:
    return (
        "module {\n"
        + '  "vendor.deep"() ({\n' * depth
        + "  }) : () -> ()\n" * depth
        + "}\n"
    )


def _nested_array_text(depth: int) -> str:
    value = "0 : i32"
    for _ in range(depth):
        value = "[%s]" % value
    return "module attributes {func.container = %s} {}\n" % value


def _shared_tuple_dag_text(depth: int) -> str:
    """Build an exponentially shared Type DAG in a small textual module."""

    aliases = ["!t0 = tuple<i32, i32>"]
    aliases.extend(
        "!t%d = tuple<!t%d, !t%d>" % (index, index - 1, index - 1)
        for index in range(1, depth + 1)
    )
    return "\n".join(
        [
            *aliases,
            "module {",
            "  func.func @shared(%%arg: !t%d) {" % depth,
            "    func.return",
            "  }",
            "}",
        ]
    ) + "\n"


def _shared_attribute_dag_text(depth: int) -> str:
    """Build the Attribute counterpart of ``_shared_tuple_dag_text``."""

    aliases = ["#a0 = [0 : i32]"]
    aliases.extend(
        "#a%d = [#a%d, #a%d]" % (index, index - 1, index - 1)
        for index in range(1, depth + 1)
    )
    return "\n".join(
        [*aliases, "module attributes {func.payload = #a%d} {}" % depth]
    ) + "\n"


def _wide_forbidden_attribute_text(count: int) -> str:
    values = ", ".join("#smt.bad" for _ in range(count))
    return "module attributes {func.payload = [%s]} {}\n" % values


def _validate(text: str, *, context=None, source_name: str):
    return StructuredAnchorIRValidator().validate_text(
        text,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        context=context,
        source_name=source_name,
    )


def _assert_resource_limit(report, *, source_name: str, object_name: str) -> None:
    assert not report.valid
    assert [item.code for item in report.diagnostics] == ["AIR-COMMON-004"]
    diagnostic = report.diagnostics[0]
    assert diagnostic.object_name == object_name
    assert diagnostic.operation_path == "builtin.module"
    assert diagnostic.object_path == ""
    assert diagnostic.hint
    assert diagnostic.location is not None
    assert diagnostic.location.file == source_name


def test_text_nesting_limit_is_identical_for_default_explicit_and_cli(tmp_path):
    source = tmp_path / "too-deep.mlir"
    source.write_text(_nested_scf_text(500), encoding="utf-8")

    default_report = _validate(source.read_text(encoding="utf-8"), source_name=str(source))
    explicit_report = _validate(
        source.read_text(encoding="utf-8"),
        context=ir.context(),
        source_name=str(source),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "triton_anchor.anchor_ir_cli",
            str(source),
            "--spec-version",
            ANCHOR_IR_SPEC_VERSION,
            "--track",
            AnchorIRTrack.LINALG.value,
            "--phase",
            AnchorIRPhase.PRE_HOOK.value,
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    _assert_resource_limit(
        default_report,
        source_name=str(source),
        object_name="text nesting depth",
    )
    assert explicit_report.to_dict() == default_report.to_dict()
    assert completed.returncode == 1
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == default_report.to_dict()


def test_array_nesting_limit_protects_explicit_context_and_cli(tmp_path):
    source = tmp_path / "too-deep-array.mlir"
    source.write_text(_nested_array_text(10_000), encoding="utf-8")
    text = source.read_text(encoding="utf-8")

    default_report = _validate(text, source_name=str(source))
    explicit_report = _validate(
        text,
        context=ir.context(),
        source_name=str(source),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "triton_anchor.anchor_ir_cli",
            str(source),
            "--spec-version",
            ANCHOR_IR_SPEC_VERSION,
            "--track",
            AnchorIRTrack.LINALG.value,
            "--phase",
            AnchorIRPhase.PRE_HOOK.value,
            "--format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    _assert_resource_limit(
        default_report,
        source_name=str(source),
        object_name="text nesting depth",
    )
    assert explicit_report.to_dict() == default_report.to_dict()
    assert completed.returncode == 1
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == default_report.to_dict()


def test_deep_unknown_text_stops_before_recursive_operation_paths():
    source_name = "nested-unknown.mlir"
    text = _nested_unknown_text(500)

    default_report = _validate(text, source_name=source_name)
    explicit_report = _validate(text, context=ir.context(), source_name=source_name)

    _assert_resource_limit(
        default_report,
        source_name=source_name,
        object_name="text nesting depth",
    )
    assert explicit_report.to_dict() == default_report.to_dict()
    assert len(default_report.diagnostics[0].operation_path) < 64


def test_module_op_nesting_limit_precedes_recursive_validator(tmp_path):
    source = tmp_path / "module-too-deep.mlir"
    source.write_text(_nested_scf_text(500), encoding="utf-8")
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)

    report = StructuredAnchorIRValidator().validate_module(
        module,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    )

    assert not report.valid
    assert [item.code for item in report.diagnostics] == ["AIR-COMMON-004"]
    diagnostic = report.diagnostics[0]
    assert diagnostic.object_name == "operation nesting depth"
    assert diagnostic.operation_path == "builtin.module"
    assert len(diagnostic.operation_path) < 64


def test_normal_nested_ir_remains_valid_in_text_and_module_entrypoints(tmp_path):
    source = tmp_path / "nested-ok.mlir"
    source.write_text(_nested_scf_text(64), encoding="utf-8")
    text = source.read_text(encoding="utf-8")

    default_report = _validate(text, source_name=str(source))
    explicit_report = _validate(text, context=ir.context(), source_name=str(source))
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    module_report = StructuredAnchorIRValidator().validate_module(
        ir.parse_mlir_module(str(source), context),
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    )

    assert default_report.valid
    assert explicit_report.to_dict() == default_report.to_dict()
    assert module_report.valid


def test_nesting_boundary_is_consistent_between_text_and_module_entrypoints(tmp_path):
    for depth in (254, 255):
        source = tmp_path / ("nested-boundary-%d.mlir" % depth)
        source.write_text(_nested_scf_text(depth), encoding="utf-8")
        text = source.read_text(encoding="utf-8")

        context = ir.context()
        ir.load_dialects(context)
        anchor.load_dialects(context)
        default_report = _validate(text, source_name=str(source))
        explicit_report = _validate(text, context=context, source_name=str(source))
        module_report = StructuredAnchorIRValidator().validate_module(
            ir.parse_mlir_module(str(source), context),
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            phase=AnchorIRPhase.PRE_HOOK,
        )

        if depth == 254:
            assert default_report.valid
            assert explicit_report.to_dict() == default_report.to_dict()
            assert module_report.valid
        else:
            assert [item.code for item in default_report.diagnostics] == [
                "AIR-COMMON-004"
            ]
            assert default_report.diagnostics[0].object_name == "text nesting depth"
            assert explicit_report.to_dict() == default_report.to_dict()
            assert [item.code for item in module_report.diagnostics] == [
                "AIR-COMMON-004"
            ]
            assert module_report.diagnostics[0].object_name == "operation nesting depth"


def test_operation_count_limit_stops_wide_text_before_diagnostic_fanout():
    text = "module {\n" + '  "vendor.wide"() : () -> ()\n' * 16385 + "}\n"
    report = _validate(text, source_name="too-wide.mlir")

    _assert_resource_limit(
        report,
        source_name="too-wide.mlir",
        object_name="operation count",
    )


@pytest.mark.parametrize(
    "text_builder",
    (_shared_tuple_dag_text, _shared_attribute_dag_text),
    ids=("type-dag", "attribute-dag"),
)
def test_shared_object_dag_has_a_module_wide_traversal_budget(
    text_builder,
    tmp_path,
):
    # Depth 20 has fewer than 600 source bytes but more than one million
    # logical visits without a global budget.  It must be rejected equally by
    # the isolated/default, explicit-context, and parsed-ModuleOp entrypoints.
    source = tmp_path / "shared-dag.mlir"
    source.write_text(text_builder(20), encoding="utf-8")
    text = source.read_text(encoding="utf-8")

    default_report = _validate(text, source_name=str(source))
    explicit_report = _validate(
        text,
        context=ir.context(),
        source_name=str(source),
    )
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    module_report = StructuredAnchorIRValidator().validate_module(
        ir.parse_mlir_module(str(source), context),
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
    )

    for report in (default_report, explicit_report, module_report):
        assert [item.code for item in report.diagnostics] == ["AIR-COMMON-004"]
        assert report.diagnostics[0].object_name == "type/attribute traversal count"
        assert report.diagnostics[0].operation_path == "builtin.module"
    assert explicit_report.to_dict() == default_report.to_dict()
    assert module_report.to_dict() == default_report.to_dict()


def test_diagnostic_budget_has_a_stable_limit_report():
    text = "module {\n" + '  "vendor.invalid"() : () -> ()\n' * 8193 + "}\n"
    report = _validate(
        text,
        context=ir.context(),
        source_name="too-many-diagnostics.mlir",
    )

    assert len(report.diagnostics) == 8192
    limits = [
        item
        for item in report.diagnostics
        if item.code == "AIR-COMMON-004" and item.object_name == "diagnostic count"
    ]
    assert len(limits) == 1
    assert limits[0].operation_path == "builtin.module"


def test_resource_limited_ir_cannot_reach_hook_lowering_or_golden():
    text = _nested_array_text(10_000)

    class RecordingHook:
        def __init__(self):
            self.calls = 0

        def on_anchor_ir_ready(self, anchor_ir):
            self.calls += 1
            return anchor_ir

    hook = RecordingHook()
    lowered = []
    with pytest.raises(AnchorIRValidationError) as captured:
        AnchorIRLifecycleOrchestrator().run_text_or_raise(
            text,
            hook=hook,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            source_name="resource-limited.mlir",
            backend_lowering=lowered.append,
        )

    assert [item.code for item in captured.value.report.diagnostics] == [
        "AIR-COMMON-004"
    ]
    assert hook.calls == 0
    assert lowered == []

    normalized = AnchorIRNormalizer().normalize_text(
        text,
        normalization_version=ANCHOR_IR_NORMALIZATION_VERSION,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.PRE_HOOK,
        source_name="resource-limited.mlir",
    )
    assert not normalized.acceptable
    assert normalized.normalized_text is None
    assert normalized.sha256 is None

    builder = AnchorIRGoldenBuilder(case_id="resource/limited")
    with pytest.raises(AnchorIRGoldenValidationError) as golden_error:
        builder.add_text(
            AnchorIRStageId.adapter_output(),
            text,
            source_name="resource-limited.mlir",
        )
    assert [item.code for item in golden_error.value.report.diagnostics] == [
        "AIR-COMMON-004"
    ]


def test_wide_attribute_diagnostics_stop_at_the_budget():
    text = _wide_forbidden_attribute_text(9_000)
    default_report = _validate(text, source_name="wide-attribute.mlir")
    explicit_report = _validate(
        text,
        context=ir.context(),
        source_name="wide-attribute.mlir",
    )

    assert len(default_report.diagnostics) == 8192
    assert sum(
        item.code == "AIR-COMMON-004"
        and item.object_name == "diagnostic count"
        for item in default_report.diagnostics
    ) == 1
    assert explicit_report.to_dict() == default_report.to_dict()
