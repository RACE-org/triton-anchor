"""Acceptance tests for the strict AnchorIR backend Hook lifecycle."""

import os
import tempfile

import pytest
from triton._C.libtriton import anchor, ir

from triton_anchor import (
    ANCHOR_IR_SPEC_VERSION,
    AnchorIRLifecycleOrchestrator,
    AnchorIRLifecycleReport,
    AnchorIRPhase,
    AnchorIRTrack,
    AnchorIRValidationError,
    StructuredAnchorIRValidator,
)

VALID_IR = """
module {
  func.func @kernel() {
    %zero = arith.constant 0 : i32
    func.return
  }
}
"""

VALID_GPU_IR = """
module attributes {
  "triton_gpu.num-warps" = 1 : i32,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  tt.make_range {start = 0 : i32, end = 16 : i32}
      : tensor<16xi32, #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>>
}
"""


class TextHook:
    def __init__(self, replacement, allowed=()):
        self.replacement = replacement
        self.allowed = allowed
        self.calls = 0

    def get_allowed_dialects(self):
        return self.allowed

    def on_anchor_ir_ready(self, anchor_ir):
        self.calls += 1
        return self.replacement(anchor_ir)


def _run(text, hook):
    return AnchorIRLifecycleOrchestrator().run_text(
        text,
        hook=hook,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        source_name="lifecycle.mlir",
    )


def test_pre_hook_failure_short_circuits_backend_hook():
    hook = TextHook(lambda text: VALID_IR)
    lowered = []

    report = AnchorIRLifecycleOrchestrator().run_text(
        'module { "smt.pre_failure"() : () -> () }',
        hook=hook,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        backend_lowering=lowered.append,
    )

    assert not report.valid
    assert not report.pre_hook.valid
    assert report.post_hook is None
    assert not report.hook_executed
    assert hook.calls == 0
    assert not report.lowering_executed
    assert lowered == []
    assert [item.code for item in report.pre_hook.diagnostics] == ["AIR-LINALG-001"]


@pytest.mark.parametrize(
    "track, invalid_ir, expected_code",
    [
        (
            AnchorIRTrack.LINALG,
            'module { ".tt.hidden"() : () -> () }',
            "AIR-COMMON-001",
        ),
        (
            AnchorIRTrack.TRITON_GPU,
            'module { ".tt.hidden"() : () -> () }',
            "AIR-COMMON-001",
        ),
        (
            AnchorIRTrack.LINALG,
            "module attributes {func.container = {smt.marker}} {}",
            "AIR-LINALG-003",
        ),
        (
            AnchorIRTrack.TRITON_GPU,
            "module attributes {func.container = {smt.marker}} {}",
            "AIR-GPU-003",
        ),
    ],
)
def test_namespace_and_dictionary_regressions_fail_closed_before_hook(
    track,
    invalid_ir,
    expected_code,
):
    hook = TextHook(lambda text: text)
    lowered = []

    with pytest.raises(AnchorIRValidationError) as captured:
        AnchorIRLifecycleOrchestrator().run_text_or_raise(
            invalid_ir,
            hook=hook,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=track,
            source_name="lifecycle-regression.mlir",
            backend_lowering=lowered.append,
        )

    assert [item.code for item in captured.value.report.diagnostics] == [
        expected_code
    ]
    assert hook.calls == 0
    assert lowered == []


def test_wide_gpu_configuration_fails_closed_without_aborting():
    hook = TextHook(lambda text: text)
    lowered = []

    with pytest.raises(AnchorIRValidationError) as captured:
        AnchorIRLifecycleOrchestrator().run_text_or_raise(
            """
module attributes {
  "triton_gpu.num-warps" = 18446744073709551615 : i65,
  "triton_gpu.threads-per-warp" = 32 : i32,
  "triton_gpu.num-ctas" = 1 : i32
} {
  tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
}
""",
            hook=hook,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.TRITON_GPU,
            source_name="wide-gpu-config.mlir",
            backend_lowering=lowered.append,
        )

    assert "AIR-GPU-011" in [
        item.code for item in captured.value.report.diagnostics
    ]
    assert hook.calls == 0
    assert lowered == []


def test_pre_hook_uses_only_core_rules_not_declared_extensions():
    hook = TextHook(lambda text: text, allowed={"backend_ext"})

    report = _run('module { "backend_ext.before_hook"() : () -> () }', hook)

    assert not report.pre_hook.valid
    assert [item.code for item in report.pre_hook.diagnostics] == ["AIR-COMMON-001"]
    assert hook.calls == 0


def test_undeclared_hook_dialect_fails_post_hook():
    hook = TextHook(lambda text: 'module { "vendor.injected"() : () -> () }')
    lowered = []

    report = AnchorIRLifecycleOrchestrator().run_text(
        VALID_IR,
        hook=hook,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        backend_lowering=lowered.append,
    )

    assert report.pre_hook.valid
    assert report.hook_executed
    assert hook.calls == 1
    assert report.post_hook is not None and not report.post_hook.valid
    assert [item.code for item in report.post_hook.diagnostics] == ["AIR-COMMON-001"]
    assert not report.lowering_executed
    assert lowered == []


def test_declared_extension_passes_post_hook():
    hook = TextHook(
        lambda text: 'module { "backend_ext.injected"() : () -> () }',
        allowed={"backend_ext"},
    )

    report = _run(VALID_IR, hook)

    assert report.valid
    assert report.pre_hook.valid
    assert report.post_hook is not None and report.post_hook.valid
    assert report.declared_extensions == ("backend_ext",)


def test_declared_extension_cannot_hide_forbidden_attribute_in_properties():
    hook = TextHook(
        lambda text: (
            'module { "backend_ext.injected"() '
            "<{payload = #smt.marker}> : () -> () }"
        ),
        allowed={"backend_ext"},
    )
    lowered = []

    report = AnchorIRLifecycleOrchestrator().run_text(
        VALID_IR,
        hook=hook,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        backend_lowering=lowered.append,
    )

    assert report.pre_hook.valid
    assert report.hook_executed
    assert report.post_hook is not None and not report.post_hook.valid
    assert any(
        diagnostic.code == "AIR-LINALG-003"
        and diagnostic.object_path == "properties.entry[payload]"
        for diagnostic in report.post_hook.diagnostics
    )
    assert not report.lowering_executed
    assert lowered == []


def test_backend_lowering_runs_only_after_successful_post_hook():
    lowered = []
    report = AnchorIRLifecycleOrchestrator().run_text(
        VALID_IR,
        hook=TextHook(lambda text: text),
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        backend_lowering=lambda text: lowered.append(text) or "lowered",
    )

    assert report.valid
    assert report.lowering_executed
    assert report.lowered_output == "lowered"
    assert lowered == [VALID_IR]


def test_text_lifecycle_rejects_non_callable_backend_lowering_before_hook():
    hook = TextHook(lambda text: text)

    with pytest.raises(TypeError, match="backend_lowering must be callable"):
        AnchorIRLifecycleOrchestrator().run_text(
            VALID_IR,
            hook=hook,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            backend_lowering=object(),
        )

    assert hook.calls == 0


@pytest.mark.parametrize("method_name", ["on_anchor_ir_ready", "get_allowed_dialects"])
def test_text_lifecycle_rejects_non_callable_hook_methods(method_name):
    class InvalidHook:
        on_anchor_ir_ready = 1

        def get_allowed_dialects(self):
            return ()

    hook = InvalidHook()
    setattr(hook, method_name, 1)
    lowered = []

    expected = (
        "callable on_anchor_ir_ready"
        if method_name == "on_anchor_ir_ready"
        else "get_allowed_dialects must be callable"
    )
    with pytest.raises(TypeError, match=expected):
        AnchorIRLifecycleOrchestrator().run_text(
            VALID_IR,
            hook=hook,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            backend_lowering=lowered.append,
        )

    assert lowered == []


def test_declaring_a_core_allowed_namespace_as_an_extension_is_rejected():
    hook = TextHook(lambda text: text, allowed={"gpu"})

    with pytest.raises(ValueError, match="redeclare core dialect.*gpu"):
        AnchorIRLifecycleOrchestrator().run_text_or_raise(
            VALID_IR,
            hook=hook,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.TRITON_GPU,
        )

    assert hook.calls == 0


def test_none_hook_result_means_in_place_mutation_contract():
    class InPlaceHook:
        def __init__(self):
            self.calls = 0

        def on_anchor_ir_ready(self, anchor_ir):
            self.calls += 1
            return None

    hook = InPlaceHook()
    report = _run(VALID_IR, hook)

    assert report.valid
    assert report.output == VALID_IR
    assert hook.calls == 1


@pytest.mark.parametrize("track, valid_ir", [
    (AnchorIRTrack.LINALG, VALID_IR),
    (AnchorIRTrack.TRITON_GPU, VALID_GPU_IR),
])
def test_extension_declaration_cannot_override_core_forbidden_namespace(
    track,
    valid_ir,
):
    hook = TextHook(
        lambda text: 'module { "smt.injected"() : () -> () }',
        allowed={"smt"},
    )
    lowered = []

    with pytest.raises(ValueError, match="core-forbidden.*smt"):
        AnchorIRLifecycleOrchestrator().run_text_or_raise(
            valid_ir,
            hook=hook,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=track,
            backend_lowering=lowered.append,
        )

    assert hook.calls == 0
    assert lowered == []


def test_linalg_extension_cannot_alias_registered_nvidia_core_namespace():
    hook = TextHook(
        lambda text: 'module { "triton_nvidia_gpu.injected"() : () -> () }',
        allowed={"triton_nvidia_gpu"},
    )
    lowered = []

    with pytest.raises(
        ValueError,
        match="core-forbidden.*triton_nvidia_gpu",
    ):
        AnchorIRLifecycleOrchestrator().run_text_or_raise(
            VALID_IR,
            hook=hook,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            backend_lowering=lowered.append,
        )

    assert hook.calls == 0
    assert lowered == []


def test_lifecycle_does_not_accept_a_caller_supplied_validator():
    with pytest.raises(TypeError, match="validator"):
        AnchorIRLifecycleOrchestrator(validator=object())


def test_module_boundary_snapshot_and_output_survive_in_place_lowering():
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", encoding="utf-8", delete=False
    ) as source:
        source.write(VALID_IR)
        source_path = source.name
    try:
        module = ir.parse_mlir_module(source_path, context)
    finally:
        os.unlink(source_path)

    lowered = []

    def mutate_lowering(lowering_module):
        lowering_module.set_attr("smt.marker", ir.make_attr([1], context))
        lowered.append(lowering_module)
        return lowering_module

    report = AnchorIRLifecycleOrchestrator().run_module(
        module,
        hook=TextHook(lambda value: value),
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        context=context,
        backend_lowering=mutate_lowering,
    )

    assert report.valid
    assert report.post_hook_snapshot is not None
    assert report.post_hook_snapshot.acceptable
    assert "smt.marker" not in report.post_hook_snapshot.normalized_text
    assert report.output is module
    assert report.lowered_output is lowered[0]
    assert report.lowered_output is not report.output
    assert StructuredAnchorIRValidator().validate_module(
        report.output,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.POST_HOOK,
    ).valid
    assert not StructuredAnchorIRValidator().validate_module(
        report.lowered_output,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        phase=AnchorIRPhase.POST_HOOK,
    ).valid


def test_post_hook_revalidates_existing_operations_not_only_additions():
    hook = TextHook(
        lambda text: text.replace(
            "func.func @kernel() {",
            "func.func @kernel() attributes {smt.marker = true} {",
        )
    )

    report = _run(VALID_IR, hook)

    assert report.pre_hook.valid
    assert report.post_hook is not None and not report.post_hook.valid
    diagnostic = report.post_hook.diagnostics[0]
    assert diagnostic.code == "AIR-LINALG-003"
    assert any(
        item.object_path == "attribute[smt.marker]"
        for item in report.post_hook.diagnostics
    )


def test_module_api_is_available_to_external_backends():
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", encoding="utf-8", delete=False
    ) as source:
        source.write(VALID_IR)
        source_path = source.name
    try:
        module = ir.parse_mlir_module(source_path, context)
    finally:
        os.unlink(source_path)

    hook = TextHook(lambda value: value)
    report = AnchorIRLifecycleOrchestrator().run_module(
        module,
        hook=hook,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        context=context,
    )

    assert report.valid
    assert report.hook_executed
    assert hook.calls == 1
    assert report.output.context is context


def test_module_lifecycle_report_retains_context_after_caller_reference_is_dropped():
    import gc

    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", encoding="utf-8", delete=False
    ) as source:
        source.write(VALID_IR)
        source_path = source.name
    try:
        module = ir.parse_mlir_module(source_path, context)
    finally:
        os.unlink(source_path)

    report = AnchorIRLifecycleOrchestrator().run_module_or_raise(
        module,
        hook=None,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        context=context,
    )
    assert report.valid
    assert report.output.context is context
    del module
    del context
    gc.collect()
    assert "func.func @kernel" in str(report.output)


def test_gpu_track_uses_the_same_complete_lifecycle():
    report = AnchorIRLifecycleOrchestrator().run_text(
        VALID_GPU_IR,
        hook=None,
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.TRITON_GPU,
    )

    assert report.valid
    assert report.pre_hook.valid
    assert report.post_hook is not None and report.post_hook.valid
    assert not report.hook_executed


def test_incomplete_lifecycle_report_is_not_valid():
    complete = _run(VALID_IR, TextHook(lambda text: text))
    incomplete = AnchorIRLifecycleReport(
        output=complete.output,
        pre_hook=complete.pre_hook,
        post_hook=None,
        hook_executed=False,
        declared_extensions=(),
    )

    assert complete.valid
    assert not incomplete.valid


def test_strict_validator_rejects_ambiguous_extension_declaration_input():
    from triton_anchor import AnchorIRPhase, StructuredAnchorIRValidator

    with pytest.raises(ValueError, match="not a string"):
        StructuredAnchorIRValidator().validate_text(
            VALID_IR,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            phase=AnchorIRPhase.POST_HOOK,
            extension_dialects="backend_ext",
        )


def test_hook_rejects_invalid_dialect_namespace_declarations():
    hook = TextHook(lambda text: text, allowed={"backend.ext"})

    with pytest.raises(TypeError, match="valid MLIR dialect namespaces"):
        _run(VALID_IR, hook)


def test_run_text_or_raise_fails_closed_before_hook_with_actionable_report():
    hook = TextHook(lambda text: VALID_IR)
    lowered = []

    with pytest.raises(AnchorIRValidationError) as captured:
        AnchorIRLifecycleOrchestrator().run_text_or_raise(
            'module { "smt.pre_failure"() : () -> () }',
            hook=hook,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            source_name="pre-invalid.mlir",
            backend_lowering=lowered.append,
        )

    error = captured.value
    assert error.report.phase is AnchorIRPhase.PRE_HOOK
    assert [item.code for item in error.report.diagnostics] == ["AIR-LINALG-001"]
    assert hook.calls == 0
    assert lowered == []
    rendered = str(error)
    for expected in (
        "AnchorIR validation: FAIL",
        "[AIR-LINALG-001]",
        "operation_path: builtin.module",
        "location: pre-invalid.mlir:",
        "hint:",
    ):
        assert expected in rendered


def test_run_text_or_raise_fails_closed_after_hook_with_object_path():
    hook = TextHook(
        lambda text: text.replace(
            "func.func @kernel() {",
            "func.func @kernel() attributes {smt.marker = true} {",
        )
    )
    lowered = []

    with pytest.raises(AnchorIRValidationError) as captured:
        AnchorIRLifecycleOrchestrator().run_text_or_raise(
            VALID_IR,
            hook=hook,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            source_name="post-invalid.mlir",
            backend_lowering=lowered.append,
        )

    error = captured.value
    assert error.report.phase is AnchorIRPhase.POST_HOOK
    assert [item.code for item in error.report.diagnostics] == ["AIR-LINALG-003"]
    assert hook.calls == 1
    assert lowered == []
    rendered = str(error)
    assert "object_path: attribute[smt.marker]" in rendered
    assert "location: post-invalid.mlir:" in rendered
    assert "hint:" in rendered


def test_run_module_or_raise_rejects_invalid_real_module_before_hook():
    context = ir.context()
    ir.load_dialects(context)
    anchor.load_dialects(context)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", encoding="utf-8", delete=False
    ) as source:
        source.write(VALID_GPU_IR)
        source_path = source.name
    try:
        module = ir.parse_mlir_module(source_path, context)
    finally:
        os.unlink(source_path)
    hook = TextHook(lambda value: value)

    with pytest.raises(AnchorIRValidationError) as captured:
        AnchorIRLifecycleOrchestrator().run_module_or_raise(
            module,
            hook=hook,
            spec_version=ANCHOR_IR_SPEC_VERSION,
            track=AnchorIRTrack.LINALG,
            context=context,
        )

    assert captured.value.report.phase is AnchorIRPhase.PRE_HOOK
    assert {diagnostic.code for diagnostic in captured.value.report.diagnostics} & {
        "AIR-LINALG-001",
        "AIR-LINALG-003",
    }
    assert hook.calls == 0


def test_run_text_or_raise_returns_complete_successful_lifecycle():
    lowered = []
    report = AnchorIRLifecycleOrchestrator().run_text_or_raise(
        VALID_IR,
        hook=TextHook(lambda text: text),
        spec_version=ANCHOR_IR_SPEC_VERSION,
        track=AnchorIRTrack.LINALG,
        backend_lowering=lambda text: lowered.append(text) or "lowered",
    )

    assert report.valid
    assert report.lowering_executed
    assert report.lowered_output == "lowered"
    assert lowered == [VALID_IR]
