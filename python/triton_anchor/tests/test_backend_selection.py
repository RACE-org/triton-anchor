"""Focused W8 tests for pure, deterministic backend selection."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from triton_anchor.backends.errors import (
    BackendPluginCapabilityError,
    BackendPluginConflictError,
    BackendPluginSelectionError,
)
from triton_anchor.backends.selection import (
    BACKEND_SELECTOR_ENV,
    SelectionMethod,
    select_backend,
)


class SpyEntryPoint:
    def __init__(self):
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        raise AssertionError("selection must never load an entry point")


@dataclass
class Record:
    record_id: str
    entry_point_name: str
    manifest: object = None
    source: str = "manifest"
    state: str = "validated"
    compatibility_status: str = "compatible"
    registry_key_override: str = ""

    def __post_init__(self):
        self.entry_point = SpyEntryPoint()

    @property
    def plugin_id(self):
        return self.manifest.plugin_id if self.manifest is not None else None

    @property
    def registry_key(self):
        if self.registry_key_override:
            return self.registry_key_override
        return self.plugin_id or f"legacy:vendor:{self.entry_point_name}"


def manifest(
    plugin_id,
    entry_point,
    *,
    targets=("mock",),
    capabilities=(),
    requires_capabilities=(),
    priority=0,
):
    return SimpleNamespace(
        plugin_id=plugin_id,
        entry_point=entry_point,
        targets=tuple(targets),
        capabilities=tuple(capabilities),
        requires_capabilities=tuple(requires_capabilities),
        priority=priority,
    )


def record(
    name,
    *,
    plugin_id=None,
    targets=("mock",),
    capabilities=(),
    requires_capabilities=(),
    priority=0,
):
    return Record(
        record_id=f"vendor-{name}:{name}",
        entry_point_name=name,
        manifest=manifest(
            plugin_id or f"vendor.{name}",
            name,
            targets=targets,
            capabilities=capabilities,
            requires_capabilities=requires_capabilities,
            priority=priority,
        ),
    )


def legacy(name="old"):
    return Record(
        record_id=f"vendor-legacy:{name}",
        entry_point_name=name,
        manifest=None,
        source="legacy",
        state="discovered",
        compatibility_status="legacy_unverified",
        registry_key_override=f"legacy:vendor-legacy:{name}",
    )


def assert_never_loaded(*records):
    assert [item.entry_point.load_calls for item in records] == [0] * len(records)


def test_python_selector_overrides_environment_and_manifest_priority():
    low = record("low", priority=1)
    high = record("high", priority=100)

    decision = select_backend(
        (high, low),
        target="mock",
        explicit_selector=low.plugin_id,
        # The lower-precedence value is deliberately malformed and must not
        # affect an explicit Python selection.
        environment={BACKEND_SELECTOR_ENV: object()},
    )

    assert decision.record is low
    assert decision.method is SelectionMethod.PYTHON_EXPLICIT
    assert decision.selector == low.plugin_id
    assert decision.candidate_record_ids == tuple(
        sorted((low.record_id, high.record_id))
    )
    assert_never_loaded(low, high)


def test_environment_selector_overrides_automatic_priority():
    selected = record("selected", priority=-10)
    other = record("other", priority=20)

    decision = select_backend(
        (other, selected),
        target=SimpleNamespace(backend="mock"),
        environment={BACKEND_SELECTOR_ENV: selected.plugin_id},
    )

    assert decision.record_id == selected.record_id
    assert decision.method is SelectionMethod.ENVIRONMENT
    assert_never_loaded(selected, other)


def test_sole_candidate_and_unique_highest_priority():
    only = record("only")
    sole = select_backend((only,), target={"backend": "mock"})
    assert sole.method is SelectionMethod.SOLE_CANDIDATE
    assert sole.record is only

    low = record("low", priority=3)
    high = record("high", priority=4)
    priority = select_backend((low, high), target="mock")
    assert priority.method is SelectionMethod.MANIFEST_PRIORITY
    assert priority.record is high
    assert_never_loaded(only, low, high)


def test_highest_priority_tie_is_deterministic_and_rejected():
    alpha = record("alpha", priority=7)
    beta = record("beta", priority=7)

    messages = []
    for records in ((alpha, beta), (beta, alpha)):
        with pytest.raises(BackendPluginSelectionError) as caught:
            select_backend(records, target="mock")
        messages.append(str(caught.value))
        assert caught.value.field == "priority"
        assert caught.value.actual == ", ".join(
            sorted((alpha.record_id, beta.record_id))
        )

    assert messages[0] == messages[1]
    assert_never_loaded(alpha, beta)


@pytest.mark.parametrize("duplicate", ["plugin_id", "entry_point"])
def test_fatal_identity_conflict_blocks_even_explicit_selection(duplicate):
    first = record("first", plugin_id="vendor.same")
    if duplicate == "plugin_id":
        second = record("second", plugin_id="vendor.same")
    else:
        second = Record(
            record_id="other:second",
            entry_point_name=first.entry_point_name,
            manifest=manifest(
                "vendor.second",
                first.entry_point_name,
            ),
        )

    with pytest.raises(BackendPluginConflictError):
        select_backend(
            (second, first),
            target="mock",
            explicit_selector=first.record_id,
        )

    assert_never_loaded(first, second)


def test_manifest_candidate_must_declare_requested_target():
    wrong = record("wrong", targets=("other",))

    with pytest.raises(BackendPluginSelectionError) as automatic:
        select_backend((wrong,), target="mock")
    assert automatic.value.field == "targets"

    with pytest.raises(BackendPluginSelectionError) as explicit:
        select_backend(
            (wrong,),
            target="mock",
            explicit_selector=wrong.plugin_id,
        )
    assert explicit.value.field == "targets"
    assert_never_loaded(wrong)


def test_capabilities_filter_candidates_and_are_recorded_in_decision():
    missing = record("missing", capabilities=("dtype.fp16",), priority=100)
    capable = record(
        "capable",
        capabilities=("runtime.launch",),
        requires_capabilities=("core.anchor_ir",),
    )

    decision = select_backend(
        (missing, capable),
        target="mock",
        core_provided_capabilities=("core.anchor_ir", "dtype.fp16"),
        kernel_required_capabilities=("runtime.launch",),
    )

    assert decision.record is capable
    assert decision.capability_report.compatible
    assert decision.capability_report.kernel_available == (
        "core.anchor_ir",
        "dtype.fp16",
        "runtime.launch",
    )
    assert decision.candidate_record_ids == (capable.record_id,)
    assert_never_loaded(missing, capable)


def test_only_capability_incompatible_candidate_reports_missing_names():
    plugin = record("limited", capabilities=("dtype.fp16",))

    with pytest.raises(BackendPluginCapabilityError) as caught:
        select_backend(
            (plugin,),
            target="mock",
            kernel_required_capabilities=(
                "runtime.launch",
                "dtype.bf16",
            ),
        )

    assert caught.value.missing_capabilities == (
        "dtype.bf16",
        "runtime.launch",
    )
    assert_never_loaded(plugin)


def test_legacy_requires_exact_explicit_record_key_and_no_kernel_requirements():
    old = legacy()

    with pytest.raises(BackendPluginSelectionError) as automatic:
        select_backend((old,), target="mock")
    assert "require explicit record_id/registry_key" in str(automatic.value)

    with pytest.raises(BackendPluginSelectionError):
        select_backend(
            (old,),
            target="mock",
            explicit_selector=old.entry_point_name,
        )

    by_id = select_backend(
        (old,),
        target="mock",
        explicit_selector=old.record_id,
    )
    assert by_id.record is old
    assert by_id.is_legacy
    assert by_id.capability_report is None

    by_key = select_backend(
        (old,),
        target="mock",
        environment={BACKEND_SELECTOR_ENV: old.registry_key},
    )
    assert by_key.record is old
    assert by_key.method is SelectionMethod.ENVIRONMENT

    with pytest.raises(BackendPluginSelectionError) as capabilities:
        select_backend(
            (old,),
            target="mock",
            explicit_selector=old.record_id,
            kernel_required_capabilities=("dtype.fp16",),
        )
    assert capabilities.value.field == "kernel_required_capabilities"
    assert_never_loaded(old)


def test_unvalidated_or_rejected_manifest_is_never_selected():
    unvalidated = record("unvalidated", priority=100)
    unvalidated.state = "discovered"
    valid = record("valid")

    decision = select_backend((unvalidated, valid), target="mock")
    assert decision.record is valid

    with pytest.raises(BackendPluginSelectionError) as caught:
        select_backend(
            (unvalidated, valid),
            target="mock",
            explicit_selector=unvalidated.plugin_id,
        )
    assert caught.value.field == "state"
    assert_never_loaded(unvalidated, valid)


@pytest.mark.parametrize(
    "state",
    ("validated", "loaded", "registered", "selected", "active"),
)
def test_validated_or_later_manifest_state_remains_selectable(state):
    plugin = record("reusable")
    plugin.state = state

    decision = select_backend((plugin,), target="mock")

    assert decision.record is plugin
    assert_never_loaded(plugin)


def test_decision_is_independent_of_input_enumeration_order():
    low = record("low", priority=1)
    high = record("high", priority=2)

    forward = select_backend((low, high), target="mock")
    reverse = select_backend((high, low), target="mock")

    assert forward == reverse
    assert forward.to_dict() == reverse.to_dict()
    assert_never_loaded(low, high)


def test_selection_decision_to_dict_has_stable_normalized_shape():
    plugin = record(
        "mock",
        capabilities=("runtime.launch",),
        priority=3,
    )

    decision = select_backend(
        (plugin,),
        target="mock",
        kernel_required_capabilities=("runtime.launch",),
    )
    payload = decision.to_dict()

    assert payload == {
        "record_id": "vendor-mock:mock",
        "registry_key": "vendor.mock",
        "plugin_id": "vendor.mock",
        "entry_point": "mock",
        "target": "mock",
        "method": "sole_candidate",
        "selector": None,
        "priority": 3,
        "is_legacy": False,
        "candidate_record_ids": ["vendor-mock:mock"],
        "capabilities": {
            "plugin_id": "vendor.mock",
            "entry_point": "mock",
            "compatible": True,
            "core_provided": [],
            "plugin_provided": ["runtime.launch"],
            "plugin_required": [],
            "kernel_required": ["runtime.launch"],
            "kernel_available": ["runtime.launch"],
            "missing": [],
            "missing_for_plugin": [],
            "missing_for_kernel": [],
        },
    }
    assert "record" not in payload
    assert json.loads(json.dumps(payload)) == payload
    assert_never_loaded(plugin)


@pytest.mark.parametrize(
    "kwargs,field",
    [
        ({"explicit_selector": ""}, "explicit_selector"),
        (
            {"environment": {BACKEND_SELECTOR_ENV: " vendor.mock"}},
            BACKEND_SELECTOR_ENV,
        ),
        (
            {"kernel_required_capabilities": "dtype.fp16"},
            "kernel_required_capabilities",
        ),
        (
            {"core_provided_capabilities": ("same", "same")},
            "core_provided_capabilities",
        ),
    ],
)
def test_invalid_selection_inputs_are_structured_errors(kwargs, field):
    plugin = record("mock")

    with pytest.raises(BackendPluginSelectionError) as caught:
        select_backend((plugin,), target="mock", **kwargs)

    assert caught.value.field == field
    assert caught.value.remediation
    assert_never_loaded(plugin)
