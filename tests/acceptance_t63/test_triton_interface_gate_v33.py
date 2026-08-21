"""Acceptance tests for the Triton 3.3 runtime-pair publication gate.

The required compiler and driver members are deliberately derived from the
checked-out Triton ABCs.  The tests do not use the three v3.3 delta names as an
implementation oracle, and they never instantiate a candidate while deciding
whether its static interface is publishable.
"""

from __future__ import annotations

import abc
import importlib
import importlib.util
import inspect
import json
import sys
import threading
import types
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import pytest

import triton_anchor.backends as backend_api
from triton_anchor.backends import (
    BackendPluginInterfaceError,
    BackendPluginLifecycleError,
    BackendPluginRegistry,
    PluginLifecycleState,
    collect_core_environment,
)


REPOSITORY = Path(__file__).resolve().parents[2]
TRITON_PACKAGE = REPOSITORY / "triton/python/triton"
TRITON_COMMIT = "523a1b235b213bc192f2d5a8999add5bf2d0fea5"


def _load_contract_module(name: str, path: Path) -> types.ModuleType:
    """Load one dependency-light ABC module without importing ``triton``."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_COMPILER_CONTRACT = _load_contract_module(
    "_t63_v33_compiler_contract",
    TRITON_PACKAGE / "backends/compiler.py",
).BaseBackend
_DRIVER_CONTRACT = _load_contract_module(
    "_t63_v33_driver_contract",
    TRITON_PACKAGE / "backends/driver.py",
).DriverBase
_CURRENT_MEMBER_CASES = tuple(
    (field, member)
    for field, contract in (
        ("compiler_cls", _COMPILER_CONTRACT),
        ("driver_cls", _DRIVER_CONTRACT),
    )
    for member in sorted(contract.__abstractmethods__)
)


def _descriptor_kind(owner: type, member: str) -> str:
    descriptor = inspect.getattr_static(owner, member)
    if isinstance(descriptor, property):
        return "property"
    if isinstance(descriptor, classmethod):
        return "classmethod"
    if isinstance(descriptor, staticmethod):
        return "staticmethod"
    if inspect.isfunction(descriptor):
        return "method"
    return type(descriptor).__name__


def _matching_implementation(
    descriptor: Any,
    *,
    calls: Counter[tuple[str, str]],
    key: tuple[str, str],
) -> Any:
    def implementation(*_args: Any, **_kwargs: Any) -> Any:
        calls[key] += 1
        if isinstance(descriptor, (classmethod, staticmethod)):
            return True
        return None

    if isinstance(descriptor, property):
        return property(implementation)
    if isinstance(descriptor, classmethod):
        return classmethod(implementation)
    if isinstance(descriptor, staticmethod):
        return staticmethod(implementation)
    return implementation


def _abstract_implementation(descriptor: Any) -> Any:
    def implementation(*_args: Any, **_kwargs: Any) -> None:
        return None

    implementation = abc.abstractmethod(implementation)
    if isinstance(descriptor, property):
        return property(implementation)
    if isinstance(descriptor, classmethod):
        return classmethod(implementation)
    if isinstance(descriptor, staticmethod):
        return staticmethod(implementation)
    return implementation


def _runtime_class(
    contract: type,
    field: str,
    name: str,
    *,
    calls: Counter[tuple[str, str]],
    structural: bool = False,
    omitted: Iterable[str] = (),
    overrides: Mapping[str, Any] | None = None,
    metaclass: type | None = None,
) -> type:
    omitted_members = frozenset(omitted)
    namespace: dict[str, Any] = {"__module__": __name__}
    for member in sorted(contract.__abstractmethods__):
        if member in omitted_members:
            continue
        descriptor = inspect.getattr_static(contract, member)
        namespace[member] = _matching_implementation(
            descriptor,
            calls=calls,
            key=(field, member),
        )
    namespace.update(dict(overrides or {}))

    def constructor(self: Any, *_args: Any, **_kwargs: Any) -> None:
        calls[(field, "__init__")] += 1

    namespace["__init__"] = constructor
    bases = (object,) if structural else (contract,)
    class_metaclass = metaclass or type(contract if not structural else object)
    return class_metaclass(name, bases, namespace)


@dataclass
class PluginFixture:
    plugin_id: str
    entry_point: str
    target: str
    plugin: Any
    distribution: Any
    entry_point_object: Any
    calls: Counter[Any]


class _EntryPoint:
    group = "triton.backends"

    def __init__(self, name: str, plugin: Any) -> None:
        self.name = name
        self.value = f"t63_f6_{name}:plugin"
        self.plugin = plugin
        self.load_calls = 0
        self.dist = None

    def load(self) -> Any:
        self.load_calls += 1
        return self.plugin


class _Distribution:
    def __init__(self, name: str, entry_point: _EntryPoint, manifest: Path):
        self.metadata = {"Name": name}
        self.name = name
        self.version = "1.0.0"
        self.entry_points = (entry_point,)
        entry_point.dist = self
        self.files = ("fixture/triton_anchor_backend.json",)
        self._manifest = manifest

    def locate_file(self, _file: Any) -> Path:
        return self._manifest

    def read_text(self, _filename: str) -> None:
        return None


@dataclass
class RuntimeHarness:
    registry: BackendPluginRegistry
    distributions: list[Any]
    backends_module: types.ModuleType
    runtime_driver_module: types.ModuleType
    compiler_contract: type
    driver_contract: type
    root: Path
    sequence: int = 0

    def fixture(
        self,
        *,
        label: str,
        compiler_cls: Any,
        driver_cls: Any,
        calls: Counter[Any] | None = None,
        target: str | None = None,
        priority: int = 0,
    ) -> PluginFixture:
        self.sequence += 1
        calls = calls if calls is not None else Counter()
        entry_point_name = f"{label}_{self.sequence}"
        target_name = target or entry_point_name
        plugin_id = f"acceptance.f6.{entry_point_name}"

        class Plugin:
            def initialize(self, context: Mapping[str, Any]) -> None:
                calls["initialize"] += 1
                calls["initialize_context"] = context

            def shutdown(self) -> None:
                calls["shutdown"] += 1

        plugin = Plugin()
        plugin.compiler_cls = compiler_cls
        plugin.driver_cls = driver_cls
        entry_point = _EntryPoint(entry_point_name, plugin)
        directory = self.root / entry_point_name
        directory.mkdir()
        manifest = directory / "triton_anchor_backend.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "plugins": [
                        {
                            "plugin_id": plugin_id,
                            "entry_point": entry_point_name,
                            "backend_protocol": ">=1.0,<2.0",
                            "requires_core": ">=0.2,<0.3",
                            "requires_triton": {
                                "version": ">=3.3,<3.4",
                                "commit": TRITON_COMMIT,
                            },
                            "targets": [target_name],
                            "capabilities": ["acceptance.f6"],
                            "isolation_mode": "python_only",
                            "priority": priority,
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        distribution = _Distribution(
            f"t63-f6-{entry_point_name}", entry_point, manifest
        )
        return PluginFixture(
            plugin_id=plugin_id,
            entry_point=entry_point_name,
            target=target_name,
            plugin=plugin,
            distribution=distribution,
            entry_point_object=entry_point,
            calls=calls,
        )

    def install(self, *fixtures: PluginFixture) -> None:
        errors = self.registry.reset()
        assert errors == ()
        self.distributions[:] = [fixture.distribution for fixture in fixtures]

    def complete_pair(
        self, *, structural: bool = False, label: str = "complete"
    ) -> tuple[type, type, Counter[Any]]:
        calls: Counter[Any] = Counter()
        compiler_cls = _runtime_class(
            self.compiler_contract,
            "compiler_cls",
            f"{label.title()}Compiler",
            calls=calls,
            structural=structural,
        )
        driver_cls = _runtime_class(
            self.driver_contract,
            "driver_cls",
            f"{label.title()}Driver",
            calls=calls,
            structural=structural,
        )
        return compiler_cls, driver_cls, calls


@pytest.fixture(scope="module")
def f6_runtime(tmp_path_factory: pytest.TempPathFactory):
    """Load the real integration module around an isolated source Registry."""
    registry_module = importlib.import_module(
        "triton_anchor.backends.registry"
    )
    previous_registry = registry_module.backend_plugin_registry
    previous_api_registry = backend_api.backend_plugin_registry
    previous_triton_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "triton" or name.startswith("triton.")
    }
    for name in tuple(previous_triton_modules):
        sys.modules.pop(name, None)

    distributions: list[Any] = []
    registry = BackendPluginRegistry(
        distribution_provider=lambda: tuple(distributions),
        environment_provider=collect_core_environment,
        preflight_profile="triton_version",
    )
    registry_module.backend_plugin_registry = registry
    backend_api.backend_plugin_registry = registry

    triton_package = types.ModuleType("triton")
    triton_package.__package__ = "triton"
    triton_package.__path__ = [str(TRITON_PACKAGE)]
    triton_package.__version__ = "3.3.0"
    sys.modules["triton"] = triton_package

    runtime_package = types.ModuleType("triton.runtime")
    runtime_package.__package__ = "triton.runtime"
    runtime_package.__path__ = [str(TRITON_PACKAGE / "runtime")]
    sys.modules["triton.runtime"] = runtime_package

    backends_module = importlib.import_module("triton.backends")
    runtime_driver_module = importlib.import_module("triton.runtime.driver")
    harness = RuntimeHarness(
        registry=registry,
        distributions=distributions,
        backends_module=backends_module,
        runtime_driver_module=runtime_driver_module,
        compiler_contract=backends_module.BaseBackend,
        driver_contract=backends_module.DriverBase,
        root=tmp_path_factory.mktemp("f6-interface-gate"),
    )
    try:
        yield harness
    finally:
        registry.reset()
        registry_module.backend_plugin_registry = previous_registry
        backend_api.backend_plugin_registry = previous_api_registry
        for name in tuple(sys.modules):
            if name == "triton" or name.startswith("triton."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_triton_modules)


def _interface_issues(error: BackendPluginInterfaceError) -> list[dict[str, Any]]:
    payload = error.to_dict()
    issues = payload.get("interface_issues")
    assert isinstance(issues, list) and issues, payload
    assert all(
        {
            "field",
            "owner",
            "member",
            "expected_kind",
            "actual_kind",
            "problem",
            "expected_signature",
            "actual_signature",
            "remediation",
        }
        <= set(issue)
        for issue in issues
    ), payload
    field_order = {"compiler_cls": 0, "driver_cls": 1}
    assert issues == sorted(
        issues,
        key=lambda issue: (
            field_order[issue["field"]],
            issue["owner"],
            issue["member"],
            issue["problem"],
        ),
    )
    return issues


def _assert_rejected_without_publication(
    harness: RuntimeHarness,
    fixture: PluginFixture,
) -> BackendPluginInterfaceError:
    with pytest.raises(BackendPluginInterfaceError) as caught:
        harness.backends_module.get_backend(fixture.target)
    error = caught.value
    payload = error.to_dict()
    assert payload["plugin_id"] == fixture.plugin_id
    assert payload["entry_point"] == fixture.entry_point
    assert payload["expected"]
    assert payload["actual"]
    assert payload["remediation"]
    record = harness.registry.list()[0]
    assert record.state is PluginLifecycleState.REJECTED
    assert not record.initialized
    assert record.errors and record.errors[0] is error
    assert fixture.calls["initialize"] == 0
    assert harness.registry.get_selection(fixture.target) is None
    assert fixture.entry_point not in harness.backends_module.backends
    return error


def test_current_v33_surface_is_derived_from_the_runtime_abcs(
    f6_runtime: RuntimeHarness,
) -> None:
    observed = {
        "compiler_cls": tuple(
            sorted(f6_runtime.compiler_contract.__abstractmethods__)
        ),
        "driver_cls": tuple(
            sorted(f6_runtime.driver_contract.__abstractmethods__)
        ),
    }
    frozen_source_oracle = {
        "compiler_cls": tuple(sorted(_COMPILER_CONTRACT.__abstractmethods__)),
        "driver_cls": tuple(sorted(_DRIVER_CONTRACT.__abstractmethods__)),
    }
    assert observed == frozen_source_oracle
    assert all(observed.values())
    for field, contract in (
        ("compiler_cls", f6_runtime.compiler_contract),
        ("driver_cls", f6_runtime.driver_contract),
    ):
        assert {
            member: _descriptor_kind(contract, member)
            for member in observed[field]
        }


def test_core_exposes_named_runtime_pair_validator_contract() -> None:
    context_cls = getattr(backend_api, "RuntimePairValidationContext", None)
    issue_cls = getattr(backend_api, "BackendPluginInterfaceIssue", None)
    method = getattr(
        BackendPluginRegistry, "register_runtime_pair_validator", None
    )
    assert context_cls is not None, "RuntimePairValidationContext is missing"
    assert issue_cls is not None, "BackendPluginInterfaceIssue is missing"
    assert callable(method), "named runtime-pair validator hook is missing"


def test_complete_v33_abc_pair_passes_before_publication(
    f6_runtime: RuntimeHarness,
) -> None:
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(label="abc")
    assert issubclass(compiler_cls, f6_runtime.compiler_contract)
    assert issubclass(driver_cls, f6_runtime.driver_contract)
    assert not inspect.isabstract(compiler_cls)
    assert not inspect.isabstract(driver_cls)
    fixture = f6_runtime.fixture(
        label="complete_abc",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    decision = f6_runtime.registry.select(fixture.target)
    assert decision.record.state is PluginLifecycleState.SELECTED
    assert decision.record.compiler_cls is compiler_cls
    assert decision.record.driver_cls is driver_cls
    assert calls["initialize"] == 1
    assert calls[("compiler_cls", "__init__")] == 0
    assert calls[("driver_cls", "__init__")] == 0


def test_complete_v33_structural_pair_passes_without_forced_abc_inheritance(
    f6_runtime: RuntimeHarness,
) -> None:
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="structural"
    )
    assert not issubclass(compiler_cls, f6_runtime.compiler_contract)
    assert not issubclass(driver_cls, f6_runtime.driver_contract)
    fixture = f6_runtime.fixture(
        label="complete_structural",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    decision = f6_runtime.registry.select(fixture.target)
    assert decision.record.state is PluginLifecycleState.SELECTED
    assert calls["initialize"] == 1


@pytest.mark.parametrize(
    "field,member",
    _CURRENT_MEMBER_CASES,
    ids=lambda value: value,
)
def test_each_current_v33_abc_abstract_member_is_required(
    f6_runtime: RuntimeHarness,
    field: str,
    member: str,
) -> None:
    assert member in getattr(f6_runtime, field.replace("_cls", "_contract")).__abstractmethods__
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(label="omission")
    contract = (
        f6_runtime.compiler_contract
        if field == "compiler_cls"
        else f6_runtime.driver_contract
    )
    incomplete = _runtime_class(
        contract,
        field,
        f"Incomplete{field.title()}",
        calls=calls,
        omitted=(member,),
    )
    if field == "compiler_cls":
        compiler_cls = incomplete
    else:
        driver_cls = incomplete
    assert inspect.isabstract(incomplete)
    fixture = f6_runtime.fixture(
        label=f"missing_{field}_{member}",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    error = _assert_rejected_without_publication(f6_runtime, fixture)
    issues = _interface_issues(error)
    assert any(
        issue["field"] == field
        and issue["member"] == member
        and "abstract" in issue["problem"]
        for issue in issues
    )


@pytest.mark.parametrize(
    "field,member",
    _CURRENT_MEMBER_CASES,
    ids=lambda value: value,
)
def test_structural_pair_missing_current_surface_member_is_rejected(
    f6_runtime: RuntimeHarness,
    field: str,
    member: str,
) -> None:
    contract = (
        f6_runtime.compiler_contract
        if field == "compiler_cls"
        else f6_runtime.driver_contract
    )
    assert member in contract.__abstractmethods__
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="structural_missing"
    )
    incomplete = _runtime_class(
        contract,
        field,
        f"StructuralIncomplete{field.title()}",
        calls=calls,
        structural=True,
        omitted=(member,),
    )
    if field == "compiler_cls":
        compiler_cls = incomplete
    else:
        driver_cls = incomplete
    fixture = f6_runtime.fixture(
        label=f"structural_missing_{field}",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    issues = _interface_issues(
        _assert_rejected_without_publication(f6_runtime, fixture)
    )
    assert any(
        issue["field"] == field
        and issue["member"] == member
        and issue["problem"] == "missing"
        for issue in issues
    )


def test_compiler_and_driver_issues_are_aggregated_in_one_error(
    f6_runtime: RuntimeHarness,
) -> None:
    calls: Counter[Any] = Counter()
    compiler_missing = sorted(
        f6_runtime.compiler_contract.__abstractmethods__
    )[:2]
    driver_missing = sorted(f6_runtime.driver_contract.__abstractmethods__)[:2]
    compiler_cls = _runtime_class(
        f6_runtime.compiler_contract,
        "compiler_cls",
        "AggregateCompiler",
        calls=calls,
        omitted=compiler_missing,
    )
    driver_cls = _runtime_class(
        f6_runtime.driver_contract,
        "driver_cls",
        "AggregateDriver",
        calls=calls,
        omitted=driver_missing,
    )
    fixture = f6_runtime.fixture(
        label="aggregate",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    error = _assert_rejected_without_publication(f6_runtime, fixture)
    issues = _interface_issues(error)
    assert error.field == "compiler_cls,driver_cls"
    assert {
        (issue["field"], issue["member"])
        for issue in issues
    } >= {
        *(("compiler_cls", member) for member in compiler_missing),
        *(("driver_cls", member) for member in driver_missing),
    }


@pytest.mark.parametrize("field", ("compiler_cls", "driver_cls"))
def test_class_that_remains_abstract_is_rejected(
    f6_runtime: RuntimeHarness,
    field: str,
) -> None:
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        label="reabstract"
    )
    contract = (
        f6_runtime.compiler_contract
        if field == "compiler_cls"
        else f6_runtime.driver_contract
    )
    member = sorted(contract.__abstractmethods__)[0]
    descriptor = inspect.getattr_static(contract, member)
    candidate = _runtime_class(
        contract,
        field,
        f"Reabstracted{field.title()}",
        calls=calls,
        overrides={member: _abstract_implementation(descriptor)},
    )
    assert inspect.isabstract(candidate)
    if field == "compiler_cls":
        compiler_cls = candidate
    else:
        driver_cls = candidate
    fixture = f6_runtime.fixture(
        label=f"reabstract_{field}",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    issues = _interface_issues(
        _assert_rejected_without_publication(f6_runtime, fixture)
    )
    assert any(
        issue["field"] == field
        and issue["member"] == member
        and "abstract" in issue["problem"]
        for issue in issues
    )


def test_class_with_an_extra_abstract_member_is_rejected(
    f6_runtime: RuntimeHarness,
) -> None:
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        label="extra_abstract"
    )

    @abc.abstractmethod
    def extra_contract_member(_self: Any) -> None:
        return None

    compiler_cls = type(
        "ExtraAbstractCompiler",
        (compiler_cls,),
        {
            "__module__": __name__,
            "extra_contract_member": extra_contract_member,
        },
    )
    assert inspect.isabstract(compiler_cls)
    fixture = f6_runtime.fixture(
        label="extra_abstract",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    issues = _interface_issues(
        _assert_rejected_without_publication(f6_runtime, fixture)
    )
    assert any(
        issue["field"] == "compiler_cls"
        and issue["member"] == "extra_contract_member"
        and "abstract" in issue["problem"]
        for issue in issues
    )


def test_structural_method_implemented_as_property_is_rejected(
    f6_runtime: RuntimeHarness,
) -> None:
    method = next(
        member
        for member in sorted(f6_runtime.compiler_contract.__abstractmethods__)
        if _descriptor_kind(f6_runtime.compiler_contract, member) == "method"
    )
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="wrong_kind"
    )
    compiler_cls = _runtime_class(
        f6_runtime.compiler_contract,
        "compiler_cls",
        "PropertyInsteadOfMethod",
        calls=calls,
        structural=True,
        overrides={method: property(lambda _self: None)},
    )
    fixture = f6_runtime.fixture(
        label="method_as_property",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    issues = _interface_issues(
        _assert_rejected_without_publication(f6_runtime, fixture)
    )
    assert any(
        issue["field"] == "compiler_cls"
        and issue["member"] == method
        and issue["expected_kind"] == "method"
        and issue["actual_kind"] == "property"
        and issue["problem"] == "kind_mismatch"
        for issue in issues
    )


def test_structural_classmethod_implemented_as_method_is_rejected(
    f6_runtime: RuntimeHarness,
) -> None:
    field, contract, member = next(
        (field, contract, member)
        for field, contract in (
            ("compiler_cls", f6_runtime.compiler_contract),
            ("driver_cls", f6_runtime.driver_contract),
        )
        for member in sorted(contract.__abstractmethods__)
        if _descriptor_kind(contract, member) == "classmethod"
    )
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="wrong_classmethod"
    )
    candidate = _runtime_class(
        contract,
        field,
        "MethodInsteadOfClassmethod",
        calls=calls,
        structural=True,
        overrides={member: lambda *_args, **_kwargs: True},
    )
    if field == "compiler_cls":
        compiler_cls = candidate
    else:
        driver_cls = candidate
    fixture = f6_runtime.fixture(
        label="classmethod_as_method",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    issues = _interface_issues(
        _assert_rejected_without_publication(f6_runtime, fixture)
    )
    assert any(
        issue["field"] == field
        and issue["member"] == member
        and issue["expected_kind"] == "classmethod"
        and issue["actual_kind"] == "method"
        and issue["problem"] == "kind_mismatch"
        for issue in issues
    )


def test_named_validator_can_report_a_synthetic_property_kind_mismatch(
    f6_runtime: RuntimeHarness,
) -> None:
    """Exercise the real dynamic surface path though v3.3 has no property."""
    member = "acceptance_synthetic_property"
    contract = f6_runtime.driver_contract

    @property
    @abc.abstractmethod
    def required_property(_self: Any) -> Any:
        return None

    original_abstract = contract.__abstractmethods__
    assert member not in original_abstract
    setattr(contract, member, required_property)
    contract.__abstractmethods__ = frozenset((*original_abstract, member))
    try:
        compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
            structural=True, label="synthetic_property"
        )
        setattr(driver_cls, member, 0)
        fixture = f6_runtime.fixture(
            label="synthetic_property",
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
            calls=calls,
        )
        f6_runtime.install(fixture)
        issues = _interface_issues(
            _assert_rejected_without_publication(f6_runtime, fixture)
        )
        assert any(
            issue["field"] == "driver_cls"
            and issue["member"] == member
            and issue["expected_kind"] == "property"
            and issue["actual_kind"] == "data"
            and issue["problem"] == "kind_mismatch"
            for issue in issues
        )
    finally:
        contract.__abstractmethods__ = original_abstract
        delattr(contract, member)


@pytest.mark.parametrize(
    "field,member",
    tuple(
        (field, member)
        for field, contract in (
            ("compiler_cls", _COMPILER_CONTRACT),
            ("driver_cls", _DRIVER_CONTRACT),
        )
        for member in sorted(contract.__abstractmethods__)
        if _descriptor_kind(contract, member) in {"method", "staticmethod"}
    ),
    ids=lambda value: value,
)
def test_safely_detectable_incompatible_signature_is_rejected(
    f6_runtime: RuntimeHarness,
    field: str,
    member: str,
) -> None:
    contract = (
        f6_runtime.compiler_contract
        if field == "compiler_cls"
        else f6_runtime.driver_contract
    )

    def incompatible(_self: Any, *, new_required_argument: Any) -> None:
        return None

    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="signature"
    )
    incompatible_cls = _runtime_class(
        contract,
        field,
        f"IncompatibleSignature{field.title()}",
        calls=calls,
        structural=True,
        overrides={member: incompatible},
    )
    if field == "compiler_cls":
        compiler_cls = incompatible_cls
    else:
        driver_cls = incompatible_cls
    fixture = f6_runtime.fixture(
        label="signature",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    issues = _interface_issues(
        _assert_rejected_without_publication(f6_runtime, fixture)
    )
    assert any(
        issue["field"] == field
        and issue["member"] == member
        and issue["problem"] == "signature_mismatch"
        and issue["expected_signature"]
        and issue["actual_signature"]
        for issue in issues
    )


def test_static_validation_does_not_execute_custom_descriptors(
    f6_runtime: RuntimeHarness,
) -> None:
    calls: Counter[Any] = Counter()
    member = next(
        member
        for member in sorted(f6_runtime.compiler_contract.__abstractmethods__)
        if _descriptor_kind(f6_runtime.compiler_contract, member) == "method"
    )

    class TrapDescriptor:
        def __get__(self, _instance: Any, _owner: Any) -> Any:
            calls["descriptor_get"] += 1
            raise AssertionError("interface validation executed a descriptor")

    compiler_cls, driver_cls, method_calls = f6_runtime.complete_pair(
        structural=True, label="descriptor"
    )
    calls.update(method_calls)
    compiler_cls = _runtime_class(
        f6_runtime.compiler_contract,
        "compiler_cls",
        "DescriptorCompiler",
        calls=calls,
        structural=True,
        overrides={member: TrapDescriptor()},
    )
    fixture = f6_runtime.fixture(
        label="descriptor",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    issues = _interface_issues(
        _assert_rejected_without_publication(f6_runtime, fixture)
    )
    assert calls["descriptor_get"] == 0
    assert any(
        issue["field"] == "compiler_cls"
        and issue["member"] == member
        and issue["problem"] == "kind_mismatch"
        for issue in issues
    )


@pytest.mark.parametrize("field", ("compiler_cls", "driver_cls"))
def test_non_type_runtime_fields_remain_rejected(
    f6_runtime: RuntimeHarness,
    field: str,
) -> None:
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(label="non_type")
    if field == "compiler_cls":
        compiler_cls = object()
    else:
        driver_cls = object()
    fixture = f6_runtime.fixture(
        label=f"non_type_{field}",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    error = _assert_rejected_without_publication(f6_runtime, fixture)
    assert field in error.invalid_fields


def test_rejection_precedes_all_plugin_code_and_public_runtime_state(
    f6_runtime: RuntimeHarness,
) -> None:
    calls: Counter[Any] = Counter()

    class TrapMeta(type):
        def __getattr__(cls, name: str) -> Any:
            calls[("metaclass_getattr", name)] += 1
            raise AssertionError("static validation executed __getattr__")

    compiler_missing = sorted(
        f6_runtime.compiler_contract.__abstractmethods__
    )[0]
    driver_missing = sorted(f6_runtime.driver_contract.__abstractmethods__)[0]
    compiler_cls = _runtime_class(
        f6_runtime.compiler_contract,
        "compiler_cls",
        "SideEffectCompiler",
        calls=calls,
        structural=True,
        omitted=(compiler_missing,),
        metaclass=TrapMeta,
    )
    driver_cls = _runtime_class(
        f6_runtime.driver_contract,
        "driver_cls",
        "SideEffectDriver",
        calls=calls,
        structural=True,
        omitted=(driver_missing,),
        metaclass=TrapMeta,
    )
    fixture = f6_runtime.fixture(
        label="side_effect_boundary",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    driver_config = f6_runtime.runtime_driver_module.driver
    default_before = driver_config.default
    assert driver_config.active is default_before
    assert default_before._obj is None

    attempts: tuple[Callable[[], Any], ...] = (
        lambda: f6_runtime.registry.select(fixture.target),
        lambda: f6_runtime.backends_module.get_backend(fixture.target),
        lambda: f6_runtime.backends_module.make_backend(fixture.target),
        lambda: f6_runtime.backends_module.get_driver_backends(),
        lambda: driver_config.active.get_current_target(),
    )
    for attempt in attempts:
        with pytest.raises(BackendPluginInterfaceError):
            attempt()

    record = f6_runtime.registry.list()[0]
    assert record.state is PluginLifecycleState.REJECTED
    assert not record.initialized
    assert calls["initialize"] == 0
    assert calls["shutdown"] == 0
    assert calls[("compiler_cls", "__init__")] == 0
    assert calls[("driver_cls", "__init__")] == 0
    assert not [key for key in calls if isinstance(key, tuple) and key[0] == "metaclass_getattr"]
    assert not [
        key
        for key, count in calls.items()
        if isinstance(key, tuple)
        and key[0] in {"compiler_cls", "driver_cls"}
        and key[1] != "__init__"
        and count
    ]
    assert f6_runtime.registry.get_selection(fixture.target) is None
    assert fixture.entry_point not in f6_runtime.backends_module.backends
    assert driver_config.active is default_before
    assert driver_config.default is default_before
    assert default_before._obj is None


def test_repeated_rejection_has_identical_issue_order_and_diagnostics(
    f6_runtime: RuntimeHarness,
) -> None:
    calls: Counter[Any] = Counter()
    compiler_cls = _runtime_class(
        f6_runtime.compiler_contract,
        "compiler_cls",
        "DeterministicCompiler",
        calls=calls,
        omitted=f6_runtime.compiler_contract.__abstractmethods__,
    )
    driver_cls = _runtime_class(
        f6_runtime.driver_contract,
        "driver_cls",
        "DeterministicDriver",
        calls=calls,
        omitted=f6_runtime.driver_contract.__abstractmethods__,
    )
    fixture = f6_runtime.fixture(
        label="deterministic",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    diagnostics = []
    for _iteration in range(3):
        f6_runtime.install(fixture)
        error = _assert_rejected_without_publication(f6_runtime, fixture)
        _interface_issues(error)
        diagnostics.append(error.to_dict())
    assert diagnostics[0] == diagnostics[1] == diagnostics[2]
    assert calls["initialize"] == 0


def test_invalid_plugin_does_not_prevent_valid_plugin_selection(
    f6_runtime: RuntimeHarness,
) -> None:
    invalid_compiler, invalid_driver, invalid_calls = f6_runtime.complete_pair(
        label="coexist_invalid"
    )
    missing = sorted(f6_runtime.driver_contract.__abstractmethods__)[0]
    invalid_driver = _runtime_class(
        f6_runtime.driver_contract,
        "driver_cls",
        "CoexistingInvalidDriver",
        calls=invalid_calls,
        omitted=(missing,),
    )
    invalid = f6_runtime.fixture(
        label="coexist_invalid",
        compiler_cls=invalid_compiler,
        driver_cls=invalid_driver,
        calls=invalid_calls,
        target="invalid_target",
    )
    valid_compiler, valid_driver, valid_calls = f6_runtime.complete_pair(
        structural=True, label="coexist_valid"
    )
    valid = f6_runtime.fixture(
        label="coexist_valid",
        compiler_cls=valid_compiler,
        driver_cls=valid_driver,
        calls=valid_calls,
        target="valid_target",
    )
    f6_runtime.install(invalid, valid)
    with pytest.raises(BackendPluginInterfaceError):
        f6_runtime.registry.select(invalid.target)
    selected = f6_runtime.registry.select(valid.target)
    assert selected.plugin_id == valid.plugin_id
    assert selected.record.state is PluginLifecycleState.SELECTED
    assert invalid_calls["initialize"] == 0
    assert valid_calls["initialize"] == 1
    assert f6_runtime.registry.get_selection(invalid.target) is None


def test_reset_never_republishes_previously_rejected_pair(
    f6_runtime: RuntimeHarness,
) -> None:
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(label="reset")
    missing = sorted(f6_runtime.compiler_contract.__abstractmethods__)[0]
    compiler_cls = _runtime_class(
        f6_runtime.compiler_contract,
        "compiler_cls",
        "ResetInvalidCompiler",
        calls=calls,
        omitted=(missing,),
    )
    fixture = f6_runtime.fixture(
        label="reset_invalid",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    first = _assert_rejected_without_publication(f6_runtime, fixture).to_dict()
    f6_runtime.install(fixture)
    second = _assert_rejected_without_publication(f6_runtime, fixture).to_dict()
    assert first == second
    assert fixture.entry_point_object.load_calls == 2
    assert calls["initialize"] == 0
    assert fixture.entry_point not in f6_runtime.backends_module.backends


def test_named_validator_registration_is_idempotent_reset_stable_and_safe(
    f6_runtime: RuntimeHarness,
) -> None:
    register = getattr(
        f6_runtime.registry, "register_runtime_pair_validator", None
    )
    assert callable(register), "named runtime-pair validator hook is missing"
    calls: Counter[str] = Counter()

    def first(_context: Any) -> None:
        calls["first"] += 1
        return None

    def replacement(_context: Any) -> tuple[()]:
        calls["replacement"] += 1
        return ()

    register("acceptance.named", first)
    register("acceptance.named", first)
    compiler_cls, driver_cls, plugin_calls = f6_runtime.complete_pair(
        label="validator_first"
    )
    first_fixture = f6_runtime.fixture(
        label="validator_first",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=plugin_calls,
    )
    f6_runtime.install(first_fixture)
    f6_runtime.registry.select(first_fixture.target)
    assert calls == {"first": 1}

    with pytest.raises(BackendPluginLifecycleError):
        register("acceptance.named", replacement)
    with pytest.raises(BackendPluginLifecycleError):
        register("acceptance.late-attach", replacement)
    assert calls == {"first": 1}

    assert f6_runtime.registry.reset() == ()
    register("acceptance.named", replacement)
    compiler_cls, driver_cls, plugin_calls = f6_runtime.complete_pair(
        label="validator_replacement"
    )
    second_fixture = f6_runtime.fixture(
        label="validator_replacement",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=plugin_calls,
    )
    f6_runtime.install(second_fixture)
    f6_runtime.registry.select(second_fixture.target)
    assert calls == {"first": 1, "replacement": 1}


def test_validator_attached_after_load_still_runs_before_initialize(
    f6_runtime: RuntimeHarness,
) -> None:
    register = getattr(
        f6_runtime.registry, "register_runtime_pair_validator", None
    )
    issue_cls = getattr(backend_api, "BackendPluginInterfaceIssue", None)
    assert callable(register) and issue_cls is not None
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="loaded_attach"
    )
    fixture = f6_runtime.fixture(
        label="loaded_attach",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    loaded = f6_runtime.registry.load(fixture.plugin_id)
    assert loaded.state is PluginLifecycleState.LOADED
    issue = issue_cls(
        field="compiler_cls",
        owner="LateAttachContract",
        member="late_requirement",
        expected_kind="method",
        actual_kind="missing",
        problem="missing",
        remediation="Implement the late-attached requirement.",
    )
    register(
        "acceptance.loaded-attach",
        lambda context: (issue,) if context.plugin_id == fixture.plugin_id else (),
    )
    error = _assert_rejected_without_publication(f6_runtime, fixture)
    assert issue.to_dict() in _interface_issues(error)
    assert calls["initialize"] == 0


def test_validator_registration_racing_runtime_pair_read_fails_closed(
    f6_runtime: RuntimeHarness,
) -> None:
    register = getattr(
        f6_runtime.registry, "register_runtime_pair_validator", None
    )
    assert callable(register)
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="registration_race"
    )
    fixture = f6_runtime.fixture(
        label="registration_race",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    entered = threading.Event()
    release = threading.Event()
    plugin_type = type(fixture.plugin)
    del fixture.plugin.compiler_cls

    def blocking_compiler(_plugin: Any) -> type:
        entered.set()
        assert release.wait(timeout=10)
        return compiler_cls

    plugin_type.compiler_cls = property(blocking_compiler)
    f6_runtime.install(fixture)
    validator_calls: Counter[str] = Counter()

    def late_validator(_context: Any) -> None:
        validator_calls["count"] += 1
        return None

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                f6_runtime.registry.select, fixture.target
            )
            assert entered.wait(timeout=10)
            with pytest.raises(BackendPluginLifecycleError):
                register("acceptance.inflight-attach", late_validator)
            release.set()
            assert future.result(timeout=10).record.state is PluginLifecycleState.SELECTED
    finally:
        release.set()
        del plugin_type.compiler_cls
        fixture.plugin.compiler_cls = compiler_cls
    assert validator_calls == {}
    assert calls["initialize"] == 1


def test_concurrent_selection_executes_each_interface_validator_once(
    f6_runtime: RuntimeHarness,
) -> None:
    register = getattr(
        f6_runtime.registry, "register_runtime_pair_validator", None
    )
    context_cls = getattr(backend_api, "RuntimePairValidationContext", None)
    assert callable(register) and context_cls is not None, (
        "generic runtime-pair validator contract is missing"
    )
    validator_calls: list[Any] = []
    validator_lock = threading.Lock()

    def count(context: Any) -> None:
        with validator_lock:
            validator_calls.append(context)
        return None

    register("acceptance.concurrent", count)
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="concurrent"
    )
    fixture = f6_runtime.fixture(
        label="concurrent",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    barrier = threading.Barrier(8)

    def select_once() -> str:
        barrier.wait(timeout=10)
        return f6_runtime.registry.select(fixture.target).record_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        record_ids = tuple(executor.map(lambda _index: select_once(), range(8)))
    assert len(set(record_ids)) == 1
    assert len(validator_calls) == 1
    context = validator_calls[0]
    assert isinstance(context, context_cls)
    assert context.plugin_id == fixture.plugin_id
    assert context.entry_point == fixture.entry_point
    assert context.compiler_cls is compiler_cls
    assert context.driver_cls is driver_cls
    with pytest.raises(FrozenInstanceError):
        context.plugin_id = "mutated"
    assert calls["initialize"] == 1


def test_concurrent_same_key_registration_never_stacks_the_validator(
    f6_runtime: RuntimeHarness,
) -> None:
    register = getattr(
        f6_runtime.registry, "register_runtime_pair_validator", None
    )
    assert callable(register)
    validator_calls: Counter[str] = Counter()
    validator_lock = threading.Lock()

    def count(_context: Any) -> None:
        with validator_lock:
            validator_calls["count"] += 1
        return None

    assert f6_runtime.registry.reset() == ()
    with ThreadPoolExecutor(max_workers=8) as executor:
        tuple(
            executor.map(
                lambda _index: register("acceptance.concurrent-register", count),
                range(32),
            )
        )

    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="concurrent_register"
    )
    fixture = f6_runtime.fixture(
        label="concurrent_register",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    f6_runtime.registry.select(fixture.target)
    assert validator_calls == {"count": 1}
    assert calls["initialize"] == 1


def test_validator_may_raise_structured_interface_error_before_initialize(
    f6_runtime: RuntimeHarness,
) -> None:
    register = getattr(
        f6_runtime.registry, "register_runtime_pair_validator", None
    )
    issue_cls = getattr(backend_api, "BackendPluginInterfaceIssue", None)
    assert callable(register) and issue_cls is not None
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="raised_error"
    )
    fixture = f6_runtime.fixture(
        label="raised_error",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    issue = issue_cls(
        field="driver_cls",
        owner="RaisedContract",
        member="raised_requirement",
        expected_kind="method",
        actual_kind="missing",
        problem="missing",
        remediation="Implement the raised requirement.",
    )
    expected_error = BackendPluginInterfaceError(
        interface_issues=(issue,),
        plugin_id=fixture.plugin_id,
        entry_point=fixture.entry_point,
    )

    def reject(context: Any) -> None:
        if context.plugin_id == fixture.plugin_id:
            raise expected_error
        return None

    register("acceptance.raised-error", reject)
    f6_runtime.install(fixture)
    error = _assert_rejected_without_publication(f6_runtime, fixture)
    assert error is expected_error
    assert _interface_issues(error) == [issue.to_dict()]
    assert calls["initialize"] == 0


def test_validator_returned_issues_are_aggregated_before_initialize(
    f6_runtime: RuntimeHarness,
) -> None:
    register = getattr(
        f6_runtime.registry, "register_runtime_pair_validator", None
    )
    issue_cls = getattr(backend_api, "BackendPluginInterfaceIssue", None)
    assert callable(register) and issue_cls is not None, (
        "generic runtime-pair issue/validator API is missing"
    )
    issues = (
        issue_cls(
            field="driver_cls",
            owner="ZDriver",
            member="z_member",
            expected_kind="method",
            actual_kind="missing",
            problem="missing",
            remediation="Implement z_member.",
        ),
        issue_cls(
            field="compiler_cls",
            owner="ACompiler",
            member="a_member",
            expected_kind="property",
            actual_kind="data",
            problem="kind_mismatch",
            remediation="Implement a_member as a property.",
        ),
    )
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="returned_issues"
    )
    fixture = f6_runtime.fixture(
        label="returned_issues",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    register(
        "acceptance.aggregate",
        lambda context: issues if context.plugin_id == fixture.plugin_id else (),
    )
    f6_runtime.install(fixture)
    error = _assert_rejected_without_publication(f6_runtime, fixture)
    payload = _interface_issues(error)
    assert payload == [issues[1].to_dict(), issues[0].to_dict()]
    assert error.field == "compiler_cls,driver_cls"
    assert calls["initialize"] == 0
