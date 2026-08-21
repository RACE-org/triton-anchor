"""Independent Triton 3.6 runtime-interface publication-gate acceptance tests.

The oracle is the abstract surface loaded from this checkout's real
``triton.backends.compiler`` and ``triton.backends.driver`` modules.  No
compiler or driver member name is copied into this test.
"""

from __future__ import annotations

import abc
import importlib
import importlib.machinery
import inspect
import json
import sys
import threading
import time
import types
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import triton_anchor.backends as backend_api
from packaging.tags import Tag
from triton_anchor.backends import (
    AbstractRuntimePairValidator,
    BackendPluginInterfaceError,
    BackendPluginLifecycleError,
    BackendPluginRegistry,
    CoreEnvironment,
    PluginLifecycleState,
    RuntimeInterfaceSurface,
    RuntimePairValidationContext,
    RuntimePairValidationResult,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRITON_PACKAGE = REPOSITORY_ROOT / "triton/python/triton"
TRITON_COMMIT = "6cc4505027d7b39fe18a44a7f89085b8babb7400"
LLVM_COMMIT = "a992f29451b9e140424f35ac5e20177db4afbdc0"


def _environment() -> CoreEnvironment:
    return CoreEnvironment(
        core_version="0.2.0",
        build_info_generated=True,
        backend_protocol_version="1.0",
        manifest_schema_version="1.0",
        triton_version="3.6.0",
        vendored_triton_commit=TRITON_COMMIT,
        expected_llvm_project_commit=LLVM_COMMIT,
        actual_llvm_version_raw="22.0.0git",
        actual_llvm_version="22.0.0",
        actual_llvm_version_suffix="git",
        actual_llvm_commit=LLVM_COMMIT,
        actual_mlir_version_raw="22.0.0git",
        actual_mlir_version="22.0.0",
        actual_mlir_version_suffix="git",
        actual_mlir_commit=LLVM_COMMIT,
        cxx_standard="17",
        cxx_compiler_id="GNU",
        cxx_compiler_version="13.3.0",
        cxx11_abi="1",
        build_type="Release",
        ttgpu=True,
        built_python_version="3.12.3",
        built_python_soabi="cpython-fixture",
        built_platform="fixture-platform",
        core_abi_fingerprint_schema="triton-anchor-core-abi-v1",
        core_library_sha256="sha256:" + "c" * 64,
        core_abi_fingerprint="sha256:" + "a" * 64,
        runtime_python_version="3.12.3",
        runtime_python_implementation="CPython",
        runtime_python_soabi="cpython-fixture",
        runtime_platform="fixture-platform",
        runtime_system="Linux",
        runtime_machine="x86_64",
    )


class _EntryPoint:
    group = "triton.backends"

    def __init__(self, name: str, plugin: Any) -> None:
        self.name = name
        self.value = f"t63_f6_v36_{name}:plugin"
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

    def read_text(self, filename: str) -> str | None:
        if filename == "WHEEL":
            return "Wheel-Version: 1.0\nTag: py3-none-any\n"
        return None


def _package(name: str, path: Path, *, version: str | None = None):
    package = types.ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(path)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        name, loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = package.__path__
    if version is not None:
        package.__version__ = version
    return package


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


def _implementation(
    descriptor: Any,
    calls: Counter[Any],
    field: str,
    member: str,
) -> Any:
    def implementation(*_args: Any, **_kwargs: Any) -> Any:
        calls[(field, member)] += 1
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


def _runtime_class(
    contract: type,
    field: str,
    name: str,
    calls: Counter[Any],
    *,
    structural: bool,
    omitted: Iterable[str] = (),
    overrides: Mapping[str, Any] | None = None,
    metaclass: type | None = None,
) -> type:
    omitted = frozenset(omitted)
    namespace: dict[str, Any] = {"__module__": __name__}
    for member in sorted(contract.__abstractmethods__):
        if member in omitted:
            continue
        descriptor = inspect.getattr_static(contract, member)
        namespace[member] = _implementation(descriptor, calls, field, member)
    namespace.update(dict(overrides or {}))

    def constructor(self: Any, *_args: Any, **_kwargs: Any) -> None:
        calls[(field, "__init__")] += 1

    namespace["__init__"] = constructor
    bases = (object,) if structural else (contract,)
    class_metaclass = metaclass or type(contract if not structural else object)
    return class_metaclass(name, bases, namespace)


@dataclass
class _PluginFixture:
    plugin_id: str
    entry_point: str
    target: str
    plugin: Any
    distribution: _Distribution
    calls: Counter[Any]


@dataclass
class _Harness:
    registry: BackendPluginRegistry
    distributions: list[Any]
    backends: types.ModuleType
    runtime_driver: types.ModuleType
    compiler_contract: type
    driver_contract: type
    root: Path
    sequence: int = 0

    def complete_pair(
        self, *, structural: bool, label: str
    ) -> tuple[type, type, Counter[Any]]:
        calls: Counter[Any] = Counter()
        compiler_cls = _runtime_class(
            self.compiler_contract,
            "compiler_cls",
            f"{label.title()}Compiler",
            calls,
            structural=structural,
        )
        driver_cls = _runtime_class(
            self.driver_contract,
            "driver_cls",
            f"{label.title()}Driver",
            calls,
            structural=structural,
        )
        return compiler_cls, driver_cls, calls

    def fixture(
        self,
        *,
        label: str,
        compiler_cls: Any,
        driver_cls: Any,
        calls: Counter[Any],
        target: str | None = None,
    ) -> _PluginFixture:
        self.sequence += 1
        entry_point_name = f"{label}_{self.sequence}"
        plugin_id = f"acceptance.f6.v36.{entry_point_name}"
        target = target or entry_point_name

        class Plugin:
            def initialize(self, _context: Mapping[str, Any]) -> None:
                calls["initialize"] += 1

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
                                "version": ">=3.6,<3.7",
                                "commit": TRITON_COMMIT,
                            },
                            "targets": [target],
                            "capabilities": ["acceptance.f6"],
                            "isolation_mode": "python_only",
                            "priority": 0,
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        distribution = _Distribution(
            f"t63-f6-v36-{entry_point_name}", entry_point, manifest
        )
        return _PluginFixture(
            plugin_id,
            entry_point_name,
            target,
            plugin,
            distribution,
            calls,
        )

    def install(self, *fixtures: _PluginFixture) -> None:
        assert self.registry.reset() == ()
        self.distributions[:] = [fixture.distribution for fixture in fixtures]

    def validator(
        self,
        contract_id: str = "acceptance.triton-v3.6.abstract-surface",
    ) -> AbstractRuntimePairValidator:
        return AbstractRuntimePairValidator(
            contract_id=contract_id,
            surfaces=(
                RuntimeInterfaceSurface(
                    field="compiler_cls",
                    abstract_base=self.compiler_contract,
                ),
                RuntimeInterfaceSurface(
                    field="driver_cls",
                    abstract_base=self.driver_contract,
                ),
            ),
        )

    def register_contract(
        self,
        contract_id: str = "acceptance.triton-v3.6.abstract-surface",
    ) -> AbstractRuntimePairValidator:
        validator = self.validator(contract_id)
        result = self.registry.register_runtime_pair_validator(contract_id, validator)
        assert result is None
        return validator


@pytest.fixture
def f6_runtime(tmp_path: Path):
    """Load canonical real v3.6 modules without importing built libtriton."""
    registry_module = importlib.import_module("triton_anchor.backends.registry")
    previous_registry = registry_module.backend_plugin_registry
    previous_api_registry = backend_api.backend_plugin_registry
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "triton" or name.startswith("triton.")
    }
    for name in tuple(previous_modules):
        sys.modules.pop(name, None)

    distributions: list[Any] = []
    registry = BackendPluginRegistry(
        distribution_provider=lambda: tuple(distributions),
        environment_provider=_environment,
        supported_tags=(Tag("py3", "none", "any"),),
        preflight_profile="triton_version",
    )
    registry_module.backend_plugin_registry = registry
    backend_api.backend_plugin_registry = registry
    sys.modules["triton"] = _package("triton", TRITON_PACKAGE, version="3.6.0")

    try:
        backends = importlib.import_module("triton.backends")
        sys.modules["triton.runtime"] = _package(
            "triton.runtime", TRITON_PACKAGE / "runtime"
        )
        runtime_driver = importlib.import_module("triton.runtime.driver")
        yield _Harness(
            registry=registry,
            distributions=distributions,
            backends=backends,
            runtime_driver=runtime_driver,
            compiler_contract=backends.BaseBackend,
            driver_contract=backends.DriverBase,
            root=tmp_path,
        )
    finally:
        registry.reset()
        registry_module.backend_plugin_registry = previous_registry
        backend_api.backend_plugin_registry = previous_api_registry
        for name in tuple(sys.modules):
            if name == "triton" or name.startswith("triton."):
                sys.modules.pop(name, None)
        sys.modules.update(previous_modules)


def _contract_for(harness: _Harness, field: str) -> type:
    return (
        harness.compiler_contract
        if field == "compiler_cls"
        else harness.driver_contract
    )


def _issues(error: BackendPluginInterfaceError) -> list[dict[str, str]]:
    payload = error.to_dict()
    issues = payload["interface_issues"]
    assert isinstance(issues, list) and issues, payload
    assert all(
        {"field", "owner", "member", "problem", "expected", "actual"} == set(issue)
        for issue in issues
    )
    field_order = {"compiler_cls": 0, "driver_cls": 1}
    assert issues == sorted(
        issues,
        key=lambda issue: (
            field_order[issue["field"]],
            issue["owner"],
            issue["member"],
            issue["problem"],
            issue["expected"],
            issue["actual"],
        ),
    )
    assert payload["contract_ids"]
    assert payload["required_surface"]
    return issues


def _assert_rejected(
    harness: _Harness,
    fixture: _PluginFixture,
    operation: Callable[[], Any] | None = None,
) -> BackendPluginInterfaceError:
    operation = operation or (lambda: harness.registry.select(fixture.target))
    with pytest.raises(BackendPluginInterfaceError) as caught:
        operation()
    error = caught.value
    assert error.plugin_id == fixture.plugin_id
    assert error.entry_point == fixture.entry_point
    record = next(
        record
        for record in harness.registry.list()
        if record.entry_point_name == fixture.entry_point
    )
    assert record.state is PluginLifecycleState.REJECTED
    assert record.error is error
    assert record.compiler_cls is None
    assert record.driver_cls is None
    assert not record.initialized
    assert fixture.calls["initialize"] == 0
    assert harness.registry.get_selection(fixture.target) is None
    assert fixture.entry_point not in harness.backends.backends
    return error


class _SyntheticDescriptorABC(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def property_member(self):
        raise NotImplementedError

    @abc.abstractmethod
    def method_member(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def class_member(cls):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def static_member():
        raise NotImplementedError


class _SyntheticSignatureABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def required_member(self, value):
        raise NotImplementedError

    @abc.abstractmethod
    def positional_only_member(self, value, /):
        raise NotImplementedError

    @abc.abstractmethod
    def optional_member(self, value=None):
        raise NotImplementedError

    @abc.abstractmethod
    def varargs_member(self, *values):
        raise NotImplementedError

    @abc.abstractmethod
    def varkw_member(self, **values):
        raise NotImplementedError

    @abc.abstractmethod
    def mixed_keyword_member(self, first, second):
        raise NotImplementedError

    @abc.abstractmethod
    def variadic_keyword_member(self, *values, flag=True, **options):
        raise NotImplementedError


def _synthetic_validator(
    contract: type,
    contract_id: str,
) -> AbstractRuntimePairValidator:
    return AbstractRuntimePairValidator(
        contract_id=contract_id,
        surfaces=(
            RuntimeInterfaceSurface(field="compiler_cls", abstract_base=contract),
        ),
    )


def test_core_exposes_version_neutral_runtime_pair_validator_api() -> None:
    expected_types = (
        "RuntimePairValidationContext",
        "RuntimePairValidationResult",
        "BackendPluginInterfaceIssue",
        "AbstractRuntimePairValidator",
    )
    missing = [
        name for name in expected_types if getattr(backend_api, name, None) is None
    ]
    register = getattr(BackendPluginRegistry, "register_runtime_pair_validator", None)
    assert not missing, "missing version-neutral Core API: " + ", ".join(missing)
    assert callable(register), "register_runtime_pair_validator is missing"


def test_real_v36_surface_is_loaded_from_canonical_modules(
    f6_runtime: _Harness,
) -> None:
    assert f6_runtime.compiler_contract.__module__ == ("triton.backends.compiler")
    assert f6_runtime.driver_contract.__module__ == "triton.backends.driver"
    assert f6_runtime.compiler_contract.__abstractmethods__
    assert f6_runtime.driver_contract.__abstractmethods__
    assert all(
        _descriptor_kind(contract, member)
        in {"property", "classmethod", "staticmethod", "method"}
        for contract in (
            f6_runtime.compiler_contract,
            f6_runtime.driver_contract,
        )
        for member in contract.__abstractmethods__
    )


def test_invalid_pair_is_rejected_before_initialize_or_publication(
    f6_runtime: _Harness,
) -> None:
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="invalid_boundary"
    )
    compiler_missing = min(f6_runtime.compiler_contract.__abstractmethods__)
    driver_missing = min(f6_runtime.driver_contract.__abstractmethods__)
    compiler_cls = _runtime_class(
        f6_runtime.compiler_contract,
        "compiler_cls",
        "IncompleteV36Compiler",
        calls,
        structural=True,
        omitted=(compiler_missing,),
    )
    driver_cls = _runtime_class(
        f6_runtime.driver_contract,
        "driver_cls",
        "IncompleteV36Driver",
        calls,
        structural=True,
        omitted=(driver_missing,),
    )
    fixture = f6_runtime.fixture(
        label="invalid_boundary",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    driver_config = f6_runtime.runtime_driver.driver
    state_before = (
        driver_config._default,
        driver_config._active,
        driver_config._reset_epoch,
    )

    with pytest.raises(BackendPluginInterfaceError) as caught:
        f6_runtime.registry.select(fixture.target)

    payload = caught.value.to_dict()
    for key in (
        "required_surface",
        "interface_issues",
        "missing_abstract_members",
        "invalid_descriptor_kinds",
    ):
        assert key in payload, payload
    rendered = json.dumps(payload, sort_keys=True)
    assert compiler_missing in rendered
    assert driver_missing in rendered
    record = f6_runtime.registry.list()[0]
    assert record.state is PluginLifecycleState.REJECTED
    assert record.plugin_object is fixture.plugin
    assert record.compiler_cls is None
    assert record.driver_cls is None
    assert not record.initialized
    assert calls["initialize"] == 0
    assert calls["shutdown"] == 0
    assert calls[("compiler_cls", "__init__")] == 0
    assert calls[("driver_cls", "__init__")] == 0
    assert not [
        key
        for key, count in calls.items()
        if isinstance(key, tuple)
        and key[0] in {"compiler_cls", "driver_cls"}
        and key[1] != "__init__"
        and count
    ]
    assert f6_runtime.registry.get_selection(fixture.target) is None
    assert fixture.entry_point not in f6_runtime.backends.backends
    assert (
        driver_config._default,
        driver_config._active,
        driver_config._reset_epoch,
    ) == state_before


def test_complete_structural_v36_pair_passes_without_abc_inheritance(
    f6_runtime: _Harness,
) -> None:
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="valid_structural"
    )
    assert not issubclass(compiler_cls, f6_runtime.compiler_contract)
    assert not issubclass(driver_cls, f6_runtime.driver_contract)
    fixture = f6_runtime.fixture(
        label="valid_structural",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    backend = f6_runtime.backends.get_backend(fixture.target)
    assert backend.compiler is compiler_cls
    assert backend.driver is driver_cls
    selected = f6_runtime.registry.get_selection(fixture.target)
    assert selected is not None
    assert selected.record.state is PluginLifecycleState.SELECTED
    assert calls["initialize"] == 1
    assert calls[("compiler_cls", "__init__")] == 0
    assert calls[("driver_cls", "__init__")] == 0


def test_complete_v36_abc_pair_passes_registered_core_contract(
    f6_runtime: _Harness,
) -> None:
    f6_runtime.register_contract()
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=False, label="valid_abc"
    )
    assert issubclass(compiler_cls, f6_runtime.compiler_contract)
    assert issubclass(driver_cls, f6_runtime.driver_contract)
    assert not inspect.isabstract(compiler_cls)
    assert not inspect.isabstract(driver_cls)
    fixture = f6_runtime.fixture(
        label="valid_abc",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    selected = f6_runtime.registry.select(fixture.target)
    assert selected.record.state is PluginLifecycleState.SELECTED
    assert selected.record.compiler_cls is compiler_cls
    assert selected.record.driver_cls is driver_cls
    assert calls["initialize"] == 1
    assert calls[("compiler_cls", "__init__")] == 0
    assert calls[("driver_cls", "__init__")] == 0


@pytest.mark.parametrize("field", ("compiler_cls", "driver_cls"))
def test_each_real_v36_abstract_member_is_required_structurally(
    f6_runtime: _Harness,
    field: str,
) -> None:
    f6_runtime.register_contract()
    contract = _contract_for(f6_runtime, field)
    assert contract.__abstractmethods__
    for member in sorted(contract.__abstractmethods__):
        compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
            structural=True, label=f"missing_{field}_{member}"
        )
        candidate = _runtime_class(
            contract,
            field,
            f"Missing{field.title()}{member.title()}",
            calls,
            structural=True,
            omitted=(member,),
        )
        if field == "compiler_cls":
            compiler_cls = candidate
        else:
            driver_cls = candidate
        fixture = f6_runtime.fixture(
            label=f"missing_{field}_{member}",
            compiler_cls=compiler_cls,
            driver_cls=driver_cls,
            calls=calls,
        )
        f6_runtime.install(fixture)
        error = _assert_rejected(f6_runtime, fixture)
        issues = _issues(error)
        assert any(
            issue["field"] == field
            and issue["member"] == member
            and issue["problem"] == "missing"
            for issue in issues
        )
        assert member in error.missing_abstract_members[field]
        assert calls[(field, "__init__")] == 0


@pytest.mark.parametrize("field", ("compiler_cls", "driver_cls"))
def test_abc_subclass_with_unrelated_abstract_member_is_rejected(
    f6_runtime: _Harness,
    field: str,
) -> None:
    f6_runtime.register_contract()
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=False, label=f"extra_abstract_{field}"
    )

    @abc.abstractmethod
    def plugin_private_abstract(self):
        raise NotImplementedError

    contract = _contract_for(f6_runtime, field)
    candidate = _runtime_class(
        contract,
        field,
        f"ExtraAbstract{field.title()}",
        calls,
        structural=False,
        overrides={"plugin_private_abstract": plugin_private_abstract},
    )
    assert inspect.isabstract(candidate)
    if field == "compiler_cls":
        compiler_cls = candidate
    else:
        driver_cls = candidate
    fixture = f6_runtime.fixture(
        label=f"extra_abstract_{field}",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    issues = _issues(_assert_rejected(f6_runtime, fixture))
    assert any(
        issue["field"] == field
        and issue["member"] == "plugin_private_abstract"
        and issue["problem"] == "class_abstract"
        for issue in issues
    )


@pytest.mark.parametrize(
    "member,wrong_descriptor",
    (
        ("property_member", lambda fn: fn),
        ("method_member", lambda fn: property(fn)),
        ("class_member", lambda fn: staticmethod(fn)),
        ("static_member", lambda fn: classmethod(fn)),
    ),
)
def test_synthetic_surface_distinguishes_all_descriptor_kinds(
    member: str,
    wrong_descriptor: Callable[[Callable[..., Any]], Any],
) -> None:
    calls: Counter[Any] = Counter()

    def implementation(*_args: Any, **_kwargs: Any) -> None:
        calls["descriptor"] += 1

    candidate = _runtime_class(
        _SyntheticDescriptorABC,
        "compiler_cls",
        f"WrongDescriptor{member.title()}",
        calls,
        structural=True,
        overrides={member: wrong_descriptor(implementation)},
    )
    validator = _synthetic_validator(
        _SyntheticDescriptorABC,
        "acceptance.synthetic-descriptors",
    )
    result = validator(
        RuntimePairValidationContext(
            compiler_cls=candidate,
            driver_cls=type("UnusedDriver", (), {}),
        )
    )
    issue = next(issue for issue in result.issues if issue.member == member)
    assert issue.problem == "descriptor_kind"
    expected_kind = _descriptor_kind(_SyntheticDescriptorABC, member)
    assert issue.expected == expected_kind
    assert issue.actual != expected_kind
    assert calls["descriptor"] == 0


def _required_bad(self):
    return None


def _positional_only_bad(self, *, value):
    return value


def _optional_bad(self, value):
    return value


def _varargs_bad(self, value=None):
    return value


def _varkw_bad(self):
    return None


def _mixed_keyword_bad(self, second, first, *values, **options):
    return second, first, values, options


def _variadic_keyword_bad(self, value=None, *, flag=True):
    return value, flag


@pytest.mark.parametrize(
    "member,incompatible",
    (
        ("required_member", _required_bad),
        ("positional_only_member", _positional_only_bad),
        ("optional_member", _optional_bad),
        ("varargs_member", _varargs_bad),
        ("varkw_member", _varkw_bad),
        ("mixed_keyword_member", _mixed_keyword_bad),
        ("variadic_keyword_member", _variadic_keyword_bad),
    ),
)
def test_signature_contract_rejects_incompatible_call_shapes(
    member: str,
    incompatible: Callable[..., Any],
) -> None:
    calls: Counter[Any] = Counter()
    candidate = _runtime_class(
        _SyntheticSignatureABC,
        "compiler_cls",
        f"IncompatibleSignature{member.title()}",
        calls,
        structural=True,
        overrides={member: incompatible},
    )
    validator = _synthetic_validator(
        _SyntheticSignatureABC,
        "acceptance.synthetic-signatures",
    )
    result = validator(
        RuntimePairValidationContext(
            compiler_cls=candidate,
            driver_cls=type("UnusedDriver", (), {}),
        )
    )
    issue = next(
        (
            issue
            for issue in result.issues
            if issue.member == member and issue.problem == "signature"
        ),
        None,
    )
    assert issue is not None, (
        f"candidate without the contract's {member} call shape was accepted"
    )
    assert issue.expected
    assert issue.actual


def test_variadic_structural_implementation_accepts_signature_surface() -> None:
    calls: Counter[Any] = Counter()
    candidate = _runtime_class(
        _SyntheticSignatureABC,
        "compiler_cls",
        "CompatibleVariadicSignature",
        calls,
        structural=True,
    )
    validator = _synthetic_validator(
        _SyntheticSignatureABC,
        "acceptance.synthetic-signatures-compatible",
    )
    result = validator(
        RuntimePairValidationContext(
            compiler_cls=candidate,
            driver_cls=type("UnusedDriver", (), {}),
        )
    )
    assert result.compatible
    assert result.issues == ()
    assert not calls


def _function_for_descriptor(descriptor: Any) -> types.FunctionType:
    if type(descriptor) is types.FunctionType:
        return descriptor
    if type(descriptor) in {classmethod, staticmethod}:
        function = descriptor.__func__
    elif type(descriptor) is property:
        function = descriptor.fget
    else:
        raise AssertionError(f"unsupported test descriptor: {type(descriptor)}")
    assert type(function) is types.FunctionType
    return function


def test_hostile_function_metadata_is_never_read_or_rendered(
    f6_runtime: _Harness,
) -> None:
    f6_runtime.register_contract()
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="hostile_function_metadata"
    )
    member = min(f6_runtime.compiler_contract.__abstractmethods__)
    descriptor = inspect.getattr_static(compiler_cls, member)
    function = _function_for_descriptor(descriptor)
    hostile_calls: Counter[str] = Counter()

    class HostileMetadata:
        def __getattribute__(self, name: str) -> Any:
            if name != "_calls":
                hostile_calls[f"getattribute:{name}"] += 1
                raise AssertionError("validator read hostile function metadata")
            return object.__getattribute__(self, name)

        def __repr__(self) -> str:
            hostile_calls["repr"] += 1
            raise AssertionError("validator rendered hostile function metadata")

        def __str__(self) -> str:
            hostile_calls["str"] += 1
            raise AssertionError("validator rendered hostile function metadata")

    hostile = HostileMetadata()
    function.__signature__ = hostile
    function.__wrapped__ = hostile
    function.__defaults__ = (hostile,)
    function.__kwdefaults__ = {"hostile_default": hostile}
    function.__annotations__ = {
        "hostile_annotation": hostile,
        "return": hostile,
    }
    fixture = f6_runtime.fixture(
        label="hostile_function_metadata",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    selected = f6_runtime.registry.select(fixture.target)
    assert selected.record.state is PluginLifecycleState.SELECTED
    assert hostile_calls == {}
    assert calls["initialize"] == 1
    assert calls[("compiler_cls", member)] == 0


def test_hostile_metaclass_lookup_is_rejected_without_execution(
    f6_runtime: _Harness,
) -> None:
    f6_runtime.register_contract()
    calls: Counter[Any] = Counter()

    class HostileMeta(type):
        def __getattribute__(cls, name: str) -> Any:
            calls[("metaclass_getattribute", name)] += 1
            raise AssertionError("validator executed metaclass __getattribute__")

        @property
        def __dict__(cls):
            calls["metaclass_dict"] += 1
            raise AssertionError("validator executed metaclass __dict__")

        @property
        def __mro__(cls):
            calls["metaclass_mro"] += 1
            raise AssertionError("validator executed metaclass __mro__")

    compiler_cls = _runtime_class(
        f6_runtime.compiler_contract,
        "compiler_cls",
        "HostileMetaclassCompiler",
        calls,
        structural=True,
        metaclass=HostileMeta,
    )
    driver_cls = _runtime_class(
        f6_runtime.driver_contract,
        "driver_cls",
        "HostileMetaclassDriver",
        calls,
        structural=True,
    )
    fixture = f6_runtime.fixture(
        label="hostile_metaclass",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    issues = _issues(_assert_rejected(f6_runtime, fixture))
    assert any(
        issue["field"] == "compiler_cls"
        and issue["member"] == "<class lookup>"
        and issue["problem"] == "metaclass_lookup"
        for issue in issues
    )
    assert not [
        key
        for key, count in calls.items()
        if count
        and (
            key in {"metaclass_dict", "metaclass_mro"}
            or (isinstance(key, tuple) and key[0] == "metaclass_getattribute")
        )
    ]


@pytest.mark.parametrize("field", ("compiler_cls", "driver_cls"))
def test_non_type_runtime_field_remains_rejected_before_validator(
    f6_runtime: _Harness,
    field: str,
) -> None:
    f6_runtime.register_contract()
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label=f"non_type_{field}"
    )
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
    error = _assert_rejected(f6_runtime, fixture)
    assert field in error.invalid_fields
    assert error.interface_issues == ()
    assert calls["initialize"] == 0


def test_compiler_and_driver_failures_have_stable_aggregate_diagnostics(
    f6_runtime: _Harness,
) -> None:
    f6_runtime.register_contract()
    calls: Counter[Any] = Counter()
    compiler_cls = _runtime_class(
        f6_runtime.compiler_contract,
        "compiler_cls",
        "FullyInvalidCompiler",
        calls,
        structural=True,
        omitted=f6_runtime.compiler_contract.__abstractmethods__,
    )
    driver_cls = _runtime_class(
        f6_runtime.driver_contract,
        "driver_cls",
        "FullyInvalidDriver",
        calls,
        structural=True,
        omitted=f6_runtime.driver_contract.__abstractmethods__,
    )
    fixture = f6_runtime.fixture(
        label="stable_dual_failure",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    diagnostics = []
    error_objects = []
    for _iteration in range(3):
        f6_runtime.install(fixture)
        error = _assert_rejected(f6_runtime, fixture)
        issues = _issues(error)
        assert {issue["field"] for issue in issues} == {
            "compiler_cls",
            "driver_cls",
        }
        diagnostics.append(error.to_dict())
        error_objects.append(error)
    assert diagnostics[0] == diagnostics[1] == diagnostics[2]
    assert len({id(error) for error in error_objects}) == 3
    assert calls["initialize"] == 0


def test_validator_identity_idempotence_late_gate_and_reset_retention(
    f6_runtime: _Harness,
) -> None:
    contract_id = "acceptance.registration-semantics"
    validator = f6_runtime.validator(contract_id)
    assert (
        f6_runtime.registry.register_runtime_pair_validator(contract_id, validator)
        is None
    )
    assert (
        f6_runtime.registry.register_runtime_pair_validator(contract_id, validator)
        is None
    )
    assert f6_runtime.registry.reset() == ()
    assert (
        f6_runtime.registry.register_runtime_pair_validator(contract_id, validator)
        is None
    )
    with pytest.raises(BackendPluginLifecycleError) as caught:
        f6_runtime.registry.register_runtime_pair_validator(
            contract_id,
            f6_runtime.validator(contract_id),
        )
    assert caught.value.field == "runtime_pair_validator"

    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="published_before_late_registration"
    )
    valid = f6_runtime.fixture(
        label="published_before_late_registration",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(valid)
    assert f6_runtime.registry.select(valid.target).record.state is (
        PluginLifecycleState.SELECTED
    )
    late_id = "acceptance.registration-too-late"
    with pytest.raises(BackendPluginLifecycleError) as caught:
        f6_runtime.registry.register_runtime_pair_validator(
            late_id, f6_runtime.validator(late_id)
        )
    assert caught.value.field == "runtime_pair_validator"
    assert caught.value.actual == "runtime pair already published"

    missing = min(f6_runtime.compiler_contract.__abstractmethods__)
    compiler_cls, driver_cls, invalid_calls = f6_runtime.complete_pair(
        structural=True, label="retained_after_reset"
    )
    compiler_cls = _runtime_class(
        f6_runtime.compiler_contract,
        "compiler_cls",
        "RetainedContractCompiler",
        invalid_calls,
        structural=True,
        omitted=(missing,),
    )
    invalid = f6_runtime.fixture(
        label="retained_after_reset",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=invalid_calls,
    )
    f6_runtime.install(invalid)
    error = _assert_rejected(f6_runtime, invalid)
    assert contract_id in error.contract_ids


def test_concurrent_invalid_selection_validates_once_and_shares_error(
    f6_runtime: _Harness,
) -> None:
    contract_id = "acceptance.concurrent-static-validation"
    delegate = f6_runtime.validator(contract_id)
    validator_calls = 0
    validator_lock = threading.Lock()

    def counting_validator(
        context: RuntimePairValidationContext,
    ) -> RuntimePairValidationResult:
        nonlocal validator_calls
        with validator_lock:
            validator_calls += 1
        return delegate(context)

    f6_runtime.registry.register_runtime_pair_validator(contract_id, counting_validator)
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="concurrent_invalid"
    )
    missing = min(f6_runtime.compiler_contract.__abstractmethods__)
    compiler_cls = _runtime_class(
        f6_runtime.compiler_contract,
        "compiler_cls",
        "ConcurrentInvalidCompiler",
        calls,
        structural=True,
        omitted=(missing,),
    )
    fixture = f6_runtime.fixture(
        label="concurrent_invalid",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    workers = 8
    barrier = threading.Barrier(workers)

    def select_once() -> BackendPluginInterfaceError:
        barrier.wait(timeout=10)
        try:
            f6_runtime.registry.select(fixture.target)
        except BackendPluginInterfaceError as error:
            return error
        raise AssertionError("invalid runtime pair was selected")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(select_once) for _index in range(workers)]
        errors = [future.result(timeout=10) for future in futures]

    assert validator_calls == 1
    assert len({id(error) for error in errors}) == 1
    assert all(error.to_dict() == errors[0].to_dict() for error in errors)
    assert calls["initialize"] == 0
    assert calls[("compiler_cls", "__init__")] == 0
    assert f6_runtime.registry.get_selection(fixture.target) is None
    assert f6_runtime.registry._loading == {}
    assert f6_runtime.registry._registering == {}
    assert f6_runtime.registry._waiting_on == {}


def test_blocking_validator_reset_wins_before_initialize_and_is_retained(
    f6_runtime: _Harness,
) -> None:
    contract_id = "acceptance.blocking-static-validation"
    delegate = f6_runtime.validator(contract_id)
    entered = threading.Event()
    release = threading.Event()
    validator_calls = 0
    validator_lock = threading.Lock()

    def blocking_validator(
        context: RuntimePairValidationContext,
    ) -> RuntimePairValidationResult:
        nonlocal validator_calls
        with validator_lock:
            validator_calls += 1
        entered.set()
        if not release.wait(timeout=10):
            raise AssertionError("test did not release static validator")
        return delegate(context)

    f6_runtime.registry.register_runtime_pair_validator(contract_id, blocking_validator)
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="blocking_reset"
    )
    fixture = f6_runtime.fixture(
        label="blocking_reset",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)

    with ThreadPoolExecutor(max_workers=2) as executor:
        selection_future: Future[Any] = executor.submit(
            f6_runtime.registry.select, fixture.target
        )
        assert entered.wait(timeout=10)
        assert (
            f6_runtime.registry.register_runtime_pair_validator(
                contract_id, blocking_validator
            )
            is None
        )
        late_id = "acceptance.during-static-validation"
        with pytest.raises(BackendPluginLifecycleError) as caught:
            f6_runtime.registry.register_runtime_pair_validator(
                late_id, f6_runtime.validator(late_id)
            )
        assert caught.value.actual == "registration in progress"

        reset_future: Future[Any] = executor.submit(f6_runtime.registry.reset)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with f6_runtime.registry._condition:
                if f6_runtime.registry._resetting:
                    break
            time.sleep(0.005)
        else:
            raise AssertionError("reset did not invalidate blocking validation")
        release.set()
        with pytest.raises(BackendPluginLifecycleError) as caught:
            selection_future.result(timeout=10)
        assert caught.value.field == "generation"
        assert reset_future.result(timeout=10) == ()

    assert validator_calls == 1
    assert calls["initialize"] == 0
    assert calls["shutdown"] == 0
    assert f6_runtime.registry.get_selection(fixture.target) is None
    assert f6_runtime.registry._registering == {}
    assert f6_runtime.registry._waiting_on == {}

    # The same version contract survives reset; the retry is newly validated
    # and only that current-generation attempt may initialize and publish.
    selected = f6_runtime.registry.select(fixture.target)
    assert selected.record.state is PluginLifecycleState.SELECTED
    assert validator_calls == 2
    assert calls["initialize"] == 1


def test_invalid_and_valid_plugins_coexist_without_cross_poisoning(
    f6_runtime: _Harness,
) -> None:
    f6_runtime.register_contract()
    invalid_compiler, invalid_driver, invalid_calls = f6_runtime.complete_pair(
        structural=True, label="coexist_invalid"
    )
    missing = min(f6_runtime.driver_contract.__abstractmethods__)
    invalid_driver = _runtime_class(
        f6_runtime.driver_contract,
        "driver_cls",
        "CoexistingInvalidDriver",
        invalid_calls,
        structural=True,
        omitted=(missing,),
    )
    invalid = f6_runtime.fixture(
        label="coexist_invalid",
        compiler_cls=invalid_compiler,
        driver_cls=invalid_driver,
        calls=invalid_calls,
        target="coexist_invalid_target",
    )
    valid_compiler, valid_driver, valid_calls = f6_runtime.complete_pair(
        structural=True, label="coexist_valid"
    )
    valid = f6_runtime.fixture(
        label="coexist_valid",
        compiler_cls=valid_compiler,
        driver_cls=valid_driver,
        calls=valid_calls,
        target="coexist_valid_target",
    )
    f6_runtime.install(invalid, valid)
    _assert_rejected(f6_runtime, invalid)
    backend = f6_runtime.backends.get_backend(valid.target)
    assert backend.compiler is valid_compiler
    assert backend.driver is valid_driver
    assert f6_runtime.registry.get_selection(invalid.target) is None
    assert f6_runtime.registry.get_selection(valid.target) is not None
    assert invalid.entry_point not in f6_runtime.backends.backends
    assert valid.entry_point in f6_runtime.backends.backends
    assert invalid_calls["initialize"] == 0
    assert valid_calls["initialize"] == 1


def test_reset_revalidates_and_reselects_a_legal_structural_pair(
    f6_runtime: _Harness,
) -> None:
    f6_runtime.register_contract()
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="valid_after_reset"
    )
    fixture = f6_runtime.fixture(
        label="valid_after_reset",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    first = f6_runtime.backends.get_backend(fixture.target)
    assert first.compiler is compiler_cls
    assert calls["initialize"] == 1
    assert fixture.entry_point in f6_runtime.backends.backends

    assert f6_runtime.registry.reset() == ()
    assert calls["shutdown"] == 1
    assert fixture.entry_point not in f6_runtime.backends.backends
    assert f6_runtime.registry.get_selection(fixture.target) is None

    second = f6_runtime.backends.get_backend(fixture.target)
    assert second.compiler is compiler_cls
    assert second.driver is driver_cls
    assert calls["initialize"] == 2
    assert calls["shutdown"] == 1
    assert fixture.distribution.entry_points[0].load_calls == 2


def test_adapter_make_backend_and_driver_state_stop_at_interface_gate(
    f6_runtime: _Harness,
) -> None:
    f6_runtime.register_contract()
    compiler_cls, driver_cls, calls = f6_runtime.complete_pair(
        structural=True, label="adapter_boundary"
    )
    compiler_missing = min(f6_runtime.compiler_contract.__abstractmethods__)
    driver_missing = min(f6_runtime.driver_contract.__abstractmethods__)
    compiler_cls = _runtime_class(
        f6_runtime.compiler_contract,
        "compiler_cls",
        "AdapterBoundaryCompiler",
        calls,
        structural=True,
        omitted=(compiler_missing,),
    )
    driver_cls = _runtime_class(
        f6_runtime.driver_contract,
        "driver_cls",
        "AdapterBoundaryDriver",
        calls,
        structural=True,
        omitted=(driver_missing,),
    )
    fixture = f6_runtime.fixture(
        label="adapter_boundary",
        compiler_cls=compiler_cls,
        driver_cls=driver_cls,
        calls=calls,
    )
    f6_runtime.install(fixture)
    driver_config = f6_runtime.runtime_driver.driver
    state_before = (
        driver_config._default,
        driver_config._active,
        driver_config._default_initializing,
        driver_config._reset_epoch,
    )

    error = _assert_rejected(
        f6_runtime,
        fixture,
        operation=lambda: f6_runtime.backends.make_backend(fixture.target),
    )
    _issues(error)
    with pytest.raises(BackendPluginInterfaceError) as caught:
        _ = driver_config.default
    assert caught.value is error
    assert (
        driver_config._default,
        driver_config._active,
        driver_config._default_initializing,
        driver_config._reset_epoch,
    ) == state_before
    assert fixture.entry_point not in f6_runtime.backends.backends
    assert f6_runtime.registry.get_selection(fixture.target) is None
    assert calls["initialize"] == 0
    assert calls[("compiler_cls", "__init__")] == 0
    assert calls[("driver_cls", "__init__")] == 0
    assert not [
        key
        for key, count in calls.items()
        if count
        and isinstance(key, tuple)
        and key[0] in {"compiler_cls", "driver_cls"}
        and key[1] != "__init__"
    ]
