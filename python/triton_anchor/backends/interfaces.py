"""Version-neutral static validation for compiler/driver runtime pairs.

The Core package intentionally has no dependency on Triton ABCs.  A Triton
adapter supplies the abstract base classes that describe its current runtime
surface, and the Registry invokes the resulting validator before plugin
initialization or publication.
"""

from __future__ import annotations

import abc
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import FunctionType, MappingProxyType
from typing import Any

_MISSING = object()
_OPTIONAL_DEFAULT = object()

# Calling ``type.__getattribute__(candidate, name)`` is not sufficient here:
# a hostile metaclass can install a data descriptor for ``__dict__`` or
# ``__mro__``.  Calling type's own C-level getsets directly reads the real
# class dictionary and the already-computed tp_mro without consulting that
# metaclass.
_TYPE_DICT_GETSET = type.__dict__["__dict__"]
_TYPE_MRO_GETSET = type.__dict__["__mro__"]
_TYPE_MODULE_GETSET = type.__dict__["__module__"]
_TYPE_QUALNAME_GETSET = type.__dict__["__qualname__"]


def _class_dict(candidate: type) -> Mapping[str, Any]:
    return _TYPE_DICT_GETSET.__get__(candidate, type(candidate))


def _class_mro(candidate: type) -> tuple[type, ...]:
    return _TYPE_MRO_GETSET.__get__(candidate, type(candidate))


def _qualified_class_name(candidate: type) -> str:
    module = _TYPE_MODULE_GETSET.__get__(candidate, type(candidate))
    qualname = _TYPE_QUALNAME_GETSET.__get__(candidate, type(candidate))
    if type(module) is not str:
        module = "<unknown>"
    if type(qualname) is not str:
        qualname = "<anonymous class>"
    return f"{module}.{qualname}"


def _raw_descriptor(candidate: type, member: str) -> Any:
    metaclass = type(candidate)
    if metaclass is type or metaclass is abc.ABCMeta:
        # These two metaclasses use the built-in lookup implementation, so
        # inspect.getattr_static cannot enter plugin code.
        return inspect.getattr_static(candidate, member, _MISSING)
    for base in _class_mro(candidate):
        namespace = _class_dict(base)
        if member in namespace:
            return namespace[member]
    return _MISSING


def _descriptor_kind(descriptor: Any) -> str:
    descriptor_type = type(descriptor)
    if descriptor_type is property:
        return "property"
    if descriptor_type is classmethod:
        return "classmethod"
    if descriptor_type is staticmethod:
        return "staticmethod"
    if descriptor_type is FunctionType:
        return "method"
    if descriptor is _MISSING:
        return "missing"
    return "unsupported_descriptor"


def _descriptor_function(descriptor: Any, kind: str) -> FunctionType | None:
    if kind == "method" and type(descriptor) is FunctionType:
        return descriptor
    if kind in {"classmethod", "staticmethod"}:
        if type(descriptor) is not (
            classmethod if kind == "classmethod" else staticmethod
        ):
            return None
        function = descriptor.__func__
        return function if type(function) is FunctionType else None
    if kind == "property" and type(descriptor) is property:
        function = descriptor.fget
        return function if type(function) is FunctionType else None
    return None


def _safe_function_signature(
    function: FunctionType,
) -> inspect.Signature | None:
    """Build a signature without consulting plugin-controlled metadata.

    ``inspect.signature`` reads ``__wrapped__`` and ``__signature__`` even on
    an exact Python function.  Both attributes are writable and may contain
    hostile objects.  Code/default layout is sufficient for call-shape
    compatibility and never renders default values or annotations.
    """
    if type(function) is not FunctionType:
        return None
    code = function.__code__
    defaults = function.__defaults__
    keyword_defaults = function.__kwdefaults__
    if defaults is not None and type(defaults) is not tuple:
        return None
    if keyword_defaults is not None and type(keyword_defaults) is not dict:
        return None

    positional_count = code.co_argcount
    positional_only_count = code.co_posonlyargcount
    keyword_only_count = code.co_kwonlyargcount
    positional_defaults = len(defaults or ())
    first_optional = positional_count - positional_defaults

    safe_keyword_defaults = set()
    if keyword_defaults:
        for name in keyword_defaults:
            if type(name) is not str:
                return None
            safe_keyword_defaults.add(name)

    parameters = []
    for index in range(positional_count):
        name = code.co_varnames[index]
        kind = (
            inspect.Parameter.POSITIONAL_ONLY
            if index < positional_only_count
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        default = (
            _OPTIONAL_DEFAULT if index >= first_optional else inspect.Parameter.empty
        )
        parameters.append(inspect.Parameter(name, kind, default=default))

    variadic_offset = positional_count + keyword_only_count
    if code.co_flags & inspect.CO_VARARGS:
        parameters.append(
            inspect.Parameter(
                code.co_varnames[variadic_offset],
                inspect.Parameter.VAR_POSITIONAL,
            )
        )
        variadic_offset += 1

    keyword_offset = positional_count
    for index in range(keyword_only_count):
        name = code.co_varnames[keyword_offset + index]
        default = (
            _OPTIONAL_DEFAULT
            if name in safe_keyword_defaults
            else inspect.Parameter.empty
        )
        parameters.append(
            inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
            )
        )

    if code.co_flags & inspect.CO_VARKEYWORDS:
        parameters.append(
            inspect.Parameter(
                code.co_varnames[variadic_offset],
                inspect.Parameter.VAR_KEYWORD,
            )
        )
    try:
        return inspect.Signature(parameters)
    except ValueError:
        return None


def _bound_signature(
    signature: inspect.Signature,
    descriptor_kind: str,
) -> inspect.Signature | None:
    if descriptor_kind == "staticmethod":
        return signature
    parameters = tuple(signature.parameters.values())
    if not parameters:
        return None
    first = parameters[0]
    if first.kind in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        parameters = parameters[1:]
    elif first.kind is not inspect.Parameter.VAR_POSITIONAL:
        return None
    try:
        return signature.replace(parameters=parameters)
    except ValueError:
        return None


def _render_signature(signature: inspect.Signature | None) -> str | None:
    if signature is None:
        return None
    rendered = []
    for parameter in signature.parameters.values():
        requirement = (
            "required"
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
            else "optional"
        )
        rendered.append(f"{parameter.name}:{parameter.kind.name.lower()}:{requirement}")
    return "(" + ",".join(rendered) + ")"


def _call_shapes(
    signature: inspect.Signature,
) -> tuple[tuple[tuple[object, ...], Mapping[str, object]], ...]:
    sentinel = object()
    parameters = tuple(signature.parameters.values())

    minimal_args = []
    minimal_kwargs: dict[str, object] = {}
    positional_only = []
    positional_or_keyword = []
    keyword_only: dict[str, object] = {}
    has_var_keyword = False

    for parameter in parameters:
        required = parameter.default is inspect.Parameter.empty
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            if required:
                minimal_args.append(sentinel)
            positional_only.append(sentinel)
        elif parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if required:
                minimal_args.append(sentinel)
            positional_or_keyword.append(parameter.name)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            if required:
                minimal_kwargs[parameter.name] = sentinel
            keyword_only[parameter.name] = sentinel
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True

    shapes = [
        (tuple(minimal_args), MappingProxyType(minimal_kwargs)),
    ]
    for split in range(len(positional_or_keyword) + 1):
        args = tuple(positional_only) + (sentinel,) * split
        kwargs = {name: sentinel for name in positional_or_keyword[split:]}
        kwargs.update(keyword_only)
        if has_var_keyword:
            extra_name = "__t63_contract_extra__"
            while extra_name in kwargs:
                extra_name += "_"
            kwargs[extra_name] = sentinel
        shapes.append((args, MappingProxyType(kwargs)))
    return tuple(shapes)


def _signature_accepts_contract(
    expected: inspect.Signature,
    candidate: inspect.Signature,
) -> bool:
    expected_kinds = {parameter.kind for parameter in expected.parameters.values()}
    candidate_kinds = {parameter.kind for parameter in candidate.parameters.values()}
    for variadic_kind in {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }:
        if variadic_kind in expected_kinds and variadic_kind not in candidate_kinds:
            return False
    for args, kwargs in _call_shapes(expected):
        try:
            candidate.bind(*args, **kwargs)
        except TypeError:
            return False
    return True


def _abstract_names(candidate: type) -> tuple[str, ...]:
    value = _class_dict(candidate).get("__abstractmethods__", _MISSING)
    if value is _MISSING:
        return ()
    if type(value) is not frozenset or any(type(name) is not str for name in value):
        return ("<unverifiable abstract state>",)
    return tuple(sorted(value))


def _metaclass_lookup_issue(
    candidate: type,
    members: tuple[str, ...],
) -> str | None:
    metaclass = type(candidate)
    if metaclass is type or metaclass is abc.ABCMeta:
        return None
    for base in _class_mro(metaclass):
        if base is type or base is abc.ABCMeta or base is object:
            continue
        namespace = _class_dict(base)
        if "__getattribute__" in namespace:
            return f"{_qualified_class_name(base)}.__getattribute__"
        for member in members:
            if member in namespace:
                return f"{_qualified_class_name(base)}.{member}"
    return None


def _descriptor_is_abstract(descriptor: Any, kind: str) -> bool:
    function = _descriptor_function(descriptor, kind)
    if function is None:
        return False
    for name, value in function.__dict__.items():
        if type(name) is str and name == "__isabstractmethod__" and value is True:
            return True
    return False


@dataclass(frozen=True)
class RuntimePairValidationContext:
    """The two plugin-provided class objects exposed to a trusted validator."""

    compiler_cls: type = field(repr=False)
    driver_cls: type = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.compiler_cls, type) or not isinstance(
            self.driver_cls, type
        ):
            raise TypeError("runtime pair context values must be classes")

    def class_for(self, field_name: str) -> type:
        if field_name == "compiler_cls":
            return self.compiler_cls
        if field_name == "driver_cls":
            return self.driver_cls
        raise KeyError(field_name)


@dataclass(frozen=True)
class RuntimeInterfaceSurface:
    """One Registry runtime field and the integration-owned abstract base."""

    field: str
    abstract_base: type = field(repr=False)

    def __post_init__(self) -> None:
        if self.field not in {"compiler_cls", "driver_cls"}:
            raise ValueError("runtime interface field must name the runtime pair")
        if not isinstance(self.abstract_base, type):
            raise TypeError("runtime interface abstract_base must be a class")


@dataclass(frozen=True)
class RuntimeInterfaceMember:
    """A stable description of one required abstract member."""

    field: str
    owner: str
    name: str
    descriptor_kind: str
    signature: str | None

    def __post_init__(self) -> None:
        values = (self.field, self.owner, self.name, self.descriptor_kind)
        if any(type(value) is not str or not value for value in values):
            raise TypeError("runtime interface member fields must be strings")
        if self.signature is not None and type(self.signature) is not str:
            raise TypeError("runtime interface signature must be a string or None")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "field": self.field,
            "owner": self.owner,
            "name": self.name,
            "descriptor_kind": self.descriptor_kind,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class BackendPluginInterfaceIssue:
    """One deterministic mismatch against an integration-owned surface."""

    field: str
    owner: str
    member: str
    problem: str
    expected: str
    actual: str

    def __post_init__(self) -> None:
        values = (
            self.field,
            self.owner,
            self.member,
            self.problem,
            self.expected,
            self.actual,
        )
        if any(type(value) is not str or not value for value in values):
            raise TypeError("runtime interface issue fields must be strings")

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "owner": self.owner,
            "member": self.member,
            "problem": self.problem,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class RuntimePairValidationResult:
    """A complete, stable result returned to the version-neutral Registry."""

    contract_id: str
    required_surface: tuple[RuntimeInterfaceMember, ...]
    issues: tuple[BackendPluginInterfaceIssue, ...] = ()

    def __post_init__(self) -> None:
        if type(self.contract_id) is not str or not self.contract_id:
            raise TypeError("runtime interface result contract_id must be a string")
        if type(self.required_surface) is not tuple or any(
            type(member) is not RuntimeInterfaceMember
            for member in self.required_surface
        ):
            raise TypeError("runtime interface result surface must be exact")
        if type(self.issues) is not tuple or any(
            type(issue) is not BackendPluginInterfaceIssue for issue in self.issues
        ):
            raise TypeError("runtime interface result issues must be exact")

    @property
    def compatible(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class AbstractRuntimePairValidator:
    """Validate a pair structurally against dynamically supplied ABCs."""

    contract_id: str
    surfaces: tuple[RuntimeInterfaceSurface, ...]
    required_surface: tuple[RuntimeInterfaceMember, ...] = field(
        init=False,
    )

    def __post_init__(self) -> None:
        if type(self.contract_id) is not str or not self.contract_id.strip():
            raise ValueError("runtime interface contract_id must be non-empty")
        if type(self.surfaces) is not tuple or not self.surfaces:
            raise ValueError("runtime interface surfaces must be a non-empty tuple")
        if any(
            type(surface) is not RuntimeInterfaceSurface for surface in self.surfaces
        ):
            raise TypeError(
                "runtime interface surfaces must contain exact surface values"
            )
        fields = tuple(surface.field for surface in self.surfaces)
        if len(set(fields)) != len(fields):
            raise ValueError("runtime interface surface fields must be unique")

        members = []
        for surface in self.surfaces:
            owner = surface.abstract_base
            owner_name = _qualified_class_name(owner)
            abstract_names = _abstract_names(owner)
            if not abstract_names:
                raise ValueError(
                    f"runtime interface owner {owner_name} has no abstract surface"
                )
            for name in abstract_names:
                descriptor = _raw_descriptor(owner, name)
                kind = _descriptor_kind(descriptor)
                function = _descriptor_function(descriptor, kind)
                raw_signature = (
                    _safe_function_signature(function) if function is not None else None
                )
                signature = (
                    _bound_signature(raw_signature, kind)
                    if raw_signature is not None
                    else None
                )
                members.append(
                    RuntimeInterfaceMember(
                        field=surface.field,
                        owner=owner_name,
                        name=name,
                        descriptor_kind=kind,
                        signature=_render_signature(signature),
                    )
                )
        object.__setattr__(self, "required_surface", tuple(members))

    def __call__(
        self,
        context: RuntimePairValidationContext,
    ) -> RuntimePairValidationResult:
        if type(context) is not RuntimePairValidationContext:
            raise TypeError("runtime pair validator requires an exact context")

        issues = []
        members_by_field: dict[str, list[RuntimeInterfaceMember]] = {}
        for member in self.required_surface:
            members_by_field.setdefault(member.field, []).append(member)

        for surface in self.surfaces:
            candidate = context.class_for(surface.field)
            owner_members = tuple(members_by_field[surface.field])
            member_names = tuple(member.name for member in owner_members)
            metaclass_override = _metaclass_lookup_issue(candidate, member_names)
            if metaclass_override is not None:
                issues.append(
                    BackendPluginInterfaceIssue(
                        field=surface.field,
                        owner=owner_members[0].owner,
                        member="<class lookup>",
                        problem="metaclass_lookup",
                        expected="built-in type/ABCMeta member lookup",
                        actual=metaclass_override,
                    )
                )

            abstract_names = _abstract_names(candidate)
            abstract_name_set = set(abstract_names)
            for member in owner_members:
                descriptor = _raw_descriptor(candidate, member.name)
                actual_kind = _descriptor_kind(descriptor)
                if descriptor is _MISSING:
                    issues.append(
                        BackendPluginInterfaceIssue(
                            field=surface.field,
                            owner=member.owner,
                            member=member.name,
                            problem="missing",
                            expected=member.descriptor_kind,
                            actual="missing",
                        )
                    )
                    continue
                if member.name in abstract_name_set or _descriptor_is_abstract(
                    descriptor, actual_kind
                ):
                    issues.append(
                        BackendPluginInterfaceIssue(
                            field=surface.field,
                            owner=member.owner,
                            member=member.name,
                            problem="abstract",
                            expected="a concrete implementation",
                            actual="abstract",
                        )
                    )
                    continue
                if actual_kind != member.descriptor_kind:
                    issues.append(
                        BackendPluginInterfaceIssue(
                            field=surface.field,
                            owner=member.owner,
                            member=member.name,
                            problem="descriptor_kind",
                            expected=member.descriptor_kind,
                            actual=actual_kind,
                        )
                    )
                    continue

                expected_descriptor = _raw_descriptor(
                    surface.abstract_base, member.name
                )
                expected_function = _descriptor_function(
                    expected_descriptor, member.descriptor_kind
                )
                candidate_function = _descriptor_function(descriptor, actual_kind)
                expected_signature = (
                    _safe_function_signature(expected_function)
                    if expected_function is not None
                    else None
                )
                candidate_signature = (
                    _safe_function_signature(candidate_function)
                    if candidate_function is not None
                    else None
                )
                expected_bound = (
                    _bound_signature(expected_signature, member.descriptor_kind)
                    if expected_signature is not None
                    else None
                )
                candidate_bound = (
                    _bound_signature(candidate_signature, actual_kind)
                    if candidate_signature is not None
                    else None
                )
                if expected_bound is not None and (
                    candidate_bound is None
                    or not _signature_accepts_contract(expected_bound, candidate_bound)
                ):
                    issues.append(
                        BackendPluginInterfaceIssue(
                            field=surface.field,
                            owner=member.owner,
                            member=member.name,
                            problem="signature",
                            expected=_render_signature(expected_bound)
                            or "a compatible call signature",
                            actual=_render_signature(candidate_bound) or "unavailable",
                        )
                    )

            unrelated_abstract = tuple(
                name for name in abstract_names if name not in set(member_names)
            )
            for name in unrelated_abstract:
                issues.append(
                    BackendPluginInterfaceIssue(
                        field=surface.field,
                        owner=owner_members[0].owner,
                        member=name,
                        problem="class_abstract",
                        expected="a concrete runtime class",
                        actual="abstract",
                    )
                )
            if (
                (type(candidate) is type or type(candidate) is abc.ABCMeta)
                and inspect.isabstract(candidate)
                and not abstract_names
            ):
                issues.append(
                    BackendPluginInterfaceIssue(
                        field=surface.field,
                        owner=owner_members[0].owner,
                        member="<class>",
                        problem="class_abstract",
                        expected="a concrete runtime class",
                        actual="abstract",
                    )
                )

        field_order = {
            surface.field: index for index, surface in enumerate(self.surfaces)
        }
        issues.sort(
            key=lambda issue: (
                field_order[issue.field],
                issue.owner,
                issue.member,
                issue.problem,
                issue.expected,
                issue.actual,
            )
        )
        return RuntimePairValidationResult(
            contract_id=self.contract_id,
            required_surface=self.required_surface,
            issues=tuple(issues),
        )


__all__ = [
    "AbstractRuntimePairValidator",
    "BackendPluginInterfaceIssue",
    "RuntimeInterfaceMember",
    "RuntimeInterfaceSurface",
    "RuntimePairValidationContext",
    "RuntimePairValidationResult",
]
