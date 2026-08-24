"""Static runtime-pair contract derived from Triton's current backend ABCs.

This module is the Triton-facing half of the generic Core Registry validator
extension.  It deliberately derives the required member names from the live
ABCs and inspects candidate classes without instantiating them or invoking any
plugin descriptor or method.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from types import FunctionType, MappingProxyType
from typing import Any, Optional, Tuple

from triton_anchor.backends import (
    BackendPluginInterfaceIssue,
    RuntimePairValidationContext,
)

from .compiler import BaseBackend
from .driver import DriverBase


TRITON_RUNTIME_INTERFACE_CONTRACT = "triton.current-runtime-interface"

_MISSING = object()
_FIELD_ORDER = {"compiler_cls": 0, "driver_cls": 1}
_TYPE_MRO_DESCRIPTOR = type.__dict__["__mro__"]
_TYPE_DICT_DESCRIPTOR = type.__dict__["__dict__"]
_CLASSMETHOD_FUNC_DESCRIPTOR = classmethod.__dict__["__func__"]
_STATICMETHOD_FUNC_DESCRIPTOR = staticmethod.__dict__["__func__"]
_PROPERTY_FGET_DESCRIPTOR = property.__dict__["fget"]
_CLASSMETHOD_GET_DESCRIPTOR = classmethod.__dict__["__get__"]
_STATICMETHOD_GET_DESCRIPTOR = staticmethod.__dict__["__get__"]
_PROPERTY_GET_DESCRIPTOR = property.__dict__["__get__"]
_FUNCTION_CODE_DESCRIPTOR = FunctionType.__dict__["__code__"]
_FUNCTION_DEFAULTS_DESCRIPTOR = FunctionType.__dict__["__defaults__"]
_FUNCTION_KWDEFAULTS_DESCRIPTOR = FunctionType.__dict__["__kwdefaults__"]
_FUNCTION_DICT_DESCRIPTOR = FunctionType.__dict__["__dict__"]
_CO_VARARGS = inspect.CO_VARARGS
_CO_VARKEYWORDS = inspect.CO_VARKEYWORDS

_POSITIONAL_ONLY = "positional_only"
_POSITIONAL_OR_KEYWORD = "positional_or_keyword"
_VAR_POSITIONAL = "var_positional"
_KEYWORD_ONLY = "keyword_only"
_VAR_KEYWORD = "var_keyword"


class _FrozenABCNamespace:
    __slots__ = ()

    ABCMeta = type(BaseBackend)


# Preserve the exact stdlib algorithm requested for ABC subclasses without
# consulting the mutable public ``inspect`` module after entry-point loading.
_FROZEN_INSPECT_ISABSTRACT = FunctionType(
    inspect.isabstract.__code__,
    {
        "__builtins__": {},
        "isinstance": isinstance,
        "type": type,
        "issubclass": issubclass,
        "hasattr": hasattr,
        "getattr": getattr,
        "abc": _FrozenABCNamespace(),
        "TPFLAGS_IS_ABSTRACT": inspect.TPFLAGS_IS_ABSTRACT,
    },
    "_frozen_inspect_isabstract",
)


@dataclass(frozen=True)
class _ParameterShape:
    name: str
    kind: str
    required: bool


@dataclass(frozen=True)
class _SignatureShape:
    parameters: Tuple[_ParameterShape, ...]


@dataclass(frozen=True)
class _InterfaceMember:
    owner: str
    member: str
    kind: str
    exists: bool
    abstract: bool
    signature: Optional[str]
    descriptor: Any = field(repr=False, compare=False)
    raw_signature: Optional[_SignatureShape] = field(
        repr=False, compare=False
    )


def _type_mro(owner: type) -> Tuple[type, ...]:
    try:
        return tuple(_TYPE_MRO_DESCRIPTOR.__get__(owner, type(owner)))
    except (AttributeError, TypeError):
        return ()


def _identity_index(values: Tuple[type, ...], needle: type) -> Optional[int]:
    for index, value in enumerate(values):
        if value is needle:
            return index
    return None


def _type_dictionary_definition(
    owner: type, name: str
) -> Tuple[Optional[type], Any]:
    """Find a class-MRO definition through ``type`` without plugin hooks."""
    try:
        # Calling ``type.__getattribute__`` directly is not sufficient here:
        # it still honors a descriptor with the same name on an untrusted
        # metaclass.  Bind the built-in ``type`` descriptors explicitly so
        # neither ``__mro__`` nor ``__dict__`` can be intercepted.
        mro = _type_mro(owner)
    except (AttributeError, TypeError):
        return None, _MISSING
    for base in mro:
        try:
            namespace = _TYPE_DICT_DESCRIPTOR.__get__(base, type(base))
        except (AttributeError, TypeError):
            return None, _MISSING
        if type(namespace) is not MappingProxyType:
            return None, _MISSING
        # ``type()`` accepts string subclasses as namespace keys.  A normal
        # mappingproxy lookup can execute such a key's ``__eq__`` on a hash
        # collision, so inspect its inert item stream and trust exact strings.
        for key, value in namespace.items():
            if type(key) is str and key == name:
                return base, value
    return None, _MISSING


def _type_dictionary_lookup(owner: type, name: str, default: Any) -> Any:
    """Read a class dictionary/MRO through ``type`` without plugin hooks."""
    _defining_class, value = _type_dictionary_definition(owner, name)
    return default if value is _MISSING else value


def _type_namespaces_are_static_safe(owner: type) -> bool:
    """Return whether static/abstract inspection can avoid hostile key hooks."""
    for base in _type_mro(owner):
        try:
            namespace = _TYPE_DICT_DESCRIPTOR.__get__(base, type(base))
        except (AttributeError, TypeError):
            return False
        if type(namespace) is not MappingProxyType or any(
            type(key) is not str for key, _value in namespace.items()
        ):
            return False
    return True


def _type_namespace_snapshot(
    owner: type,
) -> Optional[Tuple[Tuple[type, Tuple[Tuple[str, Any], ...]], ...]]:
    snapshot = []
    for base in _type_mro(owner):
        try:
            namespace = _TYPE_DICT_DESCRIPTOR.__get__(base, type(base))
        except (AttributeError, TypeError):
            return None
        if type(namespace) is not MappingProxyType:
            return None
        items = tuple(namespace.items())
        if any(type(key) is not str for key, _value in items):
            return None
        snapshot.append((base, items))
    return tuple(snapshot)


_TRUSTED_METACLASS_SNAPSHOTS = tuple(
    (
        metaclass,
        _type_namespace_snapshot(metaclass),
    )
    for index, metaclass in enumerate(
        (type, type(BaseBackend), type(DriverBase))
    )
    if metaclass not in (type, type(BaseBackend), type(DriverBase))[:index]
)


def _trusted_metaclass_is_intact(metaclass: type) -> bool:
    frozen = next(
        (
            snapshot
            for trusted, snapshot in _TRUSTED_METACLASS_SNAPSHOTS
            if metaclass is trusted
        ),
        None,
    )
    if frozen is None:
        return False
    current = _type_namespace_snapshot(metaclass)
    if current is None or len(current) != len(frozen):
        return False
    for (current_base, current_items), (frozen_base, frozen_items) in zip(
        current, frozen
    ):
        if current_base is not frozen_base or len(current_items) != len(
            frozen_items
        ):
            return False
        for (current_key, current_value), (
            frozen_key,
            frozen_value,
        ) in zip(current_items, frozen_items):
            if current_key != frozen_key or current_value is not frozen_value:
                return False
    return True


def _exact_string_dict_get(
    mapping: dict[Any, Any], name: str, default: Any
) -> Any:
    """Read an exact-string key without invoking hostile key equality.

    Function ``__dict__`` and ``__kwdefaults__`` objects may contain keys
    supplied by plugin code.  A normal hash lookup can therefore execute an
    arbitrary key's ``__eq__`` on a hash collision.  Iteration is inert, and
    restricting comparisons to exact strings keeps this boundary static.
    """
    if type(mapping) is not dict or type(name) is not str:
        return default
    for key, value in dict.items(mapping):
        if type(key) is str and key == name:
            return value
    return default


def _static_class_attribute(owner: type, name: str, default: Any) -> Any:
    """Read a class-MRO member without mutable inspection helpers."""
    raw_value = _type_dictionary_lookup(owner, name, _MISSING)
    return default if raw_value is _MISSING else raw_value


def _qualified_name(value: type) -> str:
    module = _type_dictionary_lookup(value, "__module__", None)
    qualname = _type_dictionary_lookup(value, "__qualname__", None)
    if type(module) is str and type(qualname) is str:
        return f"{module}.{qualname}"
    if type(qualname) is str:
        return qualname
    return "<runtime class>"


def _static_instance_of(value: Any, expected: type) -> bool:
    return any(base is expected for base in _type_mro(type(value)))


def _uses_builtin_wrapper_get(value: Any, expected_get: Any) -> bool:
    return (
        _type_namespaces_are_static_safe(type(value))
        and _type_dictionary_lookup(type(value), "__get__", _MISSING)
        is expected_get
    )


def _descriptor_kind(descriptor: Any) -> str:
    if descriptor is _MISSING:
        return "missing"
    if _static_instance_of(descriptor, property):
        return (
            "property"
            if _uses_builtin_wrapper_get(descriptor, _PROPERTY_GET_DESCRIPTOR)
            else "descriptor"
        )
    if _static_instance_of(descriptor, classmethod):
        return (
            "classmethod"
            if _uses_builtin_wrapper_get(
                descriptor, _CLASSMETHOD_GET_DESCRIPTOR
            )
            else "descriptor"
        )
    if _static_instance_of(descriptor, staticmethod):
        return (
            "staticmethod"
            if _uses_builtin_wrapper_get(
                descriptor, _STATICMETHOD_GET_DESCRIPTOR
            )
            else "descriptor"
        )
    if type(descriptor) is FunctionType:
        return "method"
    descriptor_method = _static_class_attribute(
        type(descriptor), "__get__", _MISSING
    )
    if descriptor_method is not _MISSING:
        return "descriptor"
    return "data"


def _descriptor_callable(descriptor: Any, kind: str) -> Any:
    try:
        if kind == "classmethod":
            return _CLASSMETHOD_FUNC_DESCRIPTOR.__get__(
                descriptor, type(descriptor)
            )
        if kind == "staticmethod":
            return _STATICMETHOD_FUNC_DESCRIPTOR.__get__(
                descriptor, type(descriptor)
            )
        if kind == "property":
            return _PROPERTY_FGET_DESCRIPTOR.__get__(
                descriptor, type(descriptor)
            )
    except (AttributeError, TypeError):
        return None
    return descriptor


def _signature_text(signature: _SignatureShape) -> str:
    """Describe a signature without rendering plugin defaults/annotations."""
    parameters = []
    for parameter in signature.parameters:
        if parameter.kind in {_VAR_POSITIONAL, _VAR_KEYWORD}:
            requirement = "variadic"
        elif parameter.required:
            requirement = "required"
        else:
            requirement = "optional"
        parameters.append(
            f"{parameter.kind}:{parameter.name}:{requirement}"
        )
    return "(" + ",".join(parameters) + ")"


def _python_function_signature(
    function: FunctionType,
) -> Optional[_SignatureShape]:
    """Build a call signature only from intrinsic CPython function fields."""
    try:
        code = _FUNCTION_CODE_DESCRIPTOR.__get__(function, FunctionType)
        defaults = _FUNCTION_DEFAULTS_DESCRIPTOR.__get__(
            function, FunctionType
        )
        keyword_defaults = _FUNCTION_KWDEFAULTS_DESCRIPTOR.__get__(
            function, FunctionType
        )
    except (AttributeError, TypeError):
        return None
    if defaults is not None and type(defaults) is not tuple:
        return None
    if keyword_defaults is not None and type(keyword_defaults) is not dict:
        return None

    positional_count = code.co_argcount
    positional_only_count = code.co_posonlyargcount
    keyword_only_count = code.co_kwonlyargcount
    names = code.co_varnames
    default_count = 0 if defaults is None else len(defaults)
    if default_count > positional_count:
        return None
    first_optional = positional_count - default_count
    parameters = []
    for index in range(positional_count):
        kind = (
            _POSITIONAL_ONLY
            if index < positional_only_count
            else _POSITIONAL_OR_KEYWORD
        )
        parameters.append(
            _ParameterShape(
                names[index], kind, required=index < first_optional
            )
        )

    keyword_start = positional_count
    variadic_index = positional_count + keyword_only_count
    if code.co_flags & _CO_VARARGS:
        parameters.append(
            _ParameterShape(
                names[variadic_index], _VAR_POSITIONAL, required=False
            )
        )
        variadic_index += 1
    for index in range(keyword_start, keyword_start + keyword_only_count):
        name = names[index]
        optional = keyword_defaults is not None and _exact_string_dict_get(
            keyword_defaults, name, _MISSING
        ) is not _MISSING
        parameters.append(
            _ParameterShape(
                name,
                _KEYWORD_ONLY,
                required=not optional,
            )
        )
    if code.co_flags & _CO_VARKEYWORDS:
        parameters.append(
            _ParameterShape(
                names[variadic_index], _VAR_KEYWORD, required=False
            )
        )
    if any(type(parameter.name) is not str for parameter in parameters):
        return None
    return _SignatureShape(tuple(parameters))


def _safe_signature(
    descriptor: Any, kind: str
) -> Tuple[Optional[_SignatureShape], Optional[str]]:
    if descriptor is _MISSING:
        return None, None
    callable_member = _descriptor_callable(descriptor, kind)
    # Python functions cannot override attribute access, but ``inspect.signature``
    # still honors attacker-controlled ``__signature__`` metadata.  Derive the
    # shape from the intrinsic code/default slots instead.
    if type(callable_member) is not FunctionType:
        return None, None
    signature = _python_function_signature(callable_member)
    if signature is None:
        return None, None
    return signature, _signature_text(signature)


def _descriptor_is_abstract(descriptor: Any, kind: str) -> bool:
    callable_member = _descriptor_callable(descriptor, kind)
    if type(callable_member) is not FunctionType:
        return False
    try:
        namespace = _FUNCTION_DICT_DESCRIPTOR.__get__(
            callable_member, FunctionType
        )
    except (AttributeError, TypeError):
        return True
    if type(namespace) is not dict:
        return True
    marker = _exact_string_dict_get(
        namespace, "__isabstractmethod__", _MISSING
    )
    # Only explicit false values are concrete.  An unexpected marker is
    # rejected without calling its potentially plugin-controlled __bool__.
    return marker is not _MISSING and marker is not False and marker is not None


def _interface_surface(owner: type) -> Tuple[_InterfaceMember, ...]:
    owner_name = _qualified_name(owner)
    abstract_names = _static_class_attribute(
        owner, "__abstractmethods__", frozenset()
    )
    allowed_containers = (set, frozenset, tuple, list)
    if not any(
        type(abstract_names) is allowed for allowed in allowed_containers
    ) or any(
        type(name) is not str for name in abstract_names
    ):
        raise TypeError("runtime ABC abstract surface must contain strings")
    members = []
    for member in sorted(abstract_names):
        raw_descriptor = _type_dictionary_lookup(owner, member, _MISSING)
        descriptor = (
            inspect.getattr_static(owner, member, _MISSING)
            if raw_descriptor is not _MISSING
            else _MISSING
        )
        kind = _descriptor_kind(descriptor)
        signature, signature_text = _safe_signature(descriptor, kind)
        members.append(
            _InterfaceMember(
                owner=owner_name,
                member=member,
                kind=kind,
                exists=descriptor is not _MISSING,
                abstract=True,
                signature=signature_text,
                descriptor=descriptor,
                raw_signature=signature,
            )
        )
    return tuple(members)


# Freeze the dynamically derived live-v3.3 surfaces while Triton's integration
# module is imported, before backend entry points are discovered or loaded.
# Untrusted plugin code therefore cannot weaken the contract by mutating an ABC.
_COMPILER_SURFACE = _interface_surface(BaseBackend)
_DRIVER_SURFACE = _interface_surface(DriverBase)


def _call_patterns(
    signature: _SignatureShape, kind: str
) -> Tuple[Tuple[int, frozenset[str]], ...]:
    """Return representative valid static call shapes for one ABC member."""
    parameters = signature.parameters
    if kind in {"method", "property"} and not parameters:
        return ()

    positional_full = 0
    keyword_full: set[str] = set()
    keyword_minimum: set[str] = set()
    positional_minimum = 0
    for index, parameter in enumerate(parameters):
        is_receiver = kind in {"method", "property"} and index == 0
        if parameter.kind == _POSITIONAL_ONLY:
            positional_full += 1
            if parameter.required:
                positional_minimum += 1
        elif parameter.kind == _POSITIONAL_OR_KEYWORD:
            positional_full += 1
            if is_receiver:
                positional_minimum += 1
            else:
                keyword_full.add(parameter.name)
                if parameter.required:
                    keyword_minimum.add(parameter.name)
        elif parameter.kind == _KEYWORD_ONLY:
            keyword_full.add(parameter.name)
            if parameter.required:
                keyword_minimum.add(parameter.name)

    candidates = (
        (
            positional_full,
            frozenset(
                parameter.name
                for parameter in parameters
                if parameter.kind == _KEYWORD_ONLY
            ),
        ),
        (positional_minimum, frozenset(keyword_minimum)),
        (positional_minimum, frozenset(keyword_full)),
    )
    return tuple(dict.fromkeys(candidates))


def _signature_accepts(
    signature: _SignatureShape,
    positional_count: int,
    keywords: frozenset[str],
) -> bool:
    positional = tuple(
        parameter
        for parameter in signature.parameters
        if parameter.kind in {_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD}
    )
    has_var_positional = any(
        parameter.kind == _VAR_POSITIONAL
        for parameter in signature.parameters
    )
    has_var_keyword = any(
        parameter.kind == _VAR_KEYWORD
        for parameter in signature.parameters
    )
    if positional_count > len(positional) and not has_var_positional:
        return False
    assigned = {
        parameter.name
        for parameter in positional[: min(positional_count, len(positional))]
    }
    named = {
        parameter.name: parameter
        for parameter in signature.parameters
        if parameter.kind
        in {_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD, _KEYWORD_ONLY}
    }
    for keyword in keywords:
        parameter = named.get(keyword)
        if parameter is None:
            if not has_var_keyword:
                return False
            continue
        if parameter.kind == _POSITIONAL_ONLY:
            if not has_var_keyword:
                return False
            continue
        if keyword in assigned:
            return False
        assigned.add(keyword)
    return all(
        not parameter.required or parameter.name in assigned
        for parameter in signature.parameters
        if parameter.kind
        in {_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD, _KEYWORD_ONLY}
    )


def _bound_classmethod_signature(
    signature: _SignatureShape,
) -> Optional[_SignatureShape]:
    parameters = signature.parameters
    if not parameters:
        return None
    first = parameters[0]
    if first.kind in {_POSITIONAL_ONLY, _POSITIONAL_OR_KEYWORD}:
        return _SignatureShape(parameters[1:])
    if first.kind == _VAR_POSITIONAL:
        return signature
    return None


def _signature_is_compatible(
    expected: _SignatureShape,
    actual: _SignatureShape,
    kind: str,
) -> Optional[bool]:
    # Triton 3.3's historical classmethod declarations do not use a uniform
    # receiver convention.  Their kind and concreteness are enforceable, but
    # comparing their raw signatures would reject valid current backends.
    comparison_kind = kind
    if kind == "classmethod":
        expected_parameters = expected.parameters
        actual = _bound_classmethod_signature(actual)
        if actual is None:
            return False
        if (
            not expected_parameters
            or expected_parameters[0].name not in {"self", "cls"}
        ):
            return None
        expected = _bound_classmethod_signature(expected)
        if expected is None:
            return False
        comparison_kind = "staticmethod"
    expected_parameters = expected.parameters
    actual_parameters = actual.parameters
    if any(
        parameter.kind == _VAR_POSITIONAL
        for parameter in expected_parameters
    ) and not any(
        parameter.kind == _VAR_POSITIONAL
        for parameter in actual_parameters
    ):
        return False
    if any(
        parameter.kind == _VAR_KEYWORD
        for parameter in expected_parameters
    ) and not any(
        parameter.kind == _VAR_KEYWORD
        for parameter in actual_parameters
    ):
        return False
    patterns = _call_patterns(expected, comparison_kind)
    if not patterns:
        return False
    for positional_count, keywords in patterns:
        if not _signature_accepts(actual, positional_count, keywords):
            return False
    return True


def _issue(
    *,
    field_name: str,
    surface: _InterfaceMember,
    actual_kind: str,
    problem: str,
    actual_signature: Optional[str] = None,
) -> BackendPluginInterfaceIssue:
    if problem == "missing":
        remediation = (
            f"Implement {surface.member} as a concrete {surface.kind} "
            f"required by {surface.owner}."
        )
    elif problem in {"still_abstract", "class_abstract"}:
        remediation = (
            f"Provide a concrete implementation of {surface.member} before "
            "publishing this runtime class."
        )
    elif problem == "kind_mismatch":
        remediation = (
            f"Implement {surface.member} with member kind {surface.kind}."
        )
    elif problem == "unsafe_metaclass":
        remediation = (
            "Use a class whose metaclass does not override static class "
            "attribute access required by interface validation."
        )
    elif problem == "unsafe_class_metadata":
        remediation = (
            "Use exact string keys in the runtime class namespace so static "
            "interface and abstractness checks cannot invoke plugin code."
        )
    else:
        remediation = (
            f"Make the static signature of {surface.member} compatible with "
            f"the {surface.owner} contract."
        )
    return BackendPluginInterfaceIssue(
        field=field_name,
        owner=surface.owner,
        member=surface.member,
        expected_kind=surface.kind,
        actual_kind=actual_kind,
        problem=problem,
        expected_signature=surface.signature,
        actual_signature=actual_signature,
        remediation=remediation,
    )


def _candidate_issues(
    field_name: str,
    owner: type,
    candidate: type,
    surface: Tuple[_InterfaceMember, ...],
) -> Tuple[BackendPluginInterfaceIssue, ...]:
    required_names = {member.member for member in surface}
    candidate_mro = _type_mro(candidate)
    is_abc_subclass = any(base is owner for base in candidate_mro)
    class_metadata_is_safe = _type_namespaces_are_static_safe(candidate)
    abstract_defining_class, abstract_names = _type_dictionary_definition(
        candidate, "__abstractmethods__"
    )
    if abstract_names is _MISSING:
        abstract_names = frozenset()
    allowed_containers = (set, frozenset, tuple, list)
    invalid_abstract_metadata = not any(
        type(abstract_names) is allowed for allowed in allowed_containers
    ) or any(type(name) is not str for name in abstract_names) or (
        is_abc_subclass and abstract_defining_class is not candidate
    )
    if invalid_abstract_metadata:
        abstract_names = ()
    abstract_names = frozenset(abstract_names)
    issues = []
    candidate_abstract = bool(abstract_names)
    trusted_metaclasses = (type, type(owner))
    metaclass_is_trusted = any(
        type(candidate) is trusted for trusted in trusted_metaclasses
    ) and _trusted_metaclass_is_intact(type(candidate))
    if not metaclass_is_trusted:
        unsafe_surface = _InterfaceMember(
            owner=_qualified_name(owner),
            member="<class>",
            kind="static-safe metaclass",
            exists=True,
            abstract=False,
            signature=None,
            descriptor=candidate,
            raw_signature=None,
        )
        issues.append(
            _issue(
                field_name=field_name,
                surface=unsafe_surface,
                actual_kind="custom metaclass attribute access",
                problem="unsafe_metaclass",
            )
        )
    elif (
        is_abc_subclass
        and class_metadata_is_safe
        and not invalid_abstract_metadata
    ):
        candidate_abstract = _FROZEN_INSPECT_ISABSTRACT(candidate)
    if not class_metadata_is_safe:
        metadata_surface = _InterfaceMember(
            owner=_qualified_name(owner),
            member="<class>",
            kind="exact-string class metadata",
            exists=True,
            abstract=False,
            signature=None,
            descriptor=candidate,
            raw_signature=None,
        )
        issues.append(
            _issue(
                field_name=field_name,
                surface=metadata_surface,
                actual_kind="untrusted class dictionary keys",
                problem="unsafe_class_metadata",
            )
        )
    if invalid_abstract_metadata:
        invalid_surface = _InterfaceMember(
            owner=_qualified_name(owner),
            member="<class>",
            kind="string abstract member names",
            exists=True,
            abstract=False,
            signature=None,
            descriptor=candidate,
            raw_signature=None,
        )
        issues.append(
            _issue(
                field_name=field_name,
                surface=invalid_surface,
                actual_kind="invalid abstract metadata",
                problem="unsafe_abstract_metadata",
            )
        )

    for member in surface:
        defining_class, raw_descriptor = _type_dictionary_definition(
            candidate, member.member
        )
        descriptor = (
            _static_class_attribute(candidate, member.member, _MISSING)
            if raw_descriptor is not _MISSING
            else _MISSING
        )
        actual_kind = _descriptor_kind(descriptor)
        if descriptor is _MISSING:
            issues.append(
                _issue(
                    field_name=field_name,
                    surface=member,
                    actual_kind="missing",
                    problem="missing",
                )
            )
            continue
        owner_position = (
            _identity_index(candidate_mro, owner)
            if is_abc_subclass
            else None
        )
        defining_position = (
            _identity_index(candidate_mro, defining_class)
            if defining_class is not None
            else None
        )
        inherited_requirement = (
            owner_position is not None
            and defining_position is not None
            and defining_position >= owner_position
        )
        if (
            inherited_requirement
            or descriptor is member.descriptor
            or member.member in abstract_names
            or _descriptor_is_abstract(descriptor, actual_kind)
        ):
            _actual_signature, actual_text = _safe_signature(
                descriptor, actual_kind
            )
            issues.append(
                _issue(
                    field_name=field_name,
                    surface=member,
                    actual_kind=actual_kind,
                    problem="still_abstract",
                    actual_signature=actual_text,
                )
            )
            continue
        if actual_kind != member.kind:
            issues.append(
                _issue(
                    field_name=field_name,
                    surface=member,
                    actual_kind=actual_kind,
                    problem="kind_mismatch",
                )
            )
            continue
        actual_signature, actual_text = _safe_signature(
            descriptor, actual_kind
        )
        if member.raw_signature is None or actual_signature is None:
            issues.append(
                _issue(
                    field_name=field_name,
                    surface=member,
                    actual_kind=actual_kind,
                    problem="signature_unavailable",
                    actual_signature=actual_text,
                )
            )
            continue
        signature_compatible = _signature_is_compatible(
            member.raw_signature, actual_signature, member.kind
        )
        if signature_compatible is False:
            issues.append(
                _issue(
                    field_name=field_name,
                    surface=member,
                    actual_kind=actual_kind,
                    problem="signature_mismatch",
                    actual_signature=actual_text,
                )
            )

    if candidate_abstract:
        for name in sorted(abstract_names - required_names):
            descriptor = _static_class_attribute(candidate, name, _MISSING)
            actual_kind = _descriptor_kind(descriptor)
            _raw_signature, actual_text = _safe_signature(
                descriptor, actual_kind
            )
            extra_surface = _InterfaceMember(
                owner=_qualified_name(candidate),
                member=name,
                kind="concrete",
                exists=descriptor is not _MISSING,
                abstract=True,
                signature=None,
                descriptor=descriptor,
                raw_signature=None,
            )
            issues.append(
                _issue(
                    field_name=field_name,
                    surface=extra_surface,
                    actual_kind=actual_kind,
                    problem="still_abstract",
                    actual_signature=actual_text,
                )
            )
        if not abstract_names:
            class_surface = _InterfaceMember(
                owner=_qualified_name(candidate),
                member="<class>",
                kind="concrete",
                exists=True,
                abstract=True,
                signature=None,
                descriptor=candidate,
                raw_signature=None,
            )
            issues.append(
                _issue(
                    field_name=field_name,
                    surface=class_surface,
                    actual_kind="abstract",
                    problem="class_abstract",
                )
            )

    return tuple(
        sorted(
            issues,
            key=lambda issue: (
                _FIELD_ORDER.get(issue.field, 99),
                issue.owner,
                issue.member,
                issue.problem,
            ),
        )
    )


def validate_triton_runtime_pair(
    context: RuntimePairValidationContext,
) -> Tuple[BackendPluginInterfaceIssue, ...]:
    """Return every static incompatibility with the live Triton ABC surface."""
    issues = (
        *_candidate_issues(
            "compiler_cls",
            BaseBackend,
            context.compiler_cls,
            _COMPILER_SURFACE,
        ),
        *_candidate_issues(
            "driver_cls", DriverBase, context.driver_cls, _DRIVER_SURFACE
        ),
    )
    return tuple(
        sorted(
            issues,
            key=lambda issue: (
                _FIELD_ORDER.get(issue.field, 99),
                issue.owner,
                issue.member,
                issue.problem,
            ),
        )
    )


__all__ = [
    "TRITON_RUNTIME_INTERFACE_CONTRACT",
    "validate_triton_runtime_pair",
]
