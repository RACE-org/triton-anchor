"""Cross-check the shipped Manifest Schema against the public parser."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from triton_anchor.backends import (
    BackendPluginManifestError,
    evaluate_manifest_semantics,
    parse_manifest,
)
from triton_anchor.backends.manifest import (
    _ABI_FINGERPRINT_PATTERN,
    _NATIVE_LIBRARY_PATH_PATTERN,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKENDS = REPO_ROOT / "python" / "triton_anchor" / "backends"
SCHEMA_PATH = BACKENDS / "schemas" / "backend_manifest.schema.json"
EXAMPLE_PATH = BACKENDS / "examples" / "triton_anchor_backend.example.json"


def _documents() -> tuple[dict, dict]:
    return (
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
        json.loads(EXAMPLE_PATH.read_text(encoding="utf-8")),
    )


def _schema_accepts(schema: dict, candidate: dict) -> bool:
    return not list(Draft202012Validator(schema).iter_errors(candidate))


def _parser_accepts(candidate: dict) -> bool:
    try:
        parse_manifest(candidate)
    except BackendPluginManifestError:
        return False
    return True


def _set_non_empty_string(candidate: dict, location: str, value: str) -> None:
    plugin = candidate["plugins"][0]
    if location == "requires_triton.version":
        plugin["requires_triton"]["version"] = value
    elif location.endswith("[]"):
        plugin[location[:-2]] = [value]
    else:
        plugin[location] = value


NON_EMPTY_STRING_LOCATIONS = (
    "display_name",
    "vendor",
    "backend_protocol",
    "requires_core",
    "requires_triton.version",
    "requires_llvm_version",
    "requires_mlir_version",
    "targets[]",
    "capabilities[]",
    "requires_capabilities[]",
)

NON_EMPTY_STRING_CASES = (
    ("empty", "", False),
    ("leading-ascii-space", " value", False),
    ("trailing-ascii-space", "value ", False),
    ("leading-unicode-nbsp", "\u00a0value", False),
    ("trailing-unicode-nbsp", "value\u00a0", False),
    ("embedded-lf", "line\nline", False),
    ("embedded-cr", "line\rline", False),
    ("embedded-crlf", "line\r\nline", False),
    ("trailing-lf", "value\n", False),
    ("trailing-cr", "value\r", False),
    ("ordinary-internal-space", "ordinary value", True),
    ("unicode", "编译器插件", True),
    ("zero-width-boundary", "\u200bvalue\u200b", True),
)


def test_manifest_schema_is_valid_draft_2020_12_and_example_conforms() -> None:
    schema, example = _documents()
    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(example)) == []
    document = parse_manifest(example)
    assert document.schema_version == "1.0"
    assert document.plugins[0].plugin_id == "example.mock_backend"


def test_public_parser_rejects_value_forbidden_by_shipped_schema() -> None:
    """Schema 1.0 is the independent oracle for accepted Manifest values.

    Frozen Schema 1.0 explicitly rejects CR/LF anywhere in ``nonEmptyString``.
    The runtime reader must not silently accept a document that its distributed
    Schema declares invalid.
    """

    schema, example = _documents()
    candidate = deepcopy(example)
    candidate["plugins"][0]["display_name"] = "line one\nline two"
    errors = list(Draft202012Validator(schema).iter_errors(candidate))
    assert len(errors) == 1
    assert list(errors[0].absolute_path) == ["plugins", 0, "display_name"]
    with pytest.raises(BackendPluginManifestError):
        parse_manifest(candidate)


@pytest.mark.parametrize("location", NON_EMPTY_STRING_LOCATIONS)
@pytest.mark.parametrize(
    "_case,value,expected",
    NON_EMPTY_STRING_CASES,
    ids=[case[0] for case in NON_EMPTY_STRING_CASES],
)
def test_manifest_non_empty_string_schema_parser_domain(
    location: str, _case: str, value: str, expected: bool
) -> None:
    schema, example = _documents()
    candidate = deepcopy(example)
    _set_non_empty_string(candidate, location, value)

    assert _schema_accepts(schema, candidate) is expected
    assert _parser_accepts(candidate) is expected


ANCHORED_FIELDS = (
    ("schema_version", "1.0"),
    ("plugin_id", "example.mock_backend"),
    ("entry_point", "mock"),
    ("requires_triton.commit", "a" * 40),
    ("requires_llvm_commit", "b" * 40),
    ("requires_mlir_commit", "c" * 40),
    ("abi_fingerprint", "sha256:" + "d" * 64),
)


@pytest.mark.parametrize("suffix", ["\n", "\r", "\r\n"], ids=["lf", "cr", "crlf"])
@pytest.mark.parametrize(
    "location,base_value",
    ANCHORED_FIELDS,
    ids=[field[0] for field in ANCHORED_FIELDS],
)
def test_manifest_anchored_pattern_schema_parser_domain(
    location: str, base_value: str, suffix: str
) -> None:
    schema, example = _documents()
    candidate = deepcopy(example)
    plugin = candidate["plugins"][0]
    if location == "schema_version":
        candidate[location] = base_value + suffix
    elif location == "requires_triton.commit":
        plugin["requires_triton"]["commit"] = base_value + suffix
    elif location == "abi_fingerprint":
        # Native isolation is structurally unsupported in 1.0.  Exercise the
        # retained low-level field definition directly so that rejection of
        # isolation_mode cannot mask the fingerprint's own structural domain.
        value = base_value + suffix
        fingerprint_schema = {
            "$schema": schema["$schema"],
            "$defs": schema["$defs"],
            "$ref": "#/$defs/abiFingerprint",
        }
        assert Draft202012Validator(fingerprint_schema).is_valid(value) is False
        assert _ABI_FINGERPRINT_PATTERN.fullmatch(value) is None
        return
    else:
        plugin[location] = base_value + suffix

    assert _schema_accepts(schema, candidate) is False
    assert _parser_accepts(candidate) is False


@pytest.mark.parametrize(
    "path,expected",
    [
        ("lib/backend.so", True),
        ("lib//backend.so", True),
        ("lib/backend.so/", True),
        ("lib/backend plugin.so", True),
        ("./backend.so", False),
        ("lib/./backend.so", False),
        ("../backend.so", False),
        ("lib/../backend.so", False),
        ("/lib/backend.so", False),
        ("lib\\backend.so", False),
        ("lib/backend.so\n", False),
        (" lib/backend.so", False),
        ("lib/backend.so ", False),
    ],
    ids=[
        "relative",
        "double-separator",
        "trailing-separator",
        "internal-space",
        "leading-dot",
        "inner-dot",
        "leading-dotdot",
        "inner-dotdot",
        "absolute",
        "backslash",
        "line-break",
        "leading-space",
        "trailing-space",
    ],
)
def test_manifest_native_library_path_schema_parser_domain(
    path: str, expected: bool
) -> None:
    schema, example = _documents()
    del example  # The operational plugin document can only be python_only.
    path_schema = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": "#/$defs/nativeLibraryPath",
    }

    # Keep the evidence-only native path definition and lightweight parser
    # predicate aligned without allowing unsupported isolation through the
    # public Manifest parser.
    assert Draft202012Validator(path_schema).is_valid(path) is expected
    assert (_NATIVE_LIBRARY_PATH_PATTERN.fullmatch(path) is not None) is expected


@pytest.mark.parametrize(
    "priority,expected",
    [
        (1, True),
        (1.0, True),
        (-0.0, True),
        (1e20, True),
        (1.5, False),
        (True, False),
        (float("nan"), False),
        (float("inf"), False),
        (float("-inf"), False),
    ],
    ids=[
        "integer",
        "integral-float",
        "negative-zero",
        "large-integral-float",
        "fractional",
        "boolean",
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_manifest_priority_schema_parser_domain(
    priority: object, expected: bool
) -> None:
    schema, example = _documents()
    candidate = deepcopy(example)
    candidate["plugins"][0]["priority"] = priority

    assert _schema_accepts(schema, candidate) is expected
    assert _parser_accepts(candidate) is expected
    if expected:
        assert type(parse_manifest(candidate).plugins[0].priority) is int


@pytest.mark.parametrize("identity", ["plugin_id", "entry_point"])
def test_manifest_cross_record_uniqueness_is_semantic(
    identity: str,
) -> None:
    schema, example = _documents()
    candidate = deepcopy(example)
    duplicate = deepcopy(candidate["plugins"][0])
    duplicate["plugin_id"] = "example.second_backend"
    duplicate["entry_point"] = "second"
    duplicate[identity] = candidate["plugins"][0][identity]
    candidate["plugins"].append(duplicate)

    assert _schema_accepts(schema, candidate)
    document = parse_manifest(candidate)
    errors = evaluate_manifest_semantics(document)
    assert len(errors) == 2
    assert {error.field for error in errors} == {f"plugins[].{identity}"}
