"""Cross-check the shipped Manifest Schema against the public parser."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from triton_anchor.backends import BackendPluginManifestError, parse_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKENDS = REPO_ROOT / "python" / "triton_anchor" / "backends"
SCHEMA_PATH = BACKENDS / "schemas" / "backend_manifest.schema.json"
EXAMPLE_PATH = BACKENDS / "examples" / "triton_anchor_backend.example.json"


def _documents() -> tuple[dict, dict]:
    return (
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
        json.loads(EXAMPLE_PATH.read_text(encoding="utf-8")),
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

    JSON Schema's ``nonEmptyString`` pattern rejects embedded newlines because
    ``.`` does not match them.  The runtime reader must not silently accept a
    document that its distributed Schema declares invalid.
    """

    schema, example = _documents()
    candidate = deepcopy(example)
    candidate["plugins"][0]["display_name"] = "line one\nline two"
    errors = list(Draft202012Validator(schema).iter_errors(candidate))
    assert len(errors) == 1
    assert list(errors[0].absolute_path) == ["plugins", 0, "display_name"]
    with pytest.raises(BackendPluginManifestError):
        parse_manifest(candidate)
