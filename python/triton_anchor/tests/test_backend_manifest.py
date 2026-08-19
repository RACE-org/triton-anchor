"""Tests for W3 static backend manifests and the Triton version example."""

import json
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

import pytest

from triton_anchor.backends import (
    BackendPluginCompatibilityError,
    BackendPluginManifestError,
    PluginIsolationMode,
    collect_core_environment,
    load_distribution_manifest,
    load_manifest,
    load_manifest_for_entry_point,
    parse_manifest,
    validate_triton_requirement,
)


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "backends"
    / "examples"
    / "triton_anchor_backend.example.json"
)
ABI_FINGERPRINT = "sha256:" + ("a" * 64)


def valid_manifest():
    return {
        "schema_version": "1.0",
        "plugins": [
            {
                "plugin_id": "vendor.mock",
                "entry_point": "mock",
                "backend_protocol": ">=1.0,<2.0",
                "requires_core": ">=0.2,<0.3",
                "requires_triton": {
                    "version": ">=3.0,<3.1",
                    "commit": "757b6a61e7df814ba806f498f8bb3160f84b120c",
                },
                "targets": ["mock"],
                "capabilities": ["anchor_ir.linalg"],
                "isolation_mode": "python_only",
            }
        ],
    }


@dataclass
class FakeEntryPoint:
    name: str
    group: str = "triton.backends"
    dist: object = None


class FakeDistribution:
    def __init__(
        self,
        root,
        *,
        manifest=None,
        entry_points=("mock",),
        duplicate_manifest=False,
    ):
        self.root = Path(root)
        self.metadata = {"Name": "vendor-backend"}
        self.version = "4.2.0"
        self.entry_points = [
            FakeEntryPoint(name=name, dist=self) for name in entry_points
        ]
        self.files = []
        if manifest is not None:
            first = PurePosixPath("vendor_backend/triton_anchor_backend.json")
            first_path = self.root / str(first)
            first_path.parent.mkdir(parents=True, exist_ok=True)
            first_path.write_text(json.dumps(manifest), encoding="utf-8")
            self.files.append(first)
            if duplicate_manifest:
                second = PurePosixPath("other/triton_anchor_backend.json")
                second_path = self.root / str(second)
                second_path.parent.mkdir(parents=True, exist_ok=True)
                second_path.write_text(json.dumps(manifest), encoding="utf-8")
                self.files.append(second)

    def locate_file(self, file):
        return self.root / str(file)


def test_packaged_example_is_valid():
    document = load_manifest(EXAMPLE_PATH)
    plugin = document.plugins[0]
    assert plugin.plugin_id == "example.mock_backend"
    assert plugin.isolation_mode is PluginIsolationMode.PYTHON_ONLY


def test_unknown_optional_fields_are_preserved_for_minor_schema_evolution():
    data = valid_manifest()
    data["future_root_field"] = {"enabled": True}
    data["plugins"][0]["future_plugin_field"] = 7
    data["plugins"][0]["requires_triton"]["future_requirement_field"] = "kept"
    document = parse_manifest(data)
    assert document.extensions["future_root_field"] == {"enabled": True}
    assert document.plugins[0].extensions["future_plugin_field"] == 7
    assert (
        document.plugins[0].requires_triton.extensions[
            "future_requirement_field"
        ]
        == "kept"
    )


@pytest.mark.parametrize(
    "mutator, match",
    [
        (lambda data: data.pop("schema_version"), "schema_version"),
        (lambda data: data.update(schema_version="2.0"), "Unsupported"),
        (lambda data: data.pop("plugins"), "plugins"),
        (
            lambda data: data["plugins"][0].pop("requires_triton"),
            "requires_triton",
        ),
        (
            lambda data: data["plugins"][0].update(isolation_mode="unknown"),
            "Unknown isolation_mode",
        ),
        (
            lambda data: data["plugins"][0].update(priority=True),
            "priority",
        ),
    ],
)
def test_invalid_manifest_fields_are_rejected(mutator, match):
    data = valid_manifest()
    mutator(data)
    with pytest.raises(BackendPluginManifestError, match=match):
        parse_manifest(data)


def test_duplicate_identity_is_rejected():
    data = valid_manifest()
    data["plugins"].append(dict(data["plugins"][0]))
    with pytest.raises(BackendPluginManifestError, match="duplicate plugin_id"):
        parse_manifest(data)


def test_native_in_process_requires_native_libraries_and_abi_fingerprint():
    data = valid_manifest()
    data["plugins"][0]["isolation_mode"] = "native_in_process"
    with pytest.raises(BackendPluginManifestError, match="native_libraries"):
        parse_manifest(data)

    data["plugins"][0]["native_libraries"] = ["vendor_backend/libvendor.so"]
    with pytest.raises(BackendPluginManifestError, match="abi_fingerprint"):
        parse_manifest(data)

    data["plugins"][0]["abi_fingerprint"] = ABI_FINGERPRINT
    plugin = parse_manifest(data).plugins[0]
    assert plugin.native_libraries == ("vendor_backend/libvendor.so",)
    assert plugin.abi_fingerprint == ABI_FINGERPRINT


@pytest.mark.parametrize(
    "value",
    [
        "sha256:placeholder",
        "sha256:" + ("a" * 63),
        "sha256:" + ("a" * 65),
        "sha256:" + ("A" * 64),
        "SHA256:" + ("a" * 64),
        "a" * 64,
    ],
)
def test_abi_fingerprint_has_one_canonical_format(value):
    data = valid_manifest()
    data["plugins"][0].update(
        isolation_mode="native_in_process",
        native_libraries=["vendor_backend/libvendor.so"],
        abi_fingerprint=value,
    )
    with pytest.raises(
        BackendPluginManifestError,
        match="64 lowercase hexadecimal characters",
    ) as caught:
        parse_manifest(data)
    diagnostic = caught.value.to_dict()
    assert diagnostic["field"] == "abi_fingerprint"
    assert diagnostic["expected"]
    assert diagnostic["actual"] == value
    assert diagnostic["remediation"]


@pytest.mark.parametrize("value", [[], ["libvendor.so"]])
def test_python_only_cannot_claim_native_libraries(value):
    data = valid_manifest()
    data["plugins"][0]["native_libraries"] = value
    with pytest.raises(BackendPluginManifestError, match="python_only"):
        parse_manifest(data)


def test_python_only_cannot_claim_abi_fingerprint():
    data = valid_manifest()
    data["plugins"][0]["abi_fingerprint"] = ABI_FINGERPRINT
    with pytest.raises(BackendPluginManifestError, match="python_only"):
        parse_manifest(data)


def test_subprocess_may_package_native_libraries_but_not_core_abi_fingerprint():
    data = valid_manifest()
    data["plugins"][0].update(
        isolation_mode="subprocess",
        native_libraries=["vendor_backend/libworker.so"],
    )
    plugin = parse_manifest(data).plugins[0]
    assert plugin.native_libraries == ("vendor_backend/libworker.so",)
    assert plugin.abi_fingerprint is None

    data["plugins"][0]["abi_fingerprint"] = ABI_FINGERPRINT
    with pytest.raises(BackendPluginManifestError, match="subprocess"):
        parse_manifest(data)


def test_distribution_without_manifest_is_legacy(tmp_path):
    distribution = FakeDistribution(tmp_path)
    assert load_distribution_manifest(distribution) is None
    assert load_manifest_for_entry_point(distribution.entry_points[0]) is None


def test_missing_distribution_file_list_is_not_assumed_legacy(tmp_path):
    distribution = FakeDistribution(tmp_path)
    distribution.files = None
    with pytest.raises(BackendPluginManifestError, match="file list"):
        load_distribution_manifest(distribution)


def test_entry_point_without_distribution_is_not_assumed_legacy():
    entry_point = FakeEntryPoint(name="mock", dist=None)
    with pytest.raises(BackendPluginManifestError, match="owning distribution"):
        load_manifest_for_entry_point(entry_point)


def test_python38_style_entry_point_accepts_explicit_distribution(tmp_path):
    distribution = FakeDistribution(tmp_path, manifest=valid_manifest())
    entry_point = FakeEntryPoint(name="mock", dist=None)
    plugin = load_manifest_for_entry_point(entry_point, distribution)
    assert plugin.plugin_id == "vendor.mock"


def test_distribution_identity_comes_from_metadata(tmp_path):
    distribution = FakeDistribution(tmp_path, manifest=valid_manifest())
    document = load_distribution_manifest(distribution)
    plugin = document.plugins[0]
    assert plugin.distribution_name == "vendor-backend"
    assert plugin.distribution_version == "4.2.0"


def test_distribution_manifest_must_cover_all_backend_entry_points(tmp_path):
    distribution = FakeDistribution(
        tmp_path,
        manifest=valid_manifest(),
        entry_points=("mock", "other"),
    )
    with pytest.raises(BackendPluginManifestError, match="missing records for other"):
        load_distribution_manifest(distribution)


def test_distribution_manifest_cannot_declare_unknown_entry_point(tmp_path):
    distribution = FakeDistribution(
        tmp_path,
        manifest=valid_manifest(),
        entry_points=(),
    )
    with pytest.raises(BackendPluginManifestError, match="unknown records for mock"):
        load_distribution_manifest(distribution)


def test_distribution_must_contain_only_one_manifest_file(tmp_path):
    distribution = FakeDistribution(
        tmp_path,
        manifest=valid_manifest(),
        duplicate_manifest=True,
    )
    with pytest.raises(BackendPluginManifestError, match="multiple"):
        load_distribution_manifest(distribution)


def test_triton_version_example_accepts_current_environment():
    plugin = parse_manifest(valid_manifest()).plugins[0]
    validate_triton_requirement(plugin, collect_core_environment())


def test_triton_version_example_rejects_before_plugin_load():
    plugin = parse_manifest(valid_manifest()).plugins[0]
    environment = replace(collect_core_environment(), triton_version="2.0.0")
    with pytest.raises(
        BackendPluginCompatibilityError, match="Incompatible Triton version"
    ):
        validate_triton_requirement(plugin, environment)


def test_triton_commit_mismatch_is_rejected():
    data = valid_manifest()
    data["plugins"][0]["requires_triton"]["commit"] = "f" * 40
    plugin = parse_manifest(data).plugins[0]
    with pytest.raises(
        BackendPluginCompatibilityError, match="vendored Triton commit"
    ):
        validate_triton_requirement(plugin, collect_core_environment())


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("plugin_id", "Vendor.Mock", "plugin_id"),
        ("plugin_id", " vendor.mock", "whitespace"),
        ("entry_point", "bad entry", "entry_point"),
    ],
)
def test_identity_fields_are_canonical(field, value, match):
    data = valid_manifest()
    data["plugins"][0][field] = value
    with pytest.raises(BackendPluginManifestError, match=match):
        parse_manifest(data)


@pytest.mark.parametrize(
    "path",
    ["/absolute/lib.so", "../escape/lib.so", "dir\\lib.so", "."],
)
def test_native_library_paths_cannot_escape_distribution(path):
    data = valid_manifest()
    data["plugins"][0]["isolation_mode"] = "subprocess"
    data["plugins"][0]["native_libraries"] = [path]
    with pytest.raises(BackendPluginManifestError, match="relative package paths"):
        parse_manifest(data)


def test_invalid_triton_specifier_is_a_manifest_error():
    data = valid_manifest()
    data["plugins"][0]["requires_triton"]["version"] = "not a specifier"
    with pytest.raises(BackendPluginManifestError, match="invalid"):
        parse_manifest(data)


def test_malformed_backend_protocol_specifier_is_structured_manifest_error():
    data = valid_manifest()
    data["plugins"][0]["backend_protocol"] = "not a specifier"

    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(data)

    diagnostic = caught.value.to_dict()
    assert diagnostic["code"] == "backend_plugin_manifest_error"
    assert diagnostic["field"] == "backend_protocol"
    assert diagnostic["expected"] == "a valid PEP 440 version specifier"
    assert diagnostic["actual"] == "not a specifier"
    assert diagnostic["remediation"]


def test_manifest_records_exact_producer_protocol_version():
    data = valid_manifest()
    data["plugins"][0]["producer_protocol_version"] = "1.2"

    plugin = parse_manifest(data).plugins[0]

    assert plugin.producer_protocol_version == "1.2"
    assert "producer_protocol_version" not in plugin.extensions


@pytest.mark.parametrize("value", [">=1.0,<2.0", "not-a-version"])
def test_producer_protocol_version_must_be_exact(value):
    data = valid_manifest()
    data["plugins"][0]["producer_protocol_version"] = value

    with pytest.raises(BackendPluginManifestError) as caught:
        parse_manifest(data)

    diagnostic = caught.value.to_dict()
    assert diagnostic["code"] == "backend_plugin_manifest_error"
    assert diagnostic["field"] == "producer_protocol_version"
    assert diagnostic["expected"] == "an exact PEP 440 version"
    assert diagnostic["actual"] == value
    assert diagnostic["remediation"]


@pytest.mark.parametrize(
    "field",
    [
        "display_name",
        "requires_core",
        "capabilities",
        "native_libraries",
        "abi_fingerprint",
    ],
)
def test_optional_manifest_fields_reject_explicit_null(field):
    data = valid_manifest()
    data["plugins"][0][field] = None
    with pytest.raises(BackendPluginManifestError, match=field):
        parse_manifest(data)


def test_duplicate_json_keys_are_rejected(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"schema_version":"1.0","schema_version":"1.1","plugins":[]}',
        encoding="utf-8",
    )
    with pytest.raises(BackendPluginManifestError, match="duplicate JSON field"):
        load_manifest(manifest)


def test_non_utf8_manifest_is_a_structured_error(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b"\xff")
    with pytest.raises(BackendPluginManifestError, match="Unable to read"):
        load_manifest(manifest)
