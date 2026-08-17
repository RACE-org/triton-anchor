"""Tests for W6 static capability declarations and subset negotiation."""

from dataclasses import fields

import pytest

from triton_anchor.backends.capabilities import (
    evaluate_capabilities,
    evaluate_plugin_capabilities,
    validate_capabilities,
    validate_plugin_capabilities,
)
from triton_anchor.backends.errors import (
    BackendPluginCapabilityError,
    BackendPluginManifestError,
)
from triton_anchor.backends.manifest import parse_manifest
from triton_anchor.backends.manifest import BackendPluginManifest


def manifest_data():
    return {
        "schema_version": "1.0",
        "plugins": [
            {
                "plugin_id": "vendor.mock",
                "entry_point": "mock",
                "backend_protocol": ">=1.0,<2.0",
                "requires_triton": {"version": ">=3.0,<3.1"},
                "targets": ["mock"],
                "capabilities": ["plugin.matmul", "plugin.reduce"],
                "requires_capabilities": ["core.ir.v1"],
                "isolation_mode": "python_only",
            }
        ],
    }


def test_manifest_distinguishes_provided_and_required_capabilities():
    plugin = parse_manifest(manifest_data()).plugins[0]
    assert plugin.capabilities == ("plugin.matmul", "plugin.reduce")
    assert plugin.requires_capabilities == ("core.ir.v1",)
    assert "requires_capabilities" not in plugin.extensions


def test_requires_capabilities_is_optional():
    data = manifest_data()
    data["plugins"][0].pop("requires_capabilities")
    assert parse_manifest(data).plugins[0].requires_capabilities == ()


def test_new_manifest_field_is_appended_for_positional_compatibility():
    assert fields(BackendPluginManifest)[-1].name == "requires_capabilities"


@pytest.mark.parametrize(
    "value",
    [
        None,
        "core.ir.v1",
        [""],
        [" core.ir.v1"],
        ["core.ir.v1", "core.ir.v1"],
        [7],
    ],
)
def test_manifest_rejects_invalid_required_capability_names(value):
    data = manifest_data()
    data["plugins"][0]["requires_capabilities"] = value
    with pytest.raises(
        BackendPluginManifestError, match="requires_capabilities"
    ):
        parse_manifest(data)


def test_plugin_requirements_are_satisfied_only_by_core():
    report = evaluate_capabilities(
        core_provided=["core.ir.v1"],
        plugin_provided=["plugin.matmul"],
        plugin_required=["core.ir.v1"],
    )
    assert report.compatible
    assert report.missing_for_plugin == ()

    report = evaluate_capabilities(
        core_provided=[],
        plugin_provided=["core.ir.v1"],
        plugin_required=["core.ir.v1"],
    )
    assert not report.compatible
    assert report.missing_for_plugin == ("core.ir.v1",)


def test_kernel_requirements_use_union_of_core_and_plugin_capabilities():
    report = evaluate_capabilities(
        core_provided=["core.ir.v1"],
        plugin_provided=["plugin.matmul"],
        kernel_required=["plugin.matmul", "core.ir.v1"],
    )
    assert report.compatible
    assert report.kernel_available == ("core.ir.v1", "plugin.matmul")


def test_exact_capability_names_are_case_sensitive_and_not_inferred():
    report = evaluate_capabilities(
        core_provided=["Core.IR.V1"],
        plugin_provided=[],
        plugin_required=["core.ir.v1"],
    )
    assert report.missing_for_plugin == ("core.ir.v1",)


def test_report_and_missing_values_are_deterministically_sorted():
    report = evaluate_capabilities(
        core_provided=["core.z", "core.a"],
        plugin_provided=["plugin.z", "plugin.a"],
        plugin_required=["missing.z", "missing.a"],
        kernel_required=["missing.kernel.z", "missing.kernel.a"],
    )
    assert report.core_provided == ("core.a", "core.z")
    assert report.plugin_provided == ("plugin.a", "plugin.z")
    assert report.missing_for_plugin == ("missing.a", "missing.z")
    assert report.missing_for_kernel == (
        "missing.kernel.a",
        "missing.kernel.z",
    )
    assert report.missing == (
        "missing.a",
        "missing.kernel.a",
        "missing.kernel.z",
        "missing.z",
    )
    assert report.to_dict()["missing"] == list(report.missing)


def test_validation_error_is_structured_and_lists_each_missing_scope():
    with pytest.raises(BackendPluginCapabilityError) as caught:
        validate_capabilities(
            core_provided=[],
            plugin_provided=[],
            plugin_required=["core.required"],
            kernel_required=["kernel.required"],
            plugin_id="vendor.mock",
            entry_point="mock",
        )

    error = caught.value
    assert error.scope == "plugin_and_kernel"
    assert error.missing_capabilities == (
        "core.required",
        "kernel.required",
    )
    assert error.to_dict()["code"] == "backend_plugin_capability_error"
    assert error.to_dict()["missing_capabilities"] == [
        "core.required",
        "kernel.required",
    ]
    assert error.to_dict()["plugin_id"] == "vendor.mock"


def test_manifest_convenience_functions_preserve_plugin_identity():
    plugin = parse_manifest(manifest_data()).plugins[0]
    report = evaluate_plugin_capabilities(
        plugin,
        core_provided=["core.ir.v1"],
        kernel_required=["plugin.matmul"],
    )
    assert report.compatible
    assert report.plugin_id == "vendor.mock"
    assert report.entry_point == "mock"

    with pytest.raises(BackendPluginCapabilityError) as caught:
        validate_plugin_capabilities(plugin, core_provided=[])
    assert caught.value.plugin_id == "vendor.mock"
    assert caught.value.missing_capabilities == ("core.ir.v1",)


@pytest.mark.parametrize(
    "argument,value",
    [
        ("core_provided", [""]),
        ("plugin_provided", ["duplicate", "duplicate"]),
        ("plugin_required", "one.capability"),
        ("kernel_required", [" spaced"]),
    ],
)
def test_runtime_capability_inputs_must_be_unique_non_empty_strings(
    argument, value
):
    arguments = {
        "core_provided": [],
        "plugin_provided": [],
        "plugin_required": [],
        "kernel_required": [],
    }
    arguments[argument] = value
    with pytest.raises(ValueError, match=argument):
        evaluate_capabilities(**arguments)
