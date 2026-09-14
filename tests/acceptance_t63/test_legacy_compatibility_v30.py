"""Legacy compatibility acceptance of the v3.0 candidate; MOCK LOGIC ONLY.

The reused fixture imports the real registry, backends bridge and runtime driver
code, but supplies synthetic distribution metadata, plugin classes, and a shell
triton package. Its triton_version preflight profile is deliberately limited.
This suite is not a native ABI, installed wheel, Sophgo, or JIT integration test.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from test_legacy_bridge_v30 import (
    LegacySpec,
    _manifest,
)
from triton_anchor.backends.errors import (
    BackendPluginCompatibilityError,
    BackendPluginConflictError,
    BackendPluginLoadError,
    BackendPluginSelectionError,
)


def _record(harness, spec):
    matches = [r for r in harness.registry.list() if r.entry_point_name == spec.name]
    assert len(matches) == 1
    return matches[0]


def _target_match(name):
    return lambda target: target.backend == name


def test_legacy_metadata_is_unverified_and_discovery_does_not_import(legacy_runtime):
    spec = legacy_runtime.fixture(LegacySpec("legacy_metadata", "sophgo"))
    legacy_runtime.install(spec)
    record = _record(legacy_runtime, spec)
    assert record.manifest is None
    assert record.compatibility_status.value == "legacy_unverified"
    assert spec.entry_point.load_calls == 0


def test_legacy_runtime_and_compiler_remain_paired_and_unverified(legacy_runtime):
    spec = legacy_runtime.fixture(LegacySpec("legacy_pair", "sophgo"))
    legacy_runtime.install(spec)
    driver = legacy_runtime.runtime_driver_module._create_driver()
    compiler = legacy_runtime.backends_module.make_backend(driver.get_current_target())
    assert type(driver) is spec.driver_cls
    assert type(compiler) is spec.compiler_cls
    assert spec.entry_point.load_calls == 1
    assert (
        _record(legacy_runtime, spec).compatibility_status.value == "legacy_unverified"
    )


@pytest.mark.parametrize(
    "broken_first", [True, False], ids=["broken-first", "healthy-first"]
)
def test_broken_legacy_import_keeps_healthy_peer_usable(
    legacy_runtime, broken_first, capsys
):
    """Old behavior is warning/diagnosis and continue, not a global raise."""
    bad_name, good_name = (
        ("a_broken", "z_good") if broken_first else ("z_broken", "a_good")
    )
    bad = legacy_runtime.fixture(
        LegacySpec(
            bad_name, "broken", load_error=RuntimeError("legacy-import-sentinel")
        )
    )
    good = legacy_runtime.fixture(LegacySpec(good_name, "sophgo"))
    legacy_runtime.install(*((bad, good) if broken_first else (good, bad)))

    mapping = legacy_runtime.backends_module._discover_backends()

    assert bad.name not in mapping
    assert mapping[good.name].driver is good.driver_cls
    bad_record = _record(legacy_runtime, bad)
    assert isinstance(bad_record.error, BackendPluginLoadError)
    assert "legacy-import-sentinel" in str(bad_record.error)
    diagnostics = capsys.readouterr().err
    assert bad.name in diagnostics
    assert "legacy-import-sentinel" in diagnostics
    driver = legacy_runtime.runtime_driver_module._create_driver()
    assert type(driver) is good.driver_cls


def test_initial_backend_module_import_survives_unrelated_broken_legacy(legacy_runtime):
    bad = legacy_runtime.fixture(
        LegacySpec(
            "a_broken_import",
            "broken",
            load_error=RuntimeError("initial-import-sentinel"),
        )
    )
    good = legacy_runtime.fixture(LegacySpec("z_healthy_import", "sophgo"))
    legacy_runtime.install(bad, good)
    original = legacy_runtime.backends_module
    sys.modules.pop("triton.backends", None)
    try:
        imported = importlib.import_module("triton.backends")
        assert imported.backends[good.name].driver is good.driver_cls
        assert bad.name not in imported.backends
    finally:
        sys.modules["triton.backends"] = original
        sys.modules["triton"].backends = original


@pytest.mark.parametrize(
    "manifest_first", [True, False], ids=["manifest-first", "legacy-first"]
)
def test_inactive_manifest_other_target_does_not_hide_active_legacy(
    legacy_runtime, manifest_first
):
    """Exercise actual _create_driver; Manifest and Legacy target DIFFERENT hardware."""
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "inactive_manifest",
            "manifest_hw",
            active=False,
            supports=_target_match("manifest_hw"),
            manifest=_manifest("inactive_manifest", "manifest_hw"),
        )
    )
    legacy = legacy_runtime.fixture(
        LegacySpec(
            "active_legacy",
            "sophgo",
            active=True,
            supports=_target_match("sophgo"),
        )
    )
    legacy_runtime.install(
        *((manifest, legacy) if manifest_first else (legacy, manifest))
    )

    driver = legacy_runtime.runtime_driver_module._create_driver()

    assert type(driver) is legacy.driver_cls
    backend = legacy_runtime.backends_module.get_backend(driver.get_current_target())
    assert backend.compiler is legacy.compiler_cls
    assert backend.driver is legacy.driver_cls
    assert manifest.calls["driver_constructor"] == 0
    assert (
        _record(legacy_runtime, legacy).compatibility_status.value
        == "legacy_unverified"
    )


def test_active_manifest_and_active_legacy_other_targets_report_ambiguity(
    legacy_runtime,
):
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "active_manifest",
            "manifest_hw",
            supports=_target_match("manifest_hw"),
            manifest=_manifest("active_manifest", "manifest_hw"),
        )
    )
    legacy = legacy_runtime.fixture(
        LegacySpec("active_legacy", "sophgo", supports=_target_match("sophgo"))
    )
    legacy_runtime.install(manifest, legacy)
    with pytest.raises(BackendPluginConflictError) as caught:
        legacy_runtime.runtime_driver_module._create_driver()
    assert caught.value.field == "driver_cls.is_active"
    # A manifestless driver may need one construction to identify its real
    # hardware target; compiler.supports_target is not ownership metadata.
    assert manifest.calls["driver_constructor"] == 0
    assert legacy.calls["driver_constructor"] <= 1
    assert legacy.calls["compiler_constructor"] == 0


def test_two_legacy_drivers_choose_unique_active_driver_and_matching_compiler(
    legacy_runtime,
):
    inactive = legacy_runtime.fixture(
        LegacySpec(
            "inactive_legacy",
            "other_hw",
            active=False,
            supports=_target_match("other_hw"),
        )
    )
    active = legacy_runtime.fixture(
        LegacySpec("active_legacy", "sophgo", supports=_target_match("sophgo"))
    )
    legacy_runtime.install(inactive, active)
    driver = legacy_runtime.runtime_driver_module._create_driver()
    compiler = legacy_runtime.backends_module.make_backend(driver.get_current_target())
    assert type(driver) is active.driver_cls
    assert type(compiler) is active.compiler_cls
    assert inactive.calls["driver_constructor"] == 0


def test_two_active_legacy_drivers_report_ambiguity_before_construction(legacy_runtime):
    first = legacy_runtime.fixture(LegacySpec("first_legacy", "first_hw"))
    second = legacy_runtime.fixture(LegacySpec("second_legacy", "second_hw"))
    legacy_runtime.install(first, second)
    with pytest.raises(BackendPluginConflictError) as caught:
        legacy_runtime.runtime_driver_module._create_driver()
    assert caught.value.field == "driver_cls.is_active"
    assert first.calls["driver_constructor"] == second.calls["driver_constructor"] == 0


def test_same_target_manifest_owner_has_priority_over_legacy(legacy_runtime):
    owner = legacy_runtime.fixture(
        LegacySpec("owner", "sophgo", manifest=_manifest("owner", "sophgo"))
    )
    legacy = legacy_runtime.fixture(LegacySpec("legacy_shadow", "sophgo"))
    legacy_runtime.install(owner, legacy)
    driver = legacy_runtime.runtime_driver_module._create_driver()
    compiler = legacy_runtime.backends_module.make_backend(driver.get_current_target())
    assert type(driver) is owner.driver_cls
    assert type(compiler) is owner.compiler_cls
    # Legacy distributions have no target metadata: discovery may import
    # them to determine overlap, but a Manifest owner must retain selection.
    assert legacy.calls["driver_constructor"] <= 1
    assert legacy.calls["compiler_constructor"] == 0
    assert legacy.calls["initialize"] == 0
    assert _record(legacy_runtime, legacy).state.value != "active"


def test_explicit_inactive_manifest_does_not_silently_fallback(
    legacy_runtime, monkeypatch
):
    owner = legacy_runtime.fixture(
        LegacySpec(
            "forced_manifest",
            "manifest_hw",
            active=False,
            manifest=_manifest("forced_manifest", "manifest_hw"),
        )
    )
    legacy = legacy_runtime.fixture(LegacySpec("otherwise_active", "sophgo"))
    legacy_runtime.install(owner, legacy)
    monkeypatch.setenv(
        "TRITON_ANCHOR_BACKEND", owner.manifest["plugins"][0]["plugin_id"]
    )
    with pytest.raises(BackendPluginSelectionError) as caught:
        legacy_runtime.runtime_driver_module._create_driver()
    assert caught.value.field == "driver_cls.is_active"
    assert caught.value.actual == "0"
    assert legacy.entry_point.load_calls == 0


def test_explicit_legacy_selector_pairs_runtime_and_compiler(
    legacy_runtime, monkeypatch
):
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "unselected_manifest",
            "manifest_hw",
            manifest=_manifest("unselected_manifest", "manifest_hw"),
        )
    )
    legacy = legacy_runtime.fixture(LegacySpec("chosen_legacy", "sophgo"))
    legacy_runtime.install(manifest, legacy)
    monkeypatch.setenv(
        "TRITON_ANCHOR_BACKEND", _record(legacy_runtime, legacy).record_id
    )
    driver = legacy_runtime.runtime_driver_module._create_driver()
    compiler = legacy_runtime.backends_module.make_backend(driver.get_current_target())
    assert type(driver) is legacy.driver_cls
    assert type(compiler) is legacy.compiler_cls
    assert manifest.entry_point.load_calls == 0


def test_wrong_triton_version_rejected_for_exact_reason_before_import(legacy_runtime):
    name = "wrong_triton"
    manifest = _manifest(name, "sophgo")
    manifest["plugins"][0]["requires_triton"]["version"] = ">=9.0,<10.0"
    spec = legacy_runtime.fixture(LegacySpec(name, "sophgo", manifest=manifest))
    legacy_runtime.install(spec)
    record = _record(legacy_runtime, spec)
    assert record.manifest.entry_point == spec.entry_point.name
    with pytest.raises(BackendPluginCompatibilityError) as caught:
        legacy_runtime.registry.validate(record.record_id)
    assert caught.value.dimension == "Triton version"
    assert caught.value.actual == "3.0.0"
    assert caught.value.expected == ">=9.0,<10.0"
    assert caught.value.entry_point == name
    assert spec.entry_point.load_calls == 0
    assert _record(legacy_runtime, spec).state.value == "rejected"


@pytest.mark.parametrize("resolution", ["compiler", "runtime"])
def test_rejected_same_target_manifest_is_not_bypassed_via_legacy(
    legacy_runtime, resolution
):
    name = "bad_version_owner"
    manifest = _manifest(name, "sophgo")
    manifest["plugins"][0]["requires_triton"]["version"] = ">=9.0,<10.0"
    owner = legacy_runtime.fixture(LegacySpec(name, "sophgo", manifest=manifest))
    legacy = legacy_runtime.fixture(LegacySpec("legacy_bypass", "sophgo"))
    legacy_runtime.install(owner, legacy)
    with pytest.raises(BackendPluginCompatibilityError) as caught:
        if resolution == "compiler":
            legacy_runtime.backends_module.get_backend(legacy_runtime.target("sophgo"))
        else:
            legacy_runtime.runtime_driver_module._create_driver()
    assert caught.value.dimension == "Triton version"
    assert owner.entry_point.load_calls == 0
    if resolution == "compiler":
        assert legacy.entry_point.load_calls == 0
    else:
        # Runtime must obtain the manifestless driver's actual target before
        # determining that the rejected Manifest governs it.
        assert legacy.entry_point.load_calls <= 1
        assert legacy.calls["driver_constructor"] <= 1
        assert legacy.calls["compiler_constructor"] == 0
        assert _record(legacy_runtime, legacy).state.value != "active"


@pytest.mark.parametrize("resolution", ["compiler", "runtime"])
def test_unknown_explicit_selector_is_not_ignored_by_fallback(
    legacy_runtime, monkeypatch, resolution
):
    spec = legacy_runtime.fixture(LegacySpec("available_legacy", "sophgo"))
    legacy_runtime.install(spec)
    monkeypatch.setenv("TRITON_ANCHOR_BACKEND", "does-not-exist")
    with pytest.raises(BackendPluginSelectionError) as caught:
        if resolution == "compiler":
            legacy_runtime.backends_module.get_backend(legacy_runtime.target("sophgo"))
        else:
            legacy_runtime.runtime_driver_module._create_driver()
    assert caught.value.field == "backend_selector"
    assert caught.value.actual == "does-not-exist"
    assert spec.entry_point.load_calls == 0


@pytest.mark.parametrize(
    "manifest_active", [False, True], ids=["inactive-owner", "active-owner"]
)
def test_broad_legacy_compiler_support_does_not_hide_actual_other_target(
    legacy_runtime, manifest_active
):
    """A broad compiler predicate must not assign the driver's hardware target."""
    manifest = legacy_runtime.fixture(
        LegacySpec(
            "broad_owner",
            "manifest_hw",
            active=manifest_active,
            manifest=_manifest("broad_owner", "manifest_hw"),
        )
    )
    legacy = legacy_runtime.fixture(
        LegacySpec(
            "broad_legacy",
            "sophgo",
            supports=True,
        )
    )
    legacy_runtime.install(manifest, legacy)

    if manifest_active:
        with pytest.raises(BackendPluginConflictError) as caught:
            legacy_runtime.runtime_driver_module._create_driver()
        assert caught.value.field == "driver_cls.is_active"
        assert legacy.calls["compiler_constructor"] == 0
    else:
        driver = legacy_runtime.runtime_driver_module._create_driver()
        compiler = legacy_runtime.backends_module.make_backend(
            driver.get_current_target()
        )
        assert type(driver) is legacy.driver_cls
        assert type(compiler) is legacy.compiler_cls
        assert driver.get_current_target().backend == "sophgo"
    assert legacy.calls["driver_constructor"] == 1
    assert manifest.calls["driver_constructor"] == 0


def test_inactive_manifest_same_target_does_not_enable_legacy_bypass(legacy_runtime):
    owner = legacy_runtime.fixture(
        LegacySpec(
            "inactive_same_owner",
            "sophgo",
            active=False,
            manifest=_manifest("inactive_same_owner", "sophgo"),
        )
    )
    legacy = legacy_runtime.fixture(LegacySpec("active_same_legacy", "sophgo"))
    legacy_runtime.install(owner, legacy)

    with pytest.raises(BackendPluginSelectionError) as caught:
        legacy_runtime.runtime_driver_module._create_driver()

    # Either selection stage may reject: no eligible active Manifest, or
    # a Legacy driver discovered to overlap the governed target.
    if caught.value.field == "targets":
        assert caught.value.expected == _record(legacy_runtime, owner).record_id
        assert caught.value.actual == "manual Legacy backend"
    else:
        assert caught.value.field == "driver_cls.is_active"
        assert caught.value.actual == "0"
    assert owner.calls["driver_constructor"] == 0
    assert legacy.calls["compiler_constructor"] == 0
    assert _record(legacy_runtime, legacy).state.value != "active"
