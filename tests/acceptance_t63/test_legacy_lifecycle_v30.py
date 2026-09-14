"""Legacy selection and reset regressions from independent review."""

from test_legacy_bridge_v30 import LegacySpec, _manifest


def test_selected_legacy_can_be_rediscovered(legacy_runtime, monkeypatch):
    h = legacy_runtime
    s = h.fixture(LegacySpec("chosen", "sophgo"))
    h.install(s)
    record = h.registry.list()[0]
    monkeypatch.setenv("TRITON_ANCHOR_BACKEND", record.record_id)
    assert h.backends_module.get_backend(h.target("sophgo")).compiler is s.compiler_cls
    h.backends_module._discover_backends()
    assert h.backends_module.get_backend(h.target("sophgo")).compiler is s.compiler_cls


def test_valid_manifest_winner_with_bad_owner_and_legacy(legacy_runtime):
    h = legacy_runtime
    good = h.fixture(
        LegacySpec("good_owner", "sophgo", manifest=_manifest("good_owner", "sophgo"))
    )
    badm = _manifest("bad_owner", "sophgo")
    badm["plugins"][0]["requires_triton"]["version"] = ">=9.0,<10.0"
    bad = h.fixture(LegacySpec("bad_owner", "sophgo", manifest=badm))
    legacy = h.fixture(LegacySpec("old", "sophgo"))
    h.install(good, bad, legacy)
    assert type(h.runtime_driver_module._create_driver()) is good.driver_cls


def test_inactive_manifest_and_two_active_legacy_targets(legacy_runtime):
    h = legacy_runtime
    owner = h.fixture(
        LegacySpec("owner", "owned", active=False, manifest=_manifest("owner", "owned"))
    )
    shadow = h.fixture(
        LegacySpec("shadow", "owned", supports=lambda t: t.backend == "owned")
    )
    other = h.fixture(
        LegacySpec("other", "other", supports=lambda t: t.backend == "other")
    )
    h.install(owner, shadow, other)
    assert type(h.runtime_driver_module._create_driver()) is other.driver_cls


def test_reset_between_registration_and_publication_does_not_restore_stale_mapping(
    legacy_runtime, monkeypatch
):
    h = legacy_runtime
    old = h.fixture(LegacySpec("removed", "sophgo"))
    h.install(old)
    publish = h.backends_module._publish_legacy_backend

    def resetting_publish(record):
        h.registry.reset()
        h.distributions.clear()
        return publish(record)

    monkeypatch.setattr(h.backends_module, "_publish_legacy_backend", resetting_publish)
    h.backends_module._discover_backends()
    assert old.name not in h.backends_module.backends


def test_failed_old_publication_does_not_poison_same_id_after_reset(
    legacy_runtime, monkeypatch
):
    h = legacy_runtime
    old = h.fixture(LegacySpec("repaired", "sophgo"))
    new = h.fixture(LegacySpec("repaired", "sophgo"))
    h.install(old)
    publish = h.backends_module._publish_legacy_backend

    def resetting_failure(record):
        h.registry.reset()
        h.distributions[:] = [new.distribution]
        raise RuntimeError("obsolete-publication-error")

    monkeypatch.setattr(h.backends_module, "_publish_legacy_backend", resetting_failure)
    h.backends_module._discover_backends()
    monkeypatch.setattr(h.backends_module, "_publish_legacy_backend", publish)
    h.backends_module._discover_backends()
    assert h.backends_module.backends[new.name].compiler is new.compiler_cls


def test_clearing_env_selector_restores_implicit_legacy_driver(
    legacy_runtime, monkeypatch
):
    h = legacy_runtime
    s = h.fixture(LegacySpec("chosen_lifetime", "sophgo"))
    h.install(s)
    record = h.registry.list()[0]
    monkeypatch.setenv("TRITON_ANCHOR_BACKEND", record.record_id)
    assert type(h.runtime_driver_module._create_driver()) is s.driver_cls
    monkeypatch.delenv("TRITON_ANCHOR_BACKEND")
    assert type(h.runtime_driver_module._create_driver()) is s.driver_cls
    assert h.backends_module.get_backend(h.target("sophgo")).compiler is s.compiler_cls
