# Triton 3.0 Legacy compatibility regressions

These tests cover the Legacy bridge, coexistence with Manifest plugins, explicit
selection, rediscovery, and Registry reset. They load the Triton bridge and runtime
Python modules from this checkout with isolated plugin metadata and valid concrete
BaseBackend/DriverBase fixtures. They do not execute kernels or replace native
wheel/JIT acceptance.

Run from the repository root with Python 3.12 and pytest/packaging available:

    PYTHONPATH="$PWD/python" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/acceptance_t63 -q

Requirements retained by the tests:

- Existing no-Manifest entry points remain available as Legacy and retain the
  legacy_unverified compatibility status.
- A failed Legacy import/publication emits a warning and does not remove healthy
  peers. Explicitly selecting the failed record still reports its failure.
- An unrelated Manifest target does not hide a healthy Legacy target.
- A Legacy target does not bypass Manifest ownership/rejection; actual driver
  targets, including architecture-dependent targets, determine overlap.
- Explicit selection errors propagate; clearing an environment selector restores
  implicit Legacy driver discovery.
- Rediscovery and Registry reset cannot restore stale mappings or failures.

The real Sophgo CModel, installed-wheel Legacy, native ABI/conflict, governance,
and protocol-evolution evidence is maintained in the separate acceptance output
directory. Passing this suite alone is not a full T6.3 or T10.2 verdict.
