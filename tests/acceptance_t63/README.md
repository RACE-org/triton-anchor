# T6.3 acceptance baseline

This directory freezes the independent Triton 3.3 acceptance oracles for
Backend Plugin Protocol 1.0.  The tests intentionally preserve known failures;
do not weaken, skip, or xfail them to accommodate an implementation.

## Oracles

- Manifest acceptance comes from the shipped JSON Schema.
- Lifecycle behavior comes from the public `BackendPluginBase` contract.
- Compatibility aggregation comes from the acceptance requirement that every
  independent failure is reported before plugin import.
- Legacy behavior is compared with the frozen `upstream/triton_v3.3` commit
  `31b073040a6d11afa63c7a062d24d546e3dd887d` using the same real entry-point
  wheel.
- Triton 3.3 compiler/driver requirements come from its public abstract base
  classes.
- Native checks use wheel `RECORD`, ELF metadata, `nm`, `readelf`, and the
  published Core ABI material.  Tests that prove an undefined rule are recorded
  as `SPEC GAP`, not as acceptance passes.

No test imports or depends on the excluded T10.2 Conformance Kit.  Repository
paths are derived from `__file__`; no developer-machine import path is embedded
in the test code.

## Canonical invocation

The test interpreter must provide `pytest`, `jsonschema`, `packaging`, and the
standard wheel-building test dependencies.  Supply the exact Core wheel through
`T63_CORE_WHEEL`; without it, wheel-gated nodes are deliberately skipped and the
result is not the frozen 177-node baseline.

```bash
cd /path/to/triton-anchor-t63-v3.3

export PYTHONPATH="$PWD/python:$PWD/triton/python"
export PYTHONDONTWRITEBYTECODE=1
export T63_PYTHON=/path/to/t63-acceptance-venv/bin/python
export T63_CORE_WHEEL=/path/to/triton_anchor-0.2.0-cp312-cp312-linux_x86_64.whl

"$T63_PYTHON" -m pytest -q \
  python/triton_anchor/tests \
  tests/acceptance_t63/test_registry_acceptance.py \
  tests/acceptance_t63/test_schema_oracle_acceptance.py \
  tests/acceptance_t63/test_native_abi_acceptance.py \
  tests/acceptance_t63/test_packaging_wheel.py \
  tests/acceptance_t63/test_packaging_entrypoint_e2e.py \
  tests/acceptance_t63/test_triton_integration_acceptance.py \
  -ra \
  --junitxml=/path/outside/worktree/phase1_baseline.xml \
  -o cache_dir=/path/outside/worktree/pytest-cache-phase1-baseline
```

The frozen result and evidence hashes are recorded in
`baseline_manifest.json`.  Logs, JUnit, wheels, pytest caches, coverage data,
and compiler build trees remain outside the Git worktree.
