# Triton 3.6 T6.3 common acceptance baseline

This directory freezes independent Triton 3.6 acceptance oracles for the
cross-version portion of Backend Plugin Protocol 1.0. The tests preserve known
failures and specification gaps until an approved contract changes their
oracle; they must not be weakened, skipped, or xfailed to match implementation
behavior.

## Scope classification

The Triton 3.3 suite was used only as a test-design source and was split before
porting:

- Class A, included here: Manifest Schema/parser equivalence, Registry
  compatibility diagnostics, lifecycle hooks, real entry-point metadata,
  pre-import sentinels, native static integrity and dynamic-load gap witnesses,
  selection, concurrency, and reset.
- Class B, deliberately excluded: compiler/driver abstract surfaces,
  `GPUTarget`, public-backend adapters, Legacy automatic compatibility,
  Triton-version-specific integration, exact wheel payload/layout, and hardware
  JIT.

No test imports the Triton 3.3 worktree, consumes its wheel, LLVM build, cache,
JUnit, or PASS result, or imports the excluded T10.2 Conformance Kit. Temporary
plugin distributions and compiled witness DSOs are created under pytest's
external temporary root.

## Independent oracles

- Manifest structural acceptance comes from the shipped Draft 2020-12 JSON
  Schema; the public parser must accept and reject the same instances.
- Lifecycle behavior comes from the public `BackendPluginBase` Protocol 1.0
  contract.
- Compatibility aggregation requires every independent failure to be reported
  before entry-point import, initialize, dynamic load, or compile.
- Native tests independently recompute wheel-style `RECORD` hashes and inspect
  ELF symbols/dynamic metadata. Undefined native rules remain `SPEC GAP` until
  the approved Protocol/ABI disposition replaces those witness expectations.
- Triton 3.6 facts are `3.6.0`, vendored commit
  `6cc4505027d7b39fe18a44a7f89085b8babb7400`, and LLVM commit
  `a992f29451b9e140424f35ac5e20177db4afbdc0` (LLVM/MLIR 22.0.0git).

## Canonical common invocation

```bash
cd /home/dingbl/race_workspace/triton-anchor-t63-v3.6
export PYTHONPATH="$PWD/python:$PWD/triton/python"
export PYTHONDONTWRITEBYTECODE=1
export T63_V36_PYTHON=/home/dingbl/race_workspace/t63-v36-acceptance-venv/bin/python
unset T63_CORE_WHEEL TRITON_PASS_PLUGIN_PATH

"$T63_V36_PYTHON" -m pytest -q \
  python/triton_anchor/tests \
  tests/acceptance_t63/test_registry_acceptance.py \
  tests/acceptance_t63/test_schema_oracle_acceptance.py \
  tests/acceptance_t63/test_native_abi_acceptance.py \
  tests/acceptance_t63/test_packaging_entrypoint_e2e.py \
  -ra \
  --junitxml=/home/dingbl/race_workspace/reports/t6.3-v3.6-evidence/phase1-common-baseline.xml \
  -o cache_dir=/home/dingbl/race_workspace/reports/t6.3-v3.6-evidence/pytest-cache-phase1-common-baseline
```

The reproducible result, node classifications, and test hashes are recorded in
`baseline_manifest.json`. Logs, JUnit, caches, wheels, coverage, and native
build artifacts remain outside this worktree.
