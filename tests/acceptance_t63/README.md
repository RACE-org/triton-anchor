# Triton 3.6 T6.3 common acceptance baseline

This directory contains the independently frozen Triton 3.6 baseline and the
current acceptance oracles for the cross-version portion of Backend Plugin
Protocol 1.0. `baseline_manifest.json` and the phase-one baseline JUnit remain
immutable evidence. Current test bodies may change only when an approved
contract changes their oracle; baseline identities must not be removed,
skipped, xfailed, or weakened to match implementation behavior.

## Scope classification

The Triton 3.3 suite was used only as a test-design source and was split before
porting:

- Class A, included here: Manifest Schema/parser equivalence, Registry
  compatibility diagnostics, lifecycle hooks, real entry-point metadata,
  pre-import sentinels, native static integrity and explicit-rejection
  witnesses, selection, concurrency, and reset. The current suite also includes
  the narrow cross-version final gate that prevents an unsupported Manifest
  record from entering Triton's public backend mapping.
- Class B, deliberately excluded: compiler/driver abstract surfaces,
  `GPUTarget`, version-specific public-backend adapter behavior, Legacy
  automatic compatibility, Triton-version-specific integration, exact wheel
  payload/layout, and hardware JIT.

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
- The approved native disposition makes Protocol/Schema 1.0 operationally
  `python_only`. `native_in_process` and `subprocess` are rejected before
  import, initialization, dynamic load, or compilation. Static `RECORD`, ELF,
  SONAME, export, and C++ ABI inspection remains evidence-only and never grants
  load authorization.
- The three original native `SPEC GAP` testcase identities are preserved. Their
  bodies are now formal explicit-rejection oracles, including a loadable DSO
  constructor sentinel proving that Registry/helper/host rejection performs no
  `dlopen`.
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
  --junitxml=/home/dingbl/race_workspace/reports/t6.3-v3.6-evidence/phase1-common-current.xml \
  -o cache_dir=/home/dingbl/race_workspace/reports/t6.3-v3.6-evidence/pytest-cache-phase1-common-current
```

The original 157-node result, classifications, hashes, and baseline evidence
paths remain recorded in `baseline_manifest.json`; that file and its referenced
JUnit are not regenerated. Current logs, JUnit, caches, wheels, coverage, and
native build artifacts remain outside this worktree.
