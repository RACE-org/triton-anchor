# Local CI Runner

This runner is for the internal server path where GitHub Actions cannot reach company dependencies or long-running backend tests.

The Docker container and backend environment are assumed to be ready already. Local CI only does the moving part for each frontend commit:

```text
poll Gitee -> enter existing Docker -> checkout frontend commit -> build/install frontend -> source backend env -> run smoke/JIT -> run FlagGems -> save logs -> optional Gitee status
```

## Expected Layout

```text
host:      /root/projects/test/workspace
container: /workspace
```

Prepared inside the container:

```text
/workspace/llvm-release
/workspace/ppl-release
/workspace/triton-anchor
/workspace/triton-sophgo-backend
/workspace/FlagGems
```

The runner does not pull or install the backend. For another backend, prepare it in the container first, then change `BACKEND_PATH`, `BACKEND_ENVSETUP_ARGS`, and the test commands in `scripts/local_ci/config.env`.

## Configure

```bash
cd /root/projects/test/workspace/triton-anchor
cp scripts/local_ci/config.example.env scripts/local_ci/config.env
```

Important defaults for Sophgo CModel:

```bash
BACKEND_PROFILE="sophgo-cmodel"
EXPECTED_TRITON_BACKEND="sophgo"
BACKEND_PATH="/workspace/triton-sophgo-backend"
BACKEND_ENVSETUP_ARGS="PIO_CMODEL"
BACKEND_TEST_COMMAND="python3 tests/test_smoke.py && python3 tests/test_jit.py"
RUN_FLAGGEMS_TESTS="false"
FLAGGEMS_PIP_PACKAGES="scipy pytest"
FLAGGEMS_TEST_COMMAND="python3 testop/batch_test_flaggems.py"
```

Set GITEE_TOKEN only if commit status should be posted back to Gitee.

Set RUN_FLAGGEMS_TESTS=true for full FlagGems validation.
Change FLAGGEMS_TEST_COMMAND if a backend uses a different FlagGems test script.

## Run Once

```bash
bash scripts/local_ci/poll_gitee_and_run.sh --once
```

## Run As A Poller

```bash
LOCAL_CI_POLL_INTERVAL=60 bash scripts/local_ci/poll_gitee_and_run.sh
```

Host-side poller logs/state:

```text
/root/projects/test/local-ci-state
```

Container-side artifacts:

```text
/workspace/local-ci-artifacts
```

## Order Notes

It is fine for the backend to be prepared before the frontend is pulled and rebuilt. The per-commit operation is frontend checkout/build/install, then source the already-prepared backend environment and run discovery/smoke/JIT. If a future frontend change breaks backend ABI/API compatibility, smoke/JIT should catch it; then that backend may need to be rebuilt separately.
