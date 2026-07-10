# Local CI Runner

This runner is for the internal server path where GitHub Actions cannot reach company dependencies or long-running backend tests.

The Docker container and backend environment are assumed to be ready already. Local CI only does the moving part for each frontend commit:

```text
poll Gitee -> enter existing Docker -> checkout frontend commit -> build/install frontend -> source backend env -> run smoke/JIT -> run FlagGems -> save logs -> push local-ci-results branch -> add commit comment
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
PYTHON_VENV_ACTIVATE="/opt/venv/bin/activate"
RUN_FLAGGEMS_TESTS="true"
FLAGGEMS_PIP_PACKAGES="scipy pytest"
FLAGGEMS_TEST_OP="add"
FLAGGEMS_TEST_COMMAND=""
```

Set GITEE_TOKEN to publish local CI results back to Gitee. The runner pushes logs to the local-ci-results branch and adds a short commit comment with the result link. The old commit status API route is not used because Gitee rejects that endpoint with HTTP 405.

The runner activates /opt/venv/bin/activate before running uv build or uv pip install. Set PYTHON_VENV_ACTIVATE to another path, or empty, if a different container layout is used.

Set RUN_FLAGGEMS_TESTS=true to run the local FlagGems check. The default command runs only the add operator through the current Sophgo script. Internally this expands to: python3 testop/batch_test_flaggems.py add by default

Change FLAGGEMS_TEST_OP for another operator if the script still supports that interface. When batch_test_flaggems.py is replaced, leave FLAGGEMS_TEST_OP alone and set FLAGGEMS_TEST_COMMAND directly, for example: python3 testop/new_flaggems_smoke.py --op add.

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

Published Gitee results are stored on the local-ci-results branch under runs/<branch>/<commit>/<run-id>/. Commit comments contain only a short summary and a link to the result directory. Full logs stay on the result branch.

## Order Notes

It is fine for the backend to be prepared before the frontend is pulled and rebuilt. The per-commit operation is frontend checkout/build/install, then source the already-prepared backend environment and run discovery/smoke/JIT. If a future frontend change breaks backend ABI/API compatibility, smoke/JIT should catch it; then that backend may need to be rebuilt separately.
