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
FLAGGEMS_TEST_OP="abs"
FLAGGEMS_TEST_COMMAND=""
```

Set GITEE_TOKEN to publish local CI results back to Gitee. The runner pulls source code from GITEE_REPO_URL, pushes logs to GITEE_RESULTS_REPO_URL on the local-ci-results branch, and adds a short commit comment on the source mirror commit with the result link. The old commit status API route is not used because Gitee rejects that endpoint with HTTP 405.

Recommended repository split:

```text
GITEE_REPO_URL=https://gitee.com/likehupochuan/triton-anchor.git
GITEE_RESULTS_REPO_URL=https://gitee.com/likehupochuan/triton-anchor-local-ci-results.git
GITEE_RESULTS_WEB_URL=https://gitee.com/likehupochuan/triton-anchor-local-ci-results
```

By default the poller only watches GITEE_BRANCH, currently jiwang-delivery-ci. To let GitHub PRs from this same repository trigger local CI, the PR source branch must also exist in the Gitee mirror and the local poller must watch it. Either list branches explicitly with GITEE_BRANCHES, or set GITEE_POLL_ALL_BRANCHES=1 and optionally narrow it with GITEE_BRANCH_INCLUDE_REGEX. The results branch is always skipped so publishing logs does not trigger another local CI run.

The runner activates /opt/venv/bin/activate before running uv build or uv pip install. Set PYTHON_VENV_ACTIVATE to another path, or empty, if a different container layout is used.

Set RUN_FLAGGEMS_TESTS=true to run the local FlagGems check. The default command runs only the abs operator through the current Sophgo script. Internally this expands to: python3 -m pytest -s tests/test_unary_pointwise_ops.py -m abs

Change FLAGGEMS_TEST_OP for another unary marker if this default pytest entry still applies. For another file or script, set FLAGGEMS_TEST_COMMAND directly, for example: python3 testop/new_flaggems_smoke.py --op add.

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

Published Gitee results are stored in the result repository on the local-ci-results branch under runs/<branch>/<commit>/<run-id>/. Commit comments contain only a short summary and a link to the result directory. The Gitee result directory intentionally keeps only selected files: delivery-summary.txt, frontend-install.log, backend-smoke-jit.log, and flaggems.log. Full local logs remain under /workspace/local-ci-artifacts.

## GitHub Status Bridge

GitHub does not need to run the hardware tests. The `Local CI Bridge` workflow waits for the Gitee `local-ci-results` branch and writes the result back to the GitHub commit status. It runs on pushes to jiwang-delivery-ci, manual workflow_dispatch, and pull_request events.

For a pull request, the bridge waits for the PR head branch and PR head SHA. This supports PRs whose source branch is in the same GitHub repository and is mirrored to Gitee. Fork PRs are intentionally rejected because the local server cannot fetch fork code through the current GitHub -> Gitee mirror path.

Configure these GitHub repository variables if the defaults change:

```text
GITEE_RESULTS_OWNER=likehupochuan
GITEE_RESULTS_REPO=triton-anchor-local-ci-results
GITEE_RESULTS_BRANCH=local-ci-results
GITEE_RESULTS_WEB_URL=https://gitee.com/likehupochuan/triton-anchor-local-ci-results
LOCAL_CI_CONTEXT=local-ci/sophgo-cmodel
LOCAL_CI_BRIDGE_TIMEOUT_SECONDS=10800
LOCAL_CI_BRIDGE_POLL_INTERVAL_SECONDS=60
```

If the Gitee result repository or result branch is private, add a GitHub repository secret named `GITEE_TOKEN` with read access to the result repository. The local server token also needs write access to the result repository, and read/comment access to the source mirror if those operations are private. The workflow uses GitHub's built-in `GITHUB_TOKEN` with `statuses: write` permission to publish the GitHub status.

## Order Notes

It is fine for the backend to be prepared before the frontend is pulled and rebuilt. The per-commit operation is frontend checkout/build/install, then source the already-prepared backend environment and run discovery/smoke/JIT. If a future frontend change breaks backend ABI/API compatibility, smoke/JIT should catch it; then that backend may need to be rebuilt separately.
