# Local CI Runner

This runner is for the internal server path where GitHub Actions cannot reach company dependencies or long-running backend tests.

The Docker container and backend environment are assumed to be ready already. Local CI only does the moving part for each frontend commit:

```text
GitHub push/PR
  -> dispatch exact head SHA to Gitee CI relay ci/* task ref
  -> for a PR, also dispatch its base SHA to ci/base/pr-<number>
  -> poll Gitee CI relay
  -> enter existing Docker
  -> checkout/build/install frontend
  -> run triton-anchor/tests/test_smoke.py
  -> rebuild backend wheel against the newly installed frontend
  -> source backend env
  -> run backend smoke/JIT and optional FlagGems
  -> benchmark add, mm, softmax, and layernorm compile time
  -> compare PR head against the cached result for its base SHA
  -> publish selected logs to local-ci-results in the same relay repository
  -> receiver writes the result to GitHub commit status
  -> scheduled bridge reconciles missed status updates
```

The Gitee CI relay is intentionally separate from the normal source mirror. One relay repository carries both sides of the protocol without mixing their branches:

```text
ci/push/<github-branch>   exact SHA dispatched by a GitHub push
ci/pr-<number>            exact PR head SHA dispatched by a GitHub PR event
ci/base/pr-<number>       exact PR base SHA; metadata only, not a standalone task
local-ci-results          local runner results only
```

The mirror process must not target this relay repository.

## Expected Layout

```text
host runner checkout: /opt/local-ci/triton-anchor-runner
host config/state:    /opt/local-ci/config.env, /root/projects/test/local-ci-state
container workspace:  /workspace
```

Keep the runner checkout separate from the code checkout under test. The runner checkout tracks the trusted branch that owns `scripts/local_ci`; set `LOCAL_CI_SCRIPT_DIR` to that fixed script directory. For each task, the host poller copies `LOCAL_CI_SCRIPT_DIR` into a per-run snapshot under `LOCAL_CI_STATE_DIR/runner/<run-id>/`, then `run_in_container.sh` copies that snapshot into the Docker container. The container workspace `/workspace/triton-anchor` is reset to the dispatched `ci/*` task commit for each PR or push. PR branches do not need to contain local CI scripts.

Prepared inside the container:

```text
/workspace/llvm-release
/workspace/ppl-release
/workspace/triton-anchor
/workspace/triton-sophgo-backend
/workspace/FlagGems
```

The runner does not pull backend source code. Backend source and dependencies must already exist in the container, but the backend wheel is rebuilt for every tested frontend commit. For another backend, prepare it in the container first, then change `BACKEND_PATH`, `BACKEND_ENVSETUP_ARGS`, and the test commands in `scripts/local_ci/config.env`.

For a PR, the poller reads `ci/base/pr-<number>`. If
`compile-time/by-sha/<base-sha>/<backend-profile>/latest.json` already exists on
`local-ci-results`, it reuses that result. Otherwise it runs the base commit
once, publishes its compile-time result, and then runs the PR head. The base
ref is excluded from normal branch discovery, so it does not create a second
independent GitHub CI status.

## Configure

```bash
cd /opt/local-ci/triton-anchor-runner
cp scripts/local_ci/config.example.env /opt/local-ci/config.env
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

Use the same independent Gitee repository for task code and results:

```bash
GITEE_REPO_URL="https://gitee.com/likehupochuan/triton-anchor-local-ci-results.git"
GITEE_OWNER="likehupochuan"
GITEE_REPO="triton-anchor-local-ci-results"
GITEE_POLL_ALL_BRANCHES="1"
GITEE_BRANCH_INCLUDE_REGEX="^ci/(pr-[0-9]+|push/.+)$"

GITEE_RESULTS_OWNER="likehupochuan"
GITEE_RESULTS_REPO="triton-anchor-local-ci-results"
GITEE_RESULTS_REPO_URL="https://gitee.com/likehupochuan/triton-anchor-local-ci-results.git"
GITEE_RESULTS_BRANCH="local-ci-results"
GITEE_RESULTS_WEB_URL="https://gitee.com/likehupochuan/triton-anchor-local-ci-results"
```

Compile-time regression defaults:

```bash
RUN_COMPILE_BENCHMARK="true"
COMPILE_BENCHMARK_KERNELS="add,mm,softmax,layernorm"
COMPILE_BENCHMARK_REPEAT="5"
COMPILE_BENCHMARK_WARMUP="1"
COMPILE_BENCHMARK_THRESHOLD="0.20"
COMPILE_BENCHMARK_TIMEOUT="30m"
```

The threshold is symmetric: a change greater than `+20%` or less than `-20%`
is reported as a warning. A missing base result is also a warning. Correctness,
build, or benchmark execution failures still fail local CI. GitHub commit
statuses have no warning state, so a warning is published as `success` with
the description `Gitee local CI passed with compile-time warning`; the detailed
comparison is linked from the Gitee result directory.

Existing server installations must update `scripts/local_ci/config.env`; changing `config.example.env` does not overwrite a local configuration. In particular, point `GITEE_REPO_URL` at the relay repository and enable the `ci/*` filter above.

Set `GITEE_TOKEN` for a private relay and for result publishing. The token needs read/write access to the relay repository. The old Gitee commit status API route is not used because Gitee rejects that endpoint with HTTP 405.

The runner activates `/opt/venv/bin/activate` before running `uv build` or `uv pip install`. Set `PYTHON_VENV_ACTIVATE` to another path, or empty, if a different container layout is used.

Set `RUN_FLAGGEMS_TESTS=true` to run the local FlagGems check. The default command runs only the `abs` operator through the current Sophgo script. Change `FLAGGEMS_TEST_OP` for another unary marker, or set `FLAGGEMS_TEST_COMMAND` directly for another file or script.

## Run

Run one discovery pass:

```bash
LOCAL_CI_CONFIG=/opt/local-ci/config.env bash scripts/local_ci/poll_gitee_and_run.sh --once
```

Run continuously:

```bash
LOCAL_CI_CONFIG=/opt/local-ci/config.env LOCAL_CI_POLL_INTERVAL=60 bash scripts/local_ci/poll_gitee_and_run.sh
```

Host-side poller logs/state:

```text
/root/projects/test/local-ci-state
```

Container-side artifacts:

```text
/workspace/local-ci-artifacts
```

Published results are stored on `local-ci-results` under `runs/<safe-task-ref>/<commit>/<run-id>/`. The result directory keeps selected delivery logs and compile-time reports; full local logs remain under `/workspace/local-ci-artifacts`.

Compile-time artifacts in each run include `compile-benchmark.json`,
`compile-benchmark.csv`, and, for PRs, `compile-time-comparison.json` and
`compile-time-comparison.md`. A stable SHA-indexed copy is also written to:

```text
compile-time/by-sha/<commit>/<backend-profile>/latest.json
compile-time/by-sha/<commit>/<backend-profile>/latest.csv
```

This directory is parallel to the existing `runs/` directory. Existing result
repositories do not need migration.

## GitHub Workflows

`Dispatch Local CI via Gitee` is the only automatic push/PR entry point. Pushes to `main` and `jiwang-delivery-ci` create `ci/push/*`; same-repository PR events create `ci/pr-*` and update the matching `ci/base/pr-*` pointer. Fork PRs are rejected because GitHub does not expose repository Gitee credentials to them.

`Receive Local CI Result` polls the existing result protocol and writes `pending`, `success`, or `failure` to the original GitHub SHA. A receiver waits up to 20,400 seconds by default, then starts the next attempt. Four attempts preserve the coworker workflow's long-running handoff behavior without changing the local runner.

`Local CI Bridge` is retained as a manual query and scheduled reconciliation fallback. It checks configured push branch heads and open same-repository PRs using the same `ci/*` task-ref mapping, so a delayed or cancelled receiver does not permanently lose a final status.

Configure these GitHub repository variables if the defaults change:

```text
GITEE_RESULTS_OWNER=likehupochuan
GITEE_RESULTS_REPO=triton-anchor-local-ci-results
GITEE_RESULTS_REPO_URL=https://gitee.com/likehupochuan/triton-anchor-local-ci-results.git
GITEE_RESULTS_BRANCH=local-ci-results
GITEE_RESULTS_WEB_URL=https://gitee.com/likehupochuan/triton-anchor-local-ci-results
LOCAL_CI_CONTEXT=local-ci/sophgo-cmodel
LOCAL_CI_RECONCILE_SOURCE_BRANCHES="main jiwang-delivery-ci"
LOCAL_CI_BRIDGE_MAX_PRS=100
LOCAL_CI_RECEIVER_REF=main
LOCAL_CI_RECEIVER_WAIT_SECONDS=20400
LOCAL_CI_RECEIVER_MAX_ATTEMPTS=4
```

Add GitHub repository secrets `GITEE_TOKEN` and, when it differs from the owner, `GITEE_USERNAME`. The workflow uses GitHub's built-in `GITHUB_TOKEN` with `actions: write` and `statuses: write` to start the receiver and publish commit statuses.

## Order Notes

It is fine for backend source and heavy dependencies to be prepared before the frontend is pulled. The per-commit operation is frontend checkout/build/install, frontend smoke, backend rebuild, then backend discovery/smoke/JIT. If a future frontend change breaks backend ABI/API compatibility, the fixed rebuild and smoke/JIT sequence should catch it.
