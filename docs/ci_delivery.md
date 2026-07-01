# Delivery CI

This document describes the CI flow for the T3.2/T3.4 delivery work.

The scope is intentionally narrower than the full CMake build job:

- T3.2 scope in this workflow: standardize the Docker build environment and
  prebuilt dependency inputs used by delivery validation.
- T3.4 scope in this workflow: build and install `triton-anchor`, run the
  existing delivery smoke checks, and upload traceable evidence artifacts.
- T3.3 scope excluded here: enabling the always-on full C++/CMake build job.

## Files

- `docker/build-env.Dockerfile`: Ubuntu-based build environment with APT,
  Python, CMake, Ninja, `uv`, and wheel build dependencies. It creates a
  virtual environment at `/opt/venv` and puts it on `PATH`.
- `scripts/ci/setup_prebuilt_deps.sh`: prepares prebuilt LLVM/PPL packages.
- `scripts/ci/build_frontend.sh`: sources `envsetup.sh`, then builds and
  installs the `triton-anchor` wheel.
- `scripts/ci/install_backend.sh`: sources `envsetup.sh`, then optionally
  installs an out-of-tree backend.
- `scripts/ci/run_delivery_smoke.sh`: sources `envsetup.sh`, then runs install
  checks and `tests/test_smoke.py`.
- `scripts/ci/collect_delivery_evidence.py`: records environment, versions,
  backend discovery, adapter discovery, and generated logs.
- `.github/workflows/delivery-ci.yml`: GitHub Actions entry point.

## Environment Setup

The build and smoke scripts source `envsetup.sh` by default. This is important
because `envsetup.sh` exports the LLVM variables consumed by Triton and
`triton-anchor` builds:

- `LLVM_SYSPATH`
- `LLVM_INCLUDE_DIRS`
- `LLVM_LIBRARY_DIR`
- `LLVM_BINARY_DIR`
- `LLVM_BUILD_DIR`

Set `SOURCE_ENVSETUP=0` only when the runner already provides an equivalent
environment.

## Package Tool

The Docker workflow uses `uv` by setting `PACKAGE_TOOL=uv`. The scripts also
support:

- `PACKAGE_TOOL=auto`: use `uv` when available, otherwise fall back to `pip`.
- `PACKAGE_TOOL=pip`: force `pip` / `python -m build`.

`uv build` receives `--no-build-isolation` by default, matching the documented
preconfigured LLVM/PPL environment. The `pip` fallback uses `python -m build
--no-isolation`.

## Backend Installation

Backend installation is optional. If neither `BACKEND_PATH` nor
`BACKEND_REPO_URL` is provided, `scripts/ci/install_backend.sh` skips backend
installation and the workflow runs as frontend-only smoke validation.

When a backend is configured, installation defaults to normal source
installation:

```bash
BACKEND_INSTALL_MODE=standard
```

Use editable mode only for backend development workflows:

```bash
BACKEND_INSTALL_MODE=editable
```

If you expect a backend to be discovered by Triton, set:

```bash
EXPECTED_TRITON_BACKEND=sophgo
```

Then `run_delivery_smoke.sh` verifies that `sophgo` is present in
`triton.backends.backends`.

## GitHub Variables

Configure these repository variables when running the full smoke job:

- `PREBUILT_LLVM_URL`: URL for the prebuilt LLVM archive.
- `PREBUILT_LLVM_SHA256`: expected SHA256 for the LLVM archive.
- `PREBUILT_LLVM_STRIP_COMPONENTS`: optional tar strip count, default `1`.
- `PREBUILT_PPL_URL`: URL for the prebuilt PPL archive.
- `PREBUILT_PPL_SHA256`: expected SHA256 for the PPL archive.
- `PREBUILT_PPL_STRIP_COMPONENTS`: optional tar strip count, default `1`.
- `LLVM_BUILD_DIR`: optional LLVM extraction target, default
  `/workspace/llvm-release`.
- `PPL_ROOT`: optional PPL extraction target, default `/workspace/ppl-release`.

The full smoke workflow can also receive an optional backend repository from
`workflow_dispatch`:

- `backend`: label written into the evidence file, for example `sophgo`.
- `expected_backend`: Triton backend name that must be discovered, for example
  `sophgo`.
- `backend_repo_url`: out-of-tree backend repository URL.
- `backend_ref`: backend branch, tag, or commit. Pin this for reproducibility.
- `backend_install_mode`: `standard` or `editable`.

## Local Usage

Build the environment image:

```bash
docker build -f docker/build-env.Dockerfile -t triton-anchor-build-env:local .
```

Run the delivery flow with prebuilt packages:

```bash
docker run --rm \
  -v "$PWD:/workspace/triton-anchor" \
  -w /workspace/triton-anchor \
  -e PREBUILT_LLVM_URL \
  -e PREBUILT_LLVM_SHA256 \
  -e PREBUILT_PPL_URL \
  -e PREBUILT_PPL_SHA256 \
  -e REQUIRE_PREBUILT_DEPS=1 \
  -e PACKAGE_TOOL=uv \
  triton-anchor-build-env:local \
  bash -lc '
    bash scripts/ci/setup_prebuilt_deps.sh &&
    bash scripts/ci/build_frontend.sh &&
    bash scripts/ci/install_backend.sh &&
    bash scripts/ci/run_delivery_smoke.sh &&
    python3 scripts/ci/collect_delivery_evidence.py \
      --output delivery-artifacts/delivery-evidence.json
  '
```

## Evidence

The workflow uploads `delivery-artifacts/`, including:

- package import log;
- Triton backend discovery log;
- optional expected backend validation log;
- smoke test log;
- `delivery-evidence.json`.

The evidence records the frontend commit, optional backend ref, configured
prebuilt dependency paths, discovered Triton backends, and discovered
`triton-anchor` adapters.