# Shared Slim Foundation

This recipe follows `docs/build.md` and `docker/build-env.Dockerfile`, without
privileged containers. It contains Ubuntu 24.04, Python 3.12, the compiler and
build tools, CPU Torch 2.8, the supplied Torch TPU wheel, and the Codex executable.
It does not contain LLVM, PPL, FlagGems or preinstalled Triton/backend wheels.

Populate an isolated build directory with this Dockerfile, the executable named
`codex`, and `wheels/*.whl`. Download the pinned requirements plus CPU Torch
2.8.0 from `https://download.pytorch.org/whl/cpu`, then add the supplied
`torch_tpu-0.17.0+torch2.8-cp312-cp312-linux_x86_64.whl`. Resolve duplicates before
creating `wheels/SHA256SUMS`. BuildKit reads both wheel files and the executable
through temporary build mounts; installation packages are not image layers.

Keep the wheel checksums and build log alongside the foundation recipe. APT
packages follow the Ubuntu repositories at build time.
Pin the final image with the top-level `image` digest in the repository's
`scripts/local_ci/prepare/config.example.json` and load it into the server's
Rootless Docker daemon, following the [deployment procedure](../../README.md).
Every profile uses this same image. An optional
profile `local_image_tag` must resolve to the configured digest.

`configure.py --config <repository-config.example.json> --image <digest> --flaggems-source <directory>
--flaggems-commit <SHA>` sets the shared image and adds a verified FlagGems source
mount to the Triton 3.0 profile after saving a private rollback configuration.
Use it only to prepare the repository source in a development checkout that can
access the dependency directory; review and commit the result. Do not point it
at the server's runtime copy or modify the deployed control checkout. Its backup
is not a configuration override and is not used by deployment.
LLVM, backend and PPL dependencies use versioned read-only mounts.
Task preparation writes a `.pth` entry in each candidate/base venv for that
profile's mounted FlagGems `src` directory. Git trusts only the exact read-only
mount path. Cache/output paths remain task-private. `validate_flaggems.py` is
available for formal backend smoke tests; it checks source imports and a numerical
Sophgo add.

Compatible Triton branches use the same image while mounting their
exact LLVM revision (and PPL/FlagGems where needed). A shared base is not a shared
mutable Python environment: each task still builds its own frontend/backend
wheels and uses isolated venvs. Environment preparation only probes basic tools,
imports and mounts, without rebuilding Wheels before the task. The shared image's
Python ABI, C++ runtime and Torch/TPU must support every configured profile.

Keep the previous validated release until the new release and formal runtime
preflight pass. Do not run global image pruning. This recipe does not publish CI results or start worker services.
