#!/usr/bin/env bash
# Load a profile recipe while preserving this task's paths and resource budget.
set -eo pipefail
setup_script="${1:?environment script required}"
shift
setup_args=()
while [[ "$#" -gt 0 && "$1" != "--" ]]; do
  setup_args+=("$1")
  shift
done
[[ "$#" -gt 1 && "$1" == "--" ]] || exit 2
shift
declare -A task_values=()
llvm_names=(LLVM_BUILD_DIR LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR LLVM_BINARY_DIR LLVM_SYSPATH LLVM_DIR MLIR_DIR LLVM_CMAKE_DIR)
for name in HOME TMPDIR TRITON_CACHE_DIR TRITON_DUMP_DIR XDG_CACHE_HOME ANCHOR_DIR BACKEND_PATH PYTHON_BIN PYTHON_VENV_ACTIVATE MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL NINJAFLAGS BASELINE_JSON "${llvm_names[@]}"; do
  if [[ -v "$name" ]]; then task_values["$name"]="${!name}"; fi
done
while IFS= read -r name; do
  task_values["$name"]="${!name}"
done < <(compgen -e | sed -n '/^LOCAL_CI_/p')
source "${setup_script}" "${setup_args[@]}"
# Preserve absence too: a setup script cannot select a different LLVM through
# an additional variable that the frozen variant never provided.
for name in "${llvm_names[@]}"; do
  if [[ ! -v "task_values[$name]" ]]; then unset "$name"; fi
done
for name in "${!task_values[@]}"; do
  printf -v "$name" '%s' "${task_values[$name]}"
  export "$name"
done
if [[ "${PYTHON_BIN:-}" == /* ]]; then
  export PATH="$(dirname "$PYTHON_BIN"):$PATH"
  export VIRTUAL_ENV="$(dirname "$(dirname "$PYTHON_BIN")")"
fi
# Keep absolute SDK dependency paths, excluding product source shadowing.
clean_pythonpath=""
IFS=: read -r -a pythonpaths <<< "${PYTHONPATH:-}"
for entry in "${pythonpaths[@]}"; do
  if [[ "$entry" == /* && "$entry" != "${ANCHOR_DIR:-}" && "$entry" != "${ANCHOR_DIR:-}/"* ]]; then
    clean_pythonpath="${clean_pythonpath:+$clean_pythonpath:}$entry"
  fi
done
export PYTHONPATH="$clean_pythonpath"
exec "$@"
