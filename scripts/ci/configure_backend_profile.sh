#!/usr/bin/env bash
# Source this file before running delivery CI scripts.  It maps a selected
# backend profile to the common BACKEND_*/PREBUILT_* environment contract.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "configure_backend_profile.sh must be sourced, not executed." >&2
  exit 1
fi

ci_default() {
  local name="$1"
  local value="${2:-}"
  if [[ -z "${!name:-}" && -n "${value}" ]]; then
    export "${name}=${value}"
  fi
}

ci_override_if_set() {
  local name="$1"
  local value="${2:-}"
  if [[ -n "${value}" ]]; then
    export "${name}=${value}"
  fi
}

BACKEND_PROFILE="${BACKEND_PROFILE:-frontend-only}"
export BACKEND_PROFILE

case "${BACKEND_PROFILE}" in
  frontend-only)
    ci_default TRITON_ANCHOR_DELIVERY_BACKEND "frontend-only"
    ci_default REQUIRE_PREBUILT_LLVM "1"
    ci_default REQUIRE_PREBUILT_PPL "0"
    ;;

  sophgo-cmodel)
    ci_default TRITON_ANCHOR_DELIVERY_BACKEND "sophgo-cmodel"
    ci_default EXPECTED_TRITON_BACKEND "sophgo"
    ci_default BACKEND_REPO_URL "${SOPHGO_BACKEND_REPO_URL:-https://github.com/RACE-org/triton-sophgo-backend.git}"
    ci_default BACKEND_REF "${SOPHGO_BACKEND_REF:-}"
    ci_default BACKEND_ENVSETUP "envsetup.sh"
    ci_default BACKEND_ENVSETUP_ARGS "PIO_CMODEL"
    ci_default BACKEND_TEST_COMMAND "python3 tests/test_smoke.py && python3 tests/test_jit.py"
    ci_default BACKEND_INSTALL_ARGS "--no-build-isolation"
    ci_default BACKEND_PIP_PACKAGES "scikit-build-core pybind11 transformers"
    ci_default BACKEND_TORCH_VERSION "2.8.0"
    ci_default BACKEND_TORCH_INDEX_URL "https://download.pytorch.org/whl/cpu"
    ci_default BACKEND_TORCH_TPU_WHEEL_URL "${SOPHGO_TORCH_TPU_WHEEL_URL:-}"
    ci_default BACKEND_TORCH_TPU_WHEEL_SHA256 "${SOPHGO_TORCH_TPU_WHEEL_SHA256:-}"
    ci_default REQUIRE_BACKEND_TORCH_TPU_WHEEL "1"
    ci_default FLAGGEMS_REPO_URL "${SOPHGO_FLAGGEMS_REPO_URL:-https://github.com/sophgo-yicong/FlagGems.git}"
    ci_default FLAGGEMS_REF "${SOPHGO_FLAGGEMS_REF:-sophgo_backend}"
    ci_default FLAGGEMS_PIP_PACKAGES "${SOPHGO_FLAGGEMS_PIP_PACKAGES:-scipy pytest}"
    ci_default FLAGGEMS_TEST_FILES "tests/test_unary_pointwise_ops.py"
    ci_default FLAGGEMS_TEST_MARKERS "abs"
    ci_override_if_set PREBUILT_LLVM_URL "${SOPHGO_LLVM_URL:-}"
    ci_override_if_set PREBUILT_LLVM_SHA256 "${SOPHGO_LLVM_SHA256:-}"
    ci_override_if_set PREBUILT_LLVM_STRIP_COMPONENTS "${SOPHGO_LLVM_STRIP_COMPONENTS:-}"
    ci_override_if_set PREBUILT_PPL_URL "${SOPHGO_PPL_URL:-}"
    ci_override_if_set PREBUILT_PPL_SHA256 "${SOPHGO_PPL_SHA256:-}"
    ci_override_if_set PREBUILT_PPL_STRIP_COMPONENTS "${SOPHGO_PPL_STRIP_COMPONENTS:-}"
    ci_default REQUIRE_PREBUILT_LLVM "1"
    ci_default REQUIRE_PREBUILT_PPL "1"
    ;;

  custom)
    ci_default TRITON_ANCHOR_DELIVERY_BACKEND "custom"
    ci_default REQUIRE_PREBUILT_LLVM "1"
    ci_default REQUIRE_PREBUILT_PPL "0"
    ;;

  *)
    echo "Unsupported BACKEND_PROFILE='${BACKEND_PROFILE}'. Use frontend-only, sophgo-cmodel, or custom." >&2
    return 1
    ;;
esac

ci_override_if_set PREBUILT_LLVM_URL "${INPUT_PREBUILT_LLVM_URL:-}"
ci_override_if_set PREBUILT_LLVM_SHA256 "${INPUT_PREBUILT_LLVM_SHA256:-}"
ci_override_if_set PREBUILT_LLVM_STRIP_COMPONENTS "${INPUT_PREBUILT_LLVM_STRIP_COMPONENTS:-}"
ci_override_if_set PREBUILT_PPL_URL "${INPUT_PREBUILT_PPL_URL:-}"
ci_override_if_set PREBUILT_PPL_SHA256 "${INPUT_PREBUILT_PPL_SHA256:-}"
ci_override_if_set PREBUILT_PPL_STRIP_COMPONENTS "${INPUT_PREBUILT_PPL_STRIP_COMPONENTS:-}"

echo "Delivery backend profile: ${BACKEND_PROFILE}"
echo "Delivery backend label: ${TRITON_ANCHOR_DELIVERY_BACKEND:-}"
echo "Expected Triton backend: ${EXPECTED_TRITON_BACKEND:-<none>}"
echo "Backend repo URL: ${BACKEND_REPO_URL:-<none>}"
echo "Backend ref: ${BACKEND_REF:-<default>}"
echo "Backend envsetup: ${BACKEND_ENVSETUP:-<none>} ${BACKEND_ENVSETUP_ARGS:-}"
echo "Run FlagGems tests: ${RUN_FLAGGEMS_TESTS:-false}"
echo "FlagGems repo URL: ${FLAGGEMS_REPO_URL:-<none>}"
echo "FlagGems ref: ${FLAGGEMS_REF:-<default>}"
echo "FlagGems test files: ${FLAGGEMS_TEST_FILES:-<none>}"
echo "FlagGems test markers: ${FLAGGEMS_TEST_MARKERS:-<none>}"
echo "FlagGems test command: ${FLAGGEMS_TEST_COMMAND:-<none>}"
echo "Require LLVM: ${REQUIRE_PREBUILT_LLVM:-${REQUIRE_PREBUILT_DEPS:-0}}"
echo "Require PPL: ${REQUIRE_PREBUILT_PPL:-${REQUIRE_PREBUILT_DEPS:-0}}"
