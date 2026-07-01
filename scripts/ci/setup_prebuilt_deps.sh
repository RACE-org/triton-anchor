#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${WORKSPACE}/llvm-release}"
PPL_ROOT="${PPL_ROOT:-${WORKSPACE}/ppl-release}"
REQUIRE_PREBUILT_DEPS="${REQUIRE_PREBUILT_DEPS:-0}"

download_file() {
  local url="$1"
  local output="$2"
  echo "Downloading ${url}"
  curl -L --fail --retry 3 --output "${output}" "${url}"
}

verify_sha256() {
  local file="$1"
  local expected="$2"
  if [[ -z "${expected}" ]]; then
    echo "No sha256 configured for ${file}; skipping checksum verification."
    return
  fi
  echo "${expected}  ${file}" | sha256sum -c -
}

extract_archive() {
  local archive="$1"
  local destination="$2"
  local strip_components="${3:-1}"
  mkdir -p "${destination}"

  case "${archive}" in
    *.tar.gz|*.tgz)
      tar -xzf "${archive}" -C "${destination}" --strip-components="${strip_components}"
      ;;
    *.tar.xz|*.txz)
      tar -xJf "${archive}" -C "${destination}" --strip-components="${strip_components}"
      ;;
    *.zip)
      unzip -q "${archive}" -d "${destination}"
      ;;
    *)
      echo "Unsupported archive format: ${archive}" >&2
      return 1
      ;;
  esac
}

setup_package() {
  local name="$1"
  local destination="$2"
  local archive_var="$3"
  local url_var="$4"
  local sha_var="$5"
  local strip_var="$6"

  local archive="${!archive_var:-}"
  local url="${!url_var:-}"
  local sha="${!sha_var:-}"
  local strip_components="${!strip_var:-1}"

  if [[ -d "${destination}" ]] && [[ -n "$(find "${destination}" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
    echo "${name} already exists at ${destination}"
    return 0
  fi

  if [[ -n "${archive}" ]]; then
    echo "Using local ${name} archive: ${archive}"
    verify_sha256 "${archive}" "${sha}"
    extract_archive "${archive}" "${destination}" "${strip_components}"
    return 0
  fi

  if [[ -n "${url}" ]]; then
    local suffix
    local tmp_archive
    suffix="${url%%\?*}"
    suffix="${suffix##*/}"
    tmp_archive="$(mktemp "/tmp/${name}.XXXXXX.${suffix}")"
    download_file "${url}" "${tmp_archive}"
    verify_sha256 "${tmp_archive}" "${sha}"
    extract_archive "${tmp_archive}" "${destination}" "${strip_components}"
    return 0
  fi

  if [[ "${REQUIRE_PREBUILT_DEPS}" == "1" ]]; then
    echo "${name} is required but neither ${archive_var} nor ${url_var} was provided." >&2
    return 1
  fi

  echo "::warning::${name} was not configured; continuing without it."
}

setup_package "llvm" "${LLVM_BUILD_DIR}" PREBUILT_LLVM_ARCHIVE PREBUILT_LLVM_URL PREBUILT_LLVM_SHA256 PREBUILT_LLVM_STRIP_COMPONENTS
setup_package "ppl" "${PPL_ROOT}" PREBUILT_PPL_ARCHIVE PREBUILT_PPL_URL PREBUILT_PPL_SHA256 PREBUILT_PPL_STRIP_COMPONENTS

echo "LLVM_BUILD_DIR=${LLVM_BUILD_DIR}"
echo "PPL_ROOT=${PPL_ROOT}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "LLVM_BUILD_DIR=${LLVM_BUILD_DIR}"
    echo "PPL_ROOT=${PPL_ROOT}"
  } >> "${GITHUB_ENV}"
fi

echo "Prebuilt dependency setup finished for ${ROOT_DIR}"