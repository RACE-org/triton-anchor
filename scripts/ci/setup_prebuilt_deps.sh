#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${WORKSPACE}/llvm-release}"
PPL_ROOT="${PPL_ROOT:-${WORKSPACE}/ppl-release}"
REQUIRE_PREBUILT_DEPS="${REQUIRE_PREBUILT_DEPS:-0}"
REQUIRE_PREBUILT_LLVM="${REQUIRE_PREBUILT_LLVM:-${REQUIRE_PREBUILT_DEPS}}"
REQUIRE_PREBUILT_PPL="${REQUIRE_PREBUILT_PPL:-${REQUIRE_PREBUILT_DEPS}}"

url_decode() {
  python3 -c 'import sys, urllib.parse; print(urllib.parse.unquote(sys.argv[1]))' "$1"
}

download_github_release_asset() {
  local url="$1"
  local output="$2"
  local clean_url owner repo tag_encoded asset_encoded tag asset_name release_json asset_id asset_api

  clean_url="${url%%\?*}"
  if [[ ! "${clean_url}" =~ ^https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/(.+)$ ]]; then
    return 2
  fi

  owner="${BASH_REMATCH[1]}"
  repo="${BASH_REMATCH[2]}"
  tag_encoded="${BASH_REMATCH[3]}"
  asset_encoded="${BASH_REMATCH[4]}"
  tag="$(url_decode "${tag_encoded}")"
  asset_name="$(url_decode "${asset_encoded}")"

  if [[ -z "${PREBUILT_DOWNLOAD_TOKEN:-}" ]]; then
    return 2
  fi

  release_json="$(mktemp /tmp/github-release.XXXXXX.json)"
  echo "Using GitHub Release Asset API for ${owner}/${repo}@${tag}/${asset_name}"

  curl -fsSL \
    -H "Authorization: Bearer ${PREBUILT_DOWNLOAD_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "https://api.github.com/repos/${owner}/${repo}/releases/tags/${tag}" \
    --output "${release_json}"

  asset_id="$(python3 - "${release_json}" "${asset_name}" <<'PY'
import json
import sys

release_path, expected_name = sys.argv[1], sys.argv[2]
with open(release_path, encoding="utf-8") as f:
    release = json.load(f)

for asset in release.get("assets", []):
    if asset.get("name") == expected_name:
        print(asset["id"])
        break
else:
    names = ", ".join(asset.get("name", "<unnamed>") for asset in release.get("assets", []))
    print(f"Asset {expected_name!r} not found. Available assets: {names}", file=sys.stderr)
    sys.exit(1)
PY
)"

  asset_api="https://api.github.com/repos/${owner}/${repo}/releases/assets/${asset_id}"
  curl -fL --retry 3 \
    -H "Authorization: Bearer ${PREBUILT_DOWNLOAD_TOKEN}" \
    -H "Accept: application/octet-stream" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    --output "${output}" \
    "${asset_api}"
}

download_file() {
  local url="$1"
  local output="$2"
  local curl_args=(-L --fail --retry 3 --output "${output}")

  if [[ -n "${PREBUILT_DOWNLOAD_TOKEN:-}" ]]; then
    echo "Using authenticated download token"
    if download_github_release_asset "${url}" "${output}"; then
      return 0
    fi
    curl_args+=(
      -H "Authorization: Bearer ${PREBUILT_DOWNLOAD_TOKEN}"
      -H "Accept: application/octet-stream"
    )
  else
    echo "No authenticated download token configured"
  fi

  echo "Downloading ${url}"
  curl "${curl_args[@]}" "${url}"
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
  local require_var="${7:-}"

  local archive="${!archive_var:-}"
  local url="${!url_var:-}"
  local sha="${!sha_var:-}"
  local strip_components="${!strip_var:-1}"
  local required="${REQUIRE_PREBUILT_DEPS}"
  if [[ -n "${require_var}" ]]; then
    required="${!require_var:-${REQUIRE_PREBUILT_DEPS}}"
  fi

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

  if [[ "${required}" == "1" ]]; then
    echo "${name} is required but neither ${archive_var} nor ${url_var} was provided." >&2
    return 1
  fi

  echo "::warning::${name} was not configured; continuing without it."
}

setup_package "llvm" "${LLVM_BUILD_DIR}" PREBUILT_LLVM_ARCHIVE PREBUILT_LLVM_URL PREBUILT_LLVM_SHA256 PREBUILT_LLVM_STRIP_COMPONENTS REQUIRE_PREBUILT_LLVM
setup_package "ppl" "${PPL_ROOT}" PREBUILT_PPL_ARCHIVE PREBUILT_PPL_URL PREBUILT_PPL_SHA256 PREBUILT_PPL_STRIP_COMPONENTS REQUIRE_PREBUILT_PPL

echo "LLVM_BUILD_DIR=${LLVM_BUILD_DIR}"
echo "PPL_ROOT=${PPL_ROOT}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "LLVM_BUILD_DIR=${LLVM_BUILD_DIR}"
    echo "PPL_ROOT=${PPL_ROOT}"
  } >> "${GITHUB_ENV}"
fi

echo "Prebuilt dependency setup finished for ${ROOT_DIR}"