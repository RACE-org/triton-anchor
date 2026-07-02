#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PACKAGE_TOOL="${PACKAGE_TOOL:-auto}"
BACKEND_PIP_PACKAGES="${BACKEND_PIP_PACKAGES:-}"
BACKEND_TORCH_VERSION="${BACKEND_TORCH_VERSION:-}"
BACKEND_TORCH_INDEX_URL="${BACKEND_TORCH_INDEX_URL:-}"
BACKEND_TORCH_TPU_WHEEL_ARCHIVE="${BACKEND_TORCH_TPU_WHEEL_ARCHIVE:-}"
BACKEND_TORCH_TPU_WHEEL_URL="${BACKEND_TORCH_TPU_WHEEL_URL:-}"
BACKEND_TORCH_TPU_WHEEL_SHA256="${BACKEND_TORCH_TPU_WHEEL_SHA256:-}"
REQUIRE_BACKEND_TORCH_TPU_WHEEL="${REQUIRE_BACKEND_TORCH_TPU_WHEEL:-0}"

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

  if [[ -z "${PREBUILT_DOWNLOAD_TOKEN:-}" ]]; then
    return 2
  fi

  owner="${BASH_REMATCH[1]}"
  repo="${BASH_REMATCH[2]}"
  tag_encoded="${BASH_REMATCH[3]}"
  asset_encoded="${BASH_REMATCH[4]}"
  tag="$(url_decode "${tag_encoded}")"
  asset_name="$(url_decode "${asset_encoded}")"
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

use_uv() {
  [[ "${PACKAGE_TOOL}" == "uv" ]] || { [[ "${PACKAGE_TOOL}" == "auto" ]] && command -v uv >/dev/null 2>&1; }
}

pip_install() {
  if use_uv; then
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
      uv pip install "$@"
    else
      uv pip install --system "$@"
    fi
  else
    "${PYTHON_BIN}" -m pip install "$@"
  fi
}

if [[ -z "${BACKEND_PIP_PACKAGES}" && -z "${BACKEND_TORCH_VERSION}" && -z "${BACKEND_TORCH_TPU_WHEEL_ARCHIVE}" && -z "${BACKEND_TORCH_TPU_WHEEL_URL}" ]]; then
  if [[ "${REQUIRE_BACKEND_TORCH_TPU_WHEEL}" == "1" ]]; then
    echo "torch_tpu wheel is required but no BACKEND_TORCH_TPU_WHEEL_ARCHIVE or BACKEND_TORCH_TPU_WHEEL_URL was provided." >&2
    exit 1
  fi
  echo "No backend-specific Python/runtime dependencies configured."
  exit 0
fi

if [[ -n "${BACKEND_PIP_PACKAGES}" ]]; then
  # shellcheck disable=SC2086
  pip_install ${BACKEND_PIP_PACKAGES}
fi

if [[ -n "${BACKEND_TORCH_VERSION}" ]]; then
  if [[ -n "${BACKEND_TORCH_INDEX_URL}" ]]; then
    pip_install "torch==${BACKEND_TORCH_VERSION}" --index-url "${BACKEND_TORCH_INDEX_URL}"
  else
    pip_install "torch==${BACKEND_TORCH_VERSION}"
  fi
fi

torch_tpu_wheel="${BACKEND_TORCH_TPU_WHEEL_ARCHIVE}"
if [[ -z "${torch_tpu_wheel}" && -n "${BACKEND_TORCH_TPU_WHEEL_URL}" ]]; then
  wheel_name="${BACKEND_TORCH_TPU_WHEEL_URL%%\?*}"
  wheel_name="${wheel_name##*/}"
  torch_tpu_wheel="$(mktemp "/tmp/torch-tpu-wheel.XXXXXX.${wheel_name}")"
  download_file "${BACKEND_TORCH_TPU_WHEEL_URL}" "${torch_tpu_wheel}"
fi

if [[ -n "${torch_tpu_wheel}" ]]; then
  verify_sha256 "${torch_tpu_wheel}" "${BACKEND_TORCH_TPU_WHEEL_SHA256}"
  pip_install "${torch_tpu_wheel}"
elif [[ "${REQUIRE_BACKEND_TORCH_TPU_WHEEL}" == "1" ]]; then
  echo "torch_tpu wheel is required but no BACKEND_TORCH_TPU_WHEEL_ARCHIVE or BACKEND_TORCH_TPU_WHEEL_URL was provided." >&2
  exit 1
fi

echo "Backend dependency installation finished."
