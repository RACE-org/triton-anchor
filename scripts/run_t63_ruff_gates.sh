#!/usr/bin/env bash

# Shared, deterministic Ruff release gates for the T6.3 Triton 3.6 candidate.
# This file is intentionally safe to source: only main performs repository I/O.

readonly RUFF_VERSION=0.15.22
readonly T63_BASE_SHA=9ceba3e222fd2c9af9cbaf6a440beb23911c197b
readonly GATE_CONTEXT_SCHEMA=triton-anchor-t63-ruff-gate-context-v1
readonly GATE_EXIT_SCHEMA=triton-anchor-t63-ruff-gate-exits-v1

readonly -a RUFF_CRITICAL_ARGS=(
  check
  --isolated
  --no-cache
  --no-fix
  --target-version
  py310
  --select
  E9,F63,F7,F82
)

readonly -a RUFF_CHANGED_ARGS=(
  check
  --isolated
  --no-cache
  --no-fix
  --target-version
  py310
)

readonly -a RUFF_FORMAT_ARGS=(
  format
  --isolated
  --no-cache
  --target-version
  py310
  --check
)

declare -a ALL_PYTHON_FILES=()
declare -a T63_CHANGED_PYTHON_FILES=()
declare -a T63_DELETED_PYTHON_FILES=()

RUFF_CRITICAL_EXIT=0
RUFF_CHANGED_EXIT=0
RUFF_FORMAT_EXIT=0
RUFF_POLICY_EXIT=0
T63_EVIDENCE_DIR=""

record_manifest() {
  local name="$1"
  local path="$2"
  local count="$3"
  local digest

  digest="$(sha256sum -- "$path")" || return 2
  digest="${digest%% *}"
  if [[ ! "$digest" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'manifest status=FAIL reason=invalid-sha256 path=%s digest=%q\n' \
      "$path" "$digest" >&2
    return 2
  fi
  printf '%s count=%s sha256=%s path=%s\n' \
    "$name" "$count" "$digest" "$path" || return 2
}

record_argv0() {
  local stage="${1:?evidence stage is required}"
  shift
  printf '%s\0' "$@" >"$T63_EVIDENCE_DIR/$stage.argv0" || return 2
}

replay_raw() {
  local path="${1:?raw evidence path is required}"
  cat -- "$path" || return 2
}

write_gate_context() {
  local head_sha="$1"
  local candidate_sha="$2"
  local head_tree_sha="$3"
  local index_tree_sha="$4"
  local ruff_version="$5"

  printf '%s\n' \
    "{\"base_is_ancestor\":true,\"base_sha\":\"$T63_BASE_SHA\",\"cached_index_clean\":true,\"candidate_sha\":\"$candidate_sha\",\"head_sha\":\"$head_sha\",\"head_tree_sha\":\"$head_tree_sha\",\"index_tree_sha\":\"$index_tree_sha\",\"ruff_version\":\"$ruff_version\",\"schema\":\"$GATE_CONTEXT_SCHEMA\",\"tracked_worktree_clean\":true}" \
    >"$T63_EVIDENCE_DIR/gate-context.json" || return 2
}

write_exit_summary() {
  printf '%s\n' \
    "{\"exits\":{\"changed\":$RUFF_CHANGED_EXIT,\"critical\":$RUFF_CRITICAL_EXIT,\"format\":$RUFF_FORMAT_EXIT,\"policy\":$RUFF_POLICY_EXIT},\"schema\":\"$GATE_EXIT_SCHEMA\"}" \
    >"$T63_EVIDENCE_DIR/gate-exit-summary.json" || return 2
}

write_evidence_checksums() {
  local -a evidence_files=(
    all-python-files.nul
    changed-python-files.nul
    deleted-python-files.nul
    gate-context.json
    critical.argv0
    critical.raw.log
    changed.argv0
    changed.raw.log
    format.argv0
    format.raw.log
    policy.argv0
    policy.raw.log
    ruff-policy-scanner.json
    gate-exit-summary.json
  )

  (
    cd "$T63_EVIDENCE_DIR" || exit 2
    sha256sum -- "${evidence_files[@]}" >SHA256SUMS
  ) || return 2
}

run_changed_gates() {
  local array_name="${1:?changed-file array name is required}"
  local -n changed_files_ref="$array_name"
  local changed_count="${#changed_files_ref[@]}"

  RUFF_CHANGED_EXIT=0
  RUFF_FORMAT_EXIT=0
  printf 'T63_CHANGED_PYTHON_FILES count=%s\n' "$changed_count"

  if ((changed_count == 0)); then
    if [[ -n "$T63_EVIDENCE_DIR" ]]; then
      : >"$T63_EVIDENCE_DIR/changed.argv0" || return 2
      : >"$T63_EVIDENCE_DIR/changed.raw.log" || return 2
      : >"$T63_EVIDENCE_DIR/format.argv0" || return 2
      : >"$T63_EVIDENCE_DIR/format.raw.log" || return 2
    fi
    printf 'changed_full status=PASS invoked=false not-invoked\n'
    printf 'changed_format status=PASS invoked=false not-invoked\n'
    return 0
  fi

  if [[ "$array_name" == "T63_CHANGED_PYTHON_FILES" ]]; then
    record_argv0 changed ruff "${RUFF_CHANGED_ARGS[@]}" -- \
      "${T63_CHANGED_PYTHON_FILES[@]}" || return 2
    if ruff "${RUFF_CHANGED_ARGS[@]}" -- "${T63_CHANGED_PYTHON_FILES[@]}" \
      >"$T63_EVIDENCE_DIR/changed.raw.log" 2>&1; then
      RUFF_CHANGED_EXIT=0
    else
      RUFF_CHANGED_EXIT=$?
    fi
    replay_raw "$T63_EVIDENCE_DIR/changed.raw.log" || return 2

    record_argv0 format ruff "${RUFF_FORMAT_ARGS[@]}" -- \
      "${T63_CHANGED_PYTHON_FILES[@]}" || return 2
    if ruff "${RUFF_FORMAT_ARGS[@]}" -- "${T63_CHANGED_PYTHON_FILES[@]}" \
      >"$T63_EVIDENCE_DIR/format.raw.log" 2>&1; then
      RUFF_FORMAT_EXIT=0
    else
      RUFF_FORMAT_EXIT=$?
    fi
    replay_raw "$T63_EVIDENCE_DIR/format.raw.log" || return 2
  else
    if ruff "${RUFF_CHANGED_ARGS[@]}" -- "${changed_files_ref[@]}"; then
      RUFF_CHANGED_EXIT=0
    else
      RUFF_CHANGED_EXIT=$?
    fi
    if ruff "${RUFF_FORMAT_ARGS[@]}" -- "${changed_files_ref[@]}"; then
      RUFF_FORMAT_EXIT=0
    else
      RUFF_FORMAT_EXIT=$?
    fi
  fi

  if ((RUFF_CHANGED_EXIT == 0)); then
    printf 'changed_full status=PASS invoked=true exit=0\n'
  else
    printf 'changed_full status=FAIL invoked=true exit=%s\n' "$RUFF_CHANGED_EXIT"
  fi
  if ((RUFF_FORMAT_EXIT == 0)); then
    printf 'changed_format status=PASS invoked=true exit=0\n'
  else
    printf 'changed_format status=FAIL invoked=true exit=%s\n' "$RUFF_FORMAT_EXIT"
  fi
  return 0
}

main() {
  set -uo pipefail

  local repo_root
  local head_sha
  local head_tree
  local index_tree
  local t63_gate_sha
  local ruff_version_output
  local evidence_dir
  local all_manifest
  local changed_manifest
  local deleted_manifest
  local scanner_output

  repo_root="$(git rev-parse --show-toplevel)" || return 2
  cd "$repo_root" || return 2

  head_sha="$(git rev-parse HEAD)" || return 2
  t63_gate_sha="$(git rev-parse "${PRODUCTION_CANDIDATE_SHA:-HEAD}")" || return 2
  if [[ "$t63_gate_sha" != "$head_sha" ]]; then
    printf 'candidate status=FAIL reason=head-mismatch expected=%s actual=%s\n' \
      "$head_sha" "$t63_gate_sha" >&2
    return 2
  fi

  index_tree="$(git write-tree)" || return 2
  head_tree="$(git rev-parse 'HEAD^{tree}')" || return 2
  if [[ "$index_tree" != "$head_tree" ]]; then
    printf 'candidate status=FAIL reason=index-tree-mismatch\n' >&2
    return 2
  fi
  if ! git merge-base --is-ancestor "$T63_BASE_SHA" "$t63_gate_sha"; then
    printf 'candidate status=FAIL reason=base-is-not-ancestor base=%s candidate=%s\n' \
      "$T63_BASE_SHA" "$t63_gate_sha" >&2
    return 2
  fi
  if ! git diff --quiet --; then
    printf 'candidate status=FAIL reason=tracked-worktree-dirty\n' >&2
    return 2
  fi
  if ! git diff --cached --quiet --; then
    printf 'candidate status=FAIL reason=index-dirty\n' >&2
    return 2
  fi

  evidence_dir="${T63_RUFF_EVIDENCE_DIR:-$repo_root/.t63-ruff-evidence}"
  mkdir -p -- "$evidence_dir" || return 2
  T63_EVIDENCE_DIR="$evidence_dir"
  all_manifest="$evidence_dir/all-python-files.nul"
  changed_manifest="$evidence_dir/changed-python-files.nul"
  deleted_manifest="$evidence_dir/deleted-python-files.nul"
  scanner_output="$evidence_dir/ruff-policy-scanner.json"

  git ls-files -z -- '*.py' '*.pyi' | LC_ALL=C sort -z >"$all_manifest" || return 2
  git diff --diff-filter=ACM --name-only -z --no-renames \
    "$T63_BASE_SHA" "$t63_gate_sha" -- '*.py' '*.pyi' \
    | LC_ALL=C sort -z >"$changed_manifest" || return 2
  git diff --diff-filter=D --name-only -z --no-renames \
    "$T63_BASE_SHA" "$t63_gate_sha" -- '*.py' '*.pyi' \
    | LC_ALL=C sort -z >"$deleted_manifest" || return 2

  mapfile -d '' -t ALL_PYTHON_FILES <"$all_manifest" || return 2
  mapfile -d '' -t T63_CHANGED_PYTHON_FILES <"$changed_manifest" || return 2
  mapfile -d '' -t T63_DELETED_PYTHON_FILES <"$deleted_manifest" || return 2

  record_manifest ALL_PYTHON_FILES "$all_manifest" \
    "${#ALL_PYTHON_FILES[@]}" || return 2
  record_manifest T63_CHANGED_PYTHON_FILES "$changed_manifest" \
    "${#T63_CHANGED_PYTHON_FILES[@]}" || return 2
  record_manifest T63_DELETED_PYTHON_FILES "$deleted_manifest" \
    "${#T63_DELETED_PYTHON_FILES[@]}" || return 2

  if ((${#T63_DELETED_PYTHON_FILES[@]} != 0)); then printf \
    'deletion status=FAIL count=%s\n' "${#T63_DELETED_PYTHON_FILES[@]}" >&2; return 3; fi
  if ((${#ALL_PYTHON_FILES[@]} == 0)); then
    printf 'critical status=FAIL reason=empty-all-python-manifest\n' >&2
    return 3
  fi

  ruff_version_output="$(ruff --version)" || {
    printf 'ruff_version status=FAIL reason=version-command-failed\n' >&2
    return 4
  }
  printf '%s\n' "$ruff_version_output"
  if [[ "$ruff_version_output" != "ruff $RUFF_VERSION" ]]; then
    printf 'ruff_version status=FAIL expected=ruff\ %s actual=%q\n' \
      "$RUFF_VERSION" "$ruff_version_output" >&2
    return 4
  fi
  printf 'ruff_version status=PASS version=%s\n' "$RUFF_VERSION"

  write_gate_context "$head_sha" "$t63_gate_sha" "$head_tree" "$index_tree" \
    "$ruff_version_output" || return 2

  record_argv0 critical ruff "${RUFF_CRITICAL_ARGS[@]}" -- \
    "${ALL_PYTHON_FILES[@]}" || return 2
  if ruff "${RUFF_CRITICAL_ARGS[@]}" -- "${ALL_PYTHON_FILES[@]}" \
    >"$T63_EVIDENCE_DIR/critical.raw.log" 2>&1; then
    RUFF_CRITICAL_EXIT=0
  else
    RUFF_CRITICAL_EXIT=$?
  fi
  replay_raw "$T63_EVIDENCE_DIR/critical.raw.log" || return 2
  if ((RUFF_CRITICAL_EXIT == 0)); then
    printf 'critical status=PASS exit=0\n'
  else
    printf 'critical status=FAIL exit=%s\n' "$RUFF_CRITICAL_EXIT"
  fi

  run_changed_gates T63_CHANGED_PYTHON_FILES || return 2

  local -a policy_command=(
    python3
    "$repo_root/scripts/check_t63_ruff_policy.py"
    --repo-root
    "$repo_root"
    --changed-manifest
    "$changed_manifest"
    --gate-script
    scripts/run_t63_ruff_gates.sh
    --workflow
    .github/workflows/ci.yml
    --output-json
    "$scanner_output"
  )
  record_argv0 policy "${policy_command[@]}" || return 2
  if "${policy_command[@]}" >"$T63_EVIDENCE_DIR/policy.raw.log" 2>&1; then
    RUFF_POLICY_EXIT=0
  else
    RUFF_POLICY_EXIT=$?
  fi
  replay_raw "$T63_EVIDENCE_DIR/policy.raw.log" || return 2
  if ((RUFF_POLICY_EXIT == 0)); then
    printf 'policy_scanner status=PASS exit=0 path=%s\n' "$scanner_output"
  else
    printf 'policy_scanner status=FAIL exit=%s path=%s\n' \
      "$RUFF_POLICY_EXIT" "$scanner_output"
  fi

  write_exit_summary || return 2
  write_evidence_checksums || return 2

  if ((RUFF_CRITICAL_EXIT != 0 \
    || RUFF_CHANGED_EXIT != 0 \
    || RUFF_FORMAT_EXIT != 0 \
    || RUFF_POLICY_EXIT != 0)); then
    return 1
  fi
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
