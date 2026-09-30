"""Shared task identity and the small Local CI result format."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

TASK_SCHEMA = "triton-anchor-local-ci-task"
RESULT_SCHEMA = "triton-anchor-local-ci"
FULL_FLAGGEMS_ROOT = "runs/ci_full_flaggems"
PREINSTALLED_SUBMODULES = frozenset({"FlagGems"})
SHA = re.compile(r"[0-9a-f]{40}\Z")
ID = re.compile(r"[0-9a-f]{64}\Z")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}\Z")
TRIGGER_ID = re.compile(r"[1-9][0-9]*:[1-9][0-9]*\Z")
LLVM_METADATA = re.compile(r"triton/cmake/llvm-(?:hash|info)(?:\.(?:txt|json))?\Z")
TRITON_VERSION_PATH = "triton/python/triton/__init__.py"
TRITON_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[a-zA-Z0-9.+-]*)\Z")
IDENTITY_FIELDS = (
    "repository",
    "event_kind",
    "pr_number",
    "target_branch",
    "tested_sha",
    "base_sha",
    "head_sha",
    "worker_revision_sha",
    "metadata_digest",
    "full",
)
RESULT_STATUSES = {"pass", "fail", "infra_error", "cancelled"}
CHECK_STATUSES = RESULT_STATUSES | {"not_selected", "not_applicable", "skipped", "warning", "limited"}


class ContractError(ValueError):
    pass


def llvm_hash_from_files(paths, read_file):
    """Read the same pinned LLVM metadata on GitHub and from the Gitee checkout."""
    revisions = set()
    for path in sorted(paths):
        if not LLVM_METADATA.fullmatch(path):
            continue
        value = read_file(path).decode().strip()
        if value.startswith("{"):
            value = json.loads(value).get("llvm_hash")
        if not isinstance(value, str) or not SHA.fullmatch(value):
            raise ContractError(f"Expected a full LLVM commit in {path}")
        revisions.add(value)
    if not revisions:
        raise ContractError("No llvm-hash or llvm-info metadata found in triton/cmake")
    if len(revisions) != 1:
        raise ContractError("Conflicting LLVM commits in triton/cmake metadata")
    return revisions.pop()


def triton_version_from_source(source: bytes) -> str:
    """Read the upstream version declaration without executing candidate code."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:
        raise ContractError("Invalid Triton version source") from exc
    versions = []
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in targets):
            if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                raise ContractError("Triton __version__ must be a literal string")
            versions.append(node.value.value)
    if len(versions) != 1 or not TRITON_VERSION.fullmatch(versions[0]):
        raise ContractError("Expected one explicit Triton major.minor.patch version")
    return versions[0]


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def metadata_digest(task: dict) -> str:
    return digest(
        {
            key: sorted(task[key]) if key == "labels" else task[key]
            for key in ("title", "description", "labels", "state", "draft")
        }
    )


def task_id(task: dict) -> str:
    identity = {key: task[key] for key in IDENTITY_FIELDS}
    if task.get("trigger_id"):
        identity["trigger_id"] = task["trigger_id"]
    if task.get("control_policy") == "worker":
        identity.pop("worker_revision_sha")
        identity["control_policy"] = "worker"
    return digest(identity)


def current_key(task: dict) -> str:
    subject = (
        f"pr:{task['pr_number']}"
        if task["pr_number"]
        else f"branch:{task['target_branch']}"
    )
    return hashlib.sha256(f"{task['repository']}:{subject}".encode()).hexdigest()


def result_task_prefix(task: dict, *, legacy: bool = False) -> str:
    """Return the shared readable directory for local runs and Gitee results."""
    branch = task.get("target_branch")
    if not isinstance(branch, str) or not branch or any(
        ord(character) < 32 or ord(character) == 127 for character in branch
    ):
        raise ContractError("Invalid result target branch")
    encoded = quote(branch, safe="")
    if len(encoded) > 180:
        encoded = "sha256-" + hashlib.sha256(branch.encode()).hexdigest()
    branch_directory = "branch-" + encoded
    pr_number = task.get("pr_number")
    if type(pr_number) is not int or pr_number < 0:
        raise ContractError("Invalid result PR number")
    identity = task["task_id"] if legacy else task["head_sha"]
    if not (ID if legacy else SHA).fullmatch(identity):
        raise ContractError("Invalid result task directory identity")
    if pr_number:
        return f"runs/pr/{branch_directory}/pr-{pr_number}/{identity}"
    return f"runs/push/{branch_directory}/{identity}"


def result_task_prefixes(task: dict) -> tuple[str, ...]:
    """Current SHA layout and both historical task-id layouts, read-only fallback."""
    return (
        result_task_prefix(task),
        result_task_prefix(task, legacy=True),
        f"runs/{task['task_id']}",
    )


def full_flaggems_result_path(task: dict, run_id: str) -> str:
    """Return the independent business-result path for one completed full run."""
    tested_sha = task.get("tested_sha")
    if not isinstance(tested_sha, str) or not SHA.fullmatch(tested_sha):
        raise ContractError("Invalid full FlagGems result SHA")
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ContractError("Invalid full FlagGems result run id")
    return f"{FULL_FLAGGEMS_ROOT}/{tested_sha}/{run_id}/flaggems-summary.json"


def is_legacy_task(task: dict) -> bool:
    return isinstance(task, dict) and str(task.get("schema", "")).startswith(
        TASK_SCHEMA + "/"
    )


def validate_task(
    task: dict,
    repositories=("RACE-org/triton-anchor",),
) -> dict:
    if not isinstance(task, dict) or task.get("schema") != TASK_SCHEMA:
        raise ContractError("Unsupported task format")
    required = {
        *IDENTITY_FIELDS,
        "task_id",
        "task_ref",
        "base_task_ref",
        "head_task_ref",
        "title",
        "description",
        "labels",
        "state",
        "draft",
        "captured_at",
        "llvm_hash",
    }
    if required - task.keys():
        raise ContractError("Task is missing required identity fields")
    if "trigger_id" in task:
        trigger = task["trigger_id"]
        if not isinstance(trigger, str) or len(trigger) > 160 or (
            trigger and not TRIGGER_ID.fullmatch(trigger)
        ):
            raise ContractError("Invalid trigger_id")
    if task.get("control_policy") not in {None, "worker"}:
        raise ContractError("Invalid control policy")
    if "_use_installed_control" in task:
        raise ContractError("Runtime-only control override is not valid task data")
    if task["repository"] not in repositories:
        raise ContractError("Task repository is not configured for this worker")
    if type(task["pr_number"]) is not int or task["pr_number"] < 0:
        raise ContractError("Invalid PR number")
    if task["event_kind"] not in {"pull_request", "push", "manual"} or (
        (task["event_kind"] == "pull_request") != bool(task["pr_number"])
    ):
        raise ContractError("Task event and PR identity disagree")
    for key in (
        "tested_sha",
        "base_sha",
        "head_sha",
        "worker_revision_sha",
        "llvm_hash",
    ):
        if not isinstance(task[key], str) or not SHA.fullmatch(task[key]):
            raise ContractError(f"Invalid {key}")
    if "variants" in task:
        variants = task["variants"]
        if not isinstance(variants, dict) or set(variants) != {"base", "candidate"}:
            raise ContractError("Task needs base and candidate source variants")
        for variant, source_sha in (("base", task["base_sha"]), ("candidate", task["tested_sha"])):
            source = variants[variant]
            if (
                not isinstance(source, dict)
                or set(source) != {"source_sha", "llvm_hash", "triton_version"}
                or source.get("source_sha") != source_sha
                or not isinstance(source.get("llvm_hash"), str)
                or not SHA.fullmatch(source["llvm_hash"])
                or not isinstance(source.get("triton_version"), str)
                or not TRITON_VERSION.fullmatch(source["triton_version"])
            ):
                raise ContractError(f"Invalid {variant} source variant")
        if task["llvm_hash"] != variants["candidate"]["llvm_hash"]:
            raise ContractError("Task LLVM alias differs from candidate source variant")
    if type(task["draft"]) is not bool or type(task["full"]) is not bool:
        raise ContractError("Task draft/full must be booleans")
    if not isinstance(task["labels"], list) or any(
        not isinstance(label, str) for label in task["labels"]
    ):
        raise ContractError("Task labels must be strings")
    for key in ("target_branch", "title", "description", "state", "captured_at"):
        if not isinstance(task[key], str):
            raise ContractError(f"Invalid task {key}")
    if task["metadata_digest"] != metadata_digest(task) or task["task_id"] != task_id(
        task
    ):
        raise ContractError("Task or metadata identity mismatch")
    prefix = (
        f"ci/pr-{task['pr_number']}/{task['task_id']}"
        if task["pr_number"]
        else f"ci/branch/{task['task_id']}"
    )
    for key, suffix in (
        ("task_ref", "tested"),
        ("base_task_ref", "base"),
        ("head_task_ref", "head"),
    ):
        if task[key] != f"{prefix}/{suffix}":
            raise ContractError("Source refs must belong to the frozen task")
    return task


def within(root: Path, relative: str, *, must_exist: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ContractError("Expected a relative task path")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ContractError("Path escapes task workspace")
    if must_exist and not path.is_file():
        raise ContractError("Task file does not exist")
    return path


def atomic_json(path: Path, value: Any, *, pretty: bool = False) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            data = (
                json.dumps(
                    value,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ).encode()
                if pretty
                else canonical(value)
            )
            stream.write(data + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate_result(result: dict, expected_task: dict | None = None) -> dict:
    if not isinstance(result, dict) or result.get("schema") != RESULT_SCHEMA:
        raise ContractError("Unsupported result format")
    task = validate_task(result.get("task"))
    if expected_task is not None and task != expected_task:
        raise ContractError("Result belongs to a different dispatched task")
    if not isinstance(result.get("run_id"), str) or not RUN_ID.fullmatch(
        result["run_id"]
    ):
        raise ContractError("Invalid result run id")
    if result.get("status") not in RESULT_STATUSES or not isinstance(
        result.get("summary"), str
    ):
        raise ContractError("Result needs a terminal status and summary")
    for name in ("checks", "reviews", "findings", "blocking_reasons", "artifacts"):
        if not isinstance(result.get(name), list):
            raise ContractError(f"Result {name} must be a list")
    for name in ("policy", "environment"):
        if not isinstance(result.get(name), dict):
            raise ContractError(f"Result {name} must be an object")
    if "limitations" in result and (
        not isinstance(result["limitations"], list)
        or any(not isinstance(item, str) for item in result["limitations"])
    ):
        raise ContractError("Result limitations must be a list of explanations")
    delivery = result.get("evidence_delivery")
    if delivery is not None and (
        not isinstance(delivery, dict)
        or delivery.get("status") not in {"complete", "incomplete"}
        or not isinstance(delivery.get("omitted", []), list)
    ):
        raise ContractError("Invalid evidence delivery status")
    for name, identity in (("checks", "tool_id"), ("reviews", "kind")):
        for row in result[name]:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get(identity), str)
                or row.get("status") not in CHECK_STATUSES
            ):
                raise ContractError(f"Invalid {identity} result")
    for row in result["artifacts"]:
        if not isinstance(row, dict):
            raise ContractError("Invalid artifact entry")
        within(Path("/artifacts"), row.get("path", ""))
    if result["status"] == "pass" and result["blocking_reasons"]:
        raise ContractError("Passing result contains blocking reasons")
    return result
