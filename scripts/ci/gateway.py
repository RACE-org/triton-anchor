#!/usr/bin/env python3
"""Trusted GitHub/Gitee control plane. Candidate text is data, never commands.

The pure contract functions and file/Git transports are also used by the offline
integration suite. Production network operations are restricted to our fork.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_ci"))
from agent_ci.progress import ReceiverProgress
from agent_ci.protocol import (
    PREINSTALLED_SUBMODULES,
    TASK_SCHEMA,
    RESULT_SCHEMA as RESULT_SCHEMA,
    ID,
    SHA,
    canonical,
    current_key,
    digest,
    metadata_digest,
    task_id as compute_task_id,
    llvm_hash_from_files,
    TRITON_VERSION_PATH,
    triton_version_from_source,
    is_legacy_task,
    result_task_prefixes,
    validate_task,
    validate_result,
    within,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dashboard_history import FULL_FLAGGEMS_DEMO_SHA, attach_full_flaggems, history_rows

CONTROL_BRANCH = "local-ci-control"
RESULTS_BRANCH = "local-ci-results"
REPOSITORY = os.getenv("GITHUB_REPOSITORY", "RACE-org/triton-anchor")
REPOSITORIES = {"RACE-org/triton-anchor"}
MARKER = "<!-- triton-anchor-local-ci -->"
RECEIVER_WAIT_SECONDS = 5 * 3600 + 40 * 60
RECEIVER_POLL_SECONDS = 60
RECEIVER_MAX_ROUNDS = 3

CHECK_NAMES = {
    # These are the visible GitHub status contexts. The internal keys remain
    # stable so task protocols and result records do not change.
    "basic": "Basic CI",
    "api": "API Compatibility",
    "security": "Security Gate",
}
SUMMARY_CONTEXT = "Summary"
SUMMARY_ALIASES = ("Local CI Summary", "local-ci/summary")
HEAD_CHECK_PREFIX = "triton-anchor-local-ci-head:"
# Approval is conditional: only external-fork PRs enter the protected
# environment.  Keeping it as a repository-wide required context would leave
# trusted/internal PRs waiting forever when no approval is applicable.
ALL_CHECK_NAMES = {
    **CHECK_NAMES,
    "approve": "Local CI Approve",
    "dispatch": "Local CI Dispatch",
}
REQUIRED_CONTEXTS = (*CHECK_NAMES.values(), ALL_CHECK_NAMES["dispatch"], SUMMARY_CONTEXT)
LEGACY_CHECK_NAMES = {"preflight": "local-ci/preflight"}
READ_CHECK_NAMES = {**ALL_CHECK_NAMES, **LEGACY_CHECK_NAMES}
CHECK_ALIASES = {
    "basic": ("github/basic", "local-ci/basic"),
    "api": ("github/api", "local-ci/api"),
    "security": ("github/security", "local-ci/security"),
    "approve": ("local-ci/approve",),
    "dispatch": ("local-ci/dispatch",),
}
# GitHub has no Check Run delete endpoint. These
# names belong to superseded gateway revisions; unfinished rows may be closed
# when the current task starts. Completed aliases remain readable only with
# matching task identity; new checks always use the current names.
RETIRED_CHECK_NAMES = frozenset({
    *(name for names in CHECK_ALIASES.values() for name in names),
    "local-ci/sophgo-cmodel",
    "local-ci/sophgo-cmodel/routing",
    "local-ci/soghgo-cmodel",
    "local-ci/sophgp-cmodel/routing",
})
RETIRED_STATUS_CONTEXTS = RETIRED_CHECK_NAMES | set(SUMMARY_ALIASES)
CHECK_CONCLUSIONS = {
    "success",
    "failure",
    "neutral",
    "cancelled",
    "skipped",
    "timed_out",
    "action_required",
}


def status_identity(row: dict) -> dict[str, str]:
    return {key: values[0] for key, values in
            parse_qs(urlparse(row.get("target_url") or "").fragment).items()}


def check_workflow_id(run: dict) -> str:
    """Read current status identity, or the marker in a legacy Check Run."""
    if "workflow_run_id" in run:
        return run["workflow_run_id"]
    match = re.search(r"<!-- local-ci-workflow:(\d+) -->\Z",
                      (run.get("output") or {}).get("summary") or "")
    return match[1] if match else ""


def check_summary(summary: str, run_id: str) -> str:
    summary = re.sub(r"\n\n<!-- local-ci-workflow:\d+ -->\Z", "", str(summary))
    marker = f"\n\n<!-- local-ci-workflow:{run_id} -->" if run_id.isdigit() else ""
    return summary[:65535 - len(marker)] + marker


def has_native_preflight(task: dict) -> bool:
    """The control branch's own push already has native checks on its tested SHA."""
    return (
        task.get("event_kind") == "push"
        and task.get("pr_number") == 0
        and task.get("target_branch") == "CI_dev"
        and task["worker_revision_sha"] == task["tested_sha"]
    )


def github_sha(task: dict) -> str:
    """Publish PR progress on head; the frozen tested SHA still owns test evidence."""
    return task["head_sha"] if task.get("pr_number") else task["tested_sha"]


def publication_shas(task: dict) -> tuple[str, ...]:
    """Both historical locations, for legacy reads and cleanup only."""
    return tuple(dict.fromkeys((task["tested_sha"], task["head_sha"])
                              if task.get("pr_number") else (task["tested_sha"],)))


class GitHubAPIError(RuntimeError):
    """A diagnostic GitHub failure that never exposes response bodies or credentials."""

    def __init__(self, code: int, method: str, path: str, reason: str = ""):
        self.code = code
        endpoint = path.split("?", 1)[0].lstrip("/")
        detail = f": {reason}" if reason else ""
        super().__init__(
            f"GitHub API {method} {endpoint} failed with HTTP {code}{detail}"
        )


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


FIELD_NAMES = {
    "summary": ("变更概述", "概述", "summary", "change summary"),
    "scope": ("影响范围", "改动范围", "scope", "change scope"),
    "validation": ("验证情况", "验证", "验证方式", "validation"),
}


def pr_fields(description: str) -> dict[str, str]:
    fields: dict[str, list[str]] = {}
    current = ""
    aliases = {alias: key for key, values in FIELD_NAMES.items() for alias in values}
    for line in description.splitlines():
        marker = re.search(r"<!--\s*field:([a-z_]+)\s*-->", line)
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if marker:
            current = marker[1] if marker[1] in FIELD_NAMES else ""
        elif heading:
            pieces = re.split(r"\s*[/|／]\s*", heading[1].lower())
            current = next((aliases[p] for p in pieces if p in aliases), "")
        elif current:
            fields.setdefault(current, []).append(line)
    return {
        key: re.sub(r"<!--.*?-->", "", "\n".join(lines), flags=re.S).strip()
        for key, lines in fields.items()
    }


def validate_pr_info(task: dict) -> list[str]:
    if not task["pr_number"]:
        return []
    fields = pr_fields(task["description"])
    errors = []
    if not task["title"].strip() or task["title"].strip().lower() in {
        "wip",
        "todo",
        "tbd",
    }:
        errors.append("请填写能描述改动目的的 PR 标题。")
    placeholders = re.compile(
        r"^(?:todo|tbd|wip|待填写|待补充|请填写.*|\.\.\.|<.*>)$", re.I | re.S
    )
    for key in ("summary", "scope", "validation"):
        value = fields.get(key, "").strip()
        if not value or placeholders.fullmatch(value):
            errors.append(f"请补充 {FIELD_NAMES[key][0]}（field:{key}）。")
    return errors


class GitHub:
    def __init__(
        self, repository: str, api_url: str | None = None, token: str | None = None
    ):
        self.repository = repository
        self.api_url = (
            api_url or os.getenv("GITHUB_API_URL", "https://api.github.com")
        ).rstrip("/")
        parsed = urlparse(self.api_url)
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            if (
                self.api_url != "https://api.github.com"
                or repository not in REPOSITORIES
            ):
                raise ValueError(
                    "Production GitHub repository/API is outside the allowlist"
                )
        self.token = token if token is not None else os.getenv("GH_TOKEN", "")
        self.run_identities: dict[str, dict] = {}

    def request(self, path: str, method: str = "GET", data: dict | None = None):
        req = Request(
            f"{self.api_url}/repos/{self.repository}/{path.lstrip('/')}",
            data=canonical(data) if data is not None else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "triton-anchor-local-ci",
            },
        )
        try:
            with urlopen(req, timeout=30) as response:
                body = response.read()
                return json.loads(body) if body else None
        except HTTPError as error:
            reason = re.sub(r"[\r\n]+", " ", str(error.reason or "")).strip()[:160]
            raise GitHubAPIError(error.code, method, path, reason) from None

    def optional(self, path: str):
        try:
            return self.request(path)
        except GitHubAPIError as error:
            if error.code == 404:
                return None
            raise

    def content(self, path: str, ref: str) -> bytes:
        data = self.request(
            f"contents/{quote(path, safe='/')}?ref={quote(ref, safe='')}"
        )
        if (
            not isinstance(data, dict)
            or data.get("type") != "file"
            or data.get("encoding") != "base64"
        ):
            raise ValueError(
                f"Expected a vendored file at {path}; configure its trusted mirror before dispatch"
            )
        return base64.b64decode(data["content"], validate=False)

    def gitlinks(self, ref: str) -> list[dict]:
        tree = self.request(f"git/trees/{ref}?recursive=1")
        if tree.get("truncated"):
            raise ValueError("Cannot freeze submodules from an incomplete GitHub tree")
        return [row for row in tree["tree"] if row.get("mode") == "160000"]

    def commit_statuses(self, sha: str) -> dict:
        """GitHub returns newest first; only the latest row per context is live."""
        latest = {}
        for page in range(1, 21):
            rows = self.request(f"commits/{sha}/statuses?per_page=100&page={page}")
            for row in rows:
                latest.setdefault(row["context"], row)
            if len(rows) < 100:
                return latest
        raise ValueError("Cannot determine Local CI ownership from incomplete statuses")

    def post_status(self, sha: str, payload: dict, previous: dict | None = None) -> bool:
        # Statuses are append-only: retries should not add identical rows.
        if previous and all(previous.get(key) == value for key, value in payload.items()):
            return False
        self.request(f"statuses/{sha}", "POST", payload)
        return True

    def status(self, task: dict, state: str, description: str, url: str = "", *,
               existing_only: bool = False, expected_pending_id: int | None = None,
               claim_request: bool = False) -> None:
        if not self.owns_task(task, workflow=claim_request, claim_request=claim_request):
            return
        if state != "error" and not is_current(self, task):
            return
        url = url or workflow_url() or f"https://github.com/{self.repository}/commit/{github_sha(task)}"
        url = url.split("#", 1)[0] + f"#local-ci-task={task['task_id']}"
        for sha, context, previous in self.summary_targets(task, existing_only=existing_only, state=state):
            identity = status_identity(previous or {})
            request_id = os.getenv("LOCAL_CI_REQUEST_ID", "") if claim_request else identity.get("local-ci-request", "")
            if identity.get("local-ci-request") and (
                (claim_request and request_id and request_id != identity["local-ci-request"])
                or (claim_request and not request_id and not identity.get("local-ci-task")
                    and previous.get("state") == "pending")
                or (not claim_request and identity.get("local-ci-task") != task["task_id"])
            ):
                return  # A newer request may arrive after the ownership check.
            # Progress must recheck the actual last-read row, not an earlier poll.
            if expected_pending_id is not None and (
                not previous or previous.get("id") != expected_pending_id
                or previous.get("state") != "pending"
                or status_identity(previous).get("local-ci-task") != task["task_id"]
            ):
                continue
            target = url + (f"&local-ci-request={request_id}" if request_id else "")
            self.post_status(sha, {"state": state, "context": context,
                                  "description": description[:140], "target_url": target}, previous)

    def summary_statuses(self, sha: str) -> dict:
        return {key: row for key, row in self.commit_statuses(sha).items()
                if key in (SUMMARY_CONTEXT, *SUMMARY_ALIASES)}

    def request_allows(self, task: dict, *, claim: bool = False) -> bool:
        row = self.summary_statuses(github_sha(task)).get(SUMMARY_CONTEXT) or {}
        identity = status_identity(row)
        owner = identity.get("local-ci-request")
        if not owner or (row.get("creator") or {}).get("login") != "github-actions[bot]":
            return True
        request_id = os.getenv("LOCAL_CI_REQUEST_ID", "")
        if claim and request_id:
            return owner == request_id
        # A routed request invalidates the previous task before freezing its successor.
        return bool(identity.get("local-ci-task")) or (claim and row.get("state") != "pending")

    def summary_targets(self, task: dict, *, existing_only: bool = False, state: str = ""):
        sha = github_sha(task)
        current = self.summary_statuses(sha)
        previous = current.get(SUMMARY_CONTEXT)
        if not existing_only or previous:
            yield sha, SUMMARY_CONTEXT, previous
        # Never reopen legacy contexts or create a status on the merge SHA.
        # A task dispatched before migration may still have a pending summary
        # there. Finish only its existing pending rows when its real result arrives.
        if state and state != "pending":
            for revision in publication_shas(task):
                rows = current if revision == sha else self.summary_statuses(revision)
                for context, row in rows.items():
                    if revision == sha and context == SUMMARY_CONTEXT:
                        continue
                    if (row.get("state") == "pending"
                            and (row.get("creator") or {}).get("login") == "github-actions[bot]"
                            and status_identity(row).get("local-ci-task") == task["task_id"]):
                        yield revision, context, row

    def latest_summary(self, task: dict, *, sha: str | None = None) -> dict | None:
        candidates = []
        for revision in dict.fromkeys((sha,) if sha else (github_sha(task), task["tested_sha"])):
            rows = self.summary_statuses(revision)
            if revision == github_sha(task) and rows.get(SUMMARY_CONTEXT):
                return rows[SUMMARY_CONTEXT]
            for context in (SUMMARY_CONTEXT, *SUMMARY_ALIASES):
                row = rows.get(context)
                if row:
                    if status_identity(row).get("local-ci-task") == task["task_id"]:
                        return row
                    candidates.append(row)
        return candidates[0] if candidates else None

    def status_matches(self, task: dict, state: str, description: str) -> bool:
        return all(
            latest and latest.get("state") == state
            and latest.get("description") == description[:140]
            and status_identity(latest).get("local-ci-task") == task["task_id"]
            for _, _, latest in self.summary_targets(task, state=state)
        )

    def stage_status(self, task: dict, key: str) -> dict:
        row = self.commit_statuses(github_sha(task)).get(ALL_CHECK_NAMES[key]) or {}
        identity = status_identity(row)
        if (not identity.get("local-ci-task")
                or (row.get("creator") or {}).get("login") != "github-actions[bot]"):
            return {}
        return {**row, "task_id": identity["local-ci-task"],
                "workflow_run_id": identity.get("local-ci-workflow", ""),
                "workflow_run_attempt": identity.get("local-ci-attempt", "1"),
                "status": "in_progress" if row["state"] == "pending" else "completed",
                "conclusion": None if row["state"] == "pending" else
                              "success" if row["state"] == "success" else "failure"}

    def check(
        self, task: dict, key: str, status: str, conclusion: str | None,
        title: str, summary: str, url: str = "", *, restart: bool = False,
        run_id: str | None = None, started_at: str = "",
    ) -> bool:
        """Publish a reached stage; the Basic status identifies the owning run."""
        if key not in ALL_CHECK_NAMES or status not in {"queued", "in_progress", "completed"}:
            raise ValueError("Invalid CI stage identity or status")
        if (status == "completed") != (conclusion is not None) or (
            conclusion and conclusion not in CHECK_CONCLUSIONS
        ):
            raise ValueError("Invalid CI stage conclusion")
        if not is_current(self, task):
            return False
        if not self.request_allows(task, claim=key == "basic" and restart):
            return False
        current_run = os.getenv("GITHUB_RUN_ID", "") if run_id is None else run_id
        attempt = os.getenv("GITHUB_RUN_ATTEMPT", "1")
        start = self.task_start(task)
        current_order = (int(current_run or 0), int(attempt))
        owner_order = (int(check_workflow_id(start) or 0), int(start.get("workflow_run_attempt", "1")))
        # Failed-jobs reruns reuse earlier successful prerequisites in the same run.
        if start and (owner_order > current_order or (
            not (key == "basic" and restart)
            and (start.get("task_id") != task["task_id"] or owner_order[0] != current_order[0])
        )):
            return False
        previous = self.stage_status(task, key)
        if previous and (int(check_workflow_id(previous) or 0),
                         int(previous.get("workflow_run_attempt", "1"))) > current_order:
            return False
        same_run = (previous.get("task_id") == task["task_id"]
                    and check_workflow_id(previous) == current_run
                    and previous.get("workflow_run_attempt") == attempt)
        if same_run and status != "completed" and previous.get("status") == "completed":
            return False  # A duplicate publisher cannot reopen a finished stage.
        state = ("pending" if status != "completed" else
                 "success" if conclusion == "success" else
                 "failure" if conclusion == "failure" else "error")
        description = " ".join((title or f"{ALL_CHECK_NAMES[key]}: {state}").split())[:140]
        url = url or workflow_url() or f"https://github.com/{self.repository}/commit/{github_sha(task)}"
        url = (url.split("#", 1)[0] + f"#local-ci-task={task['task_id']}"
               f"&local-ci-workflow={current_run}&local-ci-attempt={attempt}")
        return self.post_status(github_sha(task), {
            "context": ALL_CHECK_NAMES[key], "state": state,
            "description": description, "target_url": url,
        }, previous)

    def check_runs_named(self, task: dict, name: str, *, sha: str | None = None) -> list[dict]:
        runs = []
        for page in range(1, 21):
            response = self.request(
                f"commits/{sha or github_sha(task)}/check-runs?check_name="
                f"{quote(name, safe='')}&filter=all&per_page=100&page={page}"
            )
            batch = response.get("check_runs", [])
            runs.extend(batch)
            if len(batch) < 100:
                return runs
        raise ValueError("Cannot determine Local CI ownership from incomplete checks")

    def check_runs(self, task: dict, key: str, *, sha: str | None = None) -> list[dict]:
        return [row for name in (READ_CHECK_NAMES[key], *CHECK_ALIASES.get(key, ()))
                for row in self.check_runs_named(task, name, sha=sha)]

    def legacy_stage(self, task: dict, key: str) -> dict:
        # Prefer the tested commit's original check to an older head copy.
        for sha in publication_shas(task):
            rows = [row for row in self.check_runs(task, key, sha=sha)
                    if row.get("name") in (READ_CHECK_NAMES[key], *CHECK_ALIASES.get(key, ()))
                    and (row.get("app") or {}).get("slug") == "github-actions"
                    and str(row.get("external_id", "")).startswith(
                        (f"triton-anchor-local-ci:{key}:", f"triton-anchor-ci-v4:{key}:"))]
            if rows:
                row = max(rows, key=lambda row: int(row["id"]))
                return {**row, "task_id": row["external_id"].rsplit(":", 1)[-1],
                        "workflow_run_id": check_workflow_id(row), "workflow_run_attempt": "1"}
        return {}

    def task_start(self, task: dict) -> dict:
        if has_native_preflight(task):
            # All gateway runs share the control SHA; match the actual subject.
            name = f"Prepare exact task / Branch {task['target_branch']}"
            for row in sorted(self.check_runs_named(task, name), key=lambda row: int(row["id"]), reverse=True):
                if row.get("name") != name or (row.get("app") or {}).get("slug") != "github-actions":
                    continue
                match = re.fullmatch(rf"https://github\.com/{re.escape(self.repository)}/actions/runs/(\d+)/job/\d+", row.get("details_url") or "")
                if not match:
                    continue
                if match[1] not in self.run_identities:
                    self.run_identities[match[1]] = self.request(f"actions/runs/{match[1]}")
                run = self.run_identities[match[1]]
                if (run.get("event") in {"push", "workflow_dispatch"} and run.get("head_branch") == "CI_dev"
                        and run.get("head_sha") == task["tested_sha"]
                        and run.get("path") == ".github/workflows/ci-gateway.yml"):
                    return {**row, "native": True, "task_id": task["task_id"],
                            "workflow_run_id": match[1],
                            "workflow_run_attempt": str(run.get("run_attempt", 1))}
            return {}
        return self.stage_status(task, "basic") or self.legacy_stage(task, "basic")

    def owns_task(self, task: dict, *, workflow: bool = False, claim_request: bool = False) -> bool:
        if not self.request_allows(task, claim=claim_request):
            return False
        start = self.task_start(task)
        if not start:
            start = self.stage_status(task, "dispatch") or self.legacy_stage(task, "dispatch") or self.legacy_stage(task, "preflight")
        if start:
            if start.get("task_id") != task["task_id"]:
                return False
            if workflow and os.getenv("GITHUB_RUN_ID"):
                run_id = os.environ["GITHUB_RUN_ID"]
                attempt = int(os.getenv("GITHUB_RUN_ATTEMPT", "1"))
                if (check_workflow_id(start) != run_id
                        or int(start.get("workflow_run_attempt", "1")) > attempt):
                    return False
                # Re-run failed jobs may leave Basic in an earlier attempt.
                # Once a successor starts retrying, the old finalizer is stale.
                for row in self.commit_statuses(github_sha(task)).values():
                    identity = status_identity(row)
                    if (row.get("context") in ALL_CHECK_NAMES.values()
                            and (row.get("creator") or {}).get("login") == "github-actions[bot]"
                            and identity.get("local-ci-task") == task["task_id"]
                            and identity.get("local-ci-workflow") == run_id
                            and int(identity.get("local-ci-attempt", "1")) > attempt):
                        return False
            return True
        if workflow and os.getenv("GITHUB_RUN_ID"):
            return False
        owner = status_identity(self.latest_summary(task) or {}).get("local-ci-task")
        return owner == task["task_id"] if owner else not has_native_preflight(task)

    def retire_open_checks(self, task: dict, *, superseded: bool = False) -> None:
        """End reached stages; never manufacture statuses for future stages."""
        if not self.owns_task(task):
            return
        start = self.task_start(task)
        run_id = check_workflow_id(start)
        attempt = start.get("workflow_run_attempt", "1")
        sha = github_sha(task)
        for context, row in self.commit_statuses(sha).items():
            if context not in {*ALL_CHECK_NAMES.values(), *RETIRED_STATUS_CONTEXTS}:
                continue
            if (row.get("creator") or {}).get("login") != "github-actions[bot]":
                continue
            identity = status_identity(row)
            ours = (identity.get("local-ci-task") == task["task_id"]
                    and identity.get("local-ci-workflow", "") == run_id
                    and int(identity.get("local-ci-attempt", "1")) >= int(attempt))
            if context in ALL_CHECK_NAMES.values():
                if ours == superseded or (not superseded and row.get("state") != "pending"):
                    continue
            elif row.get("state") != "pending":
                continue
            description = ("Previous task superseded; this stage has not run for the new task"
                           if superseded else "Local CI cancelled: stage did not complete")
            self.post_status(sha, {"context": context, "state": "error", "description": description,
                                   "target_url": row.get("target_url") or ""}, row)
        # Existing Check Runs can only be completed, never deleted. No new
        # Check Runs are created, and completed historical results stay intact.
        for revision in publication_shas(task):
            for page in range(1, 21):
                rows = self.request(f"commits/{revision}/check-runs?filter=all&per_page=100&page={page}").get("check_runs", [])
                for row in rows:
                    if (row.get("status") == "completed"
                            or (row.get("app") or {}).get("slug") != "github-actions"
                            or row.get("name") not in {*READ_CHECK_NAMES.values(), *RETIRED_CHECK_NAMES}):
                        continue
                    identity = str(row.get("external_id", ""))
                    if (row.get("name") not in RETIRED_CHECK_NAMES and not identity.startswith(
                            ("triton-anchor-local-ci:", "triton-anchor-ci-v4:", HEAD_CHECK_PREFIX))):
                        continue
                    ours = identity.endswith(":" + task["task_id"]) and check_workflow_id(row) == run_id
                    if ours == superseded:
                        continue
                    self.request(f"check-runs/{row['id']}", "PATCH", {
                        "status": "completed", "conclusion": "cancelled",
                        "output": {"title": "Superseded task" if superseded else "Stage not completed",
                                   "summary": check_summary("See the current commit statuses and workflow.", check_workflow_id(row))},
                    })
                if len(rows) < 100:
                    break

    def latest_dispatch(self, task: dict) -> dict:
        start = self.task_start(task)
        row = self.stage_status(task, "dispatch") or self.legacy_stage(task, "dispatch")
        if row and row.get("task_id") == task["task_id"] and (
            not start or (check_workflow_id(row) == check_workflow_id(start)
                          and int(row.get("workflow_run_attempt", "1")) >= int(start.get("workflow_run_attempt", "1")))
        ):
            return row
        return {"status": "not_started"} if start else {}

    def finish_inactive_pr(self, pr_number: int) -> bool:
        pr = self.request(f"pulls/{pr_number}")
        if pr["state"] == "open" and not pr.get("draft"):
            return False
        seen = set()
        for sha in dict.fromkeys((pr["head"]["sha"], pr.get("merge_commit_sha"))):
            if sha:
                self._finish_inactive_sha(sha, seen)
        return True

    def _finish_inactive_sha(self, sha: str, seen: set) -> None:
        for context, old in self.commit_statuses(sha).items():
            if (context not in {SUMMARY_CONTEXT, *ALL_CHECK_NAMES.values(), *RETIRED_STATUS_CONTEXTS}
                    or old.get("state") != "pending"
                    or (old.get("creator") or {}).get("login") != "github-actions[bot]"):
                continue
            self.post_status(sha, {"context": context, "state": "error",
                                  "description": "Local CI cancelled: PR closed or became draft",
                                  "target_url": old.get("target_url") or ""}, old)
        for page in range(1, 21):
            rows = self.request(f"commits/{sha}/check-runs?filter=all&per_page=100&page={page}").get("check_runs", [])
            for row in rows:
                if (row.get("name") not in {*READ_CHECK_NAMES.values(), *RETIRED_CHECK_NAMES}
                        or (row.get("app") or {}).get("slug") != "github-actions"
                        or row.get("status") == "completed" or row["id"] in seen):
                    continue
                if (row.get("name") not in RETIRED_CHECK_NAMES and not str(row.get("external_id", "")).startswith(
                        ("triton-anchor-local-ci:", "triton-anchor-ci-v4:", HEAD_CHECK_PREFIX))):
                    continue
                seen.add(row["id"])
                self.request(f"check-runs/{row['id']}", "PATCH", {
                    "status": "completed", "conclusion": "cancelled",
                    "output": {"title": "Local CI cancelled",
                               "summary": check_summary("PR closed or became draft.", check_workflow_id(row))},
                })
            if len(rows) < 100:
                break

    def restore_preflight(self, task: dict) -> None:
        """Verify frozen evidence; legacy Check Runs are read, not copied."""
        if has_native_preflight(task):
            return
        start = self.task_start(task)
        for key in ALL_CHECK_NAMES:
            if key == "approve" and not task.get("external_fork"):
                continue
            row = self.stage_status(task, key) or self.legacy_stage(task, key)
            if not row and key in {"approve", "dispatch"}:
                continue
            if (not row or row.get("task_id") != task["task_id"]
                    or (start and (check_workflow_id(row) != check_workflow_id(start)
                                   or int(row.get("workflow_run_attempt", "1")) < int(start.get("workflow_run_attempt", "1"))))):
                raise ValueError("Missing preflight evidence for this task")
            if row.get("status") != "completed" or row.get("conclusion") != "success":
                raise ValueError("Preflight is not successful for this task")

    def approval_context(self, task: dict) -> dict:
        if not task["pr_number"]:
            return {}
        pull = self.request(f"pulls/{task['pr_number']}")
        if not is_current(self, task):
            raise ValueError("PR changed before approval context collection")
        files = []
        for page in range(1, 31):
            batch = self.request(f"pulls/{task['pr_number']}/files?per_page=100&page={page}")
            files.extend(batch)
            if len(batch) < 100:
                break
        complete = len(files) == pull.get("changed_files", len(files))
        if not is_current(self, task):
            raise ValueError("PR changed during approval context collection")
        paths = [row["filename"] for row in files]
        scopes = {}
        for path in paths:
            if path.startswith((".github/", "scripts/ci/", "scripts/local_ci/", "dashboard/")):
                scope = "CI 与结果展示"
            elif re.search(r"(^|/)(setup\.py|pyproject\.toml|CMakeLists\.txt|.*\.cmake|.*requirements.*|.*lock|Dockerfile.*|llvm-(hash|info).*|\.gitmodules)$", path):
                scope = "依赖与构建配置"
            elif path.startswith(("api_contract/", "include/")):
                scope = "公共接口与契约"
            elif re.search(r"(^|/)(tests?|testing)/|(^|/)test_[^/]+$", path):
                scope = "测试"
            elif path.startswith("docs/") or path.lower().endswith((".md", ".rst")):
                scope = "文档"
            elif path.startswith(("python/", "lib/", "src/", "adapters/", "backends/")):
                scope = "编译器与运行逻辑"
            else:
                scope = "其他"
            scopes[scope] = scopes.get(scope, 0) + 1
        attention = [
            f"{count} 个{' ' if scope[0].isascii() else ''}{scope}文件"
            for scope, count in sorted(scopes.items())
        ]
        return {"author": (pull.get("user") or {}).get("login", "未记录"),
                "source": pull["head"]["repo"]["full_name"], "branch": pull["head"]["ref"],
                "files": paths, "complete": complete, "attention": attention,
                "additions": sum(row.get("additions", 0) for row in files),
                "deletions": sum(row.get("deletions", 0) for row in files)}

    def comment(self, task: dict, body: str, *, event_key: dict | None = None,
                legacy_result_url: str = "") -> bool:
        if not task["pr_number"]:
            return False
        path = f"issues/{task['pr_number']}/comments"
        comments = []
        for page in range(1, 21):
            rows = self.request(f"{path}?per_page=100&page={page}")
            comments.extend(rows)
            if len(rows) < 100:
                break
        else:
            raise ValueError("Cannot deduplicate PR comments from an incomplete listing")
        # Immutable feedback events: new task, phase or result appends a comment.
        # A transport retry of identical content is a no-op, even after other posts.
        event = digest(event_key if event_key is not None else {
            "task_id": task.get("task_id"), "head_sha": task["head_sha"],
            "pr_number": task["pr_number"], "body": body,
        })
        marker_kind = "result" if event_key and event_key.get("kind") == "result" else "event"
        marker = f"<!-- local-ci-feedback {marker_kind}={event} -->"
        # Old result comments have body-based markers. Recognize their immutable
        # report URL during migration, without editing or reposting the comment.
        if legacy_result_url and any(
            row.get("user", {}).get("login") == "github-actions[bot]"
            and str(row.get("body", "")).startswith(MARKER)
            and "<!-- local-ci-feedback result=" not in row.get("body", "")
            and "## Local CI 审查反馈" in row.get("body", "")
            and f"]({legacy_result_url})" in row.get("body", "")
            for row in comments
        ):
            return False
        if any(
            row.get("user", {}).get("login") == "github-actions[bot]"
            and str(row.get("body", "")).startswith(f"{MARKER}\n{marker}\n")
            for row in comments
        ):
            return False
        self.request(path, "POST", {"body": f"{MARKER}\n{marker}\n{body}"[:60000]})
        return True


def verification_trigger_id() -> str:
    """Resolve a new verification once, before freezing its task manifest."""
    forwarded = os.getenv("LOCAL_CI_TRIGGER_ID", "")
    if forwarded:
        return forwarded
    request_id = os.getenv("LOCAL_CI_REQUEST_ID", "")
    action = os.getenv("LOCAL_CI_ACTION", "")
    direct_dispatch = (
        os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"
        and not request_id and not action
    )
    # Older CI Request copies already pass their entry's run and attempt.
    request_attempt = request_id.rsplit(":", 1)[-1]
    routed_rerun = request_attempt.isdigit() and int(request_attempt) > 1
    if action in {"reopened", "manual"} or direct_dispatch or routed_rerun:
        return request_id or f"{os.getenv('GITHUB_RUN_ID', '')}:{os.getenv('GITHUB_RUN_ATTEMPT') or '1'}"
    return ""


def prepare_task(
    gh: GitHub,
    worker_sha: str,
    pr_number: int = 0,
    branch: str = "",
    requested_sha: str = "",
    full: bool = False,
    event_kind: str = "push",
    trigger_id: str = "",
) -> dict:
    if not SHA.fullmatch(worker_sha):
        raise ValueError("Invalid trusted worker revision")
    if pr_number:
        pull = gh.request(f"pulls/{pr_number}")
        if pull["state"] != "open" or pull["draft"]:
            raise ValueError("PR is closed or draft")
        head = pull["head"]["sha"]
        if requested_sha and head != requested_sha:
            raise ValueError("PR changed after the routing event")
        if pull.get("mergeable") is False:
            raise ValueError("PR cannot be merged cleanly")
        merge = pull.get("merge_commit_sha") or ""
        if not isinstance(merge, str) or not SHA.fullmatch(merge):
            raise ValueError(
                "PR merge result is not ready; retry after GitHub finishes computing it"
            )
        parents = gh.request(f"git/commits/{merge}")["parents"]
        if (
            len(parents) != 2
            or parents[0]["sha"] != pull["base"]["sha"]
            or parents[1]["sha"] != head
        ):
            raise ValueError("Merge parents do not match the PR")
        base = parents[0]["sha"]
        branch = pull["base"]["ref"]
        description, title = pull.get("body") or "", pull["title"]
        labels = sorted(row["name"] for row in pull.get("labels", []))
        event_kind = "pull_request"
        ref = f"ci/pr-{pr_number}/{pull['head']['ref']}"
        base_ref, head_ref = (
            f"ci/base/pr-{pr_number}/{pull['head']['ref']}",
            f"ci/head/pr-{pr_number}/{pull['head']['ref']}",
        )
        external = pull["head"]["repo"]["full_name"] != gh.repository
    else:
        head = gh.request(f"branches/{quote(branch, safe='')}")["commit"]["sha"]
        if requested_sha and requested_sha != head:
            raise ValueError("Branch changed after routing")
        merge = head
        parents = gh.request(f"git/commits/{head}").get("parents", [])
        base = parents[0]["sha"] if parents else head
        title, description, labels = f"Branch {branch}", "Trusted branch task", []
        ref = f"ci/{'full' if full else 'push'}/{branch}"
        base_ref, head_ref = f"ci/base/push/{branch}", f"ci/head/push/{branch}"
        external = False
    variants = {}
    for variant, sha in (("base", base), ("candidate", merge)):
        llvm_files = gh.request(f"contents/triton/cmake?ref={quote(sha, safe='')}")
        variants[variant] = {
            "source_sha": sha,
            "llvm_hash": llvm_hash_from_files(
                [entry["path"] for entry in llvm_files if entry["type"] == "file"],
                lambda path: gh.content(path, sha),
            ),
            "triton_version": triton_version_from_source(gh.content(TRITON_VERSION_PATH, sha)),
        }
    task = dict(
        schema=TASK_SCHEMA,
        repository=gh.repository,
        event_kind=event_kind,
        pr_number=pr_number,
        task_ref=ref,
        base_task_ref=base_ref,
        head_task_ref=head_ref,
        tested_sha=merge,
        base_sha=base,
        head_sha=head,
        worker_revision_sha=worker_sha,
        target_branch=branch,
        title=title,
        description=description,
        labels=labels,
        state="open",
        draft=False,
        captured_at=now(),
        llvm_hash=variants["candidate"]["llvm_hash"],
        variants=variants,
        full=full,
        external_fork=external,
    )
    task["metadata_digest"] = metadata_digest(task)
    # The dispatch revision records which Gateway created the manifest.  The
    # server selects and refreshes its own trusted control checkout.
    task["control_policy"] = "worker"
    if trigger_id:
        task["trigger_id"] = trigger_id
    task["task_id"] = compute_task_id(task)
    # Different tasks never move one another's source refs; the manifest is last.
    prefix = (
        f"ci/pr-{pr_number}/{task['task_id']}"
        if pr_number
        else f"ci/branch/{task['task_id']}"
    )
    for field, suffix in (
        ("task_ref", "tested"),
        ("base_task_ref", "base"),
        ("head_task_ref", "head"),
    ):
        task[field] = f"{prefix}/{suffix}"
    mirrors = json.loads(os.getenv("GITEE_SUBMODULE_MIRRORS", "{}"))
    task["submodules"] = []
    # Mirror both variants so an optional base comparison cannot escape to GitHub.
    for variant, sha in (("candidate", merge), ("base", base)):
        for link in gh.gitlinks(sha):
            if link["path"] in PREINSTALLED_SUBMODULES:
                continue
            url = mirrors.get(link["path"], "")
            parsed = urlparse(url)
            if (
                parsed.scheme != "https"
                or parsed.hostname != "gitee.com"
                or parsed.username
                or parsed.password
            ):
                raise ValueError(
                    f"Configure the Gitee submodule mirror for {link['path']}"
                )
            task["submodules"].append(
                {
                    "path": link["path"],
                    "sha": link["sha"],
                    "variant": variant,
                    "repository_url": url,
                    "task_ref": f"{prefix}/submodule/{variant}/{hashlib.sha256(link['path'].encode()).hexdigest()}",
                }
            )
    return validate_task(task)


def is_current(gh: GitHub, task: dict) -> bool:
    if task["repository"] != gh.repository:
        raise ValueError("Task repository differs from the receiver repository")
    if task["pr_number"]:
        pull = gh.request(f"pulls/{task['pr_number']}")
        live = {
            "title": pull["title"],
            "description": pull.get("body") or "",
            "labels": [row["name"] for row in pull.get("labels", [])],
            "state": pull["state"],
            "draft": pull["draft"],
        }
        if (
            pull["head"]["sha"] != task["head_sha"]
            or pull["base"]["ref"] != task["target_branch"]
            or live["state"] != "open"
            or live["draft"]
            or metadata_digest(live) != task["metadata_digest"]
        ):
            return False
        merge = pull.get("merge_commit_sha") or ""
        return bool(
            pull.get("mergeable") is not False
            and SHA.fullmatch(merge)
            and pull["base"].get("sha") == task["base_sha"]
            and merge == task["tested_sha"]
        )
    return (
        gh.request(f"branches/{quote(task['target_branch'], safe='')}")["commit"]["sha"]
        == task["head_sha"]
    )


def validate_approval_environment(gh: GitHub) -> None:
    environment = gh.request("environments/local-ci-fork-approval")
    if not isinstance(environment, dict) or not any(
        rule.get("type") == "required_reviewers"
        and isinstance(rule.get("reviewers"), list)
        and rule["reviewers"]
        for rule in environment.get("protection_rules", [])
        if isinstance(rule, dict)
    ):
        raise ValueError(
            "local-ci-fork-approval must have non-empty required reviewers; configure the existing environment before allowing external fork Local CI"
        )


class GitStore:
    """A temporary clone with optimistic non-force commits, never a user checkout."""

    def __init__(self, url: str, branch: str):
        parsed = urlparse(url)
        local_path = Path(url).exists()
        if not parsed.scheme and not local_path:
            raise ValueError(
                "An unqualified transport must be an existing local test repository"
            )
        if (
            parsed.scheme and not local_path
            and parsed.scheme != "file"
            and (
                parsed.scheme != "https"
                or parsed.hostname != "gitee.com"
                or parsed.username
                or parsed.password
            )
        ):
            raise ValueError(
                "Gitee transport accepts HTTPS gitee.com or local test repositories"
            )
        self.temporary = tempfile.TemporaryDirectory(prefix="local-ci-transport-")
        self.root = Path(self.temporary.name) / "repo"
        self.branch = branch
        self.url = url
        self.env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        askpass = Path(self.temporary.name) / "askpass.sh"
        askpass.write_text(
            '#!/bin/sh\ncase "$1" in *Username*) printf "%s\\n" "$GITEE_USERNAME" ;; *) printf "%s\\n" "$GITEE_TOKEN" ;; esac\n'
        )
        askpass.chmod(0o700)
        self.env["GIT_ASKPASS"] = str(askpass)
        self.root.mkdir()
        self.run("init")
        self.run("remote", "add", "origin", url)
        self.run("config", "user.name", "triton-anchor-ci")
        self.run("config", "user.email", "ci@example.invalid")
        self.refresh()

    def run(self, *args: str, cwd: Path | None = None, check: bool = True) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd or self.root,
            env=self.env,
            text=True,
            capture_output=True,
        )
        if check and result.returncode:
            raise RuntimeError(
                f"Git {args[0]} failed (exit {result.returncode}); check transport/authentication"
            )
        return result.stdout.strip()

    def refresh(self) -> None:
        if self.run("ls-remote", "--heads", "origin", f"refs/heads/{self.branch}"):
            self.run(
                "fetch",
                "--depth=1",
                "origin",
                f"+refs/heads/{self.branch}:refs/remotes/origin/{self.branch}",
            )
            self.run(
                "checkout", "-B", self.branch, f"refs/remotes/origin/{self.branch}"
            )
        elif self.run("rev-parse", "--verify", "HEAD", check=False):
            self.run("checkout", "--orphan", f"init-{self.branch}-{time.time_ns()}")
            self.run("rm", "-rf", "--ignore-unmatch", ".")
        else:
            self.run("symbolic-ref", "HEAD", f"refs/heads/{self.branch}")

    def get(self, path: str):
        location = self.root / path
        return json.loads(location.read_text()) if location.is_file() else None

    def put(self, documents: dict[str, dict], immutable: tuple[str, ...] = ()) -> None:
        for attempt in range(3):
            if attempt:
                self.refresh()
            for name, document in documents.items():
                location = self.root / name
                if location.resolve().is_relative_to(self.root.resolve()) is False:
                    raise ValueError("Unsafe control path")
                old = self.get(name)
                if name in immutable and old is not None and old != document:
                    raise ValueError(f"Immutable record differs: {name}")
                location.parent.mkdir(parents=True, exist_ok=True)
                location.write_bytes(canonical(document) + b"\n")
            self.run("add", "--", *documents)
            if not self.run("diff", "--cached", "--name-only"):
                return
            self.run("commit", "-m", "ci: update v4 control records")
            push = subprocess.run(
                ["git", "push", "origin", f"HEAD:refs/heads/{self.branch}"],
                cwd=self.root,
                env=self.env,
                capture_output=True,
            )
            if push.returncode == 0:
                return
        raise RuntimeError("Gitee control publication failed after three attempts")

    def close(self) -> None:
        self.temporary.cleanup()


def enqueue(task: dict, gh: GitHub, control: GitStore, source: Path) -> None:
    validate_task(task)
    if validate_pr_info(task) or not is_current(gh, task) or not gh.owns_task(task, workflow=True):
        raise ValueError("Task information or freshness no longer permits dispatch")
    if control.get(f"cancel/{task['task_id']}.json"):
        raise ValueError(
            "Task identity was previously cancelled; request a fresh verification"
        )
    gh.check(task, "dispatch", "in_progress", None, "Dispatching Local CI task",
             "Publishing the frozen task and code to Gitee.", workflow_url())
    checked_out = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source)
        .decode()
        .strip()
    )
    if checked_out != task["tested_sha"]:
        raise ValueError("Dispatcher did not check out the frozen tested SHA")
    # All git operations are against the already fetched trusted checkout.
    refs = [
        ("tested_sha", "task_ref"),
        ("base_sha", "base_task_ref"),
        ("head_sha", "head_task_ref"),
    ]
    remote = control.run("remote", "get-url", "origin")
    for sha_key, ref_key in refs:
        subprocess.run(
            ["git", "push", remote, f"{task[sha_key]}:refs/heads/{task[ref_key]}"],
            cwd=source,
            env=control.env,
            check=True,
            capture_output=True,
        )
    for module in task.get("submodules", []):
        with tempfile.TemporaryDirectory(prefix="ci-submodule-") as temporary:
            subprocess.run(
                ["git", "init", "--bare", "--quiet", temporary],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    temporary,
                    "fetch",
                    "--no-tags",
                    module["repository_url"],
                    module["sha"],
                ],
                env=control.env,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    temporary,
                    "push",
                    remote,
                    f"{module['sha']}:refs/heads/{module['task_ref']}",
                ],
                env=control.env,
                check=True,
                capture_output=True,
            )
    if not is_current(gh, task) or not gh.owns_task(task, workflow=True):
        raise ValueError("Task changed while publishing code refs")
    key = f"current/{current_key(task)}.json"
    previous = control.get(key)
    documents = {
        f"tasks/{task['task_id']}.json": task,
        key: {
            "task_id": task["task_id"],
            "repository": task["repository"],
            "pr_number": task["pr_number"],
            "target_branch": task["target_branch"],
            "tested_sha": task["tested_sha"],
            "updated_at": now(),
        },
    }
    if previous and previous["task_id"] != task["task_id"]:
        documents[f"cancel/{previous['task_id']}.json"] = {
            "task_id": previous["task_id"],
            "reason": "superseded",
            "superseded_by": task["task_id"],
            "created_at": now(),
        }
    # On an identical retry preserve the original immutable capture timestamp.
    old = control.get(f"tasks/{task['task_id']}.json")
    if old:
        validate_task(old)
        documents[f"tasks/{task['task_id']}.json"] = old
    control.put(documents, (f"tasks/{task['task_id']}.json",))
    if not is_current(gh, task) or not gh.owns_task(task, workflow=True):
        raise ValueError("Task changed before dispatch status publication")
    description = "Local CI: task published to Gitee; awaiting worker results"
    gh.status(task, "pending", description, workflow_url())
    gh.check(task, "dispatch", "completed", "success", "Local CI task dispatched",
             "The immutable task and code were published to Gitee.", workflow_url())


def cancel_obsolete(gh: GitHub, control: GitStore, pr_number: int = 0, branch: str = "") -> int:
    if not pr_number and not branch:
        raise ValueError("Cancellation requires a PR number or source branch")
    subject = {"repository": gh.repository, "pr_number": pr_number, "target_branch": branch}
    count = 0
    for path in (control.root / "current").glob(current_key(subject) + ".json"):
        row = json.loads(path.read_text())
        task = control.get(f"tasks/{row['task_id']}.json")
        if not task or (pr_number and task["pr_number"] != pr_number):
            continue
        if is_legacy_task(task):
            continue
        validate_task(task)
        if current_key(task) != current_key(subject):
            raise ValueError("Current task pointer belongs to a different PR or branch")
        if not is_current(gh, task):
            name = f"cancel/{task['task_id']}.json"
            cancellation = control.get(name)
            if not cancellation:
                cancellation = {
                    "task_id": task["task_id"],
                    "reason": "PR/branch lifecycle or metadata changed",
                    "created_at": now(),
                }
                control.put({name: cancellation})
                count += 1
            # A newer dispatched task owns the current status. Never let
            # an old cancellation overwrite its result (including same-head edits).
            control.refresh()
            pointer = control.get(f"current/{current_key(task)}.json")
            if (
                pointer
                and pointer["task_id"] == task["task_id"]
                and not cancellation.get("github_notified")
                and gh.owns_task(task)
            ):
                gh.retire_open_checks(task)
                gh.status(
                    task,
                    "error",
                    "Local CI cancelled: PR/branch changed, closed or became draft",
                )
                cancellation["github_notified"] = True
                control.put({name: cancellation})
    return count


DISPLAY_STATES = {
    "pass": "通过", "success": "通过", "fail": "未通过", "failure": "未通过",
    "infra_error": "验证未完成（环境或执行异常）", "error": "异常",
    "cancelled": "已取消", "skipped": "未执行", "not_selected": "本次未选择",
    "warning": "提示", "limited": "验证受限",
    "not_applicable": "不适用", "queued": "等待执行", "in_progress": "执行中",
}
DISPLAY_CHECKS = {
    "prepare": "PR 信息", "basic": "基础检查", "api": "API 兼容性",
    "security": "安全检查", "pr_info": "PR 意图与属性核对", "architecture": "架构契约审查",
    "intent": "专项审查", "environment": "运行环境", "control_plane": "CI 流程验证",
    "change_validation": "变更影响与验证范围", "frontend_build": "前端构建",
    "frontend_install": "前端安装", "frontend_smoke": "前端基本功能",
    "frontend_tests": "前端测试", "backend_build": "后端构建", "backend_install": "后端安装",
    "backend_smoke": "后端基本功能", "backend_tests": "后端测试", "flaggems": "FlagGems 算子验证",
    "compile_time": "编译耗时", "pass_profile": "编译阶段性能", "ir_serialization": "IR 序列化性能",
    "diff_check": "格式检查", "source_syntax": "源码语法检查",
}
COMMENT_OMIT_STATUSES = frozenset({"not_selected", "skipped", "not_applicable"})


def feedback_text(value: object, limit: int = 1600) -> str:
    """Untrusted PR/model prose is plain text, not Markdown links or mentions."""
    text = re.sub(r"\s+", " ", str(value)).strip()[:limit]
    text = html.escape(text).replace("@", "＠").replace("|", "/").replace("`", "'")
    return re.sub(r"([\\\[\]()*_~#!])", r"\\\1", text)


def feedback_prose(value: object, limit: int = 1600) -> str:
    names = {
        **DISPLAY_CHECKS,
        "candidate/base": "候选/base", "baseline/candidate": "base/候选",
        "base/candidate": "base/候选", "candidate/baseline": "候选/base",
        "candidate": "候选", "baseline": "base",
        "冻结任务上下文": "任务信息", "冻结上下文": "验证配置",
        "task-context": "任务信息", "context": "验证配置", "variant": "源码版本",
        "pull_request": "PR",
        "profile": "环境配置", "venv": "Python 虚拟环境", "checkout": "源码目录",
        "控制面": "CI 流程", "symlink": "符号链接",
        "limited": DISPLAY_STATES["limited"], "infra_error": DISPLAY_STATES["infra_error"],
        "backend tests": "后端测试", "backend smoke": "后端基本功能验证",
        "frontend smoke": "前端基本功能验证", "smoke": "基本功能验证",
    }
    pattern = r"(?<![A-Za-z0-9_/.-])(" + "|".join(map(re.escape, names)) + r")(?![A-Za-z0-9_/.-])"
    # Literal commands and field names are not contributor-facing prose.
    text = "".join(
        part if part.startswith("`") else re.sub(pattern, lambda match: names[match[0].lower()], part, flags=re.I)
        for part in re.split(r"(`[^`]*`)", str(value))
    )
    return feedback_text(text, limit)


def dashboard_url(task: dict) -> str:
    """Return a stable, PR-filtered link to the published task dashboard."""
    configured = os.getenv("DASHBOARD_URL") or os.getenv("GITHUB_PAGES_URL")
    if configured:
        parsed = urlparse(configured)
        if parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password:
            page = configured.rstrip("/")
        else:
            page = ""
    else:
        owner, _, repository = task.get("repository", REPOSITORY).partition("/")
        page = f"https://{owner}.github.io/{repository}/local-ci.html"
    if not page:
        return ""
    if not page.lower().endswith(".html"):
        page += "/local-ci.html"
    if task.get("pr_number"):
        page += ("&" if "?" in page else "?") + f"pr={task['pr_number']}"
    return page


def display_state(value: str) -> str:
    return DISPLAY_STATES.get(value, "未知状态（" + feedback_text(value) + "）")


def feedback_evidence(item: dict, task: dict, artifact_urls: dict) -> str:
    links = []
    references = item.get("code_evidence", [])
    if isinstance(references, (str, dict)):
        references = [references]
    if not isinstance(references, list):
        return ""
    for reference in references:
        if isinstance(reference, dict):
            reference = str(reference.get("path", "")) + (
                ":" + str(reference["line"]) if reference.get("line") else ""
            )
        if not isinstance(reference, str):
            continue
        match = re.fullmatch(r"([\w./-]+)(?::([1-9][0-9]*)(?:-([1-9][0-9]*))?)?", reference)
        if match and not match[1].startswith("/") and not {"..", ".", ""}.intersection(match[1].split("/")):
            if match[1] in artifact_urls or (match[3] and int(match[3]) < int(match[2])):
                continue
            url = f"https://github.com/{task['repository']}/blob/{task['tested_sha']}/" + quote(match[1], safe="/")
            if match[2]:
                url += "#L" + match[2]
                if match[3]:
                    url += "-L" + match[3]
            links.append(f"[{feedback_text(reference, 200)}]({url})")
    return " · ".join(list(dict.fromkeys(links))[:3])


def result_comment(result: dict, result_url: str = "", artifact_urls: dict | None = None) -> str:
    task = result["task"]
    records = [*result["checks"], *result["reviews"]]
    visible_records = [
        item for item in records if item.get("status") not in COMMENT_OMIT_STATUSES
    ]
    verdict = {
        "pass": "本次要求的检查已通过。",
        "fail": "发现合入阻塞，需要处理后重新验证。",
        "infra_error": "验证尚未完成，暂不能确认可合入。",
        "cancelled": "验证已取消，不能作为当前提交的通过依据。",
    }
    lines = [
        "## Local CI 审查反馈", "", f"**结论：{verdict[result['status']]}**", "",
        f"PR 提交：`{task['head_sha']}`", "",
        f"合并后验证提交：`{task['tested_sha']}`",
        "", "### 变更意图与审查结论", "", feedback_prose(result["summary"]),
    ]
    findings = []
    has_blocking_findings = False
    limitations = [feedback_prose(item) for item in result.get("limitations", [])]
    for item in records:
        if item.get("status") in {"infra_error", "cancelled"} and "limitations" not in result:
            name = item.get("tool_id", item.get("kind", ""))
            limitations.append(feedback_prose(f"{DISPLAY_CHECKS.get(name, '补充检查')}：{item.get('summary') or display_state(item['status'])}"))
    for finding in result["findings"]:
        text = feedback_prose(finding.get("summary", ""))
        if text:
            risk = {"critical": "严重", "high": "高", "medium": "中", "low": "低", "info": "提示"}.get(finding.get("severity"), "未标注")
            blocking = finding.get("blocking") or finding.get("severity") in {"high", "critical"}
            has_blocking_findings |= bool(blocking)
            label = "合入阻塞" if blocking else f"风险：{risk}"
            if analysis := finding.get("qualification"):
                text += "  \n  分析：" + feedback_prose(analysis)
            evidence = feedback_evidence(finding, task, artifact_urls or {})
            if evidence:
                text += "  \n  代码位置：" + evidence
            findings.append(f"【{label}】{text}")
    # Findings are the reviewed issue list; failed checks are evidence, not extra defects.
    if not has_blocking_findings:
        reasons = [feedback_prose(reason) for reason in result["blocking_reasons"]]
        if result["status"] == "fail":
            findings.extend(f"【合入阻塞】{reason}" for reason in reasons if reason not in limitations)
        else:
            limitations.extend(reasons)
    if findings:
        lines.extend(["", "### 需要关注的发现", "", *(f"- {x}" for x in dict.fromkeys(findings))])
    lines.extend(["", "### 查看审查详情"])
    if visible_records:
        lines.extend(["", "<details>", "<summary>展开已执行的检查与审查记录</summary>", "",
                      "| 检查 | 结果 | 说明 |", "| --- | --- | --- |"])
    else:
        lines.extend(["", "本次评论没有可列出的已执行检查或审查记录；未选择、未执行和不适用项已保留在 Dashboard。"])
    for item in visible_records:
        name = item.get("tool_id", item.get("kind", ""))
        label = DISPLAY_CHECKS.get(name, feedback_prose(item.get("display_name") or "补充检查", 100))
        state = display_state(item["status"])
        detail = feedback_prose(item.get("summary", ""))
        lines.append(f"| {label} | {state} | {detail} |")
    if visible_records:
        lines.extend(["", "</details>"])
    dashboard = dashboard_url(task)
    if dashboard:
        lines.extend(["", f"[在 Dashboard 查看本次任务详情]({dashboard})"])
    delivery = result.get("evidence_delivery") or {}
    if delivery.get("status") == "incomplete":
        limitations.append(
            "必传检查证据未完整发布，整体结论待确认；已执行检查结果保持原状态。"
            if result["status"] == "infra_error" and any(row.get("required") for row in delivery.get("omitted", []))
            else "执行通过，证据发布不完整。"
            if result["status"] == "pass"
            else "证据发布不完整，已执行检查结果保持原状态。"
        )
    if not limitations and not findings and result["status"] != "pass":
        limitations.append("尚无完整通过结论，请补齐验证。")
    if limitations:
        lines.extend(["", "### 限制说明", "", *(f"- {x}" for x in dict.fromkeys(limitations))])
    if result_url:
        lines.extend(["", f"[完整执行报告与所选文件]({result_url})"])
    return "\n".join(lines)


def preflight_passed(stages: dict) -> bool:
    return all(stages.get(key) == "success" for key in ("prepare", *CHECK_NAMES))


def pr_info_comment(errors: list[str]) -> str:
    return (
        "## PR 信息需要补充\n\n"
        "感谢您的贡献！为了帮助维护者理解这次改动并安排合适的验证，请补充以下信息：\n\n"
        + "\n".join(f"- {feedback_text(error)}" for error in errors)
        + "\n\n请直接更新 PR 描述，系统会重新检查。PR 信息检查通过后会进入后续检查与必要验证"
    )


def contributor_mention(pull: dict | None) -> str:
    login = ((pull or {}).get("user") or {}).get("login", "")
    return f"@{login} " if re.fullmatch(r"[A-Za-z0-9-]{1,39}", str(login)) else ""


def preflight_failure_comment(error: BaseException, link: str = "", pull: dict | None = None) -> str:
    """Explain the failure boundary without blaming a workflow by default."""
    text = str(error)
    if isinstance(error, GitHubAPIError) and (error.code >= 500 or error.code == 429):
        category = "GitHub 服务暂时不可用或受到限流"
        explanation = "失败发生在 GitHub 服务请求阶段，不能据此判断 PR 代码或工作流逻辑失败。"
    elif re.search(r"merge result is not ready|Merge parents do not match|PR merge result", text, re.I):
        category = "GitHub 尚未完成 PR 合并状态准备"
        explanation = "GitHub 侧的 merge ref/父提交状态尚未稳定，不能据此判断 PR 代码或工作流逻辑失败。"
    elif re.search(r"Gitee|transport|publication|code refs|control publication|result publication", text, re.I):
        category = "Gitee 中转或投递阶段异常"
        explanation = "失败发生在任务投递或结果发布边界，不等同于 Basic、API 或 Security 检查失败。"
    else:
        category = "原因待确认"
        explanation = "目前只能确认准备/投递阶段没有完成；现有信息不足以把原因归于 GitHub、工作流、Gitee 或 PR 代码。"
    detail = feedback_text(text if isinstance(error, (ValueError, GitHubAPIError)) and text else type(error).__name__)
    lines = [
        contributor_mention(pull) + "## CI 准备或投递未完成",
        "",
        f"**初步归因：{category}。**",
        "",
        explanation,
        "",
        f"原始阶段信息：{detail}",
        "",
        "请查看本次工作流证据；确认 GitHub PR 状态和中转服务恢复后，再重新触发本次检查。",
    ]
    if link:
        lines.extend(["", f"[查看工作流证据]({link})"])
    return "\n".join(lines)


def rejected_approval(gh: GitHub) -> dict | None:
    """Return the explicit environment rejection for this workflow, if any."""
    run_id = os.getenv("GITHUB_RUN_ID", "")
    if not run_id.isdigit():
        return None
    reviews = gh.optional(f"actions/runs/{run_id}/approvals")
    if reviews is None:
        return None
    if not isinstance(reviews, list):
        raise ValueError("Invalid workflow approval history")
    for review in reversed(reviews):
        environments = review.get("environments") or []
        if (
            review.get("state") == "rejected"
            and any(row.get("name") == "local-ci-fork-approval" for row in environments)
        ):
            return review
    return None


def approval_rejection_comment(gh: GitHub, task: dict, review: dict, url: str = "") -> str:
    try:
        pull = gh.request(f"pulls/{task['pr_number']}")
    except (GitHubAPIError, OSError, ValueError):
        # The notification itself should still be attempted when the PR
        # lookup is the transiently failing GitHub request; omit only the
        # contributor mention if the login cannot be read.
        pull = None
    mention = contributor_mention(pull).strip()
    lines = [
        (mention + "   " if mention else "") + "**进入 Local CI 审批未通过**",
        "",
        f"PR 提交：`{task['head_sha']}`",
    ]
    reviewer = feedback_text(((review.get("user") or {}).get("login") or ""), 80)
    comment = feedback_text(review.get("comment") or "", 1200)
    if reviewer:
        lines.extend(["", f"审核者：{reviewer}"])
    if comment:
        lines.extend(["", "审核批注：", "", f"> {comment}"])
    lines.extend(["", "如有疑问可进一步联系审核者进行处理，感谢您的贡献！"])
    if url:
        lines.extend(["", f"[查看审批记录]({url})"])
    return "\n".join(lines)


def approval_card(task: dict, stages: dict, eligible: bool, approval_error: str = "",
                  context: dict | None = None) -> str:
    if not task.get("external_fork") or not preflight_passed(stages):
        return ""
    lines = ["## Local CI 前置检查与审批", "",
             f"PR #{task['pr_number']}：{feedback_text(task['title'])}", "",
             "| 前置检查 | 结果 |", "| --- | --- |"]
    for key in ("prepare", *CHECK_NAMES):
        lines.append(f"| {DISPLAY_CHECKS[key]} | {display_state(stages.get(key, 'skipped'))} |")
    lines.extend(["", "### 本次审批对应的固定版本", "",
                  f"- 目标分支：{feedback_text(task['target_branch'])}"])
    for key, label in (("head_sha", "PR 提交"), ("base_sha", "目标分支基线"),
                       ("tested_sha", "被测合并提交")):
        lines.append(f"- {label}：`{task[key]}`")
    if context:
        lines.extend(["", "### 本次改动概览", "",
                      f"- 贡献者：{feedback_text(context['author'])}",
                      f"- 来源：{feedback_text(context['source'])} / {feedback_text(context['branch'])}",
                      f"- 改动：{len(context['files'])} 个文件，+{context['additions']} / -{context['deletions']} 行" + ("" if context['complete'] else "（列表不完整）"),
                      f"- 改动范围（按文件路径归类）：{feedback_text('；'.join(context['attention']) or '暂无文件记录')}",
                      "", f"[查看完整文件差异](https://github.com/{task['repository']}/pull/{task['pr_number']}/files)"])
    if approval_error:
        lines.extend(["", "审批环境配置无法确认，暂不派发：" + feedback_text(approval_error)])
    if not eligible:
        lines.extend(["", "**未进入 Local CI。** 请先处理未通过或未完成的前置检查/审批配置。"])
    elif task.get("external_fork"):
        lines.extend(["", "**外部 fork：等待维护者审批。** 请在下方工作流的 `local-ci-fork-approval` 环境批准或拒绝本次运行。"])
    else:
        lines.extend(["", "**同仓库任务：前置检查已通过，无需外部 fork 审批，准备进入 Local CI。**"])
    if workflow_url():
        lines.extend(["", f"[前置检查证据与审批入口]({workflow_url()})"])
    return "\n".join(lines)


GITHUB_STATES = {
    "pass": "success",
    "fail": "failure",
    "infra_error": "error",
    "cancelled": "error",
}


def publication_description(status: str, result_digest: str) -> str:
    # The digest binds task, run and all evidence without another delivery record.
    return f"Local CI: {status} (result {result_digest})"


def workflow_url() -> str:
    run_id = os.getenv("GITHUB_RUN_ID", "")
    repository = os.getenv("GITHUB_REPOSITORY", REPOSITORY)
    server = os.getenv("GITHUB_SERVER_URL", "https://github.com").rstrip("/")
    return f"{server}/{repository}/actions/runs/{run_id}" if run_id.isdigit() else ""


def check_value(value: object, limit: int = 300) -> str:
    return (
        re.sub(r"\s+", " ", str(value))
        .strip()
        .replace("@", "＠")
        .replace("|", "/")[:limit]
        or "—"
    )


def publish_preflight_checks(
    gh: GitHub, task: dict, stages: dict, eligible: bool, *, partial: bool = False
) -> bool:
    changed = False
    for key in CHECK_NAMES:
        if partial and key not in stages:
            continue
        if not partial and stages.get("prepare", "success") != "success":
            break
        outcome = str(stages.get(key, "skipped"))
        if outcome == "skipped":
            continue
        # GitHub treats skipped/neutral required checks as passing. Keep an
        # unexecuted prerequisite blocking without misreporting a test failure.
        conclusion = {"success": "success", "failure": "failure", "cancelled": "cancelled"}.get(outcome, "action_required")
        changed |= gh.check(
            task,
            key,
            "completed",
            conclusion,
            "" if outcome == "success" else f"{CHECK_NAMES[key]}: {check_value(outcome)}",
            f"Trusted preflight stage: **{check_value(outcome)}**. Unexecuted required checks do not pass. "
            f"[Workflow evidence]({workflow_url()})",
            workflow_url(),
        )
        if not partial and outcome != "success":
            break
    return changed


def sync_preflight(gh: GitHub, task: dict, stages: dict) -> bool:
    """Publish stage progress; workflow dependencies independently gate execution."""
    if not isinstance(stages, dict) or not stages or set(stages) - CHECK_NAMES.keys() or any(
        not isinstance(value, str) or value not in {"success", "failure", "cancelled", "skipped"}
        for value in stages.values()
    ):
        raise ValueError("Invalid preflight stage results")
    if not is_current(gh, task) or not gh.owns_task(task, workflow=True) or gh.latest_dispatch(task).get("status") == "completed":
        return False
    changed = publish_preflight_checks(gh, task, stages, False, partial=True)
    keys = list(CHECK_NAMES)
    for key, outcome in stages.items():
        if outcome == "success" and keys.index(key) + 1 < len(keys):
            next_key = keys[keys.index(key) + 1]
            changed |= gh.check(task, next_key, "in_progress", None,
                                f"{CHECK_NAMES[next_key]}: running",
                                "The preceding check passed; this stage can now run.", workflow_url())
    return changed


def begin_checks(gh: GitHub, task: dict) -> None:
    if not is_current(gh, task):
        raise ValueError("Task changed before preflight initialization")
    previous = gh.task_start(task)
    previous_run = check_workflow_id(previous)
    current_run = os.getenv("GITHUB_RUN_ID", "")
    if previous_run.isdigit() and current_run.isdigit() and (
        int(previous_run), int(previous.get("workflow_run_attempt", "1"))
    ) > (int(current_run), int(os.getenv("GITHUB_RUN_ATTEMPT", "1"))):
        raise ValueError("A newer workflow owns this task")
    claimed = gh.check(task, "basic", "queued", None, "Checking task information",
                       "Basic CI will start after task information is checked.",
                       workflow_url(), restart=True)
    # A successful status write is the ownership claim. GitHub's list API
    # can briefly return the previous output immediately after that write.
    if not claimed and not gh.owns_task(task, workflow=True, claim_request=True):
        raise ValueError("A newer workflow owns this task")
    if not claimed and gh.stage_status(task, "basic").get("status") == "completed":
        return
    gh.status(task, "pending", "Local CI: preflight checks in progress", workflow_url(), claim_request=True)
    gh.retire_open_checks(task, superseded=True)


def finalize_preflight(gh: GitHub, task: dict, stages: dict) -> None:
    if not gh.owns_task(task, workflow=True):
        return
    previous = gh.latest_summary(task) or {}
    pending = (previous.get("state") == "pending" and
               status_identity(previous).get("local-ci-task") == task["task_id"])
    if not is_current(gh, task):
        gh.retire_open_checks(task)
        if pending:
            gh.status(task, "error", "Local CI: task superseded; verification stopped", workflow_url(),
                      existing_only=True, expected_pending_id=previous.get("id"))
        return
    if stages.get("enqueue") == "success":
        # The receiver owns summary after dispatch, possibly already completed.
        return
    dispatch = gh.latest_dispatch(task)
    if dispatch.get("status") == "completed" and dispatch.get("conclusion") == "success":
        # Delivery succeeded; a receiver startup error must not overwrite a result
        # that another receiver has already published.
        if pending:
            description = "Local CI: receiver startup failed; retry receive for the published task"
            gh.status(task, "error", description, workflow_url())
        return
    publish_preflight_checks(gh, task, stages, False)
    gh.retire_open_checks(task)
    if not preflight_passed(stages):
        if pending:
            failed = next((name for key, name in CHECK_NAMES.items() if stages.get(key) == "failure"), None)
            description = (f"Local CI: {failed} failed; subsequent verification not run" if failed else
                           "Local CI: preflight interrupted; task not dispatched")
            gh.status(task, "failure" if failed else "error", description, workflow_url())
        return  # Only the reached prerequisite checks are completed above.
    if task.get("external_fork") and stages.get("card", "success") != "success":
        description = "Local CI: approval card publication failed; worker verification not started"
    elif task.get("external_fork") and stages.get("approval") != "success":
        description = ("Local CI: approval cancelled; worker verification not started"
                       if stages.get("approval") == "cancelled" else
                       "Local CI: approval rejected or verification failed; worker verification not started")
    else:
        description = "Local CI: task dispatch failed; see workflow"
    key = "approve" if (
        task.get("external_fork")
        and (
            stages.get("card") in {"failure", "cancelled"}
            or stages.get("approval") in {"failure", "cancelled", "skipped"}
        )
    ) else "dispatch"
    if key == "dispatch" and stages.get("enqueue") not in {"failure", "cancelled", "skipped"}:
        return
    detail = description
    if key == "approve" and workflow_url():
        detail += f"\n\n[Open approval controls and workflow evidence]({workflow_url()})"
    gh.check(task, key, "completed", "cancelled" if "cancelled" in stages.values() else "failure",
             description, detail, workflow_url())
    review = None
    if (
        key == "approve"
        and task.get("external_fork")
        and stages.get("card") == "success"
        and stages.get("approval") == "failure"
    ):
        review = rejected_approval(gh)
    if review:
        gh.comment(
            task,
            approval_rejection_comment(gh, task, review, workflow_url()),
            event_key={
                "kind": "approval-rejected",
                "task_id": task["task_id"],
                "review": digest({
                    "reviewer": (review.get("user") or {}).get("login"),
                    "comment": review.get("comment"),
                }),
            },
        )
    if pending:
        gh.status(task, "failure" if review else "error", description, workflow_url())


def retired_task(gh: GitHub, control: GitStore, task: dict) -> bool:
    """Return true only for a task that has definitively lost ownership."""
    pointer = control.get(f"current/{current_key(task)}.json")
    return bool(
        not pointer
        or pointer.get("task_id") != task["task_id"]
        or control.get(f"cancel/{task['task_id']}.json")
        or not is_current(gh, task)
        or not gh.owns_task(task)
    )


def current_task(gh: GitHub, control: GitStore, task: dict) -> bool:
    if retired_task(gh, control, task):
        return False
    dispatch = gh.latest_dispatch(task)
    return bool(
        not dispatch
        or (
            dispatch.get("status") == "completed"
            and dispatch.get("conclusion") == "success"
        )
    )


def latest_result(task: dict, results: GitStore) -> Path | None:
    paths = sorted(
        (
            path for prefix in result_task_prefixes(task)
            for path in (results.root / prefix).glob("*/result.json")
        ),
        key=lambda path: path.parent.name,
        reverse=True,
    )
    for path in paths:
        # SHA directories may contain different frozen tasks. Never mix their results.
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                return path  # Let read_result reject it; do not reuse an old pass.
            recorded = json.loads(path.read_bytes()).get("task", {}).get("task_id")
        except (ValueError, AttributeError):
            return path
        if recorded and recorded != task["task_id"]:
            continue
        return path
    return None


def read_result(path: Path, task: dict, results: GitStore) -> tuple[dict, str]:
    raw = path.read_bytes()
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("Result exceeds the small Git document budget")
    result = validate_result(json.loads(raw), task)
    if path.parent.name != result["run_id"]:
        raise ValueError("Result path/run id mismatch")
    for artifact in result["artifacts"]:
        if not artifact.get("omitted"):
            source = within(
                path.parent / "artifacts", artifact["path"], must_exist=True
            )
            if source.stat().st_size != artifact.get("size"):
                raise ValueError("Result artifact is missing or incomplete")
    return result, hashlib.sha256(raw).hexdigest()


def receive_result(
    gh: GitHub, url: str, task_id: str, round_number: int = 1
) -> str:
    """Wait for one frozen task; only a bounded continuation creates another run."""
    if not ID.fullmatch(task_id) or not 1 <= round_number <= RECEIVER_MAX_ROUNDS:
        raise ValueError(
            "Receiver requires a valid task_id and round between 1 and 3"
        )
    deadline = time.monotonic() + RECEIVER_WAIT_SECONDS
    control = results = None
    progress = ReceiverProgress()
    try:
        while True:
            try:
                if control is None:
                    control = GitStore(url, CONTROL_BRANCH)
                else:
                    control.refresh()
                task = validate_task(control.get(f"tasks/{task_id}.json"))
                if (
                    task["task_id"] != task_id
                    or task["repository"] != gh.repository
                ):
                    raise ValueError(
                        "Receiver task identity does not match the request"
                    )
                if retired_task(gh, control, task):
                    return "obsolete"
                dispatch = gh.latest_dispatch(task)
                if not dispatch or (
                    dispatch.get("status") == "completed"
                    and dispatch.get("conclusion") == "success"
                ):
                    if results is None:
                        results = GitStore(url, RESULTS_BRANCH)
                    else:
                        results.refresh()
                    path = latest_result(task, results)
                    if path:
                        read_result(path, task, results)
                        return "ready"
                    progress.update(gh, task)
                elif dispatch.get("status") == "completed":
                    return "dispatch_failed"
            except (OSError, RuntimeError) as error:
                if (
                    isinstance(error, GitHubAPIError)
                    and error.code < 500
                    and error.code != 429
                ):
                    raise
                if time.monotonic() >= deadline:
                    raise ValueError(
                        "Local CI receiver transport failed until the waiting deadline; "
                        f"retry receive for task {task_id}. The server task was not cancelled."
                    ) from None
                print(
                    f"Receiver transport unavailable ({type(error).__name__}); retrying"
                )
            else:
                if time.monotonic() >= deadline:
                    if round_number < RECEIVER_MAX_ROUNDS:
                        gh.request(
                            "actions/workflows/ci-gateway.yml/dispatches",
                            "POST",
                            {
                                "ref": "main",
                                "inputs": {
                                    "mode": "receive",
                                    "task_id": task_id,
                                    "receiver_round": str(round_number + 1),
                                    "run_title": (
                                        f"PR #{task['pr_number']} | h:{task['head_sha'][:7]} "
                                        f"m:{task['tested_sha'][:7]}"
                                        if task["pr_number"]
                                        else f"Branch {task['target_branch']} | h:{task['tested_sha'][:7]}"
                                    ),
                                },
                            },
                        )
                        return "continued"
                    gh.status(
                        task,
                        "error",
                        "Local CI: receiver timed out; retry receive without rebuilding",
                    )
                    raise ValueError(
                        f"Local CI receiver timed out after {RECEIVER_MAX_ROUNDS} rounds; "
                        f"retry receive for task {task_id}. The server task was not cancelled."
                    )
            time.sleep(
                min(RECEIVER_POLL_SECONDS, max(0, deadline - time.monotonic()))
            )
    finally:
        if results is not None:
            results.close()
        if control is not None:
            control.close()


def result_links(results: GitStore, path: Path, result: dict) -> tuple[str, dict]:
    base = results.url.removesuffix(".git").rstrip("/")
    if not base.startswith("https://gitee.com/"):
        return "", {}
    prefix = f"{base}/blob/{RESULTS_BRANCH}/" + quote(
        path.parent.relative_to(results.root).as_posix(), safe="/"
    )
    links = {
        artifact["path"]: prefix + "/artifacts/" + quote(artifact["path"], safe="/")
        for artifact in result["artifacts"]
        if not artifact.get("omitted")
    }
    return prefix + "/result.json", links


def publication_error(gh: GitHub, control: GitStore, task: dict) -> None:
    # Failed transports may also prevent this best-effort error status.
    try:
        control.refresh()
        if current_task(gh, control, task):
            gh.status(
                task,
                "error",
                "Local CI: result validation or status publication failed; receiver will retry",
            )
    except (ValueError, OSError, RuntimeError):
        pass


def inactive_task_status(control: GitStore, task: dict) -> str:
    cancellation = control.get(f"cancel/{task['task_id']}.json")
    return "cancelled" if cancellation and cancellation.get("reason") != "superseded" else "superseded"


def owned_pending_summary(gh: GitHub, task: dict) -> dict | None:
    """Return the pending Summary only while this exact task still owns it."""
    row = gh.latest_summary(task) or {}
    if (
        row.get("state") == "pending"
        and isinstance(row.get("id"), int)
        and (row.get("creator") or {}).get("login") == "github-actions[bot]"
        and status_identity(row).get("local-ci-task") == task["task_id"]
    ):
        return row
    return None


def finish_obsolete_summary(gh: GitHub, control: GitStore, task: dict) -> bool:
    """CAS-close only an obsolete task's own pending Summary."""
    if not retired_task(gh, control, task):
        return False
    previous = owned_pending_summary(gh, task)
    if not previous:
        return False
    superseded = inactive_task_status(control, task) == "superseded"
    gh.retire_open_checks(task)
    description = (
        "Local CI: task superseded; no new worker verification started"
        if superseded
        else "Local CI: task cancelled before worker verification"
    )
    gh.status(
        task,
        "error",
        description,
        workflow_url(),
        existing_only=True,
        expected_pending_id=previous["id"],
    )
    return True


def collect_results(
    gh: GitHub, control: GitStore, results: GitStore, dashboard: Path,
    task_id: str | None = None, *, reconcile_pending: bool = False,
) -> list[dict]:
    """Read all dashboard rows; publish only explicit or safely reconciled tasks."""
    task_id = os.getenv("RECEIVER_TASK_ID", "") if task_id is None else task_id
    if task_id and not ID.fullmatch(task_id):
        raise ValueError("Receiver task ID must be an exact task identity")
    if not isinstance(reconcile_pending, bool):
        raise ValueError("Pending-result reconciliation must be explicitly enabled")
    requested_task = None
    if task_id:
        requested_task = validate_task(control.get(f"tasks/{task_id}.json"))
        if requested_task["task_id"] != task_id:
            raise ValueError("Receiver task manifest identity differs")
    rows, published = [], []
    requested_seen = False
    receiver_errors = 0
    for current in sorted((control.root / "current").glob("*.json")):
        pointer = json.loads(current.read_text())
        task = control.get(f"tasks/{pointer['task_id']}.json")
        if is_legacy_task(task):
            continue
        validate_task(task)
        retired = retired_task(gh, control, task)
        active = current_task(gh, control, task)
        requested = task["task_id"] == task_id
        requested_seen |= requested
        pending = owned_pending_summary(gh, task) if reconcile_pending and active else None
        publication_selected = requested or pending is not None
        waiting_dispatch = requested and not active and not retired
        row = {
            "task": task,
            "status": inactive_task_status(control, task) if retired else "pending",
            "historical": retired,
            "result": None,
        }
        status_published = False
        obsolete_checked = False
        try:
            if requested and retired:
                status_published = finish_obsolete_summary(gh, control, task)
                obsolete_checked = True
            elif waiting_dispatch:
                raise ValueError(
                    "Receiver task dispatch has not reached a publishable state"
                )
            path = latest_result(task, results)
            if path:
                result, result_digest = read_result(path, task, results)
                result_url, artifact_urls = result_links(results, path, result)
                row.update(
                    status=(
                        result["status"]
                        if active
                        else inactive_task_status(control, task) if retired else "pending"
                    ),
                    result=result,
                    result_url=result_url,
                    artifact_urls=artifact_urls,
                )
                if publication_selected and active:
                    # Only GitHub writes require a fresh control snapshot.
                    # Display-only rows reuse the snapshot fetched at collection start.
                    control.refresh()
                    active = current_task(gh, control, task)
                    if not active:
                        retired = retired_task(gh, control, task)
                        row.update(
                            status=inactive_task_status(control, task) if retired else "pending",
                            historical=retired,
                        )
                if publication_selected and active:
                    gh.restore_preflight(task)
                    state = GITHUB_STATES[result["status"]]
                    description = publication_description(
                        result["status"], result_digest
                    )
                    unchanged = gh.status_matches(task, state, description)
                    if not unchanged:
                        gh.status(
                            task,
                            state,
                            description,
                            result_url,
                            expected_pending_id=pending["id"] if pending else None,
                        )
                    status_published = True
                    control.refresh()
                    if not current_task(gh, control, task):
                        retired = retired_task(gh, control, task)
                        row.update(
                            status=inactive_task_status(control, task) if retired else "pending",
                            historical=retired,
                        )
                        rows.append(row)
                        continue
                    changed = gh.comment(
                        task, result_comment(result, result_url, artifact_urls),
                        event_key={"kind": "result", "task_id": task["task_id"],
                                   "run_id": result["run_id"], "result_digest": result_digest},
                        legacy_result_url=result_url,
                    )
                    if not unchanged or changed:
                        published.append(
                            {
                                "task_id": task["task_id"],
                                "run_id": result["run_id"],
                                "tested_sha": task["tested_sha"],
                                "status": result["status"],
                                "result_digest": result_digest,
                            }
                        )
        except (ValueError, OSError, RuntimeError) as error:
            if not status_published:
                row["status"] = (
                    inactive_task_status(control, task)
                    if retired
                    else "pending" if waiting_dispatch else "infra_error"
                )
            row.update(
                receiver_error=type(error).__name__,
                receiver_message="结果读取或 GitHub 发布未完成；稍后重试接收，不重跑构建。",
            )
            if publication_selected and (active or not obsolete_checked):
                receiver_errors += 1
            if requested and active and not status_published:
                publication_error(gh, control, task)
        rows.append(row)
    if requested_task and not requested_seen:
        try:
            finish_obsolete_summary(gh, control, requested_task)
        except (ValueError, OSError, RuntimeError):
            receiver_errors += 1
    dashboard.mkdir(parents=True, exist_ok=True)
    rows.extend(history_rows(results, rows))
    attach_full_flaggems(results, rows)
    (dashboard / "tasks.json").write_bytes(
        canonical(
            {
                "schema": "triton-anchor-dashboard",
                "generated_at": now(),
                "tasks": rows,
            }
        )
        + b"\n"
    )
    output("receiver_errors", receiver_errors)
    return published


def output(key: str, value: object) -> None:
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(
                f"{key}={str(value).lower() if isinstance(value, bool) else value}\n"
            )


def load_task(path: Path, expected_digest: str = "") -> dict:
    task = validate_task(json.loads(path.read_text()))
    if expected_digest and digest(task) != expected_digest:
        raise ValueError("Task artifact differs from the trusted prepare job output")
    return task


def security_diff(source: Path, base: str, tested: str) -> int:
    import importlib.util
    from dataclasses import asdict
    import sys

    spec = importlib.util.spec_from_file_location(
        "trusted_security", Path(__file__).with_name("scan_pr_security.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    command = [
        "git",
        "-C",
        str(source),
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
    ]
    names = (
        subprocess.check_output([*command, "--name-only", "-z", base, tested, "--"])
        .decode()
        .split("\0")
    )
    files = []
    for name in filter(None, names):
        patch = subprocess.check_output(
            [*command, "--unified=3", base, tested, "--", name]
        ).decode("utf-8", "replace")
        exists = (
            subprocess.run(
                ["git", "-C", str(source), "cat-file", "-e", f"{tested}:{name}"],
                capture_output=True,
            ).returncode
            == 0
        )
        files.append(
            {
                "filename": name,
                "status": "modified" if exists else "removed",
                "patch": None if "Binary files " in patch else patch,
            }
        )
    blocking, warnings = module.scan(files)
    module.print_findings(blocking + warnings)
    module.append_summary("block", blocking)
    module.append_summary("warn", warnings)
    Path("security-result.json").write_bytes(
        canonical(
            {
                "blocking": [asdict(x) for x in blocking],
                "warnings": [asdict(x) for x in warnings],
            }
        )
        + b"\n"
    )
    return int(bool(blocking))


def sarif_failures(root: Path) -> list[dict]:
    findings = []
    files = list(root.rglob("*.sarif"))
    if not files:
        raise ValueError("CodeQL produced no SARIF evidence")
    for path in files:
        document = json.loads(path.read_text())
        for run in document.get("runs", []):
            rules = {
                row["id"]: row
                for row in run.get("tool", {}).get("driver", {}).get("rules", [])
            }
            for result in run.get("results", []):
                rule = rules.get(result.get("ruleId"), {})
                severity = rule.get("properties", {}).get("security-severity")
                if (severity is not None and float(severity) >= 7) or result.get(
                    "level"
                ) == "error":
                    findings.append(result)
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "info",
            "card",
            "checks",
            "finalize",
            "approval",
            "enqueue",
            "cancel",
            "receive",
            "collect",
            "api",
            "security",
            "sarif",
        ),
    )
    parser.add_argument("--task", type=Path, default=Path("task.json"))
    parser.add_argument("--task-id", default=os.getenv("RECEIVER_TASK_ID", ""))
    parser.add_argument(
        "--reconcile-pending",
        action="store_true",
        default=os.getenv("RECONCILE_PENDING_RESULTS", "false") == "true",
    )
    parser.add_argument(
        "--round", type=int, default=int(os.getenv("RECEIVER_ROUND") or 1)
    )
    parser.add_argument(
        "--repository", default=os.getenv("GITHUB_REPOSITORY", REPOSITORY)
    )
    parser.add_argument("--worker-sha", default=os.getenv("WORKER_SHA", ""))
    parser.add_argument("--pr", type=int, default=int(os.getenv("PR_NUMBER") or 0))
    parser.add_argument("--branch", default=os.getenv("SOURCE_BRANCH", ""))
    parser.add_argument("--sha", default=os.getenv("REQUESTED_SHA", ""))
    parser.add_argument(
        "--full", action="store_true", default=os.getenv("FULL", "false") == "true"
    )
    parser.add_argument(
        "--event-kind",
        choices=("push", "manual"),
        default=os.getenv("EVENT_KIND", "push"),
    )
    parser.add_argument("--source", type=Path, default=Path("candidate"))
    parser.add_argument("--base", type=Path, default=Path("base"))
    parser.add_argument("--stages", default=os.getenv("STAGES", "{}"))
    parser.add_argument("--dashboard", type=Path, default=Path("_site/data"))
    args = parser.parse_args()
    if int(os.getenv("GITHUB_RUN_ATTEMPT") or 1) > 1 and args.command in {
        "prepare", "info", "card", "checks", "finalize", "approval", "enqueue", "api", "security",
    }:
        # Failed-only reruns may retain a successful Prepare job's old outputs.
        # The workflow redirects these attempts to a fresh, complete validation.
        parser.error("Validation reruns must start a fresh workflow; refusing the previous task artifact")
    gh = GitHub(args.repository)
    if args.command == "prepare":
        task = prepare_task(
            gh,
            args.worker_sha,
            args.pr,
            args.branch,
            args.sha,
            args.full,
            args.event_kind,
            verification_trigger_id(),
        )
        args.task.write_bytes(canonical(task) + b"\n")
        for key in ("task_id", "tested_sha", "head_sha", "base_sha", "external_fork"):
            output(key, task[key])
        output("task_digest", digest(task))
        begin_checks(gh, task)
        return 0
    if args.command == "sarif":
        failures = sarif_failures(args.source)
        print(f"CodeQL high/critical or error findings: {len(failures)}")
        return int(bool(failures))
    if args.command in {"info", "card", "checks", "finalize", "approval", "enqueue", "api", "security"}:
        task = load_task(args.task, os.getenv("EXPECTED_TASK_DIGEST", ""))
    if args.command == "checks":
        sync_preflight(gh, task, json.loads(args.stages))
        return 0
    if args.command == "finalize":
        finalize_preflight(gh, task, json.loads(args.stages))
        return 0
    if args.command == "approval":
        validate_approval_environment(gh)
        if not is_current(gh, task) or not gh.owns_task(task, workflow=True):
            raise ValueError("PR changed while waiting for approval")
        gh.check(task, "approve", "completed", "success", "Maintainer approval granted",
                 "The frozen PR revision was approved; dispatch can start.", workflow_url())
        return 0
    if args.command == "security":
        return security_diff(args.source, task["base_sha"], task["tested_sha"])
    if args.command == "info":
        if not is_current(gh, task) or not gh.owns_task(task, workflow=True):
            raise ValueError("PR changed before information publication")
        errors = validate_pr_info(task)
        if errors:
            gh.check(task, "basic", "completed", "failure", "PR information incomplete",
                     "Update the PR information before Basic CI can start.", workflow_url())
            gh.status(task, "failure", "Local CI: PR information incomplete; subsequent verification not run", workflow_url())
            gh.comment(task, pr_info_comment(errors))
        else:
            gh.check(task, "basic", "in_progress", None, f"{CHECK_NAMES['basic']}: running",
                     "Task information checked; Basic CI can now run.", workflow_url())
        return int(bool(errors))
    if args.command == "card":
        if not is_current(gh, task) or not gh.owns_task(task, workflow=True):
            if task["pr_number"]:
                gh.finish_inactive_pr(task["pr_number"])
            raise ValueError("PR changed before preflight publication")
        stages = json.loads(args.stages)
        eligible = preflight_passed(stages)
        if not eligible:
            # Defense in depth for manual CLI use or an older workflow caller.
            # Failed prerequisites get Checks/status feedback, never an approval card.
            publish_preflight_checks(gh, task, stages, False)
            gh.retire_open_checks(task)
            output("eligible", False)
            return 0
        if not task.get("external_fork"):
            # Trusted/internal tasks do not need an approval boundary.  Keep
            # the command idempotent for older/manual callers, but never emit
            # an approval comment, step summary, or approval Check Run.
            publish_preflight_checks(gh, task, stages, True)
            output("eligible", True)
            return 0
        approval_error = ""
        if eligible and task.get("external_fork"):
            try:
                validate_approval_environment(gh)
            except (ValueError, OSError) as error:
                eligible = False
                approval_error = (
                    str(error)
                    if isinstance(error, ValueError)
                    else "Cannot verify required reviewers on local-ci-fork-approval; check repository environment configuration."
                )
        publish_preflight_checks(gh, task, stages, eligible)
        context = gh.approval_context(task) if eligible else None
        body = approval_card(task, stages, eligible, approval_error, context)
        gh.comment(task, body)
        if os.getenv("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(body + "\n")
        if not eligible:
            description = "Local CI: approval configuration not ready; see review card"
        elif task.get("external_fork"):
            description = "Local CI: awaiting maintainer approval"
        else:
            description = "Local CI: trusted source; manual approval not required"
        waiting = eligible and task.get("external_fork")
        detail = description
        if workflow_url():
            detail += f"\n\n[Open approval controls and workflow evidence]({workflow_url()})"
        published = gh.check(task, "approve", "in_progress" if waiting else "completed",
                             None if waiting else "success" if eligible else "failure",
                             description, detail, workflow_url())
        if published and waiting:
            gh.status(task, "pending", description, workflow_url())
        output("eligible", eligible)
        return 0
    if args.command == "api":
        import importlib.util

        checker = (
            Path(__file__).resolve().parents[1] / "api_contract/check_public_api.py"
        )
        spec = importlib.util.spec_from_file_location("api_checker", checker)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base_scope = args.base / "api_contract/public_api.json"
        scope = (
            base_scope
            if base_scope.is_file()
            else Path(__file__).resolve().parents[2] / "api_contract/public_api.json"
        )
        candidate_scope = (
            args.source / "api_contract/public_api.json"
            if base_scope.is_file()
            else None
        )
        result = module.run_check(args.base, args.source, scope, candidate_scope)
        Path("api-result.json").write_bytes(canonical(result) + b"\n")
        Path("api-report.md").write_text(module._markdown(result))
        return int(result["status"] != "compatible")
    if args.command == "cancel" and args.pr:
        gh.finish_inactive_pr(args.pr)
    url = os.getenv("GITEE_RESULTS_REPO_URL", "")
    if not url.startswith("https://gitee.com/"):
        raise ValueError(
            "Configure GITEE_RESULTS_REPO_URL with the actual HTTPS Gitee repository"
        )
    if args.command == "receive":
        output("receiver_state", receive_result(gh, url, args.task_id, args.round))
        return 0
    control = GitStore(url, CONTROL_BRANCH)
    try:
        if args.command == "cancel":
            cancel_obsolete(gh, control, args.pr, args.branch)
        elif args.command == "enqueue":
            enqueue(task, gh, control, args.source)
        else:
            results = GitStore(url, RESULTS_BRANCH)
            try:
                if args.command == "collect":
                    if args.task_id:
                        if not ID.fullmatch(args.task_id):
                            raise ValueError("Receiver task ID must be an exact task identity")
                        received = validate_task(control.get(f"tasks/{args.task_id}.json"))
                        if received["task_id"] != args.task_id:
                            raise ValueError("Receiver task manifest identity differs")
                        cancel_obsolete(gh, control, received["pr_number"], received["target_branch"])
                    collect_results(
                        gh,
                        control,
                        results,
                        args.dashboard,
                        args.task_id,
                        reconcile_pending=args.reconcile_pending,
                    )
            finally:
                results.close()
    finally:
        control.close()
    return 0


if __name__ == "__main__":
    import sys

    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        safe_error = (
            str(error)
            if isinstance(error, (ValueError, GitHubAPIError))
            else "inspect the stage logs and transport configuration"
        )
        print(
            f"Local CI control failed: {type(error).__name__}: {safe_error}",
            file=sys.stderr,
        )
        # Errors before task.json exists still need a visible PR response.
        if (
            len(sys.argv) > 1
            and sys.argv[1] in {"prepare", "approval", "enqueue"}
            and os.getenv("GH_TOKEN")
        ):
            try:
                pr = int(os.getenv("PR_NUMBER") or 0)
                client = GitHub(os.getenv("GITHUB_REPOSITORY", REPOSITORY))
                if pr:
                    pull = client.request(f"pulls/{pr}")
                    expected = os.getenv("REQUESTED_SHA") or pull["head"]["sha"]
                    frozen = (
                        load_task(Path("task.json"))
                        if Path("task.json").is_file()
                        else None
                    )
                    if (
                        pull["state"] == "open"
                        and not pull["draft"]
                        and expected == pull["head"]["sha"]
                        and (
                            frozen is None
                            or (
                                is_current(client, frozen)
                                and client.owns_task(frozen, workflow=True)
                            )
                        )
                    ):
                        context = frozen or {
                            "head_sha": expected,
                            "pr_number": pr,
                        }
                        run_id = os.getenv("GITHUB_RUN_ID", "")
                        link = (
                            f"https://github.com/{client.repository}/actions/runs/{run_id}"
                            if run_id.isdigit()
                            else ""
                        )
                        client.comment(
                            context,
                            preflight_failure_comment(error, link, pull),
                        )
            except (ValueError, OSError, RuntimeError):
                print(
                    "PR failure notification could not be delivered; the workflow remains failed.",
                    file=sys.stderr,
                )
        raise SystemExit(1)
