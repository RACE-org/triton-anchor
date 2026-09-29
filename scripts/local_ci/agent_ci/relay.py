"""Git-based Gitee transport; production never requires server-to-GitHub access."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import urllib.parse
from pathlib import Path

from .protocol import (
    PREINSTALLED_SUBMODULES,
    ContractError,
    llvm_hash_from_files,
    TRITON_VERSION_PATH,
    triton_version_from_source,
    current_key,
    full_flaggems_result_path,
    result_task_prefix,
    result_task_prefixes,
    within,
    validate_result,
)
from .delivery import (
    MAX_FILE_BYTES,
    MAX_FULL_FLAGGEMS_BYTES,
    MAX_TOTAL_BYTES,
    MAX_RESULT_BYTES,
    validate_full_flaggems,
)


class GitRelay:
    def __init__(
        self,
        url: str,
        root: Path,
        *,
        allow_local: bool = False,
        control_branch: str = "local-ci-control",
        results_branch: str = "local-ci-results",
    ):
        parsed = urllib.parse.urlsplit(url)
        local_path = Path(url).exists()
        if parsed.scheme == "https":
            if parsed.hostname != "gitee.com" or parsed.username or parsed.password:
                raise ContractError(
                    "Production relay must be a credential-free HTTPS Gitee URL"
                )
        elif not allow_local or (not local_path and parsed.scheme not in {"", "file"}):
            raise ContractError(
                "Local relay transports are only allowed in explicit simulations"
            )
        self.url, self.root = url, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.control_branch, self.results_branch = control_branch, results_branch
        self.control_snapshot = None
        self.cache = self.root / "cache"
        self.lock = threading.RLock()
        # Writes use their own temporary checkout; a slow push must not lock the
        # control snapshot used by cancellation polling and task intake.
        self.write_lock = threading.RLock()
        self.env = {
            **os.environ,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        if os.getenv("GITEE_TOKEN"):
            askpass = self.root / "askpass.py"
            askpass.write_text(
                "#!/usr/bin/env python3\nimport os,sys\nprint(os.environ.get('GITEE_USERNAME','oauth2') if 'Username' in sys.argv[1] else os.environ['GITEE_TOKEN'])\n"
            )
            askpass.chmod(0o700)
            self.env["GIT_ASKPASS"] = str(askpass)
        if not (self.cache / ".git").exists():
            self.cache.mkdir(exist_ok=True)
            self.git(["init", "-q"], cwd=self.cache)
            self.git(["remote", "add", "origin", url], cwd=self.cache)

    def git(
        self, args: list[str], *, cwd: Path | None = None, check: bool = True
    ) -> subprocess.CompletedProcess:
        directory = (cwd or self.cache).resolve()
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={directory}",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                *args,
            ],
            cwd=directory,
            env=self.env,
            capture_output=True,
            timeout=120,
        )
        if check and result.returncode:
            detail = result.stderr.decode(errors="replace")[-2000:]
            for key, value in self.env.items():
                if (
                    any(
                        part in key.upper()
                        for part in ("TOKEN", "PASSWORD", "SECRET", "API_KEY")
                    )
                    and len(value) > 3
                ):
                    detail = detail.replace(value, "[redacted]")
            raise RuntimeError(
                f"Relay git {args[0]} failed with exit {result.returncode}: {detail.strip()}"
            )
        return result

    def refresh(self) -> None:
        with self.lock:
            self.git(
                ["fetch", "--prune", "origin", "+refs/heads/*:refs/remotes/origin/*"]
            )
            found = self.git(
                ["rev-parse", "--verify", f"refs/remotes/origin/{self.control_branch}"],
                check=False,
            )
            self.control_snapshot = (
                found.stdout.decode().strip() if found.returncode == 0 else None
            )

    def ref_sha(self, ref: str) -> str:
        return (
            self.git(["rev-parse", "--verify", f"refs/remotes/origin/{ref}^{{commit}}"])
            .stdout.decode()
            .strip()
        )

    def read(self, branch: str, path: str) -> bytes | None:
        within(self.root, path)
        with self.lock:
            revision = (
                self.control_snapshot
                if branch == self.control_branch and self.control_snapshot
                else f"refs/remotes/origin/{branch}"
            )
            result = self.git(["show", f"{revision}:{path}"], check=False)
        return result.stdout if result.returncode == 0 else None

    def read_json(self, branch: str, path: str) -> dict | None:
        raw = self.read(branch, path)
        return json.loads(raw) if raw is not None else None

    def tasks(self) -> list[dict]:
        # Manifest/current/cancel are read from one fetched control commit.
        with self.lock:
            return self._tasks()

    def _tasks(self) -> list[dict]:
        revision = self.control_snapshot or f"refs/remotes/origin/{self.control_branch}"
        result = self.git(
            ["ls-tree", "-r", "--name-only", revision, "current/"], check=False
        )
        if result.returncode:
            return []
        documents = []
        for name in result.stdout.decode().splitlines():
            current = self.read_json(self.control_branch, name)
            if current and current.get("task_id"):
                task = self.read_json(
                    self.control_branch, f"tasks/{current['task_id']}.json"
                )
                if task:
                    documents.append(task)
        return documents

    def validity(self, task: dict) -> tuple[bool, str]:
        cancel = self.read_json(self.control_branch, f"cancel/{task['task_id']}.json")
        if cancel and cancel.get("task_id") == task["task_id"]:
            return False, cancel.get("reason", "Task cancelled")
        current = self.read_json(
            self.control_branch, f"current/{current_key(task)}.json"
        )
        if not current or current.get("task_id") != task["task_id"]:
            return False, "Task superseded or no longer current"
        for field in ("task_ref", "base_task_ref", "head_task_ref"):
            sha_field = {
                "task_ref": "tested_sha",
                "base_task_ref": "base_sha",
                "head_task_ref": "head_sha",
            }[field]
            try:
                actual = self.ref_sha(task[field])
            except RuntimeError:
                return False, f"Task snapshot incomplete: {field}"
            if actual != task[sha_field]:
                return False, f"Task snapshot changed: {field}"
        if task["event_kind"] == "pull_request":
            parents = (
                self.git(["rev-list", "--parents", "-n", "1", task["tested_sha"]])
                .stdout.decode()
                .split()[1:]
            )
            if parents != [task["base_sha"], task["head_sha"]]:
                return False, "Tested merge parents do not match the frozen base/head"
        for variant, sha in (
            ("candidate", task["tested_sha"]),
            ("base", task["base_sha"]),
        ):
            tree = self.git(["ls-tree", "-r", sha]).stdout.decode().splitlines()
            links = {
                row.split("\t", 1)[1]: row.split()[2]
                for row in tree
                if row.startswith("160000 ")
                and row.split("\t", 1)[1] not in PREINSTALLED_SUBMODULES
            }
            modules = {
                row["path"]: row
                for row in task.get("submodules", [])
                if row["variant"] == variant
                and row["path"] not in PREINSTALLED_SUBMODULES
            }
            if set(links) != set(modules):
                return False, "Submodule manifest does not cover the frozen gitlinks"
            for path, module in modules.items():
                if (
                    module["sha"] != links[path]
                    or self.ref_sha(module["task_ref"]) != module["sha"]
                ):
                    return False, "Pinned Gitee submodule snapshot changed"
        try:
            self.source_variants(task)
        except (ValueError, RuntimeError) as exc:
            return False, f"Invalid source environment metadata: {exc}"
        return True, "current"

    def source_variants(self, task: dict) -> dict:
        """Verify both frozen sources, including base metadata absent in old tasks."""
        variants = {}
        for variant, sha in (("base", task["base_sha"]), ("candidate", task["tested_sha"])):
            try:
                paths = self.git(
                    ["ls-tree", "-r", "--name-only", "-z", sha, "--", "triton/cmake/"]
                ).stdout.decode().split("\0")
                source = {
                    "source_sha": sha,
                    "llvm_hash": llvm_hash_from_files(
                        paths, lambda path: self.git(["show", f"{sha}:{path}"]).stdout
                    ),
                    "triton_version": triton_version_from_source(
                        self.git(["show", f"{sha}:{TRITON_VERSION_PATH}"]).stdout
                    ),
                }
            except (ValueError, RuntimeError) as exc:
                raise ContractError(f"Cannot read {variant} LLVM/Triton metadata: {exc}") from exc
            if "variants" in task and source != task["variants"].get(variant):
                raise ContractError(f"Task {variant} environment identity does not match frozen source")
            variants[variant] = source
        if variants["candidate"]["llvm_hash"] != task["llvm_hash"]:
            raise ContractError("Task LLVM identity does not match tested source")
        return variants

    def checkout(self, sha: str, destination: Path) -> None:
        if destination.exists():
            actual = (
                self.git(["rev-parse", "HEAD"], cwd=destination).stdout.decode().strip()
            )
            dirty = self.git(
                ["status", "--porcelain", "--untracked-files=no"], cwd=destination
            ).stdout
            if actual != sha or dirty:
                raise ContractError(
                    "Existing task checkout does not match the frozen revision"
                )
            return
        self.git(
            [
                "clone",
                "--quiet",
                "--no-hardlinks",
                "--no-checkout",
                str(self.cache),
                str(destination),
            ],
            cwd=self.root,
        )
        self.git(["checkout", "--quiet", "--detach", sha], cwd=destination)

    def checkout_submodules(self, task: dict, sha: str, destination: Path) -> None:
        """Populate gitlinks from already fetched Gitee refs, ignoring candidate URLs."""
        variant = "candidate" if sha == task["tested_sha"] else "base"
        for module in task.get("submodules", []):
            if (
                module["variant"] != variant
                or module["path"] in PREINSTALLED_SUBMODULES
            ):
                continue
            location = within(destination, module["path"])
            if location.exists() and not any(location.iterdir()):
                location.rmdir()
            self.checkout(module["sha"], location)
            # Nested dependencies also need an explicit mirror manifest.
            nested = self.git(["ls-tree", "-r", module["sha"]]).stdout
            if any(row.startswith(b"160000 ") for row in nested.splitlines()):
                raise ContractError(
                    "Nested submodule requires an explicit mirrored source manifest"
                )

    def write(
        self, branch: str, files: dict[str, bytes], *, immutable: bool = False,
        message: str = "local-ci: update relay records",
    ) -> None:
        for relative in files:
            within(self.root, relative)
        with self.write_lock:
            last_error = None
            for _ in range(3):
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="publish-", dir=self.root
                    ) as temporary:
                        work = Path(temporary)
                        self.git(["init", "-q"], cwd=work)
                        self.git(["remote", "add", "origin", self.url], cwd=work)
                        found = self.git(
                            ["ls-remote", "--heads", "origin", f"refs/heads/{branch}"],
                            cwd=work,
                        ).stdout.strip()
                        if found:
                            self.git(
                                [
                                    "fetch",
                                    "--quiet",
                                    "--depth=1",
                                    "origin",
                                    f"refs/heads/{branch}",
                                ],
                                cwd=work,
                            )
                            self.git(
                                ["checkout", "--quiet", "-B", branch, "FETCH_HEAD"],
                                cwd=work,
                            )
                        else:
                            self.git(
                                ["checkout", "--quiet", "--orphan", branch], cwd=work
                            )
                        pending = files
                        for relative, content in pending.items():
                            path = within(work, relative)
                            if (
                                immutable
                                and path.exists()
                                and path.read_bytes() != content
                            ):
                                raise ContractError(
                                    f"Immutable relay artifact changed: {relative}"
                                )
                            path.parent.mkdir(parents=True, exist_ok=True)
                            path.write_bytes(content)
                        self.git(["add", "--", *pending.keys()], cwd=work)
                        if (
                            self.git(
                                ["diff", "--cached", "--quiet"], cwd=work, check=False
                            ).returncode
                            == 0
                        ):
                            return
                        self.git(
                            [
                                "-c",
                                "user.name=local-ci",
                                "-c",
                                "user.email=local-ci@example.invalid",
                                "commit",
                                "--quiet",
                                "-m",
                                message,
                            ],
                            cwd=work,
                        )
                        self.git(
                            ["push", "--quiet", "origin", f"HEAD:refs/heads/{branch}"],
                            cwd=work,
                        )
                        return
                except RuntimeError as exc:
                    last_error = exc
            raise RuntimeError(
                "Relay publish failed after three attempts"
            ) from last_error

    def publish_result(self, task: dict, run_id: str, directory: Path) -> str:
        """Publish one sealed result and its selected files in the same Git commit."""
        directory = Path(directory)
        raw = (directory / "result.json").read_bytes()
        if len(raw) > MAX_RESULT_BYTES:
            raise ContractError("Sealed result exceeds the Git document budget")
        result = validate_result(json.loads(raw), task)
        if result["run_id"] != run_id:
            raise ContractError("Sealed result run differs from publication request")
        prefix = f"{result_task_prefix(task)}/{run_id}"
        # An old sealed outbox must retry its original destination, including when
        # the previous push succeeded but its response was lost before an upgrade.
        for legacy in result_task_prefixes(task)[1:]:
            old_run = Path(legacy) / run_id
            if directory.parent.parts[-len(old_run.parts):] == old_run.parts:
                prefix = old_run.as_posix()
                break
        files = {f"{prefix}/result.json": raw}
        business = directory / "business" / "flaggems-summary.json"
        declarations = [
            row for row in result["checks"] if "business_result" in row
        ]
        if declarations and (
            len(declarations) != 1
            or declarations[0].get("tool_id") != "flaggems"
        ):
            raise ContractError("Only the full FlagGems check may declare a business result")
        if declarations and not business.is_file():
            raise ContractError("Sealed full FlagGems business result is missing")
        if business.is_file():
            destination = full_flaggems_result_path(task, run_id)
            check = declarations[0] if declarations else {}
            if (
                not task.get("full")
                or (check.get("parameters") or {}).get("mode") != "full"
                or check.get("business_result") != destination
            ):
                raise ContractError("Full FlagGems business result has no matching sealed check")
            content = business.read_bytes()
            if len(content) > MAX_FULL_FLAGGEMS_BYTES:
                raise ContractError("Full FlagGems report exceeds the business-result budget")
            try:
                validate_full_flaggems(json.loads(content))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError("Full FlagGems report is not valid JSON") from exc
            files[destination] = content
        total = 0
        for artifact in result["artifacts"]:
            if artifact.get("omitted"):
                continue
            path = within(directory / "artifacts", artifact["path"], must_exist=True)
            content = path.read_bytes()
            total += len(content)
            if len(content) != artifact.get("size") or len(content) > MAX_FILE_BYTES:
                raise ContractError("Selected artifact differs from sealed result")
            if total > MAX_TOTAL_BYTES:
                raise ContractError("Selected artifacts exceed the Git budget")
            files[f"{prefix}/artifacts/{artifact['path']}"] = content
        # Retrying after a lost push response finds identical files and makes no commit.
        self.write(
            self.results_branch, files, immutable=True,
            message=f"local-ci: {result['status']} {task['head_sha'][:12]} {run_id}",
        )
        return hashlib.sha256(raw).hexdigest()
