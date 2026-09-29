from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "gateway", ROOT / "scripts/ci/gateway.py"
)
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)


def git(root, *args):
    return (
        subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)
        .decode()
        .strip()
    )


def body(
    summary="Correct documented behavior",
    scope="README",
    validation="Reviewed the actual implementation",
):
    fields = {"summary": summary, "scope": scope, "validation": validation}
    return "\n".join(
        "<!-- field:" + key + " -->\n" + value for key, value in fields.items()
    )


def github_check(payload, check_id):
    # GITHUB_TOKEN-created checks do not retain the requested details_url.
    return {**payload, "id": check_id, "app": {"slug": "github-actions"},
            "details_url": f"https://github.com/{g.REPOSITORY}/runs/{check_id}"}


class FakeGitHub:
    repository = g.REPOSITORY

    def __init__(self, base, head, tested):
        self.base, self.head, self.tested = base, head, tested
        self.pull = {
            "state": "open",
            "draft": False,
            "user": {"login": "contributor"},
            "title": "Document the public behavior",
            "body": body(),
            "labels": [{"name": "docs"}],
            "head": {
                "sha": head,
                "ref": "docs-topic",
                "repo": {"full_name": "anteloper-c/triton-anchor"},
            },
            "base": {"ref": "main", "sha": base},
            "mergeable": True,
            "merge_commit_sha": tested,
        }
        self.statuses, self.comments = [], []
        self.latest_statuses, self.writes = {}, []
        self.cmake_files = {"triton/cmake/llvm-hash.txt": b"a" * 40 + b"\n"}
        self.source_files = {}
        self.content_reads = []
        self.environment = {
            "protection_rules": [
                {
                    "type": "required_reviewers",
                    "reviewers": [
                        {"type": "User", "reviewer": {"login": "maintainer"}}
                    ],
                }
            ]
        }
        self.approvals = []

    def request(self, path, method="GET", data=None):
        if path.startswith("contents/triton/cmake?ref="):
            ref = path.split("ref=", 1)[1]
            files = self.source_files.get(ref, self.cmake_files)
            return [{"type": "file", "path": name} for name in files if name.startswith("triton/cmake/")]
        if path == "environments/local-ci-fork-approval":
            return self.environment
        if path == "actions/runs/12345/approvals":
            return copy.deepcopy(self.approvals)
        if path == "pulls/7":
            return copy.deepcopy(self.pull)
        if path == "git/commits/" + self.tested:
            return {"parents": [{"sha": self.base}, {"sha": self.head}]}
        if path == "git/commits/" + self.head:
            return {"parents": [{"sha": self.base}]}
        if path == "branches/main":
            return {"commit": {"sha": self.head}}
        raise AssertionError(path)

    optional = request

    def gitlinks(self, ref):
        return []

    def content(self, path, ref):
        assert ref in {self.base, self.head, self.tested}
        self.content_reads.append(path)
        if path == g.TRITON_VERSION_PATH:
            return self.source_files.get(ref, {}).get(path, b"__version__ = '3.0.0'\n")
        return self.source_files.get(ref, self.cmake_files)[path]

    def status(self, task, state, description, url="", **kwargs):
        self.statuses.append((task["task_id"], state))
        self.latest_statuses[task["task_id"]] = (state, description)
        self.writes.append("status")

    def status_matches(self, task, state, description):
        return self.latest_statuses.get(task["task_id"]) == (state, description)

    def check(self, *args, **_kwargs):
        if not hasattr(self, "check_calls"):
            self.check_calls = []
        self.check_calls.append(args)
        return False

    def owns_task(self, task, **kwargs):
        return True

    def task_start(self, task):
        return {}

    def retire_open_checks(self, task, **kwargs):
        pass

    def latest_dispatch(self, task):
        return {}

    def stage_status(self, task, key):
        return {}

    def latest_summary(self, task):
        return None

    def restore_preflight(self, task):
        pass

    def approval_context(self, task):
        return {}

    def comment(self, task, content, **kwargs):
        if self.comments and self.comments[-1] == content:
            return False
        self.comments.append(content)
        self.writes.append("comment")
        return True


class RecordingGitHub(g.GitHub):
    """Exercise real publication/ownership logic against in-memory API records."""

    def __init__(self, fixture):
        super().__init__(g.REPOSITORY, token="fixture")
        self.fixture = fixture
        self.status_rows = {fixture.head: [], fixture.tested: []}
        self.check_rows = {fixture.head: [], fixture.tested: []}
        self.writes = []
        self.fail_next_status = False
        self.serial = 0

    def seed_status(self, sha, context, state, task_id, run_id="12345", attempt="1", **fields):
        self.serial += 1
        row = {
            "id": self.serial, "context": context, "state": state,
            "description": state,
            "creator": {"login": "github-actions[bot]"},
            "target_url": f"https://github.com/{g.REPOSITORY}/actions/runs/{run_id}"
                          f"#local-ci-task={task_id}&local-ci-workflow={run_id}&local-ci-attempt={attempt}",
            **fields,
        }
        self.status_rows.setdefault(sha, []).insert(0, row)
        return row

    def request(self, path, method="GET", data=None):
        if method == "GET":
            if "/statuses?" in path:
                return copy.deepcopy(self.status_rows.get(path.split("/")[1], []))
            if "/check-runs?" in path:
                rows = self.check_rows.get(path.split("/")[1], [])
                if "check_name=" in path:
                    rows = [row for row in rows
                            if f"check_name={g.quote(row['name'], safe='')}&" in path]
                return {"check_runs": copy.deepcopy(rows)}
            return self.fixture.request(path)
        if path.startswith("statuses/"):
            if self.fail_next_status:
                self.fail_next_status = False
                raise RuntimeError("Status publication temporarily unavailable")
            self.writes.append((path, method, copy.deepcopy(data)))
            return self.seed_status(path.split("/")[1], data["context"], data["state"], "", **{
                key: value for key, value in data.items() if key not in {"context", "state"}
            })
        if method == "PATCH" and path.startswith("check-runs/"):
            self.writes.append((path, method, copy.deepcopy(data)))
            for rows in self.check_rows.values():
                for row in rows:
                    if row["id"] == int(path.rsplit("/", 1)[1]):
                        row.update(data)
                        return row
        raise AssertionError((path, method))


class GatewayBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        git(self.source, "init", "-q")
        git(self.source, "config", "user.name", "Test")
        git(self.source, "config", "user.email", "test@example.invalid")
        version = self.source / "triton/python/triton/__init__.py"
        version.parent.mkdir(parents=True)
        version.write_text("__version__ = '3.0.0'\n")
        llvm = self.source / "triton/cmake/llvm-hash.txt"
        llvm.parent.mkdir(parents=True)
        llvm.write_text("a" * 40 + "\n")
        (self.source / "README.md").write_text("old\n")
        git(self.source, "add", ".")
        git(self.source, "commit", "-qm", "base")
        self.base = git(self.source, "rev-parse", "HEAD")
        (self.source / "README.md").write_text("new\n")
        git(self.source, "add", ".")
        git(self.source, "commit", "-qm", "head")
        self.head = git(self.source, "rev-parse", "HEAD")
        tree = git(self.source, "rev-parse", "HEAD^{tree}")
        self.tested = (
            subprocess.check_output(
                ["git", "commit-tree", tree, "-p", self.base, "-p", self.head],
                cwd=self.source,
                input=b"merge\n",
            )
            .decode()
            .strip()
        )
        git(self.source, "checkout", "--detach", self.tested)
        self.remote = self.root / "gitee.git"
        git(self.root, "init", "--bare", "-q", str(self.remote))
        self.gh = FakeGitHub(self.base, self.head, self.tested)
        self.task = g.prepare_task(self.gh, self.base, 7)
        receiver_env = patch.dict(g.os.environ, {"RECEIVER_TASK_ID": self.task["task_id"]})
        receiver_env.start()
        self.addCleanup(receiver_env.stop)
        self.stores = []

    def tearDown(self):
        for store in self.stores:
            store.close()
        self.tmp.cleanup()

    def store(self, branch):
        result = g.GitStore(str(self.remote), branch)
        self.stores.append(result)
        return result

    def publish_result(self, result):
        control = self.store(g.CONTROL_BRANCH)
        g.enqueue(self.task, self.gh, control, self.source)
        results = self.store(g.RESULTS_BRANCH)
        results.put({f"runs/{self.task['task_id']}/{result['run_id']}/result.json": result})
        return control, results

    def result(self):
        return {
            "schema": g.RESULT_SCHEMA,
            "task": self.task,
            "run_id": "20260907T120000Z-1",
            "status": "pass",
            "summary": "Documentation and architecture review completed.",
            "checks": [{
                "tool_id": "control_plane", "status": "pass",
                "summary": "Validated", "evidence": [],
            }],
            "reviews": [
                {"kind": "pr_info", "status": "pass", "summary": "Clear", "evidence": []},
                {
                "kind": "architecture", "status": "pass",
                "summary": "Compatible", "evidence": ["README.md"],
            },
            ],
            "findings": [],
            "blocking_reasons": [],
            "environment": {"profile": "local"},
            "policy": {"required_checks": ["control_plane"]},
            "artifacts": [],
            "completed_at": "2026-09-07T12:00:00Z",
        }

    def test_exact_identity_metadata_and_worker_revision(self):
        self.assertEqual(g.validate_task(self.task), self.task)
        self.assertTrue(self.task["external_fork"])
        for field, value in (
            ("title", "different"),
            ("full", True),
        ):
            changed = {**self.task, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                g.validate_task(changed)
        self.assertEqual(self.task["control_policy"], "worker")
        changed = {**self.task, "worker_revision_sha": "f" * 40}
        self.assertEqual(g.compute_task_id(changed), self.task["task_id"])
        g.validate_task(changed)
        for event_kind in ("push", "manual"):
            branch_task = g.prepare_task(
                self.gh, self.base, branch="main", event_kind=event_kind
            )
            self.assertEqual(branch_task["control_policy"], "worker")
            self.assertEqual(
                g.compute_task_id(
                    {**branch_task, "worker_revision_sha": "f" * 40}
                ),
                branch_task["task_id"],
            )
        legacy = dict(self.task)
        legacy.pop("control_policy")
        self.assertNotEqual(g.compute_task_id(legacy),
                            g.compute_task_id({**legacy, "worker_revision_sha": "f" * 40}))
        reopened = g.prepare_task(self.gh, self.base, 7, trigger_id="100:1")
        duplicate = g.prepare_task(self.gh, self.base, 7, trigger_id="100:1")
        rerun = g.prepare_task(self.gh, self.base, 7, trigger_id="100:2")
        self.assertEqual(reopened["task_id"], duplicate["task_id"])
        self.assertEqual(len({self.task["task_id"], reopened["task_id"], rerun["task_id"]}), 3)
        self.assertNotEqual(reopened["task_ref"], self.task["task_ref"])
        self.gh.pull["head"]["sha"] = "e" * 40
        with self.assertRaises(ValueError):
            g.prepare_task(self.gh, self.base, 7, requested_sha=self.head)
        task_file = self.root / "task.json"
        changed = {**self.task, "llvm_hash": "e" * 40}
        task_file.write_bytes(g.canonical(changed))
        with self.assertRaises(ValueError):
            g.load_task(task_file, g.digest(self.task))

    def test_verification_entry_distinguishes_routing_and_rejects_old_rerun_artifacts(self):
        env = {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_RUN_ID": "200",
               "GITHUB_RUN_ATTEMPT": "1"}
        for routed, expected in (
            ({}, "200:1"),  # Direct Run workflow.
            ({"LOCAL_CI_ACTION": "push"}, ""),
            ({"LOCAL_CI_ACTION": "opened", "LOCAL_CI_REQUEST_ID": "100:1"}, ""),
            ({"LOCAL_CI_ACTION": "opened", "LOCAL_CI_REQUEST_ID": "100:2"}, "100:2"),
            ({"LOCAL_CI_ACTION": "reopened", "LOCAL_CI_REQUEST_ID": "100:1"}, "100:1"),
            ({"LOCAL_CI_ACTION": "manual", "LOCAL_CI_REQUEST_ID": "100:1"}, "100:1"),
            ({"LOCAL_CI_ACTION": "opened", "LOCAL_CI_TRIGGER_ID": "100:2"}, "100:2"),
        ):
            with self.subTest(routed=routed), patch.dict(g.os.environ, {**env, **routed}, clear=True):
                self.assertEqual(g.verification_trigger_id(), expected)
        # Failed-only retries must not load the old successful Prepare artifact,
        # even if a dependent job is selected directly for rerun.
        for command in ("prepare", "approval", "enqueue", "checks"):
            with (
                self.subTest(command=command),
                patch.dict(g.os.environ, {**env, "GITHUB_RUN_ATTEMPT": "2"}, clear=True),
                patch("sys.argv", ["gateway.py", command]),
                patch("sys.stderr", new_callable=io.StringIO),
                patch.object(g, "GitHub") as client,
                patch.object(g, "load_task") as load,
                self.assertRaises(SystemExit) as error,
            ):
                g.main()
            self.assertEqual(error.exception.code, 2)
            client.assert_not_called()
            load.assert_not_called()

    def test_prepare_reads_llvm_metadata_without_reading_unrelated_files(self):
        llvm_json = json.dumps({"llvm_hash": "a" * 40, "build_number": 123}).encode()
        for files in (
            {"llvm-info.json": llvm_json},
            {"llvm-info": llvm_json},
            {"llvm-info.txt": b"a" * 40 + b"\n"},
            {"llvm-hash.txt": b"a" * 40, "llvm-info.json": llvm_json},
        ):
            with self.subTest(files=tuple(files)):
                self.gh.cmake_files = {"triton/cmake/" + name: value for name, value in files.items()}
                expected_reads = set(self.gh.cmake_files) | {g.TRITON_VERSION_PATH}
                self.gh.cmake_files.update({
                    "triton/cmake/amd-llvm-info.json": b"invalid",
                    "triton/cmake/llvm-build-info.json": b"invalid",
                    "vendor/llvm-info.json": b"invalid",
                })
                self.gh.content_reads.clear()
                task = g.prepare_task(self.gh, self.base, 7)
                self.assertEqual(task["llvm_hash"], "a" * 40)
                self.assertEqual(set(self.gh.content_reads), expected_reads)

    def test_prepare_rejects_missing_invalid_or_conflicting_llvm_metadata(self):
        for files in (
            {},
            {"llvm-info.json": b'{"llvm_hash": "not-a-sha"}'},
            {"llvm-info.json": b'{"version": "3.3.0"}'},
            {"llvm-info": b"revision: " + b"a" * 40},
            {"llvm-hash.txt": b"a" * 40, "llvm-info.json": json.dumps({"llvm_hash": "b" * 40}).encode()},
        ):
            with self.subTest(files=tuple(files)):
                self.gh.cmake_files = {"triton/cmake/" + name: value for name, value in files.items()}
                with self.assertRaises(ValueError):
                    g.prepare_task(self.gh, self.base, 7)

    def test_prepare_freezes_and_validates_each_source_environment(self):
        self.gh.source_files[self.tested] = {
            "triton/cmake/llvm-info.json": json.dumps({"llvm_hash": "b" * 40}).encode(),
            g.TRITON_VERSION_PATH: b"__version__ = '3.3.0'\n",
        }
        task = g.prepare_task(self.gh, self.base, 7)
        self.assertEqual(task["variants"], {
            "base": {"source_sha": self.base, "llvm_hash": "a" * 40, "triton_version": "3.0.0"},
            "candidate": {"source_sha": self.tested, "llvm_hash": "b" * 40, "triton_version": "3.3.0"},
        })
        legacy = {key: value for key, value in task.items() if key != "variants"}
        self.assertEqual(g.compute_task_id(legacy), task["task_id"])
        g.validate_task(legacy)
        for variant, key, value in (
            ("base", "source_sha", self.head),
            ("base", "llvm_hash", "invalid"),
            ("candidate", "triton_version", "3.3"),
            ("candidate", "llvm_hash", "c" * 40),
        ):
            changed = copy.deepcopy(task)
            changed["variants"][variant][key] = value
            with self.subTest(variant=variant, key=key), self.assertRaises(ValueError):
                g.validate_task(changed)

    def test_version_metadata_is_static_and_unambiguous(self):
        self.assertEqual(g.triton_version_from_source(b"__version__: str = '3.5.1'\n"), "3.5.1")
        self.assertEqual(g.triton_version_from_source(b"__version__ = '3.8.0.dev20260101'\n"), "3.8.0.dev20260101")
        for source in (
            b"__version__ = discover_version()\n",
            b"__version__ = '3.0.0'\n__version__ = '3.1.0'\n",
            b"# __version__ = '3.0.0'\n",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                g.triton_version_from_source(source)

    def test_relay_reads_legacy_base_and_verifies_new_variant_metadata(self):
        from agent_ci.relay import GitRelay

        (self.source / "triton/cmake/llvm-hash.txt").write_text("b" * 40 + "\n")
        (self.source / g.TRITON_VERSION_PATH).write_text("__version__ = '3.3.0'\n")
        git(self.source, "commit", "-qam", "upgrade compiler")
        tested = git(self.source, "rev-parse", "HEAD")
        task = {key: value for key, value in self.task.items() if key != "variants"}
        task.update(tested_sha=tested, llvm_hash="b" * 40)
        original = copy.deepcopy(task)
        relay = GitRelay(str(self.source), self.root / "relay", allow_local=True)
        relay.git(["fetch", "origin", self.base, tested])
        variants = relay.source_variants(task)
        self.assertEqual(variants["base"], {
            "source_sha": self.base, "llvm_hash": "a" * 40, "triton_version": "3.0.0",
        })
        self.assertEqual(variants["candidate"], {
            "source_sha": tested, "llvm_hash": "b" * 40, "triton_version": "3.3.0",
        })
        self.assertEqual(task, original)
        self.assertEqual(relay.source_variants({**task, "variants": variants}), variants)
        for field, value in (("llvm_hash", "c" * 40), ("triton_version", "3.1.0")):
            changed = copy.deepcopy(variants)
            changed["base"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "base environment identity"):
                relay.source_variants({**task, "variants": changed})

    def test_prepare_uses_documented_pr_merge_result_and_checks_its_identity(self):
        self.assertEqual(self.task["tested_sha"], self.gh.pull["merge_commit_sha"])
        for branch in ("main", "CI_dev_forPR", "release/next"):
            with self.subTest(branch=branch):
                self.gh.pull["base"]["ref"] = branch
                task = g.prepare_task(self.gh, self.base, 7)
                self.assertEqual(task["target_branch"], branch)
        self.gh.pull["base"]["ref"] = "main"
        self.gh.pull["mergeable"] = False
        with self.assertRaisesRegex(ValueError, "cannot be merged cleanly"):
            g.prepare_task(self.gh, self.base, 7)

        self.gh.pull["mergeable"] = True
        self.gh.pull["merge_commit_sha"] = None
        with self.assertRaisesRegex(ValueError, "merge result is not ready"):
            g.prepare_task(self.gh, self.base, 7)
        self.gh.pull["merge_commit_sha"] = self.tested
        self.gh.pull["base"]["sha"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "Merge parents do not match"):
            g.prepare_task(self.gh, self.base, 7)

    def test_preinstalled_flaggems_and_other_submodule_mirrors(self):
        links = [{"path": "FlagGems", "sha": "c" * 40}]
        with patch.object(self.gh, "gitlinks", return_value=links):
            with patch.dict(g.os.environ, {"GITEE_SUBMODULE_MIRRORS": "{}"}):
                task = g.prepare_task(self.gh, self.base, 7)
                self.assertEqual(task["submodules"], [])
            for path in ("OtherDependency", "vendor/FlagGems"):
                with self.subTest(path=path):
                    links[:] = [
                        {"path": "FlagGems", "sha": "c" * 40},
                        {"path": path, "sha": "d" * 40},
                    ]
                    with patch.dict(
                        g.os.environ, {"GITEE_SUBMODULE_MIRRORS": "{}"}
                    ):
                        with self.assertRaisesRegex(
                            ValueError, "Gitee submodule mirror"
                        ):
                            g.prepare_task(self.gh, self.base, 7)
                    with patch.dict(
                        g.os.environ,
                        {
                            "GITEE_SUBMODULE_MIRRORS": json.dumps(
                                {path: "https://gitee.com/test/dependency.git"}
                            )
                        },
                    ):
                        task = g.prepare_task(self.gh, self.base, 7)
                    self.assertEqual(
                        {row["path"] for row in task["submodules"]}, {path}
                    )
                    self.assertEqual(
                        {row["variant"] for row in task["submodules"]},
                        {"candidate", "base"},
                    )
                    self.assertTrue(
                        all(
                            task["task_id"] in row["task_ref"]
                            for row in task["submodules"]
                        )
                    )

    def test_three_required_pr_sections_without_type_specific_fields(self):
        self.assertEqual(g.validate_pr_info(self.task), [])
        for headings in (("变更概述", "影响范围", "验证情况"), ("Summary", "Scope", "Validation"),
                         ("变更概述 / Summary", "影响范围 / Scope", "验证情况 / Validation")):
            description = "\n".join(f"### {heading}\n{value}" for heading, value in
                                    zip(headings, ("Clarify behavior", "Docs only", "Reviewed diff")))
            description += "\n### 自定义字段 / Custom notes\nRollout details\n"
            with self.subTest(headings=headings):
                self.assertEqual(g.validate_pr_info({**self.task, "description": description, "labels": []}), [])
                self.assertEqual(g.pr_fields(description)["validation"], "Reviewed diff")
        for field in g.FIELD_NAMES:
            missing = body().replace(f"<!-- field:{field} -->", "<!-- field:unused -->")
            with self.subTest(field=field):
                self.assertTrue(
                    g.validate_pr_info({**self.task, "description": missing})
                )
        for placeholder in ("TODO", "待填写"):
            with self.subTest(placeholder=placeholder):
                self.assertTrue(
                    g.validate_pr_info(
                        {**self.task, "description": body(summary=placeholder)}
                    )
                )
        self.assertTrue(g.validate_pr_info({**self.task, "title": "WIP"}))
        for title in ("test", "更新"):
            with self.subTest(title=title):
                self.assertEqual(g.validate_pr_info({**self.task, "title": title}), [])

    def test_external_fork_cannot_use_an_unprotected_environment(self):
        g.validate_approval_environment(self.gh)
        for environment in (
            {},
            {"protection_rules": []},
            {"protection_rules": [{"type": "required_reviewers", "reviewers": []}]},
        ):
            self.gh.environment = environment
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                g.validate_approval_environment(self.gh)

    def test_real_git_enqueue_manifest_last_and_idempotent_retry(self):
        control = self.store(g.CONTROL_BRANCH)
        g.enqueue(self.task, self.gh, control, self.source)
        self.assertEqual(
            control.get("tasks/" + self.task["task_id"] + ".json"), self.task
        )
        self.assertEqual(
            control.get("current/" + g.current_key(self.task) + ".json")["task_id"],
            self.task["task_id"],
        )
        for key, ref in (
            ("tested_sha", "task_ref"),
            ("base_sha", "base_task_ref"),
            ("head_sha", "head_task_ref"),
        ):
            actual = git(
                self.root,
                "--git-dir=" + str(self.remote),
                "rev-parse",
                "refs/heads/" + self.task[ref],
            )
            self.assertEqual(actual, self.task[key])
        retry = {**self.task, "captured_at": "2099-01-01T00:00:00Z", "worker_revision_sha": "f" * 40}
        g.enqueue(retry, self.gh, control, self.source)
        self.assertEqual(
            control.get("tasks/" + self.task["task_id"] + ".json"), self.task
        )

    def test_cancelled_task_identity_cannot_be_enqueued_again(self):
        control = self.store(g.CONTROL_BRANCH)
        control.put({
            f"cancel/{self.task['task_id']}.json": {
                "task_id": self.task["task_id"],
                "reason": "superseded",
                "created_at": "2026-09-23T00:00:00Z",
            }
        })
        with self.assertRaisesRegex(ValueError, "fresh verification"):
            g.enqueue(self.task, self.gh, control, self.source)
        self.assertIsNone(control.get(f"tasks/{self.task['task_id']}.json"))
        self.assertIsNone(control.get(f"current/{g.current_key(self.task)}.json"))
        self.assertEqual(self.gh.writes, [])

    def test_legacy_current_does_not_block_new_dispatch_or_collection(self):
        control = self.store(g.CONTROL_BRANCH)
        legacy = {
            **self.task,
            "schema": g.TASK_SCHEMA + "/v4",
            "pr_number": 8,
            "worker_revision_sha": "f" * 40,
            "task_ref": "ci/pr-8/docs-topic",
            "base_task_ref": "ci/base/pr-8/docs-topic",
            "head_task_ref": "ci/head/pr-8/docs-topic",
        }
        legacy["task_id"] = g.compute_task_id(legacy)
        control.put(
            {
                f"tasks/{legacy['task_id']}.json": legacy,
                f"current/{g.current_key(legacy)}.json": {
                    "task_id": legacy["task_id"]
                },
            }
        )
        results = self.store(g.RESULTS_BRANCH)
        before = git(control.root, "rev-parse", "HEAD")
        self.assertEqual(g.cancel_obsolete(self.gh, control, 8), 0)
        self.assertEqual(
            g.collect_results(self.gh, control, results, self.root / "dashboard", ""), []
        )
        self.assertEqual(git(control.root, "rev-parse", "HEAD"), before)
        self.assertEqual(self.gh.writes, [])

        g.enqueue(self.task, self.gh, control, self.source)
        result = self.result()
        results.put(
            {f"runs/{self.task['task_id']}/{result['run_id']}/result.json": result}
        )
        self.assertEqual(g.cancel_obsolete(self.gh, control, 7), 0)
        published = g.collect_results(
            self.gh, control, results, self.root / "dashboard"
        )
        self.assertEqual([row["task_id"] for row in published], [self.task["task_id"]])
        self.assertEqual(self.gh.statuses[-1][1], "success")
        snapshot = json.loads((self.root / "dashboard/tasks.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [row["task"]["task_id"] for row in snapshot["tasks"]],
            [self.task["task_id"]],
        )
        self.assertEqual(control.get(f"tasks/{legacy['task_id']}.json"), legacy)
        self.assertIsNone(control.get(f"cancel/{legacy['task_id']}.json"))

    def test_damaged_current_is_not_ignored_as_legacy(self):
        control = self.store(g.CONTROL_BRANCH)
        results = self.store(g.RESULTS_BRANCH)
        for mutation in (
            {"task_ref": self.task["task_ref"] + "-damaged"},
            {"task_ref": "ci/pr-7/docs-topic"},
            {"title": "Metadata changed without updating its identity"},
        ):
            with self.subTest(mutation=mutation):
                damaged = {**self.task, **mutation}
                control.put(
                    {
                        f"tasks/{self.task['task_id']}.json": damaged,
                        f"current/{g.current_key(self.task)}.json": {
                            "task_id": self.task["task_id"]
                        },
                    }
                )
                with self.assertRaises(ValueError):
                    g.cancel_obsolete(self.gh, control, 7)
                with self.assertRaises(ValueError):
                    g.collect_results(
                        self.gh, control, results, self.root / "dashboard"
                    )
        self.assertEqual(self.gh.writes, [])

    def test_branch_cancellation_ignores_unrelated_current_records(self):
        gh = FakeGitHub(self.base, self.tested, self.tested)
        task = g.prepare_task(gh, self.base, branch="main")
        control = self.store(g.CONTROL_BRANCH)
        g.enqueue(task, gh, control, self.source)
        unrelated = [
            {**task, "target_branch": "other"},
            self.task,
            {**task, "repository": "anteloper-c/triton-anchor"},
        ]
        damaged = {
            f"current/{g.current_key(subject)}.json": {"damaged": True}
            for subject in unrelated
        }
        control.put(damaged)
        gh.head = "e" * 40
        with patch.object(gh, "request", wraps=gh.request) as request:
            self.assertEqual(g.cancel_obsolete(gh, control, branch="main"), 1)
        self.assertEqual([call.args[0] for call in request.call_args_list], ["branches/main"])
        self.assertIsNotNone(control.get(f"cancel/{task['task_id']}.json"))
        self.assertEqual(g.cancel_obsolete(gh, control, branch="missing"), 0)
        for path, row in damaged.items():
            self.assertEqual(control.get(path), row)
        with self.assertRaises(ValueError):
            g.cancel_obsolete(gh, control)

    def test_collect_cancels_only_the_receivers_branch_and_requires_a_task_scope(self):
        gh = FakeGitHub(self.base, self.tested, self.tested)
        task = g.prepare_task(gh, self.base, branch="main")
        other = {**task, "target_branch": "other"}
        other["task_id"] = g.compute_task_id(other)
        for field, suffix in (("task_ref", "tested"), ("base_task_ref", "base"), ("head_task_ref", "head")):
            other[field] = f"ci/branch/{other['task_id']}/{suffix}"
        g.validate_task(other)
        control = self.store(g.CONTROL_BRANCH)
        documents = {}
        for item in (task, other, self.task):
            documents[f"tasks/{item['task_id']}.json"] = item
            documents[f"current/{g.current_key(item)}.json"] = {"task_id": item["task_id"]}
        control.put(documents)
        gh.head = "e" * 40
        request = gh.request
        store = g.GitStore
        for received_id in ("", task["task_id"]):
            with (
                self.subTest(received_id=received_id),
                patch("sys.argv", ["gateway.py", "collect"]),
                patch.dict(g.os.environ, {
                    "GITEE_RESULTS_REPO_URL": "https://gitee.com/test/results.git",
                    "RECEIVER_TASK_ID": received_id,
                    "PR_NUMBER": "0", "SOURCE_BRANCH": "other",
                    "GITHUB_RUN_ATTEMPT": "2",  # Receive/publish reruns keep their existing scope.
                }),
                patch.object(g, "GitHub", return_value=gh),
                patch.object(g, "GitStore", side_effect=lambda _url, branch: store(str(self.remote), branch)),
                patch.object(gh, "request", side_effect=lambda path: {"commit": {"sha": gh.head}} if path == "branches/other" else request(path)),
                patch.object(g, "collect_results") as collect,
            ):
                self.assertEqual(g.main(), 0)
                collect.assert_called_once()
            control.refresh()
            self.assertEqual(bool(control.get(f"cancel/{task['task_id']}.json")), bool(received_id))
            for item in (other, self.task):
                self.assertIsNone(control.get(f"cancel/{item['task_id']}.json"))

    def test_lifecycle_cancellation_reaches_gitee(self):
        control = self.store(g.CONTROL_BRANCH)
        g.enqueue(self.task, self.gh, control, self.source)
        for update in (
            {"draft": True},
            {"state": "closed"},
            {"body": body("Different summary")},
        ):
            self.gh.pull.update(update)
            self.assertFalse(g.is_current(self.gh, self.task))
        self.assertEqual(g.cancel_obsolete(self.gh, control, 7), 1)
        self.assertEqual(g.cancel_obsolete(self.gh, control, 7), 0)
        self.assertEqual(self.gh.comments, [])
        self.assertEqual(self.gh.statuses[-1][1], "error")
        self.assertTrue(self.gh.latest_statuses[self.task["task_id"]][1].isascii())
        self.assertTrue(control.get("cancel/" + self.task["task_id"] + ".json")["github_notified"])
        self.assertEqual(
            control.get("cancel/" + self.task["task_id"] + ".json")["task_id"],
            self.task["task_id"],
        )

    def test_result_publication_is_idempotent_and_keeps_relay_readonly(self):
        result = self.result()
        name = (
            "runs/" + self.task["task_id"] + "/" + result["run_id"] + "/result.json"
        )
        control, results = self.publish_result(result)
        control_revision = git(control.root, "rev-parse", "HEAD")
        result_revision = git(results.root, "rev-parse", "HEAD")
        published = g.collect_results(self.gh, control, results, self.root / "dashboard")
        self.assertEqual(
            published[0]["result_digest"],
            hashlib.sha256((results.root / name).read_bytes()).hexdigest(),
        )
        self.assertEqual(self.gh.statuses[-1][1], "success")
        self.assertEqual(len(self.gh.comments), 1)
        before = (len(self.gh.statuses), len(self.gh.comments))
        again = g.collect_results(self.gh, control, results, self.root / "dashboard")
        self.assertEqual(again, [])
        self.assertEqual(before, (len(self.gh.statuses), len(self.gh.comments)))
        self.assertEqual(control_revision, git(control.root, "rev-parse", "HEAD"))
        self.assertEqual(result_revision, git(results.root, "rev-parse", "HEAD"))

    def test_comment_failure_retries_same_uploaded_result(self):
        result = self.result()
        control, results = self.publish_result(result)
        with patch.object(
            self.gh, "comment", side_effect=RuntimeError("comment unavailable")
        ):
            self.assertEqual(
                g.collect_results(self.gh, control, results, self.root / "dashboard"),
                [],
            )
        snapshot = json.loads((self.root / "dashboard/tasks.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshot["tasks"][0]["status"], "pass")
        self.assertIn("receiver_error", snapshot["tasks"][0])
        self.assertEqual(self.gh.statuses[-1][1], "success")
        status_count = len(self.gh.statuses)
        result_revision = git(results.root, "rev-parse", "HEAD")
        self.assertEqual(
            len(g.collect_results(self.gh, control, results, self.root / "dashboard")),
            1,
        )
        self.assertEqual(self.gh.statuses[-1][1], "success")
        self.assertEqual(len(self.gh.comments), 1)
        self.assertEqual(len(self.gh.statuses), status_count)
        self.assertEqual(result_revision, git(results.root, "rev-parse", "HEAD"))

    def test_infrastructure_failure_reports_but_cancelled_results_do_not(self):
        result = self.result()
        result.update(
            status="infra_error",
            checks=[],
            reviews=[],
            blocking_reasons=["environment failed"],
        )
        control, results = self.publish_result(result)
        published = g.collect_results(
            self.gh, control, results, self.root / "dashboard"
        )
        self.assertEqual(len(published), 1)
        self.assertEqual(self.gh.statuses[-1][1], "error")
        control.put(
            {
                "cancel/" + self.task["task_id"] + ".json": {
                    "task_id": self.task["task_id"],
                    "reason": "cancel after collection",
                }
            }
        )
        before = (len(self.gh.statuses), len(self.gh.comments))
        self.assertEqual(
            g.collect_results(self.gh, control, results, self.root / "dashboard"), []
        )
        self.assertEqual(before, (len(self.gh.statuses), len(self.gh.comments)))

    def test_inactive_dashboard_result_distinguishes_supersession_from_cancellation(self):
        result = self.result()
        control, results = self.publish_result(result)
        before = (len(self.gh.statuses), len(self.gh.comments))
        with patch.object(self.gh, "owns_task", return_value=False):
            for reason, expected in ((None, "superseded"), ("superseded", "superseded"),
                                     ("manual cancellation", "cancelled")):
                with self.subTest(reason=reason):
                    if reason:
                        control.put({f"cancel/{self.task['task_id']}.json": {
                            "task_id": self.task["task_id"], "reason": reason,
                        }})
                    self.assertEqual(g.collect_results(
                        self.gh, control, results, self.root / "dashboard"), [])
                    row = json.loads((self.root / "dashboard/tasks.json").read_text())["tasks"][0]
                    self.assertEqual(row["status"], expected)
                    self.assertTrue(row["historical"])
                    self.assertEqual(row["result"]["status"], "pass")
                    self.assertEqual(before, (len(self.gh.statuses), len(self.gh.comments)))

    def test_failed_dashboard_is_rebuilt_without_repeating_completed_writeback(self):
        result = self.result()
        control, results = self.publish_result(result)
        dashboard = self.root / "dashboard"
        dashboard.write_text("not a directory")
        with self.assertRaises(OSError):
            g.collect_results(self.gh, control, results, dashboard)
        before = (len(self.gh.statuses), len(self.gh.comments))
        self.assertEqual(self.gh.statuses[-1][1], "success")
        dashboard.unlink()
        self.assertEqual(g.collect_results(self.gh, control, results, dashboard), [])
        self.assertEqual(before, (len(self.gh.statuses), len(self.gh.comments)))
        self.assertTrue((dashboard / "tasks.json").is_file())

    def test_head_change_during_status_write_skips_old_comment(self):
        result = self.result()
        control, results = self.publish_result(result)
        status = self.gh.status

        def update_head(*args, **kwargs):
            status(*args, **kwargs)
            self.gh.pull["head"]["sha"] = "e" * 40

        with patch.object(self.gh, "status", side_effect=update_head):
            self.assertEqual(
                g.collect_results(self.gh, control, results, self.root / "dashboard"),
                [],
            )
        self.assertFalse(self.gh.comments)
        self.assertEqual(g.cancel_obsolete(self.gh, control, 7), 1)
        self.assertEqual(self.gh.statuses[-1][1], "error")

    def test_latest_invalid_result_does_not_reuse_an_old_pass(self):
        result = self.result()
        prefix = f"runs/{self.task['task_id']}"
        control, results = self.publish_result(result)
        newer = {**result, "run_id": "20260908T120000Z-2", "blocking_reasons": ["Unresolved"]}
        results.put({f"{prefix}/{newer['run_id']}/result.json": newer})
        self.assertEqual(g.collect_results(self.gh, control, results, self.root / "dashboard"), [])
        self.assertEqual(self.gh.statuses[-1][1], "error")
        self.assertFalse(self.gh.comments)

    def test_sha_results_are_selected_by_task_not_just_latest_run(self):
        results = self.store(g.RESULTS_BRANCH)
        result = self.result()
        prefix = g.result_task_prefixes(self.task)[0]
        other = copy.deepcopy(result)
        other["run_id"] = "20990101T000000Z-other"
        other["task"]["task_id"] = "f" * 64
        own_path = f"{prefix}/{result['run_id']}/result.json"
        results.put({own_path: result, f"{prefix}/{other['run_id']}/result.json": other})
        self.assertEqual(g.latest_result(self.task, results), results.root / own_path)

    def test_grouped_task_id_results_remain_readable(self):
        results = self.store(g.RESULTS_BRANCH)
        result = self.result()
        prefix = g.result_task_prefixes(self.task)[1]
        path = f"{prefix}/{result['run_id']}/result.json"
        results.put({path: result})
        self.assertEqual(g.latest_result(self.task, results), results.root / path)

    def test_invalid_results_cannot_claim_success(self):
        for mutate in (
            lambda result: result.update(blocking_reasons=["deterministic regression"]),
            lambda result: result.update(run_id="../../escape"),
            lambda result: result["task"].update(worker_revision_sha="e" * 40),
        ):
            result = copy.deepcopy(self.result())
            mutate(result)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                g.validate_result(result, self.task)

    def test_receiver_reads_selected_files_from_same_git_result(self):
        result = self.result()
        result["artifacts"] = [{"path": "report.txt", "size": 2}]
        directory = self.root / result["run_id"]
        directory.mkdir()
        path = directory / "result.json"
        path.write_bytes(g.canonical(result))
        with self.assertRaises(ValueError):
            g.read_result(path, self.task, None)
        (directory / "artifacts").mkdir()
        (directory / "artifacts/report.txt").write_text("ok")
        self.assertEqual(g.read_result(path, self.task, None)[0], result)

    def test_receiver_progress_cannot_overwrite_a_final_status_seen_during_publication(self):
        from agent_ci import progress
        gh = RecordingGitHub(self.gh)
        pending = gh.seed_status(self.head, g.SUMMARY_CONTEXT, "pending", self.task["task_id"])
        reader = progress.ReceiverProgress({})
        # A concurrent finalizer completes after the progress helper's two reads.
        gh.seed_status(self.head, g.SUMMARY_CONTEXT, "success", self.task["task_id"])
        with (
            patch.object(gh, "latest_summary", return_value=pending),
            patch.object(gh, "owns_task", return_value=True),
            patch.object(progress, "read_health", return_value={}),
            patch.object(reader, "description", return_value="Local CI: running checks"),
        ):
            reader.update(gh, self.task)
        self.assertEqual(gh.writes, [])
        self.assertEqual(gh.summary_statuses(self.head)[g.SUMMARY_CONTEXT]["state"], "success")

    def test_receiver_finishes_as_soon_as_its_result_arrives(self):
        control = self.store(g.CONTROL_BRANCH)
        g.enqueue(self.task, self.gh, control, self.source)
        results = self.store(g.RESULTS_BRANCH)
        previous = self.task
        old_result = self.result()
        prefix = g.result_task_prefixes(previous)[0]
        old_path = f"{prefix}/{old_result['run_id']}/result.json"
        results.put({old_path: old_result})
        self.task = g.prepare_task(self.gh, self.base, 7, trigger_id="200:1")
        g.enqueue(self.task, self.gh, control, self.source)
        g.enqueue(self.task, self.gh, control, self.source)  # Repeated delivery is the same task.
        result = self.result()
        result.update(run_id="20260907T130000Z-2", status="fail", blocking_reasons=["Regression"])
        result["checks"][0]["status"] = "fail"

        def arrive(_seconds):
            results.put(
                {f"{prefix}/{result['run_id']}/result.json": result}
            )

        with patch.object(g.time, "sleep", side_effect=arrive) as sleep:
            self.assertEqual(
                g.receive_result(self.gh, str(self.remote), self.task["task_id"]),
                "ready",
            )
        sleep.assert_called_once_with(g.RECEIVER_POLL_SECONDS)
        self.assertIsNone(control.get(f"cancel/{self.task['task_id']}.json"))
        self.assertEqual(control.get(f"tasks/{previous['task_id']}.json"), previous)
        self.assertEqual(results.get(old_path), old_result)
        self.assertEqual(g.read_result(g.latest_result(self.task, results), self.task, results)[0]["status"], "fail")

    def test_receiver_stops_immediately_when_task_is_obsolete(self):
        control = self.store(g.CONTROL_BRANCH)
        g.enqueue(self.task, self.gh, control, self.source)
        self.gh.pull["head"]["sha"] = "e" * 40
        with patch.object(g.time, "sleep") as sleep:
            self.assertEqual(
                g.receive_result(self.gh, str(self.remote), self.task["task_id"]),
                "obsolete",
            )
        sleep.assert_not_called()

    def test_receiver_waits_for_dispatch_visibility_instead_of_retiring_task(self):
        control, results = self.publish_result(self.result())
        with (
            patch.object(
                self.gh,
                "latest_dispatch",
                side_effect=[
                    {"status": "in_progress", "conclusion": None},
                    {"status": "completed", "conclusion": "success"},
                ],
            ),
            patch.object(g.time, "sleep") as sleep,
        ):
            self.assertEqual(
                g.receive_result(self.gh, str(self.remote), self.task["task_id"]),
                "ready",
            )
        sleep.assert_called_once_with(g.RECEIVER_POLL_SECONDS)
        self.assertIsNone(control.get(f"cancel/{self.task['task_id']}.json"))

    def test_receiver_continuations_are_bounded_and_keep_the_same_task(self):
        self.task = g.prepare_task(self.gh, self.base, 7, trigger_id="200:1")
        control = self.store(g.CONTROL_BRANCH)
        g.enqueue(self.task, self.gh, control, self.source)
        before = git(control.root, "rev-parse", "HEAD")
        request = self.gh.request
        dispatched = []

        def dispatch(path, method="GET", data=None):
            if method == "POST":
                self.assertEqual(path, "actions/workflows/ci-gateway.yml/dispatches")
                dispatched.append(data)
                return None
            return request(path, method, data)

        for round_number in range(1, g.RECEIVER_MAX_ROUNDS + 1):
            elapsed = [0]

            def advance(seconds):
                elapsed[0] += seconds

            with (
                self.subTest(round_number=round_number),
                patch.object(self.gh, "request", side_effect=dispatch),
                patch.object(g, "RECEIVER_WAIT_SECONDS", 1),
                patch.object(g.time, "monotonic", side_effect=lambda: elapsed[0]),
                patch.object(g.time, "sleep", side_effect=advance),
            ):
                if round_number < g.RECEIVER_MAX_ROUNDS:
                    self.assertEqual(
                        g.receive_result(
                            self.gh, str(self.remote), self.task["task_id"], round_number
                        ),
                        "continued",
                    )
                else:
                    with self.assertRaisesRegex(ValueError, "timed out"):
                        g.receive_result(
                            self.gh, str(self.remote), self.task["task_id"], round_number
                        )
        self.assertEqual(
            dispatched,
            [
                {"ref": "main", "inputs": {
                    "mode": "receive", "task_id": self.task["task_id"],
                    "receiver_round": str(round_number),
                    "run_title": f"PR #7 | h:{self.head[:7]} m:{self.tested[:7]}",
                }}
                for round_number in (2, 3)
            ],
        )
        self.assertEqual(self.gh.statuses[-1][1], "error")
        control.refresh()
        self.assertEqual(git(control.root, "rev-parse", "HEAD"), before)
        self.assertIsNone(control.get(f"cancel/{self.task['task_id']}.json"))

    def test_receiver_retries_transient_transport_without_another_task(self):
        result = self.result()
        control, results = self.publish_result(result)
        before = git(control.root, "rev-parse", "HEAD")
        store = g.GitStore
        attempts = []

        def connect(*args, **kwargs):
            attempts.append(args)
            if len(attempts) == 1:
                raise OSError("temporary Gitee connection failure")
            return store(*args, **kwargs)

        with (
            patch.object(g, "GitStore", side_effect=connect),
            patch.object(g.time, "sleep") as sleep,
        ):
            self.assertEqual(
                g.receive_result(self.gh, str(self.remote), self.task["task_id"]),
                "ready",
            )
        sleep.assert_called_once_with(g.RECEIVER_POLL_SECONDS)
        control.refresh()
        self.assertEqual(git(control.root, "rev-parse", "HEAD"), before)

    def test_preflight_reports_cancelled_and_unexecuted_without_passing_them(self):
        with patch.object(self.gh, "check", return_value=True) as check:
            g.publish_preflight_checks(self.gh, self.task,
                                      {"basic": "failure", "api": "skipped", "security": "cancelled"}, False)
        self.assertEqual([call.args[3] for call in check.call_args_list],
                         ["failure"])
        stages = {key: "cancelled" for key in ("prepare", *g.CHECK_NAMES)}
        with patch.object(self.gh, "check") as check, patch.object(self.gh, "retire_open_checks") as retire:
            g.finalize_preflight(self.gh, self.task, stages)
            check.assert_not_called()
            retire.assert_called_once_with(self.task)
            g.finalize_preflight(self.gh, self.task, {
                "prepare": "success", "basic": "failure", "api": "skipped", "security": "skipped",
                "card": "skipped", "approval": "skipped", "enqueue": "skipped",
            })
            self.assertEqual([call.args[1:4] for call in check.call_args_list], [("basic", "completed", "failure")])
        self.assertEqual(self.gh.statuses, [])

        # The overall verdict distinguishes check failures from infrastructure errors.
        for outcomes, expected in (
            ({"prepare": "success", "basic": "failure", "enqueue": "skipped"}, "failure"),
            ({"prepare": "success", "basic": "cancelled", "enqueue": "skipped"}, "error"),
            ({"prepare": "failure", "basic": "skipped", "enqueue": "skipped"}, "error"),
            ({**dict.fromkeys(("prepare", *g.CHECK_NAMES, "card"), "success"),
              "approval": "failure", "enqueue": "skipped"}, "error"),
            ({**dict.fromkeys(("prepare", *g.CHECK_NAMES), "success"), "enqueue": "failure"}, "error"),
        ):
            client = RecordingGitHub(self.gh)
            with self.subTest(outcomes=outcomes), patch.dict(g.os.environ, {"GITHUB_RUN_ID": "12345", "GITHUB_RUN_ATTEMPT": "1"}):
                g.begin_checks(client, self.task)
                g.finalize_preflight(client, self.task, outcomes)
                self.assertEqual(client.latest_summary(self.task)["state"], expected)
                self.assertFalse(any(row["state"] == "pending" for row in client.commit_statuses(self.head).values()))

    def test_stage_completion_advances_only_after_success_without_summary_or_comments(self):
        outcomes = {"success": "success", "failure": "failure",
                    "cancelled": "cancelled", "skipped": "action_required"}
        for stage in g.CHECK_NAMES:
            for outcome, conclusion in outcomes.items():
                with self.subTest(stage=stage, outcome=outcome), patch.object(self.gh, "check", return_value=True) as check:
                    changed = g.sync_preflight(self.gh, self.task, {stage: outcome})
                    if outcome == "skipped":
                        self.assertFalse(changed)
                        check.assert_not_called()
                        continue
                    self.assertTrue(changed)
                    self.assertEqual(check.call_args_list[0].args[1:4], (stage, "completed", conclusion))
                    advances = outcome == "success" and stage != "security"
                    self.assertEqual(check.call_count, 2 if advances else 1)
                    if advances:
                        successor = "api" if stage == "basic" else "security"
                        self.assertEqual(check.call_args.args[1:4], (successor, "in_progress", None))
        self.assertEqual(self.gh.writes, [])

    def test_push_statuses_use_tested_commit_and_cannot_overwrite_a_new_owner(self):
        task = {**self.task, "pr_number": 0, "tested_sha": self.head, "event_kind": "push"}
        client = RecordingGitHub(self.gh)
        client.seed_status(self.head, g.CHECK_NAMES["basic"], "pending", task["task_id"], "1234")
        with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "1234", "GITHUB_RUN_ATTEMPT": "1"}):
            self.assertTrue(g.sync_preflight(client, task, {"basic": "success"}))
            self.assertEqual([(path, row["context"], row["state"]) for path, _, row in client.writes], [
                (f"statuses/{self.head}", g.CHECK_NAMES["basic"], "success"),
                (f"statuses/{self.head}", g.CHECK_NAMES["api"], "pending"),
            ])
            client.seed_status(self.head, g.CHECK_NAMES["basic"], "pending", "newer-task", "2345")
            client.writes.clear()
            self.assertFalse(g.sync_preflight(client, task, {"basic": "failure"}))
            g.finalize_preflight(client, task, {"prepare": "success", "basic": "failure"})
            self.assertEqual(client.writes, [])

    def test_native_control_push_publishes_stages_without_losing_task_ownership(self):
        task = {**self.task, "pr_number": 0, "event_kind": "push",
                "target_branch": "CI_dev", "tested_sha": self.head,
                "worker_revision_sha": self.head}
        client = RecordingGitHub(self.gh)
        rows = client.check_rows[self.head]

        def add_prepare(run_id, subject):
            rows.append({"id": run_id, "name": "Prepare exact task / " + subject,
                         "app": {"slug": "github-actions"},
                         "details_url": f"https://github.com/{g.REPOSITORY}/actions/runs/{run_id}/job/{run_id}"})

        add_prepare(100, "Branch CI_dev")
        add_prepare(200, "PR #62")
        add_prepare(300, "Branch main")
        # A legacy PR display copy on the control commit must not claim this push.
        rows.append(github_check({"name": g.CHECK_NAMES["basic"],
                    "external_id": f"{g.HEAD_CHECK_PREFIX}basic:pr-task",
                    "output": {"summary": "<!-- local-ci-workflow:350 -->"}}, 350))
        request = client.request

        def api(path, method="GET", data=None):
            if path == "branches/CI_dev":
                return {"commit": {"sha": self.head}}
            if path.startswith("actions/runs/"):
                return {"event": "workflow_dispatch", "head_branch": "CI_dev",
                        "head_sha": self.head, "path": ".github/workflows/ci-gateway.yml"}
            return request(path, method, data)

        with patch.object(client, "request", side_effect=api), \
                patch.dict(g.os.environ, {"GITHUB_RUN_ID": "100", "GITHUB_RUN_ATTEMPT": "1"}):
            self.assertTrue(client.owns_task(task, workflow=True))
            self.assertEqual(client.task_start({**task, "event_kind": "manual"}), {})
            g.begin_checks(client, task)
            self.assertEqual(set(client.commit_statuses(self.head)), {g.CHECK_NAMES["basic"], g.SUMMARY_CONTEXT})
            for stage in g.CHECK_NAMES:
                self.assertTrue(g.sync_preflight(client, task, {stage: "success"}))
            statuses = client.commit_statuses(self.head)
            self.assertEqual(set(statuses), {*g.CHECK_NAMES.values(), g.SUMMARY_CONTEXT})
            self.assertTrue(all(statuses[name]["state"] == "success" for name in g.CHECK_NAMES.values()))
            self.assertEqual(statuses[g.SUMMARY_CONTEXT]["state"], "pending")
            self.assertTrue(all(path == f"statuses/{self.head}" for path, method, _ in client.writes if method == "POST"))

            client.check(task, "dispatch", "completed", "success", "Dispatched", "Dispatched")
            client.status(task, "success", "Local CI: pass")
            before = len(client.writes)
            g.begin_checks(client, task)
            self.assertEqual(len(client.writes), before)  # A duplicate prepare keeps the finished verdict.
            add_prepare(400, "Branch CI_dev")
            self.assertFalse(client.owns_task(task, workflow=True))
            self.assertFalse(g.sync_preflight(client, task, {"basic": "failure"}))
            with self.assertRaisesRegex(ValueError, "A newer workflow owns this task"):
                g.begin_checks(client, task)
            self.assertEqual(len(client.writes), before)

    def test_pr_information_failure_has_friendly_actionable_feedback(self):
        errors = ["请补充影响范围", "请说明已完成的验证"]
        with (
            patch.object(g.sys, "argv", ["gateway.py", "info"]),
            patch.object(g, "GitHub", return_value=self.gh),
            patch.object(g, "load_task", return_value=self.task),
            patch.object(g, "validate_pr_info", return_value=errors),
        ):
            self.assertEqual(g.main(), 1)
        self.assertEqual(len(self.gh.comments), 1)
        message = self.gh.comments[0]
        for text in errors:
            self.assertIn(text, message)
        self.assertEqual(self.gh.statuses, [(self.task["task_id"], "failure")])
        self.assertEqual(self.gh.check_calls[-1][1:4], ("basic", "completed", "failure"))

    def test_result_comment_separates_evidence_delivery_from_execution(self):
        result = self.result()
        result["evidence_delivery"] = {
            "status": "incomplete", "omitted": [{"path": "report.txt"}]
        }
        rendered = g.result_comment(result)
        self.assertIn("执行通过，证据发布不完整", rendered)
        self.assertNotIn("环境或执行异常", rendered)
        result["status"] = "infra_error"
        result["evidence_delivery"]["omitted"][0]["required"] = True
        rendered = g.result_comment(result)
        self.assertIn("整体结论待确认", rendered)
        self.assertIn("CI 流程验证 | 通过", rendered)
        self.assertNotIn("环境或执行异常", rendered)

    def test_dashboard_history_restores_modern_and_legacy_data_without_gate_writes(self):
        from types import SimpleNamespace
        store = SimpleNamespace(root=self.root / "archive", url="https://gitee.com/example/results.git", branch="results")
        folder = store.root / "runs/pr/branch-main/pr-7" / self.head / "run-1"
        folder.mkdir(parents=True)
        result = self.result()
        result["task"]["full"] = True
        result["task"]["metadata_digest"] = g.metadata_digest(result["task"])
        result["task"]["task_id"] = g.compute_task_id(result["task"])
        prefix = f"ci/pr-{result['task']['pr_number']}/{result['task']['task_id']}"
        result["task"].update(
            task_ref=prefix + "/tested",
            base_task_ref=prefix + "/base",
            head_task_ref=prefix + "/head",
        )
        result["checks"].append({
            "tool_id": "flaggems", "status": "pass", "summary": "full",
            "parameters": {"mode": "full"},
        })
        (folder / "result.json").write_text(json.dumps(result))
        legacy = store.root / "runs/ci_full/main" / self.head / "20260724T112410Z-old"
        legacy.mkdir(parents=True)
        (legacy / "delivery-summary.txt").write_text(
            f"target_sha: {self.head}\nbranch: old-main\nstatus: 1\nbackend_profile: sophgo-cmodel\n"
            "flaggems_test_mode: full\nflaggems_status: fail\ncompile_time_status: pass\n"
            "backend_rebuild_status: pass\nbackend_smoke_jit_status: pass\nprivate_path: /private/credentials\n")
        full_document = {
            "schema": "triton-anchor-local-ci/flaggems-v1", "mode": "full",
            "summary": {"total": 1, "passed": 0, "failed": 1, "timed_out": 0, "status": "fail"},
            "results": [{"op": "add", "test_status": "失败"}],
        }
        (legacy / "flaggems-summary.json").write_text(json.dumps(full_document))
        (legacy / "compile-benchmark.json").write_text(json.dumps({"summary": {"add": {"compile_est": {"median_ms": 12}}}}))
        rows = g.history_rows(store, [])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["historical"] for row in rows))
        self.assertNotIn("private", json.dumps(rows))
        old = next(row for row in rows if row["task"]["target_branch"] == "old-main")
        self.assertEqual(old["result"]["environment"]["variants"]["candidate"], {
            "backend_enabled": True, "backend_profile": "sophgo-cmodel", "profile": "", "triton_version": "",
        })
        old_checks = {check["tool_id"]: check for check in old["result"]["checks"]}
        self.assertEqual(old_checks["flaggems"]["details"]["flaggems-summary"]["mode"], "full")
        self.assertEqual(old_checks["backend_smoke"]["status"], "pass")
        self.assertNotIn("backend_tests", old_checks)
        self.assertTrue(old["artifact_urls"]["flaggems-summary.json"].startswith("https://gitee.com/"))

        nested = store.root / "runs/ci_full_flaggems" / result["task"]["tested_sha"] / result["run_id"]
        nested.mkdir(parents=True)
        (nested / "flaggems-summary.json").write_text(json.dumps(full_document))
        demo_document = copy.deepcopy(full_document)
        demo = store.root / "runs/ci_full_flaggems" / g.FULL_FLAGGEMS_DEMO_SHA
        demo.mkdir(parents=True)
        (demo / "flaggems-summary.json").write_text(json.dumps(demo_document))
        impact = {
            "task": {"tested_sha": g.FULL_FLAGGEMS_DEMO_SHA, "full": False},
            "result": {"run_id": "impact-run", "checks": [{
                "tool_id": "flaggems", "status": "pass",
                "parameters": {"mode": "impact"},
                "details": {"flaggems-summary": {"mode": "impact", "results": []}},
            }]},
            "historical": True,
        }
        live_demo_sha = {
            "task": {"tested_sha": g.FULL_FLAGGEMS_DEMO_SHA, "full": True},
            "result": {"run_id": "20260923-live", "checks": [{
                "tool_id": "flaggems", "status": "pass",
                "parameters": {"mode": "full"},
            }]},
            "historical": True,
        }
        live_demo_path = demo / live_demo_sha["result"]["run_id"]
        live_demo_path.mkdir()
        (live_demo_path / "flaggems-summary.json").write_text(json.dumps(full_document))
        rows.extend([impact, live_demo_sha])
        g.attach_full_flaggems(store, rows)
        modern = next(row for row in rows if row["task"].get("task_id") == result["task"]["task_id"])
        modern_full = next(check for check in modern["result"]["checks"] if check["tool_id"] == "flaggems")
        self.assertEqual(modern_full["details"]["flaggems-summary"], full_document)
        self.assertEqual(modern["business_full"]["data_mode"], "live")
        self.assertNotIn("business_full", impact)
        self.assertEqual(
            impact["result"]["checks"][0]["details"]["flaggems-summary"]["mode"],
            "impact",
        )
        self.assertEqual(live_demo_sha["business_full"]["data_mode"], "live")
        sample = next(
            row for row in rows
            if row["task"].get("tested_sha") == g.FULL_FLAGGEMS_DEMO_SHA
            and row.get("business_full", {}).get("data_mode") == "mock"
        )
        self.assertTrue(sample["historical"])
        self.assertEqual(sample["business_full"]["data_mode"], "mock")
        self.assertEqual(sample["task"]["target_branch"], "ci/full/jiwang-delivery-ci")
        self.assertEqual(len(g.history_rows(store, [{"task": result["task"], "result": result}])), 1)

    def test_result_comment_retries_ignore_presentation_and_recognize_legacy_report(self):
        client = g.GitHub(g.REPOSITORY, token="fixture")
        comments, writes = [], []
        def request(path, method="GET", data=None):
            if method == "GET":
                return comments
            self.assertEqual(method, "POST")
            writes.append(data)
            comments.append({**data, "user": {"login": "github-actions[bot]"}})
            return {}
        identity = {"kind": "result", "task_id": self.task["task_id"], "run_id": "run-1", "result_digest": "a" * 64}
        url = "https://gitee.com/example/results/blob/results/run-1/result.json"
        with patch.object(client, "request", side_effect=request):
            self.assertTrue(client.comment(self.task, "first rendering", event_key=identity))
            self.assertFalse(client.comment(self.task, "<details>new rendering</details>", event_key=identity))
            self.assertTrue(client.comment(self.task, "first rendering", event_key={**identity, "run_id": "run-2"}))
            linked_body = "## Local CI 审查反馈\n[报告](" + url + ")"
            linked_identity = {**identity, "run_id": "linked"}
            self.assertTrue(client.comment(self.task, linked_body, event_key=linked_identity, legacy_result_url=url))
            self.assertTrue(client.comment(self.task, linked_body, event_key={**linked_identity, "result_digest": "b" * 64}, legacy_result_url=url))
            comments.append({"user": {"login": "github-actions[bot]"},
                             "body": g.MARKER + "\n<!-- old marker -->\n## Local CI 审查反馈\n[完整报告](" + url + ")"})
            self.assertFalse(client.comment(self.task, "new heading", event_key={**identity, "run_id": "old"}, legacy_result_url=url))
            comments[-1]["user"]["login"] = "contributor"
            self.assertTrue(client.comment(self.task, "new heading", event_key={**identity, "run_id": "old"}, legacy_result_url=url))
        self.assertEqual(len(writes), 5)

    def test_collection_only_writes_selected_task_and_readonly_collection_is_quiet(self):
        control, results = self.store(g.CONTROL_BRANCH), self.store(g.RESULTS_BRANCH)
        other = {**self.task, "pr_number": 8}
        other["task_id"] = g.compute_task_id(other)
        for field, suffix in (("task_ref", "tested"), ("base_task_ref", "base"), ("head_task_ref", "head")):
            other[field] = f"ci/pr-8/{other['task_id']}/{suffix}"
        for task in (self.task, other):
            g.validate_task(task)
            control.put({f"current/{g.current_key(task)}.json": {"task_id": task["task_id"]},
                         f"tasks/{task['task_id']}.json": task})
            result = {**self.result(), "task": task}
            results.put({f"runs/{task['task_id']}/{result['run_id']}/result.json": result})
        with (
            patch.object(g, "retired_task", return_value=False),
            patch.object(g, "current_task", return_value=True),
        ):
            published = g.collect_results(self.gh, control, results, self.root / "dashboard", self.task["task_id"])
            self.assertEqual([row["task_id"] for row in published], [self.task["task_id"]])
            self.assertEqual([row[0] for row in self.gh.statuses], [self.task["task_id"]])
            self.assertEqual(len(self.gh.comments), 1)
            self.gh.writes.clear()
            self.assertEqual(g.collect_results(self.gh, control, results, self.root / "dashboard", ""), [])
            self.assertEqual(self.gh.writes, [])
        feed = json.loads((self.root / "dashboard/tasks.json").read_bytes())
        self.assertEqual(len(feed["tasks"]), 2)

    def test_manual_publish_reconciles_only_current_owned_pending_result(self):
        control, results = self.publish_result(self.result())
        client = RecordingGitHub(self.gh)
        client.seed_status(
            self.head,
            g.SUMMARY_CONTEXT,
            "pending",
            self.task["task_id"],
        )
        with (
            patch.object(g, "retired_task", return_value=False),
            patch.object(g, "current_task", return_value=True),
            patch.object(client, "restore_preflight"),
            patch.object(client, "comment", return_value=False),
        ):
            published = g.collect_results(
                client,
                control,
                results,
                self.root / "dashboard",
                "",
                reconcile_pending=True,
            )
            self.assertEqual(
                [row["task_id"] for row in published],
                [self.task["task_id"]],
            )
            self.assertEqual(client.latest_summary(self.task)["state"], "success")
            before = len(client.writes)
            self.assertEqual(
                g.collect_results(
                    client,
                    control,
                    results,
                    self.root / "dashboard",
                    "",
                    reconcile_pending=True,
                ),
                [],
            )
            self.assertEqual(len(client.writes), before)

    def test_obsolete_receiver_closes_only_its_pending_summary_without_reusing_result(self):
        control, results = self.publish_result(self.result())
        control.put({
            f"cancel/{self.task['task_id']}.json": {
                "task_id": self.task["task_id"],
                "reason": "superseded",
                "superseded_by": "f" * 64,
                "created_at": "2026-09-23T00:00:00Z",
            }
        })
        client = RecordingGitHub(self.gh)
        client.seed_status(
            self.head,
            g.SUMMARY_CONTEXT,
            "pending",
            self.task["task_id"],
        )
        self.assertEqual(
            g.collect_results(
                client,
                control,
                results,
                self.root / "dashboard",
                self.task["task_id"],
            ),
            [],
        )
        summary = client.latest_summary(self.task)
        self.assertEqual(summary["state"], "error")
        self.assertIn("superseded", summary["description"])
        feed = json.loads((self.root / "dashboard/tasks.json").read_bytes())
        row = next(row for row in feed["tasks"] if row["task"]["task_id"] == self.task["task_id"])
        self.assertTrue(row["historical"])
        self.assertEqual(row["status"], "superseded")
        self.assertEqual(row["result"]["status"], "pass")

    def test_collect_retries_visible_but_incomplete_dispatch_without_closing_summary(self):
        control, results = self.publish_result(self.result())
        client = RecordingGitHub(self.gh)
        client.seed_status(
            self.head,
            g.SUMMARY_CONTEXT,
            "pending",
            self.task["task_id"],
        )
        output = self.root / "github-output"
        with (
            patch.object(
                client,
                "latest_dispatch",
                return_value={"status": "in_progress", "conclusion": None},
            ),
            patch.dict(g.os.environ, {"GITHUB_OUTPUT": str(output)}),
        ):
            self.assertEqual(
                g.collect_results(
                    client,
                    control,
                    results,
                    self.root / "dashboard",
                    self.task["task_id"],
                ),
                [],
            )
        self.assertEqual(client.latest_summary(self.task)["state"], "pending")
        self.assertIn("receiver_errors=1", output.read_text())
        feed = json.loads((self.root / "dashboard/tasks.json").read_bytes())
        row = next(row for row in feed["tasks"] if row["task"]["task_id"] == self.task["task_id"])
        self.assertFalse(row["historical"])
        self.assertEqual(row["status"], "pending")

    def test_pr_stage_statuses_and_summary_use_only_head_and_retry_idempotently(self):
        client = RecordingGitHub(self.gh)
        client.seed_status(self.head, "local-ci/summary", "success", "older-task")
        with patch.dict(g.os.environ, {
            "GITHUB_RUN_ID": "12345", "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_SERVER_URL": "https://github.com", "GITHUB_REPOSITORY": g.REPOSITORY,
        }):
            with patch.object(client, "status", side_effect=RuntimeError("Summary publication failed")):
                with self.assertRaisesRegex(RuntimeError, "Summary publication failed"):
                    g.begin_checks(client, self.task)
            # Retrying initialization repairs Summary without duplicating Basic.
            g.begin_checks(client, self.task)
            self.assertEqual([row["context"] for _, _, row in client.writes], [g.CHECK_NAMES["basic"], g.SUMMARY_CONTEXT])
            client.fail_next_status = True
            with self.assertRaisesRegex(RuntimeError, "Status publication"):
                g.sync_preflight(client, self.task, {"basic": "success"})
            g.sync_preflight(client, self.task, {"basic": "success"})
            for key in ("api", "security"):
                self.assertEqual(client.latest_summary(self.task)["state"], "pending")
                g.sync_preflight(client, self.task, {key: "success"})
            client.check(self.task, "approve", "in_progress", None, "Awaiting approval", "Awaiting approval", g.workflow_url())
            approval = client.status_rows[self.head][0]
            self.assertEqual(approval["state"], "pending")
            self.assertTrue(approval["target_url"].startswith(g.workflow_url() + "#"))
            self.assertEqual(client.latest_summary(self.task)["state"], "pending")
            for key in ("approve", "dispatch"):
                client.check(self.task, key, "completed", "success", "Passed", "Passed", g.workflow_url())
            client.status(self.task, "pending", "Waiting for worker", "https://gitee.com/report")
            client.status(self.task, "success", "Local CI: pass", "https://gitee.com/report")
            self.assertTrue(client.status_matches(self.task, "success", "Local CI: pass"))
            before = len(client.writes)
            client.status(self.task, "success", "Local CI: pass", "https://gitee.com/report")
            client.check(self.task, "dispatch", "completed", "success", "Passed", "Passed", g.workflow_url())
            self.assertEqual(len(client.writes), before)
        self.assertEqual({path for path, _, _ in client.writes}, {f"statuses/{self.head}"})
        self.assertEqual({method for _, method, _ in client.writes}, {"POST"})
        self.assertEqual({row["context"] for _, _, row in client.writes}, {*g.ALL_CHECK_NAMES.values(), g.SUMMARY_CONTEXT})
        self.assertEqual(client.check_rows[self.head], [])
        self.assertEqual(client.status_rows[self.tested], [])

    def test_request_handoff_blocks_old_writers_and_preserves_request_identity(self):
        client = RecordingGitHub(self.gh)
        client.seed_status(self.head, g.CHECK_NAMES["basic"], "success", self.task["task_id"], "100")
        client.seed_status(self.head, "Local CI Summary", "pending", self.task["task_id"], "100")
        target = f"https://github.com/{g.REPOSITORY}/actions/runs/200#local-ci-request=200:1"
        client.seed_status(self.head, g.SUMMARY_CONTEXT, "pending", "", target_url=target)
        self.assertEqual(client.latest_summary(self.task)["target_url"], target)
        self.assertFalse(client.owns_task(self.task))
        client.status(self.task, "success", "Old result")
        # Also guard a request arriving after the receiver's ownership read.
        with patch.object(client, "owns_task", return_value=True), patch.dict(g.os.environ, {"LOCAL_CI_REQUEST_ID": ""}):
            client.status(self.task, "success", "Late old result")
            client.status(self.task, "pending", "Late direct dispatch", claim_request=True)
        self.assertEqual(client.writes, [])
        with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "201", "GITHUB_RUN_ATTEMPT": "1", "LOCAL_CI_REQUEST_ID": "199:1"}):
            with self.assertRaisesRegex(ValueError, "newer workflow"):
                g.begin_checks(client, self.task)
        self.assertEqual(client.writes, [])
        with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "201", "GITHUB_RUN_ATTEMPT": "1", "LOCAL_CI_REQUEST_ID": "200:1"}):
            g.begin_checks(client, self.task)
            self.assertTrue(client.owns_task(self.task, workflow=True))
            row = client.latest_summary(self.task)
            self.assertEqual(g.status_identity(row), {"local-ci-task": self.task["task_id"], "local-ci-request": "200:1"})
            self.assertEqual(row["state"], "pending")
            g.finalize_preflight(client, self.task, {"prepare": "success", "basic": "failure"})
            self.assertEqual(client.latest_summary(self.task)["state"], "failure")
            self.assertEqual(g.status_identity(client.latest_summary(self.task))["local-ci-request"], "200:1")
        self.assertEqual(client.commit_statuses(self.head)["Local CI Summary"]["state"], "error")

    def test_renamed_summary_finishes_pending_aliases_without_rewriting_completed_history(self):
        client = RecordingGitHub(self.gh)
        client.seed_status(self.head, "Local CI Summary", "pending", self.task["task_id"])
        client.seed_status(self.tested, "local-ci/summary", "success", self.task["task_id"])
        client.status(self.task, "success", "Local CI: pass")
        self.assertEqual({row["context"] for _, _, row in client.writes}, {"Summary", "Local CI Summary"})
        self.assertTrue(all(row["state"] == "success" for _, _, row in client.writes))
        self.assertEqual(client.commit_statuses(self.tested)["local-ci/summary"]["state"], "success")

    def test_legacy_result_ownership_and_checks_survive_name_and_commit_migration(self):
        client = g.GitHub(g.REPOSITORY, token="fixture")
        names = {key: g.CHECK_ALIASES[key][0] for key in (*g.CHECK_NAMES, "dispatch")}
        checks = [github_check({"name": name, "status": "completed", "conclusion": "success",
                               "external_id": f"triton-anchor-local-ci:{key}:{self.task['task_id']}"}, index)
                  for index, (key, name) in enumerate(names.items(), 1)]
        summaries = {
            self.head: [{"context": "local-ci/summary", "state": "success", "target_url": "https://github.com/run#local-ci-task=older"}],
            self.tested: [{"context": "local-ci/summary", "state": "success", "target_url": "https://github.com/run#local-ci-task=" + self.task["task_id"]}],
        }

        def request(path, method="GET", data=None):
            sha = path.split("/")[1]
            if "/statuses?" in path:
                return summaries[sha]
            if "/check-runs?" in path:
                return {"check_runs": [row for row in checks if sha == self.head
                                      and f"check_name={g.quote(row['name'], safe='')}&" in path]}
            raise AssertionError(path)

        with patch.object(client, "request", side_effect=request), patch.dict(g.os.environ, {"GITHUB_RUN_ID": ""}):
            self.assertEqual(client.latest_summary(self.task), summaries[self.tested][0])
            self.assertTrue(client.owns_task(self.task))
            with patch.object(client, "check", return_value=True) as publish:
                client.restore_preflight(self.task)
            publish.assert_not_called()  # Legacy evidence is validated without creating display copies.
            # The compatible records must not let an old receiver claim a newer task.
            checks[0].update(id=99, external_id="triton-anchor-local-ci:basic:new-task",
                             output={"summary": "<!-- local-ci-workflow:999 -->"})
            self.assertFalse(client.owns_task(self.task))
            with patch.object(g, "is_current", return_value=True), \
                    patch.dict(g.os.environ, {"GITHUB_RUN_ID": "998"}), \
                    self.assertRaisesRegex(ValueError, "newer workflow"):
                g.begin_checks(client, self.task)

    def test_reopen_closes_pending_legacy_checks_without_reopening_old_summary(self):
        client = RecordingGitHub(self.gh)
        client.check_rows[self.tested] = [github_check({
            "name": g.CHECK_NAMES["basic"], "status": "queued",
            "external_id": f"triton-anchor-local-ci:basic:{self.task['task_id']}",
        }, 42)]
        client.seed_status(self.head, "local-ci/summary", "pending", self.task["task_id"])
        client.seed_status(self.head, "external/build", "pending", "external")
        with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "12345", "GITHUB_RUN_ATTEMPT": "1"}):
            g.begin_checks(client, self.task)
        by_context = {row["context"]: row for _, _, row in client.writes if "context" in row}
        self.assertEqual(by_context[g.CHECK_NAMES["basic"]]["state"], "pending")
        self.assertEqual(by_context["local-ci/summary"]["state"], "error")
        self.assertEqual(by_context[g.SUMMARY_CONTEXT]["state"], "pending")
        self.assertNotIn("external/build", by_context)
        self.assertEqual(client.check_rows[self.tested][0]["conclusion"], "cancelled")
        self.assertFalse(any(path == "check-runs" for path, _, _ in client.writes))

    def test_approval_context_is_frozen_and_displays_contributor_claims_safely(self):
        client = g.GitHub(g.REPOSITORY, token="fixture")
        pull = {**self.gh.pull, "user": {"login": "contributor"}, "changed_files": 2}
        files = [{"filename": ".github/workflows/test.yml", "additions": 3, "deletions": 1},
                 {"filename": "pyproject.toml", "additions": 2, "deletions": 0}]
        def request(path, method="GET", data=None):
            return files if "/files?" in path else pull
        with patch.object(client, "request", side_effect=request):
            context = client.approval_context(self.task)
            card = g.approval_card(self.task, dict.fromkeys(("prepare", *g.CHECK_NAMES), "success"), True, context=context)
            for sha in (self.head, self.tested):
                self.assertIn(sha, card)
            pull["head"] = {**pull["head"], "sha": "e" * 40}
            with self.assertRaisesRegex(ValueError, "PR changed"):
                client.approval_context(self.task)

    def test_result_comment_puts_blockers_and_code_links_in_findings(self):
        result = self.result()
        result["status"] = "fail"
        result["checks"][0].update(status="infra_error", summary="工具拒绝仓内符号链接，不影响独立复现的缺陷结论")
        result["checks"].append({"tool_id": "frontend_build", "status": "not_selected", "summary": "无需构建"})
        result["blocking_reasons"] = [
            "最低必检未通过：change_validation — 插件发现回归",
            "必要审查未通过：architecture — 插件发现回归",
            "control_plane：工具拒绝仓内符号链接，不影响独立复现的缺陷结论",
        ]
        result["findings"] = [
            {"summary": "架构契约遭到破坏", "blocking": True,
             "qualification": "目录扫描遗漏独立安装的插件，导致后端无法注册",
             "code_evidence": {"path": "src/file.py", "line": 17},
             "evidence": ["validation.md", "unpublished.log:8"]},
            {"summary": "异常路径丢失资源释放", "blocking": True,
             "qualification": "初始化失败后未关闭句柄，与插件发现是独立问题",
             "code_evidence": "src/file.py:31-33"},
            {"summary": "可以改进错误提示", "blocking": False, "severity": "low",
             "qualification": "错误信息没有说明应检查哪个配置字段",
             "code_evidence": ["../secret", "/absolute/path", "https://evil.invalid", "validation.md:1", "src/file.py:23", "src/file.py:23"]},
        ]
        result["blocking_reasons"].append(result["findings"][0]["summary"])
        rendered = g.result_comment(result, artifact_urls={"validation.md": "https://example.invalid/validation.md"})
        findings = rendered.split("### 需要关注的发现", 1)[1].split("### 查看审查详情", 1)[0]
        limitations = rendered.split("### 限制说明", 1)[1]
        for reason in result["blocking_reasons"][:-1]:
            self.assertNotIn(g.feedback_text(reason), findings)
        self.assertEqual(findings.count("【合入阻塞】"), 2)
        self.assertEqual(findings.count("架构契约遭到破坏"), 1)
        self.assertEqual(sum(line.startswith("- ") for line in findings.splitlines()), 3)
        for finding in result["findings"][:2]:
            self.assertIn(
                f"- 【合入阻塞】{finding['summary']}  \n  分析：{finding['qualification']}  \n  代码位置：",
                findings,
            )
        self.assertIn(
            "- 【风险：低】可以改进错误提示  \n"
            "  分析：错误信息没有说明应检查哪个配置字段  \n"
            "  代码位置：[src/file.py:23]",
            findings,
        )
        self.assertIn("目录扫描遗漏独立安装的插件", findings)
        self.assertIn("异常路径丢失资源释放", findings)
        self.assertNotIn("工具拒绝仓内符号链接", findings)
        self.assertIn("工具拒绝仓内符号链接", limitations)
        for line in (17, 23, 31):
            self.assertIn(f"/blob/{self.tested}/src/file.py#L{line}", findings)
        self.assertIn(f"/blob/{self.tested}/src/file.py#L31-L33", findings)
        self.assertNotIn("secret", findings)
        self.assertNotIn("evil.invalid", findings)
        self.assertNotIn("/absolute/path", findings)
        self.assertNotIn("前端构建", rendered)
        self.assertNotIn("证据 1", findings)
        self.assertNotIn("validation.md", findings)
        self.assertNotIn("unpublished.log", findings)
        self.assertEqual(findings.count(f"/blob/{self.tested}/src/file.py#L23"), 1)
        self.assertNotIn(f"/blob/{self.tested}/s)", findings)

    def test_sealed_advisories_recovery_and_limits_use_contributor_language(self):
        from agent_ci.delivery import seal_result

        result = self.result()
        result["policy"] = {"required_checks": ["change_validation"]}
        result["summary"] = "candidate 的 backend_tests 已完成，control_plane 的补充范围受限。"
        limitation = "control_plane 没有现成回归套件；必要行为已通过定向断言验证。"
        result["checks"] = [
            {"tool_id": "change_validation", "status": "pass", "summary": "行为验证完成",
             "evidence": ["validation.md"]},
            {"tool_id": "backend_tests", "status": "pass", "summary": "已使用等价入口执行同一组断言。",
             "details": {"initial_status": "infra_error", "recovery": "修复导入方式后验证通过"}},
            {"tool_id": "control_plane", "status": "limited", "summary": "缺少现成回归套件",
             "limitation": limitation},
            {"tool_id": "diff_check", "status": "warning", "summary": "文档行尾空白"},
            {"tool_id": "custom_probe", "display_name": "标量内存定向验证", "status": "pass", "summary": "四组断言通过"},
        ]
        result["findings"] = [{"summary": "文档行尾空白", "severity": "low", "blocking": False}]
        result["limitations"] = [limitation]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "artifacts").mkdir()
            (source / "artifacts/validation.md").write_text("validated fixture\n")
            sealed = seal_result(self.task, result["run_id"], result, result["policy"],
                                 result["environment"], source, source / "sealed")
        self.assertEqual(sealed["status"], "pass")
        rendered = g.result_comment(sealed)
        limits = rendered.split("### 限制说明", 1)[1]
        self.assertEqual(limits.count("没有现成回归套件"), 1)
        self.assertNotIn("行尾空白", limits)
        self.assertNotIn("【合入阻塞】", rendered)
        self.assertNotIn("验证未完成（环境或执行异常）", rendered)
        self.assertIn("【风险：低】文档行尾空白", rendered)
        findings = rendered.split("### 需要关注的发现", 1)[1].split("### 查看审查详情", 1)[0]
        self.assertEqual(findings.strip(), "- 【风险：低】文档行尾空白")
        self.assertIn("| 格式检查 | 提示 |", rendered)
        self.assertIn("| CI 流程验证 | 验证受限 |", rendered)
        self.assertIn("标量内存定向验证", rendered)
        for identifier in ("control_plane", "backend_tests", "custom_probe", "diff_check"):
            self.assertNotIn(g.feedback_text(identifier), rendered)
        self.assertIn("候选 的 后端测试 已完成", rendered)

    def test_comment_uses_version_names_without_inventing_unselected_limits(self):
        result = self.result()
        result["checks"] = [{"tool_id": "source_syntax", "status": "pass", "summary": "语法检查通过"}]
        result["limitations"] = []
        result["summary"] = "baseline 和 candidate 一致，base 无变化，candidate/base 与 baseline/candidate 已比较。"
        result["checks"].append({"tool_id": "custom_probe", "display_name": "candidate profile 核对", "status": "pass",
                                 "summary": "控制面根据冻结任务上下文核对候选 profile、venv 和 checkout，通过。"})
        rendered = g.result_comment(result)
        self.assertIn("base 和 候选 一致，base 无变化", rendered)
        self.assertIn("候选/base 与 base/候选 已比较", rendered)
        self.assertIn("候选 环境配置 核对", rendered)
        self.assertIn("CI 流程根据任务信息核对候选 环境配置、Python 虚拟环境 和 源码目录，通过", rendered)
        self.assertNotIn("### 限制说明", rendered)
        self.assertNotIn("报告未列出编译器", rendered)
        self.assertEqual(g.feedback_prose("在candidate中与Baseline一致"), "在候选中与base一致")
        self.assertEqual(g.feedback_prose("pull_request 检查标为 limited"), "PR 检查标为 验证受限")
        literal = "/task/candidate/checkout src/baseline.py base_sha candidate-context `python -m venv /tmp/test` `backend_enabled=false`"
        self.assertEqual(g.feedback_prose(literal), g.feedback_text(literal))

    def test_result_comment_preserves_failure_without_findings_and_nonblocking_limits(self):
        result = self.result()
        result["status"] = "fail"
        result["blocking_reasons"] = ["独立回归失败"]
        result["limitations"] = ["性能基线不可比，不影响正确性审查"]
        rendered = g.result_comment(result)
        self.assertIn("【合入阻塞】独立回归失败", rendered)
        self.assertNotIn("分析：", rendered)
        self.assertNotIn("代码位置：", rendered)
        self.assertIn("### 限制说明\n\n- 性能基线不可比，不影响正确性审查", rendered)
        result["status"] = "pass"
        result["blocking_reasons"] = []
        rendered = g.result_comment(result)
        self.assertNotIn("【合入阻塞】", rendered)
        self.assertIn("本次要求的检查已通过", rendered)
        self.assertIn("性能基线不可比，不影响正确性审查", rendered)
        del result["limitations"]
        result["blocking_reasons"] = ["必要验证因环境异常未完成"]
        for status in ("infra_error", "cancelled"):
            with self.subTest(status=status):
                result["status"] = status
                rendered = g.result_comment(result)
                self.assertNotIn("【合入阻塞】", rendered)
                self.assertIn("必要验证因环境异常未完成", rendered.split("### 限制说明", 1)[1])

    def test_checks_appear_only_as_their_stage_is_reached(self):
        control = self.store(g.CONTROL_BRANCH)
        stages = {key: "success" for key in ("prepare", *g.CHECK_NAMES)}
        events = []
        with patch.object(self.gh, "check", side_effect=lambda *args, **kwargs: events.append((args[1], args[2])) or True) as check, \
                patch.object(self.gh, "status", side_effect=lambda *args, **kwargs: events.append(("summary", args[1]))), \
                patch.object(g, "GitHub", return_value=self.gh), \
                patch.object(g, "load_task", return_value=self.task), patch.object(g, "output"), \
                patch.dict(g.os.environ, {"GITHUB_STEP_SUMMARY": ""}):
            g.begin_checks(self.gh, self.task)
            self.assertEqual(events, [("basic", "queued"), ("summary", "pending")])
            for stage, visible in (("basic", ["basic", "api"]),
                                   ("api", ["basic", "api", "security"]),
                                   ("security", ["basic", "api", "security"])):
                self.assertTrue(g.sync_preflight(self.gh, self.task, {stage: "success"}))
                self.assertEqual(list(dict.fromkeys(key for key, _ in events if key != "summary")), visible)
            with patch.object(g.sys, "argv", ["gateway.py", "card", "--stages", json.dumps(stages)]):
                self.assertEqual(g.main(), 0)
            self.assertEqual(events[-2:], [("approve", "in_progress"), ("summary", "pending")])
            self.assertEqual(list(dict.fromkeys(key for key, _ in events if key != "summary")), ["basic", "api", "security", "approve"])
            with patch.object(g.sys, "argv", ["gateway.py", "approval"]):
                self.assertEqual(g.main(), 0)
            self.assertEqual(events[-1], ("approve", "completed"))
            g.enqueue(self.task, self.gh, control, self.source)
            self.assertEqual(events[-3:], [("dispatch", "in_progress"), ("summary", "pending"), ("dispatch", "completed")])
            self.assertEqual(check.call_args.args[3], "success")
            self.assertEqual(list(dict.fromkeys(key for key, _ in events if key != "summary")),
                             ["basic", "api", "security", "approve", "dispatch"])

    def test_same_task_receiver_waits_for_successful_dispatch(self):
        result = self.result()
        control, results = self.publish_result(result)
        before = (list(self.gh.statuses), list(self.gh.comments))
        self.assertTrue(g.current_task(self.gh, control, self.task))  # Legacy tasks have no dispatch Check Run.
        for status, conclusion in (("not_started", None), ("queued", None), ("in_progress", None), ("completed", "failure")):
            with self.subTest(status=status), patch.object(self.gh, "latest_dispatch", return_value={
                "status": status, "conclusion": conclusion,
            }), patch.object(g.time, "sleep") as sleep:
                self.assertFalse(g.retired_task(self.gh, control, self.task))
                self.assertFalse(g.current_task(self.gh, control, self.task))
                self.assertEqual(g.collect_results(self.gh, control, results, self.root / "dashboard"), [])
                self.assertFalse(g.finish_obsolete_summary(self.gh, control, self.task))
                if status == "completed":
                    self.assertEqual(
                        g.receive_result(
                            self.gh, str(self.remote), self.task["task_id"]
                        ),
                        "dispatch_failed",
                    )
                    sleep.assert_not_called()
            self.assertEqual((self.gh.statuses, self.gh.comments), before)
        with patch.object(self.gh, "latest_dispatch", return_value={"status": "completed", "conclusion": "success"}):
            self.assertTrue(g.current_task(self.gh, control, self.task))
            self.assertEqual(len(g.collect_results(self.gh, control, results, self.root / "dashboard")), 1)
        client = RecordingGitHub(self.gh)
        with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "200", "GITHUB_RUN_ATTEMPT": "2"}):
            client.seed_status(self.head, g.CHECK_NAMES["basic"], "pending", self.task["task_id"], "200", "2")
            for run_id, attempt, current in (("100", "1", False), ("200", "1", False), ("200", "2", True)):
                with self.subTest(run=run_id, attempt=attempt):
                    client.seed_status(self.head, g.ALL_CHECK_NAMES["dispatch"], "success", self.task["task_id"], run_id, attempt)
                    with patch.object(self.gh, "latest_dispatch", side_effect=client.latest_dispatch):
                        self.assertEqual(g.current_task(self.gh, control, self.task), current)

    def test_prepare_exception_comments_without_writing_summary_on_head(self):
        writes = []

        def urlopen(request, timeout=30):
            path = request.full_url.split(f"/repos/{g.REPOSITORY}/", 1)[1]
            if request.get_method() == "GET" and path == "pulls/7":
                payload = self.gh.pull
            elif request.get_method() == "GET" and path.startswith("issues/7/comments?"):
                payload = []
            elif request.get_method() == "POST" and path == "issues/7/comments":
                writes.append((path, json.loads(request.data)))
                payload = {}
            else:
                raise AssertionError((path, request.get_method()))
            return io.BytesIO(json.dumps(payload).encode())

        original = Path.cwd()
        try:
            g.os.chdir(self.root)
            with patch("urllib.request.urlopen", side_effect=urlopen), patch.object(g.sys, "argv", [
                "gateway.py", "prepare", "--worker-sha", "invalid",
            ]), patch.dict(g.os.environ, {
                "GH_TOKEN": "fixture", "PR_NUMBER": "7", "REQUESTED_SHA": self.head,
                "GITHUB_REPOSITORY": g.REPOSITORY, "GITHUB_RUN_ID": "200",
            }), self.assertRaises(SystemExit) as stopped:
                runpy.run_path(str(ROOT / "scripts/ci/gateway.py"), run_name="__main__")
            self.assertEqual(stopped.exception.code, 1)
        finally:
            g.os.chdir(original)
        self.assertEqual([path for path, _ in writes], ["issues/7/comments"])
        self.assertIn("Invalid trusted worker revision", writes[0][1]["body"])

    def test_same_head_new_run_and_attempt_invalidate_old_stages_and_block_old_results(self):
        result = self.result()
        control, results = self.publish_result(result)
        client = RecordingGitHub(self.gh)
        for context in (g.CHECK_NAMES["basic"], g.CHECK_NAMES["api"], g.ALL_CHECK_NAMES["dispatch"], g.SUMMARY_CONTEXT):
            client.seed_status(self.head, context, "success", self.task["task_id"], "100")
        with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "200", "GITHUB_RUN_ATTEMPT": "1"}):
            g.begin_checks(client, self.task)
            latest = client.commit_statuses(self.head)
            self.assertEqual(latest[g.CHECK_NAMES["basic"]]["state"], "pending")
            for key in ("api", "dispatch"):
                self.assertEqual(latest[g.ALL_CHECK_NAMES[key]]["state"], "error")
            self.assertEqual(latest[g.SUMMARY_CONTEXT]["state"], "pending")
            self.assertNotIn(g.CHECK_NAMES["security"], latest)
        before = len(client.writes)
        with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "100", "GITHUB_RUN_ATTEMPT": "1"}):
            self.assertFalse(g.sync_preflight(client, self.task, {"basic": "failure"}))
            with self.assertRaisesRegex(ValueError, "newer workflow"):
                g.begin_checks(client, self.task)
        with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "200", "GITHUB_RUN_ATTEMPT": "2"}):
            g.begin_checks(client, self.task)
        self.assertGreater(len(client.writes), before)
        before = len(client.writes)
        with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "200", "GITHUB_RUN_ATTEMPT": "1"}):
            self.assertFalse(g.sync_preflight(client, self.task, {"basic": "success"}))
            with self.assertRaisesRegex(ValueError, "newer workflow"):
                g.begin_checks(client, self.task)
        self.assertEqual(len(client.writes), before)
        # A different task can reuse the same head before it reaches Gitee.
        client.seed_status(self.head, g.CHECK_NAMES["basic"], "pending", "new-task", "300")
        with patch.object(self.gh, "owns_task", side_effect=client.owns_task):
            before = len(self.gh.statuses)
            self.assertEqual(g.collect_results(self.gh, control, results, self.root / "dashboard"), [])
            self.assertEqual(len(self.gh.statuses), before)

        # Re-run failed jobs retains the successful Basic job from attempt 1.
        for outcome in ("cancelled", "success"):
            with self.subTest(rerun=outcome):
                client = RecordingGitHub(self.gh)
                client.seed_status(self.head, g.CHECK_NAMES["basic"], "success", self.task["task_id"], "400", "1")
                client.seed_status(self.head, g.CHECK_NAMES["api"], "failure", self.task["task_id"], "400", "1")
                with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "400", "GITHUB_RUN_ATTEMPT": "2"}):
                    self.assertTrue(client.check(self.task, "api", "in_progress", None, "Retrying API", "Retrying API"))
                before = len(client.writes)
                with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "400", "GITHUB_RUN_ATTEMPT": "1"}):
                    self.assertFalse(g.sync_preflight(client, self.task, {"api": "success"}))
                    g.finalize_preflight(client, self.task, {"prepare": "success", "basic": "success", "api": "failure"})
                self.assertEqual(len(client.writes), before)
                with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "400", "GITHUB_RUN_ATTEMPT": "2"}):
                    if outcome == "cancelled":
                        g.finalize_preflight(client, self.task, {
                            "prepare": "success", "basic": "success", "api": "cancelled", "security": "skipped", "enqueue": "skipped",
                        })
                        latest = client.commit_statuses(self.head)
                        self.assertEqual(latest[g.CHECK_NAMES["api"]]["state"], "error")
                        self.assertNotIn(g.CHECK_NAMES["security"], latest)
                        self.assertNotIn(g.SUMMARY_CONTEXT, latest)
                    else:
                        for key in ("api", "security"):
                            self.assertTrue(g.sync_preflight(client, self.task, {key: "success"}))
                        for key in ("approve", "dispatch"):
                            self.assertTrue(client.check(self.task, key, "completed", "success", "Passed", "Passed"))
                        dispatch = client.latest_dispatch(self.task)
                        self.assertEqual((dispatch["status"], dispatch["conclusion"]), ("completed", "success"))
                        client.restore_preflight(self.task)

    def test_old_merge_finalize_and_receiver_cannot_change_new_head_statuses(self):
        control = self.store(g.CONTROL_BRANCH)
        g.enqueue(self.task, self.gh, control, self.source)
        client = RecordingGitHub(self.gh)
        client.check_rows[self.tested] = [github_check({
            "name": g.CHECK_NAMES["basic"], "status": "in_progress",
            "external_id": f"triton-anchor-local-ci:basic:{self.task['task_id']}",
            "output": {"summary": "<!-- local-ci-workflow:100 -->"},
        }, 10)]
        for context in (g.CHECK_NAMES["basic"], g.SUMMARY_CONTEXT):
            client.seed_status(self.head, context, "pending", "new-task", "200")
        self.gh.pull["base"]["sha"] = "e" * 40
        self.gh.pull["merge_commit_sha"] = "d" * 40
        with patch.dict(g.os.environ, {"GITHUB_RUN_ID": "100", "GITHUB_RUN_ATTEMPT": "1"}):
            self.assertFalse(client.owns_task(self.task, workflow=True))
            g.finalize_preflight(client, self.task, {"prepare": "success", "basic": "failure"})
            self.assertEqual(g.receive_result(client, str(self.remote), self.task["task_id"]), "obsolete")
        self.assertNotIn(f"statuses/{self.head}", [path for path, _, _ in client.writes])

    def test_superseded_preflight_finishes_only_its_pending_summary(self):
        for state in ("pending", "success"):
            with self.subTest(state=state), patch.dict(g.os.environ, {"GITHUB_RUN_ID": "12345", "GITHUB_RUN_ATTEMPT": "1"}):
                client = RecordingGitHub(self.gh)
                g.begin_checks(client, self.task)
                if state == "success":
                    client.status(self.task, "success", "Local CI: pass")
                with patch.object(g, "is_current", return_value=False):
                    g.finalize_preflight(client, self.task, {"prepare": "success", "basic": "cancelled"})
                self.assertEqual(client.latest_summary(self.task)["state"], "error" if state == "pending" else "success")

    def test_rejected_approval_notifies_contributor_with_commit_and_review(self):
        stages = {key: "success" for key in ("prepare", *g.CHECK_NAMES)}
        stages.update(card="success", approval="failure", enqueue="skipped")
        self.gh.approvals = [{
            "state": "rejected",
            "comment": "Please remove @generated files\n before retrying.",
            "environments": [{"name": "local-ci-fork-approval"}],
            "user": {"login": "maintainer"},
        }]
        with patch.dict(g.os.environ, {
            "GITHUB_RUN_ID": "12345", "GITHUB_REPOSITORY": g.REPOSITORY,
            "GITHUB_SERVER_URL": "https://github.com",
        }), patch.object(self.gh, "latest_summary", return_value={
            "state": "pending", "target_url": "https://github.com/run#local-ci-task=" + self.task["task_id"],
        }):
            g.finalize_preflight(self.gh, self.task, stages)
        self.assertEqual(self.gh.statuses[-1], (self.task["task_id"], "failure"))
        comment = self.gh.comments[0]
        self.assertTrue(comment.startswith("@contributor"))
        self.assertIn(f"PR 提交：`{self.head}`", comment)
        self.assertIn("审核批注：\n\n> Please remove ＠generated files before retrying.", comment)

    def test_closed_pr_finishes_pending_statuses_and_legacy_checks_without_dispatch(self):
        client = RecordingGitHub(self.gh)
        self.gh.pull["state"] = "closed"
        for sha in (self.head, self.tested):
            client.seed_status(sha, g.SUMMARY_CONTEXT, "pending", self.task["task_id"])
        client.seed_status(self.head, g.CHECK_NAMES["basic"], "pending", self.task["task_id"])
        client.seed_status(self.head, g.CHECK_NAMES["api"], "success", self.task["task_id"])
        client.seed_status(self.head, "external/build", "pending", "external")
        client.check_rows[self.tested] = [github_check({
            "name": "local-ci/basic", "status": "queued",
            "external_id": "triton-anchor-ci-v4:basic:old",
        }, 12)]
        self.assertTrue(client.finish_inactive_pr(7))
        self.assertEqual({(path, row.get("context")) for path, _, row in client.writes}, {
            (f"statuses/{self.head}", g.SUMMARY_CONTEXT),
            (f"statuses/{self.tested}", g.SUMMARY_CONTEXT),
            (f"statuses/{self.head}", g.CHECK_NAMES["basic"]),
            ("check-runs/12", None),
        })
        self.assertTrue(all(row["state"] == "error" for _, _, row in client.writes if "state" in row))
        self.assertEqual(client.check_rows[self.tested][0]["conclusion"], "cancelled")
        before = len(client.writes)
        client.finish_inactive_pr(7)
        self.assertEqual(len(client.writes), before)

    def test_optimistic_control_writes_preserve_other_writer(self):
        first, second = self.store(g.CONTROL_BRANCH), self.store(g.CONTROL_BRANCH)
        first.put({"a.json": {"a": 1}})
        second.put({"b.json": {"b": 2}})
        first.refresh()
        self.assertEqual(first.get("a.json"), {"a": 1})
        self.assertEqual(first.get("b.json"), {"b": 2})
        with self.assertRaises(ValueError):
            first.put({"a.json": {"a": 3}}, ("a.json",))

    def test_security_scans_real_git_diff(self):
        (self.source / "unsafe.py").write_text("import socket\nimport os\nos." "system('example')\n")
        git(self.source, "add", ".")
        git(self.source, "commit", "-qm", "unsafe")
        tested = git(self.source, "rev-parse", "HEAD")
        with tempfile.TemporaryDirectory() as output:
            original = Path.cwd()
            try:
                import os

                os.chdir(output)
                self.assertEqual(g.security_diff(self.source, self.base, tested), 1)
            finally:
                os.chdir(original)

    def test_sarif_severity_gate_executes(self):
        folder = self.root / "sarif"
        folder.mkdir()
        document = {
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "rules": [
                                {"id": "r", "properties": {"security-severity": "7.5"}}
                            ]
                        }
                    },
                    "results": [{"ruleId": "r", "level": "warning"}],
                }
            ]
        }
        (folder / "result.sarif").write_text(json.dumps(document))
        self.assertEqual(len(g.sarif_failures(folder)), 1)
        document["runs"][0]["results"] = []
        (folder / "result.sarif").write_text(json.dumps(document))
        self.assertEqual(g.sarif_failures(folder), [])

    def test_production_transport_allowlists(self):
        with self.assertRaises(ValueError):
            g.GitHub("likehupochuan/triton-anchor")
        with self.assertRaises(ValueError):
            g.GitStore("https://github.com/RACE-org/triton-anchor", g.CONTROL_BRANCH)

if __name__ == "__main__":
    unittest.main()
