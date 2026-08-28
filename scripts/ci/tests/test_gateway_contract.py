from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
GATEWAY = ROOT / ".github/workflows/ci-gateway.yml"
MANIFEST = ROOT / ".github/ci-gateway-manifest.json"

CONTRACT_INPUTS = {
    "gateway_contract_version",
    "mode",
    "pr_number",
    "expected_head_sha",
    "comparison_base_sha",
    "tested_sha",
    "requested_sha",
    "worker_revision_sha",
    "authorization_source",
    "source_branch",
    "target_branch",
    "task_ref",
    "context",
    "attempt",
    "started_at",
    "wait_seconds",
    "cancellation_reason",
    "delete_task_ref",
    "run_title",
}


def workflow_dispatch_inputs(text: str) -> set[str]:
    lines = text.splitlines()
    start = lines.index("  workflow_dispatch:")
    inputs_line = lines.index("    inputs:", start)
    names: set[str] = set()
    for line in lines[inputs_line + 1 :]:
        if line and not line.startswith("      "):
            break
        if line.startswith("      ") and not line.startswith("        "):
            names.add(line.strip().removesuffix(":"))
    return names


def workflow_step_script(text: str, name: str) -> str:
    marker = f"      - name: {name}\n"
    start = text.index(marker)
    body_start = text.index("        run: |\n", start) + len("        run: |\n")
    body_end = text.index("\n      - name:", body_start)
    return "\n".join(
        line[10:] if line.startswith("          ") else line
        for line in text[body_start:body_end].splitlines()
    )


class GatewayV3ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gateway = GATEWAY.read_text(encoding="utf-8")
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.security = (ROOT / ".github/workflows/security-gate.yml").read_text()
        cls.dispatcher = (ROOT / ".github/workflows/dispatch-local-ci.yml").read_text()
        cls.receiver = (ROOT / ".github/workflows/receive-local-ci-result.yml").read_text()
        cls.pages = (ROOT / ".github/workflows/backend-status-pages.yml").read_text()
        cls.poller = (ROOT / "scripts/local_ci/poll_gitee_and_run.sh").read_text()

    def test_contract_v3_interface(self) -> None:
        self.assertEqual(workflow_dispatch_inputs(self.gateway), CONTRACT_INPUTS)
        self.assertIn('GATEWAY_CONTRACT_VERSION: "3"', self.gateway)
        self.assertNotIn("expected_base_sha", self.gateway)
        self.assertNotIn("inputs.sha", self.gateway)

    def test_manifest_describes_merge_result_worker(self) -> None:
        self.assertEqual(self.manifest["schema_version"], 1)
        self.assertEqual(self.manifest["gateway_contract_version"], "3")
        self.assertEqual(self.manifest["role"], "worker")
        self.assertEqual(self.manifest["tested_revision"], "merge-result")
        self.assertTrue(
            {
                "security-scan", "codeql", "dispatch", "receive", "pages",
                "cancel", "cross-branch-pr", "cross-branch-push", "merge-result",
            }.issubset(self.manifest["capabilities"])
        )

    def test_merge_result_is_frozen_and_revalidated(self) -> None:
        self.assertGreaterEqual(self.gateway.count("pull/${prNumber}/merge"), 3)
        self.assertIn("parents[0]", self.gateway)
        self.assertIn("parents[1]", self.gateway)
        self.assertIn("tested_sha: process.env.TESTED_SHA", self.gateway)
        self.assertIn("TESTED_SHA_KIND=\"pr_merge\"", self.dispatcher)
        self.assertIn("refs/pull/${PR_NUMBER}/merge", self.dispatcher)

    def test_dispatch_metadata_v2_records_text_truncation(self) -> None:
        self.assertIn(
            "title_truncated:(($title|length)>500)", self.dispatcher
        )
        self.assertIn(
            "description_truncated:(($description|length)>8000)",
            self.dispatcher,
        )

    def test_dispatch_classifies_only_existing_codex_docs_paths(self) -> None:
        script = workflow_step_script(self.dispatcher, "Classify PR execution mode")
        cases = (
            ("docs/guide.md", "codex_only"),
            ("README.md", "codex_only"),
            ("python/triton_anchor/example.py", "full"),
            (".github/workflows/ci.yml", "full"),
            ("scripts/local_ci/codex_ai/prompts/review.md", "full"),
            ("python/old.py\ndocs/new.md", "full"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_git = root / "git"
            fake_git.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$CHANGED_PATHS" | tr "\\n" "\\0"\n',
                encoding="utf-8",
            )
            fake_git.chmod(0o700)
            for changed_paths, expected in cases:
                with self.subTest(changed_paths=changed_paths):
                    output = root / "github-output"
                    output.unlink(missing_ok=True)
                    env = os.environ.copy()
                    env.update(
                        {
                            "BASE_SHA": "a" * 40,
                            "HEAD_SHA": "b" * 40,
                            "CHANGED_PATHS": changed_paths,
                            "GITHUB_OUTPUT": str(output),
                            "PATH": f"{root}:{env['PATH']}",
                        }
                    )
                    subprocess.run(["bash", "-c", script], env=env, check=True)
                    self.assertEqual(
                        output.read_text(encoding="utf-8").strip(),
                        f"execution_mode={expected}",
                    )

    def test_receiver_reports_explicitly_skipped_stages_as_success(self) -> None:
        script = workflow_step_script(self.receiver, "Report ${{ matrix.name }}")
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.update(
                {
                    "STAGE_KEY": "frontend_build",
                    "STAGE_NAME": "Frontend build",
                    "REQUIRED_STAGE": "true",
                    "STAGE_RESULTS": json.dumps({"frontend_build": "skipped"}),
                    "TARGET_URL": "",
                    "GITHUB_STEP_SUMMARY": str(Path(directory) / "summary.md"),
                }
            )
            result = subprocess.run(["bash", "-c", script], env=env, check=False)
        self.assertEqual(result.returncode, 0)

    def test_poller_routes_codex_only_without_deterministic_ci(self) -> None:
        self.assertIn('execution_mode="full"', self.poller)
        self.assertIn('"${execution_mode}" == "codex_only"', self.poller)
        self.assertIn("frontend_build_status: skipped", self.poller)
        self.assertIn('.get("execution_mode", "full")', self.poller)
        codex_at = self.poller.rindex("run_codex_ai_ci_for_run")
        artifact_at = self.poller.index(
            'echo "Artifact dir: ${nonexecuted_artifact_dir}"', codex_at
        )
        publish_at = self.poller.index('publish_result "${sha}"', artifact_at)
        self.assertLess(codex_at, artifact_at)
        self.assertLess(artifact_at, publish_at)

    def test_external_fork_requires_live_maintainer_authorization(self) -> None:
        self.assertIn("getCollaboratorPermissionLevel", self.gateway)
        self.assertIn("write', 'maintain', 'admin", self.gateway)
        self.assertIn("approve-external-fork:", self.gateway)
        self.assertIn("local-ci-fork-approval", self.gateway)
        self.assertIn("external-fork-environment", self.gateway)
        self.assertIn("manual-maintainer:", self.gateway)
        self.assertIn("pull.head.sha !== process.env.EXPECTED_HEAD_SHA", self.gateway)

    def test_merge_result_base_comes_from_first_parent(self) -> None:
        self.assertIn("const comparisonBaseSha = parents[0]", self.gateway)
        self.assertIn("core.setOutput('base_sha', comparisonBaseSha)", self.gateway)
        self.assertIn(
            "h:${pull.head.sha.slice(0, 7)} m:${testedSha.slice(0, 7)} | dispatch",
            self.gateway,
        )
        self.assertNotIn(
            "parents[0].toLowerCase() !== pull.base.sha.toLowerCase()",
            self.gateway,
        )

    def test_security_gate_is_reusable_and_blocks_dispatch(self) -> None:
        self.assertIn("workflow_call:", self.security)
        self.assertNotIn("pull_request_target:", self.security)
        self.assertIn("trusted_ref:", self.security)
        self.assertIn("CodeQL", self.security)
        self.assertNotIn("authorize-local-ci", self.security)
        security_at = self.gateway.index("\n  security-gate:")
        dispatch_at = self.gateway.index("\n  dispatch:", security_at)
        self.assertLess(security_at, dispatch_at)
        self.assertIn("- security-gate", self.gateway[dispatch_at:])
        self.assertNotIn("workflow_run:", self.dispatcher)

    def test_fallback_is_only_for_missing_manifest(self) -> None:
        self.assertIn("let worker = await inspectWorker(pull.base.ref, true)", self.gateway)
        self.assertIn("if (worker === null)", self.gateway)
        self.assertIn("invalid JSON", self.gateway)
        self.assertIn("incompatible manifest", self.gateway)

    def test_fallback_switches_default_to_enabled(self) -> None:
        self.assertIn(
            "FALLBACK_PR_ENABLED: ${{ vars.LOCAL_CI_FALLBACK_PR_ENABLED || 'true' }}",
            self.gateway,
        )
        self.assertIn(
            "FALLBACK_PUSH_ENABLED: ${{ vars.LOCAL_CI_FALLBACK_PUSH_ENABLED || 'true' }}",
            self.gateway,
        )
        self.assertIn("PR fallback is disabled", self.gateway)
        self.assertIn("Cross-branch push fallback is disabled", self.gateway)

    def test_manual_push_and_receiver_use_explicit_sha_fields(self) -> None:
        self.assertIn("REQUESTED_SHA: ${{ inputs.requested_sha }}", self.gateway)
        self.assertIn("TESTED_SHA: ${{ inputs.tested_sha }}", self.gateway)
        pr_match_at = self.gateway.index("if (prMatch) {")
        distinct_sha_at = self.gateway.index(
            "PR receiver must distinguish head SHA from merge-result SHA"
        )
        self.assertLess(pr_match_at, distinct_sha_at)
        self.assertIn("mode=receive", self.dispatcher)
        self.assertIn("--status-sha", self.receiver)
        self.assertIn("--comparison-base-sha", self.receiver)

    def test_direct_push_dispatch_title_omits_full_sha(self) -> None:
        run_name = self.dispatcher.splitlines()[1]
        self.assertIn("format('Push {0} | dispatch'", run_name)
        self.assertNotIn("inputs.commit_sha || github.sha", run_name)

    def test_required_statuses_target_the_tested_revision(self) -> None:
        self.assertIn("`${process.env.STATUS_CONTEXT}/routing`", self.gateway)
        self.assertIn("sha: process.env.TESTED_SHA", self.gateway)
        self.assertNotIn(
            "sha: process.env.EXPECTED_HEAD_SHA || process.env.TESTED_SHA",
            self.gateway,
        )
        self.assertGreaterEqual(
            self.dispatcher.count(
                "STATUS_SHA: ${{ steps.meta.outputs.tested_sha }}"
            ),
            2,
        )
        self.assertNotIn(
            "STATUS_SHA: ${{ steps.meta.outputs.head_sha }}", self.dispatcher
        )
        self.assertIn('--status-sha "${TESTED_SHA}"', self.receiver)
        self.assertNotIn("EXPECTED_HEAD_SHA:-${TESTED_SHA}", self.receiver)

    def test_pages_are_branch_isolated(self) -> None:
        guard = "github.ref_name == (vars.LOCAL_CI_PAGES_BRANCH || 'CI_dev')"
        self.assertIn(guard, self.pages)
        self.assertIn("Cross-branch result only updates commit status", self.receiver)

    def test_pages_use_gitee_username_for_authentication(self) -> None:
        self.assertIn(
            "GITEE_RESULTS_OWNER: ${{ vars.GITEE_RESULTS_OWNER || vars.GITEE_OWNER || 'race-org' }}",
            self.pages,
        )
        self.assertIn(
            "GITEE_USERNAME: ${{ vars.GITEE_USERNAME || 'likehupochuan' }}",
            self.pages,
        )
        self.assertIn(
            '*Username*) printf \'%s\\n\' "${GITEE_USERNAME}"', self.pages
        )
        self.assertNotIn(
            '*Username*) printf \'%s\\n\' "${GITEE_RESULTS_OWNER}"', self.pages
        )

    def test_cancellation_removes_every_pr_ref(self) -> None:
        for prefix in ("ci/base/", "ci/head/", "ci/meta/"):
            self.assertIn(prefix, self.gateway)

    def test_api_compatible_resolves_old_comment(self) -> None:
        notify = (ROOT / ".github/workflows/api-breaking-notify.yml").read_text()
        self.assertIn("Resolved: the latest public API compatibility result is compatible.", notify)


if __name__ == "__main__":
    unittest.main()
