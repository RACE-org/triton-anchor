from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scan_pr_security.py"
SPEC = importlib.util.spec_from_file_location("scan_pr_security", SCRIPT)
assert SPEC and SPEC.loader
security = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = security
SPEC.loader.exec_module(security)


def pr_file(filename: str, added_lines: list[str], status: str = "modified") -> dict:
    patch = [f"@@ -0,0 +1,{len(added_lines)} @@"]
    patch.extend(f"+{line}" for line in added_lines)
    return {"filename": filename, "patch": "\n".join(patch), "status": status}


def messages(findings: list[object]) -> list[str]:
    return [finding.message for finding in findings]


class ScanSecurityTests(unittest.TestCase):
    def test_github_actions_changes_remain_blocked(self) -> None:
        blocking, _ = security.scan(
            [pr_file(".github/actions/AGENTS.md", ["# Development notes"])]
        )

        self.assertIn(security.PROTECTED_PATH_MESSAGE, messages(blocking))

    def test_harmless_setup_change_warns_without_blocking(self) -> None:
        blocking, warnings = security.scan(
            [
                pr_file(
                    "setup.py",
                    ['package_data={"triton": ["include/**/*.td"]}'],
                )
            ]
        )

        self.assertEqual(blocking, [])
        self.assertIn(security.DEPENDENCY_REVIEW_MESSAGE, messages(warnings))

    def test_setup_network_access_still_blocks(self) -> None:
        blocking, warnings = security.scan(
            [pr_file("setup.py", ["import requests", 'requests.get("url")'])]
        )

        self.assertIn("new Python network module import", messages(blocking))
        self.assertIn("new Python network request", messages(blocking))
        self.assertIn(security.DEPENDENCY_REVIEW_MESSAGE, messages(warnings))

    def test_setup_remote_git_operation_still_blocks(self) -> None:
        blocking, _ = security.scan(
            [pr_file("setup.py", ['command = "git clone https://example.test/repo"'])]
        )

        self.assertIn("new remote Git operation", messages(blocking))

    def test_requirements_custom_sources_still_block(self) -> None:
        cases = [
            (
                "pkg @ https://example.test/pkg.whl",
                "new direct URL or VCS dependency source",
            ),
            (
                "--extra-index-url https://example.test/simple",
                "new custom Python package index or link source",
            ),
            (
                "git+https://example.test/pkg.git",
                "new direct URL or VCS dependency source",
            ),
        ]

        for line, expected_message in cases:
            with self.subTest(line=line):
                blocking, warnings = security.scan(
                    [pr_file("requirements-dev.txt", [line])]
                )
                self.assertIn(expected_message, messages(blocking))
                self.assertIn(security.DEPENDENCY_REVIEW_MESSAGE, messages(warnings))

    def test_dependency_control_file_without_text_patch_blocks(self) -> None:
        blocking, warnings = security.scan(
            [{"filename": "requirements.txt", "status": "modified"}]
        )

        self.assertIn(security.UNSCANNED_PATCH_MESSAGE, messages(blocking))
        self.assertEqual(warnings, [])

    def test_missing_patch_is_restored_and_scanned(self) -> None:
        base_sha = "1" * 40
        head_sha = "2" * 40
        tested_sha = "3" * 40
        local_patch = "@@ -0,0 +1 @@\n+import requests"

        with mock.patch.object(
            security,
            "git_output",
            side_effect=[tested_sha, f"{base_sha} {head_sha}", local_patch],
        ):
            files = security.restore_missing_patches(
                [{"filename": "large.py", "status": "added"}],
                Path("/tmp/security-candidate"),
                base_sha,
                head_sha,
                tested_sha,
            )

        blocking, _ = security.scan(files)
        self.assertIn("new Python network module import", messages(blocking))

    def test_diff_fallback_rejects_wrong_merge_parents(self) -> None:
        base_sha = "1" * 40
        head_sha = "2" * 40
        tested_sha = "3" * 40

        with mock.patch.object(
            security,
            "git_output",
            side_effect=[tested_sha, f"{'4' * 40} {head_sha}"],
        ):
            with self.assertRaisesRegex(RuntimeError, "merge parents"):
                security.restore_missing_patches(
                    [{"filename": "large.cpp", "status": "added"}],
                    Path("/tmp/security-candidate"),
                    base_sha,
                    head_sha,
                    tested_sha,
                )


if __name__ == "__main__":
    unittest.main()
