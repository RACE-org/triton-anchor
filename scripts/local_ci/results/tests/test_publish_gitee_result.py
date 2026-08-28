from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "local_ci" / "results"))

import publish_gitee_result as publisher


class PublishedArtifactTests(unittest.TestCase):
    def test_publish_budget_rejects_a_file_that_would_exceed_the_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            destination = root / "destination.bin"
            source.write_bytes(b"x" * 11)
            budget = publisher.PublishBudget(10)

            with self.assertRaises(publisher.PublishBudgetExceeded):
                budget.copy(source, destination)

            self.assertFalse(destination.exists())

    def test_size_limit_result_marks_the_host_run_and_publishes_only_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            target_dir = root / "published"
            run_dir.mkdir()
            args = SimpleNamespace(
                exit_code=88,
                source_branch="ci/push/CI_dev",
                sha="a" * 40,
                run_id="run-1",
                context="local-ci/test",
                max_publish_bytes=10,
            )

            publisher.write_size_limit_result(
                run_dir, target_dir, args, Path("runs/ci_push/run-1")
            )

            marker = json.loads(
                (run_dir / "gitee-result-size-limit.json").read_text(encoding="utf-8")
            )
            result = json.loads(
                (target_dir / "result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["failure_code"], "gitee_result_size_limit")
            self.assertEqual(result["status"], 88)

    def test_candidate_result_does_not_duplicate_base_performance_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            artifact_dir = root / "artifacts"
            target_dir = root / "published"
            run_dir.mkdir()
            artifact_dir.mkdir()
            (run_dir / "local-ci.log").write_text(
                f"Artifact dir: {artifact_dir}\n", encoding="utf-8"
            )
            (artifact_dir / "delivery-summary.txt").write_text(
                "schema: triton-anchor-local-ci/v2\nstatus: 0\n",
                encoding="utf-8",
            )
            (artifact_dir / "compile-benchmark.json").write_text("{}\n")
            for name in (
                "compile-benchmark-base.json",
                "pass-profile-base.json",
                "ir-serialization-base.json",
            ):
                (artifact_dir / name).write_text("{}\n")

            args = SimpleNamespace(
                exit_code=0,
                source_branch="ci/pr-42/feature/demo",
                sha="a" * 40,
                run_id="run-1",
                context="local-ci/test",
            )
            published = publisher.copy_results(
                run_dir,
                target_dir,
                args,
                Path("runs/ci_pr/ci_pr-42_feature_demo") / args.sha / args.run_id,
                publisher.PublishBudget(1024 * 1024),
            )

            self.assertEqual(published, target_dir)
            self.assertTrue((target_dir / "compile-benchmark.json").is_file())
            for name in (
                "compile-benchmark-base.json",
                "pass-profile-base.json",
                "ir-serialization-base.json",
            ):
                self.assertFalse((target_dir / name).exists())
            manifest = json.loads(
                (target_dir / "publish-manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(
                any(name.endswith("-base.json") for name in manifest["copied_files"])
            )

    def test_frontend_profile_fallback_preserves_profile_and_skip_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            target_dir = root / "published"
            run_dir.mkdir()
            (run_dir / "local-ci.log").write_text(
                "Local CI failed before creating an artifact directory.\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                exit_code=1,
                source_branch="ci/pr-42/feature/demo",
                sha="a" * 40,
                run_id="run-1",
                context="local-ci/test",
                execution_mode="full",
                ci_profile="triton-3.3-frontend",
                llvm_hash="a66376b0dc3b2ea8a84fda26faca287980986f78",
                backend_stages_enabled="false",
                backend_skip_reason=(
                    "当前没有部署可供测试的厂商后端，未执行后端构建、JIT、"
                    "FlagGems 和性能验证。"
                ),
            )

            publisher.write_fallback_results(
                run_dir,
                target_dir,
                args,
                Path("runs/ci_pr/ci_pr-42_feature_demo") / args.sha / args.run_id,
                publisher.PublishBudget(1024 * 1024),
            )
            summary = (target_dir / "delivery-summary.txt").read_text(
                encoding="utf-8"
            )

            self.assertIn("execution_mode: full", summary)
            self.assertIn("ci_profile: triton-3.3-frontend", summary)
            self.assertIn(f"llvm_hash: {args.llvm_hash}", summary)
            self.assertIn("backend_stages_enabled: false", summary)
            self.assertIn(f"backend_skip_reason: {args.backend_skip_reason}", summary)
            self.assertIn("frontend_build_status: unavailable", summary)
            self.assertIn("frontend_smoke_status: unavailable", summary)
            for stage in (
                "backend_rebuild",
                "backend_smoke_jit",
                "flaggems",
                "compile_time",
                "pass_profile",
                "ir_serialization",
            ):
                self.assertIn(f"{stage}_status: skipped", summary)


if __name__ == "__main__":
    unittest.main()
