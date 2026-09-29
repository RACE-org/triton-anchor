"""Diff hints permit lightweight validation while explicit full keeps coverage."""
import unittest

from agent_ci import policy
from tools.basic_tools import runner


class PolicyTests(unittest.TestCase):
    def classify(self, *paths, backend=True, full=False, event_kind="pull_request", **extra):
        return policy.minimum_checks(
            [{"path": path, "status": "M", **extra} for path in paths],
            backend_enabled=backend,
            full=full,
            event_kind=event_kind,
        )

    def test_paths_do_not_force_builds_or_full_without_diff_analysis(self):
        for path in ("README.md", "python/triton_anchor/__init__.py", "csrc/pass.cpp",
                     "scripts/local_ci/agent_ci/worker.py", "unknown.cfg"):
            with self.subTest(path=path):
                selected = self.classify(path)
                self.assertEqual(selected["required_checks"], ["change_validation"])
                self.assertEqual(selected["required_parameters"], {})
        unknown = self.classify("unknown.cfg")
        self.assertEqual(unknown["impact"]["level"], "needs_analysis")
        self.assertNotIn("flaggems", unknown["recommended_checks"])

    def test_only_pr_tasks_require_pr_info_review(self):
        for event_kind in ("pull_request", "push", "manual"):
            selected = self.classify("README.md", event_kind=event_kind)
            expected = ["pr_info", "architecture"] if event_kind == "pull_request" else ["architecture"]
            self.assertEqual(selected["required_reviews"], expected)
            self.assertEqual(selected["required_checks"], ["change_validation"])

    def test_python_checks_are_hints_without_build_or_backend_expansion(self):
        selected = self.classify("python/triton_anchor/__init__.py")
        self.assertIn("frontend_tests", selected["recommended_checks"])
        self.assertNotIn("frontend_build", selected["recommended_checks"])
        self.assertNotIn("flaggems", selected["recommended_checks"])
        # Selecting the installed-package tool still requires its real preparation.
        self.assertEqual(runner.dependencies("frontend_tests"), ["frontend_install"])
        self.assertEqual(runner.dependencies("frontend_install"), ["frontend_build"])

    def test_test_paths_are_suggestions_not_a_mixed_change_coverage_cap(self):
        path = "python/triton_anchor/tests/test_ir.py"
        selected = self.classify(path)
        self.assertEqual(selected["recommended_parameters"]["frontend_tests"], {"paths": [path]})
        for selected in (self.classify(path, status="D"),
                         self.classify("python/triton_anchor/pipeline.py", path)):
            self.assertNotIn("frontend_tests", selected["recommended_parameters"])
        smoke = self.classify("tests/test_smoke.py")
        self.assertIn("frontend_smoke", smoke["recommended_checks"])
        self.assertNotIn("frontend_tests", smoke["recommended_checks"])

    def test_docs_and_control_suggest_lightweight_checks(self):
        for path in ("scripts/local_ci/prepare/README.md", "triton/README.md", "LICENSE", "assets/design.svg"):
            self.assertEqual(self.classify(path)["recommended_checks"], ["control_plane"])
        selected = self.classify("scripts/local_ci/tests/test_worker.py")
        self.assertIn("control_plane", selected["recommended_checks"])
        self.assertNotIn("frontend_build", selected["recommended_checks"])

    def test_compiler_and_interface_hints_keep_relevant_runtime_coverage(self):
        interface = self.classify("python/triton_anchor/hw_capability.py")
        self.assertIn("backend_smoke", interface["recommended_checks"])
        self.assertNotIn("flaggems", interface["recommended_checks"])
        for name in ("llvm-hash.txt", "llvm-info.json", "llvm-info"):
            with self.subTest(name=name):
                llvm = self.classify("triton/cmake/" + name)
                for tool in ("frontend_build", "backend_smoke", "flaggems", "compile_time"):
                    self.assertIn(tool, llvm["recommended_checks"])
                self.assertEqual(llvm["classification_evidence"][0]["categories"], ["llvm"])
        for path in ("triton/cmake/amd-llvm-info.json", "triton/cmake/llvm-build-info.json", "vendor/llvm-info.json"):
            self.assertNotEqual(policy.category(path), "llvm")
        renamed = self.classify("docs/example.md", old_path="csrc/old.cpp", status="R100")
        self.assertIn("compiler", renamed["categories"])
        self.assertIn("backend_smoke", renamed["recommended_checks"])
        without_backend = self.classify("triton/cmake/llvm-hash.txt", backend=False)
        self.assertNotIn("flaggems", without_backend["recommended_checks"])

    def test_explicit_full_requires_flaggems_without_forcing_performance(self):
        selected = self.classify("README.md", full=True)
        self.assertEqual(["change_validation", "flaggems"], selected["required_checks"])
        self.assertEqual(selected["required_parameters"]["flaggems"], {"mode": "full"})
        self.assertEqual(selected["recommended_parameters"], {})
        for tool in ("compile_time", "pass_profile", "ir_serialization"):
            self.assertNotIn(tool, selected["required_checks"])
        without = self.classify("README.md", full=True, backend=False)
        self.assertIn("flaggems", without["required_checks"])
        self.assertEqual(without["required_parameters"]["flaggems"], {"mode": "full"})
        self.assertIn("flaggems", without["not_applicable"])

    def test_empty_diff_is_not_implicitly_documentation(self):
        with self.assertRaises(policy.ContractError):
            policy.minimum_checks([], backend_enabled=True)
