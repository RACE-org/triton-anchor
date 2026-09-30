import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/basic_tools/flaggems"))
import batch_test_flaggems as batch
from select_flaggems_tests import select_entries


class ReadOnlyFlagGemsTests(unittest.TestCase):
    def test_test_paths_resolve_from_mount_not_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tests").mkdir()
            test = root / "tests/test_add.py"
            test.touch()
            selected = batch.SelectedOperator(
                "math", "add", "add", ("tests/test_add.py::test_add",)
            )
            command = batch.build_pytest_command(selected, sys.executable, "-q", root)
            self.assertIn(str(test) + "::test_add", command)
            empty = batch.SelectedOperator("math", "add", "add", ())
            self.assertIn(
                str(root / "tests"),
                batch.build_pytest_command(empty, sys.executable, "-q", root),
            )
            escaped = batch.SelectedOperator("math", "add", "add", ("../other.py",))
            with self.assertRaises(ValueError):
                batch.build_pytest_command(escaped, sys.executable, "-q", root)

    @unittest.skipUnless(
        importlib.util.find_spec("pytest"), "pytest needed for subprocess integration"
    )
    def test_pytest_writes_report_and_caches_outside_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            tests = source / "tests"
            tests.mkdir(parents=True)
            (tests / "test_add.py").write_text(
                "import json,os\nfrom pathlib import Path\n"
                "def test_add():\n"
                "    cwd=Path.cwd()\n"
                "    assert not cwd.is_relative_to(Path(os.environ['FLAGGEMS_ROOT']))\n"
                "    assert os.environ['PYTHONDONTWRITEBYTECODE']=='1'\n"
                "    assert Path(os.environ['FLAGGEMS_CACHE_DIR']).is_relative_to(cwd)\n"
                "    Path('result.json').write_text(json.dumps({'passed': True}))\n"
            )
            before = sorted(str(path.relative_to(source)) for path in source.rglob("*"))
            logs, dump = root / "logs", root / "dump"
            logs.mkdir()
            dump.mkdir()
            args = SimpleNamespace(
                python_bin=sys.executable,
                pytest_args="-q",
                mode="single",
                clear_cache="0",
                total_timeout_seconds=30,
                full_hard_timeout_seconds=30,
                idle_timeout_seconds=30,
            )
            selected = batch.SelectedOperator("math", "add", "", ("tests/test_add.py",))
            result = batch.run_operator(selected, 1, args, source, dump, logs)
            self.assertEqual(0, result.exit_code, (logs / "001-add.log").read_text())
            self.assertEqual(1, result.passed)
            self.assertTrue(
                json.loads((logs / "001-add-work/result.json").read_text())["passed"]
            )
            self.assertEqual(
                before,
                sorted(str(path.relative_to(source)) for path in source.rglob("*")),
            )


class OperatorSelectionTests(unittest.TestCase):
    def test_impact_includes_explicit_operator_outside_pass_whitelist(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tests").mkdir()
            (root / "tests" / "test_ops.py").write_text(
                "@pytest.mark.abs\ndef test_abs(): pass\n@pytest.mark.gelu\ndef test_gelu(): pass\n"
            )
            (root / "pass.tsv").write_text("unary abs abs\n")
            (root / "all.tsv").write_text("unary abs abs\nunary gelu gelu\n")
            args = argparse.Namespace(
                mode="impact",
                ops="gelu",
                categories="",
                flaggems_dir=str(root),
                whitelist=str(root / "pass.tsv"),
                full_list=str(root / "all.tsv"),
            )
            self.assertEqual([entry.op for entry in select_entries(args)], ["gelu"])
            args.ops = "not_an_operator"
            with self.assertRaisesRegex(ValueError, "Unknown FlagGems"):
                select_entries(args)
            args.ops = ""
            args.categories = "unary"
            self.assertEqual([entry.op for entry in select_entries(args)], ["abs", "gelu"])

    def test_empty_impact_uses_six_representative_supported_operators(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tests").mkdir()
            operators = ["abs", "maximum", "mm", "arange", "exponential_", "embedding"]
            (root / "tests/test_ops.py").write_text(
                "\n".join(f"@pytest.mark.{op}\ndef test_{op}(): pass" for op in operators)
            )
            catalogue = ROOT / "tools/basic_tools/flaggems"
            args = argparse.Namespace(
                mode="impact", ops="", categories="", flaggems_dir=str(root),
                whitelist=str(catalogue / "flaggems_pass_whitelist.tsv"),
                full_list=str(catalogue / "flaggems_all_ops.tsv"),
            )
            selected = select_entries(args)
            self.assertEqual(set(operators), {entry.op for entry in selected})
            self.assertEqual(6, len({entry.category for entry in selected}))

    def test_nonfull_limit_applies_after_expansion_and_counts_distinct_operators(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tests").mkdir()
            (root / "tests/test_ops.py").write_text(
                "\n".join(f"@pytest.mark.mark{i}\ndef test_{i}(): pass" for i in range(8))
            )
            (root / "tests/test_more.py").write_text("@pytest.mark.mark0\ndef test_more(): pass\n")
            catalogue = root / "ops.tsv"
            catalogue.write_text("\n".join(f"math op{i} mark{i}" for i in range(8)))
            args = argparse.Namespace(
                mode="impact", ops="", categories="math", flaggems_dir=str(root),
                whitelist=str(catalogue), full_list=str(catalogue), sample_size=6, seed="ci",
            )
            with self.assertRaisesRegex(ValueError, "8 operators.*--mode full"):
                select_entries(args)
            args.categories = ""
            args.ops = ",".join(f"op{i}" for i in range(7))
            with self.assertRaisesRegex(ValueError, "7 operators.*--mode full"):
                select_entries(args)
            args.ops = ",".join(f"op{i}" for i in range(6)) + ",op0,mark0"
            selected = select_entries(args)
            self.assertEqual(6, len({entry.op for entry in selected}))
            self.assertEqual(7, len(selected))  # Two test files still represent one operator.
            args.mode = "full"
            self.assertEqual(8, len({entry.op for entry in select_entries(args)}))
            args.mode = "sample"
            catalogue.write_text("\n".join(f"category{i} op{i} mark{i}" for i in range(8)))
            self.assertEqual(6, len({entry.op for entry in select_entries(args)}))
            args.sample_size = 7
            with self.assertRaisesRegex(ValueError, "--mode full"):
                select_entries(args)
