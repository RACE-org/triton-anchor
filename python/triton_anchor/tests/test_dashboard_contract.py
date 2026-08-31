from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "dashboard" / "data"
sys.path.insert(0, str(ROOT / "scripts" / "dashboard"))
import build_mock_full_test as DEMO  # noqa: E402


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class DashboardContractTest(unittest.TestCase):
    def test_manifest_sources_exist(self):
        manifest = read_json(DATA_DIR / "manifest.json")
        self.assertEqual(manifest["schema"], "triton-anchor-dashboard-manifest/v1")
        for relative_path in manifest["sources"].values():
            self.assertTrue((DATA_DIR / relative_path).is_file(), relative_path)
        for relative_path in manifest["downloads"].values():
            self.assertTrue((DATA_DIR / relative_path).is_file(), relative_path)

    def test_visible_backends_are_configured_without_removing_backend_data(self):
        manifest = read_json(DATA_DIR / "manifest.json")
        document = read_json(DATA_DIR / "backend-status.json")
        backend_ids = {row["id"] for row in document["backends"]}
        visible_ids = manifest["display"]["backend_ids"]
        self.assertEqual(visible_ids, ["sophgo-cmodel"])
        self.assertTrue(set(visible_ids).issubset(backend_ids))

    def test_full_test_operators_have_stable_shape(self):
        document = read_json(DATA_DIR / "full-test.json")
        self.assertEqual(document["schema"], "triton-anchor-full-test/v1")
        self.assertGreaterEqual(len(document["operators"]), 100)
        names = set()
        for row in document["operators"]:
            self.assertIn(row["status"], {"passed", "failed", "timeout", "unknown"})
            self.assertTrue(row["name"])
            self.assertNotIn(row["name"], names)
            names.add(row["name"])

    def test_full_test_demo_matches_latest_historical_run(self):
        document = read_json(DATA_DIR / "full-test.json")
        self.assertEqual(document["data_mode"], "mock")
        self.assertEqual(
            document["run"]["sha"],
            "demo3d4c586307dcc3c1f11e650c67529b85da3dd22f",
        )
        self.assertEqual(
            document["source_summary"],
            {
                "total": 127,
                "passed": 69,
                "failed": 14,
                "timed_out": 44,
                "status": "fail",
            },
        )
        counts = {
            status: sum(1 for row in document["operators"] if row["status"] == status)
            for status in ("passed", "failed", "timeout")
        }
        self.assertEqual(counts, {"passed": 69, "failed": 14, "timeout": 44})

        with (DATA_DIR / "full-test.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 127)

    def test_demo_builder_uses_latest_historical_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "flaggems.csv"
            output_json = root / "full-test.json"
            output_csv = root / "full-test.csv"
            source.write_text(
                "序号,算子名称,测试状态,失败阶段,耗时(ms),测试时间\n"
                "1,abs,通过,,10,10:00:00\n"
                "2,erf,失败,测试执行失败,20,10:01:00\n"
                "3,flip,超时,超时,30,10:02:00\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                sys,
                "argv",
                [
                    "build_mock_full_test.py",
                    "--input-csv",
                    str(source),
                    "--output-json",
                    str(output_json),
                    "--output-csv",
                    str(output_csv),
                ],
            ):
                self.assertEqual(DEMO.main(), 0)

            document = read_json(output_json)
            self.assertEqual(document["run"]["sha"], DEMO.DEFAULT_DEMO_SHA)
            self.assertEqual(
                document["source_summary"],
                {
                    "total": 3,
                    "passed": 1,
                    "failed": 1,
                    "timed_out": 1,
                    "status": "fail",
                },
            )

    def test_backend_statuses_are_unique(self):
        document = read_json(DATA_DIR / "backend-status.json")
        self.assertEqual(document["schema"], "triton-anchor-backend-status-list/v1")
        backend_ids = [row["id"] for row in document["backends"]]
        self.assertEqual(len(backend_ids), len(set(backend_ids)))
        self.assertIn("sophgo-cmodel", backend_ids)
        for row in document["backends"]:
            self.assertIn(
                row["state"],
                {"success", "warning", "failure", "pending", "stale", "unknown"},
            )

    def test_performance_contract_contains_required_sections(self):
        document = read_json(DATA_DIR / "performance.json")
        self.assertEqual(document["schema"], "triton-anchor-performance-summary/v1")
        sections = (
            document["compile_time"]["kernels"],
            document["pass_profile"]["hotspots"],
            document["ir_serialization"]["metrics"],
        )
        for rows in sections:
            self.assertIsInstance(rows, list)
        if document.get("data_mode") == "mock":
            self.assertTrue(all(sections))
        else:
            self.assertTrue(any(sections))

    def test_site_entrypoints_exist(self):
        for relative_path in (
            "dashboard/index.html",
            "dashboard/styles.css",
            "dashboard/xlsx.js",
            "dashboard/app.js",
        ):
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_worker_health_resource_display_contract(self):
        app = (ROOT / "dashboard" / "app.js").read_text(encoding="utf-8")
        health = (
            ROOT / "scripts" / "local_ci" / "maintenance" / "local_ci_health.py"
        ).read_text(encoding="utf-8")
        fixture = read_json(DATA_DIR / "worker-health.json")

        self.assertIn("const usedCpus = rawPercent / 100;", app)
        self.assertIn("(usedCpus / availableCpus) * 100", app)
        self.assertIn("container.available_cpus ?? limits.cpus", app)
        self.assertIn('value: memoryUsage,', app)
        self.assertIn('detail: "实际使用 / 可用内存"', app)
        self.assertIn('"available_cpus": round(available_cpus, 3)', health)
        self.assertIsNone(fixture["container"]["available_cpus"])


if __name__ == "__main__":
    unittest.main()
