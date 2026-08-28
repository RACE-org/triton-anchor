from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OUTPUT_LIMITS = load_module("local_ci_output_limits", ROOT / "shared/output_limits.py")


def test_scan_tree_reports_log_file_and_total_limits(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "stage.log").write_bytes(b"x" * 11)
    (root / "wheel.bin").write_bytes(b"y" * 21)

    total, violations = OUTPUT_LIMITS.scan_tree(
        root, max_log_bytes=10, max_file_bytes=20, max_total_bytes=30
    )

    assert total == 32
    assert {item.kind for item in violations} == {
        "log_size_limit",
        "artifact_file_size_limit",
        "artifact_size_limit",
    }


def test_scan_tree_accepts_values_exactly_at_each_limit(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "stage.log").write_bytes(b"x" * 10)
    (root / "payload.bin").write_bytes(b"y" * 20)

    total, violations = OUTPUT_LIMITS.scan_tree(
        root, max_log_bytes=10, max_file_bytes=20, max_total_bytes=30
    )

    assert total == 30
    assert violations == []


def test_compact_requires_marker_and_keeps_only_report(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    report = tmp_path / "report.json"
    report.write_text('{"failure_code":"artifact_size_limit"}\n', encoding="utf-8")
    (root / OUTPUT_LIMITS.ARTIFACT_MARKER).touch()
    (root / "large.bin").write_bytes(b"x" * 100)
    nested = root / "nested"
    nested.mkdir()
    (nested / "data").write_text("data", encoding="utf-8")

    OUTPUT_LIMITS.compact_artifact_root(root, report)

    assert sorted(path.name for path in root.iterdir()) == [
        OUTPUT_LIMITS.ARTIFACT_MARKER,
        "output-limit.json",
    ]


def test_compact_run_root_accepts_report_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / OUTPUT_LIMITS.RUN_MARKER).touch()
    report = root / "output-limit.json"
    report.write_text('{"failure_code":"artifact_size_limit"}\n', encoding="utf-8")
    (root / "oversized.bin").write_bytes(b"x" * 100)

    OUTPUT_LIMITS.compact_artifact_root(root, report)

    assert sorted(path.name for path in root.iterdir()) == [
        OUTPUT_LIMITS.RUN_MARKER,
        "output-limit.json",
    ]
    assert json.loads(report.read_text(encoding="utf-8"))["failure_code"] == (
        "artifact_size_limit"
    )


def test_capped_tee_marks_and_stops_at_limit(tmp_path: Path) -> None:
    output = tmp_path / "task.log"
    marker = tmp_path / "limit.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "shared/capped_tee.py"),
            "--output",
            str(output),
            "--max-bytes",
            "10",
            "--marker",
            str(marker),
        ],
        input=b"0123456789overflow",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 86
    assert output.stat().st_size == 10
    assert output.read_bytes().startswith(b"0123456789")
    assert json.loads(marker.read_text(encoding="utf-8"))["failure_code"] == "log_size_limit"
