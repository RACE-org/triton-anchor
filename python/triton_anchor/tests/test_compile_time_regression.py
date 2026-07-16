import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "local_ci" / "compare_compile_time.py"
SPEC = importlib.util.spec_from_file_location("compare_compile_time", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

PUBLISH_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "local_ci" / "publish_gitee_result.py"
PUBLISH_SPEC = importlib.util.spec_from_file_location("publish_gitee_result", PUBLISH_SCRIPT)
PUBLISH_MODULE = importlib.util.module_from_spec(PUBLISH_SPEC)
assert PUBLISH_SPEC and PUBLISH_SPEC.loader
PUBLISH_SPEC.loader.exec_module(PUBLISH_MODULE)


def benchmark_document(values):
    return {
        "summary": {
            kernel: {"compile_est": {"median_ms": value}}
            for kernel, value in values.items()
        }
    }


def test_compile_time_comparison_passes_within_threshold():
    kernels = ["add", "mm", "softmax", "layernorm"]
    baseline = benchmark_document({kernel: 100.0 for kernel in kernels})
    candidate = benchmark_document({kernel: 110.0 for kernel in kernels})

    result = MODULE.compare(baseline, candidate, kernels, 0.20, "base", "head")

    assert result["status"] == "pass"
    assert result["warnings"] == []


def test_compile_time_comparison_warns_outside_threshold():
    baseline = benchmark_document({"add": 100.0})
    candidate = benchmark_document({"add": 125.0})

    result = MODULE.compare(baseline, candidate, ["add"], 0.20, "base", "head")

    assert result["status"] == "warning"
    assert result["kernels"][0]["exceeds_threshold"] is True
    assert "+25.0%" in result["warnings"][0]


def test_compile_time_comparison_warns_when_baseline_missing():
    candidate = benchmark_document({"add": 100.0})

    result = MODULE.compare(None, candidate, ["add"], 0.20, "base", "head")

    assert result["status"] == "warning"
    assert result["baseline_available"] is False
    assert "No cached compile-time baseline" in result["warnings"][0]


def test_publish_compile_time_cache_uses_sha_and_profile(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    (result_dir / "compile-benchmark.json").write_text(
        '{"metadata":{"backend_profile":"sophgo-cmodel"},"summary":{}}',
        encoding="utf-8",
    )
    (result_dir / "compile-benchmark.csv").write_text("kernel,compile_est_ms\n", encoding="utf-8")

    cache_dir = PUBLISH_MODULE.publish_compile_time_cache(tmp_path, result_dir, "abc123")

    assert cache_dir == tmp_path / "compile-time" / "by-sha" / "abc123" / "sophgo-cmodel"
    assert (cache_dir / "latest.json").is_file()
    assert (cache_dir / "latest.csv").is_file()
