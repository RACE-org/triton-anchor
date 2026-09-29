"""Real tool execution, installation and numerical measurement behavior."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from tools.basic_tools import actions, runner

ROOT = Path(__file__).resolve().parents[1]


def context():
    return {
        "source_dir": "/task/candidate/checkout", "artifact_dir": "/task/artifacts",
        "target_sha": "a" * 40, "base_sha": "b" * 40,
        "profile": {"backend_enabled": True, "tools": {
            "backend_dir": "/task/backend", "backend_wheel_pattern": "backend-*.whl",
            "expected_backend": "sophgo", "backend_test_paths": ["tests"],
            "backend_smoke_argv": ["python3", "tests/jit.py"],
            "flaggems_dir": "/opt/local-ci/runtime/deps/flaggems",
        }},
    }


def test_tools_plan_build_dependencies_without_implicit_install():
    for tool in ("frontend_build", "backend_build"):
        spec = runner.plan(tool, context(), {"jobs": 1, "build_mode": "incremental"})
        expected = ["environment"] if tool == "frontend_build" else ["environment", "frontend_install"]
        assert spec["dependencies"] == expected
        assert all(c["env"]["MAX_JOBS"] == "1" for c in spec["commands"])
        assert not any("install_wheel" in c["argv"] for c in spec["commands"])
    ctx = context()
    ctx["profile"]["backend_enabled"] = False
    assert runner.plan("backend_build", ctx)["status"] == "not_applicable"


@pytest.mark.parametrize("requires_frontend", [None, False, True])
def test_backend_build_always_depends_on_frontend_wheel_install(requires_frontend):
    ctx = context()
    if requires_frontend is not None:
        ctx["profile"]["tools"]["backend_build_requires_frontend"] = requires_frontend
    assert runner.plan("frontend_install", ctx)["dependencies"] == ["frontend_build"]
    assert runner.plan("backend_build", ctx)["dependencies"] == ["environment", "frontend_install"]


def test_profile_setup_preserves_arguments_and_selected_python():
    ctx = context()
    ctx["python_bin"] = "/task/venv/bin/python"
    ctx["profile"]["tools"]["backend_env_scripts"] = [
        {"path": "/sdk/setup.sh", "args": ["argument; remains data"]}
    ]
    assert runner.plan("frontend_build", ctx)["commands"][0]["argv"][0] != "bash"
    command = runner.plan("backend_smoke", ctx)["commands"][-2]
    assert command["argv"][0] == "bash"
    assert "argument; remains data" in command["argv"]
    assert ctx["python_bin"] in command["argv"]


def test_selected_nodes_and_build_parameters():
    selected = "tests/test_math.py::test_add"
    spec = runner.plan("backend_tests", context(), {"paths": [selected], "keyword": "add"})
    assert "/task/backend/" + selected in spec["commands"][-1]["argv"]
    for paths in (["../test.py"], ["/tmp/test.py"], ["setup.py"]):
        with pytest.raises(ValueError):
            runner.plan("backend_tests", context(), {"paths": paths})
    for jobs in (0, True, 65):
        with pytest.raises(ValueError):
            runner.plan("frontend_build", context(), {"jobs": jobs})


def test_backend_pytest_supports_top_level_conftest_import(tmp_path):
    backend = tmp_path / "backend"
    (backend / "tests").mkdir(parents=True)
    (backend / "tests/conftest.py").write_text("VALUE = 42\n")
    (backend / "tests/test_jit.py").write_text("from conftest import VALUE\ndef test_jit(): assert VALUE == 42\n")
    ctx = context()
    ctx.update(python_bin=sys.executable, tools_dir=str(ROOT / "tools"), artifact_dir=str(tmp_path / "artifacts"))
    ctx["profile"]["tools"]["backend_dir"] = str(backend)
    command = runner.plan("backend_tests", ctx)["commands"][-1]
    Path(command["cwd"]).mkdir(parents=True)
    process = subprocess.run(command["argv"], cwd=command["cwd"], capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads((Path(command["cwd"]) / "tests.json").read_text())
    assert report["passed"] == 1 and report["errors"] == 0
    assert "--import-mode=importlib" in runner.plan("frontend_tests", ctx)["commands"][-1]["argv"]


@pytest.mark.parametrize("configured,explicit,expected", [
    (None, None, 12), ("4", None, 4), ("12", 32, 32),
])
def test_build_parallelism_respects_configuration(configured, explicit, expected):
    ctx = context()
    if configured is not None:
        ctx["profile"]["tools"]["env"] = {"MAX_JOBS": configured}
    spec = runner.plan("frontend_build", ctx, {} if explicit is None else {"jobs": explicit})
    for command in spec["commands"]:
        assert command["env"]["MAX_JOBS"] == str(expected)
        assert command["env"]["CMAKE_BUILD_PARALLEL_LEVEL"] == str(expected)
        assert command["env"]["NINJAFLAGS"] == f"-j{expected}"


@pytest.mark.parametrize("source,passed", [
    ("def test_ok(): assert 1 + 1 == 2\n", True),
    ("def test_bad(): assert False\n", False),
    ("import pytest\n@pytest.mark.skip\ndef test_skip(): pass\n", False),
    ("# No tests\n", False),
    ("def test_bad(): syntax error\n", False),
])
def test_pytest_exit_status_and_real_counts(tmp_path, source, passed):
    test = tmp_path / "test_actual.py"
    test.write_text(source)
    output = tmp_path / "tests.json"
    run = subprocess.run([
        sys.executable, "-I", str(ROOT / "tools/basic_tools/pytest_exec.py"),
        "--output", str(output), "--", "-q", "-p", "no:cacheprovider", str(test),
    ], capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert (run.returncode == 0) is passed
    assert (result["status"] == "pass") is passed
    assert result["passed"] == (1 if passed else 0)


def test_runner_saves_result_and_stops_after_failure(tmp_path):
    ctx = {"source_dir": str(tmp_path), "artifact_dir": str(tmp_path / "artifacts"),
           "target_sha": "a" * 40}
    commands = [
        {"argv": [sys.executable, "-c", code], "cwd": str(tmp_path), "env": {}, "timeout": 5}
        for code in ("print('actual log')", "raise SystemExit(3)", "raise RuntimeError('must not run')")
    ]
    with patch.object(runner, "plan", return_value={"status": "ready", "reason": "", "commands": commands}):
        result = runner.execute("environment", ctx)
    out = tmp_path / "artifacts/environment"
    assert result["status"] == "fail" and result["exit_code"] == 3
    assert json.loads((out / "result.json").read_text()) == result
    log = (out / "command.log").read_text()
    assert "actual log" in log and "must not run" not in log


def test_install_explicit_wheel(tmp_path):
    wheel = tmp_path / "frontend.whl"
    wheel.write_bytes(b"wheel")
    ctx = {"artifact_dir": str(tmp_path / "artifacts"), "python_bin": sys.executable}
    with patch.object(actions, "run") as run:
        actions.install_wheel({"tool_id": "frontend_install", "context": ctx,
                               "parameters": {"wheel": str(wheel)}})
    command = run.call_args.args[0]
    assert "--no-deps" in command and command[-1] == str(wheel)
    report = json.loads((tmp_path / "artifacts/frontend_install/installation.json").read_text())
    assert report["sha256"] == actions.digest(wheel)


def test_native_and_tool_commands_use_base_environment_without_candidate_leaks(tmp_path, monkeypatch):
    for key in ("LLVM_BUILD_DIR", "LLVM_BINARY_DIR", "BACKEND_PATH", "PYTHONPATH"):
        monkeypatch.setenv(key, "/candidate-only")
    monkeypatch.setenv("HTTPS_PROXY", "http://transport.invalid:8080")
    setup = tmp_path / "setup.sh"
    setup.write_text("export LLVM_BUILD_DIR=/wrong LLVM_BINARY_DIR=/wrong LLVM_DIR=/wrong\n")
    ctx = {
        "source_dir": str(tmp_path), "artifact_dir": str(tmp_path / "artifacts"),
        "target_sha": "b" * 40, "variant": "base", "tools_dir": str(ROOT / "tools"),
        "runtime_env": {"PATH": os.defpath, "PYTHON_BIN": sys.executable,
                        "LLVM_BUILD_DIR": "/base/llvm", "LLVM_BINARY_DIR": "/base/llvm/bin"},
        "profile": {"tools": {"env_scripts": [{"path": str(setup)}]}},
    }
    context_path = tmp_path / "base-context.json"
    context_path.write_text(json.dumps(ctx))
    code = (
        "import os; "
        "assert os.environ['LLVM_BUILD_DIR']=='/base/llvm'; "
        "assert os.environ['LLVM_BINARY_DIR']=='/base/llvm/bin'; "
        "assert 'LLVM_DIR' not in os.environ; "
        "assert 'BACKEND_PATH' not in os.environ; "
        "assert not os.environ.get('PYTHONPATH'); "
        "assert os.environ['HTTPS_PROXY']=='http://transport.invalid:8080'; "
        "print('base environment verified')"
    )
    argv = [sys.executable, "-c", code]
    native = subprocess.run(
        [sys.executable, str(ROOT / "tools/basic_tools/variant_exec.py"),
         "--context", str(context_path), "--", *argv], capture_output=True, text=True,
    )
    assert native.returncode == 0, native.stderr
    assert "base environment verified" in native.stdout
    command = {"argv": runner.environment_command(argv, ctx["profile"]["tools"], ctx["tools_dir"]),
               "cwd": str(tmp_path), "env": {}, "timeout": 5}
    with patch.object(runner, "plan", return_value={"status": "ready", "reason": "", "commands": [command]}):
        result = runner.execute("environment", ctx)
    assert result["status"] == "pass"
    assert result["variant"] == "base"


def test_build_cleanup_keeps_paths_outside_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (source / "build").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        actions.remove_child(source, source / "build")
    assert outside.is_dir()


class InstalledImportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_frontend_import_origin_checks_distribution_and_ignores_diagnostics(self):
        site = self.root / "site"
        site.mkdir()
        for name in ("triton", "triton_anchor"):
            package = site / name
            package.mkdir()
            (package / "__init__.py").write_text(
                "import os\nprint('python diagnostic')\nos.write(1,b'native diagnostic\\n')\n"
            )
        metadata = site / "triton_anchor-0.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text("Name: triton-anchor\nVersion: 0.0\n")
        (metadata / "RECORD").write_text(
            "triton/__init__.py,,\ntriton_anchor/__init__.py,,\n"
        )

        def probe(argv, **kwargs):
            code = "import sys; sys.path.insert(0," + repr(str(site)) + ")\n" + argv[-1]
            return subprocess.run(
                [sys.executable, "-I", "-c", code], check=True, **kwargs
            )

        with patch.object(actions, "run", side_effect=probe):
            identity = actions.import_identity(
                {"python_bin": sys.executable, "artifact_dir": str(self.root)}
            )
        self.assertEqual(set(identity["imports"]), {"triton", "triton_anchor"})

def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ir_benchmark = load(
    "local_ci_tool_ir_benchmark",
    ROOT / "tools/basic_tools/performance/ir_serialization_benchmark.py",
)
ir_compare = load(
    "local_ci_tool_ir_compare",
    ROOT / "tools/basic_tools/performance/compare_ir_serialization.py",
)
profile_compare = load(
    "local_ci_tool_profile_compare",
    ROOT / "tools/basic_tools/performance/compare_pass_profile.py",
)
profile_benchmark = load(
    "local_ci_tool_profile_benchmark",
    ROOT / "tools/basic_tools/performance/pass_profile_benchmark.py",
)
compile_benchmark = load(
    "local_ci_tool_compile_benchmark",
    ROOT / "tools/basic_tools/performance/compile_benchmark.py",
)


class MeasurementValidationTests(unittest.TestCase):
    def test_ir_comparison_covers_every_dashboard_metric(self):
        metrics = ("serialize", "write_text", "read_text", "deserialize", "roundtrip")
        baseline = {"summary": {"add": {"metrics": {
            name: {"median_ms": index + 1} for index, name in enumerate(metrics)
        }}}}
        candidate = {"summary": {"add": {"metrics": {
            name: {"median_ms": index + 2} for index, name in enumerate(metrics)
        }}}}
        result = ir_compare.compare(
            baseline, candidate, ["add"], list(ir_compare.DEFAULT_METRICS),
            0.2, 0.05, 0.05, "base", "candidate",
        )
        self.assertEqual(ir_compare.DEFAULT_METRICS, metrics)
        self.assertEqual(
            {(row["kernel"], row["metric"]) for row in result["rows"]},
            {("add", metric) for metric in metrics},
        )

    def test_profile_compare_rejects_missing_candidate_passes(self):
        with self.assertRaises(ValueError):
            profile_compare.compare(
                None,
                {"summary": {"add": {"passes": {}}}},
                ["add"],
                0.2,
                1,
                1,
                10,
                "slowdown",
                "base",
                "head",
            )

    def test_compile_benchmark_rejects_empty_or_zero_sample(self):
        for kernels, repeat in (("", 1), ("add", 0)):
            with self.assertRaises(ValueError):
                compile_benchmark.run_parent(
                    argparse.Namespace(kernels=kernels, repeat=repeat, warmup=0)
                )

    def test_pass_profile_no_events_is_an_execution_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            args = argparse.Namespace(
                kernels="add",
                repeat=1,
                warmup=0,
                top_n=10,
                flaggems_root=directory,
                cache_root=str(root / "benchmark/cache"),
                backend="fixture",
                keep_workdirs=False,
            )
            with patch.object(
                profile_benchmark, "run_child", return_value=({"compile_est_ms": 1}, [])
            ):
                with self.assertRaisesRegex(RuntimeError, "No MLIR pass timing"):
                    profile_benchmark.run_parent(args)

    def test_roundtrip_checks_canonical_content_and_verifier(self):
        class Module:
            def __init__(self, text, valid=True):
                self.text, self.valid = text, valid

            def __str__(self):
                return self.text

            def verify(self):
                return self.valid

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "module.ttir"
            ir = types.SimpleNamespace(
                parse_mlir_module=lambda p, c: Module(Path(p).read_text())
            )
            binding = types.ModuleType("triton._C.libtriton")
            binding.ir = ir
            with patch.dict(sys.modules, {"triton._C.libtriton": binding}):
                row = ir_benchmark.measure_once(
                    [Module("module {}\n")], None, [path], "add", "repeat", 0
                )
                self.assertTrue(row["roundtrip_verified"])
                ir.parse_mlir_module = lambda p, c: Module("module { changed }\n")
                with self.assertRaisesRegex(RuntimeError, "canonical"):
                    ir_benchmark.measure_once(
                        [Module("module {}\n")], None, [path], "add", "repeat", 0
                    )
                ir.parse_mlir_module = lambda p, c: Module("module {}\n", False)
                with self.assertRaisesRegex(RuntimeError, "invalid MLIR"):
                    ir_benchmark.measure_once(
                        [Module("module {}\n")], None, [path], "add", "repeat", 0
                    )


def test_missing_baseline_is_reported_but_invalid_measurements_fail(tmp_path):
    ctx = {"artifact_dir": str(tmp_path), "target_sha": "a" * 40}
    candidate = {"summary": {"add": {
        "all_correct": True, "compile_est": {"median_ms": 2.0, "count": 3},
    }}}
    out = tmp_path / "compile_time"
    actions.write_json(out / "candidate.json", candidate)
    actions.compare_performance({"tool_id": "compile_time", "context": ctx, "kernels": ["add"]})
    assert json.loads((out / "comparison.json").read_text())["status"] == "not_comparable"
    candidate["summary"]["add"]["compile_est"]["median_ms"] = float("nan")
    with pytest.raises(ValueError, match="Invalid"):
        actions.validate_measurements("compile_time", candidate, ["add"])


@pytest.mark.parametrize("same_llvm", [True, False])
def test_performance_comparison_requires_matching_llvm(tmp_path, same_llvm):
    ctx = {"artifact_dir": str(tmp_path), "target_sha": "a" * 40, "base_sha": "b" * 40,
           "environment_fingerprint": "same-runtime",
           "profile": {"id": "frontend", "llvm_revision": "c" * 40}}
    measurement = {"summary": {"add": {
        "all_correct": True, "compile_est": {"median_ms": 2.0, "count": 3},
    }}}
    baseline_path = tmp_path / "base.json"
    actions.write_json(baseline_path, {**measurement, "metadata": {
        "commit_sha": ctx["base_sha"], "environment_fingerprint": "same-runtime",
    }})
    ctx["performance_baselines"] = {"compile_time": {
        "path": str(baseline_path), "sha256": actions.digest(baseline_path),
        "base_sha": ctx["base_sha"], "profile_id": "frontend",
        "llvm_revision": ("c" if same_llvm else "d") * 40,
        "environment_fingerprint": "same-runtime",
    }}
    out = tmp_path / "compile_time"
    actions.write_json(out / "candidate.json", measurement)
    with patch.object(actions, "run") as run:
        actions.compare_performance({"tool_id": "compile_time", "context": ctx, "kernels": ["add"]})
    assert run.called is same_llvm
    if not same_llvm:
        result = json.loads((out / "comparison.json").read_text())
        assert result["status"] == "not_comparable" and result["reason"] == "llvm_revision_mismatch"
