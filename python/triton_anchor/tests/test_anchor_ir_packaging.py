"""Regression coverage for the public AnchorIR native-header artifact."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import triton
import triton_anchor


def _literal_setup_keyword(repository_root: Path, keyword_name: str):
    setup_path = repository_root / "setup.py"
    tree = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    ]
    assert len(calls) == 1
    values = [
        keyword.value
        for keyword in calls[0].keywords
        if keyword.arg == keyword_name
    ]
    assert len(values) == 1
    return ast.literal_eval(values[0])


def test_exported_anchor_ir_validator_header_contains_current_abi_fields():
    package_root = Path(triton_anchor.__file__).resolve().parent
    exported = package_root / "include/triton-anchor/Validation/AnchorIRValidator.h"

    content = exported.read_text(encoding="utf-8")

    assert "DiagnosticTemplate resourceLimit;" in content
    assert "bool resourceLimitReported = false;" in content


def test_source_build_exports_an_exact_copy_of_the_validator_header():
    repository_root = Path(__file__).resolve().parents[3]
    source = repository_root / "csrc/include/triton-anchor/Validation/AnchorIRValidator.h"
    if not source.is_file():
        # A wheel installation intentionally has no source checkout to compare
        # against; the ABI markers above still validate its exported header.
        return

    exported = (
        Path(triton_anchor.__file__).resolve().parent
        / "include/triton-anchor/Validation/AnchorIRValidator.h"
    )
    assert exported.read_bytes() == source.read_bytes()


def test_native_validator_entry_points_are_exported_for_backend_linkage():
    nm = shutil.which("nm")
    if nm is None:
        pytest.skip("nm is required to inspect native symbol visibility")

    native_library = (
        Path(triton.__file__).resolve().parent / "_C" / "libtriton.so"
    )
    if not native_library.is_file():
        pytest.skip("the current platform does not use libtriton.so")

    result = subprocess.run(
        [nm, "-D", "-C", "--defined-only", str(native_library)],
        check=True,
        capture_output=True,
        text=True,
    )
    for entry_point in (
        "validateAnchorIR(",
        "validateAnchorIRText(",
        "normalizeAnchorIR(",
        "normalizeAnchorIRText(",
    ):
        assert f"mlir::triton::anchor::{entry_point}" in result.stdout


def test_t65_documentation_marks_legacy_validator_as_non_strict_compatibility_api():
    """Public T6.5 docs must not direct callers around the strong validator."""

    repository_root = Path(__file__).resolve().parents[3]
    legacy_source = (repository_root / "python/triton_anchor/anchor_ir.py").read_text(
        encoding="utf-8"
    )
    implementation_doc = (
        repository_root / "docs/anchor_ir_validation.md"
    ).read_text(encoding="utf-8")

    assert "legacy regex compatibility" in legacy_source
    assert "not a structural validator" in legacy_source
    assert "validate_pre_hook()/validate_post_hook()" in implementation_doc
    assert "legacy regex" in implementation_doc


def test_one_click_runner_rejects_wrong_native_triton_version_for_plain_clone():
    """The source version gate must not depend on a versioned directory name."""

    repository_root = Path(__file__).resolve().parents[3]
    runner_path = repository_root / "scripts/verify_t65_all.py"
    module_name = "_triton_anchor_verify_t65_version_gate_test"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = runner
    try:
        spec.loader.exec_module(runner)
        build_python_dir = runner._find_build_python_dir(repository_root)
        assert build_python_dir is not None
        source_tree = ast.parse(
            (repository_root / "triton/python/triton/__init__.py").read_text(
                encoding="utf-8"
            )
        )
        source_version = next(
            node.value.value
            for node in source_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        context = SimpleNamespace(
            path=repository_root,
            build_python_dir=build_python_dir,
        )
        payload = {
            "anchor_path": str(repository_root / "python/triton_anchor/__init__.py"),
            "triton_path": str(build_python_dir / "triton/__init__.py"),
            "libtriton_path": str(build_python_dir / "triton/_C/libtriton.so"),
            "triton": source_version,
        }
        assert runner._probe_violations(context, payload) == []
        payload["triton"] = "0.0.0"
        assert any(
            "does not match source version" in violation
            for violation in runner._probe_violations(context, payload)
        )
    finally:
        sys.modules.pop(module_name, None)


def test_one_click_runner_reports_unreadable_source_version_as_a_failure(monkeypatch):
    """A malformed checkout must produce a summary failure, not a traceback."""

    repository_root = Path(__file__).resolve().parents[3]
    runner_path = repository_root / "scripts/verify_t65_all.py"
    module_name = "_triton_anchor_verify_t65_version_error_test"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = runner
    try:
        spec.loader.exec_module(runner)
        context = SimpleNamespace(
            label="current",
            blocked=False,
            converter_available=None,
            launcher=Path("/tmp/t65-unused-launcher"),
        )
        probe_result = runner.StepResult(
            target="current",
            name="environment/import probe",
            status="PASS",
            returncode=0,
            duration_seconds=0.0,
            output_tail='{"ttgpu_converter": false}',
        )
        monkeypatch.setattr(runner, "_run", lambda *args, **kwargs: probe_result)

        def raise_version_error(*args, **kwargs):
            raise ValueError("source Triton version is missing")

        monkeypatch.setattr(runner, "_probe_violations", raise_version_error)
        results = []
        assert runner._probe(context, results, timeout=1) is False
        assert context.blocked
        assert results[-1].status == "FAIL"
        assert "source Triton version is missing" in results[-1].note
    finally:
        sys.modules.pop(module_name, None)


def test_one_click_acceptance_runner_supports_project_minimum_python():
    """Keep the committed acceptance entry point runnable on Python 3.8."""

    repository_root = Path(__file__).resolve().parents[3]
    runner = repository_root / "scripts/verify_t65_all.py"
    tree = ast.parse(
        runner.read_text(encoding="utf-8"),
        filename=str(runner),
        feature_version=(3, 8),
    )

    python_39_string_apis = {"removeprefix", "removesuffix"}
    unsupported = sorted(
        {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in python_39_string_apis
        }
    )

    assert unsupported == [], (
        "setup.py declares Python >=3.8, but the one-click runner uses "
        f"Python 3.9-only string APIs: {unsupported}"
    )


def test_source_setup_publishes_cli_and_t65_runtime_resources():
    """The source gate must inspect current setup.py, not stale metadata."""

    repository_root = Path(__file__).resolve().parents[3]
    entry_points = _literal_setup_keyword(repository_root, "entry_points")
    assert entry_points["console_scripts"] == [
        "triton-anchor-validate = triton_anchor.anchor_ir_cli:main"
    ]

    package_data = _literal_setup_keyword(repository_root, "package_data")
    anchor_data = set(package_data["triton_anchor"])
    assert {
        "include/**/*.h",
        "spec/*.json",
        "tests/data/anchor_ir/**/*.json",
        "tests/data/anchor_ir/**/*.mlir",
    } <= anchor_data


def test_one_click_acceptance_runner_uses_source_and_selected_native_build():
    """Never mix stale Python or native artifacts into the acceptance run."""

    repository_root = Path(__file__).resolve().parents[3]
    runner_path = repository_root / "scripts/verify_t65_all.py"
    module_name = "_triton_anchor_verify_t65_all_test"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = runner
    try:
        spec.loader.exec_module(runner)
        context = runner._make_context(
            argparse.Namespace(python=None),
            repository_root,
        )
        try:
            completed = subprocess.run(
                runner._python_command(
                    context,
                    "-c",
                        (
                            "import json, triton, triton_anchor; "
                            "from pathlib import Path; "
                            "from triton._C import libtriton; "
                            "print(json.dumps({"
                            "'anchor': str(Path(triton_anchor.__file__).resolve()), "
                            "'triton': str(Path(triton.__file__).resolve()), "
                            "'native': str(Path(libtriton.__file__).resolve())}))"
                    ),
                ),
                cwd=repository_root,
                env=context.env,
                check=False,
                capture_output=True,
                text=True,
            )
        finally:
            context.launcher_dir.cleanup()
    finally:
        sys.modules.pop(module_name, None)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    imported = json.loads(completed.stdout.strip())
    source_package = (repository_root / "python/triton_anchor").resolve()
    assert context.build_python_dir is not None
    build_package = (context.build_python_dir / "triton").resolve()
    try:
        Path(imported["anchor"]).resolve().relative_to(source_package)
    except ValueError:
        pytest.fail(
            "one-click acceptance imported a stale triton_anchor copy: %s"
            % imported["anchor"]
        )
    for field in ("triton", "native"):
        try:
            Path(imported[field]).resolve().relative_to(build_package)
        except ValueError:
            pytest.fail(
                "one-click acceptance imported %s outside the selected build: %s"
                % (field, imported[field])
            )
