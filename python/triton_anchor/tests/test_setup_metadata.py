"""Packaging checks for the W2/W3 generated metadata."""

import importlib.util
import json
import os
import re
from pathlib import Path
from unittest.mock import patch

from triton_anchor._version import CORE_VERSION
from triton_anchor.backends import load_build_info


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_setup_module():
    captured = {}

    def fake_setup(**kwargs):
        captured.update(kwargs)

    spec = importlib.util.spec_from_file_location(
        "triton_anchor_setup_for_test", REPO_ROOT / "setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch("setuptools.setup", fake_setup):
        spec.loader.exec_module(module)
    return module, captured


def test_setup_uses_authoritative_core_version_and_runtime_dependency():
    _, captured = load_setup_module()
    assert captured["version"] == CORE_VERSION
    assert "packaging>=21" in captured["install_requires"]
    assert "triton_anchor.backends" in captured["packages"]


def test_setup_packages_w2_w3_json_assets():
    _, captured = load_setup_module()
    package_data = captured["package_data"]["triton_anchor"]
    assert "_build_info.json" in package_data
    assert "backends/schemas/*.json" in package_data
    assert "backends/examples/*.json" in package_data


def test_setup_preserves_ttgpu_package_data_switch():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TTGPU", None)
        _, disabled = load_setup_module()

    with patch.dict(os.environ, {"TTGPU": "1"}, clear=False):
        _, enabled = load_setup_module()

    assert disabled["exclude_package_data"]["triton_anchor"] == [
        "include/ttgpu/*",
        "include/ttgpu/**/*",
    ]
    assert enabled["exclude_package_data"]["triton_anchor"] == []


def test_generated_build_info_is_written_only_to_requested_output(tmp_path):
    module, _ = load_setup_module()
    destination = tmp_path / "build-lib" / "triton_anchor" / "_build_info.json"
    empty_cmake_dir = tmp_path / "cmake"
    module.write_build_info(destination, empty_cmake_dir)

    generated = json.loads(destination.read_text(encoding="utf-8"))
    assert generated["generated"] is True
    assert generated["core_version"] == CORE_VERSION
    assert generated["triton_version"] == "3.0.0"
    assert generated["actual_llvm_commit"] is None
    assert generated["actual_mlir_commit"] is None
    assert generated["core_library_sha256"] is None
    assert generated["core_abi_fingerprint"] is None
    assert load_build_info(destination)["generated"] is True

    source_template = json.loads(
        (
            REPO_ROOT / "python" / "triton_anchor" / "_build_info.json"
        ).read_text(encoding="utf-8")
    )
    assert source_template["generated"] is False


def test_generated_build_info_binds_actual_toolchain_and_core_bytes(tmp_path):
    module, _ = load_setup_module()
    cmake_dir = tmp_path / "cmake"
    compiler_dir = cmake_dir / "CMakeFiles" / "9.9.9"
    compiler_dir.mkdir(parents=True)

    toolchain = tmp_path / "actual-toolchain"
    llvm_dir = toolchain / "lib" / "cmake" / "llvm"
    mlir_dir = toolchain / "lib" / "cmake" / "mlir"
    revision_header = toolchain / "include" / "llvm" / "Support"
    llvm_dir.mkdir(parents=True)
    mlir_dir.mkdir(parents=True)
    revision_header.mkdir(parents=True)

    actual_commit = "a" * 40
    (llvm_dir / "LLVMConfig.cmake").write_text(
        "set(LLVM_PACKAGE_VERSION 19.0.0git)\n",
        encoding="utf-8",
    )
    # A distinct value proves MLIR was read from MLIRConfig.cmake instead of
    # inheriting the result of the LLVM lookup.
    (mlir_dir / "MLIRConfig.cmake").write_text(
        "set(LLVM_VERSION 19.0.1)\n",
        encoding="utf-8",
    )
    (revision_header / "VCSRevision.h").write_text(
        '#define LLVM_REVISION "' + actual_commit + '"\n',
        encoding="utf-8",
    )

    fake_cxx = tmp_path / "fake-cxx"
    fake_cxx.write_text(
        "#!/bin/sh\n"
        "printf '#define _GLIBCXX_USE_CXX11_ABI 1\\n'\n",
        encoding="utf-8",
    )
    fake_cxx.chmod(0o755)
    (cmake_dir / "CMakeCache.txt").write_text(
        "\n".join(
            (
                "LLVM_DIR:PATH=" + str(llvm_dir),
                "MLIR_DIR:PATH=" + str(mlir_dir),
                "CMAKE_CXX_COMPILER:FILEPATH=" + str(fake_cxx),
                "CMAKE_BUILD_TYPE:STRING=TritonRelBuildWithAsserts",
                "CMAKE_CXX_FLAGS:STRING=",
                "CMAKE_CXX_FLAGS_TRITONRELBUILDWITHASSERTS:STRING=",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (compiler_dir / "CMakeCXXCompiler.cmake").write_text(
        'set(CMAKE_CXX_COMPILER_ID "GNU")\n'
        'set(CMAKE_CXX_COMPILER_VERSION "13.3.0")\n',
        encoding="utf-8",
    )

    core_library = tmp_path / "libtriton.so"
    core_library.write_bytes(b"core-library-under-test")
    destination = tmp_path / "wheel" / "triton_anchor" / "_build_info.json"
    module.write_build_info(destination, cmake_dir, core_library)
    generated = json.loads(destination.read_text(encoding="utf-8"))

    assert generated["schema_version"] == "1.1"
    assert generated["actual_llvm_version_raw"] == "19.0.0git"
    assert generated["actual_mlir_version_raw"] == "19.0.1"
    assert generated["actual_llvm_commit"] == actual_commit
    assert generated["actual_mlir_commit"] == actual_commit
    assert generated["actual_llvm_commit"] != generated[
        "expected_llvm_project_commit"
    ]
    assert generated["cxx11_abi"] == "1"
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", generated["core_library_sha256"])
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", generated["core_abi_fingerprint"])
    assert load_build_info(destination)["core_abi_fingerprint"] == generated[
        "core_abi_fingerprint"
    ]

    original_library_digest = generated["core_library_sha256"]
    original_fingerprint = generated["core_abi_fingerprint"]
    core_library.write_bytes(b"changed-core-library")
    changed = module.collect_build_info(cmake_dir, core_library)
    assert changed["core_library_sha256"] != original_library_digest
    assert changed["core_abi_fingerprint"] != original_fingerprint

    generated["actual_mlir_version_raw"] = "19.0.2"
    destination.write_text(json.dumps(generated), encoding="utf-8")
    try:
        load_build_info(destination)
    except RuntimeError as exc:
        assert "fingerprint does not match" in str(exc)
    else:
        raise AssertionError("tampered ABI material was accepted")


def test_cxx11_abi_probe_failure_remains_unknown(tmp_path):
    module, _ = load_setup_module()
    compiler = tmp_path / "failing-cxx"
    compiler.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    compiler.chmod(0o755)
    assert module._read_cxx11_abi(
        {
            "CMAKE_CXX_COMPILER": str(compiler),
            "CMAKE_BUILD_TYPE": "Release",
        }
    ) is None
