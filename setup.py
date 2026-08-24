"""
triton-anchor build script.

Builds the embedded triton-shared frontend/core as libtriton.so plus
triton-shared-opt, then packages it together with the triton_anchor Python
orchestration layer.
"""
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import warnings
from pathlib import Path

#Suppress annoying setuptools warnings about C++ header directories looking like Python packages
warnings.filterwarnings("ignore", message=".*is absent from the `packages` configuration.*")

import pybind11
from setuptools import Extension, setup, find_namespace_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel


BASE_DIR = Path(__file__).resolve().parent
TRITON_PROJECT_DIR = BASE_DIR / "triton"
TRITON_PYTHON_ROOT = TRITON_PROJECT_DIR / "python"
TRITON_ROOT = TRITON_PYTHON_ROOT / "triton"
ANCHOR_PYTHON_ROOT = BASE_DIR / "python"
DEFAULT_BUILD_SUBDIR = Path("build") / "cmake-wheel"
CMAKE_BUILD_TARGETS = ("triton", "triton-shared-opt")
PACKAGED_TRITON_SHARED_OPT_RELATIVE_PATH = (
    Path("triton") / "csrc" / "tools" / "triton-shared-opt" / "triton-shared-opt"
)


def get_version_constants():
    """Read version constants without importing triton_anchor."""
    version_file = ANCHOR_PYTHON_ROOT / "triton_anchor" / "_version.py"
    namespace = {}
    exec(
        compile(version_file.read_text(encoding="utf-8"), str(version_file), "exec"),
        namespace,
    )
    return namespace


VERSION_CONSTANTS = get_version_constants()
BUILD_INFO_SCHEMA_VERSION = "1.1"
CORE_ABI_FINGERPRINT_SCHEMA = "triton-anchor-core-abi-v1"


def read_version() -> str:
    init_py = TRITON_ROOT / "__init__.py"
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", init_py.read_text())
    if not match:
        raise RuntimeError(f"unable to read version from {init_py}")
    return os.environ.get(
        "TRITON_ANCHOR_WHEEL_VERSION", VERSION_CONSTANTS["CORE_VERSION"]
    )


def _first_match(path, pattern):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1) if match else None


def _read_cmake_cache(cmake_dir):
    cache = {}
    cache_path = Path(cmake_dir) / "CMakeCache.txt"
    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return cache
    for line in lines:
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", 1)
        key = key_and_type.split(":", 1)[0]
        cache[key] = value
    return cache


def _read_compiler_info(cmake_dir):
    compiler_files = sorted(
        Path(cmake_dir).glob("CMakeFiles/*/CMakeCXXCompiler.cmake")
    )
    if not compiler_files:
        return None, None
    compiler_file = compiler_files[-1]
    compiler_id = _first_match(
        compiler_file, r'^set\(CMAKE_CXX_COMPILER_ID "([^"]+)"\)'
    )
    compiler_version = _first_match(
        compiler_file, r'^set\(CMAKE_CXX_COMPILER_VERSION "([^"]+)"\)'
    )
    return compiler_id, compiler_version


def _read_llvm_version(llvm_dir):
    if not llvm_dir:
        return None
    value = _first_match(
        Path(llvm_dir) / "LLVMConfig.cmake",
        r"^set\(LLVM_PACKAGE_VERSION ([^)]+)\)",
    )
    return value.strip().strip("\"'") if value else None


def _read_mlir_version(mlir_dir):
    """Read MLIR's own package metadata without copying LLVM's result."""
    if not mlir_dir:
        return None
    config_path = Path(mlir_dir) / "MLIRConfig.cmake"
    patterns = (
        r"^set\(MLIR_PACKAGE_VERSION ([^)]+)\)",
        r"^set\(MLIR_VERSION ([^)]+)\)",
        # Current upstream MLIRConfig.cmake records the llvm-project version
        # under LLVM_VERSION rather than defining an MLIR-specific variable.
        r"^set\(LLVM_PACKAGE_VERSION ([^)]+)\)",
        r"^set\(LLVM_VERSION ([^)]+)\)",
    )
    for pattern in patterns:
        value = _first_match(config_path, pattern)
        if value:
            return value.strip().strip("\"'")
    return None


def _toolchain_root(package_dir):
    if not package_dir:
        return None
    try:
        # .../<toolchain>/lib/cmake/{llvm,mlir}
        return Path(package_dir).resolve().parents[2]
    except (OSError, IndexError):
        return None


def _read_toolchain_commit(package_dir):
    """Read the revision embedded by the toolchain that is actually in use."""
    root = _toolchain_root(package_dir)
    if root is None:
        return None
    revision = _first_match(
        root / "include" / "llvm" / "Support" / "VCSRevision.h",
        r'^\s*#define\s+LLVM_REVISION\s+"([0-9a-fA-F]{40})"\s*$',
    )
    return revision.lower() if revision else None


def _read_cxx11_abi(cache):
    """Ask the configured C++ compiler which libstdc++ ABI it will use."""
    compiler = cache.get("CMAKE_CXX_COMPILER")
    if not compiler:
        return None

    build_type = (
        cache.get("CMAKE_BUILD_TYPE")
        or os.environ.get("CMAKE_BUILD_TYPE", "Release")
    ).upper()
    raw_flags = " ".join(
        value
        for value in (
            cache.get("CMAKE_CXX_COMPILER_ARG1", ""),
            cache.get("CMAKE_CXX_FLAGS", ""),
            cache.get("CMAKE_CXX_FLAGS_" + build_type, ""),
        )
        if value
    )
    try:
        flags = shlex.split(raw_flags)
        completed = subprocess.run(
            [compiler] + flags + ["-dM", "-E", "-x", "c++", "-"],
            input="#include <string>\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None
    if completed.returncode != 0:
        return None
    match = re.search(
        r"^#define\s+_GLIBCXX_USE_CXX11_ABI\s+([01])\s*$",
        completed.stdout,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def _sha256_file(path):
    if path is None:
        return None
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return "sha256:" + digest.hexdigest()


def _core_abi_material(build_info):
    """Return complete ABI inputs, or None when any critical fact is unknown."""
    keys = (
        "core_version",
        "vendored_triton_commit",
        "actual_llvm_version_raw",
        "actual_llvm_commit",
        "actual_mlir_version_raw",
        "actual_mlir_commit",
        "cxx_standard",
        "cxx_compiler_id",
        "cxx_compiler_version",
        "cxx11_abi",
        "built_python_soabi",
        "built_platform",
        "ttgpu",
        "core_library_sha256",
    )
    material = {key: build_info.get(key) for key in keys}
    if any(value is None or value == "" for value in material.values()):
        return None
    return material


def _compute_core_abi_fingerprint(build_info):
    material = _core_abi_material(build_info)
    if material is None:
        return None
    payload = {
        "schema": CORE_ABI_FINGERPRINT_SCHEMA,
        "material": material,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def collect_build_info(cmake_dir=None, core_library=None):
    """Collect reproducible build metadata without recording host paths."""
    base_dir = BASE_DIR
    cmake_dir = (
        Path(cmake_dir) if cmake_dir is not None else get_cmake_build_dir()
    )
    cache = _read_cmake_cache(cmake_dir)
    compiler_id, compiler_version = _read_compiler_info(cmake_dir)

    triton_version = _first_match(
        base_dir / "triton" / "python" / "triton" / "__init__.py",
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
    )
    triton_commit = _first_match(
        base_dir / "triton" / "TRITON_VERSION",
        r"^# Commit:\s*([0-9a-fA-F]+)\s*$",
    )
    expected_llvm_commit = (
        base_dir / "triton" / "cmake" / "llvm-hash.txt"
    ).read_text(encoding="utf-8").strip()

    llvm_dir = cache.get("LLVM_DIR")
    mlir_dir = cache.get("MLIR_DIR")
    actual_llvm_version = _read_llvm_version(llvm_dir)
    actual_mlir_version = _read_mlir_version(mlir_dir)
    actual_llvm_commit = _read_toolchain_commit(llvm_dir)
    actual_mlir_commit = _read_toolchain_commit(mlir_dir)
    core_library_sha256 = _sha256_file(core_library)

    build_info = {
        "schema_version": BUILD_INFO_SCHEMA_VERSION,
        "generated": True,
        "core_version": VERSION_CONSTANTS["CORE_VERSION"],
        "backend_protocol_version": VERSION_CONSTANTS[
            "BACKEND_PLUGIN_PROTOCOL_VERSION"
        ],
        "manifest_schema_version": VERSION_CONSTANTS[
            "BACKEND_MANIFEST_SCHEMA_VERSION"
        ],
        "triton_version": triton_version,
        "vendored_triton_commit": triton_commit,
        "expected_llvm_project_commit": expected_llvm_commit,
        # An external LLVM_SYSPATH can point at any compatible installation.
        # A pin is not evidence of the linked installation's exact commit.
        "actual_llvm_version_raw": actual_llvm_version,
        "actual_llvm_commit": actual_llvm_commit,
        "actual_mlir_version_raw": actual_mlir_version,
        "actual_mlir_commit": actual_mlir_commit,
        "cxx_standard": "17",
        "cxx_compiler_id": compiler_id,
        "cxx_compiler_version": compiler_version,
        "cxx11_abi": _read_cxx11_abi(cache),
        "build_type": cache.get("CMAKE_BUILD_TYPE")
        or os.environ.get("CMAKE_BUILD_TYPE", "Release"),
        "ttgpu": "TTGPU" in os.environ,
        "built_python_version": platform.python_version(),
        "built_python_soabi": sysconfig.get_config_var("SOABI"),
        "built_platform": sysconfig.get_platform(),
        "core_abi_fingerprint_schema": CORE_ABI_FINGERPRINT_SCHEMA,
        "core_library_sha256": core_library_sha256,
        "core_abi_fingerprint": None,
    }
    build_info["core_abi_fingerprint"] = _compute_core_abi_fingerprint(build_info)
    return build_info


def write_build_info(destination, cmake_dir=None, core_library=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            collect_build_info(cmake_dir, core_library),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def get_cmake_build_dir() -> Path:
    return Path(
        os.environ.get(
            "TRITON_ANCHOR_CMAKE_BUILD_DIR", BASE_DIR / DEFAULT_BUILD_SUBDIR
        )
    ).resolve()


def parse_cmake_cache(cache_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not cache_path.exists():
        return values
    for line in cache_path.read_text().splitlines():
        if (
            not line
            or line.startswith(("//", "#"))
            or "=" not in line
            or ":" not in line
        ):
            continue
        key_type, value = line.split("=", 1)
        key, _type = key_type.split(":", 1)
        values[key] = value
    return values


def discover_llvm_config() -> tuple[str, str]:
    mlir_dir = os.environ.get("MLIR_DIR")
    llvm_dir = os.environ.get("LLVM_DIR")
    llvm_build_dir = (
        os.environ.get("TRITON_ANCHOR_LLVM_BUILD_DIR")
        or os.environ.get("SPINE_TRITON_CORE_LLVM_BUILD_DIR")
        or os.environ.get("LLVM_BUILD_DIR")
    )

    if llvm_build_dir:
        llvm_build = Path(llvm_build_dir)
        llvm_dir = llvm_dir or str((llvm_build / "lib" / "cmake" / "llvm").resolve())
        mlir_dir = mlir_dir or str((llvm_build / "lib" / "cmake" / "mlir").resolve())

    llvm_library_dir = os.environ.get("LLVM_LIBRARY_DIR") or os.environ.get(
        "LLVM_SYSPATH"
    )
    if llvm_library_dir:
        llvm_lib = Path(llvm_library_dir)
        if (llvm_lib / "lib").exists():
            llvm_lib = llvm_lib / "lib"
        llvm_dir = llvm_dir or str((llvm_lib / "cmake" / "llvm").resolve())
        mlir_dir = mlir_dir or str((llvm_lib / "cmake" / "mlir").resolve())

    cache_values = parse_cmake_cache(
        BASE_DIR / "build" / "cmake-wheel" / "CMakeCache.txt"
    )
    llvm_dir = llvm_dir or cache_values.get("LLVM_DIR")
    mlir_dir = mlir_dir or cache_values.get("MLIR_DIR")

    if not llvm_dir or not mlir_dir:
        raise RuntimeError(
            "unable to discover LLVM/MLIR CMake directories; set "
            "TRITON_ANCHOR_LLVM_BUILD_DIR, or both LLVM_DIR and MLIR_DIR"
        )
    return llvm_dir, mlir_dir


def ensure_copy(src: Path, dst: Path, executable: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if executable:
        dst.chmod(dst.stat().st_mode | 0o111)


def copy_tree_files(src_dir: Path, dst_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    if not src_dir.exists():
        return outputs
    for src in src_dir.rglob("*"):
        if src.is_dir() or (src.is_symlink() and not src.exists()):
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        ensure_copy(src, dst)
        outputs.append(dst)
    return outputs


def get_triton_shared_opt_build_path(build_dir: Path) -> Path:
    return build_dir / PACKAGED_TRITON_SHARED_OPT_RELATIVE_PATH


class CMakeExtension(Extension):
    def __init__(self, name: str, package_path: str):
        super().__init__(name, sources=[])
        self.package_path = Path(package_path)


class BinaryWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False


class CMakeBuildPy(build_py):
    def run(self):
        self.run_command("build_ext")
        super().run()
        core_library = Path(self.build_lib) / "triton" / "_C" / "libtriton.so"
        write_build_info(
            Path(self.build_lib) / "triton_anchor" / "_build_info.json",
            get_cmake_build_dir(),
            core_library,
        )


class CMakeBuild(build_ext):
    def initialize_options(self):
        super().initialize_options()
        self._built_outputs: list[str] = []

    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exc:
            raise RuntimeError(
                "CMake must be installed to build triton-anchor"
            ) from exc
        for ext in self.extensions:
            self.build_extension(ext)

    def get_outputs(self):
        return list(self._built_outputs)

    def build_extension(self, ext: CMakeExtension):
        extdir = (Path(self.build_lib) / ext.package_path).resolve()
        package_root = extdir.parent.resolve()
        build_dir = Path(
            os.environ.get(
                "TRITON_ANCHOR_CMAKE_BUILD_DIR", BASE_DIR / DEFAULT_BUILD_SUBDIR
            )
        ).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)

        llvm_dir, mlir_dir = discover_llvm_config()
        python_include = sysconfig.get_path("platinclude") or sysconfig.get_path(
            "include"
        )
        pybind11_dir = os.environ.get("pybind11_DIR", pybind11.get_cmake_dir())

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={os.environ.get('CMAKE_BUILD_TYPE', 'Release')}",
            "-DTRITON_BUILD_PYTHON_MODULE=ON",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPython3_INCLUDE_DIR={python_include}",
            f"-Dpybind11_DIR={pybind11_dir}",
            f"-DLLVM_DIR={llvm_dir}",
            f"-DMLIR_DIR={mlir_dir}",
        ]

        if os.environ.get("TRITON_EXTRA_LLVM_TARGETS"):
            # Parse env var with flexible delimiters (space, semicolon, colon)
            import re
            targets_str = os.environ.get("TRITON_EXTRA_LLVM_TARGETS")
            targets = re.split(r'[;\s:]+', targets_str.strip())
            targets = [t for t in targets if t]  # Filter empty strings
            # CMake expects a semicolon-separated list as a single argument
            cmake_args.append(f"-DTRITON_EXTRA_LLVM_TARGETS={';'.join(targets)}")

        ninja = shutil.which("ninja")
        if ninja:
            cmake_args = ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"] + cmake_args

        env = os.environ.copy()
        subprocess.check_call(
            ["cmake", str(BASE_DIR)] + cmake_args, cwd=build_dir, env=env
        )

        build_args = ["--build", ".", "--target", *CMAKE_BUILD_TARGETS]
        max_jobs = os.environ.get("MAX_JOBS")
        if max_jobs:
            build_args.extend(["-j", max_jobs])
        subprocess.check_call(["cmake"] + build_args, cwd=build_dir, env=env)

        built_libtriton = extdir / "libtriton.so"
        if not built_libtriton.exists():
            raise RuntimeError(f"missing built libtriton.so at {built_libtriton}")

        outputs = [str(built_libtriton)]

        stub_src = TRITON_ROOT / "_C" / "libtriton"
        stub_dst = package_root / "_C" / "libtriton"
        outputs.extend(str(path) for path in copy_tree_files(stub_src, stub_dst))

        triton_shared_opt_src = get_triton_shared_opt_build_path(build_dir)
        if not triton_shared_opt_src.exists():
            raise RuntimeError(
                f"missing built triton-shared-opt at {triton_shared_opt_src}"
            )
        triton_shared_opt_dst = package_root / "bin" / "triton-shared-opt"
        ensure_copy(triton_shared_opt_src, triton_shared_opt_dst, executable=True)
        outputs.append(str(triton_shared_opt_dst))

        self._built_outputs = outputs


setup(
    name="triton-anchor",
    version=read_version(),
    description="Triton Anchor with triton-shared frontend/core integration",
    long_description="Triton Anchor with triton-shared frontend/core integration",
    license="Apache-2.0",
    author="Triton Anchor Contributors",
    python_requires=">=3.10",
    package_dir={"": "python", "triton": "triton/python/triton"},
    packages=(
        find_namespace_packages(where="triton/python", include=["triton", "triton.*"])
        + find_namespace_packages(
            where="python",
            include=["triton_anchor", "triton_anchor.*"],
            exclude=["triton_anchor.tests", "triton_anchor.tests.*"],
        )
    ),
    install_requires=["packaging>=21"],
    package_data={
        "triton": ["_C/libtriton/*.pyi"],
        "triton_anchor": [
            "_build_info.json",
            "backends/schemas/*.json",
            "backends/examples/*.json",
        ],
    },
    include_package_data=True,
    ext_modules=[CMakeExtension("triton._C.libtriton", "triton/_C")],
    cmdclass={
        "bdist_wheel": BinaryWheel,
        "build_ext": CMakeBuild,
        "build_py": CMakeBuildPy,
    },
    zip_safe=False,
    entry_points={
        "triton.adapters": [
            "triton-shared = triton_anchor.adapters.triton_shared_adapter:TritonSharedAdapter",
        ]
    },
)
