"""
triton-anchor build script.

Builds the embedded triton-shared frontend/core as libtriton.so plus
triton-shared-opt, then packages it together with the triton_anchor Python
orchestration layer.
"""

import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pybind11
from setuptools import Extension, find_namespace_packages, setup
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


def read_version() -> str:
    init_py = TRITON_ROOT / "__init__.py"
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", init_py.read_text())
    if not match:
        raise RuntimeError(f"unable to read version from {init_py}")
    return os.environ.get("TRITON_ANCHOR_WHEEL_VERSION", "0.2.0")


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
    author="Triton Anchor Contributors",
    python_requires=">=3.10",
    package_dir={"": "python", "triton": "triton/python/triton"},
    packages=(
        find_namespace_packages(where="triton/python", include=["triton", "triton.*"])
        + find_namespace_packages(
            where="python", include=["triton_anchor", "triton_anchor.*"]
        )
    ),
    install_requires=["filelock"],
    package_data={
        "triton": ["_C/libtriton/*.pyi"],
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
