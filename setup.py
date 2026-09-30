"""
triton-anchor: 统一构建脚本
===========================
替代原 Triton 的 643 行巨型 setup.py，只做三件事：
1. 调用 CMake 编译 C++ 代码（libtriton.so + _C.so）
2. 将编译产物复制到正确的 Python 包目录
3. 同时安装 triton 和 triton_anchor 两个包
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

# Suppress annoying setuptools warnings about C++ header directories looking like Python packages
warnings.filterwarnings("ignore", message=".*is absent from the `packages` configuration.*")


from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from distutils.command.clean import clean


def get_base_dir():
    return os.path.abspath(os.path.dirname(__file__))


def get_version_constants():
    """Read version constants without importing triton_anchor."""
    version_file = Path(get_base_dir()) / "python" / "triton_anchor" / "_version.py"
    namespace = {}
    exec(compile(version_file.read_text(encoding="utf-8"), str(version_file), "exec"), namespace)
    return namespace


VERSION_CONSTANTS = get_version_constants()
BUILD_INFO_SCHEMA_VERSION = "1.1"
CORE_ABI_FINGERPRINT_SCHEMA = "triton-anchor-core-abi-v1"


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

    build_type = (cache.get("CMAKE_BUILD_TYPE") or get_build_type()).upper()
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
    base_dir = Path(get_base_dir())
    cmake_dir = Path(cmake_dir) if cmake_dir is not None else get_cmake_dir()
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
        "build_type": cache.get("CMAKE_BUILD_TYPE") or get_build_type(),
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


def get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    cmake_dir = Path(get_base_dir()) / "build" / dir_name
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


def get_build_type():
    return os.environ.get("TRITON_BUILD_TYPE", "TritonRelBuildWithAsserts")


def get_env_with_keys(keys):
    for key in keys:
        val = os.environ.get(key, "")
        if val:
            return val
    return ""


def is_ttgpu_enabled():
    return "TTGPU" in os.environ


class CMakeClean(clean):
    def initialize_options(self):
        clean.initialize_options(self)
        self.build_temp = str(get_cmake_dir())


class CMakeBuildPy(build_py):
    def run(self):
        self.run_command('build_ext')
        super().run()
        core_library = (
            Path(self.build_lib) / "triton" / "_C" / "libtriton.so"
        )
        write_build_info(
            Path(self.build_lib) / "triton_anchor" / "_build_info.json",
            get_cmake_dir(),
            core_library,
        )


class CMakeExtension(Extension):
    def __init__(self, name, path, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.path = path


class CMakeBuild(build_ext):

    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed")
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        ninja_dir = shutil.which('ninja')
        # 使用 extdir 作为 CMake 的根安装目录
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_dir = get_cmake_dir()

        # Python 头文件路径
        python_include_dir = sysconfig.get_path("platinclude")

        # LLVM 路径探测
        llvm_syspath = get_env_with_keys(["LLVM_SYSPATH"])
        pybind11_syspath = get_env_with_keys(["PYBIND11_SYSPATH"])

        cmake_args = [
            "-G", "Ninja",
            "-DCMAKE_MAKE_PROGRAM=" + ninja_dir,
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DTRITON_BUILD_PYTHON_MODULE=ON",
            "-DPython3_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DPYTHON_INCLUDE_DIRS=" + python_include_dir,
        ]

        # LLVM/MLIR 路径
        if llvm_syspath:
            cmake_args += [
                "-DLLVM_LIBRARY_DIR=" + os.path.join(llvm_syspath, "lib"),
                "-DLLVM_INCLUDE_DIRS=" + os.path.join(llvm_syspath, "include"),
                "-DMLIR_DIR=" + os.path.join(llvm_syspath, "lib", "cmake", "mlir"),
            ]

        # pybind11 路径
        if pybind11_syspath:
            cmake_args += [
                "-DPYBIND11_INCLUDE_DIR=" + os.path.join(pybind11_syspath, "include"),
                "-Dpybind11_DIR=" + os.path.join(pybind11_syspath, "share", "cmake", "pybind11"),
            ]

        # 构建类型
        cfg = get_build_type()
        build_args = ["--config", cfg]

        if platform.system() != "Windows":
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
            max_jobs = os.getenv("MAX_JOBS", str(2 * os.cpu_count()))
            build_args += ['-j' + max_jobs]

        env = os.environ.copy()
        subprocess.check_call(
            ["cmake", get_base_dir()] + cmake_args,
            cwd=cmake_dir, env=env,
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=cmake_dir,
        )

        # 收集上游 Triton 头文件到 triton/python/triton/include 目录
        triton_include_out_dir = os.path.join(get_base_dir(), "triton", "python", "triton", "include")
        os.makedirs(triton_include_out_dir, exist_ok=True)
        
        # 收集 triton-anchor 扩展头文件到 python/triton_anchor/include 目录
        anchor_include_out_dir = os.path.join(get_base_dir(), "python", "triton_anchor", "include")
        os.makedirs(anchor_include_out_dir, exist_ok=True)

        def copy_headers(src_dir, out_dir):
            if not os.path.exists(src_dir): return
            for root, _, files in os.walk(src_dir):
                for f in files:
                    if f.endswith(".h") or f.endswith(".inc") or f.endswith(".def"):
                        src_path = os.path.join(root, f)
                        rel_path = os.path.relpath(src_path, src_dir)
                        dst_path = os.path.join(out_dir, rel_path)
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        shutil.copy2(src_path, dst_path)

        # 拷贝 Triton 头文件
        copy_headers(os.path.join(get_base_dir(), "triton", "include"), triton_include_out_dir)
        copy_headers(os.path.join(cmake_dir, "triton", "include"), triton_include_out_dir)

        # 拷贝 Triton-Anchor 扩展头文件
        copy_headers(os.path.join(get_base_dir(), "csrc", "include"), anchor_include_out_dir)
        copy_headers(os.path.join(cmake_dir, "csrc", "include"), anchor_include_out_dir)



def get_packages():
    """同时安装 triton 和 triton_anchor 两个 Python 包"""
    packages = [
        # 上游 Triton 前端
        "triton",
        "triton._C",
        "triton.compiler",
        "triton.language",
        "triton.language.extra",
        "triton.language.extra.cuda",
        "triton.language.extra.hip",
        "triton.runtime",
        "triton.backends",
        "triton.tools",
        # triton-anchor 编译框架
        "triton_anchor",
        "triton_anchor.adapters",
        "triton_anchor.backends",
        "triton_anchor.extensions",
        "triton_anchor.language",
        "triton_anchor.tests",
    ]
    return packages


setup(
    name="triton-anchor",
    version=VERSION_CONSTANTS["CORE_VERSION"],
    author="Triton Anchor Contributors",
    description="Unified Triton Compilation Frontend for custom AI accelerators",
    long_description="",
    license="Apache-2.0",
    package_dir={
        "": "python",
        "triton": "triton/python/triton",
    },
    packages=get_packages(),
    install_requires=["packaging>=21"],
    package_data={
        "triton.tools": ["compile.h", "compile.c"],
        "triton": ["include/**/*.h", "include/**/*.hpp", "include/**/*.inc", "include/**/*.def", "include/**/*.td"],
        "triton_anchor": [
            "_build_info.json",
            "backends/schemas/*.json",
            "backends/examples/*.json",
            "include/**/*.h",
            "include/**/*.hpp",
            "include/**/*.inc",
            "include/**/*.def",
            "include/**/*.td",
        ],
    },
    exclude_package_data={
        "triton_anchor": [] if is_ttgpu_enabled() else ["include/ttgpu/*", "include/ttgpu/**/*"],
    },
    include_package_data=True,
    ext_modules=[CMakeExtension("triton", "triton/python/triton/_C/")],
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": CMakeBuildPy,
        "clean": CMakeClean,
    },
    zip_safe=False,
    entry_points={
        "triton.adapters": [
            "hybrid = triton_anchor.adapters.hybrid_adapter:HybridAdapter",
            "triton-gpu = triton_anchor.adapters.triton_gpu_adapter:TritonGPUAdapter",
            "triton-linalg = triton_anchor.adapters.triton_linalg_adapter:TritonLinalgAdapter",
            "triton-shared = triton_anchor.adapters.triton_shared_adapter:TritonSharedAdapter",
        ]
    },
    keywords=["Compiler", "Deep Learning", "Triton"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Compilers",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)
