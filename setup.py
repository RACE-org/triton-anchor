"""
triton-anchor: 统一构建脚本
===========================
替代原 Triton 的 643 行巨型 setup.py，只做三件事：
1. 调用 CMake 编译 C++ 代码（libtriton.so + _C.so）
2. 将编译产物复制到正确的 Python 包目录
3. 同时安装 triton 和 triton_anchor 两个包
"""
import os
import re
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from distutils.command.clean import clean


def get_base_dir():
    return os.path.abspath(os.path.dirname(__file__))


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


class CMakeClean(clean):
    def initialize_options(self):
        clean.initialize_options(self)
        self.build_temp = str(get_cmake_dir())


class CMakeBuildPy(build_py):
    def run(self):
        self.run_command('build_ext')
        return super().run()


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
            "-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON",
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
        "triton_anchor.extensions",
        "triton_anchor.language",
        "triton_anchor.tests",
    ]
    return packages


setup(
    name="triton-anchor",
    version="0.2.0",
    author="Triton Anchor Contributors",
    description="Unified Triton Compilation Frontend for custom AI accelerators",
    long_description="",
    package_dir={
        "": "python",
        "triton": "triton/python/triton",
    },
    packages=get_packages(),
    install_requires=["filelock"],
    package_data={
        "triton.tools": ["compile.h", "compile.c"],
        "triton": [
            "include/**/*.h", "include/**/*.inc", "include/**/*.def",
            "include/**/**/*.h", "include/**/**/*.inc", "include/**/**/*.def",
            "include/**/**/**/*.h", "include/**/**/**/*.inc", "include/**/**/**/*.def",
        ],
        "triton_anchor": [
            "include/**/*.h", "include/**/*.inc", "include/**/*.def",
            "include/**/**/*.h", "include/**/**/*.inc", "include/**/**/*.def",
            "include/**/**/**/*.h", "include/**/**/**/*.inc", "include/**/**/**/*.def",
        ],
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
            "triton-linalg = triton_anchor.adapters.triton_linalg_adapter:TritonLinalgAdapter",
        ]
    },
    keywords=["Compiler", "Deep Learning", "Triton"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Compilers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)
