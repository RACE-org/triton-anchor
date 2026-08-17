"""Build the first W10 native collision fixture wheel."""

import json
import os
from pathlib import Path
import re

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


PACKAGE = "w10_native_collision_a"
LIBRARY = "libtriton_anchor_w10_native_collision_a.so"
SONAME = "libtriton_anchor_w10_native_collision.so"
FINGERPRINT_ENV = "TRITON_ANCHOR_TEST_CORE_ABI_FINGERPRINT"
FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class BuildManifest(build_py):

    def run(self):
        super().run()
        fingerprint = os.environ.get(FINGERPRINT_ENV, "")
        if FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
            raise RuntimeError(
                "{} must contain sha256:<64 lowercase hex characters>".format(
                    FINGERPRINT_ENV
                )
            )
        path = (
            Path(self.build_lib)
            / PACKAGE
            / "triton_anchor_backend.json"
        )
        document = json.loads(path.read_text(encoding="utf-8"))
        document["plugins"][0]["abi_fingerprint"] = fingerprint
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class FixedLibraryName(build_ext):

    def get_ext_filename(self, ext_name):
        del ext_name
        return LIBRARY


fixture_root = Path(__file__).resolve().parent

setup(
    name="triton-anchor-w10-native-collision-a-backend",
    version="1.0.0",
    description="First W10 backend in the native SONAME/symbol collision",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={PACKAGE: ["triton_anchor_backend.json"]},
    entry_points={
        "triton.backends": [
            "w10_native_collision_a={}".format(PACKAGE),
        ],
    },
    ext_modules=[
        Extension(
            PACKAGE + "._native_artifact",
            sources=["src/{}/native.c".format(PACKAGE)],
            extra_compile_args=["-fvisibility=hidden"],
            extra_link_args=[
                "-Wl,-soname,{}".format(SONAME),
                "-Wl,--version-script={}".format(
                    fixture_root / "exports.map"
                ),
                "-Wl,-z,defs",
            ],
        )
    ],
    cmdclass={
        "build_ext": FixedLibraryName,
        "build_py": BuildManifest,
    },
    zip_safe=False,
)
