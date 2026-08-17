"""Shared setuptools support for native backend fixture wheels."""

import json
import os
from pathlib import Path
import re

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


FINGERPRINT_ENV = "TRITON_ANCHOR_TEST_CORE_ABI_FINGERPRINT"
FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def build_native_fixture(
    *,
    fixture_root,
    distribution,
    description,
    package,
    entry_point,
    library,
    soname,
):
    """Configure one native fixture while keeping build mechanics shared."""

    class BuildManifest(build_py):

        def run(self):
            super().run()
            fingerprint = os.environ.get(FINGERPRINT_ENV, "")
            if FINGERPRINT_PATTERN.fullmatch(fingerprint) is None:
                raise RuntimeError(
                    f"{FINGERPRINT_ENV} must contain "
                    "sha256:<64 lowercase hex characters>"
                )
            path = Path(self.build_lib) / package / "triton_anchor_backend.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["plugins"][0]["abi_fingerprint"] = fingerprint
            path.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    class FixedLibraryName(build_ext):

        def get_ext_filename(self, ext_name):
            del ext_name
            return library

    fixture_root = Path(fixture_root).resolve()
    setup(
        name=distribution,
        version="1.0.0",
        description=description,
        python_requires=">=3.8",
        package_dir={"": "src"},
        packages=find_packages("src"),
        package_data={package: ["triton_anchor_backend.json"]},
        entry_points={"triton.backends": [f"{entry_point}={package}"]},
        ext_modules=[
            Extension(
                package + "._native_artifact",
                sources=[f"src/{package}/native.c"],
                extra_compile_args=["-fvisibility=hidden"],
                extra_link_args=[
                    f"-Wl,-soname,{soname}",
                    f"-Wl,--version-script={fixture_root / 'exports.lds'}",
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
