"""Build the installed AC4 protocol producer fixture."""

import json
import os
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


PACKAGE = "ac4_protocol_backend"


class BuildManifest(build_py):
    def run(self):
        super().run()
        version = os.environ.get("TRITON_ANCHOR_AC4_PRODUCER_VERSION", "1.0")
        path = Path(self.build_lib) / PACKAGE / "triton_anchor_backend.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["plugins"][0]["producer_protocol_version"] = version
        path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


setup(
    name="triton-anchor-ac4-protocol-backend",
    version="1.0.0",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={PACKAGE: ["triton_anchor_backend.json"]},
    entry_points={"triton.backends": ["ac4_protocol=" + PACKAGE]},
    cmdclass={"build_py": BuildManifest},
    zip_safe=False,
)
