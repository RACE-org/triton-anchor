"""Generate one minimal, no-Manifest Legacy backend wheel project."""

from __future__ import annotations

import argparse
from pathlib import Path
from string import Template
from textwrap import dedent


CASES = {
    "legacy_good": {
        "distribution": "triton-anchor-w12-legacy-good-backend",
        "module": "w12_legacy_good_backend",
        "entry_point": "w12_legacy_good",
        "compiler_binding": "LegacyCompiler",
        "driver_binding": "LegacyDriver",
    },
    "legacy_bad_interface": {
        "distribution": (
            "triton-anchor-w12-legacy-bad-interface-backend"
        ),
        "module": "w12_legacy_bad_interface_backend",
        "entry_point": "w12_legacy_bad_interface",
        "compiler_binding": "LegacyCompiler",
        "driver_binding": "object()",
    },
    "legacy_missing_compiler": {
        "distribution": "triton-anchor-w12-legacy-missing-compiler-backend",
        "module": "w12_legacy_missing_compiler_backend",
        "entry_point": "w12_legacy_missing_compiler",
        "compiler_binding": "None",
        "driver_binding": "LegacyDriver",
    },
    "legacy_illegal_compiler": {
        "distribution": "triton-anchor-w12-legacy-illegal-compiler-backend",
        "module": "w12_legacy_illegal_compiler_backend",
        "entry_point": "w12_legacy_illegal_compiler",
        "compiler_binding": "object()",
        "driver_binding": "LegacyDriver",
    },
    "legacy_missing_driver": {
        "distribution": "triton-anchor-w12-legacy-missing-driver-backend",
        "module": "w12_legacy_missing_driver_backend",
        "entry_point": "w12_legacy_missing_driver",
        "compiler_binding": "LegacyCompiler",
        "driver_binding": "None",
    },
}


PYPROJECT_TEMPLATE = Template(
    dedent(
        """\
        [build-system]
        requires = ["setuptools>=64", "wheel"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "$distribution"
        version = "1.0.0"
        description = "Installed no-Manifest Legacy backend fixture"
        requires-python = ">=3.8"

        [project.entry-points."triton.backends"]
        $entry_point = "$module"

        [tool.setuptools]
        package-dir = {"" = "src"}

        [tool.setuptools.packages.find]
        where = ["src"]
        """
    )
)


BACKEND_TEMPLATE = Template(
    dedent(
        '''\
        """Generated no-Manifest Legacy backend fixture."""

        import json
        import os
        from pathlib import Path


        TARGET = "$entry_point"
        EVENT_FILE = "$case.events.jsonl"


        def _mark(event):
            marker_dir = os.environ.get(
                "TRITON_ANCHOR_W12_LEGACY_MARKER_DIR"
            )
            if marker_dir is None:
                return
            path = Path(marker_dir) / EVENT_FILE
            with path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"event": event, "pid": os.getpid()},
                        sort_keys=True,
                    )
                    + "\\n"
                )


        class LegacyCompiler:

            @classmethod
            def supports_target(cls, target):
                return getattr(target, "backend", None) == TARGET

            def __init__(self, target):
                self.target = target
                _mark("compiler_construct")

            def compile(self):
                _mark("compiler_call")
                return {"backend": TARGET}


        class LegacyDriver:

            @classmethod
            def is_active(cls):
                _mark("driver_is_active")
                return True

            def __init__(self):
                _mark("driver_construct")

            def get_current_target(self):
                from triton.backends.compiler import GPUTarget

                _mark("runtime_call")
                return GPUTarget(TARGET, 1, 1)


        def initialize(context):
            del context
            _mark("initialize")


        compiler_cls = $compiler_binding
        driver_cls = $driver_binding

        _mark("import")
        '''
    )
)


def generate(case_name: str, output_root: Path) -> Path:
    """Render one case into an otherwise empty temporary project."""
    case = CASES[case_name]
    project_root = output_root / case_name
    project_root.mkdir(parents=True, exist_ok=False)
    source_root = project_root / "src" / case["module"]
    source_root.mkdir(parents=True)

    (project_root / "pyproject.toml").write_text(
        PYPROJECT_TEMPLATE.substitute(case),
        encoding="utf-8",
    )
    backend_values = dict(case)
    backend_values["case"] = case_name
    (source_root / "__init__.py").write_text(
        BACKEND_TEMPLATE.substitute(backend_values),
        encoding="utf-8",
    )
    return project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    project = generate(
        arguments.case,
        arguments.output_root.resolve(),
    )
    print(project)


if __name__ == "__main__":
    main()
