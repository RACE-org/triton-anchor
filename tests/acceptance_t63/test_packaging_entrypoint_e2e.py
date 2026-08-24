"""Real-wheel/importlib.metadata end-to-end acceptance fixture.

The two backend packages are built as wheels and installed into a private
``--target`` directory.  The child process uses the real PathDistribution and
EntryPoint implementations; no metadata or entry-point object is monkeypatched.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import venv
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert completed.returncode == 0, (
        f"command failed ({completed.returncode}): {command!r}\n{completed.stdout}"
    )
    return completed


def _write_backend_project(root: Path, *, ordinal: str, target: str) -> None:
    package = f"t63_acceptance_plugin_{ordinal}"
    entry_point = f"t63-{ordinal}"
    project = root / f"project-{ordinal}"
    module = project / package
    module.mkdir(parents=True)

    (project / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [build-system]
            requires = ["setuptools>=64", "wheel"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "t63-acceptance-plugin-{ordinal}"
            version = "1.0.0"
            requires-python = ">=3.10"

            [project.entry-points."triton.backends"]
            {entry_point} = "{package}:plugin"

            [tool.setuptools]
            packages = ["{package}"]

            [tool.setuptools.package-data]
            {package} = ["triton_anchor_backend.json"]
            """
        ).lstrip(),
        encoding="utf-8",
    )
    (module / "__init__.py").write_text(
        textwrap.dedent(
            f"""
            import os
            from pathlib import Path

            _sentinel_dir = os.environ.get("T63_ENTRYPOINT_SENTINEL_DIR")
            if _sentinel_dir:
                Path(_sentinel_dir, "{ordinal}.imported").write_text(
                    "imported", encoding="utf-8"
                )

            class Compiler:
                pass

            class Driver:
                pass

            class Plugin:
                compiler_cls = Compiler
                driver_cls = Driver

                def initialize(self, context):
                    if _sentinel_dir:
                        Path(_sentinel_dir, "{ordinal}.initialized").write_text(
                            context["manifest"].plugin_id, encoding="utf-8"
                        )

            plugin = Plugin()
            """
        ).lstrip(),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1.0",
        "plugins": [
            {
                "plugin_id": f"t63.acceptance.{ordinal}",
                "entry_point": entry_point,
                "backend_protocol": ">=1.0,<2.0",
                "requires_core": ">=0.2,<0.3",
                "requires_triton": {"version": ">=3.3,<3.4"},
                "targets": [target],
                "capabilities": [f"acceptance.{ordinal}"],
                "isolation_mode": "python_only",
                "priority": 0,
            }
        ],
    }
    (module / "triton_anchor_backend.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


E2E_PROBE = r'''
import importlib.metadata
import json
import pathlib
import sys

site = pathlib.Path(__import__("os").environ["T63_ENTRYPOINT_SITE"]).resolve()
sentinels = pathlib.Path(
    __import__("os").environ["T63_ENTRYPOINT_SENTINEL_DIR"]
).resolve()
sys.path.insert(0, str(site))

from triton_anchor.backends import BackendPluginRegistry, collect_core_environment

core_environment = collect_core_environment()

distributions = tuple(importlib.metadata.distributions(path=[str(site)]))
names = {dist.metadata["Name"] for dist in distributions}
expected_names = {"t63-acceptance-plugin-one", "t63-acceptance-plugin-two"}
if names != expected_names:
    raise AssertionError(f"private installed distribution mismatch: {names}")
entry_points = sorted(
    (ep.name, ep.value, ep.group)
    for dist in distributions
    for ep in dist.entry_points
    if ep.group == "triton.backends"
)
if [item[0] for item in entry_points] != ["t63-one", "t63-two"]:
    raise AssertionError(f"real entry-point metadata mismatch: {entry_points}")

provider = lambda: importlib.metadata.distributions(path=[str(site)])

# Full public discovery, inspection, diagnostics, and validation must not call
# EntryPoint.load. Source metadata must fail the production provenance gate;
# installed generated metadata must complete the full validation profile.
full = BackendPluginRegistry(distribution_provider=provider)
discovered = full.discover(strict=True)
if {record.plugin_id for record in discovered} != {
    "t63.acceptance.one", "t63.acceptance.two"
}:
    raise AssertionError([record.to_dict() for record in discovered])
full.inspect("t63.acceptance.one")
full.diagnostics("t63.acceptance.two")
if list(sentinels.iterdir()):
    raise AssertionError("discovery/query imported plugin code")
validated_full = full.validate()
if core_environment.build_info_generated:
    if any(record.state.value != "validated" for record in validated_full):
        raise AssertionError([record.to_dict() for record in validated_full])
else:
    if any(record.state.value != "rejected" for record in validated_full):
        raise AssertionError([record.to_dict() for record in validated_full])
if list(sentinels.iterdir()):
    raise AssertionError("full compatibility validation imported plugin code")

# A generated core wheel continues with the full production registry. The
# explicitly staged profile is used only by the source-tree mechanism probe.
if core_environment.build_info_generated:
    registry = full
    validated = validated_full
    selection_profile = "full"
else:
    registry = BackendPluginRegistry(
        distribution_provider=provider,
        preflight_profile="triton_version",
    )
    validated = registry.validate(strict=True)
    if any(record.state.value != "validated" for record in validated):
        raise AssertionError([record.to_dict() for record in validated])
    selection_profile = "triton_version"
if list(sentinels.iterdir()):
    raise AssertionError("selection compatibility validation imported plugin code")

one = registry.select("t63-target-one")
if one.plugin_id != "t63.acceptance.one":
    raise AssertionError(one.to_dict())
if not (sentinels / "one.imported").is_file():
    raise AssertionError("selected plugin one was not imported")
if not (sentinels / "one.initialized").is_file():
    raise AssertionError("selected plugin one was not initialized")
if (sentinels / "two.imported").exists():
    raise AssertionError("losing/unrelated plugin two imported too early")

two = registry.select("t63-target-two")
if two.plugin_id != "t63.acceptance.two":
    raise AssertionError(two.to_dict())
if not (sentinels / "two.imported").is_file():
    raise AssertionError("selected plugin two was not imported")
if not (sentinels / "two.initialized").is_file():
    raise AssertionError("selected plugin two was not initialized")

print(json.dumps({
    "distribution_names": sorted(names),
    "core_build_info_generated": core_environment.build_info_generated,
    "core_module_path": str(pathlib.Path(__import__("triton_anchor").__file__).resolve()),
    "entry_points": entry_points,
    "distribution_types": sorted({type(dist).__name__ for dist in distributions}),
    "entry_point_types": sorted({
        type(ep).__name__
        for dist in distributions
        for ep in dist.entry_points
        if ep.group == "triton.backends"
    }),
    "full_validation_states": {
        record.plugin_id: record.state.value for record in validated_full
    },
    "full_validation_errors": {
        record.plugin_id: record.error.to_dict()
        for record in validated_full
        if record.error is not None
    },
    "selection_profile": selection_profile,
    "selection_validation_states": {
        record.plugin_id: record.state.value for record in validated
    },
    "selections": [one.to_dict(), two.to_dict()],
    "sentinels": sorted(path.name for path in sentinels.iterdir()),
}, sort_keys=True))
'''


def _build_fixture_wheels(tmp_path: Path) -> tuple[list[Path], list[dict]]:
    for ordinal, target in (("one", "t63-target-one"), ("two", "t63-target-two")):
        _write_backend_project(tmp_path, ordinal=ordinal, target=target)

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for ordinal in ("one", "two"):
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheels),
                str(tmp_path / f"project-{ordinal}"),
            ],
            cwd=tmp_path,
        )
    wheel_paths = sorted(wheels.glob("*.whl"))
    assert len(wheel_paths) == 2
    wheel_evidence = []
    for path in wheel_paths:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            assert sum(name.endswith("triton_anchor_backend.json") for name in names) == 1
            assert sum(name.endswith(".dist-info/entry_points.txt") for name in names) == 1
        wheel_evidence.append(
            {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
        )
    return wheel_paths, wheel_evidence


def _install_fixture_wheels(
    python: Path | str, wheel_paths: list[Path], site: Path, tmp_path: Path
) -> None:
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--target",
            str(site),
            *(str(path) for path in wheel_paths),
        ],
        cwd=tmp_path,
    )


def _assert_probe_evidence(evidence: dict) -> None:
    assert evidence["distribution_types"] == ["PathDistribution"]
    assert evidence["entry_point_types"] == ["EntryPoint"]
    assert evidence["sentinels"] == [
        "one.imported",
        "one.initialized",
        "two.imported",
        "two.initialized",
    ]


def test_two_real_wheels_discovery_and_import_gate(tmp_path: Path) -> None:
    wheel_paths, wheel_evidence = _build_fixture_wheels(tmp_path)

    site = tmp_path / "installed-site"
    _install_fixture_wheels(sys.executable, wheel_paths, site, tmp_path)
    sentinels = tmp_path / "sentinels"
    sentinels.mkdir()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(site), str(REPO_ROOT / "python")]
    )
    env["PYTHONNOUSERSITE"] = "1"
    env["T63_ENTRYPOINT_SITE"] = str(site)
    env["T63_ENTRYPOINT_SENTINEL_DIR"] = str(sentinels)
    completed = _run(
        [sys.executable, "-c", E2E_PROBE],
        cwd=tmp_path,
        env=env,
    )
    evidence = json.loads(completed.stdout.splitlines()[-1])
    evidence["fixture_wheels"] = wheel_evidence
    _assert_probe_evidence(evidence)
    assert evidence["core_build_info_generated"] is False
    assert evidence["selection_profile"] == "triton_version"
    print("PACKAGING_ENTRYPOINT_E2E=" + json.dumps(evidence, sort_keys=True))


def test_two_real_wheels_against_freshly_installed_core_wheel(
    tmp_path: Path,
) -> None:
    core_wheel_value = os.environ.get("T63_CORE_WHEEL")
    if not core_wheel_value:
        pytest.skip("T63_CORE_WHEEL was not supplied; generated core profile unavailable")
    core_wheel = Path(core_wheel_value).resolve(strict=True)
    wheel_paths, wheel_evidence = _build_fixture_wheels(tmp_path)

    venv_dir = tmp_path / "core-venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = venv_dir / "bin/python"
    clean_env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        clean_env.pop(key, None)
    clean_env["PYTHONNOUSERSITE"] = "1"
    _run(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            str(core_wheel),
        ],
        cwd=tmp_path,
        env=clean_env,
    )
    site = tmp_path / "installed-plugins"
    _install_fixture_wheels(python, wheel_paths, site, tmp_path)
    sentinels = tmp_path / "installed-core-sentinels"
    sentinels.mkdir()
    clean_env["T63_ENTRYPOINT_SITE"] = str(site)
    clean_env["T63_ENTRYPOINT_SENTINEL_DIR"] = str(sentinels)
    completed = _run(
        [str(python), "-I", "-c", E2E_PROBE],
        cwd=tmp_path,
        env=clean_env,
    )
    evidence = json.loads(completed.stdout.splitlines()[-1])
    evidence["fixture_wheels"] = wheel_evidence
    _assert_probe_evidence(evidence)
    assert evidence["core_build_info_generated"] is True
    assert evidence["selection_profile"] == "full"
    core_module = Path(evidence["core_module_path"])
    assert venv_dir.resolve() in core_module.parents
    assert REPO_ROOT.resolve() not in core_module.parents
    print(
        "PACKAGING_INSTALLED_CORE_ENTRYPOINT_E2E="
        + json.dumps(evidence, sort_keys=True)
    )
