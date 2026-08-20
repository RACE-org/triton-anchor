"""Acceptance oracles for T6.3's Triton-facing public integration.

These tests deliberately build real wheels with real ``triton.backends``
entry-point metadata.  They do not use T10.2.  The source probes replace only
the unavailable native ``libtriton`` import and unrelated compiler services;
Registry discovery/selection, ``triton.backends``, compiler ``make_backend``,
and runtime driver/reset execute from the checked-out product sources.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path
from typing import Any

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
PROBE = Path(__file__).with_name("triton_integration_probe.py")
TRITON_COMMIT = "523a1b235b213bc192f2d5a8999add5bf2d0fea5"


def _record_digest(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
    return "sha256=" + digest.rstrip(b"=").decode("ascii")


def _build_wheel(
    directory: Path,
    *,
    distribution: str,
    module: str,
    entry_point: str,
    source: str,
    manifest: dict[str, Any] | None,
) -> Path:
    """Create a standards-compliant pure-Python wheel without build tools."""
    version = "1.0.0"
    wheel_distribution = distribution.replace("-", "_")
    dist_info = f"{wheel_distribution}-{version}.dist-info"
    files: dict[str, bytes] = {
        f"{module}/__init__.py": source.encode("utf-8"),
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {distribution}\n"
            f"Version: {version}\n"
        ).encode("utf-8"),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: T6.3 integration acceptance\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode("utf-8"),
        f"{dist_info}/entry_points.txt": (
            "[triton.backends]\n"
            f"{entry_point} = {module}\n"
        ).encode("utf-8"),
        f"{dist_info}/top_level.txt": f"{module}\n".encode("utf-8"),
    }
    if manifest is not None:
        files[f"{module}/triton_anchor_backend.json"] = (
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")

    record_path = f"{dist_info}/RECORD"
    record_stream = io.StringIO(newline="")
    writer = csv.writer(record_stream, lineterminator="\n")
    for name, payload in sorted(files.items()):
        writer.writerow((name, _record_digest(payload), str(len(payload))))
    writer.writerow((record_path, "", ""))
    files[record_path] = record_stream.getvalue().encode("utf-8")

    wheel = directory / (
        f"{wheel_distribution}-{version}-py3-none-any.whl"
    )
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(files.items()):
            archive.writestr(name, payload)
    return wheel


def _manifest(entry_point: str, target: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "plugins": [
            {
                "plugin_id": f"acceptance.{entry_point}",
                "entry_point": entry_point,
                "backend_protocol": ">=1.0,<2.0",
                "requires_core": ">=0.2,<0.3",
                "requires_triton": {
                    "version": ">=3.3,<3.4",
                    "commit": TRITON_COMMIT,
                },
                "targets": [target],
                "capabilities": ["acceptance.compile"],
                "isolation_mode": "python_only",
                "priority": 0,
            }
        ],
    }


COMPATIBLE_PLUGIN = r'''
import hashlib
import os
from pathlib import Path

from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends.driver import DriverBase

marker = os.environ.get("T63_PLUGIN_IMPORT_MARKER")
if marker:
    Path(marker).write_text("imported", encoding="utf-8")


class Options:
    def __init__(self, values):
        self.__dict__.update(values)

    def hash(self):
        payload = repr(sorted(self.__dict__.items())).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _compile_fixture(source, metadata):
    counter = os.environ.get("T63_STAGE_COUNTER")
    if counter:
        path = Path(counter)
        count = int(path.read_text(encoding="utf-8")) if path.exists() else 0
        path.write_text(str(count + 1), encoding="utf-8")
    metadata.update(
        name="fixture_kernel",
        cluster_dims=(1, 1, 1),
        num_warps=1,
        num_ctas=1,
        shared=0,
    )
    return b"t63-fixture-binary"


class Compiler(BaseBackend):
    binary_ext = "fixturebin"

    @classmethod
    def supports_target(cls, target):
        return target.backend in {"legacy_fixture", "manifest_fixture"}

    def hash(self):
        return "t63-compatible-compiler"

    def parse_options(self, options):
        return Options(options)

    def add_stages(self, stages, options):
        stages["ttir"] = lambda source, metadata: source
        stages["fixturebin"] = _compile_fixture

    def load_dialects(self, context):
        return None

    def get_module_map(self):
        return {}

    def get_codegen_implementation(self, options):
        return {}

    def pack_metadata(self, metadata):
        return metadata


class Driver(DriverBase):
    @classmethod
    def is_active(cls):
        return True

    def get_current_target(self):
        target = os.environ.get("T63_PLUGIN_TARGET", "legacy_fixture")
        return GPUTarget(target, "fixture-arch", 32)

    def get_active_torch_device(self):
        return "cpu"

    def get_benchmarker(self):
        return lambda kernel_call, quantiles, **kwargs: [0.0 for _ in quantiles]


compiler_cls = Compiler
driver_cls = Driver
'''


V30_STYLE_PLUGIN = r'''
import os
from pathlib import Path

from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends.driver import DriverBase

marker = os.environ.get("T63_PLUGIN_IMPORT_MARKER")
if marker:
    Path(marker).write_text("imported", encoding="utf-8")


class Compiler(BaseBackend):
    @classmethod
    def supports_target(cls, target):
        return target.backend == "v30_fixture"

    def hash(self):
        return "v30-style"

    def parse_options(self, options):
        return options

    def add_stages(self, stages, options):
        return None

    def load_dialects(self, context):
        return None


class Driver(DriverBase):
    @classmethod
    def is_active(cls):
        return True

    def get_current_target(self):
        return GPUTarget("v30_fixture", "fixture-arch", 32)


compiler_cls = Compiler
driver_cls = Driver
'''


INSTALLED_CACHE_SMOKE = r'''
import json
import os
from pathlib import Path

import triton
import triton_anchor
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.runtime import driver
from triton_anchor.backends import get_backend_plugin_registry

target = GPUTarget("manifest_fixture", "fixture-arch", 32)
source = os.environ["T63_TTIR_SOURCE"]
counter = Path(os.environ["T63_STAGE_COUNTER"])

first = triton.compile(source, target=target)
after_first = int(counter.read_text(encoding="utf-8"))
first_mapping = {
    name: {
        "compiler": backend.compiler.__module__ + "." + backend.compiler.__qualname__,
        "driver": backend.driver.__module__ + "." + backend.driver.__qualname__,
        "record_id": backend.record_id,
    }
    for name, backend in backends.items()
}

second = triton.compile(source, target=target)
after_second = int(counter.read_text(encoding="utf-8"))
runtime_target = driver.active.get_current_target().backend
registry = get_backend_plugin_registry()
active_state = registry.diagnostics()["plugins"][0]["state"]

reset_errors = [error.to_dict() for error in registry.reset()]
mapping_after_reset = dict(backends)
third = triton.compile(source, target=target)
after_third = int(counter.read_text(encoding="utf-8"))
runtime_target_after_reset = driver.active.get_current_target().backend

print(json.dumps({
    "triton_path": triton.__file__,
    "anchor_path": triton_anchor.__file__,
    "first_hash": first.hash,
    "second_hash": second.hash,
    "third_hash": third.hash,
    "stage_counts": [after_first, after_second, after_third],
    "mapping": first_mapping,
    "runtime_target": runtime_target,
    "active_state": active_state,
    "reset_errors": reset_errors,
    "mapping_after_reset": list(mapping_after_reset),
    "mapping_after_reselection": list(backends),
    "runtime_target_after_reset": runtime_target_after_reset,
}, sort_keys=True))
'''


def _install_wheel(wheel: Path, target: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--no-compile",
            "--target",
            str(target),
            str(wheel),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture(scope="module")
def integration_sites(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("triton-integration-wheels")

    legacy_wheel = _build_wheel(
        root,
        distribution="t63-legacy-fixture",
        module="t63_legacy_fixture",
        entry_point="legacy_fixture",
        source=COMPATIBLE_PLUGIN,
        manifest=None,
    )
    manifest_wheel = _build_wheel(
        root,
        distribution="t63-manifest-fixture",
        module="t63_manifest_fixture",
        entry_point="manifest_fixture",
        source=COMPATIBLE_PLUGIN,
        manifest=_manifest("manifest_fixture", "manifest_fixture"),
    )
    v30_wheel = _build_wheel(
        root,
        distribution="t63-v30-fixture",
        module="t63_v30_fixture",
        entry_point="v30_fixture",
        source=V30_STYLE_PLUGIN,
        manifest=_manifest("v30_fixture", "v30_fixture"),
    )

    sites = {}
    for name, wheel in (
        ("legacy", legacy_wheel),
        ("manifest", manifest_wheel),
        ("v30", v30_wheel),
    ):
        site = root / f"{name}-site"
        site.mkdir()
        _install_wheel(wheel, site)
        sites[name] = site
    sites["root"] = root
    sites["legacy_wheel"] = legacy_wheel
    sites["manifest_wheel"] = manifest_wheel
    return sites


def _run_probe(
    integration_sites: dict[str, Path], mode: str, *, baseline: bool = False
) -> dict[str, Any]:
    root = integration_sites["root"]
    marker = root / f"{mode}-{'baseline' if baseline else 'port'}.marker"
    command = [
        sys.executable,
        str(PROBE),
        "--mode",
        mode,
        "--repository",
        str(REPOSITORY),
        "--site",
        str(integration_sites[mode]),
        "--marker",
        str(marker),
    ]
    if baseline:
        command.append("--baseline")
    environment = dict(os.environ)
    environment.pop("TRITON_ANCHOR_BACKEND", None)
    environment.pop("T63_PLUGIN_TARGET", None)
    environment["PYTHONPATH"] = ""
    if mode == "manifest":
        environment["T63_PLUGIN_TARGET"] = "manifest_fixture"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, (
        f"command: {' '.join(command)}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def legacy_comparison(integration_sites):
    return {
        "baseline": _run_probe(integration_sites, "legacy", baseline=True),
        "port": _run_probe(integration_sites, "legacy", baseline=False),
    }


@pytest.fixture(scope="module")
def manifest_result(integration_sites):
    return _run_probe(integration_sites, "manifest")


@pytest.fixture(scope="module")
def v30_result(integration_sites):
    return _run_probe(integration_sites, "v30")


def test_triton_integration_upstream_v33_legacy_public_oracle(
    legacy_comparison,
) -> None:
    """The frozen target baseline is the regression oracle, not an emulation."""
    baseline = legacy_comparison["baseline"]
    assert "legacy_fixture" in baseline["mapping_after_discovery"], baseline
    assert baseline["compiler"]["ok"], baseline
    assert baseline["compiler"]["value"]["target"] == "legacy_fixture"
    assert baseline["runtime_driver"]["ok"], baseline
    assert baseline["runtime_driver"]["value"]["target"] == "legacy_fixture"
    assert baseline["reset_then_runtime_driver"]["ok"], baseline


def test_triton_integration_legacy_entry_point_remains_in_public_mapping(
    legacy_comparison,
) -> None:
    """A no-Manifest wheel must remain usable through the v3.3 public map."""
    port = legacy_comparison["port"]
    assert "legacy_fixture" in port["mapping_after_public_use"], json.dumps(
        legacy_comparison, indent=2, sort_keys=True
    )


def test_triton_integration_legacy_compiler_make_backend_regression(
    legacy_comparison,
) -> None:
    port = legacy_comparison["port"]
    assert port["compiler"]["ok"], json.dumps(
        legacy_comparison, indent=2, sort_keys=True
    )
    assert port["compiler"]["value"]["target"] == "legacy_fixture"


def test_triton_integration_legacy_runtime_driver_and_reset_regression(
    legacy_comparison,
) -> None:
    port = legacy_comparison["port"]
    assert port["runtime_driver"]["ok"], json.dumps(
        legacy_comparison, indent=2, sort_keys=True
    )
    assert port["runtime_driver"]["value"]["target"] == "legacy_fixture"
    assert port["reset_then_runtime_driver"]["ok"], json.dumps(
        legacy_comparison, indent=2, sort_keys=True
    )


def test_triton_integration_legacy_explicit_registry_selection_is_consistent(
    legacy_comparison,
) -> None:
    """Bound the regression: the new exact-key opt-in path itself still works."""
    explicit = legacy_comparison["port"]["explicit_selection"]
    assert explicit["ok"], json.dumps(
        legacy_comparison, indent=2, sort_keys=True
    )
    value = explicit["value"]
    assert value["decision"]["is_legacy"] is True
    assert value["decision"]["method"] == "python_explicit"
    assert value["runtime_driver"]["target"] == "legacy_fixture"
    mapping = value["mapping"]["legacy_fixture"]
    assert value["compiler"] == mapping["compiler"]
    assert value["runtime_driver"]["class"] == mapping["driver"]


def test_triton_integration_manifest_compiler_publishes_selected_pair(
    manifest_result,
) -> None:
    assert not manifest_result["imported_before_selection"], manifest_result
    assert manifest_result["mapping_before_selection"] == {}, manifest_result
    assert manifest_result["compiler"]["ok"], manifest_result
    mapping = manifest_result["mapping_after_compiler"]
    assert set(mapping) == {"manifest_fixture"}, manifest_result
    assert mapping["manifest_fixture"]["plugin_id"] == (
        "acceptance.manifest_fixture"
    )


def test_triton_integration_manifest_compiler_and_driver_are_one_record(
    manifest_result,
) -> None:
    assert manifest_result["runtime_driver"]["ok"], manifest_result
    mapping = manifest_result["mapping_after_compiler"]["manifest_fixture"]
    assert manifest_result["compiler"]["value"]["class"] == mapping["compiler"]
    assert (
        manifest_result["runtime_driver"]["value"]["class"]
        == mapping["driver"]
    )
    selections = manifest_result["first_diagnostics"]["selections"]
    assert selections["manifest_fixture"]["record_id"] == mapping["record_id"]
    records = manifest_result["first_diagnostics"]["plugins"]
    assert [record["state"] for record in records] == ["active"]


def test_triton_integration_manifest_registry_reset_invalidates_both_caches(
    manifest_result,
) -> None:
    assert manifest_result["reset_errors"] == [], manifest_result
    assert manifest_result["mapping_after_reset"] == {}, manifest_result
    assert manifest_result["compiler_after_reset"]["ok"], manifest_result
    assert manifest_result["runtime_after_reset"]["ok"], manifest_result
    assert set(manifest_result["mapping_after_reselection"]) == {
        "manifest_fixture"
    }
    assert manifest_result["selector_environment"] is None


def test_triton_integration_rejects_v30_abstract_classes_before_selection(
    v30_result,
) -> None:
    """New v3.3 abstract members are a load/register boundary, not JIT errors."""
    violations = []
    selection = v30_result["selection"]
    if selection["ok"]:
        violations.append("Registry.select accepted abstract compiler/driver")
    elif selection["type"] != "BackendPluginInterfaceError":
        violations.append(
            "Registry.select returned " + selection["type"]
            + " instead of BackendPluginInterfaceError"
        )

    compiler = v30_result["compiler"]
    if compiler["ok"] or compiler.get("type") == "TypeError":
        violations.append("compiler class was published and failed only at construction")

    runtime_driver = v30_result["runtime_driver"]
    if runtime_driver["ok"] or runtime_driver.get("type") in {
        "TypeError",
        "BackendPluginLifecycleError",
    }:
        violations.append("driver class reached active probe/construction")

    if "v30_fixture" in v30_result["mapping_after_late_failures"]:
        violations.append("invalid abstract pair leaked into public backends mapping")

    assert not violations, json.dumps(
        {"violations": violations, "probe": v30_result},
        indent=2,
        sort_keys=True,
    )


def test_triton_integration_v30_gap_is_exactly_the_pinned_v33_surface(
    v30_result,
) -> None:
    """Document the pinned 3.3 delta independently of Registry behavior."""
    assert v30_result["abstract_methods"] == {
        "compiler": ["get_module_map"],
        "driver": ["get_active_torch_device", "get_benchmarker"],
    }, v30_result


@pytest.mark.skipif(
    not os.environ.get("T63_CORE_WHEEL"),
    reason=(
        "BLOCKED: set T63_CORE_WHEEL to an installed Core 0.2.0 wheel for "
        "the hardware-free full-package compiler/cache smoke"
    ),
)
def test_triton_integration_installed_core_wheel_compiler_cache_smoke(
    integration_sites,
) -> None:
    core_wheel = Path(os.environ["T63_CORE_WHEEL"]).resolve()
    assert core_wheel.is_file(), f"T63_CORE_WHEEL does not exist: {core_wheel}"
    root = integration_sites["root"]
    environment_dir = root / "installed-core-venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(environment_dir)
    environment_python = environment_dir / "bin/python"

    installed = subprocess.run(
        [
            str(environment_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            str(core_wheel),
            str(integration_sites["manifest_wheel"]),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr

    ttir = root / "fixture_kernel.ttir"
    ttir.write_text(
        "module {\n"
        "  tt.func public @fixture_kernel() attributes {noinline = false} {\n"
        "    tt.return\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    cache = root / "isolated-triton-cache"
    counter = root / "installed-stage-count.txt"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("TRITON_ANCHOR_BACKEND", None)
    environment.update(
        {
            "TRITON_CACHE_DIR": str(cache),
            "T63_STAGE_COUNTER": str(counter),
            "T63_TTIR_SOURCE": str(ttir),
            "T63_PLUGIN_TARGET": "manifest_fixture",
        }
    )
    completed = subprocess.run(
        [str(environment_python), "-c", INSTALLED_CACHE_SMOKE],
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    print(
        "T63_INSTALLED_CACHE_EVIDENCE="
        + json.dumps(result, sort_keys=True)
    )

    purelib = environment_dir.resolve()
    assert Path(result["triton_path"]).resolve().is_relative_to(purelib), result
    assert Path(result["anchor_path"]).resolve().is_relative_to(purelib), result
    assert result["stage_counts"] == [1, 1, 1], result
    assert result["first_hash"] == result["second_hash"] == result["third_hash"]
    assert set(result["mapping"]) == {"manifest_fixture"}, result
    assert result["runtime_target"] == "manifest_fixture"
    assert result["active_state"] == "active"
    assert result["reset_errors"] == []
    assert result["mapping_after_reset"] == []
    assert result["mapping_after_reselection"] == ["manifest_fixture"]
    assert result["runtime_target_after_reset"] == "manifest_fixture"


@pytest.mark.skip(
    reason=(
        "BLOCKED: no physical backend device/runtime was supplied; actual "
        "JIT launch is intentionally not represented by source stubs"
    )
)
def test_triton_integration_actual_hardware_jit_launch() -> None:
    pass
