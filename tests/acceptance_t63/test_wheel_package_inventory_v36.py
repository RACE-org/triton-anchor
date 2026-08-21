"""Triton 3.6 F4 source/discovery/wheel/install package inventory."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from wheel_package_inventory import (  # noqa: E402
    FORBIDDEN_PATH_PARTS,
    capture_setup_configuration,
    install_and_probe,
    source_package_inventory,
    validate_wheel_archive,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FOCUS_PACKAGE = "triton_anchor.language.ext"
FOCUS_SOURCE = REPO_ROOT / "python/triton_anchor/language/ext/__init__.py"


def _supplied_wheel() -> Path:
    value = os.environ.get("T63_CORE_WHEEL")
    if not value:
        pytest.skip("T63_CORE_WHEEL was not supplied; current Core wheel unavailable")
    return Path(value).resolve(strict=True)


def test_source_inventory_matches_actual_setup_discovery() -> None:
    source = source_package_inventory(REPO_ROOT)
    configuration = capture_setup_configuration(REPO_ROOT)
    declared = set(configuration["packages"])

    assert FOCUS_SOURCE.is_file()
    assert FOCUS_PACKAGE in source
    assert FOCUS_PACKAGE in declared
    assert source - declared == set()
    assert declared - source == set()
    assert source
    assert configuration["package_dir"] == {
        "": "python",
        "triton": "triton/python/triton",
    }
    assert configuration["include_package_data"] is True
    assert configuration["install_requires"] == ["packaging>=21"]
    assert not {
        package
        for package in declared
        if FORBIDDEN_PATH_PARTS.intersection(package.split("."))
    }


def test_current_core_wheel_inventory_metadata_and_record() -> None:
    evidence = validate_wheel_archive(_supplied_wheel(), REPO_ROOT)
    assert evidence["archive_status"] == "PASS"
    assert evidence["source_package_count"] > 0
    assert evidence["source_package_count"] == evidence["declared_package_count"]
    assert evidence["declared_package_count"] == evidence["wheel_package_count"]
    assert evidence["a_minus_b"] == []
    assert evidence["b_minus_c"] == []
    assert evidence["a_minus_c"] == []
    print("F4_WHEEL_AUDIT=" + json.dumps(evidence, sort_keys=True))


def test_current_core_wheel_fresh_isolated_import(
    tmp_path: Path,
) -> None:
    evidence = install_and_probe(_supplied_wheel(), REPO_ROOT, tmp_path)
    assert evidence["installed_status"] == "PASS"
    assert evidence["pip_install_exit"] == 0
    assert evidence["pip_check_exit"] == 0
    assert evidence["import_probe_exit"] == 0
    assert evidence["installed_smoke_exit"] == 0
    assert evidence["pip_check_output"] == "No broken requirements found."
    assert "site-packages" in evidence["module_paths"]["triton_anchor.language.ext"]
    print("F4_INSTALLED_AUDIT=" + json.dumps(evidence, sort_keys=True))
