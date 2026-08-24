"""Triton 3.6 F4 source/discovery/wheel/install package inventory."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from tests.acceptance_t63.wheel_elf_policy import (
    audit_installed_elf,
    audit_shared_opt_cmake,
    audit_wheel_elf,
    elf_members_by_magic,
    normalize_origin_component,
    parse_dynamic_tags,
    target_uses_build_install_rpath,
    validate_needed_name,
)
from tests.acceptance_t63.wheel_package_inventory import (
    EXPECTED_ENTRY_POINT,
    FORBIDDEN_PATH_PARTS,
    SMOKE_IDENTITIES,
    SMOKE_SCHEMA,
    capture_setup_configuration,
    expected_release_file_members,
    install_and_probe,
    release_member_delta,
    source_package_inventory,
    validate_requires_dist,
    validate_wheel_archive,
    validate_zip_members,
    wheel_release_member_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FOCUS_PACKAGE = "triton_anchor.language.ext"
FOCUS_SOURCE = REPO_ROOT / "python/triton_anchor/language/ext/__init__.py"


def _supplied_wheel() -> Path:
    value = os.environ.get("T63_CORE_WHEEL")
    if not value:
        pytest.skip("T63_CORE_WHEEL was not supplied; current Core wheel unavailable")
    return Path(value).resolve(strict=True)


@pytest.fixture(scope="module")
def fresh_install_evidence(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    return install_and_probe(
        _supplied_wheel(),
        REPO_ROOT,
        tmp_path_factory.mktemp("t63-f4-fresh-install"),
    )


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
    fresh_install_evidence: dict[str, object],
) -> None:
    evidence = fresh_install_evidence
    assert evidence["installed_status"] == "PASS"
    assert evidence["pip_install_exit"] == 0
    assert evidence["pip_check_exit"] == 0
    assert evidence["import_probe_exit"] == 0
    assert evidence["installed_smoke_exit"] == 0
    assert evidence["installed_smoke_results"] == [
        {"schema": SMOKE_SCHEMA, "identity": identity, "status": "PASS"}
        for identity in SMOKE_IDENTITIES
    ]
    assert evidence["installed_smoke_summary"] == {
        "schema": SMOKE_SCHEMA,
        "ordered_identities": list(SMOKE_IDENTITIES),
        "pass": 10,
        "fail": 0,
        "skip": 0,
        "total": 10,
        "driver_sha256": evidence["installed_smoke_summary"]["driver_sha256"],
    }
    assert evidence["pip_check_output"] == "No broken requirements found."
    assert "site-packages" in evidence["module_paths"]["triton_anchor.language.ext"]
    print("F4_INSTALLED_AUDIT=" + json.dumps(evidence, sort_keys=True))


def test_release_package_policy_excludes_anchor_tests() -> None:
    source = source_package_inventory(REPO_ROOT)
    configuration = capture_setup_configuration(REPO_ROOT)
    declared = set(configuration["packages"])
    source_tests = sorted(
        package
        for package in source
        if package == "triton_anchor.tests"
        or package.startswith("triton_anchor.tests.")
    )
    declared_tests = sorted(
        package
        for package in declared
        if package == "triton_anchor.tests"
        or package.startswith("triton_anchor.tests.")
    )
    assert {"source_tests": source_tests, "declared_tests": declared_tests} == {
        "source_tests": [],
        "declared_tests": [],
    }


def test_release_member_equation_rejects_unknown_payload() -> None:
    expected = expected_release_file_members(REPO_ROOT)
    extras = {
        "triton/leaked-source.cpp",
        "triton/unknown-payload",
        "triton_anchor/tests/test_leak.py",
    }
    delta = release_member_delta(expected | extras, expected)
    assert delta == {
        "missing": [],
        "unexpected": sorted(extras),
        "tests": ["triton_anchor/tests/test_leak.py"],
    }


def test_metadata_dependency_policy_rejects_extra_or_weakened_requirements() -> None:
    assert [str(item) for item in validate_requires_dist(["packaging>=21"])] == [
        "packaging>=21"
    ]
    for invalid in (
        [],
        ["packaging>=21", "requests>=1"],
        ["packaging>=21", "packaging>=21"],
        ["packaging[extra]>=21"],
        ['packaging>=21; python_version >= "3.10"'],
        ["packaging @ https://example.invalid/packaging.whl"],
    ):
        with pytest.raises(AssertionError):
            validate_requires_dist(invalid)


def test_current_core_wheel_exact_payload_and_tests_absent() -> None:
    evidence = wheel_release_member_evidence(_supplied_wheel(), REPO_ROOT)
    print("F4_EXACT_MEMBER_AUDIT=" + json.dumps(evidence, sort_keys=True))
    assert {
        "missing": evidence["missing"],
        "unexpected": evidence["unexpected"],
        "tests": evidence["tests"],
    } == {"missing": [], "unexpected": [], "tests": []}


def test_elf_policy_rejects_adversarial_origin_and_needed_values(
    tmp_path: Path,
) -> None:
    member = "triton/bin/triton-shared-opt"
    assert normalize_origin_component(member, "$ORIGIN/../lib") == "triton/lib"
    assert normalize_origin_component(member, "${ORIGIN}") == "triton/bin"
    for invalid in (
        "",
        "$ORIGINevil/lib",
        "/absolute/lib",
        "$ORIGIN/../../outside",
        "$ORIGIN\\..\\lib",
        "$ORIGIN/evil\x00lib",
    ):
        with pytest.raises(ValueError):
            normalize_origin_component(member, invalid)

    assert validate_needed_name("libz.so.1") == "libz.so.1"
    for invalid in (
        "",
        "/tmp/libLLVM.so",
        "dir/libLLVM.so",
        "dir\\libLLVM.so",
        "bad\x00.so",
    ):
        with pytest.raises(ValueError):
            validate_needed_name(invalid)

    tags, incomplete = parse_dynamic_tags(
        " 0x000000000000001d (RUNPATH) Library runpath: [$ORIGIN/../lib]junk]\n"
    )
    assert incomplete == []
    assert tags["RUNPATH"] == ["$ORIGIN/../lib]junk"]
    tags, incomplete = parse_dynamic_tags(
        " 0x000000000000001d (RUNPATH) Library runpath: [$ORIGINevil[$ORIGIN/../lib]\n"
    )
    assert incomplete == []
    assert tags["RUNPATH"] == ["$ORIGINevil[$ORIGIN/../lib"]
    with pytest.raises(ValueError):
        normalize_origin_component(member, tags["RUNPATH"][0])

    assert target_uses_build_install_rpath(
        "set_target_properties(triton-shared-opt PROPERTIES "
        "BUILD_WITH_INSTALL_RPATH ON)\n",
        "triton-shared-opt",
    )
    current_cmake = (
        REPO_ROOT / "csrc/tools/triton-shared-opt/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    without_policy = current_cmake.replace("  BUILD_WITH_INSTALL_RPATH ON\n", "")
    configured_cmake = without_policy.replace(
        "set_target_properties(triton-shared-opt PROPERTIES",
        "set_target_properties(triton-shared-opt PROPERTIES "
        "BUILD_WITH_INSTALL_RPATH ON",
        1,
    )
    assert configured_cmake != without_policy
    assert target_uses_build_install_rpath(
        configured_cmake,
        "triton-shared-opt",
    )
    assert not target_uses_build_install_rpath(
        "# BUILD_WITH_INSTALL_RPATH ON\n"
        "set_target_properties(other-target PROPERTIES "
        "BUILD_WITH_INSTALL_RPATH ON)\n",
        "triton-shared-opt",
    )
    for ineffective in (
        (
            "set_target_properties(triton-shared-opt PROPERTIES "
            "BUILD_WITH_INSTALL_RPATH ON)\n"
            "set_target_properties(triton-shared-opt PROPERTIES "
            "BUILD_WITH_INSTALL_RPATH OFF)\n"
        ),
        (
            "if(FALSE)\n"
            "set_target_properties(triton-shared-opt PROPERTIES "
            "BUILD_WITH_INSTALL_RPATH ON)\n"
            "endif()\n"
        ),
        (
            "function(configure_later)\n"
            "set_target_properties(triton-shared-opt PROPERTIES "
            "BUILD_WITH_INSTALL_RPATH ON)\n"
            "endfunction()\n"
        ),
        (
            "set_target_properties(triton-shared-opt PROPERTIES "
            "BUILD_WITH_INSTALL_RPATH ON)\n"
            "set_property(TARGET triton-shared-opt PROPERTY "
            "BUILD_WITH_INSTALL_RPATH)\n"
        ),
        (
            "set_target_properties(triton-shared-opt PROPERTIES "
            "BUILD_WITH_INSTALL_RPATH ON)\n"
            "include(maybe-overrides-rpath.cmake)\n"
        ),
        (
            "set_target_properties(triton-shared-opt PROPERTIES "
            "BUILD_WITH_INSTALL_RPATH ON)\n"
            "set_property(TARGET ${target_name} PROPERTY "
            "BUILD_WITH_INSTALL_RPATH OFF)\n"
        ),
    ):
        assert not target_uses_build_install_rpath(
            ineffective,
            "triton-shared-opt",
        )

    misleading_archive = tmp_path / "directory-payload.zip"
    with zipfile.ZipFile(misleading_archive, "w") as archive:
        archive.writestr("triton/", b"\x7fELF-directory-payload")
    with zipfile.ZipFile(misleading_archive) as archive:
        assert elf_members_by_magic(archive) == ["triton/"]
        with pytest.raises(AssertionError, match="non-empty"):
            validate_zip_members(archive)


def test_shared_opt_target_uses_install_rpath_at_build_time() -> None:
    evidence = audit_shared_opt_cmake(REPO_ROOT)
    print("F4_CMAKE_RPATH_AUDIT=" + json.dumps(evidence, sort_keys=True))
    assert evidence["status"] == "PASS", json.dumps(
        evidence["violations"], sort_keys=True
    )


def test_current_core_wheel_elf_dynamic_policy() -> None:
    evidence = audit_wheel_elf(_supplied_wheel())
    print("F4_ELF_AUDIT=" + json.dumps(evidence, sort_keys=True))
    assert evidence["status"] == "PASS", json.dumps(
        evidence["violations"], sort_keys=True
    )


def test_current_core_wheel_fresh_tests_absent_and_installed_binary(
    fresh_install_evidence: dict[str, object],
) -> None:
    evidence = fresh_install_evidence
    shared_opt = Path(str(evidence["shared_opt_path"])).resolve(strict=True)
    libtriton = Path(str(evidence["module_paths"]["libtriton"])).resolve(strict=True)
    site_packages = [Path(str(path)).resolve() for path in evidence["site_packages"]]
    installed_elf = audit_installed_elf(
        {
            "triton/_C/libtriton.so": libtriton,
            "triton/bin/triton-shared-opt": shared_opt,
        }
    )

    violations: list[dict[str, object]] = []
    if evidence["tests_spec"] is not None:
        violations.append(
            {"rule": "F4-13-tests-spec", "actual": evidence["tests_spec"]}
        )
    if not any(
        root == shared_opt or root in shared_opt.parents for root in site_packages
    ):
        violations.append(
            {"rule": "ELF-07-site-packages-path", "actual": str(shared_opt)}
        )
    expected_command = [str(shared_opt), "--version"]
    if evidence["shared_opt_command"] != expected_command:
        violations.append(
            {
                "rule": "ELF-07-exact-command",
                "expected": expected_command,
                "actual": evidence["shared_opt_command"],
            }
        )
    if evidence["loader_environment_absent"] != {
        "LD_LIBRARY_PATH": True,
        "LD_PRELOAD": True,
        "LD_AUDIT": True,
    }:
        violations.append(
            {
                "rule": "ELF-07-loader-environment",
                "actual": evidence["loader_environment_absent"],
            }
        )
    if evidence["shared_opt_exit"] != 0 or "LLVM version 22.0.0git" not in str(
        evidence["shared_opt_output"]
    ):
        violations.append(
            {
                "rule": "ELF-07-version-smoke",
                "exit": evidence["shared_opt_exit"],
                "output": evidence["shared_opt_output"],
            }
        )
    violations.extend(
        {"rule": "installed-elf", **violation}
        for violation in installed_elf["violations"]
    )
    result = {
        "entry_points": evidence["entry_points"],
        "expected_entry_point": list(EXPECTED_ENTRY_POINT),
        "installed_elf": installed_elf,
        "loader_environment_absent": evidence["loader_environment_absent"],
        "shared_opt_command": evidence["shared_opt_command"],
        "shared_opt_exit": evidence["shared_opt_exit"],
        "shared_opt_output": evidence["shared_opt_output"],
        "tests_spec": evidence["tests_spec"],
        "violations": violations,
    }
    print("F4_FRESH_POLICY_AUDIT=" + json.dumps(result, sort_keys=True))
    assert violations == [], json.dumps(violations, sort_keys=True)
