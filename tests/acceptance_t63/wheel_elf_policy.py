#!/usr/bin/env python3
"""Static ELF RPATH/RUNPATH policy helpers for the Triton 3.6 Core wheel."""

from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_ELF_MEMBERS = frozenset(
    {
        "triton/_C/libtriton.so",
        "triton/bin/triton-shared-opt",
    }
)
EXPECTED_NEEDED = frozenset(
    {
        "libz.so.1",
        "libstdc++.so.6",
        "libm.so.6",
        "libgcc_s.so.1",
        "libc.so.6",
        "ld-linux-x86-64.so.2",
    }
)
EXPECTED_RUNPATH = {
    "triton/_C/libtriton.so": (),
    "triton/bin/triton-shared-opt": ("$ORIGIN/../lib",),
}
_DYNAMIC_TAG = re.compile(
    r"^\s*0x[0-9A-Fa-f]+\s+\((NEEDED|RPATH|RUNPATH)\)\s+"
    r"(Shared library|Library rpath|Library runpath):\s+\[(.*)\]\s*$"
)
_DYNAMIC_LABELS = {
    "NEEDED": "Shared library",
    "RPATH": "Library rpath",
    "RUNPATH": "Library runpath",
}
_PT_INTERP = re.compile(r"\[Requesting program interpreter:\s*(.*?)\]\s*$")
_ORIGIN_TOKENS = ("$ORIGIN", "${ORIGIN}")
SHARED_OPT_CMAKE = Path("csrc/tools/triton-shared-opt/CMakeLists.txt")
_ALLOWED_SHARED_OPT_CMAKE_COMMANDS = frozenset(
    {
        "get_property",
        "add_llvm_executable",
        "set_target_properties",
        "set_property",
        "llvm_update_compile_flags",
        "target_link_libraries",
        "mlir_check_all_link_libraries",
    }
)


def normalize_origin_component(member: str, component: str) -> str:
    """Validate and lexically resolve one RUNPATH component within ``triton/``."""
    if not component:
        raise ValueError("empty RUNPATH component")
    if "\x00" in component:
        raise ValueError("RUNPATH component contains NUL")
    if "\\" in component:
        raise ValueError("RUNPATH component contains a backslash")

    token = next(
        (
            candidate
            for candidate in _ORIGIN_TOKENS
            if component == candidate or component.startswith(candidate + "/")
        ),
        None,
    )
    if token is None:
        raise ValueError("RUNPATH component is not rooted at an exact ORIGIN token")

    member_path = PurePosixPath(member)
    if member_path.is_absolute() or not member_path.parts:
        raise ValueError("ELF member path is not a relative wheel path")
    stack = list(member_path.parent.parts)
    if not stack or stack[0] != "triton":
        raise ValueError("ELF member is outside the wheel-owned triton tree")

    suffix = component[len(token) :]
    suffix = suffix.removeprefix("/")
    for part in suffix.split("/") if suffix else ():
        if part in {"", "."}:
            continue
        if part == "..":
            if len(stack) <= 1:
                raise ValueError("RUNPATH component escapes the triton tree")
            stack.pop()
            continue
        stack.append(part)
    if not stack or stack[0] != "triton":
        raise ValueError("RUNPATH component resolves outside the triton tree")
    return "/".join(stack)


def validate_needed_name(name: str) -> str:
    """Require a loader basename rather than a path-bearing DT_NEEDED value."""
    if not name or name in {".", ".."}:
        raise ValueError("DT_NEEDED name is not a basename")
    if "\x00" in name or "/" in name or "\\" in name:
        raise ValueError("DT_NEEDED name contains a path separator or NUL")
    return name


def parse_dynamic_tags(output: str) -> tuple[dict[str, list[str]], list[str]]:
    tags = {"NEEDED": [], "RPATH": [], "RUNPATH": []}
    marker_counts = {name: 0 for name in tags}
    for line in output.splitlines():
        for name in tags:
            if f"({name})" in line:
                marker_counts[name] += 1
        match = _DYNAMIC_TAG.search(line)
        if match is not None and match.group(2) == _DYNAMIC_LABELS[match.group(1)]:
            tags[match.group(1)].append(match.group(3))
    incomplete = sorted(name for name in tags if marker_counts[name] != len(tags[name]))
    return tags, incomplete


def _cmake_calls(source: str) -> list[tuple[str, tuple[str, ...]]]:
    calls: list[tuple[str, tuple[str, ...]]] = []
    for match in re.finditer(r"(?ms)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)", source):
        arguments = tuple(shlex.split(match.group(2), comments=True, posix=True))
        calls.append((match.group(1).lower(), arguments))
    return calls


def target_uses_build_install_rpath(source: str, target: str) -> bool:
    """Require one unconditional, non-overridden target-local ON assignment."""
    if re.search(r"#\[(=*)\[", source):
        return False
    calls = _cmake_calls(source)
    if any(
        command not in _ALLOWED_SHARED_OPT_CMAKE_COMMANDS
        for command, _arguments in calls
    ):
        return False

    assignments: list[str] = []
    for command, arguments in calls:
        if command == "set_target_properties" and "PROPERTIES" in arguments:
            boundary = arguments.index("PROPERTIES")
            if target not in arguments[:boundary]:
                if any("$" in item for item in arguments[:boundary]):
                    assignments.append("<INDIRECT-TARGET>")
                continue
            properties = arguments[boundary + 1 :]
            for index, name in enumerate(properties):
                if name != "BUILD_WITH_INSTALL_RPATH":
                    continue
                value_index = index + 1
                assignments.append(
                    properties[value_index].upper()
                    if value_index < len(properties)
                    else "<UNSET>"
                )
        if command == "set_property" and "BUILD_WITH_INSTALL_RPATH" in arguments:
            exact_prefix = (
                "TARGET",
                target,
                "PROPERTY",
                "BUILD_WITH_INSTALL_RPATH",
            )
            if arguments[:4] == exact_prefix and len(arguments) == 5:
                assignments.append(arguments[4].upper())
            else:
                assignments.append("<INDIRECT-OR-UNSET>")
    return assignments == ["ON"]


def audit_shared_opt_cmake(repo_root: Path) -> dict[str, object]:
    """Bind ELF-05 to the exact source target that setup.py copies."""
    path = (repo_root / SHARED_OPT_CMAKE).resolve(strict=True)
    source = path.read_text(encoding="utf-8")
    calls = _cmake_calls(source)
    definitions = [
        arguments
        for command, arguments in calls
        if command == "add_llvm_executable"
        and arguments
        and arguments[0] == "triton-shared-opt"
    ]
    enabled = target_uses_build_install_rpath(source, "triton-shared-opt")
    violations: list[str] = []
    if len(definitions) != 1:
        violations.append("triton-shared-opt target must be defined exactly once")
    if not enabled:
        violations.append(
            "triton-shared-opt must set BUILD_WITH_INSTALL_RPATH ON on its own target"
        )
    return {
        "path": SHARED_OPT_CMAKE.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "definition_count": len(definitions),
        "build_with_install_rpath": enabled,
        "violations": violations,
        "status": "PASS" if not violations else "FAIL",
    }


def _violation(member: str, rule: str, actual: object) -> dict[str, object]:
    return {"member": member, "rule": rule, "actual": actual}


def _audit_paths(paths: dict[str, Path]) -> dict[str, Any]:
    readelf = shutil.which("readelf")
    violations: list[dict[str, object]] = []
    audited: list[dict[str, object]] = []
    if readelf is None:
        return {
            "audited": audited,
            "violations": [_violation("<environment>", "readelf-unavailable", None)],
            "status": "BLOCKED",
        }

    for member in sorted(paths):
        path = paths[member]
        if not path.is_file():
            violations.append(_violation(member, "missing-installed-elf", str(path)))
            continue
        with path.open("rb") as stream:
            magic = stream.read(4)
        if magic != b"\x7fELF":
            violations.append(_violation(member, "invalid-elf-magic", magic.hex()))
            continue

        completed = subprocess.run(
            [readelf, "-dW", str(path)],
            env={"LC_ALL": "C"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        program_headers = subprocess.run(
            [readelf, "-lW", str(path)],
            env={"LC_ALL": "C"},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        tags, incomplete = parse_dynamic_tags(completed.stdout)
        interpreters = [
            match.group(1)
            for line in program_headers.stdout.splitlines()
            if (match := _PT_INTERP.search(line)) is not None
        ]
        item: dict[str, object] = {
            "member": member,
            "path": str(path.resolve()),
            "readelf_exit": completed.returncode,
            "program_headers_exit": program_headers.returncode,
            "pt_interp": interpreters,
            "needed": tags["NEEDED"],
            "rpath": tags["RPATH"],
            "runpath": tags["RUNPATH"],
        }
        audited.append(item)
        if completed.returncode != 0:
            violations.append(
                _violation(member, "readelf-failed", completed.returncode)
            )
        if program_headers.returncode != 0:
            violations.append(
                _violation(
                    member,
                    "program-headers-readelf-failed",
                    program_headers.returncode,
                )
            )
        if incomplete:
            violations.append(
                _violation(member, "dynamic-tag-parse-incomplete", incomplete)
            )
        if tags["RPATH"]:
            violations.append(_violation(member, "ELF-01-DT_RPATH", tags["RPATH"]))

        runpath_components: list[str] = []
        for value in tags["RUNPATH"]:
            runpath_components.extend(value.split(":"))
        for component in runpath_components:
            try:
                normalize_origin_component(member, component)
            except ValueError as error:
                violations.append(
                    _violation(
                        member,
                        "ELF-02-invalid-RUNPATH-component",
                        {"component": component, "error": str(error)},
                    )
                )
        expected_runpath = list(EXPECTED_RUNPATH.get(member, ()))
        if member in EXPECTED_RUNPATH and tags["RUNPATH"] != expected_runpath:
            violations.append(
                _violation(
                    member,
                    "ELF-04-exact-RUNPATH",
                    {"expected": expected_runpath, "actual": tags["RUNPATH"]},
                )
            )

        invalid_needed: list[dict[str, str]] = []
        for name in tags["NEEDED"]:
            try:
                validate_needed_name(name)
            except ValueError as error:
                invalid_needed.append({"name": name, "error": str(error)})
        if invalid_needed:
            violations.append(
                _violation(member, "ELF-06-invalid-DT_NEEDED", invalid_needed)
            )
        actual_needed = set(tags["NEEDED"])
        if member in EXPECTED_ELF_MEMBERS and actual_needed != set(EXPECTED_NEEDED):
            violations.append(
                _violation(
                    member,
                    "ELF-06-exact-DT_NEEDED",
                    {
                        "expected": sorted(EXPECTED_NEEDED),
                        "actual": sorted(actual_needed),
                    },
                )
            )

    violations.sort(key=lambda item: (str(item["member"]), str(item["rule"])))
    return {
        "audited": audited,
        "violations": violations,
        "status": "PASS" if not violations else "FAIL",
    }


def elf_members_by_magic(archive: zipfile.ZipFile) -> list[str]:
    """Enumerate ELF payload by bytes, including misleading directory entries."""
    members: list[str] = []
    for info in archive.infolist():
        with archive.open(info) as stream:
            if stream.read(4) == b"\x7fELF":
                members.append(info.filename)
    return sorted(members)


def audit_wheel_elf(wheel: Path) -> dict[str, Any]:
    """Audit every ELF member in a wheel without importing or loading it."""
    wheel = wheel.resolve(strict=True)
    with zipfile.ZipFile(wheel) as archive:
        elf_members = elf_members_by_magic(archive)
        member_violations: list[dict[str, object]] = []
        if set(elf_members) != set(EXPECTED_ELF_MEMBERS):
            member_violations.append(
                _violation(
                    "<archive>",
                    "ELF-00-exact-member-set",
                    {
                        "expected": sorted(EXPECTED_ELF_MEMBERS),
                        "actual": elf_members,
                    },
                )
            )

        with tempfile.TemporaryDirectory(prefix="t63-wheel-elf-") as temporary:
            root = Path(temporary)
            paths: dict[str, Path] = {}
            for index, member in enumerate(elf_members):
                target = root / f"member-{index:04d}.elf"
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                paths[member] = target
            evidence = _audit_paths(paths)

    evidence["wheel"] = str(wheel)
    evidence["elf_members"] = elf_members
    evidence["violations"] = sorted(
        member_violations + evidence["violations"],
        key=lambda item: (str(item["member"]), str(item["rule"])),
    )
    if member_violations:
        evidence["status"] = "FAIL"
    elif evidence["status"] != "BLOCKED":
        evidence["status"] = "PASS" if not evidence["violations"] else "FAIL"
    return evidence


def audit_installed_elf(paths: dict[str, Path]) -> dict[str, Any]:
    """Apply the same policy to exact files resolved from fresh site-packages."""
    actual_members = set(paths)
    if actual_members != set(EXPECTED_ELF_MEMBERS):
        return {
            "audited": [],
            "violations": [
                _violation(
                    "<installed>",
                    "ELF-00-exact-member-set",
                    {
                        "expected": sorted(EXPECTED_ELF_MEMBERS),
                        "actual": sorted(actual_members),
                    },
                )
            ],
            "status": "FAIL",
        }
    return _audit_paths(paths)
