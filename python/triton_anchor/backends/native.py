"""Import-free, evidence-only inspection of backend wheel native files.

Backend Plugin Protocol/Manifest 1.0 authorizes only ``python_only`` plugins.
The public inspector therefore has two deliberately separate uses: it proves
that an owning ``python_only`` distribution contains no native binaries, and
it lets offline conformance/conflict tooling collect static facts from a
manually constructed native record.  Those facts never authorize importing,
loading, selecting, or activating native code.

Inventory is intentionally fail-closed.  Every owning-distribution RECORD
entry must resolve to a readable regular file beneath a verified install-
scheme root before filename and file-magic classification can be trusted.
Static native artifact parsing currently supports Linux ELF; other formats
cannot produce an evidence report until their parsers and CI coverage exist.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import platform
import re
import shutil
import subprocess
import sysconfig
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .errors import (
    BackendPluginCompatibilityError,
    BackendPluginManifestError,
)
from .manifest import BackendPluginManifest
from .protocol import PluginIsolationMode


_NATIVE_SUFFIXES = (".dylib", ".dll", ".pyd", ".so")
_ELF_MAGIC = b"\x7fELF"
_MACHO_MAGICS = {
    b"\xbe\xba\xfe\xca",
    b"\xbf\xba\xfe\xca",
    b"\xca\xfe\xba\xbe",
    b"\xca\xfe\xba\xbf",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
}
_RECORD_HASH_PATTERN = re.compile(r"^sha256=([A-Za-z0-9_-]+)$")
_SONAME_PATTERN = re.compile(
    r"\(SONAME\).*Library soname: \[([^\]]+)\]"
)
_NEEDED_PATTERN = re.compile(
    r"\(NEEDED\).*Shared library: \[([^\]]+)\]"
)
_FORBIDDEN_TOOLCHAIN_DEPENDENCY = re.compile(
    r"^(?:lib)?(?:LLVM|MLIR)(?:[-.]|$)",
    flags=re.IGNORECASE,
)
_INSTALL_SCHEME_ROOT_KEYS = ("purelib", "platlib", "scripts", "data")
_INSTALL_SCHEME_SITE_KEYS = ("purelib", "platlib")


@dataclass(frozen=True)
class NativeArtifact:
    """Normalized facts about one verified plugin native binary."""

    path: str
    sha256: str
    binary_format: str
    architecture: str
    identity: Optional[str]
    needed_libraries: Tuple[str, ...]
    exported_symbols: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "binary_format": self.binary_format,
            "architecture": self.architecture,
            "identity": self.identity,
            "needed_libraries": list(self.needed_libraries),
            "exported_symbols": list(self.exported_symbols),
        }


@dataclass(frozen=True)
class NativeInspectionReport:
    """Verified native artifacts for one Manifest record."""

    artifacts: Tuple[NativeArtifact, ...] = ()

    @property
    def paths(self) -> Tuple[str, ...]:
        return tuple(artifact.path for artifact in self.artifacts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


def _native_filename(path: str) -> bool:
    name = PurePosixPath(path).name.lower()
    return (
        name.endswith(_NATIVE_SUFFIXES)
        or ".so." in name
        or ".dylib." in name
    )


def _record_hashes(distribution: Any) -> Mapping[str, str]:
    read_text = getattr(distribution, "read_text", None)
    if not callable(read_text):
        return {}
    try:
        text = read_text("RECORD")
    except (AttributeError, OSError, TypeError, ValueError):
        return {}
    if not text:
        return {}

    hashes = {}
    try:
        rows = csv.reader(io.StringIO(text))
        for row in rows:
            if len(row) >= 2 and row[0]:
                hashes[str(PurePosixPath(row[0]))] = row[1]
    except (csv.Error, TypeError):
        return {}
    return hashes


def _package_path_hash(item: Any) -> Optional[str]:
    try:
        value = item.hash
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    if value is None:
        return None
    mode = getattr(value, "mode", None)
    digest = getattr(value, "value", None)
    if isinstance(mode, str) and isinstance(digest, str):
        return "{}={}".format(mode, digest)
    return str(value)


def _stable_exception_name(error: BaseException) -> str:
    error_type = type(error)
    return f"{error_type.__module__}.{error_type.__qualname__}"


def _distribution_file_map(
    plugin: BackendPluginManifest,
    distribution: Any,
) -> Mapping[str, Any]:
    try:
        files = distribution.files
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise BackendPluginManifestError(
            "Cannot inspect backend native files because the wheel RECORD "
            "file list is unreadable",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            field="distribution.files",
            expected="readable wheel RECORD entries",
            actual="<error: {}>".format(_stable_exception_name(exc)),
            remediation=(
                "Reinstall the backend from a standards-compliant wheel with "
                "a complete RECORD."
            ),
        ) from exc
    if files is None:
        raise BackendPluginManifestError(
            "Cannot inspect backend native files because the wheel RECORD "
            "file list is unavailable",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            field="distribution.files",
            expected="readable wheel RECORD entries",
            actual="<unavailable>",
            remediation=(
                "Reinstall the backend from a standards-compliant wheel with "
                "a complete RECORD."
            ),
        )
    result = {}
    try:
        for item in files:
            normalized = str(PurePosixPath(str(item)))
            if normalized in result:
                raise BackendPluginManifestError(
                    "Backend wheel RECORD contains a duplicate path: "
                    + normalized,
                    plugin_id=plugin.plugin_id,
                    entry_point=plugin.entry_point,
                    field="distribution.files",
                    expected="one RECORD entry per installed path",
                    actual=normalized,
                    remediation="Rebuild the backend wheel with a valid RECORD.",
                )
            result[normalized] = item
    except BackendPluginManifestError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise BackendPluginManifestError(
            "Cannot normalize backend wheel RECORD entries",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            field="distribution.files",
            expected="relative POSIX package paths",
            actual="<error: {}>".format(_stable_exception_name(exc)),
            remediation="Rebuild and reinstall the backend wheel.",
        ) from exc
    return result


def _inventory_error(
    plugin: BackendPluginManifest,
    message: str,
    *,
    actual: str,
    detail: Optional[str] = None,
) -> BackendPluginManifestError:
    """Build one stable error for an untrustworthy wheel inventory."""
    return BackendPluginManifestError(
        message,
        plugin_id=plugin.plugin_id,
        entry_point=plugin.entry_point,
        detail=detail,
        field="distribution.files",
        expected=(
            "readable regular files contained by the owning distribution's "
            "verified install-scheme roots"
        ),
        actual=actual,
        remediation=(
            "Reinstall the backend from a standards-compliant wheel with a "
            "complete RECORD and no RECORD-entry path escapes or symbolic "
            "links."
        ),
    )


def _resolved_sysconfig_path(value: Any) -> Optional[Path]:
    """Resolve one existing absolute sysconfig directory, or ignore it."""
    if not isinstance(value, str) or not value:
        return None
    try:
        path = Path(value)
        if not path.is_absolute():
            return None
        resolved = path.resolve(strict=True)
        if not resolved.is_dir():
            return None
    except (OSError, ValueError):
        return None
    return resolved


def _sysconfig_install_schemes() -> Tuple[Mapping[str, str], ...]:
    """Return the finite set of install layouts known to this interpreter."""
    schemes = []
    seen = set()
    names = tuple(sysconfig.get_scheme_names())

    # The default call is retained separately: vendors may expose a preferred
    # scheme without including its alias in get_scheme_names().
    for name in (None,) + names:
        paths = (
            sysconfig.get_paths()
            if name is None
            else sysconfig.get_paths(scheme=name)
        )
        if not isinstance(paths, Mapping):
            continue
        signature = tuple(paths.get(key) for key in _INSTALL_SCHEME_ROOT_KEYS)
        if signature in seen:
            continue
        seen.add(signature)
        schemes.append(paths)
    return tuple(schemes)


def _distribution_install_roots(root: Path) -> Tuple[Path, ...]:
    """Derive bounded RECORD roots for the distribution's install scheme.

    ``PathDistribution.locate_file("")`` is the site/purelib/platlib root.
    Wheel installers may place scripts and data elsewhere in the same scheme,
    leaving paths such as ``../../../bin/tool`` in RECORD.  Those sibling
    roots are trusted only when this distribution root matches a purelib or
    platlib root reported by sysconfig; an unrelated distribution location
    therefore cannot use ``..`` to claim files from the active environment.
    """
    allowed = {root}
    for paths in _sysconfig_install_schemes():
        resolved = {
            key: _resolved_sysconfig_path(paths.get(key))
            for key in _INSTALL_SCHEME_ROOT_KEYS
        }
        if root not in {
            resolved[key]
            for key in _INSTALL_SCHEME_SITE_KEYS
            if resolved[key] is not None
        }:
            continue
        allowed.update(
            path for path in resolved.values() if path is not None
        )
    return tuple(sorted(allowed, key=os.fspath))


def _is_within_install_roots(path: Path, roots: Iterable[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def _has_native_magic(
    plugin: BackendPluginManifest,
    relative_path: str,
    path: Path,
) -> bool:
    try:
        with path.open("rb") as stream:
            magic = stream.read(4)
    except (OSError, ValueError) as exc:
        raise _inventory_error(
            plugin,
            "Cannot read wheel file while checking for native binary magic: "
            "'{}'".format(relative_path),
            actual="{}: <error: {}>".format(
                relative_path,
                _stable_exception_name(exc),
            ),
            detail=relative_path,
        ) from exc
    return (
        magic == _ELF_MAGIC
        or magic in _MACHO_MAGICS
        or magic[:2] == b"MZ"
    )


def _discover_native_paths(
    plugin: BackendPluginManifest,
    distribution: Any,
    file_map: Mapping[str, Any],
) -> Tuple[str, ...]:
    """Find native binaries without trusting incomplete filesystem evidence."""
    result = set()
    try:
        located_root = Path(distribution.locate_file(""))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise _inventory_error(
            plugin,
            "Cannot locate the owning distribution root for native-file "
            "inventory",
            actual="<error: {}>".format(_stable_exception_name(exc)),
        ) from exc

    try:
        root = located_root.resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise _inventory_error(
            plugin,
            "Cannot resolve the owning distribution root for native-file "
            "inventory",
            actual="<error: {}>".format(_stable_exception_name(exc)),
        ) from exc
    allowed_roots = _distribution_install_roots(root)

    for relative_path in sorted(file_map):
        record_item = file_map[relative_path]
        record_path = PurePosixPath(relative_path)
        if record_path.is_absolute() or "\\" in relative_path:
            raise _inventory_error(
                plugin,
                "Wheel RECORD path is not a relative POSIX install path: "
                "'{}'".format(relative_path),
                actual=relative_path,
                detail=relative_path,
            )

        try:
            candidate = Path(distribution.locate_file(record_item))
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            raise _inventory_error(
                plugin,
                "Cannot locate wheel RECORD entry while checking for native "
                "files: '{}'".format(relative_path),
                actual="{}: <error: {}>".format(
                    relative_path,
                    _stable_exception_name(exc),
                ),
                detail=relative_path,
            ) from exc

        try:
            is_symlink = candidate.is_symlink()
        except (OSError, ValueError) as exc:
            raise _inventory_error(
                plugin,
                "Cannot determine whether wheel RECORD entry is a symbolic "
                "link: '{}'".format(relative_path),
                actual="{}: <error: {}>".format(
                    relative_path,
                    _stable_exception_name(exc),
                ),
                detail=relative_path,
            ) from exc
        if is_symlink:
            raise _inventory_error(
                plugin,
                "Wheel RECORD entry cannot be a symbolic link: '{}'".format(
                    relative_path
                ),
                actual=relative_path,
                detail=relative_path,
            )

        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, ValueError) as exc:
            raise _inventory_error(
                plugin,
                "Wheel RECORD entry is missing or cannot be resolved: '{}'".format(
                    relative_path
                ),
                actual="{}: <error: {}>".format(
                    relative_path,
                    _stable_exception_name(exc),
                ),
                detail=relative_path,
            ) from exc
        if not _is_within_install_roots(resolved, allowed_roots):
            raise _inventory_error(
                plugin,
                "Wheel RECORD entry escapes the owning distribution's "
                "verified install-scheme roots: '{}'".format(relative_path),
                actual=relative_path,
                detail=relative_path,
            )

        try:
            is_file = resolved.is_file()
        except (OSError, ValueError) as exc:
            raise _inventory_error(
                plugin,
                "Cannot determine the file type of wheel RECORD entry: "
                "'{}'".format(relative_path),
                actual="{}: <error: {}>".format(
                    relative_path,
                    _stable_exception_name(exc),
                ),
                detail=relative_path,
            ) from exc
        if not is_file:
            raise _inventory_error(
                plugin,
                "Wheel RECORD entry is not a regular file: '{}'".format(
                    relative_path
                ),
                actual=relative_path,
                detail=relative_path,
            )

        # Filename classification remains an independent conservative signal;
        # renamed binaries are additionally caught by the content signature.
        # Read every entry even when its filename is conclusive so unreadable
        # wheel contents cannot be hidden behind a native-looking suffix.
        has_native_magic = _has_native_magic(
            plugin,
            relative_path,
            resolved,
        )
        if _native_filename(relative_path) or has_native_magic:
            result.add(relative_path)
    return tuple(sorted(result))


def _validate_native_wheel_layout(
    plugin: BackendPluginManifest,
    distribution: Any,
) -> None:
    read_text = getattr(distribution, "read_text", None)
    try:
        wheel_text = read_text("WHEEL") if callable(read_text) else None
        headers = Parser().parsestr(wheel_text or "")
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise BackendPluginCompatibilityError(
            "native wheel layout",
            "readable WHEEL metadata",
            "<error: {}>".format(exc),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation="Rebuild the backend as a standards-compliant wheel.",
        ) from exc

    purelib = (headers.get("Root-Is-Purelib") or "").strip().lower()
    raw_tags = tuple(headers.get_all("Tag", ()))
    platform_tags = tuple(
        tag.rsplit("-", 1)[-1].strip().lower()
        for tag in raw_tags
        if isinstance(tag, str) and "-" in tag
    )
    if purelib != "false" or not platform_tags or all(
        tag == "any" for tag in platform_tags
    ):
        raise BackendPluginCompatibilityError(
            "native wheel layout",
            "Root-Is-Purelib: false and a platform-specific wheel Tag",
            "Root-Is-Purelib={}, Tag={}".format(
                purelib or "<missing>",
                ",".join(raw_tags) or "<missing>",
            ),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "For offline static evidence, use a platform-specific wheel. "
                "Protocol/Manifest 1.0 still does not authorize loading it."
            ),
        )


def _locate_verified_file(
    plugin: BackendPluginManifest,
    distribution: Any,
    relative_path: str,
    record_item: Any,
    record_hashes: Mapping[str, str],
) -> Tuple[Path, str]:
    try:
        root = Path(distribution.locate_file("")).resolve(strict=True)
        installed = Path(distribution.locate_file(relative_path))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise BackendPluginManifestError(
            "Cannot locate declared native library '{}'".format(relative_path),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            field="native_libraries",
            expected="an installed file inside the owning distribution",
            actual="<error: {}>".format(exc),
            remediation="Reinstall the backend wheel and retry discovery.",
        ) from exc

    if installed.is_symlink():
        raise BackendPluginManifestError(
            "Declared native library cannot be a symbolic link: '{}'".format(
                relative_path
            ),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            field="native_libraries",
            expected="a regular in-wheel file, not a symbolic link",
            actual=relative_path,
            remediation=(
                "For offline evidence, package the native binary itself in "
                "the wheel and regenerate RECORD. This does not authorize "
                "loading."
            ),
        )
    try:
        resolved = installed.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise BackendPluginManifestError(
            "Declared native library escapes or is missing from the installed "
            "distribution: '{}'".format(relative_path),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            field="native_libraries",
            expected="a regular file contained by the distribution root",
            actual=relative_path,
            remediation=(
                "Remove symlink/path escapes from the offline evidence wheel. "
                "Static verification does not authorize loading."
            ),
        ) from exc
    if not resolved.is_file():
        raise BackendPluginManifestError(
            "Declared native library is not a regular file: '{}'".format(
                relative_path
            ),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            field="native_libraries",
            expected="an installed regular file",
            actual=relative_path,
            remediation="Rebuild and reinstall the backend wheel.",
        )

    expected_hash = _package_path_hash(record_item)
    if expected_hash is None:
        expected_hash = record_hashes.get(relative_path)
    match = _RECORD_HASH_PATTERN.fullmatch(expected_hash or "")
    if match is None:
        raise BackendPluginManifestError(
            "Declared native library lacks a SHA-256 wheel RECORD hash: "
            "'{}'".format(relative_path),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            field="RECORD",
            expected="sha256=<urlsafe-base64-digest>",
            actual=expected_hash or "<missing>",
            remediation=(
                "Build a standards-compliant wheel whose RECORD hashes every "
                "native artifact with SHA-256."
            ),
        )

    try:
        content = resolved.read_bytes()
    except OSError as exc:
        raise BackendPluginManifestError(
            "Cannot read declared native library '{}'".format(relative_path),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            field="native_libraries",
            expected="a readable installed regular file",
            actual="<error: {}>".format(exc),
            remediation="Reinstall the backend wheel and verify permissions.",
        ) from exc
    digest_bytes = hashlib.sha256(content).digest()
    actual_record_digest = base64.urlsafe_b64encode(digest_bytes).decode(
        "ascii"
    ).rstrip("=")
    if actual_record_digest != match.group(1):
        raise BackendPluginManifestError(
            "Wheel RECORD hash mismatch for native library '{}'".format(
                relative_path
            ),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            field="RECORD",
            expected=expected_hash,
            actual="sha256={}".format(actual_record_digest),
            remediation=(
                "Discard the modified installation and reinstall the original "
                "backend wheel."
            ),
        )
    return resolved, hashlib.sha256(content).hexdigest()


def _run_tool(
    executable: str,
    arguments: Iterable[str],
    path: Path,
    plugin: BackendPluginManifest,
) -> str:
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            [executable] + list(arguments) + [str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            universal_newlines=True,
            env=environment,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise BackendPluginCompatibilityError(
            "native binary inspection",
            "a successful import-free binary inspection",
            "<error: {}>".format(exc),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Install readelf/nm (or the LLVM equivalents) in the validation "
                "environment; static native evidence cannot be collected "
                "without them."
            ),
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise BackendPluginCompatibilityError(
            "native binary inspection",
            "a successful import-free binary inspection",
            "{} exited {}: {}".format(
                Path(executable).name,
                completed.returncode,
                detail or "<no diagnostic>",
            ),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Rebuild the native library as a valid shared object for the "
                "current platform."
            ),
        )
    return completed.stdout


def _normalize_machine(machine: str, elf_class: str) -> str:
    lowered = machine.strip().lower()
    if "x86-64" in lowered or "amd x86-64" in lowered:
        return "x86_64"
    if "aarch64" in lowered:
        return "aarch64"
    if "risc-v" in lowered or "riscv" in lowered:
        return "riscv64" if "64" in elf_class else "riscv32"
    if "80386" in lowered or lowered == "intel 80386":
        return "i386"
    return machine.strip()


def _host_machine() -> str:
    value = platform.machine().lower()
    aliases = {
        "amd64": "x86_64",
        "arm64": "aarch64",
        "x64": "x86_64",
    }
    return aliases.get(value, value)


def _inspect_elf(
    plugin: BackendPluginManifest,
    relative_path: str,
    path: Path,
    sha256: str,
) -> NativeArtifact:
    readelf = shutil.which("readelf")
    nm = shutil.which("nm")
    if readelf is None or nm is None:
        raise BackendPluginCompatibilityError(
            "native binary inspection tools",
            "readelf and nm available",
            "readelf={}, nm={}".format(
                readelf or "<missing>",
                nm or "<missing>",
            ),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Install binutils in the validation environment to collect "
                "offline static evidence. This does not authorize loading."
            ),
        )

    header = _run_tool(readelf, ("-hW",), path, plugin)
    dynamic = _run_tool(readelf, ("-dW",), path, plugin)
    symbols = _run_tool(
        nm,
        ("-D", "--defined-only", "--extern-only", "--format=posix"),
        path,
        plugin,
    )

    class_match = re.search(r"^\s*Class:\s*(\S+)", header, flags=re.MULTILINE)
    machine_match = re.search(
        r"^\s*Machine:\s*(.+?)\s*$", header, flags=re.MULTILINE
    )
    type_match = re.search(r"^\s*Type:\s*(\S+)", header, flags=re.MULTILINE)
    if class_match is None or machine_match is None or type_match is None:
        raise BackendPluginCompatibilityError(
            "native ELF metadata",
            "readable ELF class, machine, and shared-object type",
            "<incomplete ELF header>",
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation="Rebuild the plugin as a valid ELF shared object.",
        )
    if type_match.group(1) != "DYN":
        raise BackendPluginCompatibilityError(
            "native binary type",
            "ELF shared object (DYN)",
            type_match.group(1),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation="Package a shared library, not an executable/object file.",
        )

    architecture = _normalize_machine(
        machine_match.group(1), class_match.group(1)
    )
    host = _host_machine()
    if architecture.lower() != host:
        raise BackendPluginCompatibilityError(
            "native architecture",
            host,
            architecture,
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Collect offline evidence from an artifact targeting the "
                "current machine architecture. This does not authorize "
                "loading."
            ),
        )

    sonames = tuple(sorted(set(_SONAME_PATTERN.findall(dynamic))))
    if len(sonames) > 1:
        raise BackendPluginCompatibilityError(
            "native SONAME",
            "zero or one SONAME per native library",
            ", ".join(sonames),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation="Rebuild the shared object with one deterministic SONAME.",
        )
    needed = tuple(sorted(set(_NEEDED_PATTERN.findall(dynamic))))
    forbidden_needed = tuple(
        item for item in needed if _FORBIDDEN_TOOLCHAIN_DEPENDENCY.match(item)
    )
    if forbidden_needed:
        raise BackendPluginCompatibilityError(
            "native toolchain dependencies",
            "no private LLVM/MLIR shared-library dependency",
            ", ".join(forbidden_needed),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Remove private LLVM/MLIR dependencies from the evidence "
                "artifact; Protocol/Manifest 1.0 does not authorize loading "
                "it."
            ),
        )

    exported = set()
    for line in symbols.splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith(":"):
            continue
        name = stripped.split(None, 1)[0]
        if name:
            # Treat symbol versions conservatively as one base claim for
            # cross-plugin collision detection.
            exported.add(name.split("@", 1)[0])

    return NativeArtifact(
        path=relative_path,
        sha256=sha256,
        binary_format="ELF",
        architecture=architecture,
        identity=sonames[0] if sonames else None,
        needed_libraries=needed,
        exported_symbols=tuple(sorted(exported)),
    )


def inspect_native_artifacts(
    plugin: BackendPluginManifest,
    distribution: Any,
) -> NativeInspectionReport:
    """Collect import-free static evidence about owning-wheel native files.

    ``python_only`` wheels are required to contain no native binaries.  Other
    isolation-mode values are accepted here only so offline conformance and
    conflict tooling can inspect manually constructed records; a successful
    report is never operational authorization under Protocol/Manifest 1.0.
    """
    file_map = _distribution_file_map(plugin, distribution)
    discovered_native = _discover_native_paths(plugin, distribution, file_map)
    declared = tuple(sorted(set(plugin.native_libraries)))

    if plugin.isolation_mode is PluginIsolationMode.PYTHON_ONLY:
        if discovered_native:
            raise BackendPluginCompatibilityError(
                "python_only wheel contents",
                "no native files by filename or ELF/Mach-O/PE signature",
                ", ".join(discovered_native),
                plugin_id=plugin.plugin_id,
                entry_point=plugin.entry_point,
                remediation=(
                    "Remove native artifacts from the owning distribution and "
                    "publish a genuinely Python-only backend wheel."
                ),
            )
        return NativeInspectionReport()

    missing = tuple(path for path in declared if path not in file_map)
    if missing:
        raise BackendPluginManifestError(
            "Manifest native_libraries are missing from the installed "
            "distribution: " + ", ".join(missing),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            detail=", ".join(missing),
            field="native_libraries",
            expected="exact paths present in the wheel RECORD",
            actual=", ".join(missing),
            remediation=(
                "For offline evidence, make each declared path match the "
                "evidence wheel or remove the stale record. This does not "
                "authorize loading."
            ),
        )

    undeclared = tuple(
        path for path in discovered_native if path not in set(declared)
    )
    if undeclared:
        raise BackendPluginManifestError(
            "Backend wheel contains undeclared native libraries: "
            + ", ".join(undeclared),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            detail=", ".join(undeclared),
            field="native_libraries",
            expected="every native wheel file declared exactly once",
            actual=", ".join(undeclared),
            remediation=(
                "Remove native artifacts from a Protocol/Manifest 1.0 backend "
                "wheel. Offline evidence records may describe every artifact, "
                "but do not authorize loading."
            ),
        )

    if plugin.isolation_mode is PluginIsolationMode.SUBPROCESS:
        # Preserve inventory evidence for manually constructed records without
        # implying a subprocess protocol or load contract. Protocol/Manifest
        # 1.0 rejects this mode at every operational boundary.
        return NativeInspectionReport()

    _validate_native_wheel_layout(plugin, distribution)

    if platform.system() != "Linux":
        raise BackendPluginCompatibilityError(
            "native binary format support",
            "Linux ELF (the currently validated native platform)",
            platform.system(),
            plugin_id=plugin.plugin_id,
            entry_point=plugin.entry_point,
            remediation=(
                "Collect evidence on a supported inspection platform, or "
                "publish a genuinely Python-only backend wheel. Static "
                "inspection never authorizes in-process loading."
            ),
        )

    record_hashes = _record_hashes(distribution)
    artifacts = []
    for relative_path in declared:
        installed, sha256 = _locate_verified_file(
            plugin,
            distribution,
            relative_path,
            file_map[relative_path],
            record_hashes,
        )
        try:
            magic = installed.read_bytes()[:4]
        except OSError as exc:
            raise BackendPluginManifestError(
                "Cannot read native binary magic for '{}'".format(
                    relative_path
                ),
                plugin_id=plugin.plugin_id,
                entry_point=plugin.entry_point,
                field="native_libraries",
                expected="a readable native binary",
                actual="<error: {}>".format(exc),
                remediation="Reinstall the backend wheel.",
            ) from exc
        if magic != _ELF_MAGIC:
            raise BackendPluginCompatibilityError(
                "native binary format",
                "ELF shared object",
                "magic={}".format(magic.hex() or "<empty>"),
                plugin_id=plugin.plugin_id,
                entry_point=plugin.entry_point,
                remediation=(
                    "Package a Linux ELF shared object for this wheel, or use "
                    "a platform whose binary inspector is implemented."
                ),
            )
        artifacts.append(
            _inspect_elf(
                plugin,
                relative_path,
                installed,
                sha256,
            )
        )
    return NativeInspectionReport(tuple(artifacts))


__all__ = [
    "NativeArtifact",
    "NativeInspectionReport",
    "inspect_native_artifacts",
]
