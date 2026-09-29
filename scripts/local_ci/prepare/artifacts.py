#!/usr/bin/env python3
"""Trusted source checksums, safe archive extraction and atomic state writes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SHA_RE = re.compile(r"[0-9a-f]{40}")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}")


class EnvironmentError(RuntimeError):
    """A requested environment could not be prepared or safely selected."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path) -> str:
    """Fingerprint an immutable dependency tree, including internal link targets."""
    entries = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            if not path.resolve().is_relative_to(root.resolve()):
                raise EnvironmentError(
                    "Prepared dependency contains a link outside its install tree"
                )
            entries.append([relative, "link", os.readlink(path)])
        elif path.is_file():
            entries.append(
                [relative, "file", file_digest(path), path.stat().st_mode & 0o777]
            )
        elif path.is_dir():
            entries.append([relative, "directory"])
        else:
            raise EnvironmentError("Prepared dependency contains a special file")
    return fingerprint(entries)


def safe_source(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EnvironmentError(f"{label} is required")
    parsed = urllib.parse.urlparse(value)
    if parsed.username or parsed.password:
        raise EnvironmentError(f"{label} must not contain embedded credentials")
    if parsed.scheme:
        if parsed.scheme not in {"https", "file"}:
            raise EnvironmentError(f"{label} must use HTTPS or a local file")
        if (parsed.hostname or "").lower() in {
            "github.com",
            "api.github.com",
            "raw.githubusercontent.com",
        }:
            raise EnvironmentError(
                f"{label} must use a server-reachable mirror; GitHub is not a Local CI source"
            )
    elif not Path(value).is_absolute():
        raise EnvironmentError(
            f"{label} must be an absolute local path or HTTPS mirror"
        )
    return value


def extract_verified_archive(
    archive: Path, destination: Path, expected_sha256: str, strip_components: int = 1
) -> None:
    """Extract files plus internal links, never following a link while writing."""
    if (
        not DIGEST_RE.fullmatch(expected_sha256)
        or file_digest(archive) != expected_sha256
    ):
        raise EnvironmentError(
            "Dependency archive SHA256 does not match trusted recipe"
        )
    if strip_components < 0:
        raise EnvironmentError("strip_components must not be negative")
    destination.mkdir(parents=True, exist_ok=False)
    links: list[tuple[Path, str, bool]] = []

    def target(name: str) -> Path | None:
        parts = name.replace("\\", "/").split("/")
        if (
            name.startswith(("/", "\\"))
            or any(part == ".." for part in parts)
            or ":" in parts[0]
        ):
            raise EnvironmentError("Dependency archive contains an unsafe path")
        parts = [part for part in parts if part not in {"", "."}][strip_components:]
        return destination.joinpath(*parts) if parts else None

    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as handle:
                for member in handle.infolist():
                    mode = member.external_attr >> 16
                    output = target(member.filename)
                    if output is None:
                        continue
                    if (mode & 0o170000) == 0o120000:
                        links.append(
                            (output, handle.read(member).decode("utf-8"), False)
                        )
                        continue
                    if member.is_dir():
                        output.mkdir(parents=True, exist_ok=True)
                    else:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        with handle.open(member) as source, output.open("wb") as sink:
                            shutil.copyfileobj(source, sink)
                        if mode:
                            output.chmod(mode & 0o777)
        else:
            with tarfile.open(archive) as handle:
                for member in handle:
                    if not any(
                        (
                            member.isfile(),
                            member.isdir(),
                            member.issym(),
                            member.islnk(),
                        )
                    ):
                        raise EnvironmentError(
                            "Dependency archives must not contain special device files"
                        )
                    output = target(member.name)
                    if output is None:
                        continue
                    if member.issym() or member.islnk():
                        linked = target(member.linkname) if member.islnk() else None
                        if member.islnk() and linked is None:
                            raise EnvironmentError("Invalid archive hard-link target")
                        links.append(
                            (
                                output,
                                str(linked) if linked else member.linkname,
                                member.islnk(),
                            )
                        )
                        continue
                    if member.isdir():
                        output.mkdir(parents=True, exist_ok=True)
                    else:
                        output.parent.mkdir(parents=True, exist_ok=True)
                        source = handle.extractfile(member)
                        if source is None:
                            raise EnvironmentError(
                                "Unable to read dependency archive member"
                            )
                        with source, output.open("wb") as sink:
                            shutil.copyfileobj(source, sink)
                        output.chmod(member.mode & 0o777)
        # Deferring links prevents an earlier archive entry redirecting file
        # extraction. Real LLVM releases contain internal shared-library links.
        for output, link, hard in links:
            if output.exists() or output.is_symlink():
                raise EnvironmentError("Archive link collides with an extracted file")
            if not hard and (link.startswith(("/", "\\")) or ":" in link.split("/")[0]):
                raise EnvironmentError("Archive symlink has an absolute target")
            linked = Path(link) if hard else output.parent / link
            if not linked.resolve().is_relative_to(destination.resolve()):
                raise EnvironmentError("Archive link escapes the dependency directory")
            output.parent.mkdir(parents=True, exist_ok=True)
            if hard:
                if not linked.is_file():
                    raise EnvironmentError(
                        "Archive hard-link target is not a regular file"
                    )
                os.link(linked, output)
            else:
                output.symlink_to(link)
    except BaseException:
        shutil.rmtree(destination)
        raise
