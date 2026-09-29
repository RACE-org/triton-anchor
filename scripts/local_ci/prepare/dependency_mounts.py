"""Versioned, CI-owned toolchains mounted read-only outside task storage."""

import os
from pathlib import Path
import stat

from .artifacts import DIGEST_RE, NAME_RE, EnvironmentError, tree_digest

CONTAINER_DEPS = Path("/opt/local-ci/runtime/deps")


def _directory(value):
    if not isinstance(value, str) or not value or any(c in value for c in ",\n\r\x00"):
        raise EnvironmentError(
            "Dependency source must be an absolute canonical directory"
        )
    path = Path(value)
    if (
        not path.is_absolute()
        or path.resolve() != path
        or str(path) != value
        or not path.is_dir()
    ):
        raise EnvironmentError(
            "Dependency source must be an absolute canonical directory"
        )
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o022:
        raise EnvironmentError(
            "Dependency directories must be CI-owned without group/other write access"
        )
    return path


def dependency_mounts(config, profile, *, verify_content=False):
    entries = profile.get("mounts", [])
    if not isinstance(entries, list):
        raise EnvironmentError("Dependency mounts must be a list")
    if not entries:
        return []
    root = _directory(config.get("dependency_root"))
    protected = [Path.home()]
    protected += [
        Path(config[key]).resolve()
        for key in ("control_root", "state_dir", "codex_home")
        if config.get(key)
    ]
    if root == Path("/") or any(
        root == path or root in path.parents for path in protected
    ):
        raise EnvironmentError(
            "Use a dedicated dependency_root outside control, state and credentials"
        )
    if any(path in root.parents for path in protected[1:]):
        raise EnvironmentError(
            "dependency_root cannot be inside control, state or credentials"
        )
    result, sources, targets = [], [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "source",
            "target",
            "read_only",
            "sha256",
        }:
            raise EnvironmentError(
                "Each dependency mount needs source, target, read_only and sha256"
            )
        if (
            entry["read_only"] is not True
            or not isinstance(entry["sha256"], str)
            or not DIGEST_RE.fullmatch(entry["sha256"])
        ):
            raise EnvironmentError(
                "Dependencies require read_only=true and a tree SHA256"
            )
        source = _directory(entry["source"])
        if source == root or not source.is_relative_to(root):
            raise EnvironmentError(
                "Dependency source must be a version directory inside dependency_root"
            )
        if any(
            source == path or source in path.parents or path in source.parents
            for path in sources
        ):
            raise EnvironmentError("Dependency sources must not overlap")
        for parent in source.parents:
            if parent == root:
                break
            _directory(str(parent))
        target = (
            Path(entry["target"]) if isinstance(entry["target"], str) else Path(".")
        )
        if (
            target.parent != CONTAINER_DEPS
            or not NAME_RE.fullmatch(target.name)
            or str(target) != entry["target"]
            or str(target) in targets
        ):
            raise EnvironmentError(
                "Dependency targets must be distinct directories directly under "
                + str(CONTAINER_DEPS)
            )
        if target.name in profile.get("archives", {}):
            raise EnvironmentError("Dependency mount conflicts with an image archive")
        if target.name.startswith("llvm-") and (
            profile.get("llvm", {}).get("mode") != "mount"
            or target.name != "llvm-" + str(profile["llvm"].get("commit"))
        ):
            raise EnvironmentError("LLVM mount must match the mounted LLVM recipe")
        if verify_content:
            # Mapped container UIDs need read/traverse access, but never host write access.
            for path in [source, *source.rglob("*")]:
                info = path.lstat()
                if info.st_uid != os.getuid():
                    raise EnvironmentError(
                        "Every dependency entry must belong to the CI account"
                    )
                if path.is_symlink():
                    if (
                        Path(os.readlink(path)).is_absolute()
                        or not path.resolve().is_relative_to(source)
                        or not path.exists()
                    ):
                        raise EnvironmentError(
                            "Dependency link escapes its version directory or is broken"
                        )
                    continue
                if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
                    raise EnvironmentError(
                        "Dependencies cannot contain sockets or special files"
                    )
                required = 0o005 if path.is_dir() else 0o004
                if info.st_mode & 0o022 or info.st_mode & required != required:
                    raise EnvironmentError(
                        "Dependency entries need mapped-user read/traverse access and no group/other writes"
                    )
            if tree_digest(source) != entry["sha256"]:
                raise EnvironmentError(
                    "Dependency tree SHA256 changed; prepare a new version directory"
                )
        sources.append(source)
        targets.add(str(target))
        result.append(dict(entry))
    return result


def validate_mounted_llvm(profile, mounts):
    llvm = profile.get("llvm", {})
    if llvm.get("mode") != "mount":
        return
    revision = profile.get("requested_llvm_hash", profile.get("llvm_hash"))
    if llvm.get("commit") != revision or not any(
        entry["target"] == str(CONTAINER_DEPS / ("llvm-" + str(revision)))
        for entry in mounts
    ):
        raise EnvironmentError(
            "Mounted LLVM needs a matching commit and dependency mount"
        )


def mount_arguments(mounts):
    args = []
    for entry in mounts:
        args += [
            "--mount",
            "type=bind,source="
            + entry["source"]
            + ",target="
            + entry["target"]
            + ",readonly,bind-recursive=disabled",
        ]
    return args


def verify_mounts(info, mounts):
    actual = {entry["Destination"]: entry for entry in info.get("Mounts", [])}
    for entry in mounts:
        value = actual.get(entry["target"], {})
        if (
            value.get("Type") != "bind"
            or value.get("Source") != entry["source"]
            or value.get("RW") is not False
        ):
            raise EnvironmentError("Read-only dependency mount identity changed")
