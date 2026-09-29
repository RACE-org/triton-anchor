"""Read-only runtime inputs from the deployment checkout, held by control.lock."""

from __future__ import annotations

from pathlib import Path

from .artifacts import EnvironmentError, SHA_RE

CONTROL_TARGET = "/opt/local-ci/control"


def bind_control(control_root, revision, run):
    root = Path(control_root)
    if not isinstance(revision, str) or not SHA_RE.fullmatch(revision):
        raise EnvironmentError("Control mount requires an exact committed revision")
    if not root.is_absolute() or root.resolve() != root or any(c in str(root) for c in ",\n\r"):
        raise EnvironmentError("Unsafe control checkout directory")
    # Mount only runtime inputs; .git, state and deployment credentials stay outside.
    files = run([
        "git", "-C", str(root), "ls-tree", "-r", "--name-only", "-z", revision,
        "--", "scripts", "api_contract", "envsetup.sh",
    ]).decode().split("\0")
    paths, directories = set(), set()
    for name in filter(None, files):
        path = root / name
        if path.resolve() != path:
            raise EnvironmentError("Control runtime inputs must not contain symlinks")
        paths.add(Path(name).parts[0])
        # The updater runs with umask 0077. Make only tracked runtime inputs
        # readable to the container UID, without changing executable bits.
        mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
        if path.stat().st_mode & 0o777 != mode:
            path.chmod(mode)
        directories.update(parent for parent in path.parents if parent != root and root in parent.parents)
    if "scripts" not in paths:
        raise EnvironmentError("Committed control scripts are missing")
    for directory in directories:
        if directory.stat().st_mode & 0o777 != 0o755:
            directory.chmod(0o755)
    return dict(source=str(root), revision=revision, paths=sorted(paths))


def mounts(control):
    # Old stopped runs can still use their exported directory during cleanup.
    return [
        {"source": str(Path(control["source"]) / path),
         "target": str(Path(CONTROL_TARGET) / path)}
        for path in control.get("paths", [""])
    ]


def mount_arguments(control):
    return [
        argument
        for mount in mounts(control)
        for argument in (
            "--mount",
            "type=bind,source=" + mount["source"] + ",target=" + mount["target"]
            + ",readonly,bind-recursive=disabled",
        )
    ]


def verify_mount(info, control):
    for expected in mounts(control):
        actual = [m for m in info.get("Mounts", []) if m.get("Destination") == expected["target"]]
        if (
            len(actual) != 1
            or actual[0].get("Type") != "bind"
            or actual[0].get("Source") != expected["source"]
            or actual[0].get("RW") is not False
        ):
            raise EnvironmentError("Control mount identity or readonly mode changed")
