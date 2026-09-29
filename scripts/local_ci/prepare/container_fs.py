#!/usr/bin/env python3
"""Fixed management operations inside one rootless PR container.

Never accepts host paths, arbitrary UIDs or arbitrary management commands.
Session credentials arrive on stdin and remain outside exported artifacts.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile

if __package__:
    from .python_environment import ci_python
else:
    # -I/-S removes the script directory from sys.path. Load only our sibling
    # from the read-only control mount, never a module from the task checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from python_environment import ci_python

TASK = Path("/task")
SESSION = TASK / "session"
CONTROL = TASK / ".control"
MAX_ARCHIVE = 4 * 1024**3


def checked(root: Path, relative: str, *, exists=False):
    path = Path(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"..", "."} for part in path.parts)
    ):
        raise ValueError("Only relative paths within a fixed task scope are allowed")
    target = root / path
    for part in (
        root,
        *[root.joinpath(*path.parts[:i]) for i in range(1, len(path.parts) + 1)],
    ):
        if part.is_symlink():
            raise ValueError("Symlinked management paths are forbidden")
    if exists and not target.exists():
        raise ValueError("Task object is missing")
    return target


def manifest():
    path = CONTROL / "manifest.json"
    if path.is_symlink() or not path.is_file() or path.stat().st_uid != 0:
        raise ValueError("Trusted task manifest is unavailable")
    return json.loads(path.read_text())


def own(path: Path, uid: int, gid: int, mode: int):
    if path.is_symlink():
        raise ValueError("Refusing ownership changes through a symlink")
    os.chown(path, uid, gid)
    path.chmod(mode)


def write(path: Path, payload: bytes, *, uid=0, gid=0, mode=0o600):
    if path.is_symlink():
        raise ValueError("Refusing symlinked output")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".management-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        own(Path(temp), uid, gid, mode)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def walk(root):
    for parent, directories, files in os.walk(root, followlinks=False):
        for name in [*directories, *files]:
            path = Path(parent) / name
            if path.is_symlink():
                continue
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode):
                raise ValueError("Special files are not accepted")
            if path.is_dir() and os.path.ismount(path):
                raise ValueError("Nested mounts are forbidden")
            yield path


def set_tree_identity(root, uid, gid):
    for path in [root, *walk(root)]:
        mode = path.stat().st_mode
        own(
            path, uid, gid, 0o750 if path.is_dir() else 0o750 if mode & 0o111 else 0o640
        )


def init_task(payload):
    if (CONTROL / "manifest.json").exists():
        if manifest() != payload:
            raise ValueError("Task identity cannot change within a run")
        return {"initialized": True}
    uids, gids = payload["uids"], payload["gids"]
    if (
        set(uids) != {"task"}
        or set(gids) != {"task"}
        or any(type(v) is not int or v <= 0 for v in [*uids.values(), *gids.values()])
    ):
        raise ValueError("One non-root task UID and GID are required")
    TASK.mkdir(parents=True, exist_ok=True)
    CONTROL.mkdir(exist_ok=True)
    own(CONTROL, 0, 0, 0o700)
    for name in (
        "candidate",
        "base",
        "artifacts",
        "experiments",
        "session",
        "session/home",
    ):
        root = TASK / name
        root.mkdir(parents=True, exist_ok=True)
        if name == "artifacts":
            own(root, 0, 0, 0o777)
        else:
            own(
                root,
                uids["task"],
                gids["task"],
                0o755 if name != "session/home" else 0o700,
            )
    write(CONTROL / "manifest.json", json.dumps(payload, sort_keys=True).encode())
    return {"initialized": True}


def extract(source, destination, expected_digest):
    digest = hashlib.sha256()
    with tempfile.TemporaryFile() as temporary:
        total = 0
        while block := source.read(1024 * 1024):
            total += len(block)
            if total > MAX_ARCHIVE:
                raise ValueError("Source archive exceeds the task import limit")
            temporary.write(block)
            digest.update(block)
        if digest.hexdigest() != expected_digest:
            raise ValueError("Source archive checksum mismatch")
        temporary.seek(0)
        with tarfile.open(fileobj=temporary, mode="r:*") as archive:
            members = [
                m
                for m in archive.getmembers()
                if not (m.isdir() and m.name in {".", "./"})
            ]
            total = 0
            for member in members:
                checked(destination, member.name)
                if not member.isdir() and not member.isfile() and not member.issym():
                    raise ValueError(
                        "Source archives allow only directories, files and internal symlinks"
                    )
                total += member.size
                if total > MAX_ARCHIVE:
                    raise ValueError("Expanded source archive is too large")
                if member.issym():
                    target = Path(member.name).parent / member.linkname
                    if Path(member.linkname).is_absolute() or not (
                        destination / target
                    ).resolve().is_relative_to(destination.resolve()):
                        raise ValueError("Source archive symlink escapes its checkout")
            destination.mkdir()
            # Regular entries first, links last; no extraction operation follows a link.
            for member in [m for m in members if not m.issym()] + [
                m for m in members if m.issym()
            ]:
                target = checked(destination, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.isdir():
                    target.mkdir(exist_ok=True)
                elif member.issym():
                    target.symlink_to(member.linkname)
                else:
                    if target.exists():
                        raise ValueError("Duplicate archive file")
                    with (
                        archive.extractfile(member) as content,
                        target.open("xb") as output,
                    ):
                        shutil.copyfileobj(content, output)
                    target.chmod(0o755 if member.mode & 0o111 else 0o644)


def import_checkout(params, stream):
    data = manifest()
    variant = params["variant"]
    if variant not in {"candidate", "base"}:
        raise ValueError("Invalid checkout variant")
    marker = CONTROL / (variant + "-import.json")
    destination = checked(TASK, variant + "/checkout")
    identity = {key: params[key] for key in ("sha256", "expected_sha")}
    if marker.exists():
        if json.loads(marker.read_text()) != identity or not destination.is_dir():
            raise ValueError("Existing checkout import identity changed")
        return {"imported": False, "reused": True}
    if destination.exists():
        raise ValueError("Incomplete checkout import requires a new attempt")
    extract(stream, destination, params["sha256"])
    if (
        git(destination, "rev-parse", "HEAD") != params["expected_sha"]
        or git(destination, "status", "--porcelain", "--untracked-files=no")
    ):
        raise ValueError("Imported checkout differs from frozen commit")
    set_tree_identity(destination, data["uids"]["task"], data["gids"]["task"])
    runtime = data["variants"][variant]
    backend = runtime.get("env", {}).get("BACKEND_PATH")
    if runtime.get("backend_enabled") and backend:
        target = checked(TASK, variant + "/backend")
        shutil.copytree(backend, target, symlinks=True)
        set_tree_identity(target, data["uids"]["task"], data["gids"]["task"])
    write(marker, json.dumps(identity).encode())
    return {"imported": True, "path": str(destination)}


def git(root, *arguments):
    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "safe.directory=" + str(root),
        *arguments,
    ]
    result = subprocess.run(
        command,
        cwd=root,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(CONTROL),
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
        },
        capture_output=True,
        timeout=60,
    )
    if result.returncode:
        raise ValueError("Frozen checkout Git verification failed")
    return result.stdout.decode().strip()


def seed_venv(root, environment):
    """Copy packages only from the immutable image interpreter, never PR Python."""
    seed = ci_python(environment=environment)
    # Ask the isolated trusted seed for its actual package layout, including
    # Debian dist-packages. The immutable seed is used only for environment preparation.
    query = "import json,sys; assert sys.prefix != sys.base_prefix, 'Prepared CI virtual environment required'; print(json.dumps({'version':str(sys.version_info.major)+'.'+str(sys.version_info.minor),'paths':[p for p in sys.path if p.endswith(('site-packages','dist-packages'))]}))"
    layout = json.loads(
        subprocess.run(
            [seed, "-I", "-c", query], check=True, timeout=30, capture_output=True
        ).stdout
    )
    if not re.fullmatch(r"[0-9]+\.[0-9]+", layout["version"]):
        raise ValueError("Invalid trusted seed Python layout")
    subprocess.run(
        # pip and other packages come from the prepared CI environment below;
        # do not bootstrap a different pip through the system ensurepip module.
        [seed, "-I", "-m", "venv", "--copies", "--without-pip", str(root / "venv")],
        check=True,
        timeout=120,
    )
    target = root / "venv/lib" / ("python" + layout["version"]) / "site-packages"
    # Earlier entries in sys.path take precedence, so copy them last.
    for value in reversed(layout["paths"]):
        source = Path(value)
        if not source.is_absolute() or not source.is_dir():
            continue
        shutil.copytree(source, target, dirs_exist_ok=True, symlinks=False)
    # Preserve Python console commands (pip, pytest, etc.) with task-local
    # shebangs, so PATH cannot fall through to a seed/system installation.
    for source in Path(seed).parent.iterdir():
        if not source.is_file():
            continue
        with source.open("rb") as stream:
            first_line = stream.readline(4096)
            if not re.fullmatch(rb"#![^\r\n]*[/ ]python[0-9.]*\r?\n", first_line):
                continue
            body = stream.read()
        command = root / "venv/bin" / source.name
        command.write_bytes(("#!" + str(root / "venv/bin/python") + "\n").encode() + body)
        command.chmod(source.stat().st_mode & 0o777)
    # Bind each task venv to its own profile's read-only source, never the image.
    flaggems = environment.get("FLAGGEMS_CLONE_DIR")
    (target / "local_ci_flaggems.pth").unlink(missing_ok=True)
    if flaggems:
        source = Path(flaggems)
        if source.parent != Path("/opt/local-ci/runtime/deps") or not (source / "src").is_dir():
            raise ValueError("FlagGems requires a mounted dependency with a src directory")
        (target / "local_ci_flaggems.pth").write_text(str(source / "src") + "\n")


def native_layout(root):
    return {
        "root": str(root),
        "checkout": str(root / "checkout"),
        "venv": str(root / "venv"),
        "python_bin": str(root / "venv/bin/python"),
        "home": str(root / "home"),
        "tmp": str(root / "tmp"),
        "cache": str(root / "cache"),
        "backend": str(root / "backend") if (root / "backend").is_dir() else None,
    }


def prepare_workspace(params):
    """Prepare one data version once; reject partial or mismatched environments."""
    data = manifest()
    variant = params.get("variant", "candidate")
    if variant not in {"candidate", "base"}:
        raise ValueError("Unknown task data version")
    root = checked(TASK, variant, exists=True)
    fingerprint = params["environment_fingerprint"]
    runtime = data["variants"][variant]
    if fingerprint != runtime["environment_fingerprint"]:
        raise ValueError("Workspace differs from its frozen variant environment")
    venv = root / "venv"
    marker = venv / ".local-ci-environment.json"
    expected = {"environment_fingerprint": fingerprint}
    reused = venv.exists()
    if reused:
        if (
            not (venv / "bin/python").is_file()
            or not marker.is_file()
            or marker.is_symlink()
            or json.loads(marker.read_text()) != expected
        ):
            raise ValueError("Partial or different task venv requires a new run")
    else:
        seed_venv(root, runtime["env"])
        write(marker, json.dumps(expected, sort_keys=True).encode(), mode=0o644)
        set_tree_identity(venv, data["uids"]["task"], data["gids"]["task"])
    for name in ("home", "tmp", "cache", "state"):
        (root / name).mkdir(exist_ok=True)
        own(root / name, data["uids"]["task"], data["gids"]["task"], 0o755)
    flaggems = runtime["env"].get("FLAGGEMS_CLONE_DIR")
    if flaggems:
        write(
            root / "home/.gitconfig",
            ("[safe]\n\tdirectory = " + flaggems + "\n").encode(),
            uid=data["uids"]["task"], gid=data["gids"]["task"], mode=0o644,
        )
    return {
        **native_layout(root),
        "reused": reused,
        "environment_fingerprint": fingerprint,
    }


def deploy_session(payload):
    data = manifest()
    files = payload["files"]
    if set(files) - {"config.toml", "auth.json", "AI_CI_PROGRAM.md"}:
        raise ValueError("Only trusted Codex session files may be deployed")
    for name, text in files.items():
        if not isinstance(text, str) or len(text.encode()) > 4 * 1024**2:
            raise ValueError("Invalid session file")
        target = (
            TASK / "candidate/checkout"
            if name == "AI_CI_PROGRAM.md"
            else SESSION / "home"
        ) / name
        write(
            target,
            text.encode(),
            uid=data["uids"]["task"],
            gid=data["gids"]["task"],
            mode=0o600 if name != "AI_CI_PROGRAM.md" else 0o400,
        )
    environment = payload["environment"]
    if not isinstance(environment, dict) or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
        or not isinstance(v, str)
        or "\x00" in v
        for k, v in environment.items()
    ):
        raise ValueError("Invalid private session environment")
    write(
        SESSION / "environment.json",
        json.dumps(environment).encode(),
        uid=data["uids"]["task"],
        gid=data["gids"]["task"],
    )
    return {
        "home": "/task/session/home",
        "workspace": "/task/candidate/checkout",
        "python_bin": environment["PYTHON_BIN"],
    }


def purge_credentials():
    for path in (
        SESSION / "home/config.toml",
        SESSION / "home/auth.json",
        SESSION / "environment.json",
    ):
        if path.is_symlink():
            raise ValueError("Refusing symlinked credentials")
        path.unlink(missing_ok=True)
    return {"purged": True}


def collect_artifacts():
    root = checked(TASK, "artifacts", exists=True)

    def readable(directory):
        os.fchmod(directory, 0o755)
        for name in os.listdir(directory):
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) and not (
                stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            ):
                continue
            descriptor = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
            try:
                current = os.fstat(descriptor)
                if stat.S_ISDIR(current.st_mode):
                    readable(descriptor)
                elif stat.S_ISREG(current.st_mode) and current.st_nlink == 1:
                    os.fchmod(descriptor, 0o644)
            finally:
                os.close(descriptor)

    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        readable(descriptor)
    finally:
        os.close(descriptor)
    return {"artifact_dir": str(root)}


def main():
    operation = sys.argv[1]
    if operation == "launch-codex":
        environment = json.loads((SESSION / "environment.json").read_text())
        executable = environment.pop("LOCAL_CI_CODEX_BIN")
        if not Path(executable).is_absolute():
            raise ValueError("Codex executable must be an absolute trusted image path")
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(38, 1, 0, 0, 0):
            raise OSError("Could not set no_new_privs")
        os.execvpe(executable, [executable, *sys.argv[2:]], environment)
    if os.geteuid() != 0:
        raise ValueError("Management helper requires container namespace UID 0")
    params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    if operation in {"init", "deploy-session"}:
        payload = json.load(sys.stdin)
        result = init_task(payload) if operation == "init" else deploy_session(payload)
    elif operation == "import-checkout":
        result = import_checkout(params, sys.stdin.buffer)
    elif operation == "prepare-workspace":
        result = prepare_workspace(params)
    elif operation == "collect-artifacts":
        result = collect_artifacts()
    elif operation == "purge-credentials":
        result = purge_credentials()
    elif operation == "clean-work":
        preserved = set() if params.get("scratch_only") else {"artifacts"}
        for path in TASK.iterdir():
            if path.name not in preserved:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        result = {"cleaned": True}
    else:
        raise ValueError("Unsupported fixed management operation")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("Trusted container management operation failed", file=sys.stderr)
        raise SystemExit(1)
