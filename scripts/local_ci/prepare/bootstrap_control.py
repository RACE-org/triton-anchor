#!/usr/bin/env python3
"""Bootstrap an empty Local CI server from one approved control commit.

This file is intentionally standalone so a trusted provisioning channel can
place it on a new server without first installing the control repository.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator


BOOTSTRAP_SCHEMA = "triton-anchor-local-ci-control-bootstrap"
SHA_RE = re.compile(r"[0-9a-f]{40}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_control_source(value: object, *, allow_local: bool = False) -> str:
    if not isinstance(value, str) or value != value.strip() or not value:
        raise ValueError("control_repo_url must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("control_repo_url contains control characters")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme == "https":
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("control_repo_url has an invalid port") from exc
        if (
            parsed.hostname != "gitee.com"
            or parsed.username
            or parsed.password
            or port is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "control_repo_url must be a credential-free Gitee HTTPS URL"
            )
    elif allow_local and (
        (not parsed.scheme and Path(value).is_absolute()) or parsed.scheme == "file"
    ):
        pass
    else:
        raise ValueError("Local control repositories are allowed only in tests")
    return value


def validate_control_branch(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("control_branch is not a safe Git branch name")
    invalid = (
        value == "@"
        or value.startswith(("-", ".", "/"))
        or value.endswith((".", "/"))
        or ".." in value
        or "//" in value
        or "@{" in value
        or any(part.endswith(".lock") for part in value.split("/"))
        or re.search(r"[\x00-\x20\x7f~^:?*\\\[]", value)
    )
    if invalid:
        raise ValueError("control_branch is not a safe Git branch name")
    return value


def exact_revision(value: object) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise ValueError("expected_revision must be an exact lowercase 40-character SHA")
    return value


def absolute_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute path")
    return Path(value)


def load_environment(path: Path) -> None:
    """Read a private KEY=value file without executing shell syntax."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("Credentials environment must use KEY=value lines")
        words = shlex.split(value, comments=False)
        if len(words) > 1:
            raise ValueError("Quote environment values containing whitespace")
        os.environ[key] = words[0] if words else ""


@contextlib.contextmanager
def git_environment() -> Iterator[dict[str, str]]:
    environment = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    temporary = None
    try:
        if os.getenv("GITEE_TOKEN"):
            temporary = tempfile.TemporaryDirectory(prefix="local-ci-bootstrap-")
            askpass = Path(temporary.name) / "askpass.py"
            askpass.write_text(
                "#!/usr/bin/env python3\n"
                "import os,sys\n"
                "prompt = sys.argv[1] if len(sys.argv) > 1 else ''\n"
                "print(os.environ.get('GITEE_USERNAME','oauth2') "
                "if 'Username' in prompt else os.environ['GITEE_TOKEN'])\n"
            )
            askpass.chmod(0o700)
            environment["GIT_ASKPASS"] = str(askpass)
            environment["GIT_ASKPASS_REQUIRE"] = "force"
        yield environment
    finally:
        if temporary is not None:
            temporary.cleanup()


def git(
    root: Path,
    arguments: list[str],
    environment: dict[str, str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={root.resolve()}",
            "-c",
            "credential.helper=",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            *arguments,
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
    )
    if check and result.returncode:
        operation = arguments[0] if arguments else "command"
        raise RuntimeError(
            f"Control git {operation} failed with exit {result.returncode}"
        )
    return result


def remote_revision(
    root: Path, source: str, branch: str, environment: dict[str, str]
) -> str:
    resolved = git(
        root,
        ["ls-remote", "--exit-code", "--refs", source, f"refs/heads/{branch}"],
        environment,
    ).stdout.split()
    if len(resolved) != 2 or not SHA_RE.fullmatch(resolved[0]):
        raise ValueError("Configured control branch did not resolve to one commit")
    return resolved[0]


def verify_checkout(
    root: Path, source: str, revision: str, environment: dict[str, str]
) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Existing control_root is not a dedicated directory")
    git_dir = root / ".git"
    if not git_dir.is_dir() or git_dir.is_symlink():
        raise ValueError("Existing control_root is not a dedicated Git checkout")
    if git(root, ["rev-parse", "--is-inside-work-tree"], environment).stdout.strip() != "true":
        raise ValueError("Existing control_root is not a Git checkout")
    current = git(root, ["rev-parse", "HEAD^{commit}"], environment).stdout.strip()
    if current != revision:
        raise ValueError("Existing control_root is not at expected_revision")
    dirty = git(
        root, ["status", "--porcelain=v1", "--untracked-files=all"], environment
    ).stdout.strip()
    if dirty:
        raise ValueError("Existing control_root has local changes")
    configured_source = git(
        root, ["config", "--get", "remote.origin.url"], environment
    ).stdout.strip()
    if configured_source != source:
        raise ValueError("Existing control_root has a different origin")
    install_script = root / "scripts/local_ci/prepare/install.py"
    if not install_script.is_file() or install_script.is_symlink():
        raise ValueError("Pinned control commit has no regular install.py")


def run_installer(
    root: Path, config_path: Path, credentials_path: Path, python_bin: str
) -> None:
    if not Path(python_bin).is_absolute():
        raise ValueError("python_bin must be an absolute path")
    subprocess.run(
        [
            python_bin,
            str(root / "scripts/local_ci/prepare/install.py"),
            "--config",
            str(config_path),
            "--credentials-env",
            str(credentials_path),
            "--apply",
        ],
        cwd=root,
        check=True,
        timeout=7200,
    )


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.chmod(0o600)
    os.replace(temporary, path)


def bootstrap_control(
    config: dict,
    config_path: Path,
    credentials_path: Path,
    expected_revision: str,
    *,
    apply: bool = False,
    allow_local: bool = False,
    installer: Callable[[Path, Path, Path, str], None] = run_installer,
) -> dict:
    """Clone exactly one approved revision and invoke its formal installer."""
    source = validate_control_source(
        config.get("control_repo_url"), allow_local=allow_local
    )
    branch = validate_control_branch(
        config.get("control_branch", "CI_dev")
    )
    revision = exact_revision(expected_revision)
    root = absolute_path(config.get("control_root"), "control_root")
    state_dir = absolute_path(config.get("state_dir"), "state_dir")
    python_bin = config.get("python_bin", "/usr/bin/python3")
    if not isinstance(python_bin, str):
        raise ValueError("python_bin must be a string")
    probe_root = config_path.resolve().parent
    if not probe_root.is_dir():
        raise ValueError("Config parent directory does not exist")

    with git_environment() as environment:
        observed = remote_revision(probe_root, source, branch, environment)
        if observed != revision:
            raise ValueError("Control branch tip does not match expected_revision")
        result = {
            "schema": BOOTSTRAP_SCHEMA,
            "checked_at": utc_now(),
            "branch": branch,
            "revision": revision,
            "control_root": str(root),
            "action": "verify-existing" if root.exists() else "clone",
            "applied": apply,
        }
        if not apply:
            return result

        root.parent.mkdir(parents=True, exist_ok=True)
        if root.exists() or root.is_symlink():
            verify_checkout(root, source, revision, environment)
            result["cloned"] = False
        else:
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{root.name}.bootstrap-", dir=root.parent)
            )
            try:
                git(temporary, ["init", "--quiet"], environment)
                git(temporary, ["remote", "add", "origin", source], environment)
                git(
                    temporary,
                    [
                        "fetch",
                        "--quiet",
                        "--no-tags",
                        "--depth=1",
                        "origin",
                        f"refs/heads/{branch}:refs/local-ci/bootstrap",
                    ],
                    environment,
                )
                fetched = git(
                    temporary,
                    ["rev-parse", "refs/local-ci/bootstrap^{commit}"],
                    environment,
                ).stdout.strip()
                if fetched != revision:
                    raise RuntimeError(
                        "Control branch moved during bootstrap; approve the new SHA and retry"
                    )
                git(temporary, ["checkout", "--quiet", "--detach", revision], environment)
                verify_checkout(temporary, source, revision, environment)
                if root.exists() or root.is_symlink():
                    raise ValueError("control_root appeared during bootstrap")
                os.replace(temporary, root)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            result["cloned"] = True

    installer(root, config_path.resolve(), credentials_path.resolve(), python_bin)
    result["installed_at"] = utc_now()
    atomic_json(state_dir / "control-bootstrap.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--credentials-env", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.apply and os.geteuid() == 0:
            raise ValueError("Bootstrap must run as the ordinary CI user, without sudo")
        config_path = Path(args.config).resolve()
        credentials_path = Path(args.credentials_env).resolve()
        if (
            not credentials_path.is_file()
            or credentials_path.is_symlink()
            or credentials_path.stat().st_mode & 0o077
            or credentials_path.stat().st_uid != os.getuid()
        ):
            raise ValueError(
                "Credentials EnvironmentFile must exist and be private (mode 600)"
            )
        load_environment(credentials_path)
        config = json.loads(config_path.read_text())
        result = bootstrap_control(
            config,
            config_path,
            credentials_path,
            args.expected_revision,
            apply=args.apply,
        )
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Local CI control bootstrap failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
