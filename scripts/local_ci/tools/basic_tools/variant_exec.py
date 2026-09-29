#!/usr/bin/env python3
"""Run a native command with one variant's complete task environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


# Transport settings may come from the executor. Build/runtime settings must
# come from the chosen variant, never from the caller's candidate environment.
TRANSPORT_ENV = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    "LOCAL_CI_EXECUTION_ID",
)


def command_environment(context, overrides=None):
    runtime = context.get("runtime_env")
    if runtime is None:
        # Standalone tool contexts predating variant runtimes retain their
        # ordinary shell environment. Worker contexts always set runtime_env.
        env = dict(os.environ)
    else:
        if not isinstance(runtime, dict):
            raise ValueError("runtime_env must be a complete variant environment")
        env = {key: os.environ[key] for key in TRANSPORT_ENV if key in os.environ}
        env.update({str(key): str(value) for key, value in runtime.items()})
    env.update({str(key): str(value) for key, value in (overrides or {}).items()})
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True)
    parser.add_argument("--backend", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    context = json.loads(Path(args.context).read_text(encoding="utf-8-sig"))
    if not isinstance(context.get("runtime_env"), dict):
        parser.error("native commands require a Worker variant context with runtime_env")
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide a command after --")
    profile = context.get("profile", {})
    if args.backend and not profile.get("backend_enabled"):
        parser.error("this variant has no configured backend capability")
    # The sibling module is also the ordinary tool CLI entry point.
    from runner import environment_command

    command = environment_command(
        command, profile.get("tools", {}), context["tools_dir"], backend=args.backend,
    )
    os.chdir(context["source_dir"])
    os.execvpe(command[0], command, command_environment(context, profile.get("tools", {}).get("env")))


if __name__ == "__main__":
    main()
