#!/usr/bin/env python3
"""Validate changed documentation/control files without importing candidate code."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", *args],
        cwd=root,
    ).decode("utf-8")


def yaml_documents(text: str, *, allow_local_tags: bool = False):
    try:
        import yaml
    except ImportError as exc:
        raise ValueError(
            "Trusted Python seed requires PyYAML for YAML/workflow contract checks"
        ) from exc

    class Loader(yaml.SafeLoader):
        pass

    Loader.yaml_implicit_resolvers = {
        key: [
            (tag, pattern) for tag, pattern in values if tag != "tag:yaml.org,2002:bool"
        ]
        for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError("Duplicate YAML mapping key: " + str(key))
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    if allow_local_tags:
        # MLIR configuration tags describe data, not Python object constructors.
        def local_tag(loader, tag, node):
            if isinstance(node, yaml.MappingNode):
                return mapping(loader, node)
            if isinstance(node, yaml.SequenceNode):
                return loader.construct_sequence(node)
            return loader.construct_scalar(node)

        Loader.add_multi_constructor("!", local_tag)
    return list(yaml.load_all(text, Loader=Loader))


def workflow_contract(document) -> None:
    if not isinstance(document, dict) or not document.get("on"):
        raise ValueError("Workflow requires nonempty on triggers")
    jobs = document.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        raise ValueError("Workflow requires nonempty jobs")
    for name, job in jobs.items():
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name)
            or not isinstance(job, dict)
        ):
            raise ValueError("Invalid workflow job")
        needs = job.get("needs", [])
        needs = [needs] if isinstance(needs, str) else needs
        if not isinstance(needs, list) or any(
            item not in jobs or item == name for item in needs
        ):
            raise ValueError("Workflow job has an invalid needs dependency")
        if job.get("uses"):
            if not isinstance(job["uses"], str) or "steps" in job or "runs-on" in job:
                raise ValueError(
                    "Reusable workflow job has incompatible execution fields"
                )
            continue
        if (
            not job.get("runs-on")
            or not isinstance(job.get("steps"), list)
            or not job["steps"]
        ):
            raise ValueError("Workflow job requires runs-on and nonempty steps")
        for step in job["steps"]:
            if not isinstance(step, dict) or ("run" in step) == ("uses" in step):
                raise ValueError("Workflow step requires exactly one of run or uses")
            value = step.get("run", step.get("uses"))
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Workflow step command/action must be nonempty")


def check(root: Path, base: str, tested: str) -> dict:
    root = root.resolve(strict=True)
    if not all(re.fullmatch("[a-f0-9]{40}", sha) for sha in (base, tested)):
        raise ValueError("Contract checks require frozen base/tested SHAs")
    if git(root, "rev-parse", "HEAD").strip() != tested:
        raise ValueError("Contract checkout differs from tested SHA")
    paths = git(
        root, "diff", "--name-only", "-z", "--diff-filter=ACMRT", base, tested
    ).split("\0")
    changed = git(root, "diff", "--name-status", base, tested).splitlines()
    if not changed:
        raise ValueError("Contract check has no changed files to verify")
    warnings = []
    try:
        git(root, "diff", "--check", base, tested, "--", "*.md", "*.rst", "*.txt")
    except subprocess.CalledProcessError as exc:
        if exc.returncode != 2:
            raise
        warnings.append("文档格式提示（非阻塞）：\n" + exc.output.decode("utf-8").strip())
    rows = []
    for relative in filter(None, paths):
        path = root / relative
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("Changed contract path cannot be resolved: " + relative) from exc
        if not resolved.is_relative_to(root):
            raise ValueError("Changed contract path escaped checkout: " + relative)
        if path.is_symlink():
            rows.append({
                "path": relative,
                "sha256": hashlib.sha256(str(path.readlink()).encode("utf-8")).hexdigest(),
                "target": resolved.relative_to(root).as_posix(),
                "checks": ["symlink_target"],
            })
            continue
        if not path.is_file():
            # Gitlink trees have independent policy classification/build checks.
            continue
        suffix = path.suffix.lower()
        control = relative.startswith(
            (".github/", "scripts/", "api_contract/", "dashboard/")
        ) or path.name.lower() in {
            "pyproject.toml", "setup.py", "setup.cfg", ".gitmodules",
            "license", "notice", ".gitignore", ".editorconfig",
        }
        static_asset = relative.startswith(("docs/", "assets/")) and suffix in {
            ".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif",
        }
        if (
            suffix
            not in {
                ".md",
                ".rst",
                ".txt",
                ".json",
                ".yaml",
                ".yml",
                ".py",
                ".sh",
                ".bash",
                ".toml",
                ".cfg",
                ".js",
                ".mjs",
                ".cjs",
                ".css",
                ".html",
            }
            and not control
            and not static_asset
        ):
            continue
        raw = path.read_bytes()
        if static_asset and suffix != ".svg":
            rows.append({
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "checks": [],
            })
            continue
        if b"\0" in raw:
            continue
        text = raw.decode("utf-8")
        if re.search(r"^(?:<{7}|>{7})(?:\s|$)", text, re.M):
            raise ValueError("Unresolved conflict marker in " + relative)
        checks = ["utf8", "no_conflict_markers"]
        if suffix == ".json":
            json.loads(text)
            checks.append("json_parse")
        elif suffix in {".yml", ".yaml"}:
            workflow = relative.startswith(".github/workflows/")
            documents = yaml_documents(text, allow_local_tags=not workflow)
            checks.append("yaml_parse")
            if workflow:
                if len(documents) != 1:
                    raise ValueError("Workflow requires exactly one YAML document")
                workflow_contract(documents[0])
                checks.append("workflow_contract")
        elif suffix == ".py":
            ast.parse(text, filename=relative)
            checks.append("python_ast")
        elif suffix in {".sh", ".bash"} or re.match(
            r"^#!.*\b(?:bash|sh)\s*$", text.splitlines()[0] if text else ""
        ):
            subprocess.run(["bash", "-n", str(path)], check=True)
            checks.append("shell_syntax")
        rows.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "checks": checks,
            }
        )
    if not rows and any(not row.startswith("D\t") for row in changed):
        raise ValueError(
            "No supported changed documentation/control file contract was verified"
        )
    return {
        "status": "pass",
        "base_sha": base,
        "tested_sha": tested,
        "changed_files": changed,
        "verified_files": rows,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--base", required=True)
    parser.add_argument("--tested", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = check(args.root, args.base, args.tested)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        "Verified diff and",
        len(result["verified_files"]),
        "changed documentation/control files",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
