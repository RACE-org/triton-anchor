"""Describe a frozen diff and suggest checks for Codex to select."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .protocol import ContractError, LLVM_METADATA

# Tool dependencies describe preparation once Codex selects a tool; paths do not
# expand into an obligatory execution plan.
from tools.basic_tools.runner import TOOL_IDS

TOOLS = tuple(TOOL_IDS)
FRONTEND = {
    "environment",
    "frontend_build",
    "frontend_install",
    "frontend_tests",
    "frontend_smoke",
}
BACKEND = {
    "backend_build",
    "backend_install",
    "backend_tests",
    "backend_smoke",
    "flaggems",
    "compile_time",
    "pass_profile",
    "ir_serialization",
}
CHECK_ORDER = TOOLS


def changed_files(repo: Path, base: str, tested: str) -> list[dict]:
    raw = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={repo.resolve()}",
            "-c",
            "core.fsmonitor=false",
            "diff",
            "--raw",
            "-z",
            "--no-ext-diff",
            "--no-abbrev",
            "--find-renames",
            base,
            tested,
            "--",
        ],
        cwd=repo,
    )
    parts, index, result = raw.decode("utf-8", "surrogateescape").split("\0"), 0, []
    while index < len(parts) and parts[index]:
        header = parts[index].split()
        if len(header) != 5 or not header[0].startswith(":") or index + 1 >= len(parts):
            raise ContractError("Malformed trusted git diff")
        old_mode, new_mode, _, _, status = header
        old_path = parts[index + 1]
        new_path = old_path
        index += 2
        if status.startswith(("R", "C")):
            if index >= len(parts):
                raise ContractError("Malformed rename")
            new_path = parts[index]
            index += 1
        change = {
            "old_path": old_path,
            "path": new_path,
            "status": status,
            "old_mode": old_mode[1:],
            "mode": new_mode,
        }
        result.append(change)
    return result


def category(path: str) -> str:
    p = path.lower()
    name = Path(p).name
    if name in {"agents.md", "skill.md", "ai_ci_program.md"}:
        return "control"
    if p.endswith((".md", ".rst")) or p in {"license", "notice"} or (
        p.startswith(("docs/", "assets/"))
        and p.endswith((".txt", ".png", ".jpg", ".jpeg", ".svg", ".webp", ".gif"))
    ):
        return "docs"
    if p in {".gitignore", ".editorconfig"}:
        return "control"
    if LLVM_METADATA.fullmatch(p):
        return "llvm"
    if p.endswith("cmakelists.txt") or p.endswith(".cmake"):
        return "compiler"
    if (
        p == ".gitmodules"
        or p.startswith("docker/")
        or "dockerfile" in p
        or p.endswith("envsetup.sh")
    ):
        return "environment"
    # Control-plane tests are control regressions, rather than product tests.
    if p.startswith(
        (
            ".github/",
            "scripts/ci/",
            "scripts/local_ci/",
            "dashboard/",
            "scripts/api_contract/",
        )
    ):
        return "control"
    parts = p.split("/")
    if (
        any(part in {"test", "tests"} for part in parts)
        or Path(p).name.startswith("test_")
    ) and (
        p.endswith(
            (
                ".py",
                ".c",
                ".cc",
                ".cpp",
                ".h",
                ".hpp",
                ".sh",
                ".json",
                ".toml",
                ".yaml",
                ".yml",
            )
        )
        or Path(p).name in {"pytest.ini", "tox.ini"}
    ):
        return "test"
    if p.startswith("api_contract/"):
        return "interface"
    if p.startswith("scripts/api_contract/"):
        return "control"
    if p in {"setup.py", "pyproject.toml", "manifest.in", "setup.cfg"}:
        return "packaging"
    if "requirements" in p or p.endswith((".lock", ".env")):
        return "environment"
    if p in {
        "python/triton_anchor/anchor_ir.py",
        "python/triton_anchor/hw_capability.py",
        "python/triton_anchor/adapters/base.py",
        "python/triton_anchor/adapters/registry.py",
        "python/triton_anchor/extensions/base.py",
        "python/triton_anchor/extensions/registry.py",
    }:
        return "interface"
    if p.startswith(
        (
            "csrc/",
            "triton/",
            "python/triton_anchor/adapters/",
            "python/triton_anchor/extensions/",
        )
    ) or any(
        name in p for name in ("pipeline", "anchor_ir", "hw_capability", "lowering")
    ):
        return "compiler"
    if p.startswith("python/triton_anchor/"):
        if any(v in p for v in ("jit", "cache", "concurrent")):
            return "compiler"
        return "frontend"
    if (
        p.startswith("docs/")
        or p in {"readme.md", "roadmap.md", "security.md", "license"}
    ) and p.endswith((".md", ".rst", ".txt")):
        return "docs"
    return "unknown"


def ordered(checks: set[str]) -> list[str]:
    return [tool for tool in CHECK_ORDER if tool in checks]


def minimum_checks(
    changes: list[dict], *, backend_enabled: bool, full: bool = False,
    event_kind: str = "pull_request",
) -> dict:
    if not changes:
        raise ContractError(
            "Empty diff needs an explicit branch validation task; it is not documentation"
        )
    groups: set[str] = set()
    test_paths: list[str] = []
    deleted_test = False
    classification_evidence = []
    for item in changes:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not item["path"]
            or not isinstance(item.get("old_path", item["path"]), str)
        ):
            raise ContractError("Invalid change manifest")
        item_groups = {
            category(item["path"]),
            category(item.get("old_path", item["path"])),
        }
        if item.get("mode") in {"160000", "120000"} or item.get("old_mode") in {
            "160000",
            "120000",
        }:
            item_groups.add("environment")
        groups |= item_groups
        classification_evidence.append({
            "path": item["path"],
            "old_path": item.get("old_path", item["path"]),
            "categories": sorted(item_groups),
        })
        if "test" in item_groups:
            if item.get("status", "").startswith("D"):
                deleted_test = True
            else:
                test_paths.append(item["path"])

    runtime_groups = groups - {"docs"}
    recommended = {"control_plane"} if not runtime_groups else {"environment"}
    recommended_parameters: dict[str, dict] = {}
    if "control" in runtime_groups:
        recommended.add("control_plane")
    if runtime_groups & {"frontend", "interface"}:
        recommended |= {"frontend_tests", "frontend_smoke"}
    if "packaging" in runtime_groups:
        recommended |= FRONTEND
    if "interface" in runtime_groups:
        recommended.add("backend_smoke")
    if "compiler" in runtime_groups:
        recommended |= FRONTEND | {"backend_tests", "backend_smoke", "flaggems"}
        recommended |= {"compile_time", "pass_profile", "ir_serialization"}
    if runtime_groups & {"environment", "llvm"}:
        recommended |= FRONTEND | {"control_plane", "backend_tests", "backend_smoke", "flaggems"}
        recommended |= {"compile_time", "pass_profile", "ir_serialization"}
    if "test" in runtime_groups:
        if "tests/test_smoke.py" in test_paths:
            recommended.add("frontend_smoke")
            test_paths.remove("tests/test_smoke.py")
        if test_paths or "frontend_smoke" not in recommended:
            recommended.add("frontend_tests")
        # Runnable changed tests are a useful starting point, not a coverage cap.
        selectable = [
            path
            for path in test_paths
            if Path(path).name.startswith("test_") and path.endswith(".py")
        ]
        if (
            selectable and len(selectable) == len(test_paths) and not deleted_test
            and runtime_groups <= {"test", "control"}
        ):
            recommended_parameters["frontend_tests"] = {"paths": sorted(set(selectable))}

    if full:
        level = "full"
    elif "unknown" in runtime_groups:
        level = "needs_analysis"
    elif runtime_groups & {"compiler", "environment", "llvm"}:
        level = "core"
    elif runtime_groups & {"frontend", "interface", "packaging"}:
        level = "frontend"
    elif "control" in runtime_groups:
        level = "control"
    elif runtime_groups:
        level = "test_only"
    else:
        level = "non_executable"

    unavailable = BACKEND if not backend_enabled else set()
    # `full` names the FlagGems coverage mode.  Codex still owns the execution
    # plan (including build/install preparation) and decides whether a separate
    # performance suite is relevant.
    required = {"flaggems"} if full else set()
    return {
        "categories": sorted(groups),
        "classification_evidence": classification_evidence,
        "impact": {
            "level": level,
            "classification": "path_hint"
            if runtime_groups
            else "documentation_only",
            "active_categories": sorted(runtime_groups),
        },
        "required_checks": ["change_validation", *ordered(required)],
        "required_parameters": {"flaggems": {"mode": "full"}} if full else {},
        "recommended_checks": ordered(recommended - required - unavailable),
        "recommended_parameters": recommended_parameters,
        "not_applicable": ordered((recommended | required) & unavailable),
        "capabilities": [t for t in CHECK_ORDER if t not in unavailable],
        "required_reviews": (["pr_info"] if event_kind == "pull_request" else []) + ["architecture"],
        "reason": (
            "Path categories only suggest checks. Codex must inspect the actual diff, choose relevant "
            "validation and record its reasoning and evidence in change_validation. Ordinary comments "
            "may use lightweight checks; real compiler/runtime changes need relevant builds and smoke/JIT. "
            "Explicit full requires FlagGems full; Codex prepares its dependencies and still decides "
            "whether separate performance measurements are relevant."
        ),
    }
