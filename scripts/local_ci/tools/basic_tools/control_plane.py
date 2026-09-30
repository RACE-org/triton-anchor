"""Changed-file contracts and actual candidate control-plane regressions."""

from __future__ import annotations

from pathlib import Path

from actions import candidate_python, run, write_json
from contract_checks import check


def execute(payload: dict) -> None:
    context = payload["context"]
    root = Path(context["source_dir"])
    out = Path(context["artifact_dir"]) / "control_plane"
    result = check(root, context["base_sha"], context["target_sha"])
    changed = [line.split("\t")[-1] for line in result["changed_files"]]
    regression_required = any(
        path.startswith(("scripts/", ".github/", "dashboard/"))
        and not path.endswith((".md", ".rst", ".txt"))
        for path in changed
    )
    selected = payload["parameters"].get("paths")
    if selected is None and regression_required:
        # Dashboard-only changes need Node regressions, API tooling needs its
        # own suite, and Local CI/gateway changes need Local CI regressions.
        roots = set()
        for path in changed:
            if path.endswith((".md", ".rst", ".txt")):
                continue
            if path.startswith("scripts/api_contract/"):
                roots.add("scripts/api_contract/tests")
            elif not path.startswith("dashboard/"):
                roots.add("scripts/local_ci/tests")
        selected = [
            name
            for name in sorted(roots)
            if (root / name).is_dir() and any((root / name).rglob("test_*.py"))
        ]
    dashboard_tests = []
    if any(
        path.startswith("dashboard/") or path.endswith((".js", ".mjs", ".cjs"))
        for path in changed
    ):
        dashboard_tests = sorted((root / "scripts/local_ci/tests").glob("*.test.cjs"))
        if dashboard_tests:
            run(["node", "--test", *map(str, dashboard_tests)], cwd=root)
    if selected:
        for value in selected:
            path = (root / value.split("::", 1)[0]).resolve(strict=True)
            if not path.is_relative_to(root.resolve()):
                raise ValueError(
                    "Control regression paths must stay inside the checkout"
                )
        command = [
            candidate_python(context),
            "-I",
            str(Path(__file__).with_name("pytest_exec.py")),
            "--output",
            str(out / "tests.json"),
            "--",
            "-q",
            "--import-mode=importlib",
            "-o",
            "addopts=",
        ]
        if payload["parameters"].get("keyword"):
            command += ["-k", payload["parameters"]["keyword"]]
        command += [str(root / path) for path in selected]
        run(command, cwd=out)
    missing_suite = regression_required and not selected and not dashboard_tests
    result.update(
        status="limited" if missing_suite else "pass",
        summary="仓库未提供对应回归套件；需按改动生成并执行定向测试。" if missing_suite else "契约检查与所选回归验证通过。",
        task_id=context["task_id"],
        target_sha=context["target_sha"],
        regression_required=regression_required,
        regression_paths=(selected or [])
        + [str(path.relative_to(root)) for path in dashboard_tests],
    )
    write_json(out / "control_plane.json", result)
