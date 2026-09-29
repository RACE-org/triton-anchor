"""Read published evidence for historical display; never publish GitHub statuses."""

import hashlib
import json
import re
from datetime import datetime, timezone
from urllib.parse import quote

from agent_ci.delivery import MAX_FULL_FLAGGEMS_BYTES, validate_full_flaggems
from agent_ci.protocol import FULL_FLAGGEMS_ROOT, RESULT_SCHEMA, validate_result


FULL_FLAGGEMS_DEMO_SHA = "3d4c586307dcc3c1f11e650c67529b85da3dd22f"
FULL_FLAGGEMS_DEMO_RUN = "20260724T112410Z-3d4c586307dc"


def _read(path):
    if path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Invalid historical result file")
    data = json.loads(path.read_bytes())
    if not isinstance(data, dict):
        raise ValueError("Historical report must be an object")
    return data


def _url(results, path):
    base = results.url.removesuffix(".git").rstrip("/")
    if not base.startswith("https://gitee.com/"):
        return ""
    return base + "/blob/" + results.branch + "/" + quote(path.relative_to(results.root).as_posix(), safe="/")


def _full_flaggems(path):
    if path.is_symlink() or path.stat().st_size > MAX_FULL_FLAGGEMS_BYTES:
        raise ValueError("Invalid full FlagGems business result")
    return validate_full_flaggems(json.loads(path.read_bytes()))


def _full_check(row):
    if (row.get("task") or {}).get("full") is not True:
        return None
    for check in (row.get("result") or {}).get("checks", []):
        if check.get("tool_id") != "flaggems" or check.get("status") not in {
            "pass",
            "fail",
        }:
            continue
        mode = (check.get("parameters") or {}).get("mode")
        if mode is None:
            mode = (
                (check.get("details") or {})
                .get("flaggems-summary", {})
                .get("mode")
            )
        if mode == "full":
            return check
    return None


def _demo_full_row(results, path, document):
    """Keep the named historical sample available if its old task was retired."""
    completed = "2026-07-24T11:24:10+00:00"
    status = document.get("summary", {}).get("status", "fail")
    task_id = hashlib.sha256(f"full-demo:{FULL_FLAGGEMS_DEMO_SHA}".encode()).hexdigest()
    task = {
        "task_id": task_id,
        "repository": "RACE-org/triton-anchor",
        "event_kind": "manual",
        "pr_number": 0,
        "target_branch": "ci/full/jiwang-delivery-ci",
        "head_sha": FULL_FLAGGEMS_DEMO_SHA,
        "tested_sha": FULL_FLAGGEMS_DEMO_SHA,
        "captured_at": completed,
        "full": True,
    }
    result = {
        "run_id": FULL_FLAGGEMS_DEMO_RUN,
        "completed_at": completed,
        "status": status,
        "summary": "2026-07-24 历史全量算子样例",
        "checks": [{
            "tool_id": "flaggems",
            "status": status,
            "summary": "历史全量算子测试结果",
            "parameters": {"mode": "full"},
            "details": {"flaggems-summary": document},
        }],
        "reviews": [],
        "findings": [],
        "blocking_reasons": [],
        "limitations": [],
        "artifacts": [],
        "environment": {"variants": {"candidate": {
            "backend_enabled": True,
            "backend_profile": "sophgo-cmodel",
            "profile": "",
        }}},
    }
    return {
        "task": task,
        "result": result,
        "status": status,
        "historical": True,
        "result_url": _url(results, path),
        "artifact_urls": {},
    }


def attach_full_flaggems(results, rows):
    """Join independent full reports into the generated feed, never result.json."""
    root = results.root / FULL_FLAGGEMS_ROOT
    paths = []
    if root.is_dir():
        paths.extend(root.glob("*/flaggems-summary.json"))
        paths.extend(root.glob("*/*/flaggems-summary.json"))
    for path in sorted(paths):
        try:
            relative = path.relative_to(root)
            parts = relative.parts
            sha = parts[0]
            if not re.fullmatch(r"[a-f0-9]{40}", sha):
                continue
            run_id = parts[1] if len(parts) == 3 else ""
            demo_sample = sha == FULL_FLAGGEMS_DEMO_SHA and run_id in {
                "",
                FULL_FLAGGEMS_DEMO_RUN,
            }
            document = _full_flaggems(path)
        except (OSError, ValueError, TypeError, KeyError):
            continue

        matches = [row for row in rows if row.get("task", {}).get("tested_sha") == sha]
        if run_id:
            match = next(
                (
                    row
                    for row in matches
                    if (row.get("result") or {}).get("run_id") == run_id
                    and _full_check(row) is not None
                ),
                None,
            )
        elif demo_sample:
            # The preserved sample belongs only to its named historical run.
            # Never inject it into an unrelated impact check that happens to
            # share the same tested commit.
            match = next(
                (
                    row
                    for row in matches
                    if (row.get("result") or {}).get("run_id")
                    == FULL_FLAGGEMS_DEMO_RUN
                    and _full_check(row) is not None
                ),
                None,
            )
        else:
            match = next((row for row in matches if _full_check(row)), None)
        if match is None and demo_sample:
            match = _demo_full_row(results, path, document)
            rows.append(match)
        check = _full_check(match) if match is not None else None
        if check is None:
            continue
        check.setdefault("details", {})["flaggems-summary"] = document
        check.setdefault("parameters", {})["mode"] = "full"
        match["business_full"] = {
            "data_mode": "mock" if demo_sample else "live",
            "source_path": path.relative_to(results.root).as_posix(),
            "source_note": (
                "历史样例，仅在没有新的合规全量算子结果时展示。"
                if demo_sample
                else "独立发布的全量算子结果。"
            ),
        }
    return rows


def history_rows(results, current):
    """Include completed older runs, including pre-unification delivery reports."""
    seen = {(row["task"]["task_id"], (row.get("result") or {}).get("run_id")) for row in current}
    rows = []
    root = results.root / "runs"
    paths = (path for path in root.rglob("result.json")
             if len(path.relative_to(root).parts) <= 6
             and "artifacts" not in path.relative_to(root).parts)
    for path in paths:
        try:
            result = _read(path)
            if result.get("schema") != RESULT_SCHEMA:
                continue
            validate_result(result)
            key = (result["task"]["task_id"], result["run_id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"task": result["task"], "result": result, "status": result["status"],
                         "historical": True, "result_url": _url(results, path),
                         "artifact_urls": {item["path"]: _url(results, path.parent / "artifacts" / item["path"])
                                           for item in result["artifacts"] if not item.get("omitted")}})
        except (OSError, ValueError, TypeError, KeyError):
            continue
    # Legacy reports are displayed as historical measurements, not accepted as CI gates.
    for path in root.rglob("delivery-summary.txt"):
        if "artifacts" in path.relative_to(root).parts or path.is_symlink():
            continue
        try:
            if path.stat().st_size > 64 * 1024:
                continue
            fields = dict(line.split(": ", 1) for line in path.read_text(encoding="utf-8").splitlines() if ": " in line)
            sha = fields.get("target_sha", "")
            if not re.fullmatch(r"[a-f0-9]{40}", sha):
                continue
            run_id = path.parent.name
            date = datetime.strptime(run_id[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat()
            identifier = hashlib.sha256(path.relative_to(root).as_posix().encode()).hexdigest()
            checks, artifacts, links = [], [], {}
            files = {"flaggems": "flaggems-summary.json", "compile_time": "compile-benchmark.json",
                     "pass_profile": "pass-profile.json", "ir_serialization": "ir-serialization.json"}
            stages = {"backend_rebuild": "backend_rebuild_status", "backend_smoke": "backend_smoke_jit_status",
                      **{name: name + "_status" for name in files}}
            for name, field in stages.items():
                value = fields.get(field)
                if value not in {"pass", "fail", "error", "skipped", "timeout"}:
                    continue
                details = {}
                file = path.parent / files.get(name, "unused")
                if file.is_file():
                    data = _read(file)
                    details = ({"flaggems-summary": data} if name == "flaggems" else
                               {"candidate": {"summary": data.get("summary", {})}})
                    artifacts.append({"path": file.name, "size": file.stat().st_size})
                    links[file.name] = _url(results, file)
                checks.append({"tool_id": name, "status": {"error": "infra_error", "timeout": "infra_error"}.get(value, value),
                               "summary": "历史执行记录", "details": details})
            if not checks:
                continue
            task = {"task_id": identifier, "repository": "RACE-org/triton-anchor",
                    "pr_number": 0, "head_sha": sha, "tested_sha": sha, "captured_at": date,
                    "target_branch": fields.get("branch", "历史运行"), "full": fields.get("flaggems_test_mode") == "full"}
            outcome = "pass" if fields.get("status") == "0" else "fail"
            backend_profile = fields.get("backend_profile", "").strip()
            candidate = {
                "backend_profile": backend_profile,
                "backend_enabled": bool(backend_profile),
                "profile": fields.get("triton_profile", ""),
                "triton_version": fields.get("triton_version", ""),
            }
            result = {"run_id": run_id, "completed_at": date, "status": outcome, "checks": checks,
                      "reviews": [], "findings": [], "blocking_reasons": [], "artifacts": artifacts,
                      "summary": "历史版本的测试与性能记录", "environment": {"variants": {"candidate": candidate}}}
            rows.append({"task": task, "result": result, "status": outcome, "historical": True,
                         "result_url": _url(results, path), "artifact_urls": links})
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return rows
