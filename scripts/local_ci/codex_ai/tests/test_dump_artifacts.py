import json
import subprocess
import sys
from pathlib import Path


TOOL = Path(__file__).resolve().parents[2] / "shared" / "dump_artifacts.py"
LOCAL_CI_ROOT = TOOL.parent.parent
DETERMINISTIC_RUNNER = LOCAL_CI_ROOT / "deterministic_ci" / "run_deterministic_ci.sh"
CODEX_RUNNER = LOCAL_CI_ROOT / "codex_ai" / "run_codex_ai_ci.sh"


def run_tool(*arguments: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(TOOL), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_collect_preserves_only_failure_ir_text(tmp_path: Path):
    task_root = tmp_path / "task"
    global_root = tmp_path / "global"
    output_root = tmp_path / "artifacts" / "failure-ir"
    (task_root / "kernel-a").mkdir(parents=True)
    global_root.mkdir()
    (task_root / "kernel-a" / "module.ttir").write_text("ttir", encoding="utf-8")
    (task_root / "kernel-a" / "module.linalg").write_text("linalg", encoding="utf-8")
    (task_root / "kernel-a" / "module.pplir").write_text("pplir", encoding="utf-8")
    (task_root / "kernel-a" / "kernel.so").write_bytes(b"binary")
    (task_root / "kernel-a" / "debug.log").write_text("log", encoding="utf-8")
    (global_root / "fallback.ttir").write_text("global", encoding="utf-8")

    result = run_tool(
        "collect",
        "--output-dir",
        str(output_root),
        "--stage",
        "Backend smoke and JIT",
        "--target-sha",
        "a" * 40,
        "--source",
        f"task={task_root}",
        "--source",
        f"global={global_root}",
    )

    assert result["file_count"] == 4
    assert result["stage"] == "Backend-smoke-and-JIT"
    copied_files = {
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    assert copied_files == {
        "Backend-smoke-and-JIT/task/kernel-a/module.ttir",
        "Backend-smoke-and-JIT/task/kernel-a/module.linalg",
        "Backend-smoke-and-JIT/task/kernel-a/module.pplir",
        "Backend-smoke-and-JIT/global/fallback.ttir",
    }
    manifest = json.loads(
        (output_root / "Backend-smoke-and-JIT" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["schema"] == "triton-anchor-local-ci-failure-ir/v1"
    assert manifest["target_sha"] == "a" * 40


def test_collect_without_ir_does_not_create_artifact_directory(tmp_path: Path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "failure-ir"
    source_root.mkdir()
    (source_root / "kernel.so").write_bytes(b"binary")

    result = run_tool(
        "collect",
        "--output-dir",
        str(output_root),
        "--stage",
        "failed-stage",
        "--target-sha",
        "b" * 40,
        "--source",
        f"task={source_root}",
    )

    assert result["file_count"] == 0
    assert not output_root.exists()


def test_dump_prune_removes_only_shared_fallback_dump_state(tmp_path: Path):
    dump = tmp_path / "root/.triton/dump"
    sophgo_dump = tmp_path / "workspace/triton-dump-dir"
    task_dump = tmp_path / "tmp/triton-anchor-local-ci-dump.abc123"
    cache = tmp_path / "root/.triton/cache"
    flaggems_cache = tmp_path / "root/.flaggems/code_cache"
    benchmark_dir = tmp_path / "tmp/triton_anchor_compile_bench"
    for directory in (
        dump,
        sophgo_dump,
        task_dump,
        cache,
        flaggems_cache,
        benchmark_dir,
    ):
        directory.mkdir(parents=True)
        (directory / "payload.so").write_bytes(b"unused")
    unrelated = tmp_path / "root/keep/sentinel.txt"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("keep", encoding="utf-8")

    result = run_tool("prune", "--profile", "task-dumps", "--root", str(tmp_path))

    assert result["removed_files"] == 2
    assert result["removed_bytes"] == 2 * len(b"unused")
    assert not any(dump.iterdir())
    assert not any(sophgo_dump.iterdir())
    assert (task_dump / "payload.so").read_bytes() == b"unused"
    assert (cache / "payload.so").read_bytes() == b"unused"
    assert (flaggems_cache / "payload.so").read_bytes() == b"unused"
    assert (benchmark_dir / "payload.so").read_bytes() == b"unused"
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_deterministic_runner_owns_dump_lifecycle():
    runner = DETERMINISTIC_RUNNER.read_text(encoding="utf-8")

    assert 'export TRITON_DUMP_DIR="${task_dump_dir}"' in runner
    assert 'SOPHGO_TRITON_DUMP_DIR="/workspace/triton-dump-dir"' in runner
    assert 'export TRITON_DUMP_DIR="${command_dump_dir}"' in runner
    assert "prune_task_dumps || return $?" in runner
    assert (
        'mkdir -p "${LOCAL_CI_TASK_DUMP_ROOT}/${safe_stage}" || return $?'
        in runner
    )
    assert 'if [[ ${stage_status} -ne 0 ]]; then' in runner
    assert 'collect_failure_ir "${stage_name}" "${task_dump_dir}"' in runner
    assert 'prune_task_dumps || cleanup_status=$?' in runner
    assert (
        'run_logged_in_dir "${BACKEND_PATH}" backend-rebuild '
        "rebuild_backend_command" in runner
    )
    assert "failure_ir_artifact_dir:" in runner
    assert "triton_dump_cleanup_status:" in runner


def test_codex_snapshot_does_not_clean_or_audit_source_container():
    runner = CODEX_RUNNER.read_text(encoding="utf-8")
    failure_prompt = (
        LOCAL_CI_ROOT / "codex_ai" / "prompts" / "codex_ai_failure.md"
    ).read_text(encoding="utf-8")

    assert "dump_artifacts.py" not in runner
    assert "snapshot_prune" not in runner
    assert "snapshot_hygiene" not in runner
    assert "docker commit \\" in runner
    assert '"${LOCAL_CI_CONTAINER}" "${ephemeral_image}"' in runner
    assert "LABEL triton-anchor.role=codex-ai-snapshot" in runner
    assert "export TRITON_DUMP_DIR=/tmp/triton-anchor-codex-dump" in runner
    assert "`${ARTIFACT_DIR}/failure-ir/`" in failure_prompt
    assert "不要搜索" in failure_prompt
    assert "`/root/.triton/dump`" in failure_prompt
    assert "`/workspace/triton-dump-dir`" in failure_prompt
