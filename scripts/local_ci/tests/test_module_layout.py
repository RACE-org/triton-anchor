import re
from pathlib import Path


LOCAL_CI_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_PATHS = (
    "poll_gitee_and_run.sh",
    "orchestration/run_deterministic_ci_in_container.sh",
    "orchestration/fetch_task_metadata.sh",
    "deterministic_ci/run_deterministic_ci.sh",
    "deterministic_ci/flaggems/batch_test_flaggems.py",
    "deterministic_ci/flaggems/select_flaggems_tests.py",
    "deterministic_ci/flaggems/flaggems_all_ops.tsv",
    "deterministic_ci/flaggems/flaggems_pass_whitelist.tsv",
    "deterministic_ci/performance/compile_benchmark.py",
    "deterministic_ci/performance/compare_compile_time.py",
    "deterministic_ci/performance/pass_profile_benchmark.py",
    "deterministic_ci/performance/compare_pass_profile.py",
    "deterministic_ci/performance/ir_serialization_benchmark.py",
    "deterministic_ci/performance/compare_ir_serialization.py",
    "codex_ai/run_codex_ai_ci.sh",
    "codex_ai/classify_codex_review_context.py",
    "codex_ai/prepare_codex_checkout.sh",
    "codex_ai/setup_codex_ai_container.sh",
    "codex_ai/validate_codex_ai_credentials.py",
    "codex_ai/build_codex_ai_report.py",
    "codex_ai/codex_jsonl_evidence.py",
    "codex_ai/render_codex_ai_report.py",
    "codex_ai/codex_ai_analysis.schema.json",
    "codex_ai/codex_ai_report.schema.json",
    "codex_ai/prompts/codex_ai_success.md",
    "codex_ai/prompts/codex_ai_failure.md",
    "results/publish_gitee_result.py",
    "results/bridge_gitee_to_github_status.py",
    "maintenance/local_ci_health.py",
    "maintenance/manage_local_ci_state.py",
    "shared/result_paths.py",
    "shared/finding_locations.py",
    "shared/dump_artifacts.py",
    "shared/task_tmp.py",
    "shared/path_utils.sh",
    "shared/resolve_ci_profile.py",
    "shared/validate_task_metadata.py",
    "deterministic_ci/performance/common.py",
)

RUNTIME_DIRECTORIES = (
    "orchestration",
    "deterministic_ci",
    "codex_ai",
    "results",
    "maintenance",
    "shared",
)
RUNTIME_SUFFIXES = {".json", ".py", ".sh", ".tsv"}
SERVER_ONLY_PATHS = {"maintenance/manage_local_ci_state.py"}


def staged_runner_requirements() -> set[str]:
    poller = (LOCAL_CI_ROOT / "poll_gitee_and_run.sh").read_text(encoding="utf-8")
    match = re.search(
        r"for required_path in \\\n(?P<body>.*?)\n\s*shared/validate_task_metadata\.py; do",
        poller,
        flags=re.DOTALL,
    )
    assert match is not None, "unable to locate staged runner requirement list"
    paths = []
    for line in match.group("body").splitlines():
        value = line.strip().removesuffix("\\").strip()
        if value:
            paths.append(value)
    paths.append("shared/validate_task_metadata.py")
    return set(paths)


def discovered_runtime_paths() -> set[str]:
    paths = {"poll_gitee_and_run.sh"}
    for directory in RUNTIME_DIRECTORIES:
        for path in (LOCAL_CI_ROOT / directory).rglob("*"):
            if not path.is_file() or "tests" in path.relative_to(LOCAL_CI_ROOT).parts:
                continue
            if path.suffix in RUNTIME_SUFFIXES:
                paths.add(path.relative_to(LOCAL_CI_ROOT).as_posix())
    paths.update(
        path.relative_to(LOCAL_CI_ROOT).as_posix()
        for path in (LOCAL_CI_ROOT / "codex_ai" / "prompts").glob("codex_ai_*.md")
    )
    return paths


def test_canonical_local_ci_modules_exist():
    assert len(CANONICAL_PATHS) == len(set(CANONICAL_PATHS))
    missing = [path for path in CANONICAL_PATHS if not (LOCAL_CI_ROOT / path).is_file()]
    assert not missing, f"missing canonical Local CI modules: {missing}"


def test_canonical_modules_cover_all_runtime_sources():
    assert set(CANONICAL_PATHS) == discovered_runtime_paths()


def test_staged_runner_requirements_match_canonical_modules():
    assert staged_runner_requirements() == set(CANONICAL_PATHS) - SERVER_ONLY_PATHS


def test_development_guide_is_excluded_from_formal_package():
    guide = LOCAL_CI_ROOT / "DEVELOPMENT_GUIDE.md"
    assert not guide.exists()
    assert "DEVELOPMENT_GUIDE.md" not in CANONICAL_PATHS


def test_local_ci_root_has_only_the_stable_poller_entrypoint():
    root_scripts = {
        path.name
        for path in LOCAL_CI_ROOT.iterdir()
        if path.is_file() and path.suffix in {".sh", ".py"}
    }
    assert root_scripts == {"poll_gitee_and_run.sh"}


def test_obsolete_poller_module_is_removed():
    assert not (LOCAL_CI_ROOT / "orchestration" / "poll_gitee_tasks.sh").exists()


def test_removed_codex_and_pr_mirror_modules_stay_removed():
    obsolete_paths = (
        "codex_ai/normalize_codex_ai_report.py",
        "codex_ai/repair_codex_ai_analysis.py",
        "upstream_pr_mirror",
    )
    present = [path for path in obsolete_paths if (LOCAL_CI_ROOT / path).exists()]
    assert not present, f"obsolete Local CI paths still exist: {present}"


def test_shell_runners_use_the_shared_path_normalizer():
    expected_sources = {
        "poll_gitee_and_run.sh": 'source "${LOCAL_CI_ROOT}/shared/path_utils.sh"',
        "deterministic_ci/run_deterministic_ci.sh": (
            'source "${RUNNER_ROOT}/shared/path_utils.sh"'
        ),
    }
    for relative_path, expected_source in expected_sources.items():
        text = (LOCAL_CI_ROOT / relative_path).read_text(encoding="utf-8")
        assert expected_source in text
        assert "safe_path_part() {" not in text
