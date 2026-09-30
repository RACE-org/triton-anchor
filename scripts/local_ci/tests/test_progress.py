"""Progress is optional and must never change task ownership or final results."""
import copy
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from agent_ci import progress as p


def fixture():
    now = datetime.now(timezone.utc).isoformat()
    task = {"task_id": "task-1", "repository": "org/repo", "head_sha": "a" * 40, "tested_sha": "b" * 40}
    row = {**task, "run_id": "20260920T120000Z-1", "updated_at": now,
           "stage": "running", "budget": {"codex_attempts_used": 3},
           "recovery": {"state": "recovering", "action": "resume"}}
    snapshot = {"worker_id": "worker-1", "collected_at": now, "tasks": [row]}
    status = {"id": 1, "context": "Summary", "state": "pending",
              "creator": {"login": "github-actions[bot]"}, "description": "waiting",
              "target_url": "https://github.com/org/repo/actions/runs/1#local-ci-task=task-1"}
    reader = p.ReceiverProgress({"worker_id": "worker-1", "health_stale_seconds": 1200})
    gh = Mock()
    gh.latest_summary.return_value = status
    return reader, gh, task, snapshot, status


def test_progress_polls_at_most_every_five_minutes_and_deduplicates():
    reader, gh, task, snapshot, status = fixture()
    with patch.object(p, "read_health", return_value=snapshot) as read:
        reader.update(gh, task)
        reader.update(gh, task)
        read.assert_called_once()
        gh.status.assert_called_once()
        description = gh.status.call_args.args[2]
        assert "recovering; resume; Codex attempt 3/10" in description
        status["description"] = description
        reader.next_poll = 0
        reader.update(gh, task)
        gh.status.assert_called_once()


def test_only_fresh_exact_identity_can_report_progress():
    reader, gh, task, snapshot, status = fixture()
    now = datetime.now(timezone.utc).timestamp()
    assert reader.description(snapshot, task, now)
    variants = []
    for key in ("task_id", "repository", "head_sha", "tested_sha", "run_id"):
        variant = copy.deepcopy(snapshot)
        variant["tasks"][0][key] = ""
        variants.append(variant)
    for key, value in (("worker_id", "other"), ("collected_at", "2020-01-01T00:00:00Z")):
        variants.append({**snapshot, key: value})
    variants.append({**snapshot, "tasks": [snapshot["tasks"][0]] * 2})
    variants.append({**snapshot, "tasks_available": False})
    for variant in variants:
        assert reader.description(variant, task, now) is None
    reader.last_source = now + 1
    assert reader.description(snapshot, task, now) is None


def test_no_summary_creation_or_terminal_regression_even_during_read():
    for state in (None, "success", "failure", "error"):
        reader, gh, task, snapshot, status = fixture()
        gh.latest_summary.return_value = None if state is None else {**status, "state": state}
        with patch.object(p, "read_health") as read:
            reader.update(gh, task)
            read.assert_not_called()
            gh.status.assert_not_called()
    reader, gh, task, snapshot, status = fixture()
    gh.latest_summary.side_effect = [status, {**status, "id": 2, "state": "success"}]
    with patch.object(p, "read_health", return_value=snapshot):
        reader.update(gh, task)
    gh.status.assert_not_called()


def test_optional_read_failures_do_not_break_receiver():
    for error in (TimeoutError(), ValueError(), KeyError("content"), AttributeError()):
        reader, gh, task, snapshot, status = fixture()
        with patch.object(p, "read_health", side_effect=error):
            reader.update(gh, task)
        gh.status.assert_not_called()


def test_gitee_read_has_timeout_and_size_limit():
    config = {"health_repo_url": "https://gitee.com/org/health.git", "worker_id": "worker-1"}
    response = Mock()
    response.read.return_value = b"x" * (p.MAX_BYTES + 1)
    context = Mock()
    context.__enter__ = Mock(return_value=response)
    context.__exit__ = Mock(return_value=False)
    with patch.object(p, "urlopen", return_value=context) as fetch:
        try:
            p.read_health(config)
            assert False, "oversized snapshot accepted"
        except ValueError:
            pass
        assert fetch.call_args.kwargs["timeout"] == 10
        response.read.assert_called_once_with(p.MAX_BYTES + 1)
