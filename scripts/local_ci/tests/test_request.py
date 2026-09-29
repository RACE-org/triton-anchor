"""Execute the trusted inline router and cleanup scripts against a fake GitHub API."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[3]
NODE = shutil.which("node") or shutil.which("node.exe")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is required for GitHub script tests")
REQUEST = yaml.safe_load((ROOT / ".github/workflows/ci-request.yml").read_text())
GATEWAY = yaml.safe_load((ROOT / ".github/workflows/ci-gateway.yml").read_text())
ROUTE, CLEANUP = [step["with"]["script"] for step in REQUEST["jobs"]["route"]["steps"]]
FINISH = GATEWAY["jobs"]["finish-request"]["steps"][0]["with"]["script"]
RESTART = GATEWAY["jobs"]["rerun-verification"]["steps"][0]["with"]["script"]
HEAD = "a" * 40
URL = "https://github.com/RACE-org/triton-anchor/actions/runs/100#local-ci-request=100:1"


def status(url=URL, state="pending"):
    return {"context": "Summary", "state": state, "target_url": url,
            "creator": {"login": "github-actions[bot]"}}


def execute(**options):
    driver = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const events = [], outputs = {}, rows = input.rows || [];
const head = 'a'.repeat(40);
const pull = {number: 7, state: input.closed ? 'closed' : 'open', draft: !!input.draft,
              head: {sha: head}, base: {ref: 'main'}};
const context = {repo: {owner: 'RACE-org', repo: 'triton-anchor'}, runId: 100,
                 actor: 'maintainer', eventName: input.manual ? 'workflow_dispatch' : input.push ? 'push' : 'pull_request_target',
                 ref: 'refs/heads/main', sha: head,
                 payload: input.push ? {created: !!input.created, forced: !!input.forced, deleted: false}
                                     : input.manual ? {} : {pull_request: pull, action: input.action || 'opened'}};
Object.assign(process.env, {SOURCE_BRANCH: 'main', REQUESTED_SHA: head, GITHUB_RUN_ATTEMPT: String(input.attempt || 1),
                           GITHUB_SERVER_URL: 'https://github.com', REQUEST_ID: '100:1',
                           ORIGINAL_INPUTS: JSON.stringify(input.original || {})});
const core = {info() {}, setOutput(key, value) { outputs[key] = value; },
              summary: {addHeading() {return this;}, addRaw() {return this;},
                        addLink() {return this;}, async write() {}}};
const github = {paginate: async () => rows, rest: {
  repos: {
    listCommitStatusesForRef() {},
    get: async () => ({data: {default_branch: 'main'}}),
    getCollaboratorPermissionLevel: async () => ({data: {permission: 'write'}}),
    getBranch: async ({branch}) => {
      if (input.lookupFailure && branch === 'CI_dev') throw Error('Control branch unavailable');
      return {data: {commit: {sha: branch === 'CI_dev' ? 'b'.repeat(40) : head}}};
    },
    createCommitStatus: async row => {
      events.push({kind: 'status', ...row});
      rows.unshift({...row, creator: {login: 'github-actions[bot]'}});
    }
  },
  pulls: {get: async () => ({data: {...pull, head: {sha: input.stale ? 'c'.repeat(40) : head}}})},
  actions: {createWorkflowDispatch: async row => {
    events.push({kind: 'dispatch', ...row});
    if (input.replacement) rows.unshift(input.replacement);
    if (input.dispatchFailure) throw Error('Dispatch failed');
  }}
}};
const AsyncFunction = Object.getPrototypeOf(async function() {}).constructor;
async function run(script) { await new AsyncFunction('github', 'context', 'core', script)(github, context, core); }
(async () => {
  if (input.restart) { await run(input.restartScript); }
  else if (input.finish) { await run(input.finishScript); }
  else {
    try { await run(input.route); }
    catch (error) {
      events.push({kind: 'error', message: error.message});
      if (outputs.summary_sha) {
        process.env.SUMMARY_SHA = outputs.summary_sha;
        process.env.REQUEST_URL = outputs.request_url;
        await run(input.cleanup);
      }
    }
  }
  process.stdout.write(JSON.stringify(events));
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run([NODE, "-e", driver], input=json.dumps({
        "route": ROUTE, "cleanup": CLEANUP, "finishScript": FINISH, "restartScript": RESTART, **options,
    }), text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize("options,trigger", [({}, ""), ({"manual": True}, "100:1"),
    ({"action": "synchronize"}, "100:1"), ({"action": "reopened"}, "100:1"),
    ({"action": "ready_for_review"}, "100:1"), ({"action": "edited"}, "100:1"),
    ({"action": "labeled"}, "100:1"), ({"action": "unlabeled"}, "100:1"),
    ({"attempt": 2}, "100:2"),
    ({"push": True}, ""), ({"push": True, "created": True}, "100:1"),
    ({"push": True, "forced": True}, "100:1"),
    ({"push": True, "attempt": 2}, "100:2"),
    ({"attempt": 2, "rows": [status(URL.replace('100:1', '101:1'))]}, "100:2")])
def test_request_publishes_pending_before_dispatch(options, trigger):
    events = execute(**options)
    attempt = options.get("attempt", 1)
    assert [row["kind"] for row in events] == ["status", "dispatch"]
    assert events[0]["state"] == "pending"
    assert events[0]["sha"] == HEAD
    assert events[0]["target_url"] == URL.replace("100:1", f"100:{attempt}")
    assert events[1]["inputs"]["request_id"] == f"100:{attempt}"
    assert events[1]["inputs"]["trigger_id"] == trigger
    assert REQUEST["jobs"]["route"]["permissions"]["statuses"] == "write"


@pytest.mark.parametrize("options", [{"stale": True}, {"rows": [status()]},
                                  {"rows": [status(URL.replace('100:1', '101:1'))]}])
def test_superseded_or_duplicate_requests_do_not_publish(options):
    assert execute(**options) == []


@pytest.mark.parametrize("flag,action", [("closed", "closed"), ("draft", "converted_to_draft")])
def test_inactive_pr_only_dispatches_cancellation(flag, action):
    events = execute(**{flag: True})
    assert [row["kind"] for row in events] == ["dispatch"]
    assert events[0]["inputs"]["action"] == action
    assert events[0]["inputs"]["request_id"] == ""
    assert events[0]["inputs"]["trigger_id"] == ""


def test_failed_only_rerun_dispatches_complete_verification_without_old_artifacts():
    jobs = GATEWAY["jobs"]
    restart = jobs["rerun-verification"]
    assert set(restart["needs"]) == set(jobs) - {"rerun-verification"}
    assert "always()" in restart["if"] and "github.run_attempt > 1" in restart["if"]
    assert all("github.run_attempt == 1" in job["if"]
               for key, job in jobs.items() if key != "rerun-verification")
    original = {"mode": "run", "pr_number": "7", "source_branch": "main",
                "requested_sha": HEAD, "request_id": "99:1", "trigger_id": "99:1",
                "worker_revision_sha": "c" * 40, "action": "opened", "full": True,
                "run_title": "PR #7"}
    event, = execute(restart=True, attempt=2, original=original)
    assert event["ref"] == "CI_dev"
    assert event["inputs"] == {**original, "request_id": "", "trigger_id": "100:2",
                               "worker_revision_sha": "b" * 40, "full": "true"}
    # Retrying delivery of this handoff does not invent another trigger.
    assert execute(restart=True, attempt=2, original=original) == [event]


@pytest.mark.parametrize("failure", ["lookupFailure", "dispatchFailure"])
def test_failed_request_finishes_its_pending_status(failure):
    events = execute(**{failure: True})
    assert [row["state"] for row in events if row["kind"] == "status"] == ["pending", "error"]


@pytest.mark.parametrize("replacement", [status(URL + '&local-ci-task=frozen'),
                                         status(URL.replace('100:1', '101:1'))])
def test_failed_request_cannot_overwrite_gateway_or_new_request(replacement):
    events = execute(dispatchFailure=True, replacement=replacement)
    assert [row["state"] for row in events if row["kind"] == "status"] == ["pending"]


@pytest.mark.parametrize("row,finishes", [(status(), True), (status(state="success"), False),
    (status(URL + '&local-ci-task=frozen'), False), (status(URL.replace('100:1', '101:1')), False)])
def test_initialization_cleanup_only_finishes_unclaimed_request(row, finishes):
    events = execute(finish=True, rows=[row])
    assert [row["state"] for row in events] == (["error"] if finishes else [])
