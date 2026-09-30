import assert from 'node:assert/strict';
import test from 'node:test';
import worker from './worker.mjs';

const ID = 'race-org-localci-v2';
const MARKER = `<!-- local-ci-alert:${ID} -->`;

function fixture() {
  const h = {
    now: Date.parse('2026-09-20T00:00:00Z'), stored: null, cached: null, puts: 0, cachePuts: 0, issues: [], writes: [],
    readFailure: false, alertsFailure: false, failCachePut: false,
    failPatch: false, loseCreateResponse: false, healthReads: 0, expectedHealthReads: 1,
    health: {
      schema: 'triton-anchor-worker-health', worker_id: ID, state: 'healthy',
      poller: { alive: true, heartbeat_stale: false, last_poll_status: 'success' },
      runtime: { available: true }, active_task: null, uploads: [], services: [],
      environments: { unavailable: false }, control_update: { state: 'idle' },
      storage: [{ filesystem_free_bytes: 50 * 1024 ** 3 }],
    },
  };
  h.health.collected_at = new Date(h.now).toISOString();
  h.env = {
    GITEE_TOKEN: 'PRIVATE_TOKEN',
    ALERT_STATE: {
      async get(key) {
        const stored = key.startsWith('local-ci-health-cache:') ? h.cached : h.stored;
        return stored ? JSON.parse(stored) : null;
      },
      async put(key, value) {
        if (key.startsWith('local-ci-health-cache:')) {
          h.cachePuts++;
          if (h.failCachePut) throw new Error('Cache storage unavailable');
          h.cached = value;
        } else { h.puts++; h.stored = value; }
      },
    },
  };
  h.advance = () => {
    h.now += 5 * 60 * 1000;
    h.health.collected_at = new Date(h.now).toISOString();
  };
  const json = value => new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } });
  h.fetch = async (url, options) => {
    assert.ok(!url.includes('PRIVATE_TOKEN'));
    const parsed = new URL(url);
    if (parsed.pathname.endsWith('/contents/worker-health.json')) {
      h.healthReads++;
      assert.equal(parsed.searchParams.get('ref'), `snapshot/${ID}`);
      assert.equal(options.headers.Authorization, 'Bearer PRIVATE_TOKEN');
      if (h.healthResponse) return h.healthResponse();
      if (h.readFailure) throw new Error('PRIVATE network failure');
      return json({ encoding: 'base64', content: Buffer.from(JSON.stringify(h.health)).toString('base64') });
    }
    assert.ok(!parsed.pathname.endsWith('/contents/watchdog.json'), 'watchdog has been retired');
    assert.equal(options.headers.Authorization, 'Bearer PRIVATE_TOKEN');
    if (options.method === 'GET') {
      if (h.alertsFailure) throw new Error('PRIVATE Issue error');
      return json(h.issues);
    }
    const body = JSON.parse(options.body);
    assert.equal(body.repo, 'triton-anchor-worker-health');
    assert.ok(!options.body.includes('PRIVATE'));
    h.writes.push({ method: options.method, body });
    if (options.method === 'POST') {
      const issue = { ...body, number: `I${h.issues.length + 1}`, state: 'open', updated_at: new Date(h.now).toISOString() };
      h.issues.push(issue);
      if (h.loseCreateResponse) {
        h.loseCreateResponse = false;
        throw new Error('Response lost');
      }
      return json(issue);
    }
    assert.equal(options.method, 'PATCH');
    if (h.failPatch) { h.failPatch = false; return new Response('', { status: 503 }); }
    const issue = h.issues.find(row => parsed.pathname.endsWith(`/${row.number}`));
    assert.ok(issue);
    Object.assign(issue, body);
    if (h.losePatchResponse) { h.losePatchResponse = false; throw new Error('Patch response lost'); }
    return json(issue);
  };
  h.run = async () => {
    const original = globalThis.fetch;
    globalThis.fetch = h.fetch;
    let pending;
    const before = h.puts;
    const beforeCache = h.cachePuts;
    const beforeReads = h.healthReads;
    try {
      worker.scheduled({ scheduledTime: h.now }, h.env, { waitUntil(value) { pending = value; } });
      await pending;
    } finally {
      globalThis.fetch = original;
      assert.equal(h.puts, before + 1, 'one incident-state write per execution');
      assert.equal(h.cachePuts, beforeCache + 1, 'one combined cache write per execution');
      assert.equal(h.healthReads, beforeReads + h.expectedHealthReads, 'monitor and cache share one health read');
    }
  };
  return h;
}

test('one Issue per incident, unchanged faults are quiet, changed faults update, recovery closes', async () => {
  const h = fixture();
  await h.run();
  assert.equal(h.writes.length, 0);
  h.health.poller.alive = false;
  h.health.private_error = 'PRIVATE diagnostic';
  await h.run();
  assert.equal(h.issues.length, 1);
  assert.ok(h.issues[0].body.includes(MARKER));
  h.advance();
  await h.run();
  assert.equal(h.writes.length, 1);
  h.health.runtime.available = false;
  await h.run();
  assert.equal(h.writes.length, 2);
  assert.ok(h.issues[0].body.includes('Docker'));
  h.advance();
  h.health.poller.alive = true;
  h.health.runtime.available = true;
  await h.run();
  assert.equal(h.issues[0].state, 'closed');
  assert.ok(h.issues[0].body.includes('10 分钟'));
  await h.run();
  assert.equal(h.writes.length, 3);
  h.health.runtime.available = false;
  await h.run();
  assert.equal(h.issues.length, 2);
});

test('control update faults are distinct, unknown status preserves them, and idle confirms recovery', async () => {
  const h = fixture();
  for (const state of ['failed', 'invalid', 'blocked']) {
    h.health.control_update = { state };
    await h.run();
    assert.deepEqual(JSON.parse(h.stored).codes, [`control_update_${state}`]);
    assert.equal(h.issues.length, 1);
    assert.equal(h.issues[0].state, 'open');
    const writes = h.writes.length;
    for (const unknown of [null, {}, { state: 'unknown' }]) {
      h.health.control_update = unknown;
      h.advance();
      await h.run();
      assert.deepEqual(JSON.parse(h.stored).codes, [`control_update_${state}`]);
      assert.equal(h.writes.length, writes);
    }
  }
  h.health.control_update = { state: 'idle' };
  await h.run();
  assert.equal(h.issues[0].state, 'closed');
  assert.deepEqual(JSON.parse(h.stored).codes, []);
});

test('two read failures trigger an observation alert; unreadable or stale data never clears service faults', async () => {
  const h = fixture();
  await h.run();
  const lastSuccess = new Date(h.now).toISOString();
  h.readFailure = true;
  h.advance();
  await h.run();
  assert.equal(h.issues.length, 0);
  h.advance();
  await h.run();
  assert.ok(h.issues[0].body.includes('不能据此判断服务器宕机'));
  assert.equal(JSON.parse(h.cached).health_read.last_success_at, lastSuccess);
  assert.equal(JSON.parse(h.cached).health_read.consecutive_failures, 2);
  h.advance();
  await h.run();
  assert.equal(h.writes.length, 1, 'time and failure count changes do not update the Issue');
  h.healthResponse = () => new Response('PRIVATE: denied', { status: 403 });
  h.advance();
  await h.run();
  assert.equal(h.writes.length, 2, 'changed diagnostic updates the existing Issue');
  assert.equal(h.issues.length, 1);
  assert.ok(h.issues[0].body.includes('HTTP 403'));
  h.healthResponse = null;
  h.readFailure = false;
  h.health.runtime.available = false;
  h.advance();
  await h.run();
  assert.ok(!JSON.parse(h.stored).codes.includes('source_unreadable'));
  assert.ok(!h.issues[0].body.includes('连续两次'), 'resolved observations are removed from the current alert');
  h.health.runtime.available = true;
  h.now += 25 * 60 * 1000;
  await h.run();
  assert.equal(h.issues[0].state, 'open');
  assert.ok(h.issues[0].body.includes('Docker'));
  assert.ok(h.issues[0].body.includes('20分钟未更新'));
  h.readFailure = true;
  await h.run();
  assert.equal(h.issues[0].state, 'open');
  h.readFailure = false;
  h.advance();
  await h.run();
  assert.equal(h.issues[0].state, 'closed');
  assert.equal(JSON.parse(h.cached).health_read.status, 'ok');
  assert.equal(JSON.parse(h.cached).health_read.consecutive_failures, 0);
});

test('health read failures publish only bounded diagnostics, never raw errors or response bodies', async t => {
  const warning = t.mock.method(console, 'warn', () => {});
  const cases = [
    ['network_error', null, () => { throw new Error('PRIVATE network details'); }],
    ['timeout', null, () => { throw new DOMException('PRIVATE timeout details', 'TimeoutError'); }],
    ['authentication_error', 401, () => new Response('PRIVATE unauthorized', { status: 401 })],
    ['http_error', 403, () => new Response('PRIVATE forbidden', { status: 403 })],
    ['rate_limited', 403, () => new Response('PRIVATE: Rate Limit Exceeded', { status: 403 })],
    ['rate_limited', 429, () => new Response('PRIVATE limited', { status: 429 })],
    ['invalid_document', 200, () => new Response('PRIVATE invalid JSON')],
    ['identity_mismatch', 200, null],
  ];
  for (const [code, status, response] of cases) {
    const h = fixture();
    h.healthResponse = response;
    if (!response) h.health.worker_id = 'another-worker';
    await h.run();
    const read = JSON.parse(h.cached).health_read;
    assert.equal(read.error_code, code);
    assert.equal(read.http_status, status);
    assert.equal(read.status, 'error');
    assert.equal(read.last_success_at, null);
    assert.ok(Number.isInteger(read.duration_ms) && read.duration_ms >= 0);
    assert.ok(!JSON.stringify(read).includes('PRIVATE'));
  }
  assert.equal(warning.mock.callCount(), cases.length);
  assert.ok(!JSON.stringify(warning.mock.calls).includes('PRIVATE'));
});

test('a missing Gitee secret is reported as a bounded configuration error', async t => {
  const warning = t.mock.method(console, 'warn', () => {});
  const h = fixture();
  delete h.env.GITEE_TOKEN;
  h.expectedHealthReads = 0;
  await assert.rejects(h.run(), /GITEE_TOKEN is not configured/);
  const read = JSON.parse(h.cached).health_read;
  assert.equal(read.error_code, 'configuration_error');
  assert.equal(read.http_status, null);
  assert.equal(read.status, 'error');
  assert.ok(!JSON.stringify(read).includes('GITEE_TOKEN'));
  assert.ok(!JSON.stringify(warning.mock.calls).includes('GITEE_TOKEN'));
});

test('missing or idle Codex telemetry cannot clear a known connection fault', async () => {
  const h = fixture();
  h.health.active_task = {task_id: 'task-a', run_id: 'run-1', stage: 'running', codex_status: 'connection_error',
    recovery: {state: 'retry_wait', failure_code: 'connection_error'},
    budget: {codex_attempts_used: 10, codex_attempts_limit: 10}};
  await h.run();
  for (const active of [{codex_status: 'connection_error'}, {}, null, {codex_status: 'starting'}]) {
    h.advance();
    h.health.active_task = active;
    await h.run();
    assert.equal(h.issues[0].state, 'open');
  }
  assert.equal(h.writes.length, 1);
  h.health.active_task = {task_id: 'task-a', run_id: 'run-1', stage: 'running', codex_status: 'running',
    codex_alive: true, last_progress_at: h.health.collected_at};
  await h.run();
  assert.equal(h.issues[0].state, 'closed');
});

test('a lost create response is reconciled before immediate recovery, without a duplicate Issue', async () => {
  const h = fixture();
  h.health.runtime.available = false;
  h.loseCreateResponse = true;
  await assert.rejects(h.run(), /Response lost/);
  assert.equal(h.issues.length, 1);
  h.health.runtime.available = true;
  h.advance();
  await h.run();
  assert.equal(h.issues.length, 1);
  assert.equal(h.issues[0].state, 'closed');
  assert.equal(h.writes.filter(row => row.method === 'POST').length, 1);
});

test('failed Issue updates and recovery are retried without losing the incident', async () => {
  const h = fixture();
  h.health.runtime.available = false;
  await h.run();
  h.health.poller.alive = false;
  h.failPatch = true;
  await assert.rejects(h.run(), /HTTP 503/);
  h.advance();
  await h.run();
  assert.ok(h.issues[0].body.includes('Worker未运行'));
  h.health.runtime.available = true;
  h.health.poller.alive = true;
  h.advance();
  h.failPatch = true;
  await assert.rejects(h.run(), /HTTP 503/);
  h.advance();
  await h.run();
  assert.equal(h.issues[0].state, 'closed');
  assert.equal(JSON.parse(h.stored).issue_number, null);
  assert.equal(h.issues.length, 1);
});

test('KV loss adopts a progressing Issue and preserves unknown Codex faults until explicit recovery', async () => {
  const h = fixture();
  h.health.active_task = { codex_status: 'auth_error' };
  await h.run();
  h.issues[0].state = 'progressing';
  h.stored = null;
  h.health.active_task = null;
  h.advance();
  await h.run();
  assert.equal(h.issues.length, 1);
  assert.equal(h.issues[0].state, 'progressing');
  assert.ok(h.issues[0].body.includes('Codex认证失败'));
  h.health.active_task = { codex_status: 'running' };
  h.stored = null;
  await h.run();
  assert.equal(h.issues[0].state, 'closed');
});

test('future-dated snapshots are unreadable; unavailable service telemetry does not imply recovery', async () => {
  const h = fixture();
  h.health.services = [{ name: 'health.timer', available: true, active_state: 'inactive' }];
  await h.run();
  assert.ok(h.issues[0].body.includes('维护服务失败'));
  h.health.services[0] = { name: 'health.timer', available: false, active_state: 'unknown' };
  await h.run();
  assert.equal(h.issues[0].state, 'open');
  h.health.services = [];
  await h.run();
  assert.equal(h.issues[0].state, 'open');
  h.health.collected_at = new Date(h.now + 120000).toISOString();
  await h.run();
  await h.run();
  assert.ok(h.issues[0].body.includes('连续两次无法读取'));
  h.advance();
  h.health.services = [{ name: 'health.timer', available: true, active_state: 'active' },
    { name: 'health.service', type: 'oneshot', available: true, active_state: 'inactive', result: 'success' }];
  await h.run();
  assert.equal(h.issues[0].state, 'closed');
});

test('public cache preserves original timestamps and last good data without exposing incident state or errors', async () => {
  const h = fixture();
  h.health.runtime.available = false;
  await h.run();
  const previous = JSON.parse(h.cached);
  assert.deepEqual(previous.worker, h.health);
  assert.equal(previous.watchdog, undefined);
  assert.equal(previous.events, undefined);
  assert.equal(previous.alerts.length, 1, 'newly created Issue is available in this round');
  h.advance();
  h.readFailure = h.alertsFailure = true;
  await h.run();
  const cache = JSON.parse(h.cached);
  assert.equal(cache.updated_at, new Date(h.now).toISOString());
  assert.equal(cache.worker.collected_at, previous.worker.collected_at);
  assert.deepEqual(cache.alerts, previous.alerts);
  assert.ok(Object.values(cache.errors).every(Boolean));
  assert.ok(!h.cached.includes('PRIVATE'));
  assert.equal(cache.issue_number, undefined);
});

test('Issue failure still refreshes cache, while cache failure still delivers and persists alerts', async () => {
  const h = fixture();
  h.health.runtime.available = false;
  h.loseCreateResponse = true;
  await assert.rejects(h.run(), /Response lost/);
  assert.equal(JSON.parse(h.cached).alerts.length, 1);
  h.failCachePut = true;
  h.health.poller.alive = false;
  await assert.rejects(h.run(), /Cache storage unavailable/);
  assert.ok(h.issues[0].body.includes('Worker未运行'));
  assert.ok(JSON.parse(h.stored).codes.includes('poller_unavailable'));
});

test('GET /health serves only the public KV cache with CORS and never requests Gitee', async () => {
  const h = fixture();
  const request = new Request('https://worker.example/health');
  let response = await worker.fetch(request, h.env);
  assert.equal(response.status, 503);
  await h.run();
  const original = globalThis.fetch;
  globalThis.fetch = () => { throw new Error('HTTP cache reads must not fetch Gitee'); };
  try {
    response = await worker.fetch(request, h.env);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('Access-Control-Allow-Origin'), '*');
    assert.equal(response.headers.get('Cache-Control'), 'public, max-age=60');
    assert.deepEqual(await response.json(), JSON.parse(h.cached));
    assert.equal((await worker.fetch(new Request(request, { method: 'POST' }), h.env)).status, 405);
    assert.equal((await worker.fetch(new Request(request, { method: 'OPTIONS' }), h.env)).status, 204);
    assert.equal((await worker.fetch(new Request('https://worker.example/'), h.env)).status, 404);
    h.env.ALERT_STATE.get = () => { throw new Error('PRIVATE KV failure'); };
    response = await worker.fetch(request, h.env);
    assert.equal(response.status, 503);
    assert.ok(!(await response.text()).includes('PRIVATE'));
  } finally { globalThis.fetch = original; }
});


test('older and same-time healthy snapshots cannot close a newer fault or roll back cache', async () => {
  const h = fixture();
  h.health.runtime.available = false;
  await h.run();
  const faultAt = h.health.collected_at;
  h.health.runtime.available = true;
  await h.run();
  assert.equal(h.issues[0].state, 'open', 'same source timestamp is not recovery evidence');
  h.health.collected_at = new Date(h.now - 60000).toISOString();
  await h.run();
  assert.equal(h.issues[0].state, 'open');
  assert.equal(JSON.parse(h.cached).worker.collected_at, faultAt);
  assert.match(JSON.parse(h.cached).errors.worker, /较旧/);
  h.advance();
  await h.run();
  assert.equal(h.issues[0].state, 'closed');
});

test('failed close is re-evaluated against unreadable, old or newly faulty snapshots before retry', async () => {
  for (const scenario of ['unreadable', 'old', 'fault']) {
    const h = fixture();
    h.health.runtime.available = false;
    await h.run();
    h.advance();
    h.health.runtime.available = true;
    h.failPatch = true;
    await assert.rejects(h.run(), /HTTP 503/);
    const failedClose = h.writes.length;
    assert.ok(JSON.parse(h.stored).pending_close);
    h.advance();
    if (scenario === 'unreadable') h.readFailure = true;
    if (scenario === 'old') h.health.collected_at = new Date(h.now - 600000).toISOString();
    if (scenario === 'fault') h.health.runtime.available = false;
    await h.run();
    assert.equal(h.issues[0].state, 'open');
    assert.ok(h.writes.slice(failedClose).every(row => row.body.state !== 'closed'));
    h.readFailure = false;
    h.health.runtime.available = true;
    h.advance();
    await h.run();
    assert.equal(h.issues[0].state, 'closed');
  }
});

test('automatic Codex recovery stays quiet and connection alerts only after the tenth attempt finishes', async () => {
  for (const [status, cause] of [['rate_limited','rate_limit'],['session_invalid','session_invalid'],
    ['failed','cli_failed'],['failed','result_missing']]) {
    const retry = fixture();
    const recoveringTask = {task_id: 'retry-task', run_id: 'retry-run', stage: 'running', codex_status: status,
      recovery: {state: 'retry_wait', failure_code: cause}, budget: {codex_attempts_used: 2, codex_attempts_limit: 10}};
    retry.health.tasks_available = true;
    retry.health.tasks = [recoveringTask]; retry.health.active_task = recoveringTask;
    for (const [used, state] of [[2, 'retry_wait'], [9, 'retry_wait'], [10, 'recovering']]) {
      recoveringTask.budget.codex_attempts_used = used; recoveringTask.recovery.state = state;
      retry.advance(); await retry.run();
      assert.equal(retry.issues.length, 0, status + ' ' + used + ' ' + state);
    }
    recoveringTask.recovery.state = 'retry_wait';
    retry.advance(); await retry.run();
    assert.ok(JSON.parse(retry.stored).codes.includes('codex_' + status));
    assert.equal(retry.issues[0].state, 'open', 'a finished final attempt still alerts');
    // Model a previously reported fault when the same task is now known to be retrying.
    recoveringTask.budget.codex_attempts_used = 2;
    retry.readFailure = true;
    retry.advance(); await retry.run();
    assert.equal(retry.issues[0].state, 'open', 'a failed read cannot clear the old alert');
    retry.readFailure = false;
    retry.advance(); await retry.run();
    assert.equal(retry.issues[0].state, 'closed', 'fresh managed recovery clears the previous false alarm');
    recoveringTask.budget = {};
    retry.advance(); await retry.run();
    assert.ok(JSON.parse(retry.stored).codes.includes('codex_' + status), 'unknown budget is not proof of recovery');
  }
  for (const status of ['auth_error', 'timeout']) {
    for (const cause of ['rate_limit', 'connection']) {
      const hardFailure = fixture();
      const task = {task_id: 'hard-task', run_id: 'hard-run', stage: 'running', codex_status: status,
        recovery: {state: 'retry_wait', failure_code: cause}, budget: {codex_attempts_used: 2, codex_attempts_limit: 10}};
      hardFailure.health.tasks = [task]; hardFailure.health.active_task = task;
      for (const used of [2, 10]) {
        task.budget.codex_attempts_used = used;
        hardFailure.advance(); await hardFailure.run();
        assert.ok(JSON.parse(hardFailure.stored).codes.includes('codex_' + status), 'current hard failure must remain visible');
      }
    }
  }
  const h = fixture();
  const task = {task_id: 'task-a', run_id: 'run-1', stage: 'running', codex_status: 'running', codex_alive: true,
    recovery: {state: 'recovering', failure_code: 'result_missing', action: 'resume'},
    budget: {codex_attempts_used: 4, codex_attempts_limit: 10,
      codex_deadline_at: new Date(h.now + 3600000).toISOString()}};
  h.health.tasks_available = true;
  h.health.tasks = [task]; h.health.active_task = task;
  await h.run();
  assert.equal(h.issues.length, 0, 'an in-budget internal recovery is not an incident');
  task.codex_status = 'connection_error';
  task.recovery.state = 'retry_wait';
  task.budget.codex_attempts_used = 2;
  h.advance();
  await h.run();
  assert.equal(h.issues.length, 0, 'an in-budget retry is not an incident');
  task.budget.codex_attempts_used = 10;
  task.recovery.state = 'recovering';
  task.codex_alive = true;
  h.advance();
  await h.run();
  assert.equal(h.issues.length, 0, 'the tenth attempt is not an incident while it is still running');
  task.recovery.state = 'retry_wait';
  task.recovery.failure_code = 'connection';
  task.codex_status = 'running';
  task.codex_alive = false;
  h.advance();
  await h.run();
  assert.match(h.issues[0].body, /Codex连接失败/);
  assert.match(h.issues[0].body, /第 10 次已经结束，10 次尝试已用尽/);
  h.advance(); await h.run();
  assert.equal(h.writes.length, 1, 'the same fault does not update the Issue again');
  h.advance();
  h.health.tasks = [{...task, task_id: 'task-b', codex_status: 'running', codex_alive: true,
    last_progress_at: h.health.collected_at, recovery: {state: 'normal'}}];
  h.health.active_task = h.health.tasks[0];
  await h.run();
  assert.equal(h.issues[0].state, 'closed');
});

test('finished infrastructure failure closes a task-only incident without claiming recovery', async () => {
  const h = fixture();
  const task = {task_id: 'task-a', run_id: 'run-1', stage: 'running', recovery: {state: 'exhausted'}};
  h.health.tasks = [task];
  await h.run();
  h.health.tasks = [];
  h.health.recent_tasks = [{...task, stage: 'published', result_status: 'infra_error'}];
  h.advance(); await h.run();
  assert.equal(h.issues[0].state, 'closed');
  assert.match(h.issues[0].body, /恢复失败.*不表示该任务恢复成功/);
});

test('missing service fields do not invent faults; oneshot idle is healthy and expected container loss is distinct', async () => {
  const h = fixture();
  h.health.services = [{name: 'health.service', type: 'oneshot', available: true, active_state: 'inactive', result: 'success'},
    {name: 'other.service', available: false, active_state: 'unknown'}];
  h.health.task_containers = [{task_id: 'a', run_id: 'r', status: 'exited', expected_running: false, running: false, available: true, oom_killed: false}];
  await h.run();
  assert.equal(h.issues.length, 0);
  h.health.task_containers[0] = {...h.health.task_containers[0], expected_running: true, status: 'missing', available: false};
  h.advance(); await h.run();
  assert.deepEqual(JSON.parse(h.stored).codes, ['container_failed']);
  h.health.task_containers[0] = {...h.health.task_containers[0], status: 'unknown', available: false};
  h.advance(); await h.run();
  assert.equal(h.issues[0].state, 'open');
  h.health.task_containers[0] = {...h.health.task_containers[0], status: 'running', available: true, running: true};
  h.advance(); await h.run();
  assert.equal(h.issues[0].state, 'closed');
});

test('lost close acknowledgement followed by new fault retries reopening after an API failure', async () => {
  const h = fixture();
  h.health.runtime.available = false;
  await h.run();
  h.advance(); h.health.runtime.available = true; h.losePatchResponse = true;
  await assert.rejects(h.run(), /Patch response lost/);
  assert.equal(h.issues[0].state, 'closed', 'remote close succeeded but local acknowledgement was lost');
  h.advance(); h.health.runtime.available = false; h.failPatch = true;
  await assert.rejects(h.run(), /HTTP 503/);
  assert.equal(JSON.parse(h.stored).needs_reopen, true);
  h.advance(); await h.run();
  assert.equal(h.issues[0].state, 'open');
  assert.equal(h.issues.length, 1, 'still the same continuous incident');
  assert.equal(JSON.parse(h.stored).needs_reopen, false);
});


test('unreadable Journal is unknown, not an empty recovered task or upload queue', async () => {
  const h = fixture();
  h.health.tasks_available = h.health.uploads_available = true;
  const task = {task_id: 'task-a', run_id: 'run-1', stage: 'running', recovery: {state: 'retry_wait'}};
  h.health.tasks = [task];
  h.health.uploads = [{task_id: 'upload-a', run_id: 'run-2', queued_at: new Date(h.now - 1800000).toISOString()}];
  await h.run();
  h.health.tasks_available = h.health.uploads_available = false;
  h.health.tasks = []; h.health.uploads = [];
  h.advance(); await h.run();
  const codes = JSON.parse(h.stored).codes;
  assert.ok(codes.includes('task_recovering'));
  assert.ok(codes.includes('delivery_pending'));
  assert.ok(codes.includes('task_state_unavailable'));
  assert.equal(h.issues[0].state, 'open');
  h.health.tasks_available = h.health.uploads_available = true;
  h.health.recent_tasks = [{...task, stage: 'published', result_status: 'pass', recovery: {state: 'recovered'}}];
  h.advance(); await h.run();
  assert.equal(h.issues[0].state, 'closed');
});
