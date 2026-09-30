const test = require('node:test');
const assert = require('node:assert/strict');
const { normalize, business, blockerGroups, environmentProfile } = require('../../../dashboard/data.js');
const { assess: assessHealth, readSnapshot, readAlerts, readHealth, monitorReading, source: healthSource,
  taskFacts } = require('../../../dashboard/health.js');
const task = (id, date, extra = {}) => ({task_id:id, repository:'RACE-org/triton-anchor',event_kind:'push',pr_number:0,
  target_branch:'triton_v3.0',head_sha:('f'+id).repeat(20),tested_sha:id.repeat(40),captured_at:date,full:false,...extra});
const backendEnvironment = (backend = 'sophgo-cmodel', profile = 'triton-3.0') => ({variants:{candidate:{
  backend_enabled:true, backend_profile:backend, profile, llvm_hash:'e'.repeat(40), environment_fingerprint:'env-1',
}}});
const performanceDetails = (id, tool, summary, extra = {}) => ({candidate:{metadata:{
  commit_sha:id.repeat(40), environment_fingerprint:'env-1', profile_id:'triton-3.0',
  llvm_revision:'e'.repeat(40), kernels:['add','mm','softmax','layernorm'],
  repeat:tool === 'ir_serialization' ? 20 : 3, warmup:1,
},summary},...extra});
const standardCompileSummary = value => Object.fromEntries(
  ['add','mm','softmax','layernorm'].map((kernel,index)=>[kernel,{
    all_correct:true,compile_est:{median_ms:value+index},
  }]),
);
const standardPassSummary = value => Object.fromEntries(
  ['add','mm','softmax','layernorm'].map((kernel,index)=>[kernel,{passes:{
    ['pass-'+index]:{wall_ms:{median_ms:value+index}},
  }}]),
);
const standardIrSummary = value => Object.fromEntries(
  ['add','mm','softmax','layernorm'].map((kernel,kernelIndex)=>[kernel,{metrics:Object.fromEntries(
    ['serialize','write_text','read_text','deserialize','roundtrip'].map((metric,metricIndex)=>[
      metric,{median_ms:value+kernelIndex+metricIndex},
    ]),
  )}]),
);

const healthNow = Date.parse('2026-09-17T10:00:00Z');
function healthyWorker() {
  return {schema:'triton-anchor-worker-health',worker_id:healthSource.worker,collected_at:new Date(healthNow).toISOString(),
    poller:{alive:true,heartbeat_stale:false,heartbeat_at:new Date(healthNow-30000).toISOString(),last_poll_status:'success'},
    runtime:{available:true},services:[],storage:[],active_task:null};
}

test('health freshness does not turn a normal five-minute sampling gap into a dead Worker', () => {
  const worker=healthyWorker();
  assert.equal(assessHealth(worker,{now:healthNow+240000}).issues.length,0);
  const stale=assessHealth(worker,{now:healthNow+1201000});
  assert.equal(stale.current,false);
  assert.ok(stale.issues.some(row=>row.code==='snapshot_stale'));
  const unreadable=assessHealth(worker,{now:healthNow,workerError:'浏览器请求失败'});
  assert.equal(unreadable.current,false);
  assert.deepEqual(unreadable.issues.map(row=>row.code),['health_read']);
});

test('health separates service, relay and Codex evidence without treating idle oneshots as failures', () => {
  const worker=healthyWorker();
  worker.services=[{name:'triton-anchor-local-ci-control-update.service',type:'oneshot',available:true,active_state:'inactive',result:'success'}];
  worker.active_task={stage:'running'};
  let result=assessHealth(worker,{now:healthNow});
  assert.equal(result.issues.length,0);
  worker.services.push({name:'triton-anchor-local-ci-health.timer',available:false,active_state:'unknown'});
  result=assessHealth(worker,{now:healthNow});
  assert.equal(result.issues.length,0, 'unknown service telemetry must not create a failure');
  assert.equal(Object.fromEntries(taskFacts(worker.active_task))['恢复状态'],'未上报');
  worker.services.pop();
  worker.services.push({name:'triton-anchor-local-ci-health.timer',available:true,active_state:'failed'});
  worker.poller.last_poll_status='error';
  for (const status of ['connection_error','auth_error','session_invalid','rate_limited','timeout','failed']) {
    worker.active_task={stage:'running',codex_status:status,...(status==='connection_error' ? {
      recovery:{state:'retry_wait',failure_code:'connection_error'},
      budget:{codex_attempts_used:10,codex_attempts_limit:10},
    } : {})};
    result=assessHealth(worker,{now:healthNow});
    assert.ok(result.issues.some(row=>row.code==='service_triton-anchor-local-ci-health.timer'));
    assert.ok(result.issues.some(row=>row.code==='relay_poll_failed'));
    assert.ok(result.issues.some(row=>row.code==='codex_'+status));
  }
});

test('health distinguishes update execution failure from invalid or blocked requests', () => {
  const worker=healthyWorker();
  const descriptions=new Set();
  for (const state of ['failed','invalid','blocked']) {
    worker.control_update={state};
    const issue=assessHealth(worker,{now:healthNow}).issues.find(row=>row.code==='control_update');
    descriptions.add(issue.text);
    assert.equal(issue.tone,'bad');
  }
  assert.equal(descriptions.size,3);
  for (const state of ['idle','pending','updating']) {
    worker.control_update={state};
    assert.equal(assessHealth(worker,{now:healthNow}).issues.length,0);
  }
});

test('recovery facts separate phase, budget and progress; long silence only warns', () => {
  const worker=healthyWorker();
  const current={task_id:'task',run_id:'run',stage:'running',last_progress_at:new Date(healthNow-3700000).toISOString(),
    codex_status:'connection_error',recovery:{state:'retry_wait',action:'resume',failure_code:'result_missing',next_retry_at:worker.collected_at},
    budget:{codex_attempts_used:3,codex_attempts_limit:10,execution_attempts_used:1,execution_attempts_limit:3,
      session_switches:0,session_switches_limit:1}};
  worker.active_task=current;worker.tasks=[current];
  const model=assessHealth(worker,{now:healthNow}),facts=Object.fromEntries(taskFacts(current));
  assert.equal(facts['执行阶段'],'执行中');assert.equal(facts['恢复状态'],'等待重试');
  assert.equal(facts['Codex 尝试'],'3 / 10');assert.equal(facts['恢复动作'],'复用原 session');
  assert.deepEqual(model.issues.map(row=>row.code),['task_no_progress','task_stalled']);
  assert.equal(model.cards[3].text,'自动重连中 · 3 / 10');
  assert.ok(model.issues.every(row=>row.tone==='warn'));
  assert.equal(Object.fromEntries(taskFacts(current))['异常原因'],'最终执行报告缺失或不完整');
  const budget=current.budget;current.budget={};
  const unknownBudget=assessHealth(worker,{now:healthNow});
  assert.ok(!unknownBudget.issues.some(row=>row.code==='codex_connection_error'));
  assert.equal(unknownBudget.cards[3].text,'自动重连中 · 尝试次数未上报');
  current.budget=budget;
  current.codex_status='running';current.budget.codex_attempts_used=4;current.recovery.state='recovering';
  const internalRecovery=assessHealth(worker,{now:healthNow});
  assert.ok(!internalRecovery.issues.some(row=>row.code==='task_recovering'));
  assert.equal(internalRecovery.cards[3].text,'自动恢复中 · 4 / 10');
  for (const [status, cause] of [['rate_limited','rate_limit'],['session_invalid','session_invalid'],
    ['failed','cli_failed'],['failed','result_missing']]) {
    current.codex_status=status;current.recovery.failure_code=cause;
    for (const [used,state] of [[2,'retry_wait'],[9,'retry_wait'],[10,'recovering']]) {
      current.budget.codex_attempts_used=used;current.recovery.state=state;
      const retry=assessHealth(worker,{now:healthNow});
      assert.deepEqual(retry.issues.map(row=>row.code),['task_no_progress','task_stalled'], status+' '+used+' '+state);
      assert.equal(retry.cards[3].text,'自动恢复中 · '+used+' / 10');
    }
    current.recovery.state='retry_wait';
    assert.ok(assessHealth(worker,{now:healthNow}).issues.some(row=>row.code==='codex_'+status));
    current.budget={};
    assert.ok(assessHealth(worker,{now:healthNow}).issues.some(row=>row.code==='codex_'+status), 'unknown budget is not proof of recovery');
    current.budget=budget;
  }
  for (const status of ['auth_error','timeout']) {
    current.codex_status=status;
    for (const cause of ['rate_limit','connection']) {
      current.recovery={state:'retry_wait',failure_code:cause};
      for (const used of [2,10]) {
        current.budget.codex_attempts_used=used;
        const hardFailure=assessHealth(worker,{now:healthNow});
        assert.ok(hardFailure.issues.some(row=>row.code==='codex_'+status));
        assert.equal(hardFailure.cards[3].tone,'bad');
      }
    }
  }
  current.codex_status='connection_error';
  current.budget.codex_attempts_used=10;current.recovery.state='recovering';
  assert.ok(!assessHealth(worker,{now:healthNow}).issues.some(row=>row.code==='codex_connection_error'));
  current.recovery.state='retry_wait';
  assert.ok(assessHealth(worker,{now:healthNow}).issues.some(row=>row.code==='codex_connection_error'));
  current.codex_status='failed';
  current.recovery={state:'exhausted',failure_code:'recovery_exhausted',action:'publish_infra_error'};
  assert.ok(assessHealth(worker,{now:healthNow}).issues.some(row=>row.code==='task_recovery_exhausted'));
});

test('health reader anonymously decodes the Gitee file API and rejects a different worker', async t => {
  const worker={...healthyWorker(),note:'中文'};
  t.mock.method(global,'fetch',async (url,options)=>{
    assert.match(url,/\/api\/v5\/repos\/.+\/contents\/worker-health.json\?ref=snapshot%2Frace-org-localci-v2$/);
    assert.equal(options.credentials,'omit');
    return {ok:true,json:async()=>({encoding:'base64',content:Buffer.from(JSON.stringify(worker)).toString('base64')})};
  });
  assert.deepEqual(await readSnapshot('worker-health.json','snapshot/'+healthSource.worker,worker.schema),worker);
  worker.worker_id='another-server';
  await assert.rejects(readSnapshot('worker-health.json','snapshot/'+healthSource.worker,worker.schema),/身份不匹配/);
});

test('health alerts select this worker only and link to the health repository without credentials', async t => {
  const marker = '<!-- local-ci-alert:' + healthSource.worker + ' -->';
  t.mock.method(global, 'fetch', async (url, options) => {
    assert.match(url, /\/issues\?state=all&sort=updated/);
    assert.equal(options.credentials, 'omit');
    assert.equal(options.headers, undefined);
    return {ok:true, json:async()=>[
      {number:'IABC01',title:'连接异常',body:marker,state:'open',html_url:'javascript:alert(1)'},
      {number:'IABC02',body:'<!-- local-ci-alert:another-worker -->',state:'open'},
      {number:'../settings',body:marker,state:'open'},
      {number:'IABC03',body:marker,state:'closed'},
    ]};
  });
  const alerts = await readAlerts();
  assert.deepEqual(alerts.map(row=>row.state), ['open','closed']);
  assert.equal(alerts[0].url, 'https://gitee.com/' + healthSource.repository + '/issues/IABC01');
});

test('health uses Gitee first, falls back once, and observes rate-limit cooldown', async t => {
  let now=healthNow, limited=false;
  const calls=[], worker=healthyWorker();
  const cache={schema:'triton-anchor-worker-health-cache',worker_id:healthSource.worker,
    updated_at:new Date(now).toISOString(),worker,alerts:[],errors:{}};
  t.mock.method(Date,'now',()=>now);
  t.mock.method(global,'fetch',async (url,options)=>{
    calls.push(url); assert.equal(options.credentials,'omit');
    if(url===healthSource.cacheUrl) return {ok:true,json:async()=>cache};
    if(limited) return {ok:false,status:403,text:async()=>'403 Forbidden (Rate Limit Exceeded)'};
    assert.ok(!url.includes('watchdog'), 'watchdog has been retired');
    return {ok:true,json:async()=>url.includes('/issues?') ? [] : worker};
  });
  let result=await readHealth();
  assert.equal(result.notice,'');
  assert.equal(calls.length,3);assert.ok(calls.includes(healthSource.cacheUrl),'read-only cache supplies monitor status');
  assert.equal(result.results[0].value,worker,'Gitee remains primary');
  assert.equal(result.monitor.updated_at,cache.updated_at);
  cache.errors.worker='Cloudflare 未能读取 Gitee 健康快照';
  cache.health_read={status:'error',error_code:'http_error',http_status:403,duration_ms:240,consecutive_failures:13};
  result=await readHealth();
  assert.match(monitorReading(result.monitor).text,/HTTP 403/);
  assert.equal(result.pageRead,'读取成功'); assert.equal(result.dataSource,'Gitee 直读');
  assert.equal(assessHealth(result.results[0].value,{now,workerError:result.results[0].error}).current,true);
  cache.errors.worker=''; delete cache.health_read;
  limited=true; calls.length=0;
  result=await readHealth();
  assert.equal(calls.length,3);assert.equal(calls.at(-1),healthSource.cacheUrl);
  assert.equal(result.retryAt,now+900000); assert.match(result.notice,/Cloudflare/);
  assert.equal(result.pageRead,'访问被限流'); assert.equal(result.dataSource,'Cloudflare 备用缓存');
  assert.equal(result.results[0].value.collected_at,worker.collected_at);
  calls.length=0; now+=300000;
  result=await readHealth(result.retryAt);
  assert.deepEqual(calls,[healthSource.cacheUrl]);
  limited=false; calls.length=0; now+=600001;
  result=await readHealth(result.retryAt);
  assert.equal(calls.length,3); assert.equal(result.notice,'');
});

test('monitor reading distinguishes failed reads, stale snapshots and unknown monitor state', () => {
  const monitor={updated_at:new Date(healthNow).toISOString(),source_at:new Date(healthNow-3600000).toISOString(),
    readError:'',read:{status:'ok'}};
  assert.match(monitorReading(monitor,healthNow).text,/读取成功.*快照已过期/);
  monitor.source_at=new Date(healthNow).toISOString();
  assert.equal(monitorReading(monitor,healthNow).tone,'good');
  monitor.read={status:'error',error_code:'rate_limited',http_status:403};
  assert.match(monitorReading(monitor,healthNow).text,/限流.*HTTP 403/);
  monitor.error='网页无法读取 Cloudflare 缓存';
  assert.match(monitorReading(monitor,healthNow).text,/状态未知/);
  delete monitor.error; monitor.updated_at=new Date(healthNow-1201000).toISOString();
  assert.match(monitorReading(monitor,healthNow).text,/缓存已过期.*状态未知/);
  monitor.updated_at=new Date(healthNow).toISOString(); delete monitor.read;
  monitor.readError='Cloudflare 未能读取 Gitee 健康快照';
  assert.match(monitorReading(monitor,healthNow).text,/错误详情未上报/);
  delete monitor.readError;
  assert.equal(monitorReading(monitor,healthNow).text,'读取结果未上报');
});

test('cache only fills failed reads and never presents stale or failed collection as current', async t => {
  const worker=healthyWorker();
  const cache={schema:'triton-anchor-worker-health-cache',worker_id:healthSource.worker,
    updated_at:new Date(healthNow-1201000).toISOString(),worker:{...worker,collected_at:'2026-09-17T09:00:00Z'},
    alerts:[],errors:{alerts:'Cloudflare 未能更新告警记录'}};
  t.mock.method(Date,'now',()=>healthNow);
  t.mock.method(global,'fetch',async url=>{
    if(url===healthSource.cacheUrl) return {ok:true,json:async()=>cache};
    if(url.includes('worker-health.json')) return {ok:true,json:async()=>worker};
    throw new TypeError('Network unavailable');
  });
  let result=await readHealth();
  assert.equal(result.results[0].value,worker);
  assert.match(result.monitor.error,/缓存已过期/);
  assert.match(result.results[1].error,/未能更新告警/);
  result=await readHealth(healthNow+900000);
  const model=assessHealth(result.results[0].value,
    {now:healthNow,workerError:result.results[0].error});
  assert.equal(model.current,false);
  cache.updated_at=new Date(healthNow).toISOString();
  const newer={...worker,poller:{...worker.poller,alive:false}};
  result=await readHealth(healthNow+900000,{worker:newer});
  assert.equal(result.results[0].value,newer);
  assert.match(result.results[0].error,/保留较新数据/);
  const latest={...newer,collected_at:new Date(healthNow+1000).toISOString()};
  result=await readHealth(0,{worker:latest});
  assert.equal(result.results[0].value,latest,'even Gitee cannot replace a newer browser snapshot');
});

test('failure of both health sources leaves existing data untouched and is not a server fault', async t => {
  const calls=[];
  t.mock.method(global,'fetch',async url=>{calls.push(url);throw new TypeError('offline');});
  const result=await readHealth();
  assert.equal(calls.length,3);
  assert.ok(result.results.every(row=>row.status==='rejected' && row.value===undefined));
  assert.match(result.notice,/保留最后读取的数据/);
  const model=assessHealth(healthyWorker(),
    {now:healthNow,workerError:result.notice});
  assert.equal(model.current,false);
  assert.ok(!model.issues.some(row=>row.code==='poller_unavailable'));
});

test('missing telemetry is unknown and broken task collection cannot look like an empty healthy queue', () => {
  const worker=healthyWorker();
  delete worker.runtime;delete worker.poller.alive;
  let model=assessHealth(worker,{now:healthNow});
  assert.deepEqual(model.issues,[]);
  assert.equal(model.cards[0].text,'未上报');assert.equal(model.cards[1].text,'未上报');
  worker.tasks_available=worker.uploads_available=false;
  model=assessHealth(worker,{now:healthNow});
  assert.ok(model.issues.some(row=>row.code==='task_state_unavailable' && row.tone==='warn'));
});

test('superseded tasks preserve execution results without becoming current or passing unexecuted tasks', () => {
  const data = normalize({schema:'triton-anchor-dashboard',tasks:[
    {task:task('d','2026-09-12'),status:'pending'},
    {task:task('c','2026-09-11'),status:'superseded',historical:true,result:null},
    {task:task('b','2026-09-10'),status:'superseded',historical:true,result:{status:'pass'}},
    {task:task('a','2026-09-09'),status:'cancelled',historical:true,result:{status:'pass'}},
  ]});
  assert.deepEqual(data.runs.map(run=>run.is_current),[true,false,false,false]);
  assert.deepEqual(data.runs.map(run=>run.conclusion),['waiting','superseded','success','cancelled']);
  assert.deepEqual(data.runs.map(run=>run.superseded),[false,true,true,false]);
  assert.equal(data.runs[3].local_conclusion,'passed');
});

test('omitted files and unsafe URLs are not clickable', () => {
  const artifacts = [{path:'missing',omitted:'Too large'},{path:'safe'},{path:'unsafe'}];
  const data = normalize({schema:'triton-anchor-dashboard',tasks:[{task:task('a','2026-09-10'),
    result:{artifacts},artifact_urls:{missing:'https://gitee.com/missing',safe:'https://gitee.com/report',unsafe:'javascript:alert(1)'}}]});
  assert.deepEqual(data.runs[0].artifacts.map(a=>a.url),['','https://gitee.com/report','']);
});

test('evidence delivery warnings do not change a passing execution result', () => {
  const run = normalize({schema:'triton-anchor-dashboard',tasks:[{task:task('a','2026-09-10'),status:'pass',result:{status:'pass',evidence_delivery:{status:'incomplete',omitted:[{path:'report.txt'}]}}}]}).runs[0];
  assert.equal(run.conclusion,'success');
  assert.equal(run.local_conclusion,'passed');
  assert.equal(run.evidence_delivery.status,'incomplete');
});

test('dashboard preserves not-selected, skipped, and not-applicable as distinct states', () => {
  const run = normalize({schema:'triton-anchor-dashboard',tasks:[{task:task('a','2026-09-10'),result:{checks:[
    {tool_id:'frontend_build',status:'not_selected'},
    {tool_id:'frontend_tests',status:'skipped'},
    {tool_id:'backend_build',status:'not_applicable'},
  ]}}]}).runs[0];
  assert.deepEqual(run.checks.map(check => check.status), ['not_selected','skipped','not_applicable']);
});

test('full view excludes impact-only selection and preserves failing operator details', () => {
  const run = {task:task('a','2026-09-10',{full:true}),business_full:{data_mode:'live'},result:{environment:{profile:'triton-3.0',backend_enabled:true,backend_profile:'fixture-backend'},checks:[{tool_id:'flaggems',status:'fail',details:{"flaggems-summary":{mode:'full',results:[{op:'softmax',test_status:'失败',first_failed_stage:'准确率验证',duration_seconds:2}]}}}]}};
  let data = business(normalize({schema:'triton-anchor-dashboard',tasks:[run]}));
  assert.equal(data.fullTest.operators[0].status,'failed');
  assert.equal(data.fullTest.operators[0].failure_stage,'准确率验证');
  assert.equal(data.fullTest.run.backend,'fixture-backend');
  assert.equal(data.fullTest.run.profile,'triton-3.0');
  run.result.checks[0].details["flaggems-summary"].mode='impact';
  data = business(normalize({schema:'triton-anchor-dashboard',tasks:[run]}));
  assert.equal(data.fullTest.operators.length,0);
});

test('full view prefers a valid release result and otherwise labels the preserved demo as historical sample', () => {
  const demo = {task:task('d','2026-09-09',{target_branch:'CI_dev',tested_sha:'3d4c586307dcc3c1f11e650c67529b85da3dd22f'}),
    business_full:{data_mode:'mock',source_path:'runs/ci_full_flaggems/3d4c586307dcc3c1f11e650c67529b85da3dd22f',source_note:'历史样例结果'},
    result:{environment:backendEnvironment('sophgo-cmodel',''),checks:[{tool_id:'flaggems',status:'pass',details:{'flaggems-summary':{
      mode:'full',results:[{op:'demo-add',test_status:'通过',duration_seconds:1}],
    }}}]}};
  const invalidLive = {task:task('x','2026-09-11',{full:true}),business_full:{data_mode:'live'},
    result:{environment:backendEnvironment(),checks:[{tool_id:'flaggems',status:'pass',details:{'flaggems-summary':{mode:'impact',results:[{op:'impact-only'}]}}}]}};
  let normalized = normalize({schema:'triton-anchor-dashboard',tasks:[invalidLive,demo]});
  assert.equal(normalized.runs.find(run=>run.tested_sha===demo.task.tested_sha).business_full.source_note,'历史样例结果');
  let data = business(normalized);
  assert.equal(data.fullTest.data_mode,'mock');
  assert.equal(data.fullTest.source_note,'历史样例结果');
  assert.equal(data.fullTest.run.sha,demo.task.tested_sha);
  assert.equal(data.fullTest.run.profile,'');
  assert.equal(data.fullTest.operators[0].name,'demo-add');

  const live = {task:task('l','2026-09-12',{full:true,tested_sha:'3d4c586307dcc3c1f11e650c67529b85da3dd22f'}),business_full:{data_mode:'live',source_note:'新全量实测'},
    result:{environment:backendEnvironment(),checks:[{tool_id:'flaggems',status:'pass',details:{'flaggems-summary':{
      mode:'full',results:[{op:'live-add',test_status:'通过',duration_seconds:1}],
    }}}]}};
  const foreign = {task:task('z','2026-09-13',{full:true,repository:'another/triton-anchor'}),
    business_full:{data_mode:'live'},result:{environment:backendEnvironment(),checks:[
      {tool_id:'flaggems',status:'pass',details:{'flaggems-summary':{
        mode:'full',results:[{op:'foreign-add',test_status:'通过',duration_seconds:1}],
      }}},
    ]}};
  data = business(normalize({schema:'triton-anchor-dashboard',tasks:[demo,live,foreign]}));
  assert.equal(data.fullTest.data_mode,'live');
  assert.equal(data.fullTest.source_note,'新全量实测');
  assert.equal(data.fullTest.run.sha,'3d4c586307dcc3c1f11e650c67529b85da3dd22f');
  assert.equal(data.fullTest.operators[0].name,'live-add');
});

test('variant environments preserve both sources and use candidate for business results', () => {
  const environment = {variants:{base:{profile:'triton-3.0',backend_enabled:true,backend_profile:'sophgo-cmodel'},
    candidate:{profile:'triton-3.2',backend_enabled:false,backend_profile:'sophgo-cmodel'}}};
  const feed = {schema:'triton-anchor-dashboard',tasks:[{task:task('a','2026-09-10',{full:true}),business_full:{data_mode:'live'},result:{environment,checks:[
    {tool_id:'backend_tests',status:'fail'},
    {tool_id:'compile_time',status:'pass',details:{candidate:{summary:{add:{compile_est:{median_ms:7}}}}}},
  ]}}]};
  const normalized = normalize(feed);
  assert.deepEqual(normalized.runs[0].environment,environment);
  assert.equal(environmentProfile(environment,'base'),'triton-3.0');
  let data = business(normalized);
  assert.equal(data.backends.backends.length,0);
  assert.equal(data.performance.backend,'尚无有效测量');
  assert.equal(data.performance.compile_time.kernels.length,0);
  environment.variants.candidate = {profile:'triton-3.0',backend_enabled:true,backend_profile:'sophgo-cmodel'};
  feed.tasks[0].result.checks = [{tool_id:'backend_tests',status:'pass'},
    {tool_id:'flaggems',status:'pass',details:{'flaggems-summary':{mode:'full',results:[{op:'add',exit_code:0,passed:1}]}}}];
  data = business(normalize(feed));
  assert.equal(data.backends.backends[0].profile,'triton-3.0');
  assert.equal(data.backends.backends[0].name,'sophgo-cmodel');
  assert.equal(data.fullTest.run.backend,'sophgo-cmodel');
  delete environment.variants.candidate.backend_profile;
  assert.equal(business(normalize(feed)).backends.backends.length,0);
  delete environment.variants.candidate;
  environment.profile = 'legacy';
  environment.backend_enabled = true;
  environment.backend_profile = 'sophgo-cmodel';
  assert.equal(environmentProfile(environment),'');
  data = business(normalize(feed));
  assert.equal(data.fullTest.run.backend,'尚无全量算子结果');
  assert.equal(data.backends.backends.length,0);
});

test('backend summary uses smoke results independently of the overall verdict and backend unit tests', () => {
  for (const [smoke, expected] of [['pass','passed'],['fail','failed'],['infra_error','error'],
    ['not_selected','not_selected'],['not_applicable','not_applicable'],['cancelled','cancelled'],[null,'not_recorded']]) {
    const checks = [{tool_id:'backend_tests',status:'pass'}];
    if (smoke) checks.push({tool_id:'backend_smoke',status:smoke});
    const feed = {schema:'triton-anchor-dashboard',tasks:[{
      task:task('a','2026-09-10'),
      result:{status:'fail',environment:backendEnvironment(),checks},
    }]};
    const row = business(normalize(feed)).backends.backends[0];
    assert.equal(row.tests.backend,expected);
    assert.equal(row.state,'failure');
  }
});

test('empty initial feed does not create placeholder success data', () => {
  const data = business(normalize({schema:'triton-anchor-dashboard',tasks:[]}));
  assert.equal(data.fullTest.operators.length,0);
  assert.equal(data.backends.backends.length,0);
  assert.equal(data.performance.compile_time.kernels.length,0);
});

test('historical operator and per-metric results survive pending or partially selected tasks', () => {
  const feed = {schema:'triton-anchor-dashboard',tasks:[
    {task:task('c','2026-09-12'),status:'pending'},
    {task:task('b','2026-09-11'),historical:true,result:{completed_at:'2026-09-11',environment:backendEnvironment(),checks:[
      {tool_id:'backend_tests',status:'pass'},
      {tool_id:'compile_time',status:'pass',details:performanceDetails('b','compile_time',standardCompileSummary(7))},
      {tool_id:'pass_profile',status:'pass',details:performanceDetails('b','pass_profile',standardPassSummary(2))},
      {tool_id:'ir_serialization',status:'pass',details:performanceDetails('b','ir_serialization',standardIrSummary(1))},
      {tool_id:'flaggems',status:'not_selected'},
    ]}},
    {task:task('a','2026-09-10',{full:true}),historical:true,business_full:{data_mode:'live'},result:{completed_at:'2026-09-10',environment:backendEnvironment(),checks:[
      {tool_id:'backend_tests',status:'pass'},
      {tool_id:'flaggems',status:'fail',details:{'flaggems-summary':{mode:'full',results:[{op:'add',test_status:'失败'}]}}},
    ]}},
  ]};
  const normalized=normalize(feed), data=business(normalized);
  assert.deepEqual(normalized.runs.map(run=>run.is_current),[true,false,false]);
  assert.equal(data.fullTest.operators[0].name,'add');
  assert.equal(data.fullTest.run.sha,'a'.repeat(40));
  assert.equal(data.backends.backends.length,1);
  assert.equal(data.backends.backends[0].name,'sophgo-cmodel');
  assert.equal(data.backends.backends[0].profile,'triton-3.0');
  assert.equal(data.backends.backends[0].sha,'b'.repeat(40));
  assert.equal(data.performance.compile_time.kernels[0].candidate_ms,7);
  assert.equal(data.performance.compile_time.sha,'b'.repeat(40));
  assert.equal(data.performance.pass_profile.hotspots[0].median_ms,5);
  assert.equal(data.performance.pass_profile.sha,'b'.repeat(40));
  assert.equal(data.performance.compile_time.backend,'sophgo-cmodel');
  assert.equal(data.performance.compile_time.profile,'triton-3.0');
  assert.equal(data.performance.pass_profile.backend,'sophgo-cmodel');
  assert.equal(data.performance.pass_profile.profile,'triton-3.0');
});


test('performance views read runner candidate and comparison report keys', () => {
  const kernels=['add','mm','softmax','layernorm'];
  const metrics=['serialize','write_text','read_text','deserialize','roundtrip'];
  const compileSummary=Object.fromEntries(kernels.map((kernel,index)=>[kernel,{all_correct:true,compile_est:{median_ms:12+index}}]));
  const passSummary=Object.fromEntries(kernels.map((kernel,index)=>{
    const hotspots=[
      {name:'Total',median_ms:100-index},{name:'Rest',median_ms:90-index},{name:'(A) Analysis',median_ms:80-index},
      {name:'pass-'+index+'-a',median_ms:20-index},{name:'pass-'+index+'-b',median_ms:10-index},{name:'pass-'+index+'-c',median_ms:5-index},
    ];
    return [kernel,{hotspots,passes:{
      ...Object.fromEntries(hotspots.map(row=>[
        row.name,{wall_ms:{median_ms:row.median_ms}},
      ])),
      ['complete-only-'+index]:{wall_ms:{median_ms:200-index}},
    }}];
  }));
  const irSummary=Object.fromEntries(kernels.map((kernel,kernelIndex)=>[kernel,{metrics:Object.fromEntries(
    metrics.map((metric,metricIndex)=>[metric,{median_ms:1+kernelIndex*2+metricIndex}]))}]));
  const irComparisons=kernels.flatMap((kernel,kernelIndex)=>metrics.map((metric,metricIndex)=>({kernel,metric,
    baseline_median_ms:(1+kernelIndex*2+metricIndex)/2,candidate_median_ms:1+kernelIndex*2+metricIndex,
    change_percent:100,status:'warning',exceeds_threshold:true})));
  const data = business(normalize({schema:'triton-anchor-dashboard',tasks:[{
    task:task('a','2026-09-10'),status:'pass',result:{environment:backendEnvironment(),checks:[
      {tool_id:'compile_time',status:'pass',details:performanceDetails('a','compile_time',compileSummary,{comparison:{kernels:kernels.map((kernel,index)=>({
        kernel,baseline_median_ms:10+index,candidate_median_ms:12+index,change_percent:index ? 10 : 25,exceeds_threshold:index===0,
      }))}})},
      {tool_id:'pass_profile',status:'pass',details:performanceDetails('a','pass_profile',passSummary,{comparison:{passes:[{
        kernel:'add',pass:'pass-0-a',baseline_median_ms:10,candidate_median_ms:20,change_percent:100,status:'warning',exceeds_threshold:true,
      }]}})},
      {tool_id:'ir_serialization',status:'pass',details:performanceDetails('a','ir_serialization',irSummary,{comparison:{rows:irComparisons}})}
    ]}
  }]}));
  assert.deepEqual(data.performance.compile_time.kernels.map(row=>row.name),kernels);
  assert.equal(data.performance.compile_time.kernels[0].candidate_ms,12);
  assert.equal(data.performance.compile_time.kernels[0].baseline_ms,10);
  assert.equal(data.performance.compile_time.kernels[0].delta_percent,25);
  assert.equal(data.performance.compile_time.kernels[0].status,'warning');
  assert.equal(data.performance.pass_profile.hotspots.length,10);
  assert.equal(data.performance.pass_profile.hotspots[0].name,'add / complete-only-0');
  const comparedPass=data.performance.pass_profile.hotspots.find(row=>row.name==='add / pass-0-a');
  assert.equal(comparedPass.delta_percent,100);
  assert.equal(comparedPass.status,'warning');
  assert.ok(data.performance.pass_profile.hotspots.every(row=>!/(Total|Rest|\(A\))/.test(row.name)));
  assert.deepEqual(data.performance.ir_serialization.metrics.map(row=>row.name),metrics);
  assert.equal(data.performance.ir_serialization.metrics[0].median_ms,4);
  assert.equal(data.performance.ir_serialization.metrics[0].baseline_ms,2);
  assert.equal(data.performance.ir_serialization.metrics[0].delta_percent,100);
  assert.equal(data.performance.ir_serialization.metrics[0].status,'warning');
});


const errorRun = result => normalize({schema:'triton-anchor-dashboard', tasks:[{
  task:task('a','2026-09-10'), status:result.status || 'infra_error', result,
}]}).runs[0];

test('execution errors and failed checks stay separate through task filters and business views', () => {
  const vm=require('node:vm'), fs=require('node:fs');
  const runs=normalize({schema:'triton-anchor-dashboard',tasks:['fail','infra_error'].map((status,index)=>({
    task:task(String(index),'2026-09-10',{full:true}),status,
    business_full:{data_mode:'live'},
    result:{status,environment:backendEnvironment('backend-'+index),checks:[
      {tool_id:'backend_tests',status},
      {tool_id:'flaggems',status,details:{'flaggems-summary':{mode:'full',results:[
        {op:'a',test_status:'失败'}, {op:'b',test_status:'infra_error'},
      ]}}},
    ]},
  }))}).runs;
  assert.deepEqual(runs.map(run=>run.conclusion),['failure','error']);
  const projected=business({runs});
  assert.deepEqual(projected.backends.backends.map(row=>row.state),['failure','error']);
  assert.deepEqual(projected.fullTest.operators.map(row=>row.status),['failed','error']);
  runs.forEach(run=>{ run.is_current=true; });

  const nodes=new Map();
  const document={createElement:()=>({}),querySelectorAll:()=>[],getElementById:id=>{
    if(!nodes.has(id))nodes.set(id,{value:'',addEventListener(){}});
    return nodes.get(id);
  }};
  const context=vm.createContext({document,URL,URLSearchParams,location:{search:''},
    window:{location:{search:''}},fetch:()=>new Promise(()=>{}),setInterval(){},runs});
  vm.runInContext(fs.readFileSync(require.resolve('../../../dashboard/local-ci.js'),'utf8'),context);
  vm.runInContext('model.data={runs};',context);
  assert.equal(vm.runInContext('displaySha(runs[0])',context),runs[0].tested_sha);
  nodes.get('historyFilter').value='current';
  for(const filter of ['failure','error']){
    nodes.get('resultFilter').value=filter;
    assert.equal(vm.runInContext('filteredRuns().length',context),1);
    assert.equal(vm.runInContext('filteredRuns()[0].conclusion',context),filter);
  }

  const app=vm.createContext({document,URLSearchParams,location:{search:''},LocalCIData:{normalize,business},window:{location:{search:''}},
    fetch:()=>new Promise(()=>{})});
  vm.runInContext(fs.readFileSync(require.resolve('../../../dashboard/app.js'),'utf8'),app);
  assert.match(vm.runInContext("statusBadge('not_recorded')",app),/未记录/);
  vm.runInContext("state.fullTest={operators:[{name:'fail',status:'failed'},{name:'error',status:'infra_error'}]};",app);
  const stats=vm.runInContext('computeOperatorSummary(state.fullTest.operators)',app);
  assert.equal(stats.failed,1);
  assert.equal(stats.error,1);
  assert.equal(stats.exceptions,2);
  for(const filter of ['failed','error']){
    vm.runInContext(`state.status='${filter}';`,app);
    assert.equal(vm.runInContext('filteredOperators().length',app),1);
  }
});

test('server failure in summary is separated from reviews that never completed', () => {
  const run = errorRun({status:'infra_error', summary:'Worker preparation failed',
    blocking_reasons:['必要审查未通过：pr_info','必要审查未通过：architecture']});
  const before = JSON.stringify(run);
  const groups = blockerGroups(run);
  assert.deepEqual(groups.map(group=>group.id), ['environment']);
  assert.equal(groups[0].reasons[0].reason,run.ai_review.summary);
  assert.equal(groups[0].impacts.length,2);
  assert.equal(JSON.stringify(run),before);
});

test('explicit network errors are not hidden by a generic preparation prefix', () => {
  for (const reason of ['worker preparation: Connection refused','Could not resolve host: gitee.com',
    '网络连接失败：代理不可达','certificate verify failed']) {
    assert.equal(blockerGroups(errorRun({blocking_reasons:[reason]}))[0].id,'network');
  }
  for (const reason of ['Out of memory','No space left on device',"ModuleNotFoundError: No module named 'torch'",
    'Cannot connect to the Docker daemon']) {
    assert.equal(blockerGroups(errorRun({blocking_reasons:[reason]}))[0].id,'environment');
  }
});

test('a negative review about networking is a review blocker, not an outage', () => {
  const summary = '网络连接失败时未释放资源，违反接口契约';
  const groups = blockerGroups(errorRun({status:'fail',summary,
    reviews:[{kind:'architecture',status:'fail',summary}],
    blocking_reasons:['必要审查未通过：architecture — '+summary]}));
  assert.deepEqual(groups.map(group=>group.id),['review']);
  assert.deepEqual(groups[0].reasons.map(item=>item.reason),['架构契约：'+summary]);
  const finding = {summary:'DNS failure handling leaks credentials',severity:'high'};
  const findingGroups=blockerGroups(errorRun({findings:[finding],blocking_reasons:[finding.summary]}));
  assert.equal(findingGroups[0].id,'review');
  assert.deepEqual(findingGroups[0].reasons.map(item=>item.reason),[finding.summary]);
});

test('test failure, incomplete coverage, and absent evidence stay distinct', () => {
  const run = errorRun({status:'fail',policy:{required_checks:['frontend_tests','frontend_build']},
    checks:[{tool_id:'frontend_tests',status:'fail',summary:'断言失败：期望 2，实际 3'},
      {tool_id:'frontend_build',status:'skipped',summary:'尚未执行'}],
    blocking_reasons:['frontend_tests：断言失败：期望 2，实际 3',
      '最低必检未通过：frontend_build','frontend_smoke 引用的证据文件不存在：smoke.log']});
  assert.deepEqual(blockerGroups(run).map(group=>group.id),['validation','publication','unknown']);
  assert.ok(blockerGroups(run).find(group=>group.id==='unknown').impacts.length);
});

test('ambiguous failures and generic timeouts do not guess network or server causes', () => {
  assert.equal(blockerGroups(errorRun({summary:'Codex task time budget exhausted'}))[0].id,'execution');
  assert.equal(blockerGroups(errorRun({blocking_reasons:['Command timed out']}))[0].id,'execution');
  assert.equal(blockerGroups(errorRun({blocking_reasons:['任务返回 137，原因未记录']}))[0].id,'unknown');
  assert.equal(blockerGroups(errorRun({}))[0].id,'unknown');
  for (const summary of ['Rootless Docker preparation succeeded; unspecified failure later',
    'TLS handshake succeeded; unspecified failure later','结果校验通过，另有问题待定位']) {
    assert.equal(blockerGroups(errorRun({summary}))[0].id,'unknown');
  }
});

test('successful runs stay successful when publishing has a separate error', () => {
  const run = errorRun({status:'pass',summary:'验证通过'});
  assert.deepEqual(blockerGroups(run),[]);
  run.receiver_message = '结果读取或 GitHub 发布未完成；稍后重试接收，不重跑构建。';
  assert.equal(blockerGroups(run)[0].id,'publication');
  assert.equal(run.conclusion,'success');
});

test('all original reasons survive classification, with exact duplicates collapsed', () => {
  const reasons = ['worker preparation failed','tested tracked source changed during execution',
    '未知异常 <img src=x onerror=alert(1)>','未知异常 <img src=x onerror=alert(1)>'];
  const groups = blockerGroups(errorRun({blocking_reasons:reasons}));
  assert.deepEqual(new Set(groups.flatMap(group=>group.reasons.map(item=>item.reason))),new Set(reasons));
  assert.equal(groups.flatMap(group=>group.reasons).length,3);
});

test('task details preserve findings, limitations, environment and evidence states safely', () => {
  const vm = require('node:vm');
  const fs = require('node:fs');
  class Element {
    constructor(tag) { this.tag=tag; this.children=[]; this.textContent=''; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children=[...children]; }
    addEventListener() {}
    set innerHTML(_value) { throw new Error('Remote text must not be rendered as HTML'); }
  }
  const nodes = new Map();
  const document = {createElement:tag=>new Element(tag), getElementById:id=>{
    if(!nodes.has(id))nodes.set(id,new Element('div'));
    return nodes.get(id);
  }};
  const root = new Element('main');
  const original = '未知异常 <img src=x onerror=alert(1)>';
  const context = {document,URLSearchParams,URL,setInterval:()=>{},fetch:()=>new Promise(()=>{}),
    location:{search:''},LocalCIData:{blockerGroups,environmentProfile},root,
    run:errorRun({summary:'Worker preparation failed',
      blocking_reasons:['必要审查未通过：architecture',original]})};
  vm.runInNewContext(fs.readFileSync(require.resolve('../../../dashboard/local-ci.js'),'utf8')+
    '\nrenderBlockers(root,run);',context);
  const flatten = node => [node,...node.children.flatMap(flatten)];
  const rendered = flatten(root);
  assert.ok(rendered.some(node=>node.textContent===original));
  assert.ok(!rendered.some(node=>node.tag==='img'));
  for (const [environment,expected] of [
    [{variants:{base:{profile:'triton-3.0',backend_enabled:true},candidate:{profile:'triton-3.1',backend_enabled:false}}},
      ['base 环境','triton-3.0','candidate 环境','triton-3.1']],
    [{variants:{candidate:{profile:original}}},['base 环境','未记录','candidate 环境',original]],
    [{profile:'legacy'},['环境','legacy']],
    [{generation:'legacy-generation'},['环境','legacy-generation']],
    [{},['环境','未记录']],
  ]) {
    context.run=errorRun({environment});
    vm.runInNewContext('renderDetail(run)',context);
    const facts=flatten(nodes.get('taskDetail')).find(node=>node.className==='ci-facts');
    assert.deepEqual(facts.children.slice(2).map(node=>node.textContent),expected);
  }
  const findings=[{summary:'插件未被发现',severity:'high',qualification:'目录扫描遗漏独立安装插件',code_evidence:['src/file.py:17']},
    {summary:'初始化泄漏资源',blocking:true}];
  context.run=errorRun({status:'fail',findings,blocking_reasons:findings.map(item=>item.summary),
    reviews:[{kind:'architecture',status:'fail',summary:'架构审查重复诊断'}],
    checks:[{tool_id:'control_plane',status:'infra_error',summary:'工具异常'}],limitations:[original,original]});
  assert.deepEqual(context.run.limitations,[original,original]);
  vm.runInNewContext('renderDetail(run)',context);
  const detail=nodes.get('taskDetail');
  const blockers=detail.children.find(node=>node.className==='ci-blockers');
  const items=blockers.children.find(node=>node.tag==='ul').children;
  assert.deepEqual(items.map(node=>node.textContent),findings.map(item=>item.summary));
  assert.ok(flatten(blockers).some(node=>node.textContent==='src/file.py:17'));
  assert.ok(flatten(blockers).some(node=>node.textContent===findings[0].qualification));
  assert.ok(!flatten(blockers).some(node=>['工具异常',original,'架构审查重复诊断'].includes(node.textContent)));
  for (const status of ['pass','infra_error','cancelled']) {
    context.run=errorRun({status,limitations:[original,original],
      checks:[{tool_id:'control_plane',status:status==='pass'?'pass':'infra_error',summary:'工具异常'}]});
    vm.runInNewContext('renderDetail(run)',context);
    assert.deepEqual(blockerGroups(context.run),[]);
    const limits=detail.children.find(node=>node.children.some(child=>child.textContent==='限制说明'));
    assert.deepEqual(limits.children[1].children.map(node=>node.textContent),[original]);
    assert.ok(!flatten(limits).some(node=>node.tag==='img'));
    assert.equal(context.run.local_conclusion,{pass:'passed',infra_error:'error',cancelled:'cancelled'}[status]);
  }
  for (const reasons of [['无 finding 的独立失败'],[]]) {
    context.run=errorRun({status:'fail',summary:'失败摘要',blocking_reasons:reasons,limitations:[]});
    vm.runInNewContext('renderDetail(run)',context);
    assert.deepEqual(blockerGroups(context.run)[0].reasons.map(item=>item.reason),reasons.length?reasons:['失败摘要']);
    assert.ok(!flatten(detail).some(node=>node.textContent==='限制说明'));
  }
  context.run=errorRun({status:'infra_error',checks:[{tool_id:'frontend_smoke',status:'pass'}],
    reviews:[{kind:'architecture',status:'pass'}],
    evidence_delivery:{status:'incomplete',omitted:[{path:'smoke.log',required:true}]},
    blocking_reasons:['必传检查证据未完整发布，整体结论待确认：smoke.log']});
  assert.equal(blockerGroups(context.run)[0].id,'publication');
  assert.equal(vm.runInNewContext('evidencePending(run)',context),true);
  for(const status of ['error','failed']) {
    context.run.checks[0].status=status;
    assert.equal(vm.runInNewContext('evidencePending(run)',context),false);
  }
});

test('incompletion is an impact under one recorded cause, otherwise its cause is unknown', () => {
  const missing = '必要审查未通过：architecture';
  for (const [reason, category] of [['Could not resolve host: gitee.com','network'],
    ['Codex task time budget exhausted','execution']]) {
    const groups = blockerGroups(errorRun({summary:reason,blocking_reasons:[missing]}));
    assert.deepEqual(groups.map(group=>group.id),[category]);
    assert.equal(groups[0].impacts[0].reason,missing);
  }
  const unknown = blockerGroups(errorRun({blocking_reasons:[missing]}));
  assert.deepEqual(unknown.map(group=>group.id),['unknown']);
  assert.equal(unknown[0].impacts[0].reason,missing);
  const mixed = blockerGroups(errorRun({blocking_reasons:['Out of memory','Connection refused',missing]}));
  assert.deepEqual(mixed.map(group=>group.id),['environment','network','unknown']);
  assert.equal(mixed.find(group=>group.id==='unknown').impacts[0].reason,missing);
  assert.ok(mixed.filter(group=>group.id!=='unknown').every(group=>group.impacts.length===0));
});
