/* Render remote result text as text nodes. No result field can inject HTML. */
const labels = {success:'通过',passed:'通过',failure:'未通过',failed:'未通过',warning:'提示',limited:'验证受限',error:'执行错误',evidence_pending:'证据待确认',cancelled:'已取消',superseded:'已失效 / 被替代',skipped:'未执行',not_selected:'本次未选择',not_applicable:'不适用',healthy:'正常',degraded:'异常',offline:'离线',snapshot_stale:'快照已过期',unknown:'状态未知',waiting:'等待',ready:'已交付',pending:'待交付',expired:'已到期',not_comparable:'无可比基线'};
const names = {environment:'环境与依赖',frontend_build:'前端构建',frontend_install:'前端安装与导入',frontend_tests:'前端测试',wheel_install:'Wheel 安装与导入',frontend_smoke:'前端基本功能验证',backend_build:'后端构建',backend_install:'后端安装与发现',backend_tests:'后端测试',backend_rebuild:'后端重新构建',backend_smoke:'后端基本功能与 JIT 验证',flaggems:'FlagGems',compile_time:'编译时间性能',pass_profile:'编译阶段性能剖析',ir_serialization:'IR 序列化',pr_information:'PR 说明与改动核验',architecture_review:'架构与接口约束审查',control_plane:'CI 流程检查',custom_test:'定向测试'};
const friendlyReasons = {'cancelled':'任务已取消','missing required check':'必检尚未完成','PR intent and attributes were not reviewed':'PR 意图与属性尚未完成核验','all commands completed successfully':'命令执行成功','not selected for this change':'本次改动未触发该项检查','minimum frontend coverage':'编译器改动的最低检查范围','frontend code or test behavior changed':'前端代码或测试行为发生变化','architecture contract review is mandatory':'架构审查为必检项'};
const model = {data:null, selected:null};
const $ = id => document.getElementById(id);
const arr = value => Array.isArray(value) ? value : [];
const txt = value => typeof value === 'string' ? value : value == null ? '' : JSON.stringify(value);
function el(tag, className, text) { const node=document.createElement(tag); if(className)node.className=className; if(text!=null)node.textContent=txt(text); return node; }
function badge(status) { const tone=['success','passed','healthy','ready'].includes(status)?'good':['failure','failed','offline'].includes(status)?'bad':['error','evidence_pending','degraded','waiting','pending','expired','snapshot_stale'].includes(status)?'warn':status==='cancelled'?'info':''; return el('span','ci-badge '+tone,labels[status]||status||'未知'); }
function evidencePending(run) {
  return run.local_conclusion==='error' && arr(run.evidence_delivery?.omitted).some(item=>item.required===true) &&
    [...arr(run.checks),...['pr_info','architecture','intent'].map(kind=>run.ai_review?.[kind]).filter(Boolean)]
      .every(item=>['passed','not_selected','not_applicable'].includes(item.status));
}
function date(value) { const d=new Date(typeof value==='number'?value*1000:value); return value!=null&&!Number.isNaN(d.valueOf())?d.toLocaleString('zh-CN',{hour12:false}):'尚无时间记录'; }
function link(label, url) { try { const target=new URL(url); if(target.protocol!=='https:')return null; const a=el('a','',label); a.href=target.href; a.target='_blank'; a.rel='noopener noreferrer'; return a; } catch { return null; } }
function empty(container, message) { container.append(el('div','ci-empty',message)); }
function section(parent,title) { const s=el('section'); s.append(el('h3','',title)); parent.append(s); return s; }
function key(run) { return run.task_id+'/'+run.run_id; }
function title(run) { return run.pr_number?'PR #'+run.pr_number+' · '+run.target_branch:run.target_branch+' · '+(run.event_kind==='push'?'分支提交':'手动任务'); }
function displaySha(run) { return run.pr_number&&run.head_sha?run.head_sha:run.tested_sha; }
function reason(value) { return friendlyReasons[value]||txt(value)||'没有记录原因'; }
function publicText(value) { let text=txt(value);for(const [id,label] of Object.entries(names))text=text.replaceAll(id+':',label+'：');for(const [message,label] of Object.entries(friendlyReasons))text=text.replaceAll(message,label);return text.replaceAll('Codex cancelled','AI 验证已取消').replaceAll('base..merge','基准提交与合并验证提交之间').replace(/\bhead\(([^)]+)\)/g,'PR 提交 $1').replace(/\bmerge\(([^)]+)\)/g,'合并验证提交 $1').replaceAll('changed_paths','影响文件列表').replaceAll('Frontend','前端').replaceAll('Backend','后端').replaceAll(' smoke','基本功能验证').replaceAll(' wheel',' wheel 包'); }

function facts(parent,rows) {const list=el('dl','ci-facts');for(const [name,value] of rows){list.append(el('dt','',name),el('dd','',value??'未采集'));}parent.append(list);}

function filteredRuns() {
  const query=$('taskSearch').value.trim().toLowerCase(); const filter=$('resultFilter').value;
  const prQuery=query.match(/^#?([1-9][0-9]*)$/); const includeHistory=$('historyFilter').value==='all';
  return arr(model.data.runs).filter(run=>(includeHistory||run.is_current)&&(filter==='all'||run.conclusion===filter)&&(prQuery?String(run.pr_number)===prQuery[1]:[run.pr_number,run.target_branch,displaySha(run)].join(' ').toLowerCase().includes(query)));
}

function renderList() {
  const root=$('taskList'); root.replaceChildren(); const runs=filteredRuns();
  $('taskCount').textContent=runs.length;
  if(!runs.some(run=>key(run)===model.selected))model.selected=runs.length?key(runs[0]):null;
  if(!runs.length)empty(root,arr(model.data.runs).length?'没有符合筛选条件的任务。':'尚未发布 Local CI 结果。');
  for(const run of runs) {
    const button=el('button','ci-task'); button.type='button'; button.setAttribute('aria-pressed',String(model.selected===key(run)));
    button.append(el('strong','',title(run)),badge(run.conclusion==='error'&&evidencePending(run)?'evidence_pending':run.conclusion),el('small','',displaySha(run).slice(0,12)+(run.is_current?' · 当前结果':run.superseded?' · 已失效 / 被替代 · 历史记录':' · 历史记录')),el('small','',date(run.completed_at)));
    button.addEventListener('click',()=>{model.selected=key(run);renderList();}); root.append(button);
  }
  renderDetail(runs.find(run=>key(run)===model.selected));
}

function evidenceList(parent, entries) {
  const list=el('ul','ci-evidence');
  for(const entry of arr(entries))list.append(el('li','',typeof entry==='string'?entry:[entry.path,entry.line?'L'+entry.line:'',entry.reason||entry.summary||''].filter(Boolean).join(' · ')));
  parent.append(list);
}

function incompleteText(value) {
  const match=txt(value).match(/^(?:必要审查未通过|最低必检未通过)：([a-z_]+)(.*)$/);
  if(!match)return publicText(value);
  const label=({pr_info:'PR 信息核验',architecture:'架构契约审查',intent:'变更意图审查'})[match[1]]||names[match[1]]||match[1];
  return label+'尚未完成'+publicText(match[2]);
}

function shortBlocker(item, category) {
  const text=txt(item.reason);
  // Review findings describe code behavior; do not translate their vocabulary
  // into an infrastructure diagnosis.
  if(category==='environment') {
    const rules=[
      [/worker revision differs from installed control/i,'控制版本不匹配'],
      [/Trusted profile and exact LLVM revision are required/i,'服务器 Profile 或 LLVM 版本配置不匹配'],
      [/LLVM.{0,30}(?:mismatch|not found|missing)|dependency version mismatch|依赖.{0,15}不匹配/i,'依赖版本不匹配'],
      [/no space left on device|磁盘空间不足/i,'磁盘空间不足'],
      [/out of memory|\bOOM(?:Killed)?\b|内存不足/i,'内存不足'],
      [/permission denied|权限不足/i,'权限不足'],
      [/no module named|ModuleNotFoundError|shared librar(?:y|ies).{0,40}(?:not found|cannot open)|依赖.{0,15}缺失/i,'依赖缺失或无法加载'],
      [/cannot connect to the docker daemon|rootless Docker.{0,40}(?:required|failed|invalid)|容器.{0,12}(?:启动失败|不可用)/i,'容器运行环境不可用'],
      [/worker preparation(?: failed|:|$)|环境准备.{0,15}(?:失败|未完成)/i,'服务器环境准备失败'],
    ];
    for(const [pattern,label] of rules)if(pattern.test(text))return label;
  }
  if(category==='network')return '网络连接异常';
  if(category==='execution'&&/timed? ?out|timeout|time budget exhausted|超时/i.test(text))return '执行超时';
  if(item.source==='展示说明'||item.source==='任务状态')return '原因未明确，需查看执行日志';
  return publicText(text).replace(/\s+/g,' ').trim();
}

function renderBlockers(parent, run) {
  const groups=LocalCIData.blockerGroups(run);
  if(!groups.length)return;
  {
    const selected=groups;
    const box=el('section','ci-blockers');
    box.append(el('h3','','阻塞原因'));
    const list=el('ul'),visible=[];
    for(const group of selected){
      if(Array.isArray(run.limitations)||group.reasons.some(item=>item.finding)) {
        for(const item of group.reasons) {
          const row=el('li','',publicText(item.reason));
          if(item.finding?.qualification)row.append(el('p','ci-muted',publicText(item.finding.qualification)));
          if(arr(item.finding?.code_evidence).length)evidenceList(row,item.finding.code_evidence);
          list.append(row);
        }
        continue;
      }
      // Show one root cause per category. Check rows and generic incomplete
      // impacts remain available in the detail table.
      const direct=group.reasons.filter(item=>!String(item.source||'').startsWith('检查：') && item.source!=='展示说明');
      const reasons=direct.length ? direct : group.reasons.filter(item=>item.source!=='展示说明');
      if(reasons.length) {
        const summary=shortBlocker(reasons[0],group.id);
        visible.push(reasons.length>1 ? `${summary}（另有 ${reasons.length-1} 项同类原因，详见执行报告）` : summary);
      } else if(group.impacts.length) visible.push('部分必检项未完成，具体原因见执行报告');
    }
    for(const text of new Set(visible))list.append(el('li','',text));
    box.append(list);
    parent.append(box);
  }
}

function renderDetail(run) {
  const root=$('taskDetail'); root.replaceChildren();
  if(!run) { empty(root,'选择一个任务以查看检查范围、审查结论和执行证据。尚无数据时不会显示通过状态。'); return; }
  const pendingEvidence=evidencePending(run);
  const head=el('div','ci-detail-head'); const heading=el('div'); heading.append(el('p','eyebrow','验证与审查'),el('h2','',title(run))); head.append(heading,badge(run.conclusion==='error'&&pendingEvidence?'evidence_pending':run.conclusion));root.append(head);
  const identity=el('div','ci-identity'); identity.append(el('code','',displaySha(run)),el('span','',date(run.completed_at))); root.append(identity);
  const environmentRows = run.environment.variants
    ? ['base','candidate'].map(variant => {
      const profile = LocalCIData.environmentProfile(run.environment,variant) || '未记录';
      return [variant+' 环境',profile];
    })
    : [['环境',LocalCIData.environmentProfile(run.environment)||'未记录']];
  facts(root,[['本地验证',pendingEvidence?'证据待确认':labels[run.local_conclusion]||run.local_conclusion],...environmentRows]);
  if(run.superseded)root.append(el('p','ci-notice','任务已失效或被新任务替代；以下保留该次执行记录，不作为当前提交的合入依据。'));
  if(run.evidence_delivery?.status==='incomplete')root.append(el('p','ci-notice',pendingEvidence?'必传检查证据未完整发布，整体结论待确认；已执行检查结果保持原状态。':run.local_conclusion==='passed'?'执行通过，证据发布不完整。':'证据发布不完整，已执行检查结果保持原状态。'));
  if(run.receiver_message)root.append(el('p','ci-notice',run.receiver_message));
  const links=el('div','ci-links'); for(const [label,url] of [['查看完整结果',run.result_url],['查看执行产物',run.artifacts_url]]) { const a=link(label,url); if(a)links.append(a); }root.append(links);
  const metrics=el('div','ci-metrics'); const checks=arr(run.checks); const values=[[checks.filter(c=>c.required).length,'最低必检项'],[checks.filter(c=>c.status==='passed').length,'已通过检查'],[checks.filter(c=>['skipped','not_selected','not_applicable'].includes(c.status)).length,'未选择 / 未执行 / 不适用'],[arr(run.artifacts).filter(artifact=>!artifact.omitted).length,'所选证据文件']];
  for(const [value,label] of values){const box=el('div','ci-metric');box.append(el('strong','',value),el('span','',label));metrics.append(box);}root.append(metrics);
  renderBlockers(root,run);
  if(arr(run.limitations).length) {
    const limits=section(root,'限制说明');
    const list=el('ul','ci-evidence');
    for(const limitation of new Set(run.limitations))list.append(el('li','',publicText(limitation)));
    limits.append(list);
  }
  const scope=section(root,'检查选择与执行结果');
  const policy=run.policy||{};scope.append(el('p','ci-muted',policy.docs_only?'文档变更：依规则免构建；架构审查仍需提供证据。':policy.manual_full?'维护者手动触发全量测试。':'按改动影响选择检查，并满足主机控制面规定的最低要求。'));
  const wrap=el('div','ci-table-shell'),table=el('table','ci-table'),thead=el('thead'),header=el('tr'); for(const s of ['检查','要求','结果','结果说明'])header.append(el('th','',s));thead.append(header);table.append(thead);const tbody=el('tbody');
  for(const check of checks){const tr=el('tr'); const name=el('td','',names[check.id]||'补充检查');const requirement=el('td','',check.required?'必检':'按影响选择');const result=el('td');result.append(badge(check.status));const explanation=el('td','',publicText(reason(check.reason)));if(policy.reasons?.[check.id])explanation.append(el('small','','选择依据：'+publicText(reason(policy.reasons[check.id]))));tr.append(name,requirement,result,explanation);tbody.append(tr);}table.append(tbody);wrap.append(table);scope.append(wrap);
  const review=run.ai_review||{};
  const failedReview=LocalCIData.blockerGroups(run).some(group=>group.id==='review');
  if(!failedReview){
  const ai=section(root,'Codex 审查与定向验证');const summary=el('div','ci-review');summary.append(el('p','',publicText(review.summary)||'未收到完整审查结论。'));ai.append(summary);
  for(const [kind,label] of [['pr_info','PR 信息'],['architecture','架构契约'],['intent','变更意图']]){
    const item=review[kind];if(!item&&kind==='intent')continue;
    const card=el('div','ci-review');
    card.append(el('strong','',label+' '),badge(item?.status||'unknown'),el('p','',publicText(item?.summary)||'尚未收到此项审查结果。'));
    evidenceList(card,item?.evidence);ai.append(card);
  }
  for(const finding of arr(review.findings)){const risk=({critical:'严重',high:'高',medium:'中',low:'低',info:'提示'})[finding.severity]||'未标注';const card=el('div','ci-review');card.append(el('strong','',(finding.blocking?'合入阻塞':'非阻塞发现')+' · 风险：'+risk),el('p','',publicText(finding.summary||finding.title)||'发现'));if(finding.qualification)card.append(el('p','ci-muted',publicText(finding.qualification)));evidenceList(card,finding.code_evidence);ai.append(card);}
  }
  const performance=section(root,'性能变化');performance.append(el('p','ci-muted','性能回退或纯耗时变化仅报告；基准执行失败仍会阻塞。适用能力以该任务的环境声明为准。'));
  if(!arr(run.performance).length)performance.append(el('p','ci-muted','本次没有可展示的性能测量或基线对比。请结合上方检查状态判断是否适用。'));
  for(const item of arr(run.performance)){const card=el('div','ci-performance');card.append(el('strong','',names[item.tool]||'性能记录'),el('p','ci-muted',item.summary||item.reason||'已记录性能结果，详细数据见完整结果。'));performance.append(card);}
  const files=section(root,'所选日志与报告');
  if(!arr(run.artifacts).length)empty(files,'本次未选择需要发布的文件。完整运行记录保留在 CI 主机。');
  const list=el('ul','ci-evidence');
  for(const artifact of arr(run.artifacts)){
    const row=el('li');const label=artifact.label||artifact.path;
    row.append(link(label,artifact.url)||el('span','',label));
    if(artifact.omitted)row.append(el('span','ci-muted',' · '+artifact.omitted));
    else row.append(el('span','ci-muted',' · '+artifact.size+' bytes'));
    list.append(row);
  }
  files.append(list);
  const context=el('details','ci-receipt');const commitBox=el('div','ci-evidence');commitBox.append(el('p','',run.pr_number?'PR 提交：'+run.head_sha:'被测提交：'+run.tested_sha));if(run.pr_number&&run.tested_sha!==run.head_sha)commitBox.append(el('p','',`与 ${run.target_branch} 合并后的验证提交：${run.tested_sha}`));if(arr(policy.changed_paths).length){commitBox.append(el('p','','影响文件：'));evidenceList(commitBox,policy.changed_paths);}context.append(el('summary','','被测提交与影响文件'),commitBox);root.append(context);
}

async function load() {
  const notice=$('dataNotice'); const refresh=$('refresh'); refresh.disabled=true;refresh.textContent='读取中…';
  try {
    const requested=new URLSearchParams(location.search).get('data');
    const source=requested&&/^data\/[A-Za-z0-9_.-]+\.json$/.test(requested)?requested:'data/tasks.json';
    const response=await fetch(source+(source.includes('?')?'&':'?')+'refresh='+Date.now(),{cache:'no-store'});if(!response.ok)throw new Error('HTTP '+response.status);
    const data=LocalCIData.normalize(await response.json());
    model.data=data;notice.hidden=data.data_mode!=='fixture';notice.textContent='本机界面样例 · 以下数据用于验证展示，不构成编译、硬件或部署验收证据。';
    $('updatedAt').textContent=data.generated_at?'数据生成：'+date(data.generated_at)+' · 本页读取：'+new Date().toLocaleTimeString('zh-CN',{hour12:false}):'尚未同步真实结果';
    const warnings=$('syncWarnings');warnings.replaceChildren();warnings.hidden=!arr(data.warnings).length;if(!warnings.hidden){warnings.append(el('strong','','部分发布数据未通过校验'));const list=el('ul');for(const warning of data.warnings)list.append(el('li','',warning.path+'：'+warning.reason));warnings.append(list);}
    renderList();
  } catch(error) { notice.hidden=false;notice.textContent='无法加载 Local CI 数据：'+error.message+'。页面不据此推断 CI 通过。';if(!model.data){empty($('taskDetail'),'结果数据不可用。');} }
  finally {refresh.disabled=false;refresh.textContent='查询最新快照';}
}
$('refresh').addEventListener('click',load);$('taskSearch').addEventListener('input',()=>model.data&&renderList());$('historyFilter').addEventListener('change',()=>model.data&&renderList());$('resultFilter').addEventListener('change',()=>model.data&&renderList());
const initialPr=new URLSearchParams(location.search).get('pr');if(/^[1-9][0-9]*$/.test(initialPr||''))$('taskSearch').value=initialPr;
load();setInterval(()=>{if(!document.hidden)load();},300000);
