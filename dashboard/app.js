const state = {
  manifest: null,
  fullTest: null,
  backends: null,
  performance: null,
  query: "",
  status: "all",
  stage: "all",
  page: 1,
  pageSize: 50,
};


const statusLabels = {
  passed: "通过",
  success: "通过",
  failed: "未通过",
  failure: "未通过",
  error: "执行错误",
  timeout: "超时",
  warning: "警告",
  pending: "等待 / 待交付",
  expired: "证据已到期",
  not_comparable: "无可比基线",
  not_applicable: "不适用",
  not_selected: "本次未选择",
  not_recorded: "未记录",
  skipped: "未执行",
  stale: "已过期",
  healthy: "正常",
  busy: "执行中",
  degraded: "异常",
  offline: "离线",
  idle: "空闲",
  polling: "轮询中",
  running: "执行中",
  not_run: "尚未轮询",
  unknown: "未知",
  disabled: "未启用",
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeStatus(value) {
  const status = String(value || "unknown").toLowerCase();
  if (status === "success" || status === "pass") return "passed";
  if (status === "waiting") return "pending";
  if (status === "failure" || status === "fail") return "failed";
  if (status === "infra_error" || status === "error") return "error";
  if (status === "healthy" || status === "idle") return "passed";
  if (status === "busy" || status === "polling" || status === "running") return "pending";
  if (status === "degraded") return "warning";
  if (status === "offline") return "failed";
  if (status === "not_run") return "unknown";
  return status;
}

function statusBadge(value) {
  const normalized = normalizeStatus(value);
  const label = statusLabels[value] || statusLabels[normalized] || value || "未知";
  return `<span class="status-badge status-${escapeHtml(normalized)}">${escapeHtml(label)}</span>`;
}

function formatDate(value) {
  if (!value) return "--";
  const date = new Date(typeof value === "number" ? value * 1000 : value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

function formatDuration(ms) {
  if (ms === null || ms === undefined || ms === "") return "--";
  const value = Number(ms);
  if (!Number.isFinite(value)) return "--";
  if (value >= 60000) return `${(value / 60000).toFixed(1)} min`;
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(2)} ms`;
}

function safeExternalUrl(value) {
  if (!value) return "";
  try {
    const url = new URL(value, window.location.href);
    return url.protocol === "https:" || url.protocol === "http:" ? url.href : "";
  } catch {
    return "";
  }
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

async function loadData() {
  const requested = new URLSearchParams(location.search).get("data");
  const source = requested && /^data\/[A-Za-z0-9_.-]+\.json$/.test(requested) ? requested : "data/tasks.json";
  const data = LocalCIData.normalize(await fetchJson(source));
  Object.assign(state, LocalCIData.business(data));
}

function renderHeader() {
  $("#generatedAt").textContent = (state.manifest.mode === "mock" ? "界面样例（非真实验证结果） · " : "") + `数据更新：${formatDate(state.manifest.generated_at)}`;
  $("#schemaVersion").textContent = `全量算子与后端性能结果 · 发布分支：${state.manifest.source_branch}`;
  $("#operatorHeading").textContent = '全量算子测试';
  $("#fullRunBackend").textContent = [state.fullTest.run.backend, state.fullTest.run.profile].filter(Boolean).join(' · ');
  $("#fullRunSha").textContent = [state.fullTest.run.sha.slice(0, 12),
    state.fullTest.run.measured_at ? formatDate(state.fullTest.run.measured_at) : ''].filter(Boolean).join(' · ');
  const csv = [["序号","算子","状态","失败阶段","耗时(ms)"],...state.fullTest.operators.map(row => [row.index,row.name,row.status,row.failure_stage,row.duration_ms])]
    .map(row => row.map(value => '"' + String(value ?? '').replaceAll('"','""') + '"').join(',')).join('\r\n');
  $("#downloadRawCsv").href = URL.createObjectURL(new Blob(['\ufeff' + csv], {type:'text/csv;charset=utf-8'}));
  $("#downloadRawCsv").download = 'full-operators.csv';
}

function computeOperatorSummary(rows) {
  const summary = { total: rows.length, passed: 0, failed: 0, error: 0, timeout: 0 };
  rows.forEach((row) => {
    const status = normalizeStatus(row.status);
    if (status in summary) summary[status] += 1;
  });
  summary.exceptions = summary.failed + summary.error + summary.timeout;
  summary.passRate = summary.total ? (summary.passed / summary.total) * 100 : 0;
  return summary;
}

function renderOperatorMetrics() {
  const summary = computeOperatorSummary(state.fullTest.operators);
  const metrics = [
    ["算子总数", summary.total, ""],
    ["通过", summary.passed, ""],
    ["未通过", summary.failed, ""],
    ["执行错误", summary.error, ""],
    ["超时", summary.timeout, ""],
    ["通过率", `${summary.passRate.toFixed(1)}%`, `${summary.exceptions} 项异常`],
  ];
  $("#operatorMetrics").innerHTML = metrics
    .map(
      ([label, value, detail]) => `
        <div class="metric-item">
          <span class="metric-label">${escapeHtml(label)}</span>
          <span class="metric-value">${escapeHtml(value)}</span>
          ${detail ? `<span class="metric-detail">${escapeHtml(detail)}</span>` : ""}
        </div>`,
    )
    .join("");
}

function populateStageFilter() {
  const stages = [...new Set(state.fullTest.operators.map((row) => row.failure_stage).filter(Boolean))].sort();
  $("#stageFilter").insertAdjacentHTML(
    "beforeend",
    stages.map((stage) => `<option value="${escapeHtml(stage)}">${escapeHtml(stage)}</option>`).join(""),
  );
}

function filteredOperators() {
  const query = state.query.trim().toLowerCase();
  return state.fullTest.operators.filter((row) => {
    const status = normalizeStatus(row.status);
    const matchesQuery = !query || row.name.toLowerCase().includes(query);
    const matchesStage = state.stage === "all" || row.failure_stage === state.stage;
    let matchesStatus = state.status === "all" || status === state.status;
    if (state.status === "exception") matchesStatus = ['failed','error','timeout'].includes(status);
    return matchesQuery && matchesStage && matchesStatus;
  });
}

function renderOperators() {
  const rows = filteredOperators();
  const totalPages = state.pageSize === "all" ? 1 : Math.max(1, Math.ceil(rows.length / state.pageSize));
  state.page = Math.min(state.page, totalPages);
  const start = state.pageSize === "all" ? 0 : (state.page - 1) * state.pageSize;
  const visible = state.pageSize === "all" ? rows : rows.slice(start, start + state.pageSize);

  $("#operatorRows").innerHTML = visible
    .map(
      (row) => `
        <tr>
          <td>${escapeHtml(row.index)}</td>
          <td>${escapeHtml(row.name)}${safeExternalUrl(row.log_url) ? ` <a class="details-link" target="_blank" rel="noreferrer" href="${escapeHtml(safeExternalUrl(row.log_url))}">证据</a>` : ""}</td>
          <td>${statusBadge(row.status)}</td>
          <td>${escapeHtml(row.failure_stage || "--")}</td>
          <td>${formatDuration(row.duration_ms)}</td>
        </tr>`,
    )
    .join("");

  $("#operatorEmpty").hidden = rows.length !== 0;
  const rangeEnd = Math.min(start + visible.length, rows.length);
  $("#rowCount").textContent = rows.length
    ? `显示 ${start + 1}-${rangeEnd}，共 ${rows.length} 项`
    : "共 0 项";
  $("#pageIndicator").textContent = `${state.page} / ${totalPages}`;
  $("#previousPage").disabled = state.page <= 1;
  $("#nextPage").disabled = state.page >= totalPages;
}

function renderBackends() {
  const configuredIds = state.manifest.display?.backend_ids;
  const visibleIds = Array.isArray(configuredIds) ? new Set(configuredIds) : null;
  const visibleBackends = state.backends.backends.filter(
    (backend) => !visibleIds || visibleIds.has(backend.id),
  );
  $("#backendRows").innerHTML = visibleBackends
    .map((backend) => {
      const tests = backend.tests || {};
      const resultUrl = safeExternalUrl(backend.result_url);
      const detail = resultUrl
        ? `<a class="details-link" href="${escapeHtml(resultUrl)}" target="_blank" rel="noreferrer">详情</a>`
        : "--";
      return `
        <tr>
          <td><strong>${escapeHtml(backend.name)}</strong></td>
          <td>${escapeHtml(backend.profile || '未记录')}</td>
          <td>${statusBadge(backend.state)}</td>
          <td>${statusBadge(tests.backend)}</td>
          <td>${statusBadge(tests.compile_time)}</td>
          <td>${statusBadge(tests.pass_profile)}</td>
          <td>${statusBadge(tests.ir_serialization)}</td>
          <td><code>${escapeHtml((backend.sha || "-------").slice(0, 7))}</code></td>
          <td>${escapeHtml(formatDate(backend.tested_at))}</td>
          <td>${detail}</td>
        </tr>`;
    })
    .join("");
  $("#backendEmpty").hidden = visibleBackends.length !== 0;
}

function renderPerformanceList(target, rows, valueKey, maxValue, formatter) {
  if (!rows.length) {
    $(target).innerHTML = '<div class="empty-state">当前结果中暂无该项数据。</div>';
    return;
  }
  $(target).innerHTML = rows
    .map((row) => {
      const value = Number(row[valueKey] || 0);
      const width = maxValue ? Math.max(2, Math.min(100, (value / maxValue) * 100)) : 0;
      const status = normalizeStatus(row.status || "success");
      const barClass = status === "failed" ? "failure" : status === "warning" ? "warning" : "";
      return `
        <div class="performance-row">
          <span class="performance-name" title="${escapeHtml(row.name)}">${escapeHtml(row.name)}</span>
          <div class="performance-track"><div class="performance-bar ${barClass}" style="width:${width}%"></div></div>
          <span class="performance-value">${formatter(row)}</span>
        </div>`;
    })
    .join("");
}

function renderPerformance() {
  const compileRows = state.performance.compile_time.kernels;
  const passRows = state.performance.pass_profile.hotspots;
  const irRows = state.performance.ir_serialization.metrics;
  $("#performanceBackend").textContent = state.performance.backend;
  for (const [id, key] of [['compileSource','compile_time'],['passSource','pass_profile'],['irSource','ir_serialization']]) {
    const measurement = state.performance[key];
    $('#' + id).textContent = measurement.sha
      ? [measurement.backend, measurement.profile, measurement.sha.slice(0, 12), formatDate(measurement.measured_at)].filter(Boolean).join(' · ')
      : '尚无有效测量';
  }

  renderPerformanceList(
    "#compileMetrics",
    compileRows,
    "candidate_ms",
    Math.max(...compileRows.map((row) => Number(row.candidate_ms || 0))),
    (row) => {
      if (row.delta_percent === null || row.delta_percent === undefined) {
        return `${Number(row.candidate_ms).toFixed(1)} ms <span>无历史基线</span>`;
      }
      const delta = Number(row.delta_percent);
      const deltaClass = delta > 0 ? "delta-up" : delta < 0 ? "delta-down" : "";
      const sign = delta > 0 ? "+" : "";
      return `${Number(row.candidate_ms).toFixed(1)} ms <span class="${deltaClass}">${sign}${delta.toFixed(1)}%</span>`;
    },
  );
  renderPerformanceList(
    "#passMetrics",
    passRows,
    "median_ms",
    Math.max(...passRows.map((row) => Number(row.median_ms || 0))),
    (row) => `${Number(row.median_ms).toFixed(2)} ms`,
  );
  renderPerformanceList(
    "#irMetrics",
    irRows,
    "median_ms",
    Math.max(...irRows.map((row) => Number(row.median_ms || 0))),
    (row) => `${Number(row.median_ms).toFixed(3)} ms`,
  );
}

function downloadFilteredXlsx() {
  const headers = ["序号", "算子名称", "测试状态", "失败阶段", "耗时(ms)"];
  const rows = filteredOperators().map((row) => [
    row.index,
    row.name,
    statusLabels[normalizeStatus(row.status)] || row.status,
    row.failure_stage,
    row.duration_ms,
  ]);
  window.TritonXlsx.downloadWorkbook(
    `operator-results-${state.status}.xlsx`,
    "算子测试结果",
    headers,
    rows,
  );
}

function bindEvents() {
  $("#operatorSearch").addEventListener("input", (event) => {
    state.query = event.target.value;
    state.page = 1;
    renderOperators();
  });
  $$(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      $$(".segment").forEach((item) => item.classList.toggle("active", item === button));
      state.status = button.dataset.status;
      state.page = 1;
      renderOperators();
    });
  });
  $("#stageFilter").addEventListener("change", (event) => {
    state.stage = event.target.value;
    state.page = 1;
    renderOperators();
  });
  $("#pageSize").addEventListener("change", (event) => {
    state.pageSize = event.target.value === "all" ? "all" : Number(event.target.value);
    state.page = 1;
    renderOperators();
  });
  $("#previousPage").addEventListener("click", () => {
    state.page -= 1;
    renderOperators();
  });
  $("#nextPage").addEventListener("click", () => {
    state.page += 1;
    renderOperators();
  });
  $("#downloadFilteredXlsx").addEventListener("click", downloadFilteredXlsx);
}

async function initialize() {
  const selectedView = new URLSearchParams(window.location.search).get("view") === "performance" ? "performance" : "operators";
  $$(".tab-button[data-view]").forEach((link) => {
    const selected = link.dataset.view === selectedView;
    link.classList.toggle("active", selected);
    if (selected) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `${selectedView}View`));
  try {
    await loadData();
    renderHeader();
    renderOperatorMetrics();
    populateStageFilter();
    renderOperators();
    renderBackends();
    renderPerformance();

    bindEvents();
  } catch (error) {
    $("#operatorsView").classList.remove("active");
    $("#performanceView").classList.remove("active");

    $("#loadError").hidden = false;
    $("#loadErrorMessage").textContent = error instanceof Error ? error.message : String(error);
  }
}

initialize();
