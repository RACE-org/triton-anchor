const state = {
  manifest: null,
  fullTest: null,
  backends: null,
  performance: null,
  workerHealth: null,
  workerHealthError: "",
  query: "",
  status: "all",
  stage: "all",
  page: 1,
  pageSize: 50,
};

let workerHealthRefreshTimer = null;
let workerHealthRefreshPromise = null;

const statusLabels = {
  passed: "通过",
  success: "通过",
  failed: "失败",
  failure: "失败",
  error: "错误",
  timeout: "超时",
  warning: "警告",
  pending: "运行中",
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
  if (status === "success") return "passed";
  if (status === "failure" || status === "error") return "failed";
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
  const date = new Date(value);
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

function formatElapsed(seconds) {
  if (seconds === null || seconds === undefined || seconds === "") return "--";
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return "--";
  if (value < 60) return `${Math.floor(value)} 秒`;
  if (value < 3600) return `${Math.floor(value / 60)} 分 ${Math.floor(value % 60)} 秒`;
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  return `${hours} 小时 ${minutes} 分`;
}

function formatBytes(bytes) {
  if (bytes === null || bytes === undefined || bytes === "") return "--";
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) return "--";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let scaled = value;
  let index = 0;
  while (scaled >= 1024 && index < units.length - 1) {
    scaled /= 1024;
    index += 1;
  }
  const digits = index === 0 || scaled >= 100 ? 0 : scaled >= 10 ? 1 : 2;
  return `${scaled.toFixed(digits)} ${units[index]}`;
}

function formatCpuUsage(cpuPercent, cpuCapacity) {
  const rawPercent = Number.parseFloat(String(cpuPercent ?? "").replace(/%$/, ""));
  const availableCpus = Number(cpuCapacity);
  if (!Number.isFinite(rawPercent) || rawPercent < 0) {
    return { used: "--", utilization: "--", ratio: "--" };
  }

  const usedCpus = rawPercent / 100;
  const used = `${usedCpus.toFixed(2)} CPU`;
  if (!Number.isFinite(availableCpus) || availableCpus <= 0) {
    return { used, utilization: "--", ratio: `${usedCpus.toFixed(2)} / -- CPU` };
  }

  const capacity = Number.isInteger(availableCpus) ? availableCpus.toFixed(0) : availableCpus.toFixed(2);
  return {
    used,
    utilization: `${((usedCpus / availableCpus) * 100).toFixed(2)}%`,
    ratio: `${usedCpus.toFixed(2)} / ${capacity} CPU`,
  };
}

function shortSha(value) {
  return value ? String(value).slice(0, 12) : "--";
}

function renderFactList(target, rows, emptyText = "暂无数据。") {
  if (!rows.length) {
    $(target).innerHTML = `<p class="fact-empty">${escapeHtml(emptyText)}</p>`;
    return;
  }
  $(target).innerHTML = rows
    .map(([label, value, kind]) => {
      let rendered = escapeHtml(value === null || value === undefined || value === "" ? "--" : value);
      if (kind === "code") {
        rendered = `<code title="${escapeHtml(value || "")}">${escapeHtml(value || "--")}</code>`;
      } else if (kind === "status") {
        rendered = statusBadge(value);
      }
      return `<div class="fact-row"><dt>${escapeHtml(label)}</dt><dd>${rendered}</dd></div>`;
    })
    .join("");
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

function decodeBase64Utf8(value) {
  const binary = atob(String(value || "").replaceAll(/\s/g, ""));
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  return new TextDecoder().decode(bytes);
}

async function fetchLiveWorkerHealth() {
  const source = state.manifest.live_sources?.worker_health;
  if (!source?.url || source.kind !== "gitee_contents_api") {
    throw new Error("Worker health live source is not configured");
  }
  const response = await fetchJson(source.url);
  if (response.encoding !== "base64" || typeof response.content !== "string") {
    throw new Error("Gitee worker health response is invalid");
  }
  const document = JSON.parse(decodeBase64Utf8(response.content));
  if (document.schema !== "triton-anchor-local-ci-worker-health/v1") {
    throw new Error("Worker health snapshot schema is invalid");
  }
  return document;
}

function secondsSince(value) {
  if (!value) return null;
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return null;
  return Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
}

function workerHealthForDisplay(document) {
  const health = document && typeof document === "object" ? document : {};
  const snapshotAge = secondsSince(health.collected_at);
  const staleSeconds = Number(
    state.manifest?.live_sources?.worker_health?.stale_seconds || 1200,
  );
  const poller = health.poller && typeof health.poller === "object" ? health.poller : {};
  const task = health.active_task && typeof health.active_task === "object"
    ? health.active_task
    : null;
  const stale = health.data_mode === "live"
    && (snapshotAge === null || snapshotAge > staleSeconds);
  return {
    ...health,
    state: stale ? "offline" : health.state,
    snapshot_age_seconds: snapshotAge,
    poller: {
      ...poller,
      heartbeat_age_seconds: secondsSince(poller.heartbeat_at),
    },
    active_task: task
      ? { ...task, elapsed_seconds: secondsSince(task.started_at) }
      : null,
  };
}

async function refreshWorkerHealth() {
  if (workerHealthRefreshPromise) return workerHealthRefreshPromise;
  if (state.workerHealth?.data_mode !== "live") {
    $("#workerSnapshotAt").textContent = "正在读取最新快照...";
  }
  workerHealthRefreshPromise = fetchLiveWorkerHealth()
    .then((document) => {
      state.workerHealth = document;
      state.workerHealthError = "";
    })
    .catch((error) => {
      state.workerHealthError = error instanceof Error ? error.message : String(error);
      console.warn("Unable to refresh worker health", error);
    })
    .finally(() => {
      workerHealthRefreshPromise = null;
      renderWorkerHealth();
    });
  return workerHealthRefreshPromise;
}

function setWorkerHealthRefreshEnabled(enabled) {
  if (workerHealthRefreshTimer !== null) {
    window.clearInterval(workerHealthRefreshTimer);
    workerHealthRefreshTimer = null;
  }
  if (!enabled) return;
  refreshWorkerHealth();
  const refreshSeconds = Math.max(
    60,
    Number(state.manifest.live_sources?.worker_health?.refresh_seconds || 300),
  );
  workerHealthRefreshTimer = window.setInterval(() => {
    if (document.visibilityState === "visible") refreshWorkerHealth();
  }, refreshSeconds * 1000);
}

async function loadData() {
  state.manifest = await fetchJson("data/manifest.json");
  const sources = state.manifest.sources;
  [state.fullTest, state.backends, state.performance, state.workerHealth] = await Promise.all([
    fetchJson(`data/${sources.full_test}`),
    fetchJson(`data/${sources.backend_status}`),
    fetchJson(`data/${sources.performance}`),
    fetchJson(`data/${sources.worker_health}`),
  ]);
}

function renderHeader() {
  $("#generatedAt").textContent = `数据更新：${formatDate(state.manifest.generated_at)}`;
  $("#schemaVersion").textContent = state.manifest.schema;
  $("#fullRunBackend").textContent = state.fullTest.run.backend;
  $("#fullRunSha").textContent = state.fullTest.run.sha.slice(0, 12);
  $("#downloadRawCsv").href = `data/${state.manifest.downloads.full_test_csv}`;
}

function computeOperatorSummary(rows) {
  const summary = { total: rows.length, passed: 0, failed: 0, timeout: 0 };
  rows.forEach((row) => {
    const status = normalizeStatus(row.status);
    if (status in summary) summary[status] += 1;
  });
  summary.exceptions = summary.failed + summary.timeout;
  summary.passRate = summary.total ? (summary.passed / summary.total) * 100 : 0;
  return summary;
}

function renderOperatorMetrics() {
  const summary = computeOperatorSummary(state.fullTest.operators);
  const metrics = [
    ["算子总数", summary.total, ""],
    ["通过", summary.passed, ""],
    ["失败", summary.failed, ""],
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
    if (state.status === "exception") matchesStatus = status === "failed" || status === "timeout";
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
          <td>${escapeHtml(row.name)}</td>
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
          <td><strong>${escapeHtml(backend.name)}</strong><br><small>${escapeHtml(backend.profile)}</small></td>
          <td>${statusBadge(backend.state)}</td>
          <td>${statusBadge(tests.delivery)}</td>
          <td>${statusBadge(tests.compile_time)}</td>
          <td>${statusBadge(tests.pass_profile)}</td>
          <td>${statusBadge(tests.ir_serialization)}</td>
          <td><code>${escapeHtml((backend.sha || "-------").slice(0, 7))}</code></td>
          <td>${escapeHtml(formatDate(backend.tested_at))}</td>
          <td>${detail}</td>
        </tr>`;
    })
    .join("");
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

function renderWorkerHealth() {
  const health = workerHealthForDisplay(state.workerHealth);
  const poller = health.poller || {};
  const task = health.active_task;
  const container = health.container || {};
  const limits = container.limits || {};
  const stats = container.stats || {};
  const lastResult = health.last_result;
  const cpuUsage = formatCpuUsage(stats.cpu_percent, container.available_cpus ?? limits.cpus);
  const memoryUsage = stats.memory_usage || "--";

  $("#workerProfile").textContent = health.profile || "unknown";
  $("#workerId").textContent = health.worker_id || "--";
  const snapshotLabel = health.collected_at
    ? `快照 ${formatDate(health.collected_at)}（${formatElapsed(health.snapshot_age_seconds)}前）`
    : "快照 --";
  $("#workerSnapshotAt").textContent = state.workerHealthError
    ? `${snapshotLabel} · 刷新失败`
    : snapshotLabel;

  const metrics = [
    {
      label: "Worker 状态",
      value: statusBadge(health.state || "unknown"),
      detail: health.data_mode === "live" ? "实时快照" : "暂无快照",
      html: true,
    },
    {
      label: "Poller 心跳",
      value: formatElapsed(poller.heartbeat_age_seconds),
      detail: formatDate(poller.heartbeat_at),
    },
    {
      label: "发现 task ref",
      value: Number.isInteger(poller.task_ref_count) ? poller.task_ref_count : "--",
      detail: `最近轮询 ${formatDate(poller.last_poll_finished_at)}`,
    },
    {
      label: "容器 CPU",
      value: cpuUsage.utilization,
      detail: cpuUsage.ratio,
    },
    {
      label: "容器内存",
      value: memoryUsage,
      detail: "实际使用 / 可用内存",
    },
  ];
  $("#workerMetrics").innerHTML = metrics
    .map(
      (metric) => `
        <div class="metric-item">
          <span class="metric-label">${escapeHtml(metric.label)}</span>
          <span class="metric-value worker-metric-value">${metric.html ? metric.value : escapeHtml(metric.value)}</span>
          <span class="metric-detail worker-metric-detail">${escapeHtml(metric.detail)}</span>
        </div>`,
    )
    .join("");

  const activeTaskState = task
    ? "busy"
    : health.state === "healthy"
      ? "idle"
      : health.state || "unknown";
  $("#activeTaskState").innerHTML = statusBadge(activeTaskState);
  renderFactList(
    "#activeTaskDetails",
    task
      ? [
          ["阶段", task.stage],
          ["任务分支", task.branch, "code"],
          ["SHA", shortSha(task.sha), "code"],
          ["Run ID", task.run_id, "code"],
          ["Profile", task.profile],
          ["执行模式", task.execution_mode],
          ["开始时间", formatDate(task.started_at)],
          ["已运行", formatElapsed(task.elapsed_seconds)],
        ]
      : [],
    "当前没有正在执行的任务。",
  );

  const pollerStatus = poller.alive ? poller.state || "unknown" : health.state === "offline" ? "offline" : "unknown";
  $("#pollerState").innerHTML = statusBadge(pollerStatus);
  renderFactList("#pollerDetails", [
    ["进程", poller.alive ? "运行中" : "未确认"],
    ["PID", poller.pid],
    ["启动时间", formatDate(poller.started_at)],
    ["最后心跳", formatDate(poller.heartbeat_at)],
    ["心跳距今", formatElapsed(poller.heartbeat_age_seconds)],
    ["最近轮询", poller.last_poll_status || "unknown", "status"],
    ["轮询结束", formatDate(poller.last_poll_finished_at)],
    ["错误码", poller.last_error_code || "--", "code"],
  ]);

  const containerStatus = !container.available
    ? "unknown"
    : container.running
      ? "healthy"
      : "offline";
  $("#containerState").innerHTML = statusBadge(containerStatus);
  renderFactList("#containerDetails", [
    ["名称", container.name || "--", "code"],
    ["Docker 状态", container.status || "unknown", "status"],
    ["启动时间", formatDate(container.started_at)],
    ["重启次数", container.restart_count],
    [
      "OOM killed",
      typeof container.oom_killed === "boolean" ? (container.oom_killed ? "是" : "否") : "--",
    ],
    ["CPU 使用", cpuUsage.used],
    ["CPU 使用率", cpuUsage.utilization],
    ["内存使用", memoryUsage],
    ["PID 数量", stats.pids || "--"],
    ["Block I/O", stats.block_io || "--"],
    ["Network I/O", stats.network_io || "--"],
  ]);

  const storageRows = Array.isArray(health.storage) ? health.storage : [];
  $("#workerStorageRows").innerHTML = storageRows
    .map(
      (row) => `
        <tr>
          <td><strong>${escapeHtml(row.label || "--")}</strong></td>
          <td><code title="${escapeHtml(row.path || "")}">${escapeHtml(row.path || "--")}</code></td>
          <td>${formatBytes(row.directory_bytes)}</td>
          <td>${row.directory_percent === null || row.directory_percent === undefined ? "--" : `${escapeHtml(row.directory_percent)}%`}</td>
        </tr>`,
    )
    .join("");
  $("#workerStorageEmpty").hidden = storageRows.length !== 0;

  $("#lastResultState").innerHTML = statusBadge(lastResult?.status || "unknown");
  renderFactList(
    "#lastResultDetails",
    lastResult
      ? [
          ["任务分支", lastResult.branch, "code"],
          ["SHA", shortSha(lastResult.sha), "code"],
          ["Run ID", lastResult.run_id, "code"],
          ["Profile", lastResult.profile],
          ["结果", lastResult.status, "status"],
          ["退出码", lastResult.exit_code],
          ["发布退出码", lastResult.publish_status],
          ["错误码", lastResult.failure_code || "--", "code"],
          ["开始时间", formatDate(lastResult.started_at)],
          ["结束时间", formatDate(lastResult.finished_at)],
        ]
      : [],
    "还没有可展示的任务结果。",
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
  $$(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      $$(".tab-button").forEach((item) => item.classList.toggle("active", item === button));
      $$(".view").forEach((view) => view.classList.remove("active"));
      $(`#${button.dataset.view}View`).classList.add("active");
      setWorkerHealthRefreshEnabled(button.dataset.view === "worker");
    });
  });

  document.addEventListener("visibilitychange", () => {
    const workerButton = $('.tab-button[data-view="worker"]');
    if (document.visibilityState === "visible" && workerButton?.classList.contains("active")) {
      refreshWorkerHealth();
    }
  });

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
  try {
    await loadData();
    renderHeader();
    renderOperatorMetrics();
    populateStageFilter();
    renderOperators();
    renderBackends();
    renderPerformance();
    renderWorkerHealth();
    bindEvents();
  } catch (error) {
    $("#operatorsView").classList.remove("active");
    $("#performanceView").classList.remove("active");
    $("#workerView").classList.remove("active");
    $("#loadError").hidden = false;
    $("#loadErrorMessage").textContent = error instanceof Error ? error.message : String(error);
  }
}

initialize();
