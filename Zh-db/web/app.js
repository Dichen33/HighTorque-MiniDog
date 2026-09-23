const state = {
  ws: null,
  runs: [],
  run: null,
  metricInfo: {},
  activeRunId: null,
  filter: "all",
  query: "",
  xAxis: "iteration",
};

const metricOrder = [
  "mean_reward",
  "mean_episode_length",
  "error_vel_xy",
  "error_vel_yaw",
  "reward_track_lin_vel",
  "reward_track_yaw",
  "reward_height",
  "reward_orientation",
  "reward_action_rate",
  "reward_joint_torques",
  "loss_value",
  "loss_surrogate",
  "loss_entropy",
  "loss_learning_rate",
  "policy_mean_std",
  "total_fps",
  "learning_time",
  "collection_time",
  "termination_time_out",
  "termination_nan_detection",
  "termination_bad_orientation",
  "termination_base_ground_contact",
  "curriculum_max",
];

const palette = ["#2f6fed", "#10a37f", "#f97316", "#8b5cf6", "#ef4444", "#0ea5e9", "#64748b"];

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${protocol}://${location.host}/ws`);

  state.ws.addEventListener("open", () => {
    setStatus(true, "WebSocket 已连接");
    send({ type: "list_runs" });
  });

  state.ws.addEventListener("close", () => {
    setStatus(false, "连接断开，正在重连");
    setTimeout(connect, 1000);
  });

  state.ws.addEventListener("message", (event) => {
    handleMessage(JSON.parse(event.data));
  });
}

function send(message) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(message));
  }
}

function handleMessage(message) {
  if (message.type === "runs") {
    state.runs = message.runs || [];
    state.metricInfo = message.metric_info || {};
    renderFilters();
    renderRuns();
    if (!state.activeRunId && state.runs.length) {
      selectRun(state.runs[state.runs.length - 1].id);
    }
  }

  if (message.type === "run") {
    state.run = normalizeRun(message.run);
    if (state.run) {
      state.metricInfo = state.run.metric_info || state.metricInfo;
      renderFilters();
      renderRun();
    }
  }

  if (message.type === "run_updated") {
    state.runs = message.runs || state.runs;
    renderRuns();
    if (state.activeRunId && message.run && message.run.id === state.activeRunId) {
      state.run = normalizeRun(message.run);
      renderRun();
    }
  }

  if (message.type === "run_deleted") {
    state.runs = message.runs || state.runs;
    state.metricInfo = message.metric_info || state.metricInfo;
    if (message.run_id && message.run_id === state.activeRunId) {
      state.activeRunId = null;
      state.run = null;
      document.querySelector("#runTitle").textContent = "请选择一条训练";
      document.querySelector("#runMeta").textContent = "等待数据";
      document.querySelector("#configBox").textContent = "{}";
      document.querySelector("#charts").innerHTML = "";
      document.querySelector("#analysisBox").className = "analysis empty";
      document.querySelector("#analysisBox").textContent = "点击“生成训练总结”，这里会根据历史曲线生成中文诊断。";
      document.querySelector("#lastIteration").textContent = "-";
      document.querySelector("#lastStep").textContent = "-";
      document.querySelector("#lastReward").textContent = "-";
      document.querySelector("#lastVelErr").textContent = "-";
    }
    renderFilters();
    renderRuns();
  }

  if (message.type === "analysis") {
    const box = document.querySelector("#analysisBox");
    box.classList.remove("empty");
    if (message.error) {
      box.textContent = message.error;
    } else if (message.summary) {
      renderAnalysis(message);
    } else {
      box.textContent = `来源：${message.source}\n\n${message.text}`;
    }
  }

  if (message.type === "export_csv") {
    renderExportResult(message);
  }
}

function normalizeRun(run) {
  if (!run) return null;
  const config = run.config || {};
  const rolloutSize = Number(config.rollout_size || (config.num_envs && config.steps_per_env ? config.num_envs * config.steps_per_env : 0));
  const metrics = (run.metrics || []).map((row, index) => {
    const sampleStep = numberOr(row.sample_step, row.step, 0);
    const iteration = typeof row.iteration === "number"
      ? row.iteration
      : rolloutSize > 0 && typeof row.step === "number"
        ? row.step / rolloutSize
        : index + 1;
    return { ...row, sample_step: sampleStep, iteration };
  });
  return { ...run, metrics };
}

function setStatus(online, text) {
  const dot = document.querySelector("#statusDot");
  dot.className = online ? "online" : "offline";
  document.querySelector("#statusText").textContent = text;
}

function renderFilters() {
  const select = document.querySelector("#metricFilter");
  const modules = new Set(Object.values(state.metricInfo).map((info) => info[1]));
  select.innerHTML = `<option value="all">全部模块</option>`;
  [...modules].sort().forEach((module) => {
    const option = document.createElement("option");
    option.value = module;
    option.textContent = module;
    select.appendChild(option);
  });
  select.value = state.filter;
}

function renderRuns() {
  const list = document.querySelector("#runList");
  const query = state.query.trim().toLowerCase();
  const runs = state.runs.filter((run) => `${run.id} ${run.name}`.toLowerCase().includes(query));
  list.innerHTML = "";

  if (!runs.length) {
    list.innerHTML = `<div class="empty-list">没有匹配的训练记录</div>`;
    return;
  }

  runs.forEach((run, index) => {
    const item = document.createElement("div");
    item.className = `run-item ${run.id === state.activeRunId ? "active" : ""}`;
    item.setAttribute("role", "button");
    item.tabIndex = 0;
    const iteration = numberOr(run.last_iteration, deriveListIteration(run), 0);
    item.innerHTML = `
      <div class="run-row">
        <strong>${escapeHtml(run.name || run.id)}</strong>
        <span class="status-pill">${escapeHtml(run.status || "未知")}</span>
      </div>
      <span>第${index + 1} 次 · ${escapeHtml(run.created_at || "")}</span>
      <div class="run-metrics">
        <b>${fmt(iteration)}</b><small>轮次</small>
        <b>${fmt(run.last_reward)}</b><small>奖励</small>
      </div>
      <button class="run-delete" type="button">删除</button>
    `;
    item.addEventListener("click", (event) => {
      if (event.target && event.target.closest(".run-delete")) return;
      selectRun(run.id);
    });
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") selectRun(run.id);
    });
    item.querySelector(".run-delete").addEventListener("click", (event) => {
      event.stopPropagation();
      deleteRun(run.id, run.name || run.id);
    });
    list.appendChild(item);
  });
}

function deriveListIteration(run) {
  if (run.last_step && run.config && run.config.rollout_size) {
    return run.last_step / run.config.rollout_size;
  }
  return run.steps || 0;
}

function selectRun(runId) {
  state.activeRunId = runId;
  document.querySelector("#analysisBox").className = "analysis empty";
  document.querySelector("#analysisBox").textContent = "点击“生成训练总结”，这里会根据历史曲线生成中文诊断。";
  renderRuns();
  send({ type: "get_run", run_id: runId });
}

function renderRun() {
  const run = state.run;
  if (!run) return;
  const config = run.config || {};
  document.querySelector("#runTitle").textContent = run.name || run.id;
  document.querySelector("#runMeta").textContent = `${run.status || "未知"} · ${run.created_at || ""} · ${config.experiment_name || "默认实验"}`;
  document.querySelector("#configBox").textContent = JSON.stringify(config, null, 2);
  renderStats(run);
  renderCharts(run);
}

function renderStats(run) {
  const metrics = run.metrics || [];
  const last = metrics[metrics.length - 1] || {};
  document.querySelector("#lastIteration").textContent = fmt(last.iteration);
  document.querySelector("#lastStep").textContent = fmt(last.sample_step);
  document.querySelector("#lastReward").textContent = fmt(last.mean_reward);
  document.querySelector("#lastVelErr").textContent = fmt(last.error_vel_xy);
}

function renderCharts(run) {
  const charts = document.querySelector("#charts");
  const metrics = run.metrics || [];
  charts.innerHTML = "";

  if (!metrics.length) {
    charts.innerHTML = `<div class="empty-chart">当前训练没有曲线数据</div>`;
    return;
  }

  const numericKeys = discoverNumericKeys(metrics);
  const knownKeys = metricOrder.filter((key) => numericKeys.includes(key));
  const extraKeys = numericKeys
    .filter((key) => !metricOrder.includes(key))
    .sort((a, b) => groupKey(a).localeCompare(groupKey(b)));
  const available = [...knownKeys, ...extraKeys];

  available.forEach((key, index) => {
    const info = state.metricInfo[key] || [key, "其他", key];
    if (state.filter !== "all" && info[1] !== state.filter) return;
    const points = metrics
      .filter((row) => typeof row[key] === "number" && typeof row[state.xAxis] === "number")
      .map((row) => ({ x: Number(row[state.xAxis]), y: Number(row[key]), row }));
    if (!points.length) return;

    const latest = points[points.length - 1].y;
    const card = document.createElement("article");
    card.className = "chart-card";
    card.innerHTML = `
      <div class="chart-head">
        <div>
          <h3>${escapeHtml(info[0])}</h3>
          <p>${escapeHtml(info[2] || key)}</p>
        </div>
        <div class="chart-side">
          <span class="module-tag">${escapeHtml(info[1])}</span>
          <strong>${fmt(latest)}</strong>
        </div>
      </div>
      <canvas height="230"></canvas>
      <div class="chart-tooltip" hidden></div>
    `;
    charts.appendChild(card);
    drawChart(card, card.querySelector("canvas"), points, palette[index % palette.length], axisLabel(), info[0]);
  });
}

function discoverNumericKeys(metrics) {
  const keys = [];
  const excluded = new Set(["step", "sample_step", "iteration"]);
  metrics.forEach((row) => {
    Object.entries(row).forEach(([key, value]) => {
      if (!excluded.has(key) && typeof value === "number" && !keys.includes(key)) {
        keys.push(key);
      }
    });
  });
  return keys;
}

function renderAnalysis(message) {
  const box = document.querySelector("#analysisBox");
  const summary = message.summary;
  const score = Number(summary.health_score || 0);
  const scoreColor = score >= 78 ? "#10a37f" : score >= 55 ? "#f59e0b" : "#ef4444";
  box.className = "analysis";
  box.innerHTML = `
    <div class="summary-report">
      <div class="summary-score">
        <div class="score-ring" style="--score:${score}%; background: conic-gradient(${scoreColor} ${score}%, #e5e7eb 0);">${score}</div>
        <div>
          <strong>${escapeHtml(summary.verdict || "未知")}</strong>
          <div>来源：${escapeHtml(message.source || "规则分析")}</div>
          <div>${escapeHtml(summary.title || "")}</div>
        </div>
      </div>
      ${renderKeyNumbers(summary.key_numbers || [])}
      ${renderListSection("优势", summary.strengths || [])}
      ${renderListSection("风险", summary.risks || [])}
      ${renderListSection("建议", summary.suggestions || [])}
      ${message.text ? `<div class="summary-section"><h3>完整总结</h3><div class="ai-text">${escapeHtml(message.text)}</div></div>` : ""}
    </div>
  `;
}

function exportCurrentRun() {
  if (!state.activeRunId) return;
  const box = document.querySelector("#exportBox");
  box.className = "export-box";
  box.textContent = "正在从本地 JSON 导出当前曲线数据...";
  send({
    type: "export_csv",
    run_id: state.activeRunId,
    metric_filter: state.filter,
  });
}

function deleteRun(runId, runName) {
  if (!confirm(`确认删除训练记录「${runName}」？删除后 JSON 和导出的 CSV 都会被清理。`)) return;
  send({ type: "delete_run", run_id: runId });
}
function renderExportResult(message) {
  const box = document.querySelector("#exportBox");
  box.className = `export-box ${message.ok ? "" : "error"}`;
  if (!message.ok) {
    box.textContent = `导出失败：${message.error || "未知错误"}`;
    return;
  }
  box.innerHTML = `
    <strong>CSV 已导出</strong>
    <span>文件：${escapeHtml(message.csv_path)}</span>
    <span>行数：${escapeHtml(message.row_count)}，列数：${escapeHtml((message.columns || []).length)}</span>
    <span>源 JSON SHA256：${escapeHtml(shortHash(message.source_json_sha256))}</span>
    <span>CSV SHA256：${escapeHtml(shortHash(message.csv_sha256))}</span>
    <span>哈希链：${escapeHtml(message.hash_chain_status || "not_available")}</span>
    <span>校验信息：${escapeHtml(message.meta_path)}</span>
  `;
}

function renderKeyNumbers(items) {
  if (!items.length) return "";
  return `
    <div class="summary-section">
      <h3>关键指标</h3>
      <div class="key-grid">
        ${items.map((item) => `
          <div class="key-item">
            <span>${escapeHtml(item.name)}</span>
            <strong>${escapeHtml(item.value)}</strong>
          </div>
        `).join("")}
      </div>
    </div>
  `;
}

function renderListSection(title, items) {
  if (!items.length) return "";
  return `
    <div class="summary-section">
      <h3>${escapeHtml(title)}</h3>
      <ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
    </div>
  `;
}

function drawChart(card, canvas, points, color, axisLabelText, metricLabel) {
  const tooltip = card.querySelector(".chart-tooltip");
  const view = { hover: -1, pinned: -1 };
  canvas.style.cursor = "crosshair";
  card.style.position = "relative";

  function computeBounds() {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    let minY = Math.min(...ys);
    let maxY = Math.max(...ys);
    if (minY === maxY) {
      minY -= 1;
      maxY += 1;
    }
    const yPad = (maxY - minY) * 0.12;
    return {
      minX,
      maxX,
      minY: minY - yPad,
      maxY: maxY + yPad,
      pad: { left: 54, right: 18, top: 18, bottom: 34 },
    };
  }

  function draw(activeIndex = view.hover >= 0 ? view.hover : view.pinned) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(320, Math.floor(rect.width * dpr));
    canvas.height = Math.floor(230 * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const width = canvas.width / dpr;
    const height = canvas.height / dpr;
    const { minX, maxX, minY, maxY, pad } = computeBounds();

    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#e8ebf0";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#697386";
    ctx.font = "12px Microsoft YaHei, Segoe UI, sans-serif";

    for (let i = 0; i <= 4; i++) {
      const y = pad.top + ((height - pad.top - pad.bottom) * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(width - pad.right, y);
      ctx.stroke();
      const value = maxY - ((maxY - minY) * i) / 4;
      ctx.fillText(shortNum(value), 8, y + 4);
    }

    const gradient = ctx.createLinearGradient(0, pad.top, 0, height - pad.bottom);
    gradient.addColorStop(0, alpha(color, 0.2));
    gradient.addColorStop(1, alpha(color, 0));

    ctx.beginPath();
    points.forEach((point, index) => {
      const x = scale(point.x, minX, maxX, pad.left, width - pad.right);
      const y = scale(point.y, minY, maxY, height - pad.bottom, pad.top);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(scale(points[points.length - 1].x, minX, maxX, pad.left, width - pad.right), height - pad.bottom);
    ctx.lineTo(scale(points[0].x, minX, maxX, pad.left, width - pad.right), height - pad.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = scale(point.x, minX, maxX, pad.left, width - pad.right);
      const y = scale(point.y, minY, maxY, height - pad.bottom, pad.top);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const last = points[points.length - 1];
    const lx = scale(last.x, minX, maxX, pad.left, width - pad.right);
    const ly = scale(last.y, minY, maxY, height - pad.bottom, pad.top);
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(lx, ly, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    if (activeIndex >= 0 && points[activeIndex]) {
      const point = points[activeIndex];
      const x = scale(point.x, minX, maxX, pad.left, width - pad.right);
      const y = scale(point.y, minY, maxY, height - pad.bottom, pad.top);
      ctx.save();
      ctx.strokeStyle = "rgba(47, 111, 237, 0.35)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, height - pad.bottom);
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      tooltip.hidden = false;
      tooltip.innerHTML = `
        <strong>${escapeHtml(metricLabel)}</strong>
        <span>${escapeHtml(axisLabelText)}: ${escapeHtml(fmt(point.x))}</span>
        <span>Y: ${escapeHtml(fmt(point.y))}</span>
      `;
      const box = tooltip.getBoundingClientRect();
      const px = Math.min(width - box.width - 8, Math.max(8, x + 12));
      const py = Math.min(height - box.height - 8, Math.max(8, y - box.height - 12));
      tooltip.style.left = `${px}px`;
      tooltip.style.top = `${py}px`;
    } else {
      tooltip.hidden = true;
    }

    ctx.fillStyle = "#697386";
    ctx.fillText(`${axisLabelText} ${shortNum(minX)}`, pad.left, height - 10);
    ctx.fillText(shortNum(maxX), width - 62, height - 10);
  }

  function nearestIndex(event) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const x = ((event.clientX - rect.left) * canvas.width) / rect.width / dpr;
    const y = ((event.clientY - rect.top) * canvas.height) / rect.height / dpr;
    const { minX, maxX, minY, maxY, pad } = computeBounds();
    let best = 0;
    let bestDist = Number.POSITIVE_INFINITY;
    points.forEach((point, index) => {
      const px = scale(point.x, minX, maxX, pad.left, canvas.width / dpr - pad.right);
      const py = scale(point.y, minY, maxY, canvas.height / dpr - pad.bottom, pad.top);
      const dist = (px - x) ** 2 + (py - y) ** 2;
      if (dist < bestDist) {
        best = index;
        bestDist = dist;
      }
    });
    return best;
  }

  canvas.addEventListener("mousemove", (event) => {
    view.hover = nearestIndex(event);
    draw();
  });
  canvas.addEventListener("mouseleave", () => {
    view.hover = -1;
    draw();
  });
  canvas.addEventListener("click", (event) => {
    const index = nearestIndex(event);
    view.pinned = view.pinned === index ? -1 : index;
    draw();
  });

  draw();
}

function setAxis(axis) {
  state.xAxis = axis;
  document.querySelector("#axisIterationBtn").classList.toggle("active", axis === "iteration");
  document.querySelector("#axisSampleBtn").classList.toggle("active", axis === "sample_step");
  renderRun();
}

function axisLabel() {
  return state.xAxis === "iteration" ? "轮次" : "样本";
}

function groupKey(key) {
  const info = state.metricInfo[key] || ["", "其他"];
  return `${info[1]}:${key}`;
}

function numberOr(...values) {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return values[values.length - 1];
}

function fmt(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(3);
}

function shortNum(value) {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(1)}k`;
  if (Math.abs(value) >= 10) return value.toFixed(0);
  return value.toFixed(2);
}

function alpha(hex, opacity) {
  const value = hex.replace("#", "");
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

function scale(value, minIn, maxIn, minOut, maxOut) {
  if (maxIn === minIn) return (minOut + maxOut) / 2;
  return minOut + ((value - minIn) / (maxIn - minIn)) * (maxOut - minOut);
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[ch]);
}

function shortHash(value) {
  const text = String(value || "");
  return text.length > 18 ? `${text.slice(0, 10)}...${text.slice(-8)}` : text;
}

document.querySelector("#refreshBtn").addEventListener("click", () => send({ type: "list_runs" }));
document.querySelector("#runSearch").addEventListener("input", (event) => {
  state.query = event.target.value;
  renderRuns();
});
document.querySelector("#analyzeBtn").addEventListener("click", () => {
  if (!state.activeRunId) return;
  const box = document.querySelector("#analysisBox");
  box.className = "analysis";
  box.textContent = "正在读取历史曲线并生成训练总结...";
  send({ type: "analyze", run_id: state.activeRunId });
});
document.querySelector("#exportBtn").addEventListener("click", exportCurrentRun);
document.querySelector("#metricFilter").addEventListener("change", (event) => {
  state.filter = event.target.value;
  renderRun();
});
document.querySelector("#axisIterationBtn").addEventListener("click", () => setAxis("iteration"));
document.querySelector("#axisSampleBtn").addEventListener("click", () => setAxis("sample_step"));
window.addEventListener("resize", () => renderRun());

connect();
