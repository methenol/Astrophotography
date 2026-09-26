/* AstroPhoto Studio front-end (vanilla JS, no build step) */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { } },
};

const S = {
  system: null, folder: null, status: null, frames: [], params: {}, job: null,
  view: "after", sort: { k: "idx", dir: 1 }, split: 0.5,
  zoom: { s: 1, x: 0, y: 0 }, imgW: 0, imgH: 0, previewSeq: 0,
};

async function api(path, opts = {}) {
  const o = { headers: {}, ...opts };
  if (o.body && typeof o.body !== "string") { o.body = JSON.stringify(o.body); o.headers["Content-Type"] = "application/json"; }
  const r = await fetch(path, o);
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.detail || msg; } catch { }
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r;
}

function toast(msg, err = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast" + (err ? " err" : ""); t.hidden = false;
  clearTimeout(t._h); t._h = setTimeout(() => (t.hidden = true), err ? 6000 : 3000);
}
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : (typeof v === "number" ? v.toFixed(d) : v);

/* ------------------------------------------------------------ init */
async function init() {
  S.system = await api("/api/system");
  $("#versionBadge").textContent = "v" + S.system.version;
  const dev = S.system.devices;
  const gpu = (dev.gpus || [])[0];
  $("#deviceBadge").textContent = gpu ? `⚡ ${gpu.name}${gpu.vram_gb ? " · " + gpu.vram_gb + " GB" : ""}` : "CPU only";
  const ds = $("#sp-device");
  ds.innerHTML = `<option value="auto">auto (${dev.default})</option>` +
    (dev.gpus || []).map(g => `<option value="${g.id}">${g.name}</option>`).join("") + `<option value="cpu">CPU</option>`;
  buildParamUI();
  buildPresets();
  bindUI();
  await loadDatasets();
  const last = store.get("lastFolder");
  if (last && [...$("#datasetSelect").options].some(o => o.value === last)) {
    $("#datasetSelect").value = last; openDataset(last);
  }
}

async function loadDatasets() {
  const r = await api("/api/datasets");
  const sel = $("#datasetSelect");
  if (!r.datasets.length) { sel.innerHTML = `<option value="">No FITS folders under ${r.root}</option>`; return; }
  sel.innerHTML = `<option value="">Select a dataset…</option>` + r.datasets.map(d => {
    const tag = d.cached.denoised ? " ✓✓✓" : d.cached.stacked ? " ✓✓" : d.cached.analysed ? " ✓" : "";
    return `<option value="${d.path}">${d.object || d.name} — ${d.n_fits}× ${d.exptime || "?"}s ${d.filter || ""} (${d.total_min} min)${tag}</option>`;
  }).join("");
}

/* ------------------------------------------------------------ params UI */
function currentDefaults() { return { ...S.system.defaults }; }

function buildParamUI() {
  const groups = {};
  for (const p of S.system.param_spec) (groups[p.group] ||= []).push(p);
  const host = $("#paramGroups");
  host.innerHTML = "";
  Object.entries(groups).forEach(([g, items], gi) => {
    const det = document.createElement("details");
    det.className = "pgroup"; det.open = gi < 4;
    det.innerHTML = `<summary>${g}</summary><div class="pbody"></div>`;
    const body = $(".pbody", det);
    for (const p of items) {
      const el = document.createElement("div");
      el.className = "param";
      if (p.type === "range") {
        el.innerHTML = `<div class="plabel"><span>${p.label}</span><output></output></div>
          <input type="range" min="${p.min}" max="${p.max}" step="${p.step}" data-key="${p.key}">`;
      } else if (p.type === "bool") {
        el.innerHTML = `<label class="plabel check" style="justify-content:flex-start;gap:8px"><input type="checkbox" data-key="${p.key}"> ${p.label}</label>`;
      } else {
        el.innerHTML = `<div class="plabel"><span>${p.label}</span></div><select data-key="${p.key}">${p.options.map(o => `<option>${o}</option>`).join("")}</select>`;
      }
      body.appendChild(el);
    }
    host.appendChild(det);
  });
  $$("#paramGroups [data-key]").forEach(inp => {
    const ev = inp.type === "range" ? "input" : "change";
    inp.addEventListener(ev, () => {
      const k = inp.dataset.key;
      S.params[k] = inp.type === "checkbox" ? inp.checked : inp.type === "range" ? parseFloat(inp.value) : inp.value;
      syncParamOutputs();
      saveParams();
      schedulePreview();
    });
  });
}

function applyParamsToUI() {
  $$("#paramGroups [data-key]").forEach(inp => {
    const v = S.params[inp.dataset.key];
    if (inp.type === "checkbox") inp.checked = !!v; else inp.value = v;
  });
  syncParamOutputs();
}
function syncParamOutputs() {
  $$("#paramGroups input[type=range]").forEach(inp => { inp.parentElement.querySelector("output").textContent = (+inp.value).toFixed(inp.step < 0.05 ? 3 : 2); });
}
function saveParams() { if (S.folder) store.set("params:" + S.folder, S.params); }

function buildPresets() {
  const bar = $("#presets");
  bar.innerHTML = Object.keys(S.system.presets).map(n => `<button data-preset="${n}">${n}</button>`).join("");
  $$("button", bar).forEach(b => b.onclick = () => {
    S.params = { ...currentDefaults(), ...S.system.presets[b.dataset.preset] };
    applyParamsToUI(); saveParams(); schedulePreview(0);
    toast(`Preset: ${b.dataset.preset}`);
  });
}

/* ------------------------------------------------------------ dataset */
async function openDataset(folder) {
  if (!folder) return;
  try {
    const r = await api("/api/open", { method: "POST", body: { folder } });
    S.folder = folder; store.set("lastFolder", folder);
    if (typeof Explore !== "undefined") Explore.reset();
    S.status = r.status; S.frames = r.frames;
    S.params = { ...currentDefaults(), ...store.get("params:" + folder, {}) };
    applyParamsToUI();
    $("#datasetTitle").textContent = `${r.status.object || ""} · ${folder}`;
    const nb = r.status.narrowband;
    $("#datasetInfo").innerHTML = `<span>Object</span><b>${r.status.object || "–"}</b><span>Filter</span><b>${r.status.filter || "–"} ${nb ? "(dual-band → HOO)" : "(broadband RGB)"}</b><span>Frames</span><b>${r.status.n_files}</b><span>Cache</span><b class="small">${r.status.workdir.split("/").slice(-2).join("/")}</b>`;
    updateSteps(); renderFrames();
    if (r.status.stacked) schedulePreview(0); else showPlaceholder(true);
    refreshExports(); refreshDiag();
  } catch (e) { toast(e.message, true); }
}

function updateSteps() {
  const st = S.status || {};
  $("#step-analyse").classList.toggle("done", !!st.analysed);
  $("#step-stack").classList.toggle("done", !!st.stacked);
  $("#step-denoise").classList.toggle("done", !!st.denoised);
}

/* ------------------------------------------------------------ settings from an experiment */
let EXPERIMENTS = [];
async function loadExperimentChoices() {
  try { EXPERIMENTS = (await api("/api/lab/best")).studies; } catch { return; }
  const sel = $("#sp-experiment"), cur = sel.value;
  const f = v => typeof v === "number" ? +v.toPrecision(4) : v;
  sel.innerHTML = `<option value="">—</option>` + EXPERIMENTS.map(e =>
    `<option value="${e.id}">${e.name} · ${e.task_label} · trial #${e.trial} (${e.objectives.map((o, i) => `${o.metric} ${f(e.values[i])}`).join(", ")})${e.state === "running" ? " · still running" : ""}</option>`).join("");
  if (EXPERIMENTS.some(e => e.id === cur)) sel.value = cur;
}
function applyExperiment(id) {
  const e = EXPERIMENTS.find(x => x.id === id);
  if (!e) { $("#sp-experiment-info").textContent = ""; return; }
  const set = [];
  for (const [k, v] of Object.entries(e.settings)) {
    const el = $("#sp-" + k); if (!el) continue;
    if (el.type === "checkbox") el.checked = !!v; else el.value = String(v);
    if (k === "deconv_method") el.dispatchEvent(new Event("change", { bubbles: true }));
    set.push(`${k} = ${v}`);
  }
  const other = Object.entries(e.not_pipeline || {}).map(([k, v]) => `${k} = ${v}`);
  $("#sp-experiment-info").innerHTML = `Applied trial #${e.trial} of “${e.name}” (${e.criterion}) on ${e.dataset}: ${set.join(", ")}.` +
    (other.length ? ` Not pipeline settings (not applied): ${other.join(", ")}.` : "") + " Re-run the step to use them.";
  toast(`Settings of “${e.name}” applied`);
}
document.addEventListener("change", e => { if (e.target.id === "sp-experiment") applyExperiment(e.target.value); });

/* ------------------------------------------------------------ jobs */
// ImageMM options only matter for the ImageMM restoration
document.addEventListener("change", e => {
  if (e.target.id === "sp-deconv_method") $("#imagemm-opts").hidden = e.target.value !== "imagemm";
});
function stackParams() {
  const g = id => $("#sp-" + id);
  return {
    mode: g("mode").value, scale: parseFloat(g("scale").value), sensitivity: parseFloat(g("sensitivity").value),
    sigma_low: parseFloat(g("sigma_low").value), sigma_high: parseFloat(g("sigma_high").value),
    local_norm: g("local_norm").checked, denoise_iters: parseInt(g("denoise_iters").value), device: g("device").value,
    deconv_method: g("deconv_method").value, ai_deconvolution: g("deconv_method").value !== "none",
    imagemm_r: parseInt(g("imagemm_r").value), imagemm_sigma: parseFloat(g("imagemm_sigma").value) || 0,
    imagemm_robust: g("imagemm_robust").checked, imagemm_epsilon: parseFloat(g("imagemm_epsilon").value) || 1e-6,
    imagemm_stop: g("imagemm_stop").value,
    imagemm_delta: parseFloat(g("imagemm_delta").value) || 2, imagemm_kappa: parseFloat(g("imagemm_kappa").value) || 2,
    imagemm_max_iters: parseInt(g("imagemm_max_iters").value) || 1000, imagemm_psf: g("imagemm_psf").value,
    imagemm_groups: parseInt(g("imagemm_groups").value) || 0, imagemm_accelerate: g("imagemm_accelerate").checked,
    imagemm_n2n: g("imagemm_n2n").checked, network_groups: parseInt(g("network_groups").value) || 0,
  };
}
function exportOpts() {
  return { quality: parseInt($("#ex-quality").value), upscale: parseFloat($("#ex-upscale").value), tiff: $("#ex-tiff").checked };
}

async function startJob(kind) {
  if (!S.folder) return toast("Open a dataset first", true);
  try {
    const job = await api("/api/jobs", { method: "POST", body: { kind, folder: S.folder, params: S.params, stack_params: stackParams(), export: exportOpts() } });
    S.job = job; $("#jobCard").hidden = false; setJobButtons(true);
    const stepEl = { analyse: "#step-analyse", stack: "#step-stack", denoise: "#step-denoise" }[kind];
    if (stepEl) $(stepEl).classList.add("running");
    pollJob();
  } catch (e) { toast(e.message, true); }
}
function setJobButtons(busy) { $$("[data-job], #exportBtn, #exportBtn2").forEach(b => b.disabled = busy); }

async function pollJob() {
  if (!S.job) return;
  const j = await api(`/api/jobs/${S.job.id}`);
  $("#progBar").style.width = (j.progress * 100).toFixed(1) + "%";
  $("#progMsg").textContent = `${j.message} · ${j.elapsed}s`;
  $("#progLog").innerHTML = j.log.map(([t, m]) => `<div>${t.toFixed(0).padStart(4)}s  ${m}</div>`).join("");
  $("#progLog").scrollTop = 1e9;
  if (j.state === "running" || j.state === "queued") { setTimeout(pollJob, 700); return; }
  setJobButtons(false);
  $$(".steps li").forEach(li => li.classList.remove("running"));
  S.job = null;
  if (j.state === "done") {
    toast(`${j.kind} finished in ${j.elapsed}s`);
    const r = await api("/api/open", { method: "POST", body: { folder: S.folder } });
    S.status = r.status; S.frames = r.frames; updateSteps(); renderFrames();
    if (S.status.stacked) schedulePreview(0);
    if (j.kind === "export" || j.kind === "all") { refreshExports(); switchTab("export"); }
    refreshDiag();
  } else if (j.state === "error") {
    toast(j.message, true); console.error(j.traceback);
    // keep the failure on screen: the stage it reached, the traceback and the device memory
    $("#jobCard").hidden = false;
    $("#progMsg").innerHTML = `<b style="color:var(--bad)">${esc(j.message)}</b> · ${j.elapsed}s` +
      (j.stage ? `<div class="muted small">Last stage: ${esc(j.stage)}</div>` : "") +
      `<details open class="small" style="margin-top:6px"><summary>Traceback (also in job_errors.log in the session folder)</summary>` +
      `<pre class="log" style="max-height:320px;user-select:text">${esc((j.device_state ? j.device_state + "\n\n" : "") + (j.traceback || ""))}</pre></details>`;
  } else toast("Cancelled");
}

/* ------------------------------------------------------------ preview */
let previewTimer = null;
function schedulePreview(delay = 350) {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(requestPreview, delay);
}
function showPlaceholder(v) { $("#placeholder").style.display = v ? "flex" : "none"; }

async function fetchImage(which) {
  const size = $("#previewSize").value;
  const r = await fetch("/api/preview", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder: S.folder, params: S.params, which, size: size === "full" ? null : parseInt(size) }) });
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail; } catch { } throw new Error(m); }
  return { url: URL.createObjectURL(await r.blob()), t: r.headers.get("X-Render-Time"), w: +r.headers.get("X-Width"), h: +r.headers.get("X-Height") };
}

async function requestPreview() {
  if (!S.folder || !S.status?.stacked) return;
  const seq = ++S.previewSeq;
  $("#spinner").hidden = false;
  try {
    const after = await fetchImage("after");
    if (seq !== S.previewSeq) return;
    const first = !$("#imgAfter").src;
    const sizeChanged = after.w !== S.imgW;
    $("#imgAfter").src = after.url;
    S.imgW = after.w; S.imgH = after.h;
    $("#renderInfo").textContent = `${after.w}×${after.h}px · rendered in ${after.t}s`;
    showPlaceholder(false);
    if (S.view !== "after") { const b = await fetchImage("before"); $("#imgBefore").src = b.url; }
    $("#imgBefore").style.width = S.imgW + "px"; $("#imgBefore").style.height = S.imgH + "px";
    if (first || sizeChanged) fitView();
    layoutSplit();
  } catch (e) { toast(e.message, true); }
  finally { if (seq === S.previewSeq) $("#spinner").hidden = true; }
}

/* ------------------------------------------------------------ viewer zoom/pan/split */
function applyZoom() {
  const z = S.zoom;
  $("#stage").style.transform = `translate(${z.x}px, ${z.y}px) scale(${z.s})`;
}
function fitView() {
  const c = $("#canvas").getBoundingClientRect();
  if (!S.imgW) return;
  const s = Math.min(c.width / S.imgW, c.height / S.imgH) * 0.98;
  S.zoom = { s, x: (c.width - S.imgW * s) / 2, y: (c.height - S.imgH * s) / 2 };
  applyZoom();
}
function zoomAt(factor, cx, cy) {
  const z = S.zoom;
  const ns = Math.min(8, Math.max(0.05, z.s * factor));
  z.x = cx - (cx - z.x) * (ns / z.s); z.y = cy - (cy - z.y) * (ns / z.s); z.s = ns;
  applyZoom();
}
function layoutSplit() {
  const wrap = $("#beforeWrap"), handle = $("#splitHandle");
  if (S.view === "after") { wrap.style.display = "none"; handle.hidden = true; return; }
  wrap.style.display = "block";
  const w = S.view === "before" ? S.imgW : S.imgW * S.split;
  wrap.style.width = w + "px"; wrap.style.height = S.imgH + "px";
  handle.hidden = S.view !== "split"; handle.style.left = w + "px";
}

function bindViewer() {
  const c = $("#canvas");
  c.addEventListener("wheel", e => { e.preventDefault(); const r = c.getBoundingClientRect(); zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top); }, { passive: false });
  let drag = null;
  c.addEventListener("mousedown", e => {
    const r = c.getBoundingClientRect();
    if (S.view === "split") {
      const ix = (e.clientX - r.left - S.zoom.x) / S.zoom.s;
      if (Math.abs(ix - S.imgW * S.split) < 12 / S.zoom.s) { drag = { split: true }; return; }
    }
    drag = { x: e.clientX, y: e.clientY, zx: S.zoom.x, zy: S.zoom.y }; c.classList.add("dragging");
  });
  window.addEventListener("mousemove", e => {
    if (!drag) return;
    if (drag.split) {
      const r = c.getBoundingClientRect();
      S.split = Math.min(1, Math.max(0, ((e.clientX - r.left - S.zoom.x) / S.zoom.s) / S.imgW)); layoutSplit(); return;
    }
    S.zoom.x = drag.zx + e.clientX - drag.x; S.zoom.y = drag.zy + e.clientY - drag.y; applyZoom();
  });
  window.addEventListener("mouseup", () => { drag = null; c.classList.remove("dragging"); });
  c.addEventListener("dblclick", fitView);
  window.addEventListener("resize", fitView);
  $("#zoomFit").onclick = fitView;
  $("#zoom100").onclick = () => { const r = c.getBoundingClientRect(); zoomAt(1 / S.zoom.s, r.width / 2, r.height / 2); };
  $$(".segbtn[data-view]").forEach(b => b.onclick = async () => {
    $$(".segbtn[data-view]").forEach(x => x.classList.toggle("active", x === b));
    S.view = b.dataset.view;
    if (S.view !== "after" && S.folder) {
      try { const r = await fetchImage("before"); $("#imgBefore").src = r.url; $("#imgBefore").style.width = S.imgW + "px"; $("#imgBefore").style.height = S.imgH + "px"; } catch (e) { toast(e.message, true); }
    }
    layoutSplit();
  });
  $("#previewSize").onchange = () => schedulePreview(0);
}

/* ------------------------------------------------------------ frames */
function renderFrames() {
  const fr = S.frames.map((f, i) => ({ ...f, idx: i + 1 }));
  const acc = fr.filter(f => f.accepted);
  $("#framesCount").textContent = fr.length ? `${acc.length}/${fr.length}` : "";
  if (!fr.length) { $("#framesSummary").innerHTML = `<span class="muted">Run “Analyse frames” to measure every sub.</span>`; $("#framesTable tbody").innerHTML = ""; $("#charts").innerHTML = ""; return; }
  const exp = acc.reduce((a, f) => a + f.exptime, 0);
  const med = k => { const v = acc.map(f => f[k]).filter(x => x !== null && !Number.isNaN(x)).sort((a, b) => a - b); return v[Math.floor(v.length / 2)]; };
  const partial = fr.filter(f => f.accepted && f.obstructed > 0).length;
  $("#framesSummary").innerHTML = [
    [`${acc.length} / ${fr.length}`, "frames used"], [`${(exp / 60).toFixed(1)} min`, "integration"],
    [fmt(med("fwhm")) + " px", "median FWHM"], [fmt(med("elongation")), "median elongation"],
    [String(fr.length - acc.length), "rejected"], [String(partial), "partially masked"],
  ].map(([v, l]) => `<div class="stat"><b>${v}</b><span>${l}</span></div>`).join("");

  const { k, dir } = S.sort;
  fr.sort((a, b) => ((a[k] ?? -1e9) > (b[k] ?? -1e9) ? 1 : -1) * dir);
  $("#framesTable tbody").innerHTML = fr.map(f => `
    <tr class="${f.accepted ? "" : "rej"} ${f.is_reference ? "ref" : ""}" data-name="${f.name}">
      <td>${f.idx}</td><td title="${f.name}">${f.time ? f.time.slice(11, 19) : f.name}</td>
      <td><button class="toggle ${f.accepted ? "on" : ""} ${f.overridden ? "ovr" : ""}" title="${f.overridden ? "manual override (click to cycle)" : "automatic"}"></button></td>
      <td>${fmt(f.fwhm)}</td><td>${fmt(f.elongation)}</td><td>${f.n_stars}</td><td>${fmt(f.transparency)}</td>
      <td>${f.obstructed > 0 ? Math.round(f.obstructed * 100) + "%" : ""}</td><td>${fmt(f.background, 0)}</td><td>${fmt(f.weight)}</td>
      <td class="notes">${f.reasons.join("; ")}</td></tr>`).join("");
  $$("#framesTable tbody tr").forEach(tr => {
    tr.onmouseenter = () => showFramePreview(tr.dataset.name);
    $(".toggle", tr).onclick = async ev => {
      ev.stopPropagation();
      const f = S.frames.find(x => x.name === tr.dataset.name);
      // cycle: auto -> forced opposite -> back to auto
      const next = f.overridden ? null : !f.accepted;
      const r = await api("/api/frames/override", { method: "POST", body: { folder: S.folder, name: f.name, accepted: next } });
      S.frames = r.frames; renderFrames();
      toast("Frame selection changed – re-run “Register & integrate” to apply");
    };
  });
  renderCharts();
}

function showFramePreview(name) {
  const f = S.frames.find(x => x.name === name);
  if (!f) return;
  const tm = f.tile_mask || [];
  const cols = tm[0]?.length || 0;
  $("#framePreview").innerHTML = `<img src="/api/thumb?folder=${encodeURIComponent(S.folder)}&name=${encodeURIComponent(name)}" alt="">
    <div class="small" style="margin-top:8px"><b>${name}</b><br>${f.accepted ? "✓ used" : "✗ rejected"} · weight ${fmt(f.weight)} · anomaly score ${fmt(f.anomaly, 1)}</div>
    ${cols ? `<div class="small muted" style="margin-top:8px">Obstruction map (reference orientation, red = masked)</div>
    <div class="tilegrid" style="grid-template-columns:repeat(${cols},1fr)">${tm.flat().map(v => `<div style="background:${v > 0.5 ? "#2b3a2f" : "#a33"}"></div>`).join("")}</div>` : ""}`;
}

function renderCharts() {
  const fr = S.frames;
  const defs = [["fwhm", "FWHM (px)"], ["n_stars", "Detected stars"], ["transparency", "Transparency"], ["background", "Sky background"]];
  $("#charts").innerHTML = defs.map(([k, label]) => {
    const vals = fr.map(f => f[k]).filter(v => v !== null && !Number.isNaN(v));
    if (!vals.length) return "";
    const sorted = [...vals].sort((a, b) => a - b);
    const lo = sorted[Math.floor(sorted.length * 0.01)], hi = sorted[Math.ceil(sorted.length * 0.99) - 1];
    const W = 300, H = 70, n = fr.length;
    const y = v => H - 4 - (Math.min(Math.max(v, lo), hi) - lo) / ((hi - lo) || 1) * (H - 8);
    const pts = fr.map((f, i) => (f[k] === null || Number.isNaN(f[k])) ? "" :
      `<circle cx="${(i / Math.max(n - 1, 1) * (W - 6) + 3).toFixed(1)}" cy="${y(f[k]).toFixed(1)}" r="1.8" fill="${f.accepted ? "#4fb3c9" : "#e5534b"}"/>`).join("");
    return `<div class="chart"><h4>${label}</h4><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${pts}</svg></div>`;
  }).join("");
}

/* ------------------------------------------------------------ export & diag */
async function refreshExports() {
  if (!S.folder) return;
  const r = await api(`/api/exports?folder=${encodeURIComponent(S.folder)}`);
  $("#exportsList").innerHTML = r.exports.length ? r.exports.map(e => `
    <div class="exp"><a href="/api/download?inline=1&path=${encodeURIComponent(e.jpg)}" target="_blank"><img src="/api/download?inline=1&path=${encodeURIComponent(e.jpg)}" loading="lazy" alt=""></a>
      <div class="meta"><b>${e.name}</b><span class="muted">${e.created} · ${e.size_mb} MB</span>
      <span><a href="/api/download?path=${encodeURIComponent(e.jpg)}">JPG</a>${e.tif ? `<a href="/api/download?path=${encodeURIComponent(e.tif)}">16-bit TIFF</a>` : ""}</span></div></div>`).join("")
    : `<p class="muted">No exports yet.</p>`;
}
async function refreshDiag() {
  if (!S.folder) return;
  const q = `folder=${encodeURIComponent(S.folder)}&t=${Date.now()}`;
  $("#diagRef").src = `/api/diagnostic?kind=reference&${q}`;
  $("#diagCov").src = `/api/diagnostic?kind=coverage&${q}`;
  $("#diagRej").src = `/api/diagnostic?kind=rejection&${q}`;
  let li = {};
  try { li = await api(`/api/linear_info?folder=${encodeURIComponent(S.folder)}`); } catch { }
  $("#diagInfo").textContent = JSON.stringify({ stack: S.status?.stack_meta, processing: li }, null, 1);
}

function switchTab(name) {
  $$(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  $$(".tabpane").forEach(p => p.classList.toggle("active", p.id === "tab-" + name));
  if (name === "process") setTimeout(fitView, 50);
  if (name === "diag") refreshDiag();
  if (name === "explore") setTimeout(() => Explore.show(), 30);
  if (name === "lab") Lab.show();
}

/* ------------------------------------------------------------ bindings */
function bindUI() {
  $("#datasetSelect").onchange = e => openDataset(e.target.value);
  $("#openCustom").onclick = async () => {
    const p = $("#customPath").value.trim(); if (!p) return;
    await openDataset(p);
  };
  $$("[data-job]").forEach(b => b.onclick = () => startJob(b.dataset.job));
  $("#cancelJob").onclick = () => S.job && api(`/api/jobs/${S.job.id}/cancel`, { method: "POST" });
  $("#exportBtn").onclick = $("#exportBtn2").onclick = () => startJob("export");
  $("#resetParams").onclick = () => { S.params = currentDefaults(); applyParamsToUI(); saveParams(); schedulePreview(0); };
  $$(".tab").forEach(t => t.onclick = () => switchTab(t.dataset.tab));
  $$("#framesTable th[data-k]").forEach(th => th.onclick = () => {
    const k = th.dataset.k; S.sort = { k, dir: S.sort.k === k ? -S.sort.dir : 1 }; renderFrames();
  });
  const sens = $("#sensSlider");
  sens.oninput = () => ($("#sensOut").textContent = (+sens.value).toFixed(1));
  sens.onchange = async () => {
    if (!S.folder || !S.frames.length) return;
    const r = await api("/api/frames/sensitivity", { method: "POST", body: { folder: S.folder, sensitivity: +sens.value } });
    S.frames = r.frames; renderFrames(); $("#sp-sensitivity").value = sens.value;
  };
  const sps = $("#sp-sensitivity");
  sps.oninput = () => (sps.nextElementSibling.textContent = (+sps.value).toFixed(1));
  sps.nextElementSibling.textContent = "1.0";
  const q = $("#ex-quality"); q.oninput = () => (q.nextElementSibling.textContent = q.value);
  bindViewer();
  loadExperimentChoices();
  $(".sidebar details.adv").addEventListener("toggle", ev => { if (ev.target.open) loadExperimentChoices(); });
}

init().catch(e => toast(e.message, true));
