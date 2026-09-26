/* Experiments tab: configure, dispatch, monitor and review Optuna studies (vanilla JS) */
const Lab = (() => {
  const L = { tasks: null, byName: {}, data: null, space: {}, sel: null, detail: null, trialSel: null, timer: 0, synthDefaults: null };
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const f4 = v => v === null || v === undefined || Number.isNaN(v) ? "–" : (typeof v === "number" ? (Math.abs(v) >= 1000 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : +v.toPrecision(4)) : String(v));
  const RUNNING = ["starting", "preparing", "running"];

  /* ---------------------------------------------------------------- loading */
  async function show() {
    if (!L.tasks) await init();
    await Promise.all([loadData(), loadStudies()]);
    tick();
  }
  async function init() {
    const r = await api("/api/lab/tasks");
    L.tasks = r.tasks; L.synthDefaults = r.synthetic_defaults;
    r.tasks.forEach(t => L.byName[t.name] = t);
    $("#lbTask").innerHTML = r.tasks.map(t => `<option value="${t.name}">${esc(t.label)}</option>`).join("");
    $("#lbSampler").innerHTML = r.samplers.map(s => `<option value="${s.name}">${esc(s.label)}</option>`).join("");
    const dev = S.system.devices;
    $("#lbDevice").innerHTML = `<option value="auto">auto (${dev.default})</option>` +
      (dev.gpus || []).map(g => `<option value="${g.id}">${esc(g.name)}</option>`).join("") + `<option value="cpu">CPU</option>`;
    buildSynthForm();
    $("#lbTask").onchange = buildTask;
    $("#lbData").onchange = buildObjectives;
    $("#lbGo").onclick = start;
    $("#lbRefresh").onclick = () => { loadData(); loadStudies(); };
    $("#lbSynthGo").onclick = synth;
    buildTask();
  }

  async function loadData() {
    L.data = await api("/api/lab/datasets");
    const cur = $("#lbData").value;
    const real = L.data.real.map(d => `<option value='${esc(JSON.stringify({ kind: "real", folder: d.folder, name: d.name }))}' ${d.stacked ? "" : "data-unstacked=1"}>${esc(d.name)} — ${d.n_fits} subs${d.stacked ? "" : " (analysed, not stacked)"}</option>`).join("");
    const syn = L.data.synthetic.map(d => {
      const ok = d.status.state === "done";
      return `<option value='${esc(JSON.stringify({ kind: "synthetic", dir: d.dir, name: d.name }))}' ${ok ? "" : "disabled"}>${esc(d.name)} — ${d.spec.n_subs || "?"} subs ${d.spec.width}×${d.spec.height}${ok ? "" : ` (${esc(d.status.state)}: ${esc(d.status.message || "")})`}</option>`;
    }).join("");
    $("#lbData").innerHTML = `<optgroup label="Real sessions">${real || "<option disabled>none stacked yet</option>"}</optgroup><optgroup label="Synthetic (with ground truth)">${syn || "<option disabled>none yet</option>"}</optgroup>`;
    if (cur && [...$("#lbData").options].some(o => o.value === cur && !o.disabled)) $("#lbData").value = cur;
    buildObjectives();
    // keep polling while a dataset is being generated
    if (L.data.synthetic.some(d => ["running", "new"].includes(d.status.state))) { clearTimeout(L.dataTimer); L.dataTimer = setTimeout(loadData, 4000); }
  }
  const dataset = () => { try { return JSON.parse($("#lbData").value); } catch { return null; } };

  /* ---------------------------------------------------------------- form */
  function buildSynthForm() {
    const D = L.synthDefaults;
    const fields = [["name", "Name", "text", "synthetic-" + new Date().toISOString().slice(5, 10)], ["width", "Width (px)", "number"], ["height", "Height (px)", "number"],
      ["n_subs", "Subs", "number"], ["exptime", "Exposure (s)", "number"], ["seeing_fwhm", "Seeing FWHM (px)", "number"],
      ["seeing_spread", "Seeing spread (log σ)", "number"], ["star_density", "Stars per 10⁶ px", "number"],
      ["nebula", "Nebula", ["emission", "reflection", "none"]], ["n_galaxies", "Galaxies", "number"],
      ["filter", "Filter", ["IRCUT", "LP"]], ["rotation_deg", "Field rotation (°)", "number"], ["dither_px", "Dither σ (px)", "number"],
      ["trail_fraction", "Subs with satellite trails", "number"], ["cloudy_fraction", "Cloudy subs", "number"], ["seed", "Seed", "number"]];
    $("#lbSynthFields").innerHTML = fields.map(([k, lab, t, dflt]) => {
      const v = dflt ?? D[k];
      if (Array.isArray(t)) return `<label>${lab}<select data-syn="${k}">${t.map(o => `<option ${o === v ? "selected" : ""}>${o}</option>`).join("")}</select></label>`;
      return `<label>${lab}<input data-syn="${k}" type="${t}" value="${esc(v)}" step="any"></label>`;
    }).join("");
  }
  async function synth() {
    const spec = {};
    $$("[data-syn]").forEach(i => { const k = i.dataset.syn; spec[k] = i.type === "number" ? parseFloat(i.value) : i.value; });
    ["width", "height", "n_subs", "n_galaxies", "seed"].forEach(k => spec[k] = Math.round(spec[k]));
    try {
      await api("/api/lab/synthetic", { method: "POST", body: { name: spec.name, spec, device: $("#lbDevice").value } });
      toast("Generating the synthetic dataset — it appears in the list when ready");
      $("#lbSynth").open = false; loadData();
    } catch (e) { toast(e.message, true); }
  }

  function buildTask() {
    const t = L.byName[$("#lbTask").value];
    $("#lbTaskDesc").textContent = t.description;
    $("#lbOptions").innerHTML = t.options.map(o => o.type === "categorical"
      ? `<label>${esc(o.label)}<select data-opt="${o.name}">${o.choices.map(c => `<option ${c === o.default ? "selected" : ""}>${c}</option>`).join("")}</select></label>`
      : `<label>${esc(o.label)}<input data-opt="${o.name}" data-t="${o.type}" type="number" step="any" value="${o.default}"></label>`).join("");
    L.space = {};
    t.params.forEach(p => L.space[p.name] = { tune: !!p.tune, low: p.low, high: p.high, log: !!p.log, step: null, choices: p.choices, value: p.default });
    renderSpace();
    buildObjectives();
  }

  function renderSpace() {
    const t = L.byName[$("#lbTask").value];
    $("#lbSpace").innerHTML = t.params.map(p => {
      const s = L.space[p.name];
      let body = "";
      if (s.tune) {
        if (p.type === "float" || p.type === "int") body = `<div class="rng">from <input type="number" step="any" data-k="low" value="${s.low}"> to <input type="number" step="any" data-k="high" value="${s.high}">
          <label class="small"><input type="checkbox" data-k="log" ${s.log ? "checked" : ""}> log</label> step <input type="number" step="any" data-k="step" value="${s.step ?? ""}" placeholder="any"></div>`;
        else if (p.type === "categorical") body = `<div class="rng">choices <input type="text" class="wide" data-k="choices" value="${esc(s.choices.join(", "))}"></div>`;
        else body = `<div class="rng muted small">on / off</div>`;
      }
      const fixed = s.tune ? "" : p.type === "bool" ? `<select class="fixed" data-k="value"><option value="true" ${s.value ? "selected" : ""}>on</option><option value="false" ${!s.value ? "selected" : ""}>off</option></select>`
        : p.type === "categorical" ? `<select class="fixed" data-k="value">${p.choices.map(c => `<option ${String(c) === String(s.value) ? "selected" : ""}>${c}</option>`).join("")}</select>`
        : `<input class="fixed" type="number" step="any" data-k="value" value="${s.value}">`;
      return `<div class="lbparam" data-p="${p.name}"><div class="hd"><input type="checkbox" data-k="tune" ${s.tune ? "checked" : ""}>
        <span title="${esc(p.pipeline ? "pipeline setting: " + p.pipeline : "not a pipeline setting")}">${esc(p.label)}</span>${fixed}</div>${body}</div>`;
    }).join("");
    $$("#lbSpace [data-k]").forEach(inp => inp.onchange = () => {
      const name = inp.closest("[data-p]").dataset.p, s = L.space[name], k = inp.dataset.k;
      const p = t.params.find(q => q.name === name);
      if (k === "tune") { s.tune = inp.checked; renderSpace(); return; }
      if (k === "log") s.log = inp.checked;
      else if (k === "choices") s.choices = inp.value.split(",").map(x => x.trim()).filter(Boolean).map(x => (x !== "" && !Number.isNaN(+x)) ? +x : x);
      else if (k === "value") s.value = p.type === "bool" ? inp.value === "true" : p.type === "categorical" ? (Number.isNaN(+inp.value) ? inp.value : +inp.value) : parseFloat(inp.value);
      else s[k] = inp.value === "" ? null : parseFloat(inp.value);
    });
  }

  function buildObjectives() {
    const t = L.byName[$("#lbTask").value]; if (!t) return;
    const ds = dataset();
    const real = !ds || ds.kind === "real";
    const opts = Object.entries(t.metrics).map(([k, m]) => `<option value="${k}" ${m.truth && real ? "disabled" : ""}>${esc(m.label)} (${m.direction === "minimize" ? "↓" : "↑"})${m.truth && real ? " — synthetic only" : ""}</option>`).join("");
    const o1 = $("#lbObj1").value, o2 = $("#lbObj2").value;
    $("#lbObj1").innerHTML = opts;
    $("#lbObj2").innerHTML = `<option value="">— nothing (single objective)</option>` + opts;
    $("#lbObj1").value = (o1 && t.metrics[o1] && !(t.metrics[o1].truth && real)) ? o1 : t.default_objective;
    if (o2 && t.metrics[o2]) $("#lbObj2").value = o2;
  }

  async function start() {
    const t = L.byName[$("#lbTask").value];
    const ds = dataset();
    if (!ds) return toast("Choose a dataset", true);
    const objectives = [$("#lbObj1").value, $("#lbObj2").value].filter(Boolean).filter((v, i, a) => a.indexOf(v) === i)
      .map(m => ({ metric: m, direction: t.metrics[m].direction }));
    const options = {};
    $$("[data-opt]").forEach(i => options[i.dataset.opt] = i.tagName === "SELECT" ? i.value : parseFloat(i.value));
    const space = {};
    for (const p of t.params) {
      const s = L.space[p.name];
      space[p.name] = s.tune ? { tune: true, low: s.low, high: s.high, log: s.log, step: s.step, choices: s.choices } : { tune: false, value: s.value };
    }
    if (!Object.values(space).some(s => s.tune)) return toast("Tick at least one parameter to tune", true);
    const sampler = $("#lbSampler").value;
    const cfg = { name: $("#lbName").value.trim(), task: t.name, dataset: ds, options, space, objectives, sampler,
      n_trials: parseInt($("#lbTrials").value) || 30, timeout_min: parseFloat($("#lbTimeout").value) || 0,
      seed: parseInt($("#lbSeed").value) || 0, device: $("#lbDevice").value, baseline: $("#lbBaseline").checked };
    try {
      const r = await api("/api/lab/studies", { method: "POST", body: { config: cfg } });
      toast("Study started"); L.sel = r.id; await loadStudies(); loadDetail(); tick();
    } catch (e) { toast(e.message, true); }
  }

  /* ---------------------------------------------------------------- studies */
  async function loadStudies() {
    const r = await api("/api/lab/studies");
    L.studies = r.studies;
    $("#lbStudies").innerHTML = r.studies.length ? r.studies.map(s => `<div class="lbst ${s.id === L.sel ? "sel" : ""}" data-id="${esc(s.id)}">
      <b>${esc(s.name)}</b><span class="state ${esc(s.state)}">${esc(s.state)}</span>
      <span class="meta">${esc(L.byName[s.task]?.label || s.task)} · ${esc(s.dataset)} · ${s.n_complete}/${s.n_trials} trials</span>
      <span class="meta">${s.best !== null && s.best !== undefined ? `best ${esc(s.objectives[0].metric)} ${f4(s.best)}` : s.objectives.map(o => o.metric).join(" × ")}</span></div>`).join("")
      : `<p class="muted small">No studies yet. Configure one on the left.</p>`;
    $$("#lbStudies .lbst").forEach(el => el.onclick = () => { L.sel = el.dataset.id; L.trialSel = null; loadStudies(); loadDetail(); });
    if (!L.sel && r.studies.length) { L.sel = r.studies[0].id; loadDetail(); }
  }

  async function loadDetail() {
    if (!L.sel) { $("#lbDetail").innerHTML = ""; return; }
    try { L.detail = await api(`/api/lab/studies/${encodeURIComponent(L.sel)}`); } catch (e) { $("#lbDetail").innerHTML = `<p class="muted">${esc(e.message)}</p>`; return; }
    renderDetail();
  }

  function tick() {
    clearTimeout(L.timer);
    L.timer = setTimeout(async () => {
      if (!$("#tab-lab").classList.contains("active")) return;
      const running = (L.studies || []).some(s => RUNNING.includes(s.state) || s.state === "new");
      if (running) { await loadStudies(); if (L.detail) await loadDetail(); }
      tick();
    }, 3000);
  }

  /* ---------------------------------------------------------------- detail */
  function objectiveOf(D, k = 0) { return D.config.objectives[k]; }
  function isBetter(a, b, dir) { return dir === "minimize" ? a < b : a > b; }

  function renderDetail() {
    const D = L.detail, cfg = D.config, st = D.status;
    const done = D.trials.filter(t => t.state === "COMPLETE");
    const tuned = Object.entries(cfg.space).filter(([, v]) => v.tune).map(([k]) => k);
    const o1 = objectiveOf(D, 0), o2 = cfg.objectives[1];
    const base = D.trials.find(t => t.baseline && t.state === "COMPLETE");
    const bestNum = D.best?.[0];
    const shown = D.trials.find(t => t.number === (L.trialSel ?? bestNum)) || done[0];
    const running = RUNNING.includes(st.state);
    const head = `<div class="card"><h3>${esc(cfg.name)} <span class="state ${esc(st.state)}">${esc(st.state)}</span></h3>
      <div class="small">${esc(L.byName[cfg.task]?.label || cfg.task)} on <b>${esc(cfg.dataset.name)}</b> (${esc(cfg.dataset.kind)}) · ${esc(cfg.sampler)} · ${done.length}/${cfg.n_trials} trials complete${D.trials.length - done.length ? `, ${D.trials.length - done.length} other` : ""} · device ${esc(cfg.device)}</div>
      <div class="small muted" style="margin-top:4px">${esc(st.message || "")}</div>
      <div class="row gap" style="margin-top:10px">
        ${running ? `<button class="btn small" id="lbStop">Stop</button>` : `<button class="btn small" id="lbCont">Run more trials</button> <input id="lbMore" type="number" value="10" min="1" style="width:70px">
        <button class="btn small ghost" id="lbDel">Delete study</button>`}
      </div></div>`;
    // charts
    const charts = [];
    if (done.length) {
      if (o2) charts.push(chartPareto(D, done));
      charts.push(chartHistory(D, done, 0));
      const imp = D.importance?.[o1.metric];
      if (imp && !imp._error && Object.keys(imp).length) charts.push(chartImportance(imp, o1.metric));
      for (const k of tuned.slice(0, 8)) charts.push(chartSlice(D, done, k));
    }
    // selected / best trial
    let bestCard = "";
    if (shown) {
      const m = shown.metrics || {};
      const mrows = Object.entries(D.metrics).filter(([k]) => m[k] !== undefined && m[k] !== null).map(([k, def]) => {
        const v = m[k], b = base?.metrics?.[k];
        let delta = "";
        if (base && shown.number !== base.number && typeof v === "number" && typeof b === "number" && b !== 0) {
          const better = isBetter(v, b, def.direction);
          delta = `<span style="color:${v === b ? "var(--muted)" : better ? "var(--good)" : "var(--bad)"}"> ${v > b ? "+" : ""}${f4((v - b) / Math.abs(b) * 100)} %</span>`;
        }
        return `<span>${esc(def.label)}</span><b>${f4(v)}${delta}</b>`;
      }).join("");
      const params = shown.metrics ? (Object.entries(shown.params).map(([k, v]) => `<span>${esc(k)}</span><b>${esc(f4(v))}</b>`).join("")) : "";
      const applicable = Object.keys(shown.params).filter(k => D.pipeline_map[k]);
      bestCard = `<div class="card"><h3>${shown.number === bestNum ? "Best trial" : "Trial"} #${shown.number}${shown.baseline ? " (baseline: current pipeline settings)" : ""}</h3>
        <div class="lbbest"><div>
          <div class="kv">${params}</div>
          <div class="kv" style="margin-top:10px">${mrows}</div>
          ${base && shown.number !== base.number ? `<p class="muted small">Percentages: change from the baseline trial #${base.number}.</p>` : ""}
          ${applicable.length ? `<button class="btn small" id="lbApply" style="margin-top:10px">Apply to the pipeline settings</button>
            <p class="muted small">Sets ${applicable.map(k => D.pipeline_map[k]).join(", ")} in “Integration &amp; compute options”.${Object.keys(shown.params).length > applicable.length ? " Not pipeline settings: " + Object.keys(shown.params).filter(k => !D.pipeline_map[k]).join(", ") + "." : ""}</p>` : ""}
          ${shown.error ? `<p style="color:var(--bad)" class="small">${esc(shown.error)}</p>` : ""}
        </div><div>${shown.image ? `<img src="/api/lab/studies/${encodeURIComponent(D.id)}/trial/${shown.number}.jpg" alt="">
          <p class="muted small">Same stretch for every trial of this study.</p>` : ""}</div></div></div>`;
    }
    // trials table
    const mcols = Object.keys(D.metrics).filter(k => D.trials.some(t => t.metrics && typeof t.metrics[k] === "number") && !cfg.objectives.some(o => o.metric === k)).slice(0, 5);
    const table = `<div class="card"><h3>Trials</h3><div class="lbtable"><table><thead><tr><th>#</th><th>State</th>
      ${cfg.objectives.map(o => `<th>${esc(o.metric)} ${o.direction === "minimize" ? "↓" : "↑"}</th>`).join("")}
      ${tuned.map(k => `<th>${esc(k)}</th>`).join("")}${mcols.map(k => `<th>${esc(k)}</th>`).join("")}<th>Time</th></tr></thead><tbody>
      ${[...D.trials].reverse().map(t => `<tr data-n="${t.number}" class="${(D.best || []).includes(t.number) ? "best" : ""} ${t.state === "FAIL" ? "fail" : ""}" title="${esc(t.error || "")}">
        <td>${t.number}${t.baseline ? " ●" : ""}</td><td>${esc(t.state.toLowerCase())}</td>
        ${cfg.objectives.map((o, i) => `<td class="num">${t.values ? f4(t.values[i]) : "–"}</td>`).join("")}
        ${tuned.map(k => `<td class="num">${esc(f4(t.params[k]))}</td>`).join("")}
        ${mcols.map(k => `<td class="num">${f4(t.metrics?.[k])}</td>`).join("")}
        <td class="num">${t.seconds ? f4(t.seconds) + " s" : "–"}</td></tr>`).join("")}
      </tbody></table></div><p class="muted small">● = baseline (the pipeline's current settings). Click a row to show that trial above; hover for its preview.</p></div>`;
    const log = `<details class="card"><summary class="small muted">Log</summary><pre class="lblog">${esc(D.log)}</pre></details>`;
    $("#lbDetail").innerHTML = head + (charts.length ? `<div class="lbcharts">${charts.join("")}</div>` : "") + bestCard + table + log;
    // bindings
    $("#lbStop")?.addEventListener("click", async () => { try { await api(`/api/lab/studies/${encodeURIComponent(D.id)}/stop`, { method: "POST" }); toast("Stopping after cancelling the running trial"); } catch (e) { toast(e.message, true); } });
    $("#lbCont")?.addEventListener("click", async () => { try { await api(`/api/lab/studies/${encodeURIComponent(D.id)}/continue`, { method: "POST", body: { extra: parseInt($("#lbMore").value) || 10 } }); toast("Continuing"); await loadStudies(); loadDetail(); tick(); } catch (e) { toast(e.message, true); } });
    $("#lbDel")?.addEventListener("click", async () => {
      if (!confirm(`Delete the study “${cfg.name}” and all its trials? This cannot be undone.`)) return;
      try { await api(`/api/lab/studies/${encodeURIComponent(D.id)}`, { method: "DELETE" }); L.sel = null; L.detail = null; $("#lbDetail").innerHTML = ""; loadStudies(); } catch (e) { toast(e.message, true); }
    });
    $("#lbApply")?.addEventListener("click", () => applyToPipeline(D, shown));
    $$("#lbDetail tbody tr").forEach(tr => {
      tr.onclick = () => { L.trialSel = +tr.dataset.n; renderDetail(); };
      const t = D.trials.find(q => q.number === +tr.dataset.n);
      if (!t?.image) return;
      tr.onmouseenter = e => { const im = document.createElement("img"); im.className = "lbimgpop"; im.src = `/api/lab/studies/${encodeURIComponent(D.id)}/trial/${t.number}.jpg`;
        im.style.left = Math.min(e.clientX + 20, window.innerWidth - 380) + "px"; im.style.top = Math.max(10, e.clientY - 200) + "px"; document.body.appendChild(im); tr._pop = im; };
      tr.onmouseleave = () => { tr._pop?.remove(); tr._pop = null; };
    });
    $$("#lbDetail [data-trial]").forEach(el => el.onclick = () => { L.trialSel = +el.dataset.trial; renderDetail(); });
  }

  function applyToPipeline(D, t) {
    const set = [];
    for (const [k, v] of Object.entries(t.params)) {
      const key = D.pipeline_map[k]; if (!key) continue;
      const el = $("#sp-" + key); if (!el) continue;
      if (el.type === "checkbox") el.checked = !!v; else el.value = String(v);
      set.push(`${key} = ${v}`);
    }
    const task = D.config.task;
    if (task === "imagemm" || task === "network") { $("#sp-deconv_method").value = task; $("#sp-deconv_method").dispatchEvent(new Event("change", { bubbles: true })); }
    $(".sidebar details.adv").open = true;
    if (typeof loadExperimentChoices === "function") loadExperimentChoices();
    toast(`Applied: ${set.join(", ")}. Re-run the pipeline step to use them.`);
  }

  /* ---------------------------------------------------------------- charts (SVG) */
  const W = 340, H = 180, P = { l: 44, r: 10, t: 8, b: 26 };
  function scale(vals, log) {
    const v = vals.filter(x => Number.isFinite(x) && (!log || x > 0));
    let lo = Math.min(...v), hi = Math.max(...v);
    if (log) { lo = Math.log10(lo); hi = Math.log10(hi); }
    if (lo === hi) { lo -= 0.5; hi += 0.5; }
    const pad = (hi - lo) * 0.06;
    lo -= pad; hi += pad;
    return { lo, hi, f: x => (log ? Math.log10(x) : x), log };
  }
  const X = (s, x) => P.l + (s.f(x) - s.lo) / (s.hi - s.lo) * (W - P.l - P.r);
  const Y = (s, y) => H - P.b - (s.f(y) - s.lo) / (s.hi - s.lo) * (H - P.t - P.b);
  function axes(sx, sy, xl, yl) {
    const t = (s, v) => f4(s.log ? Math.pow(10, v) : v);
    const xt = sx.labels ? sx.labels.map((l, i) => `<text x="${X(sx, i)}" y="${H - 12}" text-anchor="middle">${esc(l)}</text>`).join("")
      : `<text x="${P.l}" y="${H - 12}">${t(sx, sx.lo)}</text><text x="${W - P.r}" y="${H - 12}" text-anchor="end">${t(sx, sx.hi)}</text>`;
    return `<line x1="${P.l}" y1="${H - P.b}" x2="${W - P.r}" y2="${H - P.b}" stroke="#2a3346"/><line x1="${P.l}" y1="${P.t}" x2="${P.l}" y2="${H - P.b}" stroke="#2a3346"/>
      ${xt}
      <text x="${(P.l + W - P.r) / 2}" y="${H - 2}" text-anchor="middle">${esc(xl)}</text>
      <text x="${P.l - 4}" y="${H - P.b}" text-anchor="end">${t(sy, sy.lo)}</text><text x="${P.l - 4}" y="${P.t + 8}" text-anchor="end">${t(sy, sy.hi)}</text>
      <text x="10" y="${(P.t + H - P.b) / 2}" transform="rotate(-90 10 ${(P.t + H - P.b) / 2})" text-anchor="middle">${esc(yl)}</text>`;
  }
  const dot = (x, y, t, extra = "") => `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${t.baseline ? 5 : 3.2}" fill="${t.baseline ? "none" : "#4fb3c9"}" stroke="${t.baseline ? "#e6b450" : "none"}" stroke-width="2" data-trial="${t.number}" style="cursor:pointer" ${extra}><title>#${t.number}</title></circle>`;

  function chartHistory(D, done, k) {
    const o = D.config.objectives[k];
    const sx = scale(done.map(t => t.number)), sy = scale(done.map(t => t.values[k]));
    let best = null, path = "";
    for (const t of [...done].sort((a, b) => a.number - b.number)) {
      const v = t.values[k];
      if (best === null || isBetter(v, best, o.direction)) best = v;
      path += `${path ? "L" : "M"}${X(sx, t.number).toFixed(1)},${Y(sy, best).toFixed(1)} `;
    }
    return `<div class="lbchart"><h4>Optimisation history — ${esc(o.metric)}</h4><svg viewBox="0 0 ${W} ${H}">${axes(sx, sy, "trial", o.metric)}
      <path d="${path}" fill="none" stroke="#e0527a" stroke-width="1.5"/>${done.map(t => dot(X(sx, t.number), Y(sy, t.values[k]), t)).join("")}</svg></div>`;
  }
  function chartPareto(D, done) {
    const [o1, o2] = D.config.objectives;
    const sx = scale(done.map(t => t.values[0]), false), sy = scale(done.map(t => t.values[1]), false);
    const front = done.filter(t => (D.best || []).includes(t.number)).sort((a, b) => a.values[0] - b.values[0]);
    const path = front.map((t, i) => `${i ? "L" : "M"}${X(sx, t.values[0]).toFixed(1)},${Y(sy, t.values[1]).toFixed(1)}`).join(" ");
    return `<div class="lbchart"><h4>Pareto front — ${esc(o1.metric)} vs ${esc(o2.metric)}</h4><svg viewBox="0 0 ${W} ${H}">${axes(sx, sy, o1.metric, o2.metric)}
      <path d="${path}" fill="none" stroke="#4cc38a" stroke-width="1.5"/>${done.map(t => dot(X(sx, t.values[0]), Y(sy, t.values[1]), t, (D.best || []).includes(t.number) ? 'fill="#4cc38a"' : "")).join("")}</svg></div>`;
  }
  function chartImportance(imp, metric) {
    const e = Object.entries(imp).sort((a, b) => b[1] - a[1]);
    const bh = 18, h = e.length * bh + 10;
    return `<div class="lbchart"><h4>Parameter importance (fANOVA) — ${esc(metric)}</h4><svg viewBox="0 0 ${W} ${h}" style="height:${h}px">
      ${e.map(([k, v], i) => `<text x="120" y="${i * bh + 14}" text-anchor="end">${esc(k)}</text><rect x="126" y="${i * bh + 4}" width="${(v * (W - 170)).toFixed(1)}" height="${bh - 6}" fill="#4fb3c9" rx="2"/>
        <text x="${130 + v * (W - 170)}" y="${i * bh + 14}">${(v * 100).toFixed(0)}%</text>`).join("")}</svg></div>`;
  }
  function chartSlice(D, done, k) {
    const o = D.config.objectives[0], spec = D.config.space[k] || {};
    const vals = done.map(t => t.params[k]);
    const cat = vals.some(v => typeof v !== "number");
    let sx, xs;
    if (cat) {
      const levels = [...new Set(vals.map(String))];
      sx = { lo: -0.5, hi: levels.length - 0.5, f: x => x, log: false };
      xs = done.map(t => levels.indexOf(String(t.params[k])) + ((t.number * 0.618) % 1 - 0.5) * 0.3);
      sx.labels = levels;
    } else { sx = scale(vals, !!spec.log); xs = vals; }
    const sy = scale(done.map(t => t.values[0]));
    const ax = axes(sx, sy, k, o.metric);
    return `<div class="lbchart"><h4>${esc(k)} vs ${esc(o.metric)}</h4><svg viewBox="0 0 ${W} ${H}">${ax}${done.map((t, i) => dot(X(sx, xs[i]), Y(sy, t.values[0]), t)).join("")}</svg></div>`;
  }

  return { show };
})();
