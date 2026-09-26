/* Explore tab: the processed image with a plate-solved overlay / star map (vanilla JS, canvas 2D) */
const Explore = (() => {
  const COL = { star: "#ffd27a", variable: "#ff8c5a", multiple: "#b58cff", galaxy: "#5ad1ff", nebula: "#4cc38a", other: "#9aa4b8" };
  const X = {
    folder: null, data: null, img: null, imgKey: null, mode: "overlay", view: { s: 1, x: 0, y: 0 },
    layers: { stars: true, grid: true, labels: true, compass: true, star: true, variable: true, multiple: true, galaxy: true, nebula: true, other: false },
    mag: 13, opacity: 0.85, hover: null, pinned: null, mouse: null, idx: null, raf: 0, polling: false,
  };
  const dpr = () => window.devicePixelRatio || 1;
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const num = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v)) ? null : Number(v).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });

  /* ---------------------------------------------------------------- data */
  function reset() {
    X.folder = null; X.data = null; X.img = null; X.imgKey = null; X.hover = X.pinned = null; X.idx = null;
    $("#exPop").hidden = true; $("#exList").innerHTML = ""; $("#exLayers").innerHTML = "";
    $("#exField").innerHTML = `<h3>Field</h3><div class="muted small">Not solved yet.</div>`;
    draw();
  }

  function placeholder(text, solveBtn = false) {
    $("#exPlaceholder").style.display = text ? "flex" : "none";
    if (text) $("#exPhText").textContent = text;
    $("#exSolve").hidden = !solveBtn; $("#exSolveNote").hidden = !solveBtn;
  }

  async function show() {
    layout();
    if (!S.folder) return placeholder("Choose a dataset first.");
    if (!S.status?.stacked) return placeholder("Stack the dataset first: the sky map is matched to the stacked image.");
    if (X.folder !== S.folder) { reset(); X.folder = S.folder; }
    const st = await api(`/api/explore/status?folder=${encodeURIComponent(S.folder)}`);
    if (st.state === "running") return pollSolve();
    if (!st.solved) {
      return placeholder(st.state === "error" ? `Plate solving failed: ${st.message}` :
        "This image has not been identified yet.", true);
    }
    await load();
  }

  async function solve() {
    try {
      await api("/api/explore/solve", { method: "POST", body: { folder: S.folder } });
      pollSolve();
    } catch (e) { toast(e.message, true); }
  }

  async function pollSolve() {
    if (X.polling) return;
    X.polling = true;
    $("#exSpinner").hidden = false;
    try {
      for (;;) {
        const st = await api(`/api/explore/status?folder=${encodeURIComponent(S.folder)}`);
        if (st.state === "running") { placeholder(`${st.message}…`); await new Promise(r => setTimeout(r, 1000)); continue; }
        if (st.state === "error") { placeholder(`Plate solving failed: ${st.message}`, true); break; }
        await load(); break;
      }
    } finally { X.polling = false; $("#exSpinner").hidden = true; }
  }

  async function load() {
    const size = $("#exSize").value;
    const key = JSON.stringify([S.folder, S.params, size]);
    $("#exSpinner").hidden = false;
    try {
      if (!X.data || X.dataKey !== JSON.stringify([S.folder, S.params])) {
        placeholder("Loading the sky map…");
        X.data = await api("/api/explore/data", { method: "POST", body: { folder: S.folder, params: S.params } });
        X.dataKey = JSON.stringify([S.folder, S.params]);
        buildIndex(); buildPanel();
      }
      if (X.imgKey !== key) {
        const r = await fetch("/api/preview", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ folder: S.folder, params: S.params, which: "after", size: size === "full" ? null : parseInt(size) }) });
        if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail; } catch { } throw new Error(m); }
        const url = URL.createObjectURL(await r.blob());
        const img = new Image();
        await new Promise((ok, bad) => { img.onload = ok; img.onerror = bad; img.src = url; });
        const first = !X.img;
        X.img = img; X.imgKey = key;
        if (first) fit();
      }
      placeholder(null);
      draw();
    } catch (e) { placeholder(e.message); toast(e.message, true); }
    finally { $("#exSpinner").hidden = true; }
  }

  /* ---------------------------------------------------------------- index for hover */
  function buildIndex() {
    const d = X.data, C = 48;
    const cells = new Map();
    const add = (kind, i, x, y) => {
      const k = Math.floor(x / C) + "," + Math.floor(y / C);
      (cells.get(k) || cells.set(k, []).get(k)).push([kind, i]);
    };
    d.stars.x.forEach((x, i) => add("s", i, x, d.stars.y[i]));
    d.objects.forEach((o, i) => add("o", i, o.x, o.y));
    X.idx = { C, cells, ext: d.objects.map((o, i) => i).filter(i => d.objects[i].ellipse).sort((a, b) => area(d.objects[a]) - area(d.objects[b])) };
  }
  const area = o => o.ellipse[0] * o.ellipse[1];

  function visibleObj(o) { return X.layers[o.category]; }

  function pick(ix, iy) {
    const d = X.data; if (!d) return null;
    const R = Math.max(10 / X.view.s, 1.5);
    const { C, cells } = X.idx;
    let best = null, bd = R * R;
    for (let gx = Math.floor((ix - R) / C); gx <= Math.floor((ix + R) / C); gx++)
      for (let gy = Math.floor((iy - R) / C); gy <= Math.floor((iy + R) / C); gy++)
        for (const [kind, i] of cells.get(gx + "," + gy) || []) {
          let x, y;
          if (kind === "s") { x = d.stars.x[i]; y = d.stars.y[i]; }
          else { const o = d.objects[i]; if (!visibleObj(o) || o.gaia !== null) continue; x = o.x; y = o.y; }
          const dd = (x - ix) ** 2 + (y - iy) ** 2;
          // catalogued objects win ties with anonymous stars
          const w = kind === "o" ? dd * 0.7 : dd;
          if (w < bd) { bd = w; best = { kind, i }; }
        }
    if (best) return best;
    for (const i of X.idx.ext) {
      const o = d.objects[i]; if (!visibleObj(o)) continue;
      const [a, b, t] = o.ellipse, dx = ix - o.x, dy = iy - o.y;
      const u = dx * Math.cos(t) + dy * Math.sin(t), v = -dx * Math.sin(t) + dy * Math.cos(t);
      if ((u / a) ** 2 + (v / b) ** 2 <= 1) return { kind: "o", i };
    }
    return null;
  }

  /* ---------------------------------------------------------------- view */
  function canvases() { return X.mode === "side" ? [$("#exImg"), $("#exChart")] : X.mode === "chart" ? [$("#exChart")] : [$("#exImg")]; }
  function layout() {
    $("#exImg").hidden = X.mode === "chart"; $("#exChart").hidden = X.mode === "overlay";
    for (const c of [$("#exImg"), $("#exChart")]) {
      const r = c.getBoundingClientRect();
      c.width = Math.max(1, Math.round(r.width * dpr())); c.height = Math.max(1, Math.round(r.height * dpr()));
    }
  }
  function fit() {
    const d = X.data; if (!d) return;
    const r = canvases()[0].getBoundingClientRect();
    const s = Math.min(r.width / d.W, r.height / d.H) * 0.98;
    X.view = { s, x: (r.width - d.W * s) / 2, y: (r.height - d.H * s) / 2 };
    X.fitS = s;
    draw();
  }
  function zoomAt(f, cx, cy) {
    const v = X.view, ns = Math.min(40, Math.max(0.02, v.s * f));
    v.x = cx - (cx - v.x) * (ns / v.s); v.y = cy - (cy - v.y) * (ns / v.s); v.s = ns; draw();
  }
  function centreOn(o) {
    const r = canvases()[0].getBoundingClientRect();
    const s = Math.max(X.view.s, (X.fitS || X.view.s) * 4);
    X.view = { s, x: r.width / 2 - o.x * s, y: r.height / 2 - o.y * s };
    draw();
  }
  function draw() {
    cancelAnimationFrame(X.raf);
    X.raf = requestAnimationFrame(render);
  }

  /* ---------------------------------------------------------------- rendering */
  function starColour(bp) {
    if (bp === null || Number.isNaN(bp)) return "#e8ecf5";
    const stops = [[-0.4, [155, 176, 255]], [0.0, [202, 215, 255]], [0.6, [255, 246, 234]], [1.0, [255, 228, 196]], [1.6, [255, 196, 140]], [2.6, [255, 160, 110]]];
    let i = 0; while (i < stops.length - 2 && bp > stops[i + 1][0]) i++;
    const [a, ca] = stops[i], [b, cb] = stops[i + 1], t = Math.min(1, Math.max(0, (bp - a) / (b - a)));
    return `rgb(${ca.map((c, k) => Math.round(c + (cb[k] - c) * t)).join(",")})`;
  }

  function render() {
    const d = X.data;
    for (const c of [$("#exImg"), $("#exChart")]) {
      if (c.hidden) continue;
      const ctx = c.getContext("2d");
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = c.id === "exChart" ? "#03050a" : "#05070a";
      ctx.fillRect(0, 0, c.width, c.height);
      if (!d) continue;
      const k = dpr(), v = X.view;
      if (c.id === "exImg" && X.img) {
        ctx.setTransform(k * v.s, 0, 0, k * v.s, k * v.x, k * v.y);
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(X.img, 0, 0, d.W, d.H);
      }
      ctx.setTransform(k, 0, 0, k, 0, 0);
      const chart = c.id === "exChart";
      if (chart || X.mode === "overlay") drawSky(ctx, c, chart);
      drawHighlight(ctx);
    }
  }

  function drawSky(ctx, c, chart) {
    const d = X.data, v = X.view, W = c.width / dpr(), H = c.height / dpr();
    const sx = x => x * v.s + v.x, sy = y => y * v.s + v.y;
    const onScreen = (x, y, m = 20) => x > -m && x < W + m && y > -m && y < H + m;
    ctx.save();
    if (!chart) ctx.globalAlpha = X.opacity;
    if (chart) {                                        // frame of the image on the map
      ctx.strokeStyle = "#2a3346"; ctx.lineWidth = 1;
      ctx.strokeRect(sx(0), sy(0), d.W * v.s, d.H * v.s);
    }
    if (X.layers.grid) {
      ctx.strokeStyle = chart ? "#23304a" : "#7aa2ff55"; ctx.lineWidth = 1;
      ctx.fillStyle = chart ? "#5d6b88" : "#9fb6ffaa"; ctx.font = "11px system-ui, sans-serif";
      for (const L of d.grid) {
        ctx.beginPath();
        L.pts.forEach(([x, y], i) => i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y)));
        ctx.stroke();
        // label where the line enters the visible image area
        const p = L.pts.find(([x, y]) => x >= 0 && y >= 0 && x <= d.W && y <= d.H && onScreen(sx(x), sy(y), -8));
        if (p) ctx.fillText(L.label, sx(p[0]) + 3, sy(p[1]) - 3);
      }
    }
    const zoom = X.fitS ? v.s / X.fitS : 1;
    if (chart && X.layers.stars) {
      const st = d.stars, grow = Math.sqrt(Math.min(Math.max(zoom, 1), 6));
      for (let i = 0; i < st.x.length; i++) {
        const g = st.g[i]; if (g > X.mag) continue;
        const x = sx(st.x[i]), y = sy(st.y[i]); if (!onScreen(x, y)) continue;
        const r = Math.max(0.7, 0.75 * (X.mag + 1.2 - g)) * grow;
        ctx.fillStyle = starColour(st.bp_rp[i]);
        ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
      }
    } else if (!chart && X.layers.stars && zoom > 2.5) {
      ctx.strokeStyle = "#ffffff55"; ctx.lineWidth = 1;
      const st = d.stars;
      for (let i = 0; i < st.x.length; i++) {
        if (st.g[i] > X.mag) continue;
        const x = sx(st.x[i]), y = sy(st.y[i]); if (!onScreen(x, y)) continue;
        ctx.beginPath(); ctx.arc(x, y, 7, 0, 6.2832); ctx.stroke();
      }
    }
    // catalogued objects
    let nLabels = 0;
    const labels = [];
    for (const o of d.objects) {
      if (!visibleObj(o)) continue;
      const x = sx(o.x), y = sy(o.y);
      const col = COL[o.category];
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.3;
      if (o.ellipse) {
        const [a, b, t] = o.ellipse;
        if (!onScreen(x, y, Math.max(a, b) * v.s + 20)) continue;
        ctx.beginPath(); ctx.ellipse(x, y, Math.max(a * v.s, 4), Math.max(b * v.s, 3), t, 0, 6.2832); ctx.stroke();
      } else {
        if (!onScreen(x, y)) continue;
        if (o.category === "star" || o.category === "multiple" || o.category === "variable") {
          // thousands of catalogued stars: marked once zoomed in, named ones always
          if (!named(o) && zoom < 2.5) continue;
          const r = chart ? 6 : 8;
          ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.stroke();
          if (o.category === "multiple") { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.lineTo(x + r + 4, y); ctx.stroke(); }
          if (o.category === "variable") { ctx.beginPath(); ctx.arc(x, y, r + 3, -0.6, 0.6); ctx.stroke(); }
        } else if (o.category === "galaxy") {
          ctx.beginPath(); ctx.ellipse(x, y, 7, 3.5, -0.5, 0, 6.2832); ctx.stroke();
        } else if (o.category === "nebula") {
          ctx.strokeRect(x - 6, y - 6, 12, 12);
        } else {
          ctx.beginPath(); ctx.moveTo(x, y - 6); ctx.lineTo(x + 6, y); ctx.lineTo(x, y + 6); ctx.lineTo(x - 6, y); ctx.closePath(); ctx.stroke();
        }
      }
      if (X.layers.labels) {
        const lab = labelFor(o, zoom);
        if (lab) labels.push([lab, x, y, col, o.ellipse ? o.ellipse[1] * v.s : 8]);
      }
    }
    ctx.font = "12px system-ui, sans-serif";
    ctx.shadowColor = "#000"; ctx.shadowBlur = 3;
    for (const [lab, x, y, col, off] of labels) {
      if (nLabels++ > 180) break;
      ctx.fillStyle = col; ctx.fillText(lab, x + Math.min(off, 60) + 4, y - 4);
    }
    ctx.shadowBlur = 0;
    if (X.layers.compass) drawCompass(ctx, W, H);
    ctx.restore();
  }

  function named(o) { return !!(o.names?.length || o.messier || o.ngc); }
  function title(o) { return o.messier ? (o.names?.[0] ? `${o.messier} · ${o.names[0]}` : o.messier) : (o.names?.[0] || o.ngc || o.designation || o.id); }
  function labelFor(o, zoom) {
    if (named(o)) return o.messier || o.names?.[0] || o.ngc;
    if ((o.category === "galaxy" || o.category === "nebula") && zoom > 1.5) return o.id;
    if (zoom > 4 && o.designation) return o.designation;
    if (zoom > 8) return o.id;
    return null;
  }

  function drawCompass(ctx, W, H) {
    const f = X.data.field, cx = W - 46, cy = H - 46, L = 26;
    ctx.globalAlpha = 1; ctx.lineWidth = 2; ctx.font = "bold 11px system-ui, sans-serif";
    for (const [vec, lab, col] of [[f.north, "N", "#ff6b8a"], [f.east, "E", "#7fd3ff"]]) {
      ctx.strokeStyle = col; ctx.fillStyle = col;
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + vec[0] * L, cy + vec[1] * L); ctx.stroke();
      ctx.fillText(lab, cx + vec[0] * (L + 9) - 4, cy + vec[1] * (L + 9) + 4);
    }
  }

  function drawHighlight(ctx) {
    const h = X.pinned || X.hover; if (!h || !X.data) return;
    const p = posOf(h), v = X.view, x = p[0] * v.s + v.x, y = p[1] * v.s + v.y;
    ctx.save(); ctx.globalAlpha = 1; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(x, y, 12, 0, 6.2832); ctx.stroke();
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { ctx.beginPath(); ctx.moveTo(x + dx * 15, y + dy * 15); ctx.lineTo(x + dx * 22, y + dy * 22); ctx.stroke(); }
    ctx.restore();
  }
  const posOf = h => h.kind === "s" ? [X.data.stars.x[h.i], X.data.stars.y[h.i]] : [X.data.objects[h.i].x, X.data.objects[h.i].y];

  /* ---------------------------------------------------------------- pop-out */
  function colourWord(bp) {
    if (bp === null || Number.isNaN(bp)) return null;
    return bp < 0 ? "blue-white" : bp < 0.5 ? "white" : bp < 0.85 ? "yellowish-white (Sun-like)" : bp < 1.3 ? "yellow-orange" : bp < 2 ? "orange" : "red";
  }
  function brighter(g) {
    const f = Math.pow(10, 0.4 * (g - 6));
    return g > 6.5 ? `${num(f, f < 10 ? 1 : 0)}× fainter than the faintest stars visible to the naked eye` : "visible to the naked eye under dark skies";
  }
  function ly(v) { return v >= 1e9 ? `${num(v / 1e9, 2)} billion` : v >= 1e6 ? `${num(v / 1e6, 1)} million` : num(v, 0); }

  function popHTML(h) {
    const d = X.data;
    let o = null, s = null;
    if (h.kind === "s") { s = h.i; if (d.stars.simbad[s] >= 0) o = d.objects[d.stars.simbad[s]]; }
    else { o = d.objects[h.i]; if (o.gaia !== null) s = o.gaia; }
    const rows = [];
    const st = d.stars;
    const head = o ? title(o) : `Gaia DR3 ${st.source_id[s]}`;
    const aka = o ? [...(o.names || []).slice(o.messier ? 0 : 1), o.ngc, o.designation, o.id].filter((q, i, a) => q && q !== head && !head.includes(q) && a.indexOf(q) === i) : [];
    let type = o ? o.type : "Star";
    if (s !== null && st.sp_class[s]) type += ` · class ${st.sp_class[s]} (from its temperature)`;
    else if (o?.sp_type) type += ` · spectral type ${o.sp_type}`;
    if (o?.morph_type) type += ` · ${o.morph_type}`;
    if (s !== null) {
      const g = st.g[s];
      rows.push(["Brightness", `G = ${num(g, 2)} mag — ${brighter(g)}`]);
      const cw = colourWord(st.bp_rp[s]);
      if (cw) {
        let t = `appears ${cw}${st.teff[s] ? `; its surface is ≈ ${num(st.teff[s], 0)} K (the Sun: 5,772 K)` : ""}`;
        if (st.ebr[s] !== null && st.ebr[s] >= 0.15) t += ` — interstellar dust has reddened its light (E(BP−RP) ≈ ${num(st.ebr[s], 2)} mag)`;
        rows.push(["Colour", t]);
      }
      if (st.dist_ly[s]) {
        rows.push(["Distance", `≈ ${ly(st.dist_ly[s])} light-years (${st.dist_src[s] === 1 ? "Gaia parallax" : "Gaia spectro-photometric estimate"}) — the light you captured left it ${ly(st.dist_ly[s])} years ago`]);
        if (st.abs_g[s] !== null) {
          const L = Math.pow(10, -0.4 * (st.abs_g[s] - d.field.solar_abs_g));
          rows.push(["Luminosity", `≈ ${L >= 1 ? num(L, L < 10 ? 1 : 0) + "×" : "1/" + num(1 / L, 0) + " of"} the Sun's in Gaia's G band` +
            (st.ag[s] !== null ? ` (corrected for ${num(st.ag[s], 2)} mag of dust dimming)` : " (not corrected for dust dimming, so a lower limit)")]);
        }
      }
      const pm = Math.hypot(st.pmra[s] || 0, st.pmdec[s] || 0);
      if (st.pmra[s] !== null) rows.push(["Motion", `${num(pm, 1)} milliarcsec/yr across the sky${st.rv[s] !== null ? `; ${st.rv[s] < 0 ? "approaching" : "receding"} at ${num(Math.abs(st.rv[s]), 1)} km/s` : ""}`]);
      const fl = [st.variable[s] ? "Gaia saw its brightness vary" : null, st.nss[s] ? "Gaia found it is not a single star" : null].filter(Boolean);
      if (fl.length) rows.push(["Gaia", fl.join("; ")]);
    } else if (o) {
      if (o.vmag !== null && o.vmag !== undefined) rows.push(["Brightness", `V = ${num(o.vmag, 2)} mag — ${brighter(o.vmag)}`]);
    }
    if (o) {
      if (o.size_arcmin) rows.push(["Size", `${num(o.size_arcmin[0], 1)}′${o.size_arcmin[1] ? " × " + num(o.size_arcmin[1], 1) + "′" : ""} on the sky${o.size_ly ? ` — about ${ly(o.size_ly)} light-years across` : ""}`]);
      if (o.dist_ly && s === null) rows.push(["Distance", `≈ ${ly(o.dist_ly)} light-years (${o.dist_method})`]);
      if (o.z) rows.push(["Redshift", `z = ${num(o.z, 4)}${o.lookback_yr ? ` — the light you captured left it ${ly(o.lookback_yr)} years ago` : ""}`]);
      if (o.nbref) rows.push(["Studied in", `${num(o.nbref, 0)} scientific papers`]);
    }
    const ra = o ? o.ra : null, dec = o ? o.dec : null;
    const links = [];
    if (o) links.push(`<a href="https://simbad.cds.unistra.fr/simbad/sim-id?Ident=${encodeURIComponent(o.id)}" target="_blank" rel="noopener">SIMBAD</a>`);
    if (o && named(o)) links.push(`<a href="https://en.wikipedia.org/wiki/Special:Search?search=${encodeURIComponent(o.messier || o.names?.[0] || o.ngc)}" target="_blank" rel="noopener">Wikipedia</a>`);
    if (s !== null) links.push(`<a href="https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=I/355/gaiadr3&Source=${st.source_id[s]}" target="_blank" rel="noopener">Gaia DR3</a>`);
    return `<span class="close" data-close>✕</span><h4>${esc(head)}</h4>
      ${aka.length ? `<div class="muted small">also ${esc(aka.slice(0, 4).join(" · "))}</div>` : ""}
      <div class="ty">${esc(type)}</div>
      ${o?.note ? `<div class="note">${esc(o.note)}</div>` : ""}
      <div class="kv">${rows.map(([k, v]) => `<span>${k}</span><b>${esc(v)}</b>`).join("")}</div>
      ${links.length ? `<div class="small" style="margin-top:8px">${links.join(" · ")}</div>` : ""}`;
  }

  function showPop(h, sx, sy, pinned) {
    const p = $("#exPop");
    if (!h) { p.hidden = true; return; }
    p.innerHTML = popHTML(h);
    p.classList.toggle("pinned", !!pinned);
    p.hidden = false;
    const st = $("#exStage").getBoundingClientRect(), r = p.getBoundingClientRect();
    let x = sx + 18, y = sy + 18;
    if (x + r.width > st.width - 8) x = sx - r.width - 18;
    if (y + r.height > st.height - 8) y = Math.max(8, st.height - r.height - 8);
    p.style.left = Math.max(8, x) + "px"; p.style.top = y + "px";
    const c = $("[data-close]", p); if (c) c.onclick = () => { X.pinned = null; p.hidden = true; draw(); };
  }

  /* ---------------------------------------------------------------- side panel */
  function buildPanel() {
    const d = X.data, f = d.field;
    const fmtDeg = v => v >= 1 ? `${num(v, 2)}°` : `${num(v * 60, 1)}′`;
    $("#exField").innerHTML = `<h3>Field</h3><div class="exfield">
      <span>Constellation</span><b>${esc(f.constellation)}</b>
      <span>Centre</span><b>${esc(f.center_text)}</b>
      <span>Size</span><b>${fmtDeg(f.width_deg)} × ${fmtDeg(f.height_deg)} (${num(f.width_deg * f.height_deg / 0.0491, 0)}× the full Moon's area)</b>
      <span>Scale</span><b>${num(f.scale_arcsec, 2)}″ per pixel</b>
      <span>North</span><b>${num(f.north_angle_deg, 0)}° clockwise from up${f.mirrored ? " (mirrored)" : ""}</b>
      <span>Galactic</span><b>l ${num(f.galactic[0], 1)}°, b ${num(f.galactic[1], 1)}°${Math.abs(f.galactic[1]) < 10 ? " — in the Milky Way's disc" : ""}</b>
      <span>Stars</span><b>${num(f.n_stars, 0)} Gaia stars to G ${f.gmax}; half are detected down to G ≈ ${f.depth_g ?? "?"}</b>
      <span>Solution</span><b>${num(f.n_matched, 0)} stars matched, ${num(f.rms_arcsec, 2)}″ rms</b></div>`;
    const counts = f.counts || {};
    const lay = [["stars", "Gaia stars", "#e8ecf5", f.n_stars], ...Object.entries(d.categories).map(([k, v]) => [k, v, COL[k], counts[k] || 0]),
      ["grid", "RA / Dec grid"], ["labels", "Labels"], ["compass", "North / east arrows"]];
    $("#exLayers").innerHTML = lay.map(([k, lab, col, n]) => `<label><input type="checkbox" data-layer="${k}" ${X.layers[k] ? "checked" : ""}>
      ${col ? `<span class="sw" style="background:${col}"></span>` : ""}${esc(lab)}${n !== undefined ? `<span class="n">${num(n, 0)}</span>` : ""}</label>`).join("");
    $$("#exLayers [data-layer]").forEach(i => i.onchange = () => { X.layers[i.dataset.layer] = i.checked; draw(); renderList(); });
    renderList();
  }

  function renderList() {
    const d = X.data; if (!d) return;
    const q = $("#exSearch").value.trim().toLowerCase();
    const inImg = o => o.x >= 0 && o.y >= 0 && o.x <= d.W && o.y <= d.H;
    const match = o => !q || [o.id, o.type, o.ngc, o.messier, o.designation, ...(o.names || [])].some(s => s && s.toLowerCase().includes(q));
    const items = d.objects.map((o, i) => [o, i]).filter(([o]) => inImg(o) && visibleObj(o) && match(o));
    const byMag = (a, b) => (a[0].vmag ?? 99) - (b[0].vmag ?? 99);
    const groups = [
      ["Named objects", items.filter(([o]) => named(o))],
      ["Galaxies", items.filter(([o]) => o.category === "galaxy" && !named(o)).sort((a, b) => (b[0].nbref || 0) - (a[0].nbref || 0))],
      ["Nebulae & clusters", items.filter(([o]) => o.category === "nebula" && !named(o))],
      ["Variable stars", items.filter(([o]) => o.category === "variable" && !named(o)).sort(byMag)],
      ["Double & multiple stars", items.filter(([o]) => o.category === "multiple" && !named(o)).sort(byMag)],
      ["Stars", items.filter(([o]) => o.category === "star" && !named(o)).sort(byMag)],
      ["Other", items.filter(([o]) => o.category === "other" && !named(o)).sort((a, b) => (b[0].nbref || 0) - (a[0].nbref || 0))],
    ];
    const LIM = 60;
    $("#exList").innerHTML = groups.filter(g => g[1].length).map(([h, arr]) =>
      `<h5>${h} (${num(arr.length, 0)})</h5>` + arr.slice(0, LIM).map(([o, i]) =>
        `<div class="it" data-i="${i}"><span class="sw" style="background:${COL[o.category]};width:8px;height:8px;border-radius:50%;margin:0"></span>
         <b>${esc(title(o))}</b><span>${esc(o.type)}${o.vmag !== null && o.vmag !== undefined ? " · " + num(o.vmag, 1) : ""}</span></div>`).join("") +
      (arr.length > LIM ? `<div class="muted small" style="padding:4px 6px">+ ${num(arr.length - LIM, 0)} more — search to narrow</div>` : "")).join("")
      || `<p class="muted small">Nothing matches.</p>`;
    $$("#exList .it").forEach(el => el.onclick = () => {
      const i = +el.dataset.i, o = d.objects[i];
      X.pinned = { kind: "o", i }; centreOn(o);
      const v = X.view; showPop(X.pinned, o.x * v.s + v.x + (X.mode === "side" ? 0 : 0), o.y * v.s + v.y, true);
    });
  }

  /* ---------------------------------------------------------------- cursor coordinates */
  function skyAt(ix, iy) {
    const L = X.data.lookup, n = L.n;
    const fx = Math.min(Math.max(ix / L.W * n, 0), n - 1e-6), fy = Math.min(Math.max(iy / L.H * n, 0), n - 1e-6);
    const i = Math.floor(fx), j = Math.floor(fy), u = fx - i, w = fy - j;
    const bil = A => A[j][i] * (1 - u) * (1 - w) + A[j][i + 1] * u * (1 - w) + A[j + 1][i] * (1 - u) * w + A[j + 1][i + 1] * u * w;
    return [((bil(L.ra) % 360) + 360) % 360, bil(L.dec)];
  }
  function fmtRA(ra) { const h = ra / 15, m = (h % 1) * 60; return `${String(Math.floor(h)).padStart(2, "0")}h${String(Math.floor(m)).padStart(2, "0")}m${((m % 1) * 60).toFixed(1).padStart(4, "0")}s`; }
  function fmtDec(dec) { const a = Math.abs(dec), m = (a % 1) * 60; return `${dec < 0 ? "−" : "+"}${String(Math.floor(a)).padStart(2, "0")}°${String(Math.floor(m)).padStart(2, "0")}′${((m % 1) * 60).toFixed(0).padStart(2, "0")}″`; }

  /* ---------------------------------------------------------------- events */
  function bind() {
    $("#exSolve").onclick = solve;
    $("#exFit").onclick = fit;
    $("#exSize").onchange = () => X.data && load();
    $("#exOpacity").oninput = e => { X.opacity = +e.target.value; draw(); };
    $("#exMag").oninput = e => { X.mag = +e.target.value; $("#exMagOut").textContent = X.mag; draw(); };
    $("#exSearch").oninput = () => renderList();
    $$("[data-exmode]").forEach(b => b.onclick = () => {
      $$("[data-exmode]").forEach(x => x.classList.toggle("active", x === b));
      X.mode = b.dataset.exmode; layout(); fit();
    });
    window.addEventListener("resize", () => { if ($("#tab-explore").classList.contains("active")) { layout(); draw(); } });
    window.addEventListener("keydown", e => { if (e.key === "Escape" && X.pinned) { X.pinned = null; $("#exPop").hidden = true; draw(); } });
    for (const c of [$("#exImg"), $("#exChart")]) {
      let drag = null;
      const local = e => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
      const stageXY = e => { const r = $("#exStage").getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
      c.addEventListener("wheel", e => { e.preventDefault(); const [x, y] = local(e); zoomAt(e.deltaY < 0 ? 1.18 : 1 / 1.18, x, y); }, { passive: false });
      c.addEventListener("mousedown", e => { const [x, y] = local(e); drag = { x, y, vx: X.view.x, vy: X.view.y, moved: false }; });
      c.addEventListener("mousemove", e => {
        if (!X.data) return;
        const [x, y] = local(e);
        if (drag && (Math.abs(x - drag.x) + Math.abs(y - drag.y) > 3)) {
          drag.moved = true; X.view.x = drag.vx + x - drag.x; X.view.y = drag.vy + y - drag.y; draw(); return;
        }
        const ix = (x - X.view.x) / X.view.s, iy = (y - X.view.y) / X.view.s;
        if (ix >= 0 && iy >= 0 && ix <= X.data.W && iy <= X.data.H) {
          const [ra, dec] = skyAt(ix, iy);
          $("#exCursor").textContent = `RA ${fmtRA(ra)}  Dec ${fmtDec(dec)}`;
        } else $("#exCursor").textContent = "";
        if (X.pinned) return;
        const h = pick(ix, iy);
        const changed = JSON.stringify(h) !== JSON.stringify(X.hover);
        X.hover = h;
        if (h) showPop(h, ...stageXY(e), false); else $("#exPop").hidden = true;
        if (changed) draw();
      });
      c.addEventListener("mouseup", e => {
        if (drag && !drag.moved && X.data) {
          const [x, y] = local(e);
          const h = pick((x - X.view.x) / X.view.s, (y - X.view.y) / X.view.s);
          X.pinned = h;
          if (h) showPop(h, ...stageXY(e), true); else $("#exPop").hidden = true;
          draw();
        }
        drag = null;
      });
      c.addEventListener("mouseleave", () => { drag = null; if (!X.pinned) { X.hover = null; $("#exPop").hidden = true; draw(); } });
      c.addEventListener("dblclick", fit);
    }
  }
  bind();
  return { show, reset };
})();
