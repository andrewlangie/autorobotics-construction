// Fatigue & posture panel for the dashboard.
// Polls GET /api/polar-status (written by backend/polar_coach.py via polar_status.py)
// every 5 s and shows: fatigue level + HRV changes, posture level + angles,
// the last spoken message, and 10-minute trends. Plain JavaScript, no build step.

const POLL_MS = 5000;
const LINE = '#1d9bd8';            // validated against the dark panel surface
const GRID = 'rgba(148,163,184,0.18)';
const INK = '#cbd5e1';
const MUTED = '#9ca3af';

const root = document.getElementById('polar-coach-panel');

const css = `
#polar-coach-panel { padding: 16px; margin-bottom: 16px; }
#polar-coach-panel .pc-head { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:12px; flex-wrap:wrap; }
#polar-coach-panel h3 { margin:0; }
#polar-coach-panel .pc-grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
#polar-coach-panel .pc-card { background: rgba(255,255,255,0.03); border:1px solid rgba(148,163,184,0.12); border-radius:14px; padding:14px; }
#polar-coach-panel .pc-row { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-bottom:6px; }
#polar-coach-panel .pc-title { font-weight:700; font-size:15px; }
#polar-coach-panel .pc-cause { font-size:13px; color:#dbeafe; min-height:18px; }
#polar-coach-panel .pc-stats { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:8px; margin:10px 0; }
#polar-coach-panel .pc-stat { background: rgba(255,255,255,0.03); border-radius:10px; padding:8px 10px; border:1px solid rgba(148,163,184,0.10); }
#polar-coach-panel .pc-stat .k { font-size:11px; color:${MUTED}; }
#polar-coach-panel .pc-stat .v { font-size:18px; font-weight:700; margin-top:2px; font-variant-numeric: tabular-nums; }
#polar-coach-panel .pc-chart-title { font-size:12px; color:${MUTED}; margin:6px 0 4px; }
#polar-coach-panel svg { width:100%; height:150px; display:block; background: rgba(255,255,255,0.02); border-radius:10px; border:1px solid rgba(148,163,184,0.12); }
#polar-coach-panel .pc-msg { margin-top:16px; }
#polar-coach-panel .pc-note { font-size:12px; color:${MUTED}; }
#polar-coach-panel .pc-empty { color:${MUTED}; font-size:13px; padding:8px 0; }
#polar-coach-panel .pc-tip { position:absolute; pointer-events:none; background:#0b1220; border:1px solid #334155; border-radius:8px; padding:6px 8px; font-size:12px; color:#e5e7eb; white-space:nowrap; display:none; z-index:5; }
#polar-coach-panel .pc-chart { position:relative; }
#polar-coach-panel table.pc-table { width:100%; border-collapse:collapse; font-size:12px; margin-top:6px; }
#polar-coach-panel table.pc-table td, #polar-coach-panel table.pc-table th { border-bottom:1px solid rgba(148,163,184,0.12); padding:3px 6px; text-align:right; }
#polar-coach-panel details summary { cursor:pointer; font-size:12px; color:${MUTED}; margin-top:8px; }
@media (max-width: 1200px) { #polar-coach-panel .pc-grid { grid-template-columns: 1fr; } }
`;

function el(tag, attrs = {}, text) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

// Level -> existing dashboard badge classes. Text always says the level (never colour alone).
function badgeClass(level) {
  if (level == null) return 'badge';
  if (level <= 1) return 'badge normal';
  if (level === 2) return 'badge caution';
  return 'badge high';
}

const fmtPct = v => (v == null ? '–' : `${v > 0 ? '+' : ''}${v.toFixed(0)}%`);
const fmtNum = (v, d = 0, unit = '') => (v == null ? '–' : `${v.toFixed(d)}${unit}`);
const pretty = s => (s ? s.replace(/_/g, ' ').replace(/^\w/, c => c.toUpperCase()) : '');

function build() {
  root.classList.add('panel');
  const style = el('style'); style.textContent = css; document.head.appendChild(style);
  root.innerHTML = `
    <div class="pc-head">
      <div>
        <h3>Polar H10 · Fatigue &amp; posture</h3>
        <div class="pc-note">From <code>backend/polar_coach.py</code>. The strap allows one connection:
          use either this coach or the EXG panel above, not both.</div>
      </div>
      <div class="pill" id="pc-status">Checking…</div>
    </div>
    <div class="pc-grid">
      <div class="pc-card" id="pc-fatigue">
        <div class="pc-row"><span class="pc-title">Fatigue</span><span id="pc-f-badge" class="badge">–</span></div>
        <div class="pc-cause" id="pc-f-cause"></div>
        <div class="pc-stats">
          <div class="pc-stat"><div class="k">Heart rate</div><div class="v" id="pc-hr">–</div></div>
          <div class="pc-stat"><div class="k">RMSSD</div><div class="v" id="pc-rmssd">–</div></div>
          <div class="pc-stat"><div class="k">SDNN</div><div class="v" id="pc-sdnn">–</div></div>
        </div>
        <div class="pc-chart-title">Fatigue score, last 10 min (0 low – 1 very high)</div>
        <div class="pc-chart" id="pc-f-chart"></div>
      </div>
      <div class="pc-card" id="pc-posture">
        <div class="pc-row"><span class="pc-title">Posture</span><span id="pc-p-badge" class="badge">–</span></div>
        <div class="pc-cause" id="pc-p-cause"></div>
        <div class="pc-stats">
          <div class="pc-stat"><div class="k">Forward bend</div><div class="v" id="pc-fwd">–</div></div>
          <div class="pc-stat"><div class="k">Side bend</div><div class="v" id="pc-side">–</div></div>
          <div class="pc-stat"><div class="k">Holding still</div><div class="v" id="pc-still">–</div></div>
        </div>
        <div class="pc-chart-title">Forward bend, last 10 min (degrees; poor &gt; 20°, critical &gt; 45°)</div>
        <div class="pc-chart" id="pc-p-chart"></div>
      </div>
    </div>
    <div class="pc-msg">
      <div class="pc-row"><span class="pc-title">Last spoken feedback</span><span class="small" id="pc-msg-meta"></span></div>
      <div class="recommendation" id="pc-msg">Nothing spoken yet.</div>
      <div class="pc-note" id="pc-msg-goal" style="margin-top:6px"></div>
    </div>
    <details><summary>Show trend data as a table</summary><div id="pc-table"></div></details>`;
}

// Single-series line chart (SVG) with dashed threshold lines and a hover tooltip.
function lineChart(container, points, { key, yMax, thresholds, unit, digits }) {
  const W = container.clientWidth || 480, H = 150, L = 34, R = 10, T = 10, B = 22;
  const data = points.filter(p => p[key] != null);
  container.innerHTML = '';
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('role', 'img');
  const last = data.length ? data[data.length - 1] : null;
  svg.setAttribute('aria-label', last ? `Latest ${last[key].toFixed(digits)}${unit} at minute ${last.m.toFixed(0)}` : 'No data yet');
  container.appendChild(svg);
  const tip = el('div', { class: 'pc-tip' }); container.appendChild(tip);

  const add = (name, attrs, text) => {
    const n = document.createElementNS(svgNS, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text !== undefined) n.textContent = text;
    svg.appendChild(n);
    return n;
  };
  if (!data.length) {
    add('text', { x: W / 2, y: H / 2, fill: MUTED, 'font-size': 12, 'text-anchor': 'middle' }, 'Waiting for data…');
    return;
  }
  const mMax = data[data.length - 1].m, mMin = Math.max(data[0].m, mMax - 10);
  const span = Math.max(mMax - mMin, 1);
  const x = m => L + ((m - mMin) / span) * (W - L - R);
  const y = v => T + (1 - Math.min(v, yMax) / yMax) * (H - T - B);

  // recessive grid: baseline + thresholds
  add('line', { x1: L, x2: W - R, y1: y(0), y2: y(0), stroke: GRID });
  for (const t of thresholds) {
    add('line', { x1: L, x2: W - R, y1: y(t.v), y2: y(t.v), stroke: GRID, 'stroke-dasharray': '4 4' });
    add('text', { x: W - R - 2, y: y(t.v) - 3, fill: MUTED, 'font-size': 10, 'text-anchor': 'end' }, t.label);
  }
  add('text', { x: L - 4, y: y(yMax) + 4, fill: MUTED, 'font-size': 10, 'text-anchor': 'end' }, `${yMax}${unit}`);
  add('text', { x: L - 4, y: y(0) + 4, fill: MUTED, 'font-size': 10, 'text-anchor': 'end' }, `0`);
  add('text', { x: L, y: H - 6, fill: MUTED, 'font-size': 10 }, `min ${mMin.toFixed(0)}`);
  add('text', { x: W - R, y: H - 6, fill: MUTED, 'font-size': 10, 'text-anchor': 'end' }, `min ${mMax.toFixed(0)}`);

  // the line (breaks where data is missing)
  let d = '', prev = null;
  for (const p of data) {
    if (p.m < mMin) continue;
    d += `${prev && p.m - prev.m < 0.5 ? 'L' : 'M'}${x(p.m).toFixed(1)},${y(p[key]).toFixed(1)} `;
    prev = p;
  }
  add('path', { d, fill: 'none', stroke: LINE, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' });
  // latest point, labelled directly
  add('circle', { cx: x(last.m), cy: y(last[key]), r: 4, fill: LINE, stroke: '#111827', 'stroke-width': 2 });

  // hover: crosshair + tooltip on the nearest point
  const cross = add('line', { y1: T, y2: H - B, stroke: INK, 'stroke-width': 1, opacity: 0 });
  const dot = add('circle', { r: 4, fill: LINE, stroke: '#111827', 'stroke-width': 2, opacity: 0 });
  const hit = add('rect', { x: L, y: 0, width: W - L - R, height: H, fill: 'transparent' });
  hit.addEventListener('mousemove', ev => {
    const box = svg.getBoundingClientRect();
    const mx = ((ev.clientX - box.left) / box.width) * W;
    const m = mMin + ((mx - L) / (W - L - R)) * span;
    let best = data[0];
    for (const p of data) if (Math.abs(p.m - m) < Math.abs(best.m - m)) best = p;
    cross.setAttribute('x1', x(best.m)); cross.setAttribute('x2', x(best.m)); cross.setAttribute('opacity', 0.5);
    dot.setAttribute('cx', x(best.m)); dot.setAttribute('cy', y(best[key])); dot.setAttribute('opacity', 1);
    tip.textContent = `min ${best.m.toFixed(1)} · ${best[key].toFixed(digits)}${unit}`;
    tip.style.display = 'block';
    const px = (x(best.m) / W) * box.width;
    tip.style.left = `${Math.min(px + 10, box.width - 120)}px`;
    tip.style.top = '6px';
  });
  hit.addEventListener('mouseleave', () => {
    cross.setAttribute('opacity', 0); dot.setAttribute('opacity', 0); tip.style.display = 'none';
  });
}

function renderTable(history) {
  const rows = history.slice(-24).map(h => `<tr><td>${h.m.toFixed(1)}</td><td>${h.score ?? '–'}</td><td>${h.f_level ?? '–'}</td><td>${h.forward ?? '–'}</td><td>${h.p_level ?? '–'}</td></tr>`).join('');
  document.getElementById('pc-table').innerHTML =
    `<table class="pc-table"><tr><th>minute</th><th>fatigue score</th><th>fatigue level</th><th>forward °</th><th>posture level</th></tr>${rows}</table>`;
}

function setBadge(id, state) {
  const b = document.getElementById(id);
  if (state.status !== 'ok') { b.className = 'badge'; b.textContent = pretty(state.status || 'waiting'); return; }
  b.className = badgeClass(state.level);
  b.textContent = state.label;
}

function render(s) {
  const pill = document.getElementById('pc-status');
  if (!s.available) {
    pill.textContent = 'Coach not running';
    document.getElementById('pc-f-cause').textContent = s.reason;
    return;
  }
  const src = s.source === 'live' ? 'Live strap' : s.source === 'replay' ? 'Replay' : 'Simulation';
  pill.textContent = s.stale ? `${src} · not updating (${Math.round(s.age_s)} s)` : `${src} · minute ${Math.floor(s.minute)}`;

  const f = s.fatigue, p = s.posture;
  setBadge('pc-f-badge', f);
  setBadge('pc-p-badge', p);
  document.getElementById('pc-f-cause').textContent =
    f.status !== 'ok' ? pretty(f.status) + (f.status === 'measuring baseline' ? ' (first ~4 min)' : '')
      : f.cause ? `Cause: ${pretty(f.cause)}` : 'No fatigue pattern';
  document.getElementById('pc-p-cause').textContent =
    p.status !== 'ok' ? pretty(p.status) : p.cause ? `Cause: ${pretty(p.cause)}` : 'Neutral working position';
  document.getElementById('pc-hr').textContent = f.hr != null ? `${f.hr.toFixed(0)} bpm` : '–';
  document.getElementById('pc-rmssd').textContent = f.status === 'ok' ? fmtPct(f.rmssd_pct) : fmtNum(f.rmssd, 0, ' ms');
  document.getElementById('pc-sdnn').textContent = f.status === 'ok' ? fmtPct(f.sdnn_pct) : fmtNum(f.sdnn, 0, ' ms');
  document.getElementById('pc-fwd').textContent = fmtNum(p.forward_deg, 0, '°');
  document.getElementById('pc-side').textContent = fmtNum(p.side_deg, 0, '°');
  document.getElementById('pc-still').textContent = p.still_s ? `${(p.still_s / 60).toFixed(1)} min` : '–';

  const m = s.last_message;
  if (m) {
    document.getElementById('pc-msg').textContent = `“${m.message}”`;
    document.getElementById('pc-msg-meta').textContent = `${pretty(m.track)} · ${m.level_label} · ${m.cause} · minute ${Math.floor(m.minute)}`;
    document.getElementById('pc-msg-goal').textContent = `Goal: ${m.goal}  ·  Closed-loop action: ${m.action}`;
  }

  lineChart(document.getElementById('pc-f-chart'), s.history, {
    key: 'score', yMax: 1, unit: '', digits: 2,
    thresholds: [{ v: 0.25, label: 'medium' }, { v: 0.5, label: 'high' }, { v: 0.75, label: 'very high' }],
  });
  lineChart(document.getElementById('pc-p-chart'), s.history, {
    key: 'forward', yMax: 90, unit: '°', digits: 0,
    thresholds: [{ v: 20, label: 'poor' }, { v: 45, label: 'critical' }],
  });
  renderTable(s.history);
}

async function poll() {
  try {
    const r = await fetch(`./api/polar-status?t=${Date.now()}`, { cache: 'no-store' });
    render(await r.json());
  } catch (e) {
    document.getElementById('pc-status').textContent = 'Dashboard server not reachable';
  }
}

if (root) {
  build();
  poll();
  setInterval(poll, POLL_MS);
}
