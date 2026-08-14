/* Curling Arena v2 — touch-first client. Views: home / game / replay.
   Modes: vs champion, pass-&-play (hot seat), online (share link, poll sync).
   Coordinates from the API are compact meters: along (0=button, − = in front /
   guard side, + = behind), lateral (+ = right). Board drawn portrait, guards up. */
"use strict";

const $ = (id) => document.getElementById(id);
const S = {
  view: "home", ends: 4, match: null, coach: localStorage.coach === "1",
  mode: "draw", hitAct: "remove", target: null, hitSlot: null, tapTarget: null,
  heat: null, busy: false, replay: null, step: 0, solved: null,
  mySides: new Set(["A"]), online: false, seenThrows: 0, pollTimer: null,
  rHeatOn: false, rHeatCache: {}, pendingAdvance: null,
  params: [2.50, 0.0, 7.0, 0.0], actionConfig: null,
  noiseEnabled: true, noiseScales: [1, 1, 1, 1],
  scenario: null, scenarioStones: [], scenarioTool: "A",
};
const APP_VERSION = "2.4.0";
const PLAY_RATE = 5;          // fixed playback: sim-seconds per real-second (no normalization)
const REPLAY_RATE = 6;
const _tele = [];
function tele(ev) {
  ev.t = Date.now(); ev.v = APP_VERSION; ev.ua = (navigator.userAgent || "").slice(0, 80);
  _tele.push(ev);
  console.log("[anim]", JSON.stringify(ev));
  if (_tele.length >= 4) {
    const batch = _tele.splice(0);
    fetch("/api/client_log", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ events: batch }) }).catch(() => {});
  }
}
let _busyTimer = null, _busyT0 = 0, _busyLabel = "";
function busyShow(label) {
  _busyLabel = label; _busyT0 = Date.now();
  const el = $("busy");
  el.textContent = label; el.classList.remove("hidden");
  clearInterval(_busyTimer);
  _busyTimer = setInterval(() => {
    const s = Math.round((Date.now() - _busyT0) / 1000);
    if (s >= 4) el.textContent = `${_busyLabel} ${s}s`;
  }, 1000);
}
function busyHide() { clearInterval(_busyTimer); _busyTimer = null; $("busy").classList.add("hidden"); }

/* ---------------- api ---------------- */
async function api(path, opts = {}) {
  const ctl = new AbortController();
  const kill = setTimeout(() => ctl.abort(), opts.timeoutMs || 90000);
  let r;
  try {
    r = await fetch(path, {
      headers: { "content-type": "application/json" },
      signal: ctl.signal,
      ...opts, body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
  } catch (e) {
    clearTimeout(kill);
    throw new Error(e.name === "AbortError" ? "Request timed out — check your connection" : e.message);
  }
  clearTimeout(kill);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail; } catch (e) { /* keep statusText */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}
function toast(msg, ms = 2600) {
  const t = $("toast");
  t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.add("hidden"), ms);
}

/* ---------------- board rendering ---------------- */
const VIEW = { latHalf: 2.55, alongTop: -6.9, alongBot: 2.45 };
const TEAM_FILL = { A: "#d33a2f", B: "#e8b71a" };
const TEAM_NAME = { A: "Red", B: "Yellow" };
const RINGS = [[1.83, "#3f7fbf"], [1.22, "#f5f7f9"], [0.61, "#d24646"], [0.15, "#f5f7f9"]];

function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const ratio = (VIEW.alongBot - VIEW.alongTop) / (2 * VIEW.latHalf);
  const parentW = (cv.parentElement ? cv.parentElement.clientWidth : cv.clientWidth) || cv.clientWidth;
  const maxH = Math.max(420, (window.innerHeight || 900) * 0.78);
  let w = parentW, h = w * ratio;
  if (h > maxH) { h = maxH; w = h / ratio; }         // tall view: fit the screen
  cv.style.width = w + "px"; cv.style.height = h + "px";
  cv.style.marginLeft = "auto"; cv.style.marginRight = "auto";
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
function mkMap(cv) {
  const w = parseFloat(cv.style.width) || cv.clientWidth, ppm = w / (2 * VIEW.latHalf);
  return {
    px: (lat) => (lat + VIEW.latHalf) * ppm,
    py: (along) => (along - VIEW.alongTop) * ppm,
    invLat: (x) => x / ppm - VIEW.latHalf,
    invAlong: (y) => y / ppm + VIEW.alongTop,
    ppm,
  };
}
function heatColor(v, lo, mid, hi) {
  // diverging around the median spot value: RED = better (hot), BLUE = worse (cold).
  // Near-median cells stay almost transparent so only real signal colors the ice.
  let t;
  if (v >= mid) {
    t = hi > mid ? Math.min(1, (v - mid) / (hi - mid)) : 0;
    return `rgba(198,44,34,${0.06 + 0.66 * t})`;
  }
  t = mid > lo ? Math.min(1, (mid - v) / (mid - lo)) : 0;
  return `rgba(23,111,208,${0.06 + 0.66 * t})`;
}
function drawBoard(cv, board, opts = {}) {
  const ctx = setupCanvas(cv), m = mkMap(cv);
  const w = parseFloat(cv.style.width) || cv.clientWidth, h = parseFloat(cv.style.height);
  ctx.fillStyle = "#eef4fa"; ctx.fillRect(0, 0, w, h);
  for (const [r, col] of RINGS) {
    ctx.beginPath(); ctx.arc(m.px(0), m.py(0), r * m.ppm, 0, 2 * Math.PI);
    ctx.fillStyle = col; ctx.globalAlpha = opts.heat ? 0.35 : 0.9; ctx.fill();
    ctx.globalAlpha = 1;
  }
  if (opts.heat) {
    const { alongs, lats, v } = opts.heat;
    const vals = v.flat().filter((x) => x != null).sort((a, b) => a - b);
    if (vals.length) {
      const q = (p) => vals[Math.floor(p * (vals.length - 1))];
      const lo = q(0.05), mid = q(0.5), hi = q(0.95);
      const ca = (alongs[1] - alongs[0]) * m.ppm, cl = (lats[1] - lats[0]) * m.ppm;
      for (let ai = 0; ai < alongs.length; ai++)
        for (let li = 0; li < lats.length; li++) {
          const val = v[ai][li];
          if (val == null) continue;
          ctx.fillStyle = heatColor(val, lo, mid, hi);
          ctx.fillRect(m.px(lats[li]) - cl / 2, m.py(alongs[ai]) - ca / 2, cl + 0.5, ca + 0.5);
        }
    }
  }
  const w2 = parseFloat(cv.style.width);
  ctx.strokeStyle = "rgba(40,70,110,.25)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(m.px(0), 0); ctx.lineTo(m.px(0), h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, m.py(0)); ctx.lineTo(w2, m.py(0)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, m.py(-1.974)); ctx.lineTo(w2, m.py(-1.974)); ctx.stroke();
  ctx.strokeStyle = "rgba(178,60,60,.5)"; ctx.lineWidth = 2.5;       // hog line
  ctx.beginPath(); ctx.moveTo(0, m.py(-6.4)); ctx.lineTo(w2, m.py(-6.4)); ctx.stroke();
  if (opts.traj && opts.traj.frames && opts.traj.frames.length > 1) {
    const slot = opts.traj.stone_slot;
    ctx.save(); ctx.setLineDash([9, 7]); ctx.lineWidth = 2.5;
    ctx.strokeStyle = "rgba(45,110,170,.8)"; ctx.beginPath();
    let started = false;
    for (const f of opts.traj.frames) {
      const p = f[slot];
      if (!p || p[0] == null) continue;
      if (!started) { ctx.moveTo(m.px(p[1]), m.py(p[0])); started = true; }
      else ctx.lineTo(m.px(p[1]), m.py(p[0]));
    }
    ctx.stroke(); ctx.restore();
  }
  const R = 0.145 * m.ppm;
  for (const s of board || []) {
    ctx.beginPath(); ctx.arc(m.px(s.lateral), m.py(s.along), R, 0, 2 * Math.PI);
    ctx.fillStyle = TEAM_FILL[s.team]; ctx.fill();
    ctx.lineWidth = s.slot === opts.hilite ? 3.5 : 1.6;
    ctx.strokeStyle = s.slot === opts.hilite ? "#0ea5e9" : "rgba(20,25,35,.8)";
    ctx.stroke();
    ctx.beginPath(); ctx.arc(m.px(s.lateral), m.py(s.along), R * 0.45, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(255,255,255,.55)"; ctx.fill();
  }
  if (opts.predicted) {
    for (const s of opts.predicted) {
      ctx.beginPath(); ctx.arc(m.px(s.lateral), m.py(s.along), R, 0, 2 * Math.PI);
      ctx.setLineDash([4, 4]); ctx.lineWidth = 2;
      ctx.strokeStyle = TEAM_FILL[s.team]; ctx.stroke(); ctx.setLineDash([]);
    }
  }
  if (opts.target) {
    const [al, la] = opts.target;
    ctx.strokeStyle = "#0b76c4"; ctx.lineWidth = 2.5;
    const r = 11;
    ctx.beginPath(); ctx.arc(m.px(la), m.py(al), r, 0, 2 * Math.PI); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(m.px(la) - r - 6, m.py(al)); ctx.lineTo(m.px(la) + r + 6, m.py(al));
    ctx.moveTo(m.px(la), m.py(al) - r - 6); ctx.lineTo(m.px(la), m.py(al) + r + 6);
    ctx.stroke();
  }
  return m;
}

/* Real-time-paced throw animation. traj.dt = sim seconds per frame. */
function animateTraj(cv, traj, finalBoard, rate) {
  return new Promise((resolve) => {
    let done = false, watchdog = null, frames = null, animT0 = 0, ticks = 0;
    const finish = (why) => {
      if (done) return;
      done = true;
      clearTimeout(watchdog);
      try { if (finalBoard) drawBoard(cv, finalBoard, {}); } catch (e) { /* draw-only */ }
      if (traj && traj.frames)
        tele({ ev: "anim", slot: traj.stone_slot, nAll: traj.frames.length,
               nVis: frames ? frames.length : -1, ticks,
               ms: Math.round(performance.now() - (animT0 || performance.now())), why: why || "?" });
      resolve();
    };
    if (!traj || !traj.frames || traj.frames.length < 2) { finish("no-traj"); return; }
    frames = traj.frames;
    const dt = traj.dt || 0.1;
    // a draw spends its first seconds far above the visible view — skip ahead
    // to just before the thrown stone enters the board so motion is visible
    const slot = traj.stone_slot;
    if (slot != null) {
      const first = frames.findIndex((f) => {
        const p = f[slot];
        return p && p[0] != null && p[0] > VIEW.alongTop - 0.8;
      });
      if (first > 3) frames = frames.slice(first - 3);
    }
    const simRate = rate || PLAY_RATE;
    const durMs = (frames.length * dt / simRate) * 1000;
    animT0 = performance.now();
    // the promise MUST settle even if rAF stalls (hidden tab, throttling, a
    // rendering exception) — a stuck animation used to freeze "Throwing…"
    watchdog = setTimeout(() => finish("watchdog"), durMs + 3000);
    const t0 = performance.now();
    const tick = (now) => {
      if (done) return;
      try {
        // Safari's rAF timestamp can PRECEDE a performance.now() captured in the
        // same frame -> negative idx -> frames[-1] undefined -> instant finish.
        const idx = Math.max(0, Math.floor(((now - t0) / 1000) * simRate / dt));
        const f = frames[Math.min(idx, frames.length - 1)] || frames[0];
        const stones = [];
        for (let slot = 0; slot < f.length; slot++) {
          const p = f[slot];
          if (p && p[0] != null) stones.push({ slot, team: slot < 6 ? "A" : "B", along: p[0], lateral: p[1] });
        }
        drawBoard(cv, stones, {});
        ticks++;
        if (idx >= frames.length + 2) { finish("complete"); return; }
      } catch (e) { finish("draw-error:" + (e && e.message ? e.message.slice(0, 60) : "?")); return; }
      if (document.hidden) setTimeout(() => tick(performance.now()), 120);
      else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}
function animateThrow(cv, result, rate = PLAY_RATE) {
  return animateTraj(cv, result?.trajectory, result?.board, rate);
}

/* ---------------- view switching ---------------- */
function show(view) {
  S.view = view;
  for (const v of ["home", "game", "sandbox", "replay"])
    $(`view-${v}`).classList.toggle("hidden", v !== view);
  $("homeBtn").classList.toggle("hidden", view === "home");
  $("scorechip").classList.toggle("hidden", view !== "game");
  if (view !== "game") stopPolling();
  if (view === "home") { loadMatches(); loadScenarios(); }
  if (view === "sandbox") drawSandbox();
}

/* ---------------- home ---------------- */
async function loadMatches() {
  try {
    const d = await api("/api/matches");
    const rows = d.matches.filter((x) => Object.values(x.players).includes("human") ||
                                          x.status === "finished").slice(0, 12);
    if (!rows.length) { $("matchList").innerHTML = '<p class="hint">No matches yet — play one!</p>'; return; }
    $("matchList").innerHTML = "";
    for (const r of rows) {
      const b = document.createElement("button");
      b.className = "matchrow";
      const la = r.labels?.A || r.players.A, lb = r.labels?.B || r.players.B;
      const created = typeof r.created === "number" ? r.created * 1000 : r.created;
      const when = new Date(created).toLocaleDateString(undefined, { month: "short", day: "numeric" });
      const live = r.status === "in_progress";
      b.innerHTML = `<span class="who"><span class="names">${la} vs ${lb}</span>
        <span class="sub">${when} · end ${r.end}/${r.ends_scheduled}</span></span>
        <span class="score">${r.totals.A}:${r.totals.B}</span>
        <span class="tag">${live ? "continue ▶" : "replay"}</span>`;
      b.onclick = () => live && Object.values(r.players).includes("human")
        ? resumeMatch(r.id) : openReplay(r.id);
      $("matchList").appendChild(b);
    }
  } catch (e) { $("matchList").innerHTML = `<p class="hint">${e.message}</p>`; }
}

async function loadScenarios() {
  try {
    const d = await api("/api/scenarios");
    const host = $("scenarioList"); host.innerHTML = "";
    if (!d.scenarios.length) {
      host.innerHTML = '<p class="hint">No saved positions yet.</p>'; return;
    }
    for (const sc of d.scenarios.slice(0, 12)) {
      const b = document.createElement("button"); b.className = "matchrow";
      b.innerHTML = `<span class="who"><span class="names">${escapeHtml(sc.name)}</span>` +
        `<span class="sub">End ${sc.end} · next throw ${sc.throw} · ${sc.stones.length} stones</span></span>` +
        `<span class="tag">edit ▶</span>`;
      b.onclick = () => openSandbox(sc);
      host.appendChild(b);
    }
  } catch (e) { $("scenarioList").innerHTML = `<p class="hint">${e.message}</p>`; }
}
function escapeHtml(v) {
  return String(v).replace(/[&<>'"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
    "'": "&#39;", '"': "&quot;" })[c]);
}

async function loadArenaConfig() {
  try {
    const d = await api("/api/config");
    S.actionConfig = d.action;
    document.querySelectorAll(".param-range").forEach((el) => {
      const i = +el.dataset.i; el.min = d.action.low[i]; el.max = d.action.high[i];
    });
    if (!S.params || S.params.length !== 4) S.params = d.action.defaults.map(Number);
    syncParamControls();
  } catch (e) { /* bundled limits remain usable offline */ }
}

function tabToken() {
  if (!sessionStorage.arenaTok)
    sessionStorage.arenaTok = Math.random().toString(36).slice(2) + Date.now().toString(36);
  return sessionStorage.arenaTok;
}
async function claimSeat(mid) {
  // per-TAB token via sessionStorage (localStorage is shared across tabs and
  // caused both players landing on the same color); server assigns the seat.
  const d = await api(`/api/match/${mid}/claim`, { method: "POST", body: { token: tabToken() } });
  return d.side;   // null => spectator
}
function sidesFor(match) {
  const humans = Object.keys(match.players).filter((s) => match.players[s] === "human");
  if (humans.length <= 1) return new Set(humans);
  return new Set(humans);   // hot-seat default: this device controls both
}

async function newMatch(kind) {
  try {
    const players = kind === "champ" ? { A: "human", B: "champion" } : { A: "human", B: "human" };
    const labels = kind === "champ" ? { A: "You", B: "Champion" } : { A: "Red", B: "Yellow" };
    const d = await api("/api/match", { method: "POST", body: {
      players, labels, ends: S.ends, first_hammer: "random", mode: kind,
      noise: S.noiseEnabled, noise_scales: S.noiseScales } });
    S.match = d.match;
    adoptNoise(d.match);
    S.online = kind === "online";
    if (kind === "online") {
      const side = await claimSeat(d.match.id);        // creator -> "A"
      S.mySides = new Set(side ? [side] : []);
      $("shareLink").value = `${location.origin}/join/${d.match.id}`;
      $("shareModal").classList.remove("hidden");
    } else {
      S.mySides = sidesFor(d.match);
    }
    S.seenThrows = countThrows(d.match);
    resetShot(); show("game"); renderGame();
    maybePowerPlay();
    if (S.online) startPolling();
  } catch (e) { toast(e.message); }
}
async function resumeMatch(mid) {
  try {
    const d = await api(`/api/match/${mid}`);
    S.match = d.match;
    adoptNoise(d.match);
    const humans = Object.values(d.match.players).filter((p) => p === "human").length;
    S.online = humans === 2 && d.match.mp_mode === "online";
    if (S.online) {
      const side = await claimSeat(mid);
      S.mySides = new Set(side ? [side] : []);
      if (!side) toast("Both seats taken — watching live", 3200);
    } else {
      S.mySides = sidesFor(d.match);
    }
    S.seenThrows = countThrows(d.match);
    resetShot(); show("game"); renderGame(); maybePowerPlay();
    if (S.online) startPolling();
  } catch (e) { toast(e.message); }
}
async function joinFromLink(mid) {
  try {
    const d = await api(`/api/match/${mid}`);
    history.replaceState(null, "", "/");
    S.match = d.match; S.online = true;
    adoptNoise(d.match);
    const side = await claimSeat(mid);
    S.mySides = new Set(side ? [side] : []);
    S.seenThrows = countThrows(d.match);
    resetShot(); show("game"); renderGame(); maybePowerPlay(); startPolling();
    toast(side ? `You are ${TEAM_NAME[side]} — good curling!` : "Both seats taken — watching live", 3600);
  } catch (e) { toast("Couldn't join: " + e.message, 4000); show("home"); }
}

function adoptNoise(match) {
  S.noiseEnabled = match.noise !== false;
  S.noiseScales = (match.noise_scales || [1, 1, 1, 1]).map(Number);
  syncNoiseControls();
}

/* ---------------- sandbox scenario editor ---------------- */
function openSandbox(sc = null) {
  S.scenario = sc ? { ...sc } : null;
  S.scenarioStones = (sc?.stones || []).map((x) => ({ ...x }));
  S.scenarioTool = "A";
  $("scenarioName").value = sc?.name || "Untitled position";
  $("scenarioEnd").value = sc?.end || 1;
  $("scenarioThrow").value = sc?.throw || 1;
  $("scenarioHammer").value = sc?.hammer || "B";
  $("scenarioScoreA").value = sc?.totals?.A || 0;
  $("scenarioScoreB").value = sc?.totals?.B || 0;
  $("scenarioEnds").value = Math.max(S.ends, sc?.end || 1);
  document.querySelectorAll(".stone-tool").forEach((b) => b.classList.toggle("sel", b.dataset.tool === "A"));
  show("sandbox"); drawSandbox();
}
function scenarioTurn() {
  const hammer = $("scenarioHammer").value;
  const first = hammer === "A" ? "B" : "A";
  return ((Math.max(1, +$("scenarioThrow").value) - 1) % 2) ? hammer : first;
}
function drawSandbox() {
  if (!$("sboard")) return;
  drawBoard($("sboard"), S.scenarioStones, {});
  const turn = scenarioTurn();
  $("scenarioHint").textContent = `${TEAM_NAME[turn]} throws next · ${S.scenarioStones.length} stones placed. ` +
    "Choose a team and tap the sheet; use Remove to erase a stone.";
}
function sandboxTapped(al, la) {
  if (S.view !== "sandbox") return;
  let nearest = null, best = 0.24;
  for (const s of S.scenarioStones) {
    const d = Math.hypot(s.along - al, s.lateral - la);
    if (d < best) { best = d; nearest = s; }
  }
  if (S.scenarioTool === "erase") {
    if (nearest) S.scenarioStones = S.scenarioStones.filter((s) => s.slot !== nearest.slot);
    drawSandbox(); return;
  }
  if (al < -6.7056 || al > 1.974 || Math.abs(la) > 2.23) {
    toast("Place stone centers inside the playable sheet", 2600); return;
  }
  if (nearest || S.scenarioStones.some((s) => Math.hypot(s.along - al, s.lateral - la) < 0.290)) {
    toast("Stones cannot overlap", 2200); return;
  }
  const lo = S.scenarioTool === "A" ? 0 : 6, hi = lo + 6;
  const used = new Set(S.scenarioStones.map((s) => s.slot));
  const slot = Array.from({ length: 6 }, (_, i) => lo + i).find((i) => !used.has(i));
  if (slot == null || slot >= hi) { toast(`${TEAM_NAME[S.scenarioTool]} already has six stones`); return; }
  S.scenarioStones.push({ slot, team: S.scenarioTool,
    along: +al.toFixed(4), lateral: +la.toFixed(4) });
  drawSandbox();
}
function scenarioPayload() {
  return {
    name: $("scenarioName").value.trim() || "Untitled position",
    end: Math.max(1, Math.min(20, +$("scenarioEnd").value || 1)),
    throw: Math.max(1, Math.min(10, +$("scenarioThrow").value || 1)),
    hammer: $("scenarioHammer").value,
    totals: { A: Math.max(0, +$("scenarioScoreA").value || 0),
      B: Math.max(0, +$("scenarioScoreB").value || 0) },
    stones: S.scenarioStones,
  };
}
async function saveScenario() {
  const update = S.scenario?.id;
  const d = await api(update ? `/api/scenario/${update}` : "/api/scenario", {
    method: update ? "PUT" : "POST", body: scenarioPayload(),
  });
  S.scenario = d.scenario;
  S.scenarioStones = d.scenario.stones.map((s) => ({ ...s }));
  return d.scenario;
}
async function saveScenarioOnly() {
  try { await saveScenario(); toast("Sandbox position saved"); drawSandbox(); }
  catch (e) { toast(e.message, 4000); }
}
async function playScenario() {
  $("scenarioPlay").disabled = true;
  try {
    const sc = await saveScenario();
    const me = $("scenarioPlayAs").value, other = me === "A" ? "B" : "A";
    const players = { [me]: "human", [other]: "champion" };
    const labels = { [me]: "You", [other]: "Champion" };
    const ends = Math.max(sc.end, Math.min(20, +$("scenarioEnds").value || S.ends));
    const d = await api(`/api/scenario/${sc.id}/play`, { method: "POST", body: {
      players, labels, ends, noise: S.noiseEnabled, noise_scales: S.noiseScales,
    }, timeoutMs: 180000 });
    S.match = d.match; S.online = false; S.mySides = new Set([me]);
    S.seenThrows = countThrows(d.match); adoptNoise(d.match);
    resetShot(); show("game"); renderGame(); maybePowerPlay();
  } catch (e) { toast(e.message, 4500); }
  finally { $("scenarioPlay").disabled = false; }
}

/* ---------------- game ---------------- */
function countThrows(m) { return m.ends.reduce((a, e) => a + (e.throws ? e.throws.length : 0), 0); }
function humanOnTurn() {
  const m = S.match;
  return m && m.status === "in_progress" && S.mySides.has(m.turn.team);
}
function resetShot() {
  _previewSeq++;
  S.target = null; S.hitSlot = null; S.tapTarget = null; S.heat = null; S.solved = null;
  $("heatlegend").classList.add("hidden");
}
function curEnd() { return S.match.ends[S.match.ends.length - 1]; }

function formatParam(i, v) {
  if (i === 0) return `${v.toFixed(3)} m/s`;
  if (i === 1) return `${v >= 0 ? "+" : ""}${v.toFixed(4)} rad`;
  if (i === 2) return `${v >= 0 ? "+" : ""}${v.toFixed(1)} rad/s`;
  return `${v >= 0 ? "+" : ""}${v.toFixed(3)} m`;
}
function syncParamControls() {
  document.querySelectorAll(".param-range").forEach((el) => {
    const i = +el.dataset.i;
    el.value = S.params[i];
    $(`paramV${i}`).textContent = formatParam(i, S.params[i]);
  });
}
function setParams(action) {
  if (!action || action.length !== 4) return;
  S.params = action.map(Number);
  syncParamControls();
}
function selectShotMode(mode) {
  S.mode = mode;
  document.querySelectorAll(".modechip").forEach((x) => x.classList.toggle("sel", x.dataset.m === mode));
  $("hitopts").classList.toggle("hidden", mode !== "hit");
}
function syncNoiseControls() {
  if (!$("setupNoiseOn")) return;
  $("setupNoiseOn").checked = S.noiseEnabled;
  $("gameNoiseOn").checked = S.noiseEnabled;
  for (const cls of ["setup-noise", "game-noise"])
    document.querySelectorAll(`.${cls}`).forEach((el) => {
      const i = +el.dataset.i; el.value = S.noiseScales[i];
      $(`${cls === "setup-noise" ? "setup" : "game"}NoiseV${i}`).textContent = `${S.noiseScales[i].toFixed(1)}×`;
    });
  const allOne = S.noiseScales.every((v) => Math.abs(v - 1) < 1e-6);
  $("gameNoiseSummary").textContent = `${S.noiseEnabled ? "on" : "off"} · ${allOne ? "1× each" : S.noiseScales.map((v) => v.toFixed(1) + "×").join("/")}`;
}
let _noiseSaveTimer = null;
function queueNoiseSave() {
  syncNoiseControls();
  if (!S.match || S.view !== "game") return;
  clearTimeout(_noiseSaveTimer);
  _noiseSaveTimer = setTimeout(async () => {
    try {
      const d = await api(`/api/match/${S.match.id}/noise`, { method: "POST",
        body: { enabled: S.noiseEnabled, scales: S.noiseScales } });
      S.match = d.match; toast("Execution noise updated", 1500);
    } catch (e) { toast(e.message); }
  }, 350);
}

function renderGame() {
  const m = S.match;
  if (!m) return;
  $("scA").textContent = m.totals.A; $("scB").textContent = m.totals.B;
  const t = m.turn;
  if (m.status === "finished") {
    $("statusbar").innerHTML = `<b>Match over</b> — ${m.totals.A}:${m.totals.B}`;
  } else {
    const mine = S.mySides.has(t.team);
    const who = m.players[t.team] === "champion" ? "champion…"
      : mine ? (S.mySides.size > 1 ? `<b>${TEAM_NAME[t.team]}'s throw</b>` : "<b>your throw</b>")
      : `waiting for ${TEAM_NAME[t.team]}…`;
    const mode = curEnd().mode;
    $("statusbar").innerHTML =
      `End <b>${t.end}</b>/${m.ends_scheduled} · Throw <b>${t.throw}</b>/10 · ` +
      `<span class="dot ${t.hammer === "A" ? "red" : "yel"}"></span> hammer` +
      (mode !== "standard" ? ` · <b>${mode.replace("pp_", "power play ")}</b>` : "") +
      ` · ${who}`;
  }
  drawBoard($("board"), m.board, {
    target: S.mode === "draw" ? S.target : S.tapTarget,
    hilite: S.hitSlot, heat: S.heat,
    predicted: S.solved?.preview?.predicted_board || null,
  });
  $("undoBtn").classList.toggle("hidden", S.online);
  syncParamControls();
  syncNoiseControls();
  updateShotUI();
  renderCoach();
}
function updateShotUI() {
  const my = humanOnTurn();
  $("shotbar").style.opacity = my ? 1 : 0.55;
  const rejectedShot = S.solved?.solver?.solvable === false;
  const ready = my && !S.busy &&
    ((S.mode === "params" && S.solved) || (S.mode === "draw" && S.target) ||
     (S.mode === "hit" && S.hitSlot != null && (S.hitAct === "remove" || S.tapTarget))) &&
    !rejectedShot;
  $("throwBtn").disabled = !ready;
  $("previewBtn").classList.toggle("hidden", !(S.coach && ready));
  $("hint").textContent = !my ? (S.match.status === "finished" ? "" : (S.online && S.mySides.size === 0 ? "Watching live…" : "Waiting for the other team…"))
    : S.mode === "params" ? (S.solved ? "Parameter throw previewed — Throw when ready."
       : "Adjust a delivery parameter; the predicted path will update.")
    : S.mode === "draw" ? (S.target ? "Target set — Throw when ready." : "Tap the ice where the stone should stop.")
    : S.hitSlot == null ? "Tap a stone on the board."
    : S.hitAct === "remove" ? "Take-out selected — Throw when ready."
    : S.hitAct === "roll" ? (S.tapTarget ? "Roll target set — Throw when ready."
       : "Now tap where YOUR stone should roll after the hit.")
    : (S.tapTarget ? "Target set — Throw when ready (tap another stone to switch)."
       : "Now tap where that stone should end up (even right beside another stone).");
}

function boardTapped(al, la) {
  if (S.view !== "game" || !humanOnTurn() || S.busy) return;
  if (S.mode === "params") return;
  _previewSeq++;                            // invalidate any solve still in flight
  if (S.mode === "draw") { S.target = [al, la]; S.solved = null; }
  else {
    // Once a stone is picked in tap mode, the next tap is the TARGET — except a
    // tap directly ON a stone (tight radius), which switches the selection.
    // (A generous radius here used to swallow targets near stones, e.g. freezes.)
    const targeting = S.hitSlot != null && (S.hitAct === "tap" || S.hitAct === "roll");
    let best = null, bd = targeting ? 0.22 : 0.5;
    for (const s of S.match.board) {
      const d = Math.hypot(s.along - al, s.lateral - la);
      if (d < bd) { bd = d; best = s.slot; }
    }
    if (best != null && best !== S.hitSlot) { S.hitSlot = best; S.tapTarget = null; }
    else if (targeting) S.tapTarget = [al, la];
    S.solved = null;
  }
  renderGame();
  if ((S.mode === "draw" && S.target) ||
      (S.mode === "hit" && S.hitSlot != null && (S.hitAct === "remove" || S.tapTarget)))
    autoPreviewShot();
}

let _previewSeq = 0;
async function autoPreviewShot() {
  // Solve constrained shots immediately and show the predicted result before
  // the user commits; Throw then replays the solved action exactly.
  const seq = ++_previewSeq;
  $("hint").textContent = "Working out the shot…";
  $("throwBtn").disabled = true;
  try {
    const d = await api(`/api/match/${S.match.id}/solve`, { method: "POST", body: shotBody(true) });
    if (seq !== _previewSeq) return;             // selection changed meanwhile
    S.solved = d;
    setParams(d.intended);
    renderGame();
    if (d.solver?.solvable === false) {
      $("hint").textContent = (d.solver.solvability_reason || "That shot is not reliable.") +
        " Choose another target, stone, or shot.";
      $("throwBtn").disabled = true;
      return;
    }
    if (S.mode === "params") {
      $("hint").textContent = "Parameter throw previewed — Throw when ready.";
    } else if (S.mode === "draw") {
      const err = d.solver?.achieved_error_m;
      $("hint").textContent = err == null ? "Ready — Throw when ready."
        : `Stone should finish within ~${Math.round(err * 100)} cm. Throw when ready.`;
    } else if (S.hitAct === "roll") {
      const err = d.solver?.expected_roll_error_m;
      const contact = d.solver?.contact_reliability;
      const rel = d.solver?.removal_reliability;
      const surv = d.solver?.shooter_survives;
      let msg = err == null ? "Ready — Throw when ready."
        : err > 0.9 ? `⚠️ Hard roll — your stone typically ends ~${err.toFixed(1)} m from that spot.`
        : `Your stone typically rolls to ~${Math.round(err * 100)} cm of that spot.`;
      if (contact != null && contact < 0.7) msg += ` Makes the hit ~${Math.round(contact * 100)}% of the time.`;
      if (d.solver?.takeout_allowed === false) msg += " The no-takeout rule is active, so the hit stone stays in play.";
      else if (rel != null && rel < 0.6) msg += ` Removes the hit stone only ~${Math.round(rel * 100)}% of the time.`;
      if (surv != null && surv < 0.7) msg += ` ⚠️ Your shooter may roll out (${Math.round(surv * 100)}% stays).`;
      $("hint").textContent = msg + " Throw when ready.";
    } else if (S.hitAct === "remove") {
      const rel = d.solver?.removal_reliability;
      $("hint").textContent = rel != null && rel < 0.6
        ? `⚠️ Risky from here — removes it only ~${Math.round(rel * 100)}% of the time. Throw anyway or pick another stone.`
        : rel != null ? `Good angle — removes it ~${Math.round(rel * 100)}% of the time. Throw when ready.`
        : "Looks good — Throw when ready.";
    } else {
      // expected error INCLUDES execution noise — the honest number for a tap
      const err = d.solver?.expected_error_m ?? d.solver?.achieved_error_m;
      $("hint").textContent = err == null ? "Ready — Throw when ready."
        : err > 0.9 ? `⚠️ Hard from here — it typically ends ~${err.toFixed(1)} m from that spot. Adjust or throw anyway.`
        : `It typically ends ~${Math.round(err * 100)} cm from your spot (throws vary — same for the champion). Throw when ready.`;
    }
    $("throwBtn").disabled = !humanOnTurn();
  } catch (e) {
    if (seq === _previewSeq) { $("hint").textContent = e.message; }
  }
}

function shotBody(preview) {
  const side = S.match.turn.team;
  if (!preview && S.solved?.intended)
    return { side, type: "params", action: S.solved.intended };   // throw the previewed shot
  if (S.mode === "params") return { side, type: "params", action: S.params, preview };
  if (S.mode === "draw") return { side, type: "draw", target: S.target, preview };
  if (S.hitAct === "remove") return { side, type: "after_contact", stone_slot: S.hitSlot, remove: true, preview };
  if (S.hitAct === "roll") return { side, type: "hit_roll", stone_slot: S.hitSlot, target: S.tapTarget, preview };
  return { side, type: "after_contact", stone_slot: S.hitSlot, target: S.tapTarget, preview };
}

let _paramPreviewTimer = null;
function parameterChanged(i, value) {
  _previewSeq++;
  S.params[i] = Number(value);
  S.solved = null;
  S.target = null; S.hitSlot = null; S.tapTarget = null;
  selectShotMode("params");
  syncParamControls();
  renderGame();
  clearTimeout(_paramPreviewTimer);
  _paramPreviewTimer = setTimeout(() => {
    if (humanOnTurn() && S.mode === "params" && !S.busy) autoPreviewShot();
  }, 220);
}

async function doThrow() {
  if (S.busy) return;
  if (!S.solved || S.solved.solver?.solvable === false) {
    toast("Choose a shot the solver can complete first.", 3600);
    return;
  }
  S.busy = true;
  $("hint").textContent = "Delivering…";
  $("throwBtn").disabled = true;
  try {
    const body = shotBody(false);
    body.auto_reply = false;                     // opponent thinks AFTER our stone lands
    const d = await api(`/api/match/${S.match.id}/throw`, { method: "POST", body });
    await animateThrow($("board"), d.result);
    if (d.result?.end_result) {
      holdEnd(d.match, d.result.end_result.end, d.result.end_result.score, d.result.board);
    } else {
      S.match = d.match; S.seenThrows = countThrows(d.match);
      resetShot(); renderGame();
      S.busy = false;
      await championIfOnTurn();
    }
  } catch (e) { toast(e.message, 4200); }
  finally { S.busy = false; busyHide(); if (!S.pendingAdvance) renderGame(); }
}
async function doPreview() {
  try {
    const d = await api(`/api/match/${S.match.id}/solve`, { method: "POST", body: shotBody(true) });
    S.solved = d;
    setParams(d.intended);
    const v = d.preview?.predicted_value_A;
    toast(`Solver ${d.solver?.achieved_error_m != null ? d.solver.achieved_error_m + " m off target · " : ""}` +
          (v != null ? `champion eval ${v > 0 ? "+" : ""}${v} (red persp.)` : ""), 4200);
    renderGame();
  } catch (e) { toast(e.message, 4000); }
}

/* Hold at end-of-end: freeze the completed end's final board and wait for the
   user's "Next end" tap before advancing to the new pre-placement. */
function holdEnd(newMatch, endNo, score, finalBoard) {
  S.pendingAdvance = newMatch;
  if (finalBoard) drawBoard($("board"), finalBoard, {});
  $("scA").textContent = newMatch.totals.A; $("scB").textContent = newMatch.totals.B;
  const finished = newMatch.status === "finished";
  const name = score && score.team
    ? (newMatch.labels?.[score.team] || TEAM_NAME[score.team]) : null;
  $("endTitle").textContent = `End ${endNo} complete`;
  const verb = name === "You" ? "score" : "scores";
  $("endBody").textContent = (name ? `${name} ${verb} ${score.points}.` : "Blank end — no score.") +
    `  (${newMatch.totals.A} : ${newMatch.totals.B})`;
  $("nextEndBtn").textContent = finished ? "Final result ▶" : "Next end ▶";
  $("statusbar").innerHTML = `<b>End ${endNo} complete</b>`;
  $("endModal").classList.remove("hidden");
}
async function advanceEnd() {
  $("endModal").classList.add("hidden");
  const m = S.pendingAdvance;
  S.pendingAdvance = null;
  if (!m) return;
  S.match = m; S.seenThrows = countThrows(m);
  resetShot(); renderGame();
  if (m.status === "finished") matchOver();
  else await maybePowerPlay();
}
function matchOver() {
  stopPolling();
  const m = S.match;
  const la = m.labels?.A || TEAM_NAME.A, lb = m.labels?.B || TEAM_NAME.B;
  const w = m.totals.A > m.totals.B ? la : m.totals.B > m.totals.A ? lb : null;
  $("overTitle").textContent = w ? (w === "You" ? "🏆 You win!" : `🏆 ${w} wins!`) : "Tied match";
  $("overBody").textContent = `Final score ${m.totals.A} : ${m.totals.B} over ${m.ends_scheduled} ends.`;
  $("overModal").classList.remove("hidden");
}

/* power play + champion resume */
function ppAvailable() {
  const m = S.match, e = curEnd();
  return m.status === "in_progress" && e.mode !== "sandbox" && m.turn.throw === 1 &&
    !(e.throws || []).length && e.hammer &&
    m.players[e.hammer] === "human" && S.mySides.has(e.hammer) &&
    !m.power_play_used[e.hammer];
}
async function maybePowerPlay() {
  if (ppAvailable()) { $("ppModal").classList.remove("hidden"); return; }
  await championIfOnTurn();
}
async function championIfOnTurn() {
  const m = S.match;
  if (m.status === "in_progress" && m.players[m.turn.team] === "champion") {
    S.busy = true;
    busyShow(m.players[m.turn.team] === "champion" ? "Champion thinking…" : "Opponent thinking…");
    try {
      const d = await api(`/api/match/${m.id}/champion_move`, { method: "POST", body: {} });
      busyHide();
      await animateThrow($("board"), d.result);
      let er = d.result?.end_result, board = d.result?.board;
      for (const rep of d.replies || []) {
        await animateThrow($("board"), rep);
        if (rep.end_result) { er = rep.end_result; board = rep.board; }
      }
      if (er) {
        holdEnd(d.match, er.end, er.score, board);
      } else {
        S.match = d.match; S.seenThrows = countThrows(d.match);
        resetShot();
        if (d.match.status === "finished") matchOver();
        else maybePowerPlay();
      }
    } catch (e) { if (!/not on turn/.test(e.message)) toast(e.message); }
    finally { S.busy = false; busyHide(); if (!S.pendingAdvance) renderGame(); }
  }
}
async function choosePP(wing) {
  $("ppModal").classList.add("hidden");
  if (wing) {
    try {
      const d = await api(`/api/match/${S.match.id}/powerplay`, { method: "POST",
        body: { side: curEnd().hammer, wing } });
      S.match = d.match; renderGame();
      toast(`Power play — ${wing} wing`);
    } catch (e) { toast(e.message); }
  }
  await championIfOnTurn();
}

/* ---------------- online sync (polling) ---------------- */
function startPolling() {
  stopPolling();
  S.pollTimer = setInterval(pollOnce, 2500);
}
function stopPolling() {
  if (S.pollTimer) { clearInterval(S.pollTimer); S.pollTimer = null; }
}
async function pollOnce() {
  if (!S.match || S.busy || S.view !== "game") return;
  if (humanOnTurn()) return;                    // it's our move; nothing to fetch
  try {
    const d = await api(`/api/match/${S.match.id}`, { timeoutMs: 8000 });
    const newCount = countThrows(d.match);
    if (newCount > S.seenThrows) {
      S.busy = true;
      try {
        toast("Opponent throwing…", 1400);
        const recs = [];
        for (const e of d.match.ends)
          for (const r of e.throws || []) recs.push(r);
        for (const r of recs.slice(S.seenThrows)) {
          try {
            const tt = await api(`/api/match/${S.match.id}/throw_traj?end=${r.end}&n=${r.n}`);
            await animateTraj($("board"), tt.trajectory, tt.board, PLAY_RATE);
          } catch (err) { /* skip animation, still update */ }
        }
        S.seenThrows = newCount;
        const prevEnds = S.match.ends.length;
        if (d.match.ends.length > prevEnds || d.match.status === "finished") {
          const doneEnd = d.match.ends.length > prevEnds
            ? d.match.ends[prevEnds - 1] : d.match.ends[d.match.ends.length - 1];
          const board = doneEnd.state
            ? null : null;   // board already drawn by the last animation
          holdEnd(d.match, prevEnds, doneEnd.score, board);
        } else {
          S.match = d.match; resetShot(); renderGame();
          maybePowerPlay();
        }
      } finally {
        S.busy = false;                          // a stuck busy here used to kill the UI
        busyHide();
      }
    } else {
      S.match = d.match; renderGame();
    }
  } catch (e) { /* transient network — keep polling */ }
}

/* coach */
function renderCoach() {
  $("coachBtn").classList.toggle("active", S.coach);
  $("coachpanel").classList.toggle("hidden", !(S.coach && S.view === "game"));
  if (!S.coach || !S.match) return;
  const tr = $("evaltrace"); tr.innerHTML = "";
  const throws = (curEnd().throws || []).filter((r) => r.value_A != null);
  for (const r of throws.slice(-8)) {
    const d = document.createElement("span");
    d.className = "evalpill " + (r.value_A >= 0 ? "up" : "down");
    d.textContent = `${TEAM_NAME[r.team]} ${r.value_A > 0 ? "+" : ""}${r.value_A.toFixed(2)}`;
    tr.appendChild(d);
  }
}
async function loadHeat() {
  if (!humanOnTurn()) { toast("Heatmap is for your throw"); return; }
  $("heatBtn").disabled = true; $("heatBtn").textContent = "computing…";
  try {
    S.heat = await api(`/api/match/${S.match.id}/heatmap`);
    $("heatlegend").classList.remove("hidden");
    renderGame();
  } catch (e) { toast(e.message); }
  $("heatBtn").disabled = false; $("heatBtn").textContent = "🔥 Best-spot map";
}

/* ---------------- replay ---------------- */
async function openReplay(mid) {
  try {
    const d = await api(`/api/match/${mid}/replay`);
    S.replay = d; S.step = 0; S.rHeatOn = false; S.rHeatCache = {};
    $("rHeatBtn").classList.remove("active");
    $("rSlider").max = Math.max(d.steps.length - 1, 0);
    $("rSlider").value = 0;
    const la = d.labels?.A || d.players.A, lb = d.labels?.B || d.players.B;
    $("replayHeader").innerHTML = `<b>${la}</b> (red) vs <b>${lb}</b> (yellow) — final ${d.totals.A}:${d.totals.B}`;
    show("replay"); renderStep();
  } catch (e) { toast(e.message); }
}
async function renderStep(animate = false) {
  const d = S.replay;
  if (!d || !d.steps.length) return;
  const st = d.steps[S.step];
  let heat = null;
  if (S.rHeatOn) {
    heat = S.rHeatCache[S.step];
    if (heat === undefined) {
      $("rLabel").textContent = "computing heatmap…";
      try {
        heat = await api(`/api/match/${d.id}/heatmap?end=${st.end}&n=${st.n}`);
      } catch (e) { heat = null; }
      S.rHeatCache[S.step] = heat;
    }
  }
  $("rHeatLegend").classList.toggle("hidden", !(S.rHeatOn && heat));
  if (animate && !S.rHeatOn) {
    drawBoard($("rboard"), st.board_before, {});
    await animateTraj($("rboard"), st.traj, st.board_after, REPLAY_RATE);
  } else if (S.rHeatOn && heat) {
    // coach view: the board BEFORE the throw + where the next stone should go
    drawBoard($("rboard"), st.board_before, { heat, traj: st.traj });
  } else {
    drawBoard($("rboard"), st.board_after, { traj: st.traj });
  }
  const who = st.team === "A" ? (d.labels?.A || "Red") : (d.labels?.B || "Yellow");
  $("rLabel").innerHTML = `End ${st.end} · Throw ${st.n}/10 · <b>${who}</b>` +
    (st.illegal ? " · ⚠️ forfeited (early take-out)" : "") +
    (st.end_score && st.end_score.team ? ` — <b>${st.end_score.team} scores ${st.end_score.points}</b>` : "") +
    (S.coach && st.value_A != null ? ` · eval ${st.value_A > 0 ? "+" : ""}${st.value_A}` : "");
  $("rSlider").value = S.step;
}

/* ---------------- wiring ---------------- */
function bindCanvas(cv, handler) {
  cv.addEventListener("pointerdown", (ev) => {
    const rect = cv.getBoundingClientRect();
    const map = mkMap(cv);
    handler(map.invAlong(ev.clientY - rect.top), map.invLat(ev.clientX - rect.left));
  });
}
window.addEventListener("resize", () => {
  if (S.view === "game") renderGame();
  if (S.view === "sandbox") drawSandbox();
  if (S.view === "replay") renderStep();
});

document.addEventListener("DOMContentLoaded", () => {
  bindCanvas($("board"), boardTapped);
  bindCanvas($("sboard"), sandboxTapped);
  loadArenaConfig();
  syncParamControls(); syncNoiseControls();
  document.querySelectorAll(".newgame").forEach((b) => b.onclick = () => newMatch(b.dataset.kind));
  $("sandboxBtn").onclick = () => openSandbox();
  $("joinBtn").onclick = () => {
    const code = $("joinCode").value.trim().split("/").pop();
    if (code) joinFromLink(code);
  };
  document.querySelectorAll(".endsel").forEach((b) => b.onclick = () => {
    document.querySelectorAll(".endsel").forEach((x) => x.classList.remove("sel"));
    b.classList.add("sel"); S.ends = +b.dataset.ends;
  });
  document.querySelectorAll(".param-range").forEach((el) =>
    el.oninput = () => parameterChanged(+el.dataset.i, el.value));
  $("setupNoiseOn").onchange = (e) => { S.noiseEnabled = e.target.checked; syncNoiseControls(); };
  document.querySelectorAll(".setup-noise").forEach((el) => el.oninput = () => {
    S.noiseScales[+el.dataset.i] = +el.value; syncNoiseControls();
  });
  $("gameNoiseOn").onchange = (e) => { S.noiseEnabled = e.target.checked; queueNoiseSave(); };
  document.querySelectorAll(".game-noise").forEach((el) => el.oninput = () => {
    S.noiseScales[+el.dataset.i] = +el.value; queueNoiseSave();
  });
  document.querySelectorAll(".modechip").forEach((b) => b.onclick = () => {
    _previewSeq++;
    selectShotMode(b.dataset.m);
    S.target = null; S.hitSlot = null; S.tapTarget = null; S.solved = null; renderGame();
    if (S.mode === "params" && humanOnTurn()) autoPreviewShot();
  });
  document.querySelectorAll(".hitopt").forEach((b) => b.onclick = () => {
    _previewSeq++;
    document.querySelectorAll(".hitopt").forEach((x) => x.classList.remove("sel"));
    b.classList.add("sel"); S.hitAct = b.dataset.h; S.tapTarget = null; S.solved = null; renderGame();
    if (S.mode === "hit" && S.hitSlot != null && S.hitAct === "remove") autoPreviewShot();
  });
  $("throwBtn").onclick = doThrow;
  $("previewBtn").onclick = doPreview;
  $("undoBtn").onclick = async () => {
    try {
      const d = await api(`/api/match/${S.match.id}/undo`, { method: "POST", body: {} });
      S.match = d.match; S.seenThrows = countThrows(d.match);
      resetShot(); renderGame(); toast("Rolled back the last throw");
    } catch (e) { toast(e.message); }
  };
  $("coachBtn").onclick = () => {
    S.coach = !S.coach; localStorage.coach = S.coach ? "1" : "0";
    if (S.view === "game") renderGame();
    renderCoach();
    if (S.view === "replay") renderStep();
  };
  $("homeBtn").onclick = () => show("home");
  $("sandboxBack").onclick = () => show("home");
  document.querySelectorAll(".stone-tool").forEach((b) => b.onclick = () => {
    S.scenarioTool = b.dataset.tool;
    document.querySelectorAll(".stone-tool").forEach((x) => x.classList.toggle("sel", x === b));
    drawSandbox();
  });
  $("scenarioClear").onclick = () => { S.scenarioStones = []; drawSandbox(); };
  $("scenarioSave").onclick = saveScenarioOnly;
  $("scenarioPlay").onclick = playScenario;
  for (const id of ["scenarioEnd", "scenarioThrow", "scenarioHammer", "scenarioScoreA", "scenarioScoreB"])
    $(id).oninput = drawSandbox;
  $("heatBtn").onclick = loadHeat;
  document.querySelectorAll(".ppchoice").forEach((b) => b.onclick = () => choosePP(b.dataset.w));
  $("ppSkip").onclick = () => choosePP(null);
  $("nextEndBtn").onclick = advanceEnd;
  $("overHome").onclick = () => { $("overModal").classList.add("hidden"); show("home"); };
  $("overReplay").onclick = () => { $("overModal").classList.add("hidden"); openReplay(S.match.id); };
  $("replayBack").onclick = () => show("home");
  $("rSlider").oninput = (e) => { S.step = +e.target.value; renderStep(); };
  $("rPrev").onclick = () => { S.step = Math.max(0, S.step - 1); renderStep(); };
  $("rNext").onclick = () => { S.step = Math.min(S.replay.steps.length - 1, S.step + 1); renderStep(true); };
  $("rHeatBtn").onclick = () => {
    S.rHeatOn = !S.rHeatOn;
    $("rHeatBtn").classList.toggle("active", S.rHeatOn);
    renderStep();
  };
  $("shareCopy").onclick = async () => {
    try { await navigator.clipboard.writeText($("shareLink").value); toast("Link copied"); }
    catch (e) { $("shareLink").select(); document.execCommand("copy"); toast("Link copied"); }
  };
  $("shareClose").onclick = () => $("shareModal").classList.add("hidden");
  renderCoach();

  const joinMatch = location.pathname.match(/^\/join\/([A-Za-z0-9]+)/);
  if (joinMatch) joinFromLink(joinMatch[1]);
  else show("home");
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/sw.js").catch(() => {});
    // when a new version activates, reload once so stale clients can't linger
    navigator.serviceWorker.addEventListener("controllerchange", () => {
      if (sessionStorage.swReloaded) return;
      sessionStorage.swReloaded = "1";
      toast("Updating to the latest version…", 1500);
      setTimeout(() => location.reload(), 600);
    });
  }
  console.log("Curling Arena client v" + APP_VERSION);
  const vl = $("verline"); if (vl) vl.textContent = "v" + APP_VERSION;
});
