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
  rHeatOn: false, rHeatCache: {},
};
const PLAY_RATE = 5;          // sim-seconds per real-second (real throw ≈ 24 s → ≈ 5 s)
const REPLAY_RATE = 7;

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
const VIEW = { latHalf: 2.55, alongTop: -4.7, alongBot: 2.45 };
const TEAM_FILL = { A: "#d33a2f", B: "#e8b71a" };
const TEAM_NAME = { A: "Red", B: "Yellow" };
const RINGS = [[1.83, "#3f7fbf"], [1.22, "#f5f7f9"], [0.61, "#d24646"], [0.15, "#f5f7f9"]];

function setupCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth;
  const h = w * (VIEW.alongBot - VIEW.alongTop) / (2 * VIEW.latHalf);
  cv.style.height = h + "px";
  cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
function mkMap(cv) {
  const w = cv.clientWidth, ppm = w / (2 * VIEW.latHalf);
  return {
    px: (lat) => (lat + VIEW.latHalf) * ppm,
    py: (along) => (along - VIEW.alongTop) * ppm,
    invLat: (x) => x / ppm - VIEW.latHalf,
    invAlong: (y) => y / ppm + VIEW.alongTop,
    ppm,
  };
}
function heatColor(v, lo, mid, hi) {
  // diverging around the median spot value: blue = better, red = worse.
  // Near-median cells stay almost transparent so only real signal colors the ice.
  let t;
  if (v >= mid) {
    t = hi > mid ? Math.min(1, (v - mid) / (hi - mid)) : 0;
    return `rgba(23,111,208,${0.06 + 0.66 * t})`;
  }
  t = mid > lo ? Math.min(1, (mid - v) / (mid - lo)) : 0;
  return `rgba(198,44,34,${0.06 + 0.66 * t})`;
}
function drawBoard(cv, board, opts = {}) {
  const ctx = setupCanvas(cv), m = mkMap(cv);
  const w = cv.clientWidth, h = parseFloat(cv.style.height);
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
  ctx.strokeStyle = "rgba(40,70,110,.25)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(m.px(0), 0); ctx.lineTo(m.px(0), h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, m.py(0)); ctx.lineTo(w, m.py(0)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, m.py(-1.974)); ctx.lineTo(w, m.py(-1.974)); ctx.stroke();
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
    let done = false, watchdog = null;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(watchdog);
      try { if (finalBoard) drawBoard(cv, finalBoard, {}); } catch (e) { /* draw-only */ }
      resolve();
    };
    if (!traj || !traj.frames || traj.frames.length < 2) { finish(); return; }
    const frames = traj.frames, dt = traj.dt || 0.1;
    const durMs = (frames.length * dt / rate) * 1000;
    // the promise MUST settle even if rAF stalls (hidden tab, throttling, a
    // rendering exception) — a stuck animation used to freeze "Throwing…"
    watchdog = setTimeout(finish, durMs + 3000);
    const t0 = performance.now();
    const tick = (now) => {
      if (done) return;
      try {
        const idx = Math.floor(((now - t0) / 1000) * rate / dt);
        const f = frames[Math.min(idx, frames.length - 1)];
        const stones = [];
        for (let slot = 0; slot < f.length; slot++) {
          const p = f[slot];
          if (p && p[0] != null) stones.push({ slot, team: slot < 6 ? "A" : "B", along: p[0], lateral: p[1] });
        }
        drawBoard(cv, stones, {});
        if (idx >= frames.length + 2) { finish(); return; }
      } catch (e) { finish(); return; }
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
  for (const v of ["home", "game", "replay"]) $(`view-${v}`).classList.toggle("hidden", v !== view);
  $("homeBtn").classList.toggle("hidden", view === "home");
  $("scorechip").classList.toggle("hidden", view === "home");
  if (view !== "game") stopPolling();
  if (view === "home") loadMatches();
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
      const when = new Date(r.created * 1000).toLocaleDateString(undefined, { month: "short", day: "numeric" });
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
      players, labels, ends: S.ends, first_hammer: "random", mode: kind } });
    S.match = d.match;
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
    const side = await claimSeat(mid);
    S.mySides = new Set(side ? [side] : []);
    S.seenThrows = countThrows(d.match);
    resetShot(); show("game"); renderGame(); maybePowerPlay(); startPolling();
    toast(side ? `You are ${TEAM_NAME[side]} — good curling!` : "Both seats taken — watching live", 3600);
  } catch (e) { toast("Couldn't join: " + e.message, 4000); show("home"); }
}

/* ---------------- game ---------------- */
function countThrows(m) { return m.ends.reduce((a, e) => a + (e.throws ? e.throws.length : 0), 0); }
function humanOnTurn() {
  const m = S.match;
  return m && m.status === "in_progress" && S.mySides.has(m.turn.team);
}
function resetShot() {
  S.target = null; S.hitSlot = null; S.tapTarget = null; S.heat = null; S.solved = null;
  $("heatlegend").classList.add("hidden");
}
function curEnd() { return S.match.ends[S.match.ends.length - 1]; }

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
  updateShotUI();
  renderCoach();
}
function updateShotUI() {
  const my = humanOnTurn();
  $("shotbar").style.opacity = my ? 1 : 0.55;
  const ready = my && !S.busy &&
    ((S.mode === "draw" && S.target) ||
     (S.mode === "hit" && S.hitSlot != null && (S.hitAct === "remove" || S.tapTarget)));
  $("throwBtn").disabled = !ready;
  $("previewBtn").classList.toggle("hidden", !(S.coach && ready));
  $("hint").textContent = !my ? (S.match.status === "finished" ? "" : (S.online && S.mySides.size === 0 ? "Watching live…" : "Waiting for the other team…"))
    : S.mode === "draw" ? (S.target ? "Target set — Throw when ready." : "Tap the ice where the stone should stop.")
    : S.hitSlot == null ? "Tap a stone on the board."
    : S.hitAct === "remove" ? "Take-out selected — Throw when ready."
    : (S.tapTarget ? "Target set — Throw when ready (tap another stone to switch)."
       : "Now tap where that stone should end up (even right beside another stone).");
}

function boardTapped(al, la) {
  if (S.view !== "game" || !humanOnTurn() || S.busy) return;
  if (S.mode === "draw") { S.target = [al, la]; S.solved = null; }
  else {
    // Once a stone is picked in tap mode, the next tap is the TARGET — except a
    // tap directly ON a stone (tight radius), which switches the selection.
    // (A generous radius here used to swallow targets near stones, e.g. freezes.)
    const targeting = S.hitSlot != null && S.hitAct === "tap";
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
}

function shotBody(preview) {
  const side = S.match.turn.team;
  if (S.mode === "draw") return { side, type: "draw", target: S.target, preview };
  if (S.hitAct === "remove") return { side, type: "after_contact", stone_slot: S.hitSlot, remove: true, preview };
  return { side, type: "after_contact", stone_slot: S.hitSlot, target: S.tapTarget, preview };
}

async function doThrow() {
  if (S.busy) return;
  S.busy = true; $("busy").classList.remove("hidden"); $("busy").textContent = "Throwing…";
  try {
    const d = await api(`/api/match/${S.match.id}/throw`, { method: "POST", body: shotBody(false) });
    await animateThrow($("board"), d.result);
    for (const rep of d.replies || []) {
      $("busy").textContent = "Champion is thinking…";
      await animateThrow($("board"), rep);
      if (rep.end_result) announceEnd(rep.end_result);
    }
    if (d.result?.end_result) announceEnd(d.result.end_result);
    S.match = d.match; S.seenThrows = countThrows(d.match);
    resetShot(); renderGame();
    if (d.match.status === "finished") matchOver();
    else maybePowerPlay();
  } catch (e) { toast(e.message, 4200); }
  finally { S.busy = false; $("busy").classList.add("hidden"); renderGame(); }
}
async function doPreview() {
  try {
    const d = await api(`/api/match/${S.match.id}/solve`, { method: "POST", body: shotBody(true) });
    S.solved = d;
    const v = d.preview?.predicted_value_A;
    toast(`Solver ${d.solver?.achieved_error_m != null ? d.solver.achieved_error_m + " m off target · " : ""}` +
          (v != null ? `champion eval ${v > 0 ? "+" : ""}${v} (red persp.)` : ""), 4200);
    renderGame();
  } catch (e) { toast(e.message, 4000); }
}

function announceEnd(er) {
  if (!er || !er.score) return;
  const t = er.score.team, p = er.score.points;
  const name = S.match && S.match.labels ? (S.match.labels[t] || TEAM_NAME[t]) : TEAM_NAME[t];
  toast(t ? `End ${er.end}: ${name} scores ${p}` : `End ${er.end}: blanked`, 3800);
}
function matchOver() {
  stopPolling();
  const m = S.match;
  const la = m.labels?.A || TEAM_NAME.A, lb = m.labels?.B || TEAM_NAME.B;
  const w = m.totals.A > m.totals.B ? la : m.totals.B > m.totals.A ? lb : null;
  $("overTitle").textContent = w ? `🏆 ${w} wins!` : "Tied match";
  $("overBody").textContent = `Final score ${m.totals.A} : ${m.totals.B} over ${m.ends_scheduled} ends.`;
  $("overModal").classList.remove("hidden");
}

/* power play + champion resume */
function ppAvailable() {
  const m = S.match, e = curEnd();
  return m.status === "in_progress" && !(e.throws || []).length && e.hammer &&
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
    S.busy = true; $("busy").classList.remove("hidden"); $("busy").textContent = "Champion is thinking…";
    try {
      const d = await api(`/api/match/${m.id}/champion_move`, { method: "POST", body: {} });
      await animateThrow($("board"), d.result);
      for (const rep of d.replies || []) await animateThrow($("board"), rep);
      if (d.result?.end_result) announceEnd(d.result.end_result);
      S.match = d.match; S.seenThrows = countThrows(d.match);
      resetShot();
      if (d.match.status === "finished") matchOver();
      else maybePowerPlay();
    } catch (e) { if (!/not on turn/.test(e.message)) toast(e.message); }
    finally { S.busy = false; $("busy").classList.add("hidden"); renderGame(); }
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
    const d = await api(`/api/match/${S.match.id}`);
    const newCount = countThrows(d.match);
    if (newCount > S.seenThrows) {
      S.busy = true;
      // animate each unseen throw in order
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
      S.match = d.match; resetShot(); renderGame();
      for (const e of d.match.ends) if (e.score && e.throws?.length === 0) { /* noop */ }
      if (d.match.status === "finished") matchOver();
      else maybePowerPlay();
      S.busy = false;
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
window.addEventListener("resize", () => { if (S.view === "game") renderGame(); if (S.view === "replay") renderStep(); });

document.addEventListener("DOMContentLoaded", () => {
  bindCanvas($("board"), boardTapped);
  document.querySelectorAll(".newgame").forEach((b) => b.onclick = () => newMatch(b.dataset.kind));
  $("joinBtn").onclick = () => {
    const code = $("joinCode").value.trim().split("/").pop();
    if (code) joinFromLink(code);
  };
  document.querySelectorAll(".endsel").forEach((b) => b.onclick = () => {
    document.querySelectorAll(".endsel").forEach((x) => x.classList.remove("sel"));
    b.classList.add("sel"); S.ends = +b.dataset.ends;
  });
  document.querySelectorAll(".modechip").forEach((b) => b.onclick = () => {
    document.querySelectorAll(".modechip").forEach((x) => x.classList.remove("sel"));
    b.classList.add("sel"); S.mode = b.dataset.m;
    $("hitopts").classList.toggle("hidden", S.mode !== "hit");
    S.target = null; S.hitSlot = null; S.tapTarget = null; S.solved = null; renderGame();
  });
  document.querySelectorAll(".hitopt").forEach((b) => b.onclick = () => {
    document.querySelectorAll(".hitopt").forEach((x) => x.classList.remove("sel"));
    b.classList.add("sel"); S.hitAct = b.dataset.h; S.tapTarget = null; renderGame();
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
  $("heatBtn").onclick = loadHeat;
  document.querySelectorAll(".ppchoice").forEach((b) => b.onclick = () => choosePP(b.dataset.w));
  $("ppSkip").onclick = () => choosePP(null);
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
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
});
