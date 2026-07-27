/* Curling Arena frontend.
 * Sheet coordinates: along (m, 0 = tee/button, negative toward hog line),
 * lateral (m, positive = right). Canvas: along on the vertical axis (hog at
 * top, back line at bottom), lateral on the horizontal axis. */

"use strict";

const VIEW = { minAlong: -11.8, maxAlong: 2.8, minLat: -2.62, maxLat: 2.62 };
const GEOM = {
  houseR: [1.829, 1.219, 0.610, 0.1524],
  stoneR: 0.145, hog: -6.401, back: 1.829, sideHalf: 2.375, hogToTee: 28.35,
};
// default CurlingParams (must match csas_v3 curling_sim_jax.CurlingParams)
const PHYS = { dt: 0.02, substeps: 2, maxSteps: 1500, vStop: 0.03, vCap: 6.0,
               aLinear: 0.11, kCurl: 0.10, gammaSpin: 0.15, curlSpeedCap: 2.5 };

const S = {
  matchId: null, match: null, mode: "draw", weight: "medium",
  selSlot: null, solved: null, targetMark: null, guide: null,
  faded: [],   // pre-throw positions of stones displaced by the current throw
  evalA: 0, busy: false, animating: false, previewTimer: null,
  solveSeq: 0, hintTimer: null,
};

const canvas = document.getElementById("sheet");
const ctx = canvas.getContext("2d");
const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ */
/* Canvas mapping                                                      */
/* ------------------------------------------------------------------ */
function sizeCanvas() {
  const spanLat = VIEW.maxLat - VIEW.minLat, spanAlong = VIEW.maxAlong - VIEW.minAlong;
  const maxH = Math.min(window.innerHeight * 0.86, 780);
  const h = Math.max(430, maxH), w = h * (spanLat / spanAlong);
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
function ppm() { return (canvas.clientHeight || 1) / (VIEW.maxAlong - VIEW.minAlong); }
function px(lat) { return (lat - VIEW.minLat) * ppm(); }
function py(along) { return (along - VIEW.minAlong) * ppm(); }
function evToM(ev) {
  const r = canvas.getBoundingClientRect();
  const lat = (ev.clientX - r.left) / ppm() + VIEW.minLat;
  const along = (ev.clientY - r.top) / ppm() + VIEW.minAlong;
  return [along, lat];
}

/* ------------------------------------------------------------------ */
/* Drawing                                                             */
/* ------------------------------------------------------------------ */
function drawSheet(boardOverride) {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#dfe9f0"; ctx.fillRect(0, 0, w, h);
  // out-of-sheet shading beyond side lines
  ctx.fillStyle = "rgba(120,140,158,0.35)";
  ctx.fillRect(0, 0, px(-GEOM.sideHalf), h);
  ctx.fillRect(px(GEOM.sideHalf), 0, w - px(GEOM.sideHalf), h);
  ctx.fillRect(0, py(2.438), w, h - py(2.438)); // beyond the removal boundary

  // house
  const rings = [["#7fa8c9", GEOM.houseR[0]], ["#f3f7fa", GEOM.houseR[1]],
                 ["#cf6454", GEOM.houseR[2]], ["#f3f7fa", GEOM.houseR[3]]];
  for (const [color, r] of rings) {
    ctx.beginPath(); ctx.arc(px(0), py(0), r * ppm(), 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
    ctx.strokeStyle = "rgba(30,40,50,0.25)"; ctx.lineWidth = 1; ctx.stroke();
  }
  // lines: center, tee, back, hog
  ctx.strokeStyle = "rgba(30,40,50,0.4)"; ctx.lineWidth = 1.2;
  vline(0); hline(0); hline(GEOM.back);
  ctx.strokeStyle = "rgba(180,60,50,0.75)"; ctx.lineWidth = 3; hline(GEOM.hog);

  drawTrajectory();
  drawGuide();
  drawBoard(boardOverride);
  drawMarks();
}
function hline(along) { ctx.beginPath(); ctx.moveTo(0, py(along)); ctx.lineTo(canvas.clientWidth, py(along)); ctx.stroke(); }
function vline(lat) { ctx.beginPath(); ctx.moveTo(px(lat), 0); ctx.lineTo(px(lat), canvas.clientHeight); ctx.stroke(); }

function stoneColor(team) { return team === "A" ? "#d1584a" : "#e5b93c"; }

function drawStoneAt(along, lat, team, opts = {}) {
  const r = GEOM.stoneR * ppm();
  ctx.beginPath(); ctx.arc(px(lat), py(along), r, 0, Math.PI * 2);
  if (opts.faded) {
    ctx.save(); ctx.globalAlpha = 0.22;
    ctx.fillStyle = stoneColor(team); ctx.fill();
    ctx.globalAlpha = 0.35; ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(20,28,36,0.8)"; ctx.stroke();
    ctx.restore();
  } else if (opts.ghost) {
    ctx.strokeStyle = stoneColor(team); ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
    ctx.stroke(); ctx.setLineDash([]);
  } else {
    ctx.fillStyle = stoneColor(team); ctx.fill();
    ctx.lineWidth = opts.selected ? 3.5 : 1.5;
    ctx.strokeStyle = opts.selected ? "#ffffff" : "rgba(20,28,36,0.65)";
    ctx.stroke();
    if (opts.label != null) {
      ctx.fillStyle = "rgba(255,255,255,0.92)";
      ctx.font = `${Math.max(9, r * 0.95)}px system-ui`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(opts.label, px(lat), py(along) + 0.5);
    }
  }
}

function drawBoard(boardOverride) {
  const board = boardOverride || (S.match && S.match.board) || [];
  for (const f of S.faded) drawStoneAt(f.along, f.lateral, f.team, { faded: true });
  for (const s of board) {
    drawStoneAt(s.along, s.lateral, s.team,
                { label: (s.slot % 6) + 1, selected: s.slot === S.selSlot });
  }
  if (S.solved && S.solved.preview && !S.animating) {
    for (const s of S.solved.preview.predicted_board || []) {
      drawStoneAt(s.along, s.lateral, s.team, { ghost: true });
    }
  }
}

function trajPath(traj) {
  if (!traj || !traj.frames) return [];
  const pts = [];
  for (const f of traj.frames) {
    const xy = f[traj.stone_slot];
    if (xy && xy[0] != null) pts.push(xy);
  }
  return pts;
}

function drawTrajectory() {
  const traj = S.solved && !S.animating ? S.solved.preview?.intended_trajectory : null;
  const pts = trajPath(traj);
  if (pts.length < 2) return;
  ctx.save();
  ctx.setLineDash([10, 8]); ctx.lineWidth = 3; ctx.strokeStyle = "rgba(75,159,212,0.85)";
  ctx.beginPath(); ctx.moveTo(px(pts[0][1]), py(pts[0][0]));
  for (const p of pts.slice(1)) ctx.lineTo(px(p[1]), py(p[0]));
  ctx.stroke();
  if (traj.contact && traj.contact.thrown_at) {
    const [a, l] = traj.contact.thrown_at;
    cross(a, l, "rgba(220,120,40,0.95)");
  }
  ctx.restore();
}

function drawGuide() {
  if (S.mode !== "params" || !S.guide || S.animating) return;
  ctx.save();
  ctx.setLineDash([6, 7]); ctx.lineWidth = 2.5; ctx.strokeStyle = "rgba(111,191,115,0.9)";
  ctx.beginPath();
  let started = false;
  for (const p of S.guide.pts) {
    if (p[0] < VIEW.minAlong) continue;
    if (!started) { ctx.moveTo(px(p[1]), py(p[0])); started = true; }
    else ctx.lineTo(px(p[1]), py(p[0]));
  }
  ctx.stroke();
  if (S.guide.contact) cross(S.guide.contact[0], S.guide.contact[1], "rgba(220,120,40,0.95)");
  ctx.restore();
}

function cross(along, lat, color) {
  const r = 7;
  ctx.save(); ctx.setLineDash([]); ctx.strokeStyle = color; ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.moveTo(px(lat) - r, py(along) - r); ctx.lineTo(px(lat) + r, py(along) + r);
  ctx.moveTo(px(lat) - r, py(along) + r); ctx.lineTo(px(lat) + r, py(along) - r);
  ctx.stroke(); ctx.restore();
}

function drawMarks() {
  if (S.targetMark) {
    const [a, l] = S.targetMark;
    ctx.save(); ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(px(l), py(a), 9, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(px(l), py(a), 2, 0, Math.PI * 2);
    ctx.fillStyle = "#fff"; ctx.fill(); ctx.restore();
  }
}

/* ------------------------------------------------------------------ */
/* Client-side guide physics (params mode) — mirrors _ice_forces        */
/* ------------------------------------------------------------------ */
function buildGuide() {
  if (!S.match) return null;
  const speed = +$("speed").value, angle = +$("angle").value;
  let omega = +$("spin").value;
  const y0 = +$("y0").value;
  const stones = (S.match.board || []).map((s) => [s.along, s.lateral]);
  let x = -GEOM.hogToTee, y = y0;
  let vx = Math.cos(angle) * speed, vy = Math.sin(angle) * speed;
  const micro = PHYS.dt / PHYS.substeps;
  const pts = [];
  let contact = null;
  for (let step = 0; step < PHYS.maxSteps; step++) {
    pts.push([x, y]);
    let done = false;
    for (let sub = 0; sub < PHYS.substeps; sub++) {
      const sp = Math.hypot(vx, vy);
      const hx = vx / (sp + 1e-8), hy = vy / (sp + 1e-8);
      const sEff = PHYS.curlSpeedCap * Math.tanh(sp / PHYS.curlSpeedCap);
      const ax = -PHYS.aLinear * hx + PHYS.kCurl * omega * -hy * sEff;
      const ay = -PHYS.aLinear * hy + PHYS.kCurl * omega * hx * sEff;
      vx += ax * micro; vy += ay * micro;
      const sp2 = Math.hypot(vx, vy);
      if (sp2 > PHYS.vCap) { vx *= PHYS.vCap / sp2; vy *= PHYS.vCap / sp2; }
      x += vx * micro; y += vy * micro;
      omega += -PHYS.gammaSpin * omega * micro;
      for (const st of stones) {
        if (Math.hypot(st[0] - x, st[1] - y) < 2 * GEOM.stoneR) { contact = [x, y]; done = true; break; }
      }
      if (done) break;
      if (Math.hypot(vx, vy) < PHYS.vStop) done = true;
    }
    if (done || x > VIEW.maxAlong + 1) { pts.push([x, y]); break; }
  }
  return { pts, contact };
}

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */
async function api(path, method = "GET", body = null) {
  const res = await fetch(path, {
    method, headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });
  if (!res.ok) {
    const t = await res.text();
    let msg = t; try { msg = JSON.parse(t).detail || t; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

/* ------------------------------------------------------------------ */
/* Match state / UI sync                                               */
/* ------------------------------------------------------------------ */
function playerLabel(side) {
  if (!S.match) return side;
  return S.match.labels?.[side] || S.match.players[side];
}

function humanTurn() {
  const t = S.match?.turn;
  return t && t.team && S.match.players[t.team] !== "champion";
}

function applyMatch(m) {
  S.match = m;
  S.matchId = m.id;
  localStorage.setItem("arena_match", m.id);
  $("nameA").textContent = playerLabel("A");
  $("nameB").textContent = playerLabel("B");
  $("totalA").textContent = m.totals.A;
  $("totalB").textContent = m.totals.B;

  const grid = $("endsGrid"); grid.innerHTML = "";
  m.ends.forEach((e, i) => {
    const cell = document.createElement("div"); cell.className = "cell";
    let v = "–";
    if (e.score) v = e.score.team ? `${e.score.team}+${e.score.points}` : "0";
    cell.innerHTML = `${i + 1}<b>${v}</b>`;
    grid.appendChild(cell);
  });

  const t = m.turn;
  if (m.status === "finished") {
    $("turnLine").innerHTML = `<b>Match over — Team ${m.winner} (${playerLabel(m.winner)}) wins ${m.totals.A}:${m.totals.B}</b>`;
    $("sheetHint").textContent = "Match over. Create a new match to play again.";
  } else {
    const pp = m.ends[m.ends.length - 1].mode;
    $("turnLine").innerHTML =
      `End <b>${t.end}</b>/${m.ends_scheduled} &middot; throw <b>${t.throw}</b>/10 &middot; ` +
      `hammer <b>${t.hammer}</b>${pp !== "standard" ? ` &middot; <b>${pp}</b>` : ""} — ` +
      `<b class="${t.team}">Team ${t.team}</b> (${playerLabel(t.team)}) to throw`;
    $("sheetHint").textContent = modeHint();
  }

  // last known eval
  const throws = m.ends[m.ends.length - 1].throws || [];
  const withV = throws.filter((r) => r.value_A != null);
  if (withV.length) setEval(withV[withV.length - 1].value_A);

  // power play availability + champion resume (power-play window)
  const e = m.ends[m.ends.length - 1];
  const side = t?.team;
  const ppOk = m.status === "in_progress" && side && !(e.throws || []).length &&
    e.hammer && m.players[e.hammer] !== "champion" &&
    !m.power_play_used[e.hammer] && t.end <= m.ends_scheduled;
  $("ppBtn").classList.toggle("hidden", !ppOk);
  const holdOk = m.status === "in_progress" && side && m.players[side] === "champion";
  $("resumeBtn").classList.toggle("hidden", !holdOk);
  if (holdOk && ppOk) {
    $("sheetHint").textContent = "You have hammer: call your power play now, or let the champion throw first.";
  }
  $("undoBtn").disabled = !(m.status === "in_progress" &&
    (e.throws || []).some((r) => m.players[r.team] !== "champion"));

  syncThrowBtn();
  drawSheet();
}

function setEval(v) {
  S.evalA = v;
  const clip = Math.max(-3, Math.min(3, v));
  const half = 50 * Math.abs(clip) / 3;
  const fill = $("evalFill");
  fill.style.width = `${half}%`;
  fill.style.left = clip >= 0 ? `${50 - half}%` : "50%";
  fill.style.background = clip >= 0 ? "var(--red)" : "var(--yellow)";
  $("evalNum").textContent = (v >= 0 ? "+" : "") + Number(v).toFixed(2);
}

function syncThrowBtn() {
  $("throwBtn").disabled = !(S.solved && humanTurn() && !S.busy && !S.animating &&
                             S.match?.status === "in_progress");
}

function modeHint() {
  switch (S.mode) {
    case "draw": return "Click where your stone should come to REST.";
    case "contact": return "Click the point your stone should reach at the moment of impact (pick a weight).";
    case "after_contact": return S.selSlot == null ?
      "Click a stone to move, then its destination (or 'take out')." :
      "Now click where that stone should END UP (or use 'take out').";
    case "params": return "Set the throw with the sliders, then Throw.";
  }
  return "";
}

function log(msg, cls = "") {
  const div = document.createElement("div");
  if (cls) div.className = cls;
  div.textContent = msg;
  $("log").prepend(div);
}

/* ------------------------------------------------------------------ */
/* Solving + throwing                                                  */
/* ------------------------------------------------------------------ */
function fmtAction(a) {
  return `speed ${a[0].toFixed(3)}  angle ${(a[1] * 1000).toFixed(1)}mrad  ` +
         `spin ${a[2].toFixed(1)}  y0 ${a[3].toFixed(3)}`;
}

async function runSolve(body) {
  // Non-blocking: a newer solve supersedes an in-flight one (the server
  // serialises sim work anyway), so the sheet never feels dead while solving.
  if (!S.match) return;
  const seq = ++S.solveSeq;
  S.solved = null; syncThrowBtn();
  $("solveOut").innerHTML = "solving…";
  try {
    const out = await api(`/api/match/${S.matchId}/solve`, "POST", body);
    if (seq !== S.solveSeq) return;   // superseded by a newer click
    S.solved = out;
    const info = out.solver || {};
    const err = info.achieved_error_m != null ? `± ${info.achieved_error_m} m off target` :
                (info.removed != null ? (info.removed ? "removal predicted" : "REMOVAL NOT ACHIEVED") : "");
    const v = out.preview?.predicted_value_A;
    $("solveOut").innerHTML =
      `<b>${body.type}</b> solved: ${fmtAction(out.intended)}\n` +
      `${err}${out.preview?.illegal_takeout ? "  ⚠ ILLEGAL (early takeout — throw would be forfeited)" : ""}` +
      (v != null ? `\nchampion eval after this shot: ${v >= 0 ? "+" : ""}${v} (A perspective)` : "");
  } catch (e) {
    if (seq !== S.solveSeq) return;
    $("solveOut").innerHTML = `solve failed: ${e.message}`;
  }
  syncThrowBtn(); drawSheet();
}

function flashHint(msg) {
  $("sheetHint").textContent = msg;
  if (S.hintTimer) clearTimeout(S.hintTimer);
  S.hintTimer = setTimeout(() => { $("sheetHint").textContent = modeHint(); }, 2200);
}

async function throwNow() {
  if (!S.solved || !humanTurn() || S.busy) return;
  const side = S.match.turn.team;
  S.busy = true; syncThrowBtn();
  $("solveOut").innerHTML = "throwing…";
  try {
    const out = await api(`/api/match/${S.matchId}/throw`, "POST",
                          { side, type: "params", action: S.solved.intended });
    S.solved = null; S.targetMark = null; S.selSlot = null;
    let prev = S.match.board;
    await animateResult(out.result, side, prev);
    prev = out.result.board;
    for (const rep of out.replies || []) {
      log(`champion throws (${fmtAction(rep.throw.realized)})`, rep.throw.team);
      await animateResult(rep, rep.throw.team, prev);
      prev = rep.board;
    }
    applyMatch(out.match);
    $("solveOut").innerHTML = "";
  } catch (e) {
    $("solveOut").innerHTML = `throw failed: ${e.message}`;
  }
  S.busy = false; syncThrowBtn();
}

function computeFaded(prevBoard, newBoard) {
  const out = [];
  for (const s of prevBoard || []) {
    const n = (newBoard || []).find((q) => q.slot === s.slot);
    if (!n || Math.hypot(n.along - s.along, n.lateral - s.lateral) > 0.02) {
      out.push({ team: s.team, along: s.along, lateral: s.lateral });
    }
  }
  return out;
}

async function animateResult(result, side, prevBoard) {
  const rec = result.throw;
  if (rec.illegal_takeout) {
    log(`Team ${side} throw ${rec.n}: ILLEGAL early takeout — forfeited, board restored`, side);
  } else {
    log(`Team ${side} throw ${rec.n}/10`, side);
  }
  if (result.trajectory) {
    S.faded = computeFaded(prevBoard, result.board);  // origins fade during the throw
    await animateTrajectory(result.trajectory, result.board);
  } else {
    S.faded = [];
  }
  if (rec.value_A != null) setEval(rec.value_A);
  if (result.end_result) {
    S.faded = [];
    const er = result.end_result;
    const sc = er.score.team ? `Team ${er.score.team} scores ${er.score.points}` : "blank end";
    log(`End ${er.end}: ${sc} — totals A ${er.totals.A} : ${er.totals.B} B`, "sys");
    if (er.match_over) log(`MATCH OVER — Team ${er.winner} wins`, "sys");
  }
}

function animateTrajectory(traj, finalBoard) {
  return new Promise((resolve) => {
    if (!traj || !traj.frames?.length) { resolve(); return; }
    S.animating = true;
    let i = 0;
    const timer = setInterval(() => {
      const frame = traj.frames[Math.min(i, traj.frames.length - 1)];
      const board = [];
      frame.forEach((xy, slot) => {
        if (xy && xy[0] != null) board.push({ slot, team: slot < 6 ? "A" : "B",
                                              along: xy[0], lateral: xy[1] });
      });
      drawSheet(board);   // faded origin markers render underneath (drawBoard)
      i += 1;
      if (i > traj.frames.length + 4) {
        clearInterval(timer);
        S.animating = false;
        S.faded = [];      // the throw is over: origin markers disappear
        drawSheet(finalBoard);
        resolve();
      }
    }, 30);
  });
}

/* ------------------------------------------------------------------ */
/* Input handling                                                      */
/* ------------------------------------------------------------------ */
canvas.addEventListener("click", (ev) => {
  if (!S.match) { flashHint("create a match first (button top right)"); return; }
  if (S.match.status !== "in_progress") { flashHint("match is over — start a new match"); return; }
  if (S.busy || S.animating) { flashHint("wait — a throw is in progress"); return; }
  if (!humanTurn()) {
    const t = S.match.turn;
    flashHint(t && S.match.players[t.team] === "champion"
      ? 'champion’s turn — press “Let champion throw”'
      : "not your turn");
    return;
  }
  const [along, lat] = evToM(ev);
  const side = S.match.turn.team;
  if (S.mode === "draw") {
    S.targetMark = [along, lat];
    drawSheet();
    runSolve({ side, type: "draw", target: [along, lat] });
  } else if (S.mode === "contact") {
    S.targetMark = [along, lat];
    drawSheet();
    runSolve({ side, type: "contact", target: [along, lat], weight: S.weight });
  } else if (S.mode === "after_contact") {
    const near = nearestStone(along, lat);
    if (S.selSlot == null || (near && near.d < 0.35)) {
      if (!near || near.d > 0.6) return;
      S.selSlot = near.s.slot;
      S.targetMark = null; S.solved = null;
      $("afterStatus").textContent =
        `Selected ${near.s.team}${near.s.slot % 6 + 1}. Click its destination, or take it out.`;
      $("removeBtn").classList.remove("hidden");
      $("sheetHint").textContent = modeHint();
      drawSheet();
    } else {
      S.targetMark = [along, lat];
      drawSheet();
      runSolve({ side, type: "after_contact", stone_slot: S.selSlot, target: [along, lat] });
    }
  }
});

function nearestStone(along, lat) {
  let best = null;
  for (const s of S.match.board || []) {
    const d = Math.hypot(s.along - along, s.lateral - lat);
    if (!best || d < best.d) best = { s, d };
  }
  return best;
}

$("removeBtn").addEventListener("click", () => {
  if (S.selSlot == null || !humanTurn()) return;
  S.targetMark = null;
  runSolve({ side: S.match.turn.team, type: "after_contact",
             stone_slot: S.selSlot, remove: true });
});

document.querySelectorAll("#modeTabs .tab").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll("#modeTabs .tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    S.mode = b.dataset.mode;
    S.solved = null; S.targetMark = null; S.selSlot = null; S.guide = null;
    $("removeBtn").classList.add("hidden");
    $("afterStatus").textContent = "Click one of the stones on the sheet.";
    $("paramsControls").classList.toggle("hidden", S.mode !== "params");
    $("contactControls").classList.toggle("hidden", S.mode !== "contact");
    $("afterControls").classList.toggle("hidden", S.mode !== "after_contact");
    $("modeHelp").textContent = modeHint();
    $("sheetHint").textContent = modeHint();
    $("solveOut").innerHTML = "";
    if (S.mode === "params") onParamChange();
    syncThrowBtn(); drawSheet();
  });
});

document.querySelectorAll(".wbtn").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll(".wbtn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    S.weight = b.dataset.w;
    if (S.targetMark) runSolve({ side: S.match.turn.team, type: "contact",
                                 target: S.targetMark, weight: S.weight });
  });
});

function onParamChange() {
  $("speedVal").textContent = `${(+$("speed").value).toFixed(3)} m/s`;
  $("angleVal").textContent = `${(+$("angle").value * 1000).toFixed(1)} mrad`;
  $("spinVal").textContent = `${(+$("spin").value).toFixed(1)} rad/s ${+$("spin").value >= 0 ? "(curls right)" : "(curls left)"}`;
  $("y0Val").textContent = `${(+$("y0").value).toFixed(3)} m`;
  S.guide = buildGuide();
  drawSheet();
  if (S.previewTimer) clearTimeout(S.previewTimer);
  S.previewTimer = setTimeout(() => {
    if (S.mode !== "params" || !humanTurn()) return;
    runSolve({ side: S.match.turn.team, type: "params",
               action: [+$("speed").value, +$("angle").value, +$("spin").value, +$("y0").value] });
  }, 450);
}
["speed", "angle", "spin", "y0"].forEach((id) => $(id).addEventListener("input", onParamChange));

$("throwBtn").addEventListener("click", throwNow);

$("ppBtn").addEventListener("click", async () => {
  const e = S.match.ends[S.match.ends.length - 1];
  const wing = (window.prompt('Power play: "left" or "right"?', "right") || "").trim().toLowerCase();
  if (wing !== "left" && wing !== "right") return;
  try {
    const out = await api(`/api/match/${S.matchId}/powerplay`, "POST",
                          { side: e.hammer, wing });
    log(`Team ${e.hammer} calls the ${wing} power play`, "sys");
    applyMatch(out.match);
    // if the champion throws first this end, it was waiting for this decision
    const t = out.match.turn;
    if (t.team && out.match.players[t.team] === "champion") await championResume();
  } catch (err) { log(`power play refused: ${err.message}`, "sys"); }
});

async function championResume() {
  if (S.busy) return;
  S.busy = true;
  S.solved = null; S.targetMark = null;
  $("sheetHint").textContent = "champion thinking…";
  try {
    const out = await api(`/api/match/${S.matchId}/champion_move`, "POST", {});
    let prev = S.match.board;
    for (const rep of [out.result, ...(out.replies || [])]) {
      log(`champion throws (${fmtAction(rep.throw.realized)})`, rep.throw.team);
      await animateResult(rep, rep.throw.team, prev);
      prev = rep.board;
    }
    applyMatch(out.match);
  } catch (e) { log(`champion move failed: ${e.message}`, "sys"); }
  S.busy = false; syncThrowBtn();
}
$("resumeBtn").addEventListener("click", championResume);

$("undoBtn").addEventListener("click", async () => {
  if (S.busy || S.animating) { flashHint("wait — a throw is in progress"); return; }
  try {
    const out = await api(`/api/match/${S.matchId}/undo`, "POST", {});
    S.solved = null; S.targetMark = null; S.selSlot = null; S.faded = [];
    $("solveOut").innerHTML = "";
    log(`undo: rolled back to throw ${out.undo.back_to_throw} `
        + `(${out.undo.throws_undone} throw${out.undo.throws_undone > 1 ? "s" : ""} discarded)`, "sys");
    applyMatch(out.match);
  } catch (e) { flashHint(`undo refused: ${e.message}`); }
});

/* ------------------------------------------------------------------ */
/* New match                                                           */
/* ------------------------------------------------------------------ */
$("newMatchBtn").addEventListener("click", () => $("newMatchDlg").showModal());

$("newMatchDlg").addEventListener("close", async () => {
  if ($("newMatchDlg").returnValue !== "default") return;
  const side = $("nmSide").value, opp = side === "A" ? "B" : "A";
  const players = {}; players[side] = "human"; players[opp] = $("nmOpp").value;
  const labels = {}; labels[side] = $("nmName").value || "human";
  if ($("nmOpp").value === "champion") labels[opp] = "champion az_v14d";
  const hammerSel = $("nmHammer").value;
  const first_hammer = hammerSel === "random" ? "random" : (hammerSel === "you" ? side : opp);
  $("sheetHint").textContent = "creating match (first run loads the model — up to a minute)…";
  try {
    const out = await api("/api/match", "POST", {
      players, labels, ends: +$("nmEnds").value, noise: $("nmNoise").checked, first_hammer,
    });
    S.solved = null; S.targetMark = null; S.selSlot = null; S.faded = [];
    applyMatch(out.match);
    log(`new match ${out.match.id} — you are Team ${side}`, "sys");
    if (out.champion_opening) {
      for (const rep of out.champion_opening) await animateResult(rep, rep.throw.team);
      const ref = await api(`/api/match/${S.matchId}`);
      applyMatch(ref.match);
    }
  } catch (e) {
    $("sheetHint").textContent = `failed: ${e.message}`;
  }
});

/* ------------------------------------------------------------------ */
/* Boot                                                                */
/* ------------------------------------------------------------------ */
window.addEventListener("resize", () => { sizeCanvas(); drawSheet(); });

(async function boot() {
  sizeCanvas();
  $("modeHelp").textContent = modeHint();
  drawSheet();
  const prev = localStorage.getItem("arena_match");
  if (prev) {
    try {
      const out = await api(`/api/match/${prev}`);
      applyMatch(out.match);
      log(`resumed match ${prev}`, "sys");
    } catch (e) { localStorage.removeItem("arena_match"); }
  }
  fetch("/api/warmup", { method: "POST" }).catch(() => {});
})();
