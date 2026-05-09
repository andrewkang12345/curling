const state = {
  scenarios: [],
  currentScenario: null,
  board: [],
  animationFrames: null,
  animationIndex: 0,
  animTimer: null,
  intendedTrajectory: null,
  liveGuidePath: null,
  previewTimer: null,
  previewRequestId: 0,
};

const canvas = document.getElementById("sheetCanvas");
const ctx = canvas.getContext("2d");

const scenarioSelect = document.getElementById("scenarioSelect");
const speedInput = document.getElementById("speedInput");
const angleInput = document.getElementById("angleInput");
const spinInput = document.getElementById("spinInput");
const y0Input = document.getElementById("y0Input");

const speedValue = document.getElementById("speedValue");
const angleValue = document.getElementById("angleValue");
const spinValue = document.getElementById("spinValue");
const y0Value = document.getElementById("y0Value");

const preValueOut = document.getElementById("preValueOut");
const valueOut = document.getElementById("valueOut");
const yourDvOut = document.getElementById("yourDvOut");
const athleteDvOut = document.getElementById("athleteDvOut");
const athleteOut = document.getElementById("athleteOut");
const terminalOut = document.getElementById("terminalOut");
const previewOut = document.getElementById("previewOut");
const sampledParamsOut = document.getElementById("sampledParamsOut");
const modelOut = document.getElementById("modelOut");

const scenarioName = document.getElementById("scenarioName");
const scenarioDescription = document.getElementById("scenarioDescription");

const SHEET = {
  minX: -7.1,
  maxX: 2.1,
  minY: -2.375,
  maxY: 2.375,
  nearTeeX: 0.0,
  hogToTee: 6.401,
  houseRadii: [1.829, 1.219, 0.610, 0.1524],
  stoneRadius: 0.145,
  pad: 0,
};

const GUIDE_PHYSICS = {
  dt: 0.02,
  substeps: 2,
  maxSteps: 1500,
  vStop: 0.03,
  vCap: 6.0,
  aLinear: 0.11,
  cQuadratic: 0.0,
  kCurl: 0.10,
  gammaSpin: 0.15,
  curlSpeedCap: 2.5,
  radius: 0.145,
};

function resizeCanvasForAuthenticScale() {
  const dpr = window.devicePixelRatio || 1;
  const worldW = SHEET.maxY - SHEET.minY;
  const worldH = SHEET.maxX - SHEET.minX;
  const aspect = worldH / worldW;
  const parent = canvas.parentElement;
  const parentStyle = parent ? window.getComputedStyle(parent) : null;
  const padX = parentStyle ? parseFloat(parentStyle.paddingLeft) + parseFloat(parentStyle.paddingRight) : 0;
  const padY = parentStyle ? parseFloat(parentStyle.paddingTop) + parseFloat(parentStyle.paddingBottom) : 0;
  const maxWidth = Math.max(320, (parent?.clientWidth || 900) - padX);
  const maxHeight = Math.max(420, (parent?.clientHeight || 0) - padY);
  let cssWidth = maxWidth;
  let cssHeight = Math.round(cssWidth * aspect);
  if (window.matchMedia("(min-width: 1100px)").matches && cssHeight > maxHeight) {
    cssHeight = Math.round(maxHeight);
    cssWidth = Math.round(cssHeight / aspect);
  }
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.scale(dpr, dpr);
}

function sheetMetrics() {
  const worldW = SHEET.maxY - SHEET.minY;
  const worldH = SHEET.maxX - SHEET.minX;
  const cssW = canvas.clientWidth || canvas.width;
  const cssH = canvas.clientHeight || canvas.height;
  const usableW = cssW - SHEET.pad * 2;
  const usableH = cssH - SHEET.pad * 2;
  const ppm = Math.min(usableW / worldW, usableH / worldH);
  const drawW = worldW * ppm;
  const drawH = worldH * ppm;
  const offsetX = (cssW - drawW) / 2;
  const offsetY = (cssH - drawH) / 2;
  return { ppm, offsetX, offsetY };
}

function mToCanvasX(y) {
  const { ppm, offsetX } = sheetMetrics();
  return offsetX + (y - SHEET.minY) * ppm;
}

function mToCanvasY(x) {
  const { ppm, offsetY } = sheetMetrics();
  return offsetY + (x - SHEET.minX) * ppm;
}

function drawHouse() {
  const rings = [
    { r: 1.829, fill: "#dfe9f2" },
    { r: 1.219, fill: "#ffffff" },
    { r: 0.610, fill: "#f4d9d4" },
    { r: 0.1524, fill: "#ffffff" },
  ];
  const { ppm } = sheetMetrics();
  ctx.save();
  ctx.translate(mToCanvasX(0), mToCanvasY(SHEET.nearTeeX));
  for (const ring of rings) {
    ctx.beginPath();
    ctx.arc(0, 0, ring.r * ppm, 0, Math.PI * 2);
    ctx.fillStyle = ring.fill;
    ctx.fill();
    ctx.strokeStyle = "rgba(31,42,51,0.15)";
    ctx.stroke();
  }
  ctx.restore();
}

function drawSheetLines() {
  ctx.strokeStyle = "rgba(44,111,178,0.16)";
  ctx.lineWidth = 2;
  const centerX = mToCanvasX(0);
  ctx.beginPath();
  ctx.moveTo(centerX, mToCanvasY(SHEET.minX));
  ctx.lineTo(centerX, mToCanvasY(SHEET.maxX));
  ctx.stroke();

  for (const x of [
    SHEET.nearTeeX,
    SHEET.nearTeeX - 2.13,
    SHEET.nearTeeX - SHEET.hogToTee,
  ]) {
    ctx.beginPath();
    ctx.moveTo(mToCanvasX(SHEET.minY), mToCanvasY(x));
    ctx.lineTo(mToCanvasX(SHEET.maxY), mToCanvasY(x));
    ctx.stroke();
  }
}

function stonePathFromTrajectory(trajectory) {
  if (!trajectory || !trajectory.frames || !trajectory.frames.length) return [];
  const slot = trajectory.stone_slot;
  const path = [];
  for (const frame of trajectory.frames) {
    const xy = frame[slot];
    if (xy && Number.isFinite(xy[0]) && Number.isFinite(xy[1])) {
      path.push({ x: xy[0], y: xy[1], slot });
    }
  }
  return path;
}

function nextSlotForClientBoard() {
  const occupied = new Set((state.board || []).map((s) => s.slot));
  const start = Math.round(Number(state.currentScenario?.stone_block || 0)) === 0 ? 0 : 6;
  for (let slot = start; slot < start + 6; slot += 1) {
    if (!occupied.has(slot)) return slot;
  }
  return start + 5;
}

function buildLiveGuidePath() {
  if (!state.currentScenario) return null;
  const speed = Number(speedInput.value);
  const angle = Number(angleInput.value);
  let omega = Number(spinInput.value);
  const y0 = Number(y0Input.value);
  const slot = nextSlotForClientBoard();
  const staticStones = (state.board || [])
    .filter((s) => Number.isFinite(s.x) && Number.isFinite(s.y))
    .map((s) => ({ x: Number(s.x), y: Number(s.y) }));
  let x = -SHEET.hogToTee;
  let y = y0;
  let vx = Math.cos(angle) * speed;
  let vy = Math.sin(angle) * speed;
  const frames = [];
  const microDt = GUIDE_PHYSICS.dt / GUIDE_PHYSICS.substeps;

  function speedClip() {
    const sp = Math.hypot(vx, vy);
    if (sp > GUIDE_PHYSICS.vCap) {
      const scale = GUIDE_PHYSICS.vCap / (sp + 1e-8);
      vx *= scale;
      vy *= scale;
    }
  }

  function inContact() {
    const thresh = 2 * GUIDE_PHYSICS.radius;
    for (const stone of staticStones) {
      if (Math.hypot(stone.x - x, stone.y - y) < thresh) return true;
    }
    return false;
  }

  for (let step = 0; step < GUIDE_PHYSICS.maxSteps; step += 1) {
    const frame = Array.from({ length: 12 }, () => [null, null]);
    frame[slot] = [x, y];
    frames.push(frame);
    let collided = false;
    let done = false;
    for (let sub = 0; sub < GUIDE_PHYSICS.substeps; sub += 1) {
      const speedNow = Math.hypot(vx, vy);
      const vhatX = vx / (speedNow + 1e-8);
      const vhatY = vy / (speedNow + 1e-8);
      const aLinX = -GUIDE_PHYSICS.aLinear * vhatX;
      const aLinY = -GUIDE_PHYSICS.aLinear * vhatY;
      const aQuadX = -GUIDE_PHYSICS.cQuadratic * speedNow * vx;
      const aQuadY = -GUIDE_PHYSICS.cQuadratic * speedNow * vy;
      const perpX = -vhatY;
      const perpY = vhatX;
      const sEff = GUIDE_PHYSICS.curlSpeedCap * Math.tanh(speedNow / GUIDE_PHYSICS.curlSpeedCap);
      const aLatX = GUIDE_PHYSICS.kCurl * omega * perpX * sEff;
      const aLatY = GUIDE_PHYSICS.kCurl * omega * perpY * sEff;

      vx += (aLinX + aQuadX + aLatX) * microDt;
      vy += (aLinY + aQuadY + aLatY) * microDt;
      speedClip();
      x += vx * microDt;
      y += vy * microDt;
      omega += (-GUIDE_PHYSICS.gammaSpin * omega) * microDt;

      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(vx) || !Number.isFinite(vy) || !Number.isFinite(omega)) {
        done = true;
        break;
      }
      if (inContact()) {
        collided = true;
        break;
      }
      if (Math.hypot(vx, vy) < GUIDE_PHYSICS.vStop) {
        done = true;
      }
    }
    if (collided) break;
    if (x > SHEET.maxX || y < SHEET.minY || y > SHEET.maxY) break;
    if (done) {
      const finalFrame = Array.from({ length: 12 }, () => [null, null]);
      finalFrame[slot] = [x, y];
      frames.push(finalFrame);
      break;
    }
  }
  return { stone_slot: slot, frames };
}

function drawIntendedTrajectory() {
  const path = stonePathFromTrajectory(state.liveGuidePath || state.intendedTrajectory);
  if (!path.length) return;
  ctx.save();
  ctx.setLineDash([14, 12]);
  ctx.lineWidth = 5;
  ctx.strokeStyle = "rgba(181,66,47,0.88)";
  ctx.beginPath();
  ctx.moveTo(mToCanvasX(path[0].y), mToCanvasY(path[0].x));
  for (const point of path.slice(1)) {
    ctx.lineTo(mToCanvasX(point.y), mToCanvasY(point.x));
  }
  ctx.stroke();
  const finalPoint = path[path.length - 1];
  ctx.beginPath();
  ctx.arc(mToCanvasX(finalPoint.y), mToCanvasY(finalPoint.x), 9, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(181,66,47,0.9)";
  ctx.fill();
  ctx.restore();
}

function drawStone(stone, fill) {
  const { ppm } = sheetMetrics();
  const r = SHEET.stoneRadius * ppm;
  ctx.beginPath();
  ctx.arc(mToCanvasX(stone.y), mToCanvasY(stone.x), r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(31,42,51,0.5)";
  ctx.stroke();
}

function drawBoard(stones) {
  resizeCanvasForAuthenticScale();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cssW = canvas.clientWidth || canvas.width;
  const cssH = canvas.clientHeight || canvas.height;
  ctx.fillStyle = "#f7fbff";
  ctx.fillRect(0, 0, cssW, cssH);
  const { offsetX, offsetY, ppm } = sheetMetrics();
  ctx.fillStyle = "rgba(221, 231, 238, 0.65)";
  ctx.fillRect(0, 0, cssW, offsetY);
  ctx.fillRect(0, offsetY + (SHEET.maxX - SHEET.minX) * ppm, cssW, cssH);
  ctx.fillRect(0, 0, offsetX, cssH);
  ctx.fillRect(offsetX + (SHEET.maxY - SHEET.minY) * ppm, 0, cssW, cssH);
  drawSheetLines();
  drawHouse();
  drawIntendedTrajectory();

  for (const stone of stones) {
    drawStone(stone, stone.team === "A" ? "#c95a48" : "#3878bd");
  }
}

function stopAnimation() {
  if (state.animTimer) {
    clearInterval(state.animTimer);
    state.animTimer = null;
  }
}

function animateTrajectory(trajectory) {
  stopAnimation();
  state.animationFrames = trajectory?.frames || [];
  state.animationIndex = 0;
  state.animTimer = setInterval(() => {
    const frame = state.animationFrames[state.animationIndex];
    const stones = frame.map((xy, idx) => ({
      x: xy[0],
      y: xy[1],
      team: idx < 6 ? "A" : "B",
      slot: idx,
    })).filter((s) => Number.isFinite(s.x) && Number.isFinite(s.y));
    drawBoard(stones);
    state.animationIndex += 1;
    if (state.animationIndex >= state.animationFrames.length) {
      stopAnimation();
      drawBoard(state.board);
    }
  }, 34);
}

function syncControlLabels() {
  speedValue.textContent = Number(speedInput.value).toFixed(3);
  angleValue.textContent = `${(Number(angleInput.value) * 180 / Math.PI).toFixed(2)} deg`;
  spinValue.textContent = Number(spinInput.value).toFixed(3);
  y0Value.textContent = `${Number(y0Input.value).toFixed(3)} m`;
}

function currentPayload() {
  return {
    scenario_id: state.currentScenario.id,
    speed: Number(speedInput.value),
    angle: Number(angleInput.value),
    spin: Number(spinInput.value),
    y0: Number(y0Input.value),
  };
}

async function fetchPreview() {
  if (!state.currentScenario) return;
  const requestId = state.previewRequestId;
  const res = await fetch("/api/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(currentPayload()),
  });
  if (!res.ok) return;
  const data = await res.json();
  if (requestId !== state.previewRequestId) return;
  state.intendedTrajectory = data.intended_trajectory;
  state.liveGuidePath = null;
  previewOut.textContent =
    `pre=${Number(data.pre_value).toFixed(3)}\n` +
    `intended post=${Number(data.intended_post_value).toFixed(3)}\n` +
    `intended dv=${Number(data.intended_value_diff).toFixed(3)}`;
  drawBoard(state.board);
}

function schedulePreview() {
  syncControlLabels();
  state.previewRequestId += 1;
  state.liveGuidePath = buildLiveGuidePath();
  drawBoard(state.board);
  if (state.previewTimer) clearTimeout(state.previewTimer);
  state.previewTimer = setTimeout(fetchPreview, 220);
}

function loadScenarioById(id) {
  const scenario = state.scenarios.find((item) => item.id === id);
  if (!scenario) return;
  state.currentScenario = scenario;
  state.board = scenario.pre_board;
  state.intendedTrajectory = null;
  state.liveGuidePath = null;
  scenarioName.textContent = scenario.name;
  scenarioDescription.textContent = scenario.description;
  athleteOut.textContent = scenario.athlete_label;
  athleteDvOut.textContent = Number(scenario.athlete_value_diff).toFixed(3);
  preValueOut.textContent = Number(scenario.pre_value).toFixed(3);
  valueOut.textContent = "-";
  yourDvOut.textContent = "-";
  terminalOut.textContent = "-";
  sampledParamsOut.textContent = "-";
  previewOut.textContent = "-";
  speedInput.value = scenario.defaults.speed;
  angleInput.value = scenario.defaults.angle;
  spinInput.value = scenario.defaults.spin;
  y0Input.value = scenario.defaults.y0;
  syncControlLabels();
  state.previewRequestId += 1;
  state.liveGuidePath = buildLiveGuidePath();
  drawBoard(state.board);
  fetchPreview();
}

async function fetchScenarios() {
  const res = await fetch("/api/scenarios");
  const data = await res.json();
  state.scenarios = data.scenarios;
  scenarioSelect.innerHTML = "";
  for (const scenario of state.scenarios) {
    const option = document.createElement("option");
    option.value = scenario.id;
    option.textContent = scenario.name;
    scenarioSelect.appendChild(option);
  }
  loadScenarioById(state.scenarios[0].id);
}

window.addEventListener("resize", () => {
  if (state.board.length) drawBoard(state.board);
});

async function throwShot() {
  if (!state.currentScenario) return;
  const res = await fetch("/api/play", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(currentPayload()),
  });
  if (!res.ok) {
    const err = await res.text();
    alert(err);
    return;
  }
  const data = await res.json();
  state.board = data.final_board;
  state.intendedTrajectory = data.intended_trajectory;
  preValueOut.textContent = Number(data.pre_value).toFixed(3);
  valueOut.textContent = Number(data.your_post_value).toFixed(3);
  yourDvOut.textContent = Number(data.your_value_diff).toFixed(3);
  athleteDvOut.textContent = Number(data.athlete_value_diff).toFixed(3);
  athleteOut.textContent = data.athlete_label;
  terminalOut.textContent = data.terminal ? "Yes" : "No";
  sampledParamsOut.textContent =
    `speed=${data.sampled_params.speed.toFixed(3)}\n` +
    `angle=${(data.sampled_params.angle * 180 / Math.PI).toFixed(2)} deg\n` +
    `spin=${data.sampled_params.spin.toFixed(3)}\n` +
    `y0=${data.sampled_params.y0.toFixed(3)} m`;
  previewOut.textContent =
    `athlete dv=${Number(data.observed_value_diff).toFixed(3)}\n` +
    `your dv=${Number(data.your_value_diff).toFixed(3)}\n` +
    `observed post=${Number(data.observed_post_value).toFixed(3)}`;
  modelOut.textContent = data.model_path;
  animateTrajectory(data.trajectory);
}

document.getElementById("shootBtn").addEventListener("click", throwShot);
document.getElementById("resetBtn").addEventListener("click", () => loadScenarioById(state.currentScenario.id));
document.getElementById("randomScenarioBtn").addEventListener("click", () => {
  const choice = state.scenarios[Math.floor(Math.random() * state.scenarios.length)];
  scenarioSelect.value = choice.id;
  loadScenarioById(choice.id);
});
scenarioSelect.addEventListener("change", (e) => loadScenarioById(e.target.value));

for (const el of [speedInput, angleInput, spinInput, y0Input]) {
  el.addEventListener("input", schedulePreview);
}

fetchScenarios();
