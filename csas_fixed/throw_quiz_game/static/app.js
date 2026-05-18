const canvas = document.getElementById("sheet");
const ctx = canvas.getContext("2d");
const optionRow = document.getElementById("optionRow");
const scenarioName = document.getElementById("scenarioName");
const scenarioDesc = document.getElementById("scenarioDesc");
const scenarioText = document.getElementById("scenarioText");
const throwingTeam = document.getElementById("throwingTeam");
const preValue = document.getElementById("preValue");
const resultCard = document.getElementById("resultCard");
const decisionValue = document.getElementById("decisionValue");
const executionValue = document.getElementById("executionValue");
const choiceRank = document.getElementById("choiceRank");
const bestOption = document.getElementById("bestOption");
const nextBtn = document.getElementById("nextBtn");
const randomBtn = document.getElementById("randomBtn");

const colors = {
  A: "#c95031",
  B: "#395d9a",
  C: "#3c7a5d",
  D: "#946b2d",
  teamA: "#171717",
  teamB: "#f8f4e8",
};

let current = null;
let currentIndex = 0;
let locked = false;
let animationTimer = null;
let revealed = false;
let selectedOptionId = null;
let lastResult = null;

const view = {
  xMin: -2.45,
  xMax: 28.55,
  yMin: -2.45,
  yMax: 2.45,
  pad: 34,
};

function sheetScale() {
  return Math.min(
    (canvas.width - 2 * view.pad) / (view.yMax - view.yMin),
    (canvas.height - 2 * view.pad) / (view.xMax - view.xMin),
  );
}

function mToCanvasX(y) {
  const scale = sheetScale();
  const usedWidth = (view.yMax - view.yMin) * scale;
  const left = (canvas.width - usedWidth) / 2;
  return left + (y - view.yMin) * scale;
}

function mToCanvasY(x) {
  const scale = sheetScale();
  const usedHeight = (view.xMax - view.xMin) * scale;
  const top = (canvas.height - usedHeight) / 2;
  return top + (x - view.xMin) * scale;
}

function fmt(x, n = 3) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "-";
  return Number(x).toFixed(n);
}

function drawHouse() {
  const cx = mToCanvasX(0);
  const cy = mToCanvasY(0);
  ctx.save();
  ctx.strokeStyle = "rgba(23,32,27,0.32)";
  ctx.lineWidth = 2;
  ctx.strokeRect(
    mToCanvasX(view.yMin),
    mToCanvasY(view.xMin),
    (view.yMax - view.yMin) * sheetScale(),
    (view.xMax - view.xMin) * sheetScale(),
  );
  ctx.strokeStyle = "#5d625e";
  ctx.lineWidth = 2;
  [1.829, 1.219, 0.610, 0.145].forEach((r) => {
    ctx.beginPath();
    ctx.arc(cx, cy, r * sheetScale(), 0, Math.PI * 2);
    ctx.stroke();
  });
  ctx.strokeStyle = "rgba(23,32,27,0.16)";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  ctx.moveTo(mToCanvasX(view.yMin), cy);
  ctx.lineTo(mToCanvasX(view.yMax), cy);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(cx, mToCanvasY(view.xMin));
  ctx.lineTo(cx, mToCanvasY(view.xMax));
  ctx.stroke();
  [6.4, 28.35].forEach((xLine) => {
    ctx.beginPath();
    ctx.moveTo(mToCanvasX(view.yMin), mToCanvasY(xLine));
    ctx.lineTo(mToCanvasX(view.yMax), mToCanvasY(xLine));
    ctx.stroke();
  });
  ctx.restore();
}

function drawStone(stone, highlight = false) {
  const x = mToCanvasX(stone.y);
  const y = mToCanvasY(stone.x);
  const isThrowingTeam = current && stone.team === current.throwing_team;
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, highlight ? 16 : 13, 0, Math.PI * 2);
  ctx.fillStyle = isThrowingTeam ? colors.teamA : colors.teamB;
  ctx.fill();
  ctx.lineWidth = highlight ? 4 : 2;
  ctx.strokeStyle = isThrowingTeam ? "#ffffff" : "#171717";
  ctx.stroke();
  ctx.fillStyle = isThrowingTeam ? "#ffffff" : "#171717";
  ctx.font = "bold 12px ui-sans-serif, system-ui";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(stone.slot + 1), x, y);
  ctx.restore();
}

function frameToStones(frame) {
  const stones = [];
  frame.forEach((xy, slot) => {
    if (!xy || xy.length < 2) return;
    if (!Number.isFinite(xy[0]) || !Number.isFinite(xy[1])) return;
    stones.push({ slot, team: slot < 6 ? "A" : "B", x: xy[0], y: xy[1] });
  });
  return stones;
}

function drawPath(trajectory, color, width = 3, alpha = 0.8) {
  const frames = trajectory?.frames || [];
  const slot = trajectory?.stone_slot;
  const pts = [];
  frames.forEach((frame) => {
    const xy = frame?.[slot];
    if (xy && Number.isFinite(xy[0]) && Number.isFinite(xy[1])) {
      pts.push([mToCanvasX(xy[1]), mToCanvasY(xy[0])]);
    }
  });
  if (pts.length < 2) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = alpha;
  ctx.lineWidth = width;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
  ctx.stroke();
  ctx.restore();
}

function drawBoard(stones, options = [], selected = null, frameStones = null) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fbfaf3";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawHouse();

  if (current) {
    options.forEach((opt) => {
      const color = colors[opt.id] || "#c95031";
      const width = selected && selected === opt.id ? 5 : 2.5;
      const alpha = selected && selected !== opt.id ? 0.18 : 0.72;
      drawPath(opt.intended_trajectory, color, width, alpha);
    });
  }

  const drawStones = frameStones || stones;
  drawStones.forEach((stone) => drawStone(stone));

  ctx.save();
  ctx.strokeStyle = "#171717";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(mToCanvasX(0), mToCanvasY(6.4), 7, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function setActiveOption(optionId, finalBoard = null) {
  selectedOptionId = optionId;
  [...document.querySelectorAll(".option-card")].forEach((card) => {
    card.classList.toggle("selected", card.dataset.optionId === optionId);
  });
  drawBoard(current.pre_board, current.options, optionId, finalBoard);
}

function renderOptions(options, showValues = false) {
  optionRow.innerHTML = "";
  options.forEach((opt) => {
    const card = document.createElement("button");
    card.className = "option-card";
    card.dataset.optionId = opt.id;
    card.style.setProperty("--option-color", colors[opt.id] || "#c95031");
    card.disabled = locked && !showValues;
    card.innerHTML = `
      <div class="option-header">
        <span class="badge">${opt.label}</span>
        <h3>${opt.kind === "observed" ? "Observed throw" : `Throw ${opt.label}`}</h3>
      </div>
      <div class="param-grid">
        <div><span>Speed</span><strong>${fmt(opt.speed, 2)}</strong></div>
        <div><span>Angle</span><strong>${fmt(opt.angle, 3)}</strong></div>
        <div><span>Spin</span><strong>${fmt(opt.spin, 2)}</strong></div>
        <div><span>Y0</span><strong>${fmt(opt.y0, 2)}</strong></div>
      </div>
      <div class="reveal ${showValues ? "" : "hidden"}">
        Decision value: <strong>${fmt(opt.decision_value, 3)}</strong>
      </div>
    `;
    card.addEventListener("mouseenter", () => {
      if (!current || (locked && !showValues)) return;
      drawBoard(current.pre_board, current.options, opt.id);
    });
    card.addEventListener("mouseleave", () => {
      if (!current || (locked && !showValues)) return;
      if (revealed && selectedOptionId) {
        const finalBoard = lastResult?.selected_option_id === selectedOptionId ? lastResult.final_board : null;
        drawBoard(current.pre_board, current.options, selectedOptionId, finalBoard);
      } else {
        drawBoard(current.pre_board, current.options);
      }
    });
    card.addEventListener("click", () => {
      if (showValues) {
        chooseOption(opt.id);
        return;
      }
      chooseOption(opt.id);
    });
    optionRow.appendChild(card);
  });
}

function setLoading(text) {
  scenarioText.textContent = text;
  nextBtn.disabled = true;
  randomBtn.disabled = true;
}

async function loadScenario(index = 0, random = false) {
  if (animationTimer) clearInterval(animationTimer);
  locked = true;
  revealed = false;
  selectedOptionId = null;
  lastResult = null;
  resultCard.classList.add("hidden");
  setLoading("Generating three diverse near-optimal throws...");
  optionRow.innerHTML = "";
  const res = await fetch(random ? "/api/random" : `/api/scenario?index=${index}`);
  if (!res.ok) throw new Error(await res.text());
  current = await res.json();
  currentIndex = current.index;
  locked = false;
  scenarioName.textContent = current.name;
  scenarioDesc.textContent = current.description;
  throwingTeam.textContent = `${current.throwing_team}; thrower team is black`;
  preValue.textContent = fmt(current.pre_value, 3);
  scenarioText.textContent = `${current.index + 1} / ${current.count}`;
  nextBtn.disabled = false;
  randomBtn.disabled = false;
  renderOptions(current.options, false);
  drawBoard(current.pre_board, current.options);
}

async function chooseOption(optionId) {
  if (!current || locked) return;
  locked = true;
  nextBtn.disabled = true;
  randomBtn.disabled = true;
  scenarioText.textContent = `Executing throw ${optionId}...`;
  const res = await fetch("/api/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario_id: current.id, option_id: optionId }),
  });
  if (!res.ok) {
    locked = false;
    scenarioText.textContent = "Selection failed.";
    throw new Error(await res.text());
  }
  const result = await res.json();
  decisionValue.textContent = fmt(result.decision_value, 3);
  executionValue.textContent = fmt(result.execution_value, 3);
  choiceRank.textContent = `${result.selected_rank} / ${result.options.length}`;
  bestOption.textContent = result.best_option_id;
  resultCard.classList.remove("hidden");
  revealed = true;
  selectedOptionId = optionId;
  lastResult = result;
  current.options = result.options;
  renderOptions(current.options, true);
  setActiveOption(optionId);
  animateResult(result, optionId);
}

function animateResult(result, optionId) {
  const frames = result.trajectory?.frames || [];
  let i = 0;
  const baseOptions = current.options;
  drawBoard(current.pre_board, baseOptions, optionId);
  if (animationTimer) clearInterval(animationTimer);
  animationTimer = setInterval(() => {
    if (i >= frames.length) {
      clearInterval(animationTimer);
      animationTimer = null;
      lastResult = result;
      selectedOptionId = optionId;
      drawBoard(current.pre_board, baseOptions, optionId, result.final_board);
      scenarioText.textContent = `Decision ${fmt(result.decision_value, 3)} | execution ${fmt(result.execution_value, 3)}`;
      locked = false;
      nextBtn.disabled = false;
      randomBtn.disabled = false;
      return;
    }
    drawBoard(current.pre_board, baseOptions, optionId, frameToStones(frames[i]));
    i += 1;
  }, 24);
}

nextBtn.addEventListener("click", () => loadScenario(currentIndex + 1));
randomBtn.addEventListener("click", () => loadScenario(0, true));

loadScenario(0).catch((err) => {
  console.error(err);
  scenarioText.textContent = "Failed to load scenario. See server logs.";
});
