const cameraCanvas = document.getElementById("cameraCanvas");
const cameraCtx = cameraCanvas.getContext("2d");
const spaceCanvas = document.getElementById("spaceCanvas");
const spaceCtx = spaceCanvas.getContext("2d");
const trackRows = document.getElementById("trackRows");
const assetList = document.getElementById("assetList");
const eventLog = document.getElementById("eventLog");
const frameCounter = document.getElementById("frameCounter");
const trackCount = document.getElementById("trackCount");

const camera = {
  hfovDeg: 62,
  vfovDeg: 38,
  focalPx: 980,
};

const assets = [
  { name: "Asset A", type: "optical observer", status: "ready", track: null },
  { name: "Asset B", type: "response drone", status: "ready", track: null },
  { name: "Asset C", type: "response drone", status: "hold", track: null },
  { name: "Asset D", type: "operator team", status: "offline", track: null },
];

const targets = [
  makeTarget(1, "unknown_uav", -520, 180, 1750, 10.5, 0.15, -20),
  makeTarget(2, "unknown_uav", 310, 240, 2250, -8.2, -0.2, -12),
  makeTarget(3, "unknown_uav", 760, 150, 2850, -12.5, 0.1, -18),
  makeTarget(4, "bird", -260, 120, 1150, 4.3, 0.35, 8),
  makeTarget(5, "unknown_uav", 80, 320, 3150, 2.0, -0.3, -24),
];

const events = [
  "Model checkpoint loaded from latest best.pt.",
  "Camera calibration profile active.",
  "Optical track-space initialized.",
];

let frame = 0;

function makeTarget(id, className, x, y, z, vx, vy, vz) {
  return {
    id,
    className,
    x,
    y,
    z,
    vx,
    vy,
    vz,
    confidence: 0.7,
    trail: [],
    box: null,
    priority: 0,
  };
}

function resizeCanvases() {
  resizeCanvasToDisplay(cameraCanvas);
  resizeCanvasToDisplay(spaceCanvas);
}

function resizeCanvasToDisplay(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function tick() {
  resizeCanvases();
  updateTargets();
  assignAssets();
  drawCamera();
  drawTrackSpace();
  renderAssets();
  renderTrackRows();
  renderEvents();
  frameCounter.textContent = String(frame).padStart(6, "0");
  trackCount.textContent = `${targets.filter((target) => target.className !== "bird").length} tracks`;
  frame += 1;
  requestAnimationFrame(tick);
}

function updateTargets() {
  for (const target of targets) {
    target.x += target.vx;
    target.y += target.vy + Math.sin(frame / 55 + target.id) * 0.12;
    target.z += target.vz;

    if (Math.abs(target.x) > 1250) target.vx *= -1;
    if (target.y < 90 || target.y > 470) target.vy *= -1;
    if (target.z < 820 || target.z > 3450) target.vz *= -1;

    target.confidence = clamp(
      0.58 + 0.28 * Math.abs(Math.sin(frame / 70 + target.id * 1.7)),
      0.5,
      0.94,
    );

    const rangeScore = 1 - clamp(target.z / 3500, 0, 1);
    const centerScore = 1 - clamp(Math.abs(target.x) / 1300, 0, 1);
    const classScore = target.className === "bird" ? 0.25 : 1;
    target.priority = Math.round(
      100 * classScore * (0.46 * target.confidence + 0.34 * rangeScore + 0.2 * centerScore),
    );

    target.trail.push({ x: target.x, y: target.y, z: target.z });
    if (target.trail.length > 54) target.trail.shift();
  }

  if (frame % 180 === 0) {
    const top = rankedTargets()[0];
    if (top) {
      events.unshift(`Track ${pad(top.id)} priority update: ${top.priority} at ${Math.round(top.z)}m.`);
      events.splice(8);
    }
  }
}

function assignAssets() {
  for (const asset of assets) asset.track = null;
  const readyAssets = assets.filter((asset) => asset.status !== "offline");
  rankedTargets()
    .filter((target) => target.className !== "bird")
    .slice(0, readyAssets.length)
    .forEach((target, index) => {
      readyAssets[index].track = target.id;
    });
}

function drawCamera() {
  const w = cameraCanvas.width;
  const h = cameraCanvas.height;
  const horizon = h * 0.64;

  const sky = cameraCtx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, "#718aa0");
  sky.addColorStop(0.55, "#b7c4cf");
  sky.addColorStop(1, "#313942");
  cameraCtx.fillStyle = sky;
  cameraCtx.fillRect(0, 0, w, h);

  drawCloudBand(w, h);
  drawCitySilhouette(w, h, horizon);
  drawReticle(w, h);

  for (const target of targets) {
    const projection = projectCamera(target, w, h);
    target.box = projection;
    if (!projection.visible) continue;
    drawTarget(cameraCtx, target, projection);
  }

  drawCameraOverlays(w, h);
}

function drawCloudBand(w, h) {
  cameraCtx.save();
  cameraCtx.globalAlpha = 0.25;
  cameraCtx.fillStyle = "#d7e0e6";
  for (let i = 0; i < 8; i += 1) {
    const x = ((frame * 0.22 + i * 240) % (w + 280)) - 140;
    const y = h * (0.16 + (i % 3) * 0.08);
    cameraCtx.beginPath();
    cameraCtx.ellipse(x, y, 80, 18, 0, 0, Math.PI * 2);
    cameraCtx.ellipse(x + 56, y + 6, 65, 14, 0, 0, Math.PI * 2);
    cameraCtx.fill();
  }
  cameraCtx.restore();
}

function drawCitySilhouette(w, h, horizon) {
  cameraCtx.fillStyle = "#1a2229";
  cameraCtx.fillRect(0, horizon, w, h - horizon);
  cameraCtx.fillStyle = "#202a32";
  const block = w / 28;
  for (let i = 0; i < 30; i += 1) {
    const height = 30 + ((i * 17) % 70);
    cameraCtx.fillRect(i * block - 12, horizon - height, block * 0.68, height);
  }
  cameraCtx.fillStyle = "rgba(79, 224, 161, 0.14)";
  cameraCtx.fillRect(0, horizon - 2, w, 2);
}

function drawReticle(w, h) {
  cameraCtx.strokeStyle = "rgba(237, 244, 247, 0.18)";
  cameraCtx.lineWidth = 1;
  cameraCtx.beginPath();
  cameraCtx.moveTo(w / 2 - 26, h / 2);
  cameraCtx.lineTo(w / 2 + 26, h / 2);
  cameraCtx.moveTo(w / 2, h / 2 - 26);
  cameraCtx.lineTo(w / 2, h / 2 + 26);
  cameraCtx.stroke();
}

function projectCamera(target, w, h) {
  const sx = w / 2 + (target.x / target.z) * camera.focalPx * (w / 1280);
  const sy = h * 0.58 - (target.y / target.z) * camera.focalPx * (h / 720);
  const size = clamp((900 / target.z) * 34 * (w / 1280), 7, 34);
  return {
    x: sx,
    y: sy,
    size,
    visible: sx > -40 && sx < w + 40 && sy > -40 && sy < h + 40,
  };
}

function drawTarget(ctx, target, projection) {
  const color = target.className === "bird" ? "#f1c66d" : "#4fe0a1";
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;

  const boxW = projection.size * (target.className === "bird" ? 2.2 : 1.5);
  const boxH = projection.size;
  const x = projection.x - boxW / 2;
  const y = projection.y - boxH / 2;
  ctx.strokeRect(x, y, boxW, boxH);

  ctx.beginPath();
  if (target.className === "bird") {
    ctx.moveTo(projection.x - projection.size, projection.y);
    ctx.lineTo(projection.x, projection.y - projection.size * 0.24);
    ctx.lineTo(projection.x + projection.size, projection.y);
  } else {
    ctx.arc(projection.x, projection.y, Math.max(2, projection.size * 0.16), 0, Math.PI * 2);
    ctx.moveTo(projection.x - projection.size * 0.45, projection.y);
    ctx.lineTo(projection.x + projection.size * 0.45, projection.y);
    ctx.moveTo(projection.x, projection.y - projection.size * 0.42);
    ctx.lineTo(projection.x, projection.y + projection.size * 0.42);
  }
  ctx.stroke();

  ctx.font = `${Math.max(12, cameraCanvas.width / 95)}px ui-sans-serif, system-ui`;
  ctx.fillText(
    `T${pad(target.id)} ${target.className} ${target.confidence.toFixed(2)} ${Math.round(target.z)}m`,
    x,
    Math.max(18, y - 8),
  );
  ctx.restore();
}

function drawCameraOverlays(w, h) {
  cameraCtx.fillStyle = "rgba(8, 12, 15, 0.62)";
  cameraCtx.fillRect(18, 18, 280, 96);
  cameraCtx.fillStyle = "#edf4f7";
  cameraCtx.font = `${Math.max(13, w / 86)}px ui-sans-serif, system-ui`;
  cameraCtx.fillText("PASSIVE OPTICAL DETECTION", 34, 48);
  cameraCtx.fillStyle = "#9fb0ba";
  cameraCtx.fillText("long-range airborne object tracks", 34, 76);
  cameraCtx.fillStyle = "#4fe0a1";
  cameraCtx.fillText(`${rankedTargets().length} active tracks`, 34, 102);
}

function drawTrackSpace() {
  const w = spaceCanvas.width;
  const h = spaceCanvas.height;
  spaceCtx.clearRect(0, 0, w, h);
  spaceCtx.fillStyle = "#0e1419";
  spaceCtx.fillRect(0, 0, w, h);

  const origin = { x: w * 0.42, y: h * 0.76 };
  drawGrid(origin, w, h);

  for (const target of targets) {
    if (target.className === "bird") continue;
    drawSpaceTrail(target, origin);
    drawSpacePoint(target, origin);
  }
}

function drawGrid(origin, w, h) {
  const xAxis = { x: w * 0.34, y: -h * 0.16 };
  const zAxis = { x: w * 0.34, y: h * 0.13 };
  const yAxis = { x: 0, y: -h * 0.52 };

  spaceCtx.strokeStyle = "rgba(159, 176, 186, 0.22)";
  spaceCtx.lineWidth = 1;
  for (let i = 1; i <= 5; i += 1) {
    line(origin.x + (xAxis.x / 5) * i, origin.y + (xAxis.y / 5) * i, origin.x + (xAxis.x / 5) * i + zAxis.x, origin.y + (xAxis.y / 5) * i + zAxis.y);
    line(origin.x + (zAxis.x / 5) * i, origin.y + (zAxis.y / 5) * i, origin.x + (zAxis.x / 5) * i + xAxis.x, origin.y + (zAxis.y / 5) * i + xAxis.y);
  }

  spaceCtx.strokeStyle = "#73b7ff";
  line(origin.x, origin.y, origin.x + xAxis.x, origin.y + xAxis.y);
  spaceCtx.strokeStyle = "#4fe0a1";
  line(origin.x, origin.y, origin.x + yAxis.x, origin.y + yAxis.y);
  spaceCtx.strokeStyle = "#f1c66d";
  line(origin.x, origin.y, origin.x + zAxis.x, origin.y + zAxis.y);

  spaceCtx.fillStyle = "#9fb0ba";
  spaceCtx.font = `${Math.max(12, w / 42)}px ui-sans-serif, system-ui`;
  spaceCtx.fillText("X lateral", origin.x + xAxis.x - 58, origin.y + xAxis.y - 8);
  spaceCtx.fillText("Y elevation", origin.x + yAxis.x + 8, origin.y + yAxis.y + 12);
  spaceCtx.fillText("Z range", origin.x + zAxis.x - 4, origin.y + zAxis.y + 18);
  spaceCtx.fillText("camera", origin.x - 22, origin.y + 24);
}

function drawSpaceTrail(target, origin) {
  if (target.trail.length < 2) return;
  spaceCtx.beginPath();
  target.trail.forEach((point, index) => {
    const p = projectSpace(point, origin);
    if (index === 0) spaceCtx.moveTo(p.x, p.y);
    else spaceCtx.lineTo(p.x, p.y);
  });
  spaceCtx.strokeStyle = "rgba(79, 224, 161, 0.35)";
  spaceCtx.lineWidth = 2;
  spaceCtx.stroke();
}

function drawSpacePoint(target, origin) {
  const p = projectSpace(target, origin);
  const future = projectSpace(
    {
      x: target.x + target.vx * 28,
      y: target.y + target.vy * 28,
      z: target.z + target.vz * 28,
    },
    origin,
  );

  spaceCtx.strokeStyle = "rgba(79, 224, 161, 0.55)";
  spaceCtx.setLineDash([5, 5]);
  line(p.x, p.y, future.x, future.y);
  spaceCtx.setLineDash([]);

  spaceCtx.fillStyle = "#4fe0a1";
  spaceCtx.beginPath();
  spaceCtx.arc(p.x, p.y, 5 + target.priority / 30, 0, Math.PI * 2);
  spaceCtx.fill();
  spaceCtx.fillStyle = "#edf4f7";
  spaceCtx.font = `${Math.max(11, spaceCanvas.width / 48)}px ui-sans-serif, system-ui`;
  spaceCtx.fillText(`T${pad(target.id)}`, p.x + 11, p.y - 6);
}

function projectSpace(point, origin) {
  const sx = origin.x + point.x * 0.075 + point.z * 0.07;
  const sy = origin.y - point.y * 0.24 + point.z * 0.026;
  return { x: sx, y: sy };
}

function renderAssets() {
  assetList.innerHTML = assets
    .map((asset) => {
      const statusClass = asset.status === "ready" ? "" : asset.status === "hold" ? "hold" : "offline";
      const label = asset.track ? `assigned T${pad(asset.track)}` : asset.status;
      return `
        <div class="asset ${statusClass}">
          <div class="dot"></div>
          <div>
            <strong>${asset.name}</strong>
            <span>${asset.type}</span>
          </div>
          <code>${label}</code>
        </div>
      `;
    })
    .join("");
}

function renderTrackRows() {
  trackRows.innerHTML = rankedTargets()
    .slice(0, 6)
    .map((target) => {
      const bearing = bearingDeg(target);
      return `
        <div class="track-row">
          <span class="id">T${pad(target.id)}</span>
          <span>${target.className}</span>
          <span>${target.confidence.toFixed(2)}</span>
          <span>${Math.round(target.z)} m</span>
          <span>${bearing.toFixed(1)} deg</span>
          <span class="priority">${target.priority}</span>
        </div>
      `;
    })
    .join("");
}

function renderEvents() {
  eventLog.innerHTML = events.map((event) => `<div class="event">${event}</div>`).join("");
}

function rankedTargets() {
  return [...targets].sort((a, b) => b.priority - a.priority);
}

function bearingDeg(target) {
  return 35 + Math.atan2(target.x, target.z) * (180 / Math.PI);
}

function line(x1, y1, x2, y2) {
  spaceCtx.beginPath();
  spaceCtx.moveTo(x1, y1);
  spaceCtx.lineTo(x2, y2);
  spaceCtx.stroke();
}

function pad(value) {
  return String(value).padStart(2, "0");
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

window.addEventListener("resize", resizeCanvases);
resizeCanvases();
tick();
