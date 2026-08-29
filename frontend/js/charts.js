/**
 * charts.js — All chart rendering (Canvas 2D, no external dependency)
 */

const PALETTE = [
  '#3b82f6', '#8b5cf6', '#10d070', '#f59e0b', '#f43f5e',
  '#06b6d4', '#ec4899', '#84cc16', '#f97316', '#a78bfa',
];

function pct(v) { return `${(v * 100).toFixed(1)}%`; }
function formatVND(v) {
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)} bil`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)} mil`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(0)} k`;
  return v.toFixed(0);
}

// ── Equity curve + benchmark ───────────────────────────────────────────────
export function drawEquityCurve(canvasId, data, rollingData = []) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !data?.length) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.offsetWidth * dpr;
  canvas.height = canvas.offsetHeight * dpr;
  ctx.scale(dpr, dpr);
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  const pad = { top: 24, right: 24, bottom: 40, left: 56 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top - pad.bottom;

  const allVals = data.flatMap(d => [d.portfolio, d.benchmark]);
  const minV = Math.min(...allVals) * 0.995;
  const maxV = Math.max(...allVals) * 1.005;

  function xPos(i) { return pad.left + (i / (data.length - 1)) * cW; }
  function yPos(v) { return pad.top + cH - ((v - minV) / (maxV - minV)) * cH; }

  // Grid
  ctx.strokeStyle = 'rgba(99,172,255,0.08)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const y = pad.top + (i / 5) * cH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    const val = maxV - (i / 5) * (maxV - minV);
    ctx.fillStyle = '#4d6380';
    ctx.font = `11px "JetBrains Mono", monospace`;
    ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(3), pad.left - 8, y + 4);
  }

  // X-axis labels
  const step = Math.max(1, Math.floor(data.length / 8));
  ctx.fillStyle = '#4d6380';
  ctx.textAlign = 'center';
  ctx.font = '10px Inter, sans-serif';
  data.forEach((d, i) => {
    if (i % step === 0) ctx.fillText(d.period, xPos(i), H - 8);
  });

  // Benchmark line
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(100,116,139,0.6)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  data.forEach((d, i) => {
    const x = xPos(i), y = yPos(d.benchmark);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  // Portfolio fill
  const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
  grad.addColorStop(0, 'rgba(59,130,246,0.3)');
  grad.addColorStop(1, 'rgba(59,130,246,0.0)');
  ctx.beginPath();
  data.forEach((d, i) => {
    const x = xPos(i), y = yPos(d.portfolio);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.lineTo(xPos(data.length - 1), yPos(minV));
  ctx.lineTo(xPos(0), yPos(minV));
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Portfolio line
  ctx.beginPath();
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 2.5;
  data.forEach((d, i) => {
    const x = xPos(i), y = yPos(d.portfolio);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ── Drawdown chart ─────────────────────────────────────────────────────────
export function drawDrawdown(canvasId, data) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !data?.length) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.offsetWidth * dpr;
  canvas.height = canvas.offsetHeight * dpr;
  ctx.scale(dpr, dpr);
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  const pad = { top: 16, right: 24, bottom: 32, left: 56 };
  const cW = W - pad.left - pad.right, cH = H - pad.top - pad.bottom;

  const minDD = Math.min(...data.map(d => d.drawdown));

  function xP(i) { return pad.left + (i / (data.length - 1)) * cW; }
  function yP(v) { return pad.top + ((v) / minDD) * cH; }  // 0 at top, minDD at bottom

  // Grid
  ctx.strokeStyle = 'rgba(244,63,94,0.08)'; ctx.lineWidth = 1;
  [0, 0.25, 0.5, 0.75, 1].forEach(t => {
    const y = pad.top + t * cH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    const val = t * minDD;
    ctx.fillStyle = '#4d6380'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    ctx.fillText(pct(val), pad.left - 6, y + 4);
  });

  // Fill
  const grad = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
  grad.addColorStop(0, 'rgba(244,63,94,0.0)');
  grad.addColorStop(1, 'rgba(244,63,94,0.25)');
  ctx.beginPath();
  ctx.moveTo(xP(0), yP(0));
  data.forEach((d, i) => ctx.lineTo(xP(i), yP(d.drawdown)));
  ctx.lineTo(xP(data.length - 1), yP(0));
  ctx.closePath(); ctx.fillStyle = grad; ctx.fill();

  // Line
  ctx.beginPath(); ctx.strokeStyle = '#f43f5e'; ctx.lineWidth = 1.5;
  data.forEach((d, i) => { const x = xP(i), y = yP(d.drawdown); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke();
}

// ── Sector donut ───────────────────────────────────────────────────────────
export function drawSectorDonut(canvasId, sectorData) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.offsetWidth * dpr;
  canvas.height = canvas.offsetHeight * dpr;
  ctx.scale(dpr, dpr);
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  const cx = W / 2, cy = H / 2, r = Math.min(W, H) / 2 - 16, inner = r * 0.58;

  const entries = Object.entries(sectorData);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  let angle = -Math.PI / 2;

  entries.forEach(([sector, weight], i) => {
    const sweep = (weight / total) * 2 * Math.PI;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, angle, angle + sweep);
    ctx.closePath();
    ctx.fillStyle = PALETTE[i % PALETTE.length];
    ctx.fill();
    angle += sweep;
  });

  // Hole
  ctx.beginPath();
  ctx.arc(cx, cy, inner, 0, 2 * Math.PI);
  ctx.fillStyle = '#0d1420'; ctx.fill();

  // Centre text
  ctx.fillStyle = '#e8edf7'; ctx.textAlign = 'center'; ctx.font = `bold 13px Inter`;
  ctx.fillText(`${entries.length}`, cx, cy - 4);
  ctx.fillStyle = '#4d6380'; ctx.font = `10px Inter`;
  ctx.fillText('SECTORS', cx, cy + 12);
}

// ── Rolling Sharpe ─────────────────────────────────────────────────────────
export function drawRollingSharpe(canvasId, data) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !data?.length) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.offsetWidth * dpr;
  canvas.height = canvas.offsetHeight * dpr;
  ctx.scale(dpr, dpr);
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  const pad = { top: 16, right: 24, bottom: 32, left: 48 };
  const cW = W - pad.left - pad.right, cH = H - pad.top - pad.bottom;

  const vals = data.map(d => d.sharpe);
  const minV = Math.min(...vals, 0) - 0.2, maxV = Math.max(...vals) + 0.2;

  function xP(i) { return pad.left + (i / (data.length - 1)) * cW; }
  function yP(v) { return pad.top + cH - ((v - minV) / (maxV - minV)) * cH; }

  // Zero line
  ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.lineWidth = 1;
  const yZero = yP(0);
  ctx.beginPath(); ctx.moveTo(pad.left, yZero); ctx.lineTo(W - pad.right, yZero); ctx.stroke();

  // Fill
  ctx.beginPath();
  data.forEach((d, i) => { const x = xP(i), y = yP(d.sharpe); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.lineTo(xP(data.length - 1), yP(0)); ctx.lineTo(xP(0), yP(0)); ctx.closePath();
  const g = ctx.createLinearGradient(0, pad.top, 0, H);
  g.addColorStop(0, 'rgba(16,208,112,0.3)'); g.addColorStop(1, 'rgba(16,208,112,0.0)');
  ctx.fillStyle = g; ctx.fill();

  ctx.beginPath(); ctx.strokeStyle = '#10d070'; ctx.lineWidth = 2;
  data.forEach((d, i) => { const x = xP(i), y = yP(d.sharpe); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
  ctx.stroke();
}

// ── Metric bar chart (for model comparison) ────────────────────────────────
export function drawMetricBars(canvasId, labels, values, color = '#3b82f6') {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.offsetWidth * dpr;
  canvas.height = canvas.offsetHeight * dpr;
  ctx.scale(dpr, dpr);
  const W = canvas.offsetWidth, H = canvas.offsetHeight;
  const pad = { top: 8, right: 16, bottom: 40, left: 48 };
  const barW = ((W - pad.left - pad.right) / labels.length) * 0.6;
  const gap = (W - pad.left - pad.right) / labels.length;
  const maxV = Math.max(...values) * 1.1;

  values.forEach((v, i) => {
    const x = pad.left + i * gap + gap * 0.2;
    const barH = ((v / maxV) * (H - pad.top - pad.bottom));
    const y = H - pad.bottom - barH;
    const g = ctx.createLinearGradient(0, y, 0, H - pad.bottom);
    g.addColorStop(0, color); g.addColorStop(1, color + '44');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, [4, 4, 0, 0]);
    ctx.fill();
    ctx.fillStyle = '#8da4c4'; ctx.font = '9px Inter'; ctx.textAlign = 'center';
    ctx.fillText(labels[i], x + barW / 2, H - pad.bottom + 14);
    ctx.fillStyle = '#e8edf7'; ctx.font = '10px monospace';
    ctx.fillText(v.toFixed(3), x + barW / 2, y - 4);
  });
}
