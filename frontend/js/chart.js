export function drawEquityChart(canvas, points) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const pad = { left: 48, right: 18, top: 20, bottom: 32 };
  const values = points.flatMap(p => [p.portfolio, p.benchmark]);
  const min = Math.min(...values) * .98;
  const max = Math.max(...values) * 1.02;
  const x = i => pad.left + i * (width - pad.left - pad.right) / Math.max(points.length - 1, 1);
  const y = value => pad.top + (max - value) * (height - pad.top - pad.bottom) / Math.max(max - min, .01);
  ctx.font = "10px ui-monospace";
  ctx.fillStyle = "#75807a";
  ctx.strokeStyle = "#e1e6e3";
  for (let i = 0; i <= 4; i++) {
    const value = min + (max - min) * i / 4;
    const py = y(value);
    ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(width - pad.right, py); ctx.stroke();
    ctx.fillText(`${((value - 1) * 100).toFixed(0)}%`, 7, py + 3);
  }
  [["portfolio", "#126a4a"], ["benchmark", "#356c9a"]].forEach(([key, color]) => {
    ctx.beginPath(); ctx.lineWidth = 2.25; ctx.strokeStyle = color;
    points.forEach((point, i) => i ? ctx.lineTo(x(i), y(point[key])) : ctx.moveTo(x(i), y(point[key])));
    ctx.stroke();
  });
  const labelEvery = Math.max(Math.ceil(points.length / 6), 1);
  points.forEach((point, i) => { if (i % labelEvery === 0) ctx.fillText(point.period, x(i) - 8, height - 9); });
}
