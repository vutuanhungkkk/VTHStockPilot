/**
 * portfolio.js — Portfolio builder + risk analysis views
 */
import { store } from './state.js';
import { api } from './api.js';
import { drawSectorDonut } from './charts.js';

const PALETTE = ['#3b82f6','#8b5cf6','#10d070','#f59e0b','#f43f5e','#06b6d4','#ec4899','#84cc16'];

function fmt(v, d = 1) { return `${(v * 100).toFixed(d)}%`; }
function fmtVND(v) {
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)} bil`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)} mil`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(0)} k`;
  return `${v.toFixed(0)} VND`;
}

// ── Portfolio builder ─────────────────────────────────────────────────────
export async function handleBuildPortfolio() {
  const btn = document.getElementById('build-portfolio-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Optimising…'; }
  store.set('riskReportLoading', true);

  const profile = store.get('profile');
  try {
    const [reco, risk] = await Promise.all([
      api.buildPortfolio(profile),
      api.riskReport(profile),
    ]);
    store.set('recommendation', reco);
    store.set('riskReport', risk);
    renderPortfolioBuilder(reco, risk);
    renderRiskAnalysis(risk);
  } catch (e) {
    console.error(e);
    const el = document.getElementById('portfolio-error');
    if (el) { el.textContent = `Error: ${e.message}`; el.style.display = 'block'; }
  } finally {
    store.set('riskReportLoading', false);
    if (btn) { btn.disabled = false; btn.textContent = 'Build Portfolio'; }
  }
}

// ── Portfolio builder render ──────────────────────────────────────────────
function renderPortfolioBuilder(reco, risk) {
  // Metrics
  const p = reco.portfolio;
  const setM = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  setM('pb-return', fmt(p.expected_return));
  setM('pb-vol', fmt(p.expected_volatility));
  setM('pb-sharpe', p.sharpe_ratio.toFixed(2));
  setM('pb-sortino', p.sortino_ratio?.toFixed(2) ?? '--');
  setM('pb-divs', p.diversification_score.toFixed(3));
  setM('pb-eff-n', p.effective_n?.toFixed(1) ?? '--');
  setM('pb-maxdd', fmt(p.max_drawdown_estimate ?? 0));

  // Positions table
  const tbody = document.getElementById('portfolio-tbody');
  if (tbody) {
    tbody.innerHTML = risk.positions.map((pos, i) => `<tr>
      <td><strong>${pos.symbol}</strong><br><small style="color:var(--txt-muted)">${pos.company}</small></td>
      <td><span class="tag">${pos.sector}</span></td>
      <td style="font-family:var(--font-mono)">${fmt(pos.weight)}</td>
      <td style="font-family:var(--font-mono)">${fmtVND(pos.allocation)}</td>
      <td><span class="tag positive">${fmt(pos.expected_return)}</span></td>
      <td><span class="tag">${fmt(pos.volatility)}</span></td>
      <td style="font-family:var(--font-mono)">${pos.beta.toFixed(2)}</td>
    </tr>`).join('');
  }

  // Sector donut
  drawSectorDonut('sector-donut', p.sector_concentration || {});

  // Sector legend
  const legend = document.getElementById('sector-legend');
  if (legend) {
    legend.innerHTML = Object.entries(p.sector_concentration || {}).map(([s, w], i) =>
      `<div style="display:flex;align-items:center;gap:6px;font-size:var(--fs-xs);color:var(--txt-secondary)">
        <span style="width:10px;height:10px;border-radius:50%;background:${PALETTE[i % PALETTE.length]};flex-shrink:0"></span>
        <span>${s}</span><span style="margin-left:auto;color:var(--txt-accent);font-family:var(--font-mono)">${fmt(w)}</span>
      </div>`
    ).join('');
  }
}

// ── Risk analysis render ──────────────────────────────────────────────────
function renderRiskAnalysis(risk) {
  const setR = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  setR('ra-var95', fmt(risk.var_95 ?? 0));
  setR('ra-cvar95', fmt(risk.cvar_95 ?? 0));
  setR('ra-eff-n', risk.metrics.effective_n?.toFixed(1) ?? '--');
  setR('ra-maxdd', fmt(risk.metrics.max_drawdown_estimate ?? 0));

  // Correlation matrix
  const container = document.getElementById('corr-matrix');
  if (!container || !risk.correlation_matrix?.length) return;
  const n = risk.symbols.length;
  container.innerHTML = '';

  // Header row
  const header = document.createElement('div');
  header.style.cssText = `display:grid;grid-template-columns:60px repeat(${n}, 36px);gap:2px;margin-bottom:2px`;
  header.appendChild(Object.assign(document.createElement('div'), { style: 'width:60px' }));
  risk.symbols.forEach(sym => {
    const cell = document.createElement('div');
    cell.textContent = sym.slice(0, 5);
    cell.style.cssText = 'font-size:9px;color:var(--txt-muted);text-align:center;writing-mode:vertical-rl;transform:rotate(180deg);height:36px;display:flex;align-items:center;justify-content:center';
    header.appendChild(cell);
  });
  container.appendChild(header);

  // Data rows
  risk.correlation_matrix.forEach((row, i) => {
    const rowEl = document.createElement('div');
    rowEl.style.cssText = `display:grid;grid-template-columns:60px repeat(${n}, 36px);gap:2px`;

    const label = document.createElement('div');
    label.textContent = risk.symbols[i]?.slice(0, 6) ?? '';
    label.style.cssText = 'font-size:9px;color:var(--txt-secondary);display:flex;align-items:center;font-family:var(--font-mono)';
    rowEl.appendChild(label);

    row.forEach((val, j) => {
      const cell = document.createElement('div');
      cell.className = 'corr-cell';
      const abs = Math.abs(val);
      const alpha = Math.round(abs * 200 + 55);
      const color = val >= 0
        ? `rgba(59,130,246,${abs.toFixed(2)})`
        : `rgba(244,63,94,${abs.toFixed(2)})`;
      cell.style.background = color;
      cell.textContent = val.toFixed(2);
      cell.style.fontSize = '9px';
      rowEl.appendChild(cell);
    });
    container.appendChild(rowEl);
  });
}
