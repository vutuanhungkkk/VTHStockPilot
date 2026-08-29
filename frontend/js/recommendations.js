/**
 * recommendations.js — Recommendation ranking view
 */
import { store } from './state.js';
import { api } from './api.js';
import { runWithProgress } from './websocket.js';

const PALETTE = ['#3b82f6','#8b5cf6','#10d070','#f59e0b','#f43f5e','#06b6d4','#ec4899','#84cc16','#f97316','#a78bfa'];

function fmt(v, digits = 1) { return `${(v * 100).toFixed(digits)}%`; }
function fmtVND(v) {
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)} bil`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)} mil`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(0)} k`;
  return `${v.toFixed(0)} VND`;
}

// ── Build profile from sidebar form ──────────────────────────────────────
function buildProfile() {
  const preferred = [...document.querySelectorAll('#sector-checks input:checked')].map(el => el.value);
  return {
    capital: +document.getElementById('capital').value,
    risk_level: document.querySelector('input[name="risk"]:checked')?.value || 'balanced',
    horizon_months: +document.getElementById('horizon').value,
    max_positions: +document.getElementById('positions').value,
    max_position_weight: +document.getElementById('max-weight').value / 100,
    preferred_sectors: preferred,
    excluded_sectors: [],
    esg_filter: document.getElementById('esg-filter')?.checked || false,
  };
}

// ── Render recommendation table ──────────────────────────────────────────
function renderTable(items) {
  const tbody = document.getElementById('reco-tbody');
  if (!tbody) return;
  tbody.innerHTML = items.map((item, i) => {
    const rankClass = i < 3 ? `rank-${i + 1}` : '';
    const scoreWidth = Math.min(100, item.score * 300);
    const dotColors = Object.values(item.signals).map((v, j) => {
      const hex = PALETTE[j % PALETTE.length];
      const alpha = Math.round(v * 255).toString(16).padStart(2,'0');
      return `<span class="signal-dot" style="background:${hex}${alpha}" data-tooltip="${Object.keys(item.signals)[j]}: ${(v*100).toFixed(0)}%"></span>`;
    }).join('');

    const flags = item.risk_flags.map(f => `<span class="tag warning">${f}</span>`).join(' ');
    return `<tr>
      <td><span class="rank-badge ${rankClass}">${item.rank}</span></td>
      <td>
        <div style="font-weight:600;font-size:var(--fs-sm)">${item.company}</div>
        <div style="font-size:var(--fs-xs);color:var(--txt-muted);font-family:var(--font-mono)">${item.symbol} · ${item.sector}</div>
      </td>
      <td>
        <div class="score-bar-wrap">
          <div class="score-bar"><div class="score-bar-fill" style="width:${scoreWidth}%"></div></div>
          <span class="score-num">${item.score.toFixed(3)}</span>
        </div>
      </td>
      <td><span class="tag positive">${fmt(item.expected_return)}</span></td>
      <td><span class="tag ${item.volatility > 0.28 ? 'warning' : ''}">${fmt(item.volatility)}</span></td>
      <td style="font-weight:600;color:var(--txt-accent);font-family:var(--font-mono)">${fmt(item.weight)}</td>
      <td>
        <div class="signals">${dotColors}</div>
        ${flags ? `<div style="margin-top:4px">${flags}</div>` : ''}
      </td>
    </tr>`;
  }).join('');
}

// ── Render portfolio metrics ─────────────────────────────────────────────
function renderMetrics(portfolio, regime) {
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('metric-return', fmt(portfolio.expected_return));
  set('metric-volatility', fmt(portfolio.expected_volatility));
  set('metric-sharpe', portfolio.sharpe_ratio.toFixed(2));
  set('metric-sortino', portfolio.sortino_ratio?.toFixed(2) ?? '--');
  set('metric-divs', portfolio.diversification_score.toFixed(3));
  set('metric-eff-n', portfolio.effective_n?.toFixed(1) ?? '--');

  const regimeEl = document.getElementById('regime-badge');
  if (regimeEl) {
    regimeEl.className = `regime-badge ${regime}`;
    regimeEl.textContent = regime.replace('_', ' ').toUpperCase();
  }
}

// ── Render allocation bars ───────────────────────────────────────────────
function renderAllocation(items) {
  const container = document.getElementById('alloc-bars');
  if (!container) return;
  container.innerHTML = items.map((item, i) => `
    <div class="alloc-item">
      <span class="alloc-symbol">${item.symbol}</span>
      <div class="alloc-bar-wrap">
        <div class="alloc-bar-fill" style="width:${item.weight * 100}%;background:${PALETTE[i % PALETTE.length]}"></div>
      </div>
      <span class="alloc-pct">${fmt(item.weight)}</span>
      <span class="alloc-vnd">${fmtVND(item.allocation)}</span>
    </div>`).join('');
}

// ── Render SHAP + explanation ────────────────────────────────────────────
function renderExplanations(items) {
  const container = document.getElementById('explanations');
  if (!container) return;
  container.innerHTML = items.map(item => {
    const shap = item.shap_contributions || {};
    const maxAbs = Math.max(...Object.values(shap).map(Math.abs), 0.001);
    const shapRows = Object.entries(shap)
      .filter(([, v]) => Math.abs(v) > 0.001)
      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
      .slice(0, 8)
      .map(([k, v]) => {
        const w = Math.abs(v) / maxAbs * 100;
        const cls = v >= 0 ? 'pos' : 'neg';
        return `<div class="shap-row">
          <span class="shap-label">${k.replace(/_/g,' ')}</span>
          <div class="shap-bar-wrap"><div class="shap-bar-fill ${cls}" style="width:${w}%"></div></div>
          <span class="shap-val">${v.toFixed(3)}</span>
        </div>`;
      }).join('');

    return `<div class="card" style="margin-bottom:var(--sp-4)">
      <div class="card-header">
        <div>
          <div class="card-eyebrow">Signal Breakdown</div>
          <h3>${item.company} <span style="color:var(--txt-muted);font-size:var(--fs-sm)">${item.symbol}</span></h3>
        </div>
        <span class="tag ${item.confidence > 0.7 ? 'positive' : ''}">${(item.confidence*100).toFixed(0)}% confidence</span>
      </div>
      ${item.explanation_text ? `<div class="explanation">${item.explanation_text}</div>` : ''}
      ${shapRows ? `<div style="margin-top:var(--sp-4)"><div style="font-size:var(--fs-xs);color:var(--txt-muted);margin-bottom:var(--sp-2)">SIGNAL CONTRIBUTIONS</div>${shapRows}</div>` : ''}
    </div>`;
  }).join('');
}

// ── Audit trail ──────────────────────────────────────────────────────────
function renderAudit(data) {
  const el = document.getElementById('audit');
  if (!el) return;
  el.innerHTML = `
    <div><dt>Recommendation ID</dt><dd style="font-family:var(--font-mono);font-size:var(--fs-xs)">${data.recommendation_id.slice(0,8)}…</dd></div>
    <div><dt>Model version</dt><dd>${data.model_version}</dd></div>
    <div><dt>Risk mandate</dt><dd>${data.profile.risk_level}</dd></div>
    <div><dt>Data as of</dt><dd>${data.data_as_of}</dd></div>
    <div><dt>Pipeline nodes</dt><dd>${data.pipeline_stages?.length ?? '–'}</dd></div>
  `;
}

// ── Progress UI ───────────────────────────────────────────────────────────
function showProgress(stage, pct) {
  const wrap = document.getElementById('progress-wrap');
  if (wrap) wrap.classList.add('visible');
  const fill = document.getElementById('progress-fill');
  if (fill) fill.style.width = `${pct}%`;
  const lbl = document.getElementById('progress-label');
  if (lbl) lbl.textContent = stage;
}
function hideProgress() {
  const wrap = document.getElementById('progress-wrap');
  if (wrap) wrap.classList.remove('visible');
}

// ── Main generate handler ─────────────────────────────────────────────────
export async function handleGenerate() {
  const btn = document.getElementById('generate-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Running workflow…'; }
  store.set('recommendationLoading', true);
  store.set('recommendationError', null);

  const profile = buildProfile();
  store.update({ profile });

  const chanId = `reco_${Date.now()}`;
  const wsChannel = `recommendations/${chanId}`;

  try {
    await runWithProgress(wsChannel, profile, {
      onProgress: (stage, pct) => showProgress(stage, pct),
      onResult: (data) => {
        store.set('recommendation', data);
        store.set('recommendationLoading', false);
        hideProgress();
        renderTable(data.recommendations);
        renderMetrics(data.portfolio, data.market_regime);
        renderAllocation(data.recommendations);
        renderExplanations(data.recommendations);
        renderAudit(data);
        if (btn) { btn.disabled = false; btn.innerHTML = '<span>Generate recommendation</span><b>→</b>'; }
      },
      onError: (msg) => {
        store.set('recommendationError', msg);
        store.set('recommendationLoading', false);
        hideProgress();
        const tbody = document.getElementById('reco-tbody');
        if (tbody) tbody.innerHTML = `<tr class="empty-row"><td colspan="7">⚠ ${msg}</td></tr>`;
        if (btn) { btn.disabled = false; btn.innerHTML = '<span>Generate recommendation</span><b>→</b>'; }
      },
    });
  } catch (e) {
    // Fallback: REST
    try {
      const data = await api.recommend(profile);
      store.set('recommendation', data);
      renderTable(data.recommendations);
      renderMetrics(data.portfolio, data.market_regime);
      renderAllocation(data.recommendations);
      renderExplanations(data.recommendations);
      renderAudit(data);
    } catch (e2) {
      const tbody = document.getElementById('reco-tbody');
      if (tbody) tbody.innerHTML = `<tr class="empty-row"><td colspan="7">⚠ ${e2.message}</td></tr>`;
    } finally {
      hideProgress();
      if (btn) { btn.disabled = false; btn.innerHTML = '<span>Generate recommendation</span><b>→</b>'; }
      store.set('recommendationLoading', false);
    }
  }
}

// ── Sector checkboxes init ────────────────────────────────────────────────
export function initSectorChecks(sectors) {
  const container = document.getElementById('sector-checks');
  if (!container) return;
  container.innerHTML = sectors.map(s =>
    `<label class="check-item"><input type="checkbox" value="${s}"><span>${s}</span></label>`
  ).join('');
}
