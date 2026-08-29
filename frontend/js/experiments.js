/**
 * experiments.js — Model performance + Data freshness views
 */
import { store } from './state.js';
import { api } from './api.js';
import { drawMetricBars } from './charts.js';

function fmt(v, d = 3) { return typeof v === 'number' ? v.toFixed(d) : '--'; }
function pct(v) { return typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '--'; }

export async function loadExperiments() {
  store.set('experimentsLoading', true);
  try {
    const [data, freshness] = await Promise.all([
      api.experiments(),
      api.freshness(),
    ]);
    store.set('experiments', data);
    renderExperiments(data);
    renderFreshness(freshness);
  } catch (e) {
    console.error('Failed to load experiments:', e);
  } finally {
    store.set('experimentsLoading', false);
  }
}

function renderExperiments(data) {
  // Champion card
  renderModelCard('champion-card', data.champion, true);
  renderModelCard('challenger-card', data.challenger, false);

  // Experiments table
  const tbody = document.getElementById('experiments-tbody');
  if (!tbody) return;
  tbody.innerHTML = data.experiments.map(exp => {
    const m = exp.metrics;
    return `<tr>
      <td style="font-family:var(--font-mono);font-size:var(--fs-xs)">${exp.run_id.slice(0, 8)}</td>
      <td><span class="stage-badge ${exp.stage}">${exp.stage}</span></td>
      <td>${exp.parameters.model || '--'}</td>
      <td style="font-family:var(--font-mono)">${fmt(m.mean_validation_ndcg)}</td>
      <td style="font-family:var(--font-mono)">${pct(m.precision_at_5)}</td>
      <td style="font-family:var(--font-mono)">${fmt(m.rank_ic)}</td>
      <td style="font-family:var(--font-mono)">${fmt(m.sharpe_ratio)}</td>
      <td style="font-family:var(--font-mono);color:var(--clr-negative)">${pct(m.max_drawdown)}</td>
      <td style="font-size:var(--fs-xs);color:var(--txt-muted)">${exp.created_at.slice(0, 10)}</td>
    </tr>`;
  }).join('');

  // Metric comparison chart
  if (data.champion && data.challenger) {
    const metrics = ['ndcg', 'precision_at_5', 'hit_rate', 'sharpe_ratio'];
    const labels = ['NDCG', 'Prec@5', 'Hit Rate', 'Sharpe'];
    setTimeout(() => {
      const champVals = metrics.map(m =>
        data.champion.metrics[`mean_validation_${m}`] || data.champion.metrics[m] || 0
      );
      drawMetricBars('champion-metric-chart', labels, champVals, '#3b82f6');
      const chalVals = metrics.map(m =>
        data.challenger.metrics[`mean_validation_${m}`] || data.challenger.metrics[m] || 0
      );
      drawMetricBars('challenger-metric-chart', labels, chalVals, '#8b5cf6');
    }, 50);
  }
}

function renderModelCard(id, model, isChampion) {
  const el = document.getElementById(id);
  if (!el || !model) return;
  const m = model.metrics;
  el.innerHTML = `
    <div class="card-header">
      <div>
        <div class="card-eyebrow">${isChampion ? 'CHAMPION' : 'CHALLENGER'}</div>
        <h3>${model.parameters.model || model.parameters.challenger_model || 'Model'}</h3>
      </div>
      <span class="stage-badge ${model.stage}">${model.stage}</span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:var(--sp-3);margin-bottom:var(--sp-4)">
      ${[
        ['NDCG@K', fmt(m.mean_validation_ndcg || m.lgb_ndcg)],
        ['Precision@5', pct(m.precision_at_5 || m.lgb_precision_at_k)],
        ['Rank IC', fmt(m.rank_ic || m.lgb_rank_ic)],
        ['Sharpe', fmt(m.sharpe_ratio)],
        ['Max DD', pct(m.max_drawdown)],
        ['Hit Rate', pct(m.hit_rate)],
      ].map(([label, val]) => `
        <div style="background:var(--clr-bg-700);padding:var(--sp-3);border-radius:var(--r-sm)">
          <div style="font-size:var(--fs-xs);color:var(--txt-muted);text-transform:uppercase;letter-spacing:0.8px">${label}</div>
          <div style="font-family:var(--font-mono);font-weight:600;margin-top:4px;color:var(--txt-primary)">${val}</div>
        </div>`).join('')}
    </div>
    <canvas id="${id.replace('-card', '')}-metric-chart" style="width:100%;height:120px"></canvas>
    <div style="font-size:var(--fs-xs);color:var(--txt-muted);margin-top:var(--sp-3)">
      Run ID: <span style="font-family:var(--font-mono)">${model.run_id.slice(0, 16)}…</span> · ${model.created_at.slice(0, 10)}
    </div>
  `;
}

function renderFreshness(data) {
  const container = document.getElementById('freshness-panel');
  if (!container) return;
  const driftOk = !data.feature_drift_detected && !data.prediction_drift_detected;
  container.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:var(--sp-4)">
      <div class="metric-tile">
        <div class="metric-label">Model Version</div>
        <div class="metric-value" style="font-size:var(--fs-md)">${data.model_version}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Last Trained</div>
        <div class="metric-value" style="font-size:var(--fs-md)">${data.model_last_trained?.slice(0, 10) ?? '--'}</div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">ETL Status</div>
        <div class="metric-value ${data.etl_status === 'success' ? 'positive' : 'negative'}" style="font-size:var(--fs-lg)">
          ${data.etl_status === 'success' ? '✓' : '✗'} ${data.etl_status}
        </div>
      </div>
      <div class="metric-tile">
        <div class="metric-label">Drift Status</div>
        <div class="metric-value ${driftOk ? 'positive' : 'negative'}" style="font-size:var(--fs-lg)">
          ${driftOk ? '✓ Clean' : '⚠ Drift'}
        </div>
        <div class="metric-sublabel">
          Feature: ${data.feature_drift_detected ? '⚠' : '✓'} ·
          Prediction: ${data.prediction_drift_detected ? '⚠' : '✓'}
        </div>
      </div>
    </div>
  `;
}
