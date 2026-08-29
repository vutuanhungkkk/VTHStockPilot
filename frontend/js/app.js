/**
 * app.js — Main application entry point
 */
import { store } from './state.js';
import { api } from './api.js';
import { runWithProgress } from './websocket.js';
import { handleGenerate, initSectorChecks } from './recommendations.js';
import { handleBuildPortfolio } from './portfolio.js';
import { loadExperiments } from './experiments.js';

import { drawEquityCurve, drawDrawdown, drawRollingSharpe } from './charts.js';

// ── Navigation ────────────────────────────────────────────────────────────
function initNav() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(btn => {
    btn.addEventListener('click', () => {
      const view = btn.dataset.view;
      store.set('currentView', view);
      navItems.forEach(b => b.classList.toggle('active', b === btn));
      document.querySelectorAll('.view').forEach(v => {
        v.classList.toggle('active', v.id === `${view}-view`);
      });
      // Lazy load on first navigation
      if (view === 'experiments' && !store.get('experiments')) {
        loadExperiments();
      }

    });
  });
}

// ── Profile form sync ─────────────────────────────────────────────────────
function initProfileForm() {
  const weightSlider = document.getElementById('max-weight');
  const weightOutput = document.getElementById('weight-output');
  if (weightSlider && weightOutput) {
    weightSlider.addEventListener('input', () => {
      weightOutput.textContent = `${weightSlider.value}%`;
    });
  }
}

// ── Generate button ───────────────────────────────────────────────────────
function initGenerateBtn() {
  document.getElementById('generate-btn')?.addEventListener('click', handleGenerate);
}

// ── Portfolio buttons ─────────────────────────────────────────────────────
function initPortfolioButtons() {
  document.getElementById('build-portfolio-btn')?.addEventListener('click', handleBuildPortfolio);
}

// ── Backtest ──────────────────────────────────────────────────────────────
async function runBacktest() {
  const btn = document.getElementById('run-backtest-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Running…'; }
  store.set('backtestLoading', true);

  const profile = store.get('profile');
  const months = +document.getElementById('backtest-months')?.value || 24;
  const freq = document.getElementById('backtest-freq')?.value || 'monthly';
  const tcost = +document.getElementById('backtest-tcost')?.value || 10;

  const payload = {
    profile,
    months,
    rebalance_frequency: freq,
    transaction_cost_bps: tcost,
  };

  const chanId = `bt_${Date.now()}`;

  const setM = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  const showBtProgress = (stage, pct) => {
    const fill = document.getElementById('bt-progress-fill');
    if (fill) fill.style.width = `${pct}%`;
    const lbl = document.getElementById('bt-progress-label');
    if (lbl) lbl.textContent = stage;
    document.getElementById('bt-progress-wrap')?.classList.add('visible');
  };

  try {
    await runWithProgress(`backtest/${chanId}`, payload, {
      onProgress: showBtProgress,
      onResult: (data) => {
        store.set('backtest', data);
        store.set('backtestLoading', false);
        document.getElementById('bt-progress-wrap')?.classList.remove('visible');
        renderBacktestResults(data, setM);
        if (btn) { btn.disabled = false; btn.textContent = 'Run backtest'; }
      },
      onError: (msg) => {
        document.getElementById('bt-progress-wrap')?.classList.remove('visible');
        // Fallback to REST
        api.backtest(payload).then(data => {
          store.set('backtest', data);
          renderBacktestResults(data, setM);
        }).catch(console.error).finally(() => {
          store.set('backtestLoading', false);
          if (btn) { btn.disabled = false; btn.textContent = 'Run backtest'; }
        });
      },
    });
  } catch {
    // REST fallback
    try {
      const data = await api.backtest(payload);
      store.set('backtest', data);
      renderBacktestResults(data, setM);
    } finally {
      store.set('backtestLoading', false);
      if (btn) { btn.disabled = false; btn.textContent = 'Run backtest'; }
    }
  }
}

function renderBacktestResults(data, setM) {
  function pct(v) { return `${(v * 100).toFixed(1)}%`; }
  setM('bt-return', pct(data.annualized_return));
  setM('bt-benchmark', pct(data.benchmark_return));
  setM('bt-vol', pct(data.annualized_volatility));
  setM('bt-sharpe', data.sharpe_ratio.toFixed(2));
  setM('bt-sortino', data.sortino_ratio.toFixed(2));
  setM('bt-drawdown', pct(data.max_drawdown));
  setM('bt-hit', pct(data.hit_rate));
  setM('bt-calmar', data.calmar_ratio?.toFixed(2) ?? '--');
  setM('bt-ir', data.information_ratio?.toFixed(2) ?? '--');

  // Charts (defer to next tick so canvas sizes are resolved)
  requestAnimationFrame(() => {
    drawEquityCurve('equity-chart', data.equity_curve);
    if (data.drawdown_series?.length) drawDrawdown('drawdown-chart', data.drawdown_series);
    if (data.rolling_sharpe?.length) drawRollingSharpe('rolling-sharpe-chart', data.rolling_sharpe);
  });
}

// ── Startup ───────────────────────────────────────────────────────────────
async function init() {
  initNav();
  initProfileForm();
  initGenerateBtn();
  initPortfolioButtons();

  document.getElementById('run-backtest-btn')?.addEventListener('click', runBacktest);

  try {
    const [health, meta] = await Promise.all([api.health(), api.metadata()]);
    document.getElementById('model-version').textContent = health.version;
    document.getElementById('as-of').textContent = new Date().toLocaleDateString('vi-VN');
    store.set('metadata', meta);
    initSectorChecks(meta.sectors || []);
    document.getElementById('universe-size').textContent = meta.universe_size || '--';
  } catch (e) {
    console.warn('API health check failed:', e.message);
  }
}

document.addEventListener('DOMContentLoaded', init);
