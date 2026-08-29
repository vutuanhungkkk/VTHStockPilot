/**
 * api.js — REST client for VTH-StockPilot API
 */

const API_BASE = '/api/v1';

async function _fetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  health: () => _fetch('/health'),
  metadata: () => _fetch('/metadata'),

  // Recommendations
  recommend: (profile) =>
    _fetch('/recommendations', { method: 'POST', body: JSON.stringify(profile) }),

  // Portfolio
  buildPortfolio: (profile) =>
    _fetch('/portfolio/build', { method: 'POST', body: JSON.stringify(profile) }),

  riskReport: (profile) =>
    _fetch('/portfolio/risk-report', { method: 'POST', body: JSON.stringify(profile) }),

  sectors: () => _fetch('/portfolio/sectors'),

  // Backtest
  backtest: (request) =>
    _fetch('/backtests', { method: 'POST', body: JSON.stringify(request) }),

  // Experiments
  experiments: () => _fetch('/experiments'),
  freshness: () => _fetch('/experiments/freshness'),
  metricDefinitions: () => _fetch('/experiments/metrics/definitions'),


};
