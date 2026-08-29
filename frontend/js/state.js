/**
 * state.js — Observable application state (tiny observer pattern)
 */

function createStore(initial) {
  let state = { ...initial };
  const listeners = new Map();

  return {
    get: (key) => state[key],
    getAll: () => ({ ...state }),

    set(key, value) {
      state = { ...state, [key]: value };
      (listeners.get(key) || []).forEach((fn) => fn(value, state));
      (listeners.get('*') || []).forEach((fn) => fn(state));
    },

    update(partial) {
      Object.entries(partial).forEach(([k, v]) => this.set(k, v));
    },

    on(key, fn) {
      if (!listeners.has(key)) listeners.set(key, []);
      listeners.get(key).push(fn);
      return () => {
        const arr = listeners.get(key) || [];
        const idx = arr.indexOf(fn);
        if (idx >= 0) arr.splice(idx, 1);
      };
    },
  };
}

export const store = createStore({
  // App
  currentView: 'recommendations',
  metadata: null,

  // Recommendation
  recommendation: null,
  recommendationLoading: false,
  recommendationError: null,

  // Portfolio
  riskReport: null,
  riskReportLoading: false,

  // Backtest
  backtest: null,
  backtestLoading: false,
  backtestMonths: 24,
  rebalanceFrequency: 'monthly',

  // Experiments
  experiments: null,
  experimentsLoading: false,

  // Profile form
  profile: {
    capital: 100_000_000,
    risk_level: 'balanced',
    horizon_months: 12,
    max_positions: 5,
    max_position_weight: 0.3,
    preferred_sectors: [],
    excluded_sectors: [],
    esg_filter: false,
  },
});
