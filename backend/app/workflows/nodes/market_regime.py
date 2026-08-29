"""Node 2 — Market Regime Agent.

Classifies current market regime using:
- Cross-sectional average momentum (trend signal)
- Cross-sectional volatility spread (risk-on/risk-off)
- Average sentiment

Outputs: regime label + per-signal regime multipliers.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.domain.schemas import MarketRegime, MarketSnapshot
from app.workflows.state import RecommendationState


def _classify(mean_momentum: float, mean_vol: float, mean_sentiment: float) -> tuple[MarketRegime, float]:
    """Return (regime, confidence) using threshold rules."""
    # Bull: high momentum + moderate vol + positive sentiment
    if mean_momentum >= 0.65 and mean_vol < 0.28 and mean_sentiment >= 0.55:
        return MarketRegime.bull, round(min(1.0, mean_momentum * 1.1), 3)

    # Bear: low momentum + negative sentiment
    if mean_momentum < 0.45 and mean_sentiment < 0.45:
        return MarketRegime.bear, round(min(1.0, (1 - mean_momentum) * 1.1), 3)

    # High-volatility: vol spike regardless of direction
    if mean_vol >= 0.30:
        return MarketRegime.high_volatility, round(min(1.0, mean_vol / 0.35), 3)

    return MarketRegime.neutral, 0.6


def _regime_weights(regime: MarketRegime) -> dict[str, float]:
    """Per-signal multipliers adjusted for regime."""
    base = {"momentum": 1.0, "quality": 1.0, "value": 1.0, "sentiment": 1.0, "forecast": 1.0}
    if regime == MarketRegime.bull:
        base.update({"momentum": 1.25, "sentiment": 1.15})
    elif regime == MarketRegime.bear:
        base.update({"quality": 1.30, "value": 1.20, "momentum": 0.70})
    elif regime == MarketRegime.high_volatility:
        base.update({"quality": 1.40, "momentum": 0.60, "sentiment": 0.80})
    return base


def market_regime_node(state: RecommendationState) -> dict[str, Any]:
    universe: list[MarketSnapshot] = state["universe"]

    momenta = np.array([x.momentum for x in universe])
    vols = np.array([x.volatility for x in universe])
    sentiments = np.array([x.sentiment for x in universe])

    mean_mom = float(np.mean(momenta))
    mean_vol = float(np.mean(vols))
    mean_sent = float(np.mean(sentiments))

    regime, confidence = _classify(mean_mom, mean_vol, mean_sent)
    weights = _regime_weights(regime)

    stages: list[dict] = state.get("pipeline_stages", [])
    stages.append({
        "node": "market_regime",
        "regime": regime.value,
        "confidence": confidence,
        "mean_momentum": round(mean_mom, 3),
        "mean_volatility": round(mean_vol, 3),
    })

    return {
        "regime": regime,
        "regime_confidence": confidence,
        "regime_weights": weights,
        "pipeline_stages": stages,
    }
