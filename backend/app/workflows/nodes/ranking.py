"""Node 7 — Ranking Agent.

Combines signals from all upstream nodes into a versioned recommendation score.
Score version is embedded so champion/challenger models can be tracked in MLflow.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.domain.schemas import (
    ForecastScore,
    FundamentalScore,
    InvestorProfile,
    MarketSnapshot,
    RankingScore,
    RiskLevel,
)
from app.workflows.state import RecommendationState

SCORE_VERSION = "v2.0.0-langgraph"

# Base signal weights
_BASE_WEIGHTS = {
    "forecast_excess_return": 0.25,
    "fundamental_composite": 0.20,
    "momentum": 0.18,
    "quality": 0.12,
    "value": 0.10,
    "sentiment": 0.08,
    "liquidity": 0.07,
}

# Risk-level volatility penalty
_RISK_PENALTIES = {
    RiskLevel.conservative: 0.35,
    RiskLevel.balanced: 0.22,
    RiskLevel.growth: 0.10,
}


def _rank_score(
    asset: MarketSnapshot,
    fund: FundamentalScore,
    forecast: ForecastScore,
    regime_weights: dict[str, float],
    preference_boost: float,
    risk_penalty: float,
) -> tuple[float, dict[str, float]]:
    """Return (final_score, signal_contributions)."""
    contributions: dict[str, float] = {
        "forecast_excess_return": _BASE_WEIGHTS["forecast_excess_return"]
            * regime_weights.get("forecast", 1.0)
            * forecast.expected_excess_return * 5,  # scale to [0,1] range
        "fundamental_composite": _BASE_WEIGHTS["fundamental_composite"]
            * fund.composite,
        "momentum": _BASE_WEIGHTS["momentum"]
            * regime_weights.get("momentum", 1.0)
            * asset.momentum,
        "quality": _BASE_WEIGHTS["quality"] * asset.quality,
        "value": _BASE_WEIGHTS["value"]
            * regime_weights.get("value", 1.0)
            * asset.value,
        "sentiment": _BASE_WEIGHTS["sentiment"]
            * regime_weights.get("sentiment", 1.0)
            * asset.sentiment,
        "liquidity": _BASE_WEIGHTS["liquidity"] * asset.liquidity_score,
        "volatility_penalty": -risk_penalty * asset.volatility,
        "preference_boost": preference_boost,
    }
    score = sum(contributions.values())
    return score, {k: round(v, 5) for k, v in contributions.items()}


def ranking_node(state: RecommendationState) -> dict[str, Any]:
    eligible: list[MarketSnapshot] = state.get("eligible_universe", [])
    profile: InvestorProfile = state["profile"]
    fundamentals: list[FundamentalScore] = state.get("fundamental_scores", [])
    forecasts: list[ForecastScore] = state.get("forecast_scores", [])
    regime_weights: dict[str, float] = state.get("regime_weights", {})
    preference_boosts: dict[str, float] = state.get("preference_boosts", {})

    fund_map = {f.symbol: f for f in fundamentals}
    forecast_map = {f.symbol: f for f in forecasts}
    risk_penalty = _RISK_PENALTIES[profile.risk_level]

    _default_fund = FundamentalScore(
        symbol="", quality_score=0.5, value_score=0.5,
        growth_score=0.5, profitability_score=0.5, leverage_score=0.5, composite=0.5,
    )
    _default_fc = ForecastScore(
        symbol="", expected_excess_return=0.0, return_volatility=0.2,
        outperform_probability=0.5, confidence=0.5,
    )

    ranked: list[RankingScore] = []
    for asset in eligible:
        fund = fund_map.get(asset.symbol, _default_fund)
        fc = forecast_map.get(asset.symbol, _default_fc)
        boost = preference_boosts.get(asset.symbol, 0.0)
        score, contribs = _rank_score(asset, fund, fc, regime_weights, boost, risk_penalty)
        ranked.append(RankingScore(
            symbol=asset.symbol,
            final_score=round(score, 5),
            score_version=SCORE_VERSION,
            signal_contributions=contribs,
        ))

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    n_select = profile.max_positions
    selected_scores = ranked[:n_select]
    selected_assets = [next(a for a in eligible if a.symbol == r.symbol) for r in selected_scores]
    selected = list(zip(selected_assets, selected_scores))

    stages: list[dict] = state.get("pipeline_stages", [])
    stages.append({
        "node": "ranking",
        "score_version": SCORE_VERSION,
        "ranked_total": len(ranked),
        "selected": [r.symbol for r in selected_scores],
    })

    return {
        "ranking_scores": ranked,
        "selected": selected,
        "pipeline_stages": stages,
    }
