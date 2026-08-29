"""Node 8 — Portfolio Optimization Agent.

Optimizes portfolio weights using utility-based approach
with optional mean-variance adjustment.

Strategies:
- utility_based: expected_return / variance (fast, intuitive)
- risk_parity: equal risk contribution
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from app.domain.schemas import InvestorProfile, MarketSnapshot, PortfolioMetrics, RankingScore
from app.workflows.state import RecommendationState

_RISK_FREE_RATE = 0.035  # 3.5% annualised
_CORR_ASSUMPTION = 0.25  # conservative cross-asset correlation


def _cap_redistribute(weights: np.ndarray, cap: float) -> np.ndarray:
    weights = weights.copy()
    for _ in range(30):
        over = weights > cap
        if not over.any():
            break
        excess = float((weights[over] - cap).sum())
        weights[over] = cap
        under = ~over
        if not under.any():
            break
        weights[under] += excess * weights[under] / weights[under].sum()
    total = weights.sum()
    return weights / total if total > 0 else weights


def _utility_weights(assets: list[MarketSnapshot], cap: float) -> np.ndarray:
    utility = np.array([
        max(a.expected_return, 0.01) / max(a.volatility ** 2, 0.0001)
        for a in assets
    ])
    w = utility / utility.sum()
    return _cap_redistribute(w, cap)


def _risk_parity_weights(assets: list[MarketSnapshot], cap: float) -> np.ndarray:
    inv_vol = np.array([1.0 / max(a.volatility, 0.01) for a in assets])
    w = inv_vol / inv_vol.sum()
    return _cap_redistribute(w, cap)


def _compute_metrics(
    weights: np.ndarray,
    assets: list[MarketSnapshot],
    profile: InvestorProfile,
) -> PortfolioMetrics:
    er = float(np.dot(weights, [a.expected_return for a in assets]))
    variance = float(np.sum([(w * a.volatility) ** 2 for w, a in zip(weights, assets)]))
    covariance = _CORR_ASSUMPTION * float(sum(
        2 * weights[i] * weights[j] * assets[i].volatility * assets[j].volatility
        for i in range(len(assets)) for j in range(i + 1, len(assets))
    ))
    vol = math.sqrt(max(variance + covariance, 1e-8))
    sharpe = (er - _RISK_FREE_RATE) / vol
    sortino_vol = float(np.sqrt(np.mean([
        (w * a.volatility * 0.7) ** 2 for w, a in zip(weights, assets)
    ])))
    sortino = (er - _RISK_FREE_RATE) / max(sortino_vol, 0.001)
    hhi = float(np.sum(weights ** 2))
    eff_n = round(1.0 / hhi, 2) if hhi > 0 else 0.0
    max_dd = round(-1.65 * vol * math.sqrt(profile.horizon_months / 12), 4)

    sector_w: dict[str, float] = defaultdict(float)
    for w, a in zip(weights, assets):
        sector_w[a.sector] += float(w)

    return PortfolioMetrics(
        expected_return=round(er, 4),
        expected_volatility=round(vol, 4),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        diversification_score=round(1 - hhi, 4),
        max_drawdown_estimate=max_dd,
        sector_concentration=dict(sector_w),
        effective_n=eff_n,
    )


def portfolio_optimizer_node(state: RecommendationState) -> dict[str, Any]:
    selected: list[tuple[MarketSnapshot, RankingScore]] = state.get("selected", [])
    profile: InvestorProfile = state["profile"]

    if not selected:
        return {"weights": [], "portfolio_metrics": None, "pipeline_stages": state.get("pipeline_stages", [])}

    assets = [s[0] for s in selected]

    # Choose optimisation strategy by risk level
    if profile.risk_level.value == "growth":
        weights = _utility_weights(assets, profile.max_position_weight)
        strategy = "utility_based"
    else:
        weights = _risk_parity_weights(assets, profile.max_position_weight)
        strategy = "risk_parity"

    metrics = _compute_metrics(weights, assets, profile)

    stages: list[dict] = state.get("pipeline_stages", [])
    stages.append({
        "node": "portfolio_optimizer",
        "strategy": strategy,
        "expected_return": metrics.expected_return,
        "expected_volatility": metrics.expected_volatility,
        "sharpe_ratio": metrics.sharpe_ratio,
    })

    return {
        "weights": [round(float(w), 6) for w in weights],
        "portfolio_metrics": metrics,
        "pipeline_stages": stages,
    }
