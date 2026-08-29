"""LangGraph typed state for the recommendation pipeline."""
from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict

from app.domain.schemas import (
    ForecastScore,
    FundamentalScore,
    InvestorProfile,
    MarketRegime,
    MarketSnapshot,
    PortfolioMetrics,
    RankingScore,
    RecommendationExplanation,
)


class RecommendationState(TypedDict, total=False):
    """
    Immutable-by-convention typed dict flowing through each LangGraph node.
    Each node receives the full state and returns a partial dict of updated keys.
    No node may mutate the state dict in place.
    """
    # ── Inputs ─────────────────────────────────────────────────────────────
    profile: InvestorProfile
    raw_universe: list[MarketSnapshot]          # full universe from MarketDataService

    # ── Data Quality node output ────────────────────────────────────────────
    universe: list[MarketSnapshot]              # clean, non-stale assets
    data_quality_flags: list[str]               # warnings from DQ checks

    # ── Market Regime node output ───────────────────────────────────────────
    regime: MarketRegime
    regime_confidence: float
    regime_weights: dict[str, float]            # per-signal regime multipliers

    # ── Fundamental node output ─────────────────────────────────────────────
    fundamental_scores: list[FundamentalScore]

    # ── Forecast node output ────────────────────────────────────────────────
    forecast_scores: list[ForecastScore]

    # ── Customer Preference node output ─────────────────────────────────────
    preference_universe: list[MarketSnapshot]   # filtered by customer prefs
    preference_boosts: dict[str, float]         # symbol → boost delta

    # ── Risk & Compliance node output ───────────────────────────────────────
    eligible_universe: list[MarketSnapshot]     # post-risk-filter
    risk_flags: dict[str, list[str]]            # symbol → flag list

    # ── Ranking node output ─────────────────────────────────────────────────
    ranking_scores: list[RankingScore]
    selected: list[tuple[MarketSnapshot, RankingScore]]

    # ── Portfolio Optimization node output ──────────────────────────────────
    weights: list[float]
    portfolio_metrics: PortfolioMetrics

    # ── Explanation node output ─────────────────────────────────────────────
    explanations: list[RecommendationExplanation]

    # ── Pipeline audit trail ─────────────────────────────────────────────────
    pipeline_stages: list[dict[str, Any]]
    errors: list[str]
