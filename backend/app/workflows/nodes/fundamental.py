"""Node 3 — Fundamental Scoring Agent.

Computes quality, value, growth, profitability, and leverage scores
for each asset in the clean universe.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.domain.schemas import FundamentalScore, MarketSnapshot
from app.workflows.state import RecommendationState


def _cross_section_rank(values: list[float]) -> list[float]:
    """Return percentile rank in [0, 1] — avoids scale sensitivity."""
    arr = np.array(values, dtype=float)
    ranks = arr.argsort().argsort().astype(float)
    n = len(arr)
    return list(ranks / max(n - 1, 1))


def _score_asset(asset: MarketSnapshot, ranks: dict[str, float]) -> FundamentalScore:
    # Quality: ROE + low leverage + high liquidity
    quality = 0.4 * ranks["roe"] + 0.3 * (1 - ranks["debt_to_equity"]) + 0.3 * ranks["quality"]

    # Value: low P/E, low P/B → higher rank means cheaper
    pe_rank = 1 - ranks["pe_ratio"] if asset.pe_ratio > 0 else 0.5
    pb_rank = 1 - ranks["pb_ratio"] if asset.pb_ratio > 0 else 0.5
    value = 0.4 * pe_rank + 0.4 * pb_rank + 0.2 * ranks["value"]

    # Growth: revenue growth + expected return forecast
    growth = 0.5 * ranks["revenue_growth"] + 0.5 * ranks["expected_return"]

    # Profitability: ROE proxy + quality signal
    profitability = 0.6 * ranks["roe"] + 0.4 * ranks["quality"]

    # Leverage (lower is better → we flip rank)
    leverage = 1 - ranks["debt_to_equity"]

    composite = (
        0.28 * quality + 0.22 * value + 0.20 * growth
        + 0.18 * profitability + 0.12 * leverage
    )

    return FundamentalScore(
        symbol=asset.symbol,
        quality_score=round(quality, 4),
        value_score=round(value, 4),
        growth_score=round(growth, 4),
        profitability_score=round(profitability, 4),
        leverage_score=round(leverage, 4),
        composite=round(composite, 4),
    )


def fundamental_node(state: RecommendationState) -> dict[str, Any]:
    universe: list[MarketSnapshot] = state["universe"]
    n = len(universe)

    if n == 0:
        return {"fundamental_scores": [], "pipeline_stages": state.get("pipeline_stages", [])}

    # Build per-field cross-sectional ranks
    field_ranks = {
        field: _cross_section_rank([getattr(a, field) for a in universe])
        for field in ("quality", "value", "expected_return", "pe_ratio", "pb_ratio",
                      "roe", "debt_to_equity", "revenue_growth")
    }

    scores = [
        _score_asset(asset, {f: field_ranks[f][i] for f in field_ranks})
        for i, asset in enumerate(universe)
    ]

    stages: list[dict] = state.get("pipeline_stages", [])
    stages.append({
        "node": "fundamental",
        "assets_scored": n,
        "top_symbol": max(scores, key=lambda s: s.composite).symbol,
    })

    return {"fundamental_scores": scores, "pipeline_stages": stages}
