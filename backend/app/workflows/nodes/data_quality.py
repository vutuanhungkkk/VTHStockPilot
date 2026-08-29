"""Node 1 — Data Quality Agent.

Checks:
- Missing / zero-price assets
- Stale data (data_age_hours > threshold)
- Outlier expected returns
- Insufficient liquidity
"""
from __future__ import annotations

import statistics
from typing import Any

from app.domain.schemas import MarketSnapshot
from app.workflows.state import RecommendationState

_STALE_HOURS = 24.0
_MIN_LIQUIDITY = 0.5
_MAX_RETURN_ZSCORE = 3.5


def data_quality_node(state: RecommendationState) -> dict[str, Any]:
    raw: list[MarketSnapshot] = state.get("raw_universe", [])
    flags: list[str] = []

    # --- basic validity ---
    valid = [x for x in raw if x.price > 0 and x.liquidity_score >= _MIN_LIQUIDITY]
    removed_basic = len(raw) - len(valid)
    if removed_basic:
        flags.append(f"{removed_basic} assets removed: zero price or low liquidity")

    # --- staleness ---
    fresh = [x for x in valid if not x.is_stale and x.data_age_hours <= _STALE_HOURS]
    stale_count = len(valid) - len(fresh)
    if stale_count:
        flags.append(f"{stale_count} assets removed: stale data (>{_STALE_HOURS}h)")

    # --- outlier expected returns (z-score) ---
    if len(fresh) >= 3:
        returns = [x.expected_return for x in fresh]
        mean_r = statistics.mean(returns)
        std_r = statistics.stdev(returns)
        if std_r > 0:
            clean = [x for x in fresh if abs((x.expected_return - mean_r) / std_r) <= _MAX_RETURN_ZSCORE]
            outlier_count = len(fresh) - len(clean)
            if outlier_count:
                flags.append(f"{outlier_count} assets removed: outlier expected return (>{_MAX_RETURN_ZSCORE}σ)")
        else:
            clean = fresh
    else:
        clean = fresh

    stages: list[dict] = state.get("pipeline_stages", [])
    stages.append({
        "node": "data_quality",
        "input_count": len(raw),
        "output_count": len(clean),
        "flags": flags,
    })

    return {
        "universe": clean,
        "data_quality_flags": flags,
        "pipeline_stages": stages,
    }
