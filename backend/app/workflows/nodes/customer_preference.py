"""Node 5 — Customer Preference Agent.

Applies investor profile preferences to adjust the scoring universe:
- Boost for preferred sectors and watchlist stocks
- Soft-exclude (not hard-exclude — that's Risk node) low-preference sectors
- ESG filter (placeholder — flags assets with high leverage as ESG risk)
- Feedback history: previously liked stocks get a boost
"""
from __future__ import annotations

from typing import Any

from app.domain.schemas import InvestorProfile, MarketSnapshot
from app.workflows.state import RecommendationState

_PREFERRED_SECTOR_BOOST = 0.06

_ESG_LEVERAGE_THRESHOLD = 1.5       # debt_to_equity ratio


def customer_preference_node(state: RecommendationState) -> dict[str, Any]:
    universe: list[MarketSnapshot] = state["universe"]
    profile: InvestorProfile = state["profile"]

    preference_boosts: dict[str, float] = {}

    for asset in universe:
        boost = 0.0

        # Preferred sector
        if asset.sector in profile.preferred_sectors:
            boost += _PREFERRED_SECTOR_BOOST



        # ESG filter — flag high-leverage assets (boost = negative)
        if profile.esg_filter and asset.debt_to_equity > _ESG_LEVERAGE_THRESHOLD:
            boost -= 0.05

        if boost != 0.0:
            preference_boosts[asset.symbol] = round(boost, 4)

    # Filtered universe respects excluded sectors (soft-pass through for Risk node)
    preference_universe = [a for a in universe if a.sector not in profile.excluded_sectors]

    stages: list[dict] = state.get("pipeline_stages", [])
    stages.append({
        "node": "customer_preference",
        "boosted_assets": len(preference_boosts),
        "excluded_sectors": list(profile.excluded_sectors),
        "esg_filter_active": profile.esg_filter,
        "universe_after_exclusion": len(preference_universe),
    })

    return {
        "preference_universe": preference_universe,
        "preference_boosts": preference_boosts,
        "pipeline_stages": stages,
    }
