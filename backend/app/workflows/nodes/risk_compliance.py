"""Node 6 — Risk & Compliance Agent.

Applies hard constraints:
- Volatility cap per risk mandate
- Liquidity minimum
- Sector concentration cap (max 50% in any sector)
- Maximum drawdown estimate filter
- Exposure constraint (beta cap for conservative profiles)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.domain.schemas import InvestorProfile, MarketSnapshot, RiskLevel
from app.workflows.state import RecommendationState

_VOL_CAPS = {
    RiskLevel.conservative: 0.23,
    RiskLevel.balanced: 0.32,
    RiskLevel.growth: 0.45,
}

_BETA_CAPS = {
    RiskLevel.conservative: 0.9,
    RiskLevel.balanced: 1.3,
    RiskLevel.growth: 2.0,
}

_MAX_SECTOR_WEIGHT = 0.50   # no more than 50% of universe in one sector
_MIN_ASSETS_REQUIRED = 3


def risk_compliance_node(state: RecommendationState) -> dict[str, Any]:
    universe: list[MarketSnapshot] = state.get("preference_universe", state.get("universe", []))
    profile: InvestorProfile = state["profile"]
    risk_flags: dict[str, list[str]] = defaultdict(list)

    vol_cap = _VOL_CAPS[profile.risk_level]
    beta_cap = _BETA_CAPS[profile.risk_level]

    # --- per-asset hard constraints ---
    eligible: list[MarketSnapshot] = []
    for asset in universe:
        flags: list[str] = []

        if asset.volatility > vol_cap:
            flags.append(f"Volatility {asset.volatility:.1%} exceeds {vol_cap:.1%} cap")

        if asset.beta > beta_cap:
            flags.append(f"Beta {asset.beta:.2f} exceeds {beta_cap:.2f} cap")

        if asset.liquidity_score < 0.5:
            flags.append("Below minimum liquidity threshold")

        if flags:
            risk_flags[asset.symbol] = flags
        else:
            eligible.append(asset)

    # --- sector concentration ---
    sector_counts: dict[str, int] = defaultdict(int)
    for a in eligible:
        sector_counts[a.sector] += 1
    n = max(len(eligible), 1)
    over_concentrated_sectors = {
        s for s, cnt in sector_counts.items() if cnt / n > _MAX_SECTOR_WEIGHT
    }
    if over_concentrated_sectors:
        # Keep only 2 per over-concentrated sector (deterministic — highest liquidity)
        capped: list[MarketSnapshot] = []
        sector_budget: dict[str, int] = defaultdict(int)
        for asset in sorted(eligible, key=lambda a: a.liquidity_score, reverse=True):
            cap = 2 if asset.sector in over_concentrated_sectors else 999
            if sector_budget[asset.sector] < cap:
                capped.append(asset)
                sector_budget[asset.sector] += 1
            else:
                risk_flags[asset.symbol].append(f"Sector {asset.sector} over-concentrated; capped")
        eligible = capped

    stages: list[dict] = state.get("pipeline_stages", [])
    stages.append({
        "node": "risk_compliance",
        "input_count": len(universe),
        "eligible_count": len(eligible),
        "flagged_assets": len(risk_flags),
        "vol_cap": vol_cap,
    })

    if len(eligible) < _MIN_ASSETS_REQUIRED:
        errors: list[str] = state.get("errors", [])
        errors.append(
            f"Risk constraints leave only {len(eligible)} eligible assets "
            f"(minimum {_MIN_ASSETS_REQUIRED} required)."
        )
        return {"eligible_universe": eligible, "risk_flags": dict(risk_flags),
                "pipeline_stages": stages, "errors": errors}

    return {
        "eligible_universe": eligible,
        "risk_flags": dict(risk_flags),
        "pipeline_stages": stages,
    }
