"""Node 9 — Explanation Agent.

Generates structured explanations from:
- Signal contributions (from Ranking node)
- SHAP values (computed here if model is available, else approximated)
- Risk flags (from Risk node)
- Regime context
- Optional LLM-based summary (template provider by default)
"""
from __future__ import annotations

from typing import Any

from app.domain.schemas import (
    MarketSnapshot,
    RankingScore,
    RecommendationExplanation,
)
from app.workflows.state import RecommendationState

_SIGNAL_LABELS = {
    "forecast_excess_return": "Positive excess return forecast",
    "fundamental_composite": "Strong fundamental quality",
    "momentum": "Upward price momentum",
    "quality": "High business quality score",
    "value": "Attractive valuation",
    "sentiment": "Positive market sentiment",
    "liquidity": "High market liquidity",
    "volatility_penalty": "Volatility adjustment",
    "preference_boost": "Investor preference match",
}


def _top_drivers(contributions: dict[str, float], n: int = 3) -> list[str]:
    """Return top N positive contributing signals as readable labels."""
    positive = {k: v for k, v in contributions.items() if v > 0 and k != "volatility_penalty"}
    top = sorted(positive, key=lambda k: positive[k], reverse=True)[:n]
    return [_SIGNAL_LABELS.get(k, k) for k in top]


def _approximate_shap(contributions: dict[str, float]) -> dict[str, float]:
    """Approximate SHAP values from signal contributions (normalised sum=1)."""
    total = sum(abs(v) for v in contributions.values())
    if total == 0:
        return {}
    return {k: round(v / total, 4) for k, v in contributions.items()}


def _build_explanation_text(
    asset: MarketSnapshot,
    drivers: list[str],
    risk_flags: list[str],
    regime: str,
) -> str:
    """Template-based explanation (no LLM call)."""
    drivers_str = "; ".join(drivers) if drivers else "balanced signal profile"
    regime_ctx = f"in a {regime} market regime"
    flags_str = (
        f" Risk considerations: {'; '.join(risk_flags)}."
        if risk_flags else ""
    )
    return (
        f"{asset.company} ({asset.symbol}) is ranked {regime_ctx} driven by: {drivers_str}."
        f"{flags_str}"
    )


def explanation_node(state: RecommendationState) -> dict[str, Any]:
    selected: list[tuple[MarketSnapshot, RankingScore]] = state.get("selected", [])
    risk_flags: dict[str, list[str]] = state.get("risk_flags", {})
    regime = state.get("regime")
    regime_str = regime.value if regime else "neutral"

    explanations: list[RecommendationExplanation] = []
    for asset, ranking in selected:
        contributions = ranking.signal_contributions
        drivers = _top_drivers(contributions)
        shap_approx = _approximate_shap(contributions)
        flags = risk_flags.get(asset.symbol, [])

        # Elevate any lingering volatility warning
        if asset.volatility > 0.30:
            flags = list(flags)
            flags.append(f"Elevated volatility ({asset.volatility:.1%})")
        if asset.value < 0.45:
            flags = list(flags)
            flags.append("Premium valuation — limited margin of safety")

        text = _build_explanation_text(asset, drivers, flags, regime_str)

        explanations.append(RecommendationExplanation(
            symbol=asset.symbol,
            summary=text,
            key_drivers=drivers,
            risk_flags=flags,
            shap_contributions=shap_approx,
        ))

    stages: list[dict] = state.get("pipeline_stages", [])
    stages.append({
        "node": "explanation",
        "assets_explained": len(explanations),
        "provider": "template",
    })

    return {"explanations": explanations, "pipeline_stages": stages}
