"""LangGraph recommendation workflow.

Assembles a deterministic StateGraph with 9 specialist nodes.
Each node receives the full RecommendationState and returns a partial dict
of updated keys — LangGraph merges these automatically.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

try:
    from langgraph.graph import StateGraph, END
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False

from app.domain.schemas import (
    InvestorProfile,
    MarketSnapshot,
    RecommendationExplanation,
    RecommendationItem,
    RecommendationResponse,
    PortfolioMetrics,
)
from app.workflows.state import RecommendationState
from app.workflows.nodes.data_quality import data_quality_node
from app.workflows.nodes.market_regime import market_regime_node
from app.workflows.nodes.fundamental import fundamental_node
from app.workflows.nodes.forecast import forecast_node
from app.workflows.nodes.customer_preference import customer_preference_node
from app.workflows.nodes.risk_compliance import risk_compliance_node
from app.workflows.nodes.ranking import ranking_node
from app.workflows.nodes.portfolio_optimizer import portfolio_optimizer_node
from app.workflows.nodes.explanation import explanation_node

# Ordered node sequence (deterministic — no conditional branching)
_NODES = [
    ("data_quality", data_quality_node),
    ("market_regime", market_regime_node),
    ("fundamental", fundamental_node),
    ("forecast", forecast_node),
    ("customer_preference", customer_preference_node),
    ("risk_compliance", risk_compliance_node),
    ("ranking", ranking_node),
    ("portfolio_optimizer", portfolio_optimizer_node),
    ("explanation", explanation_node),
]


def _build_langgraph() -> Any:
    """Build a real LangGraph StateGraph."""
    graph = StateGraph(RecommendationState)
    node_names = [name for name, _ in _NODES]
    for name, fn in _NODES:
        graph.add_node(name, fn)
    graph.set_entry_point(node_names[0])
    for i in range(len(node_names) - 1):
        graph.add_edge(node_names[i], node_names[i + 1])
    graph.add_edge(node_names[-1], END)
    return graph.compile()


def _build_fallback():
    """Pure-Python deterministic fallback (no langgraph package needed)."""
    async def run(state: RecommendationState) -> RecommendationState:
        for _, fn in _NODES:
            partial = fn(state)
            state = {**state, **partial}
        return state
    return run


class RecommendationGraph:
    """Facade wrapping the LangGraph workflow."""

    def __init__(self) -> None:
        if _LANGGRAPH_AVAILABLE:
            self._graph = _build_langgraph()
            self._mode = "langgraph"
        else:
            self._graph = _build_fallback()
            self._mode = "fallback"

    @property
    def mode(self) -> str:
        return self._mode

    async def run(
        self,
        profile: InvestorProfile,
        universe: list[MarketSnapshot],
        progress_cb=None,
    ) -> RecommendationState:
        initial: RecommendationState = {
            "profile": profile,
            "raw_universe": universe,
            "pipeline_stages": [],
            "errors": [],
        }
        if _LANGGRAPH_AVAILABLE:
            result = await self._graph.ainvoke(initial)
        else:
            result = await self._graph(initial)
        return result


# ── Response assembly ─────────────────────────────────────────────────────────

def build_response(
    state: RecommendationState,
    profile: InvestorProfile,
    model_version: str,
    data_as_of: str,
    recommendation_id: str,
) -> RecommendationResponse:
    """Convert final state into a RecommendationResponse."""
    import uuid
    from app.core.config import get_settings

    selected = state.get("selected", [])
    weights: list[float] = state.get("weights", [])
    explanations: list[RecommendationExplanation] = state.get("explanations", [])
    expl_map = {e.symbol: e for e in explanations}
    fund_map = {f.symbol: f for f in state.get("fundamental_scores", [])}
    fc_map = {f.symbol: f for f in state.get("forecast_scores", [])}

    items: list[RecommendationItem] = []
    for idx, ((asset, ranking), weight) in enumerate(zip(selected, weights), start=1):
        expl = expl_map.get(asset.symbol)
        fund = fund_map.get(asset.symbol)
        fc = fc_map.get(asset.symbol)
        items.append(RecommendationItem(
            rank=idx,
            symbol=asset.symbol,
            company=asset.company,
            sector=asset.sector,
            price=asset.price,
            score=ranking.final_score,
            expected_return=asset.expected_return,
            volatility=asset.volatility,
            confidence=round(fc.confidence if fc else 0.5 + 0.3 * asset.quality, 3),
            weight=round(weight, 4),
            allocation=round(profile.capital * weight, 0),
            signals={
                "momentum": asset.momentum,
                "quality": asset.quality,
                "value": asset.value,
                "sentiment": asset.sentiment,
                "liquidity": asset.liquidity_score,
            },
            fundamental={
                "quality_score": fund.quality_score if fund else 0.0,
                "value_score": fund.value_score if fund else 0.0,
                "growth_score": fund.growth_score if fund else 0.0,
                "composite": fund.composite if fund else 0.0,
            },
            forecast={
                "excess_return": fc.expected_excess_return if fc else 0.0,
                "outperform_prob": fc.outperform_probability if fc else 0.5,
                "confidence": fc.confidence if fc else 0.5,
            },
            reasons=expl.key_drivers if expl else [],
            risk_flags=expl.risk_flags if expl else [],
            shap_contributions=expl.shap_contributions if expl else {},
            explanation_text=expl.summary if expl else "",
        ))

    regime = state.get("regime")

    return RecommendationResponse(
        recommendation_id=recommendation_id,
        market_regime=regime.value if regime else "neutral",
        model_version=model_version,
        data_as_of=data_as_of,
        profile=profile,
        recommendations=items,
        portfolio=state.get("portfolio_metrics") or PortfolioMetrics(
            expected_return=0, expected_volatility=0, sharpe_ratio=0,
            diversification_score=0,
        ),
        pipeline_stages=state.get("pipeline_stages", []),
    )
