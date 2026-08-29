"""Portfolio API router."""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException

from app.domain.schemas import (
    InvestorProfile,
    PortfolioPosition,
    PortfolioRiskReport,
    RecommendationResponse,
)
from app.services.market_data import MarketDataService
from app.workflows.recommendation_graph import RecommendationGraph, build_response
from app.core.config import get_settings

import math
import numpy as np

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])
_graph = RecommendationGraph()
_market = MarketDataService()


@router.post("/build", response_model=RecommendationResponse)
async def build_portfolio(profile: InvestorProfile) -> RecommendationResponse:
    """Build a portfolio from an investor profile — alias for recommendation with full output."""
    settings = get_settings()
    try:
        universe = _market.get_universe()
        state = await _graph.run(profile, universe)
        errors = state.get("errors", [])
        if errors:
            raise ValueError(errors[0])
        return build_response(state, profile, settings.model_version, _market.data_as_of(), str(uuid.uuid4()))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/risk-report", response_model=PortfolioRiskReport)
async def portfolio_risk_report(profile: InvestorProfile) -> PortfolioRiskReport:
    """Compute detailed risk analytics for the optimal portfolio."""
    settings = get_settings()
    try:
        universe = _market.get_universe()
        state = await _graph.run(profile, universe)
        errors = state.get("errors", [])
        if errors:
            raise ValueError(errors[0])

        selected = state.get("selected", [])
        weights: list[float] = state.get("weights", [])
        metrics = state.get("portfolio_metrics")

        if not selected or not weights or not metrics:
            raise ValueError("Portfolio optimization produced no result.")

        assets = [s[0] for s in selected]
        n = len(assets)

        # --- VaR 95% (parametric) ---
        port_vol = metrics.expected_volatility
        var_95 = round(-1.645 * port_vol * math.sqrt(1 / 12), 4)
        cvar_95 = round(-2.063 * port_vol * math.sqrt(1 / 12), 4)

        # --- Correlation matrix (approx: beta-based) ---
        betas = np.array([a.beta for a in assets])
        corr = np.outer(betas, betas)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        positions = [
            PortfolioPosition(
                symbol=a.symbol, company=a.company, sector=a.sector,
                weight=round(w, 4), allocation=round(profile.capital * w, 0),
                expected_return=a.expected_return, volatility=a.volatility, beta=a.beta,
            )
            for a, w in zip(assets, weights)
        ]

        return PortfolioRiskReport(
            portfolio_id=str(uuid.uuid4()),
            metrics=metrics,
            positions=positions,
            var_95=var_95,
            cvar_95=cvar_95,
            correlation_matrix=corr.tolist(),
            symbols=[a.symbol for a in assets],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/sectors")
async def sector_universe() -> dict[str, Any]:
    """Return available sectors with aggregate stats."""
    from collections import defaultdict
    universe = _market.get_universe()
    sectors: dict[str, dict] = defaultdict(lambda: {"count": 0, "avg_momentum": 0.0, "avg_volatility": 0.0})
    for a in universe:
        s = sectors[a.sector]
        s["count"] += 1
        s["avg_momentum"] += a.momentum
        s["avg_volatility"] += a.volatility
    for s_data in sectors.values():
        n = s_data["count"]
        s_data["avg_momentum"] = round(s_data["avg_momentum"] / n, 3)
        s_data["avg_volatility"] = round(s_data["avg_volatility"] / n, 3)
    return {"sectors": dict(sectors)}
