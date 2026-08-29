import math
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.domain.schemas import (
    InvestorProfile, MarketSnapshot, PortfolioMetrics, RecommendationItem,
    RecommendationResponse, RiskLevel,
)
from app.services.market_data import MarketDataService

ProgressCallback = Callable[[str, int], Awaitable[None]]


class RecommendationService:
    def __init__(self, market_data: MarketDataService | None = None):
        self.market_data = market_data or MarketDataService()

    async def recommend(
        self, profile: InvestorProfile, progress: ProgressCallback | None = None
    ) -> RecommendationResponse:
        async def emit(stage: str, percent: int) -> None:
            if progress:
                await progress(stage, percent)

        state: dict[str, Any] = {"profile": profile, "universe": self.market_data.get_universe()}
        graph = self._build_graph(emit)
        state = await graph(state)
        return self._to_response(state)

    def _build_graph(self, emit):
        async def run(state):
            # These nodes form the same typed state transitions used by the LangGraph adapter.
            for node in (self._data_quality, self._market_regime, self._score_assets,
                         self._risk_filter, self._portfolio_optimize, self._explain):
                state = await node(state, emit)
            return state
        return run

    async def _data_quality(self, state, emit):
        await emit("Validating market data", 12)
        state["universe"] = [x for x in state["universe"] if x.price > 0 and x.liquidity_score >= .5]
        return state

    async def _market_regime(self, state, emit):
        await emit("Detecting market regime", 27)
        mean_momentum = float(np.mean([x.momentum for x in state["universe"]]))
        state["regime"] = "risk-on" if mean_momentum >= .65 else "neutral"
        return state

    async def _score_assets(self, state, emit):
        await emit("Scoring cross-sectional signals", 48)
        profile = state["profile"]
        risk_penalty = {RiskLevel.conservative: .30, RiskLevel.balanced: .20, RiskLevel.growth: .10}[profile.risk_level]
        scored = []
        for asset in state["universe"]:
            preference = .04 if asset.sector in profile.preferred_sectors else 0
            score = (
                .27 * asset.expected_return + .18 * asset.momentum + .18 * asset.quality
                + .14 * asset.value + .10 * asset.sentiment + .08 * asset.liquidity_score
                - risk_penalty * asset.volatility + preference
            )
            scored.append((asset, score))
        state["scored"] = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return state

    async def _risk_filter(self, state, emit):
        await emit("Applying suitability and risk constraints", 65)
        profile = state["profile"]
        max_vol = {RiskLevel.conservative: .25, RiskLevel.balanced: .31, RiskLevel.growth: .40}[profile.risk_level]
        eligible = [p for p in state["scored"] if p[0].sector not in profile.excluded_sectors and p[0].volatility <= max_vol]
        state["selected"] = eligible[:profile.max_positions]
        if len(state["selected"]) < 3:
            raise ValueError("Risk and sector constraints leave fewer than three eligible assets")
        return state

    async def _portfolio_optimize(self, state, emit):
        await emit("Optimizing portfolio weights", 80)
        profile = state["profile"]
        assets = [x[0] for x in state["selected"]]
        utility = np.array([max(a.expected_return, .01) / max(a.volatility ** 2, .01) for a in assets])
        weights = utility / utility.sum()
        weights = self._cap_and_redistribute(weights, profile.max_position_weight)
        state["weights"] = weights
        expected_return = float(sum(w * a.expected_return for w, a in zip(weights, assets)))
        # Conservative correlation assumption avoids presenting false precision.
        variance = sum((w * a.volatility) ** 2 for w, a in zip(weights, assets))
        covariance = .25 * sum(
            2 * weights[i] * weights[j] * assets[i].volatility * assets[j].volatility
            for i in range(len(assets)) for j in range(i + 1, len(assets))
        )
        volatility = math.sqrt(variance + covariance)
        state["metrics"] = PortfolioMetrics(
            expected_return=round(expected_return, 4), expected_volatility=round(volatility, 4),
            sharpe_ratio=round((expected_return - .045) / volatility, 2),
            diversification_score=round(1 - float(np.sum(weights ** 2)), 3),
        )
        return state

    async def _explain(self, state, emit):
        await emit("Generating structured explanations", 94)
        explanations = []
        for asset, score in state["selected"]:
            strengths = sorted(
                ((asset.momentum, "Strong price momentum"), (asset.quality, "High fundamental quality"),
                 (asset.value, "Attractive relative valuation"), (asset.sentiment, "Positive market sentiment")),
                reverse=True,
            )
            flags = []
            if asset.volatility > .28:
                flags.append("Elevated volatility")
            if asset.value < .5:
                flags.append("Premium valuation")
            explanations.append((score, [x[1] for x in strengths[:2]], flags))
        state["explanations"] = explanations
        await emit("Recommendation ready", 100)
        return state

    @staticmethod
    def _cap_and_redistribute(weights: np.ndarray, cap: float) -> np.ndarray:
        weights = weights.copy()
        for _ in range(20):
            over = weights > cap
            if not over.any():
                break
            excess = float((weights[over] - cap).sum())
            weights[over] = cap
            under = ~over
            if not under.any():
                break
            weights[under] += excess * weights[under] / weights[under].sum()
        return weights / weights.sum()

    def _to_response(self, state) -> RecommendationResponse:
        profile = state["profile"]
        items = []
        for index, ((asset, score), weight, (_, reasons, flags)) in enumerate(
            zip(state["selected"], state["weights"], state["explanations"]), start=1
        ):
            items.append(RecommendationItem(
                rank=index, symbol=asset.symbol, company=asset.company, sector=asset.sector,
                price=asset.price, score=round(score, 4), expected_return=asset.expected_return,
                volatility=asset.volatility, confidence=round(.55 + .4 * asset.quality, 3),
                weight=round(float(weight), 4), allocation=round(profile.capital * float(weight), 0),
                signals={"momentum": asset.momentum, "quality": asset.quality, "value": asset.value,
                         "sentiment": asset.sentiment, "liquidity": asset.liquidity_score},
                reasons=reasons, risk_flags=flags,
            ))
        return RecommendationResponse(
            recommendation_id=str(uuid.uuid4()), market_regime=state["regime"],
            model_version=get_settings().model_version, data_as_of=self.market_data.data_as_of(),
            profile=profile, recommendations=items, portfolio=state["metrics"],
        )
