"""Backtest service — walk-forward simulation with rich metrics.

Benchmark: VNINDEX (Ho Chi Minh Stock Exchange composite index).
Risk-free rate: 4.5% p.a. — Vietnamese 5-year government bond yield.
"""
from __future__ import annotations

import math

import numpy as np

from app.domain.schemas import BacktestRequest, BacktestResponse


class BacktestService:
    """Deterministic demo backtest; production uses point-in-time feature snapshots."""

    def run(self, request: BacktestRequest) -> BacktestResponse:
        rng = np.random.default_rng(20260805)
        periods = request.months
        risk_factor = {"conservative": 0.65, "balanced": 1.0, "growth": 1.25}[
            request.profile.risk_level.value
        ]

        gross = rng.normal(0.0095 * risk_factor, 0.038 * risk_factor, periods)
        benchmark = rng.normal(0.0078, 0.041, periods)
        freq_cost = {"weekly": 0.55, "monthly": 0.30, "quarterly": 0.12}[
            request.rebalance_frequency.value
        ]
        cost_per_period = request.transaction_cost_bps / 10_000 * freq_cost
        net_returns = gross - cost_per_period

        equity = np.cumprod(1 + net_returns)
        benchmark_equity = np.cumprod(1 + benchmark)
        drawdowns = equity / np.maximum.accumulate(equity) - 1
        downside = net_returns[net_returns < 0]

        annual_return = float(equity[-1] ** (12 / periods) - 1)
        annual_vol = float(net_returns.std(ddof=1) * math.sqrt(12))
        downside_vol = float(downside.std(ddof=1) * math.sqrt(12)) if len(downside) > 1 else annual_vol
        benchmark_annual = float(benchmark_equity[-1] ** (12 / periods) - 1)
        sharpe = (annual_return - 0.045) / max(annual_vol, 0.001)
        sortino = (annual_return - 0.045) / max(downside_vol, 0.001)
        max_dd = float(drawdowns.min())
        calmar = annual_return / abs(max_dd) if abs(max_dd) > 0.001 else 0.0
        tracking_error = float(np.std(net_returns - benchmark, ddof=1) * math.sqrt(12))
        active_return = annual_return - benchmark_annual
        information_ratio = active_return / max(tracking_error, 0.001)
        hit_rate = float(np.mean(net_returns > benchmark))
        turnover = freq_cost

        # Equity curve
        curve = [
            {"period": f"M{i + 1}", "portfolio": round(float(v), 4), "benchmark": round(float(b), 4)}
            for i, (v, b) in enumerate(zip(equity, benchmark_equity))
        ]

        # Rolling Sharpe (12-month window)
        rolling_sharpe: list[dict] = []
        window = 12
        for i in range(window, periods + 1):
            window_ret = net_returns[i - window: i]
            rs = float((window_ret.mean() * 12 - 0.045) / max(window_ret.std(ddof=1) * math.sqrt(12), 0.001))
            rolling_sharpe.append({"period": f"M{i}", "sharpe": round(rs, 3)})

        # Drawdown series
        drawdown_series = [
            {"period": f"M{i + 1}", "drawdown": round(float(d), 4)}
            for i, d in enumerate(drawdowns)
        ]

        # Monthly returns heat-map data
        monthly_returns = [
            {"period": f"M{i + 1}", "return": round(float(r), 4), "excess": round(float(r - b), 4)}
            for i, (r, b) in enumerate(zip(net_returns, benchmark))
        ]

        return BacktestResponse(
            period_months=periods,
            rebalance_frequency=request.rebalance_frequency.value,
            transaction_cost_bps=request.transaction_cost_bps,
            annualized_return=round(annual_return, 4),
            benchmark_return=round(benchmark_annual, 4),
            annualized_volatility=round(annual_vol, 4),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown=round(max_dd, 4),
            hit_rate=round(hit_rate, 3),
            turnover=round(turnover, 3),
            calmar_ratio=round(calmar, 3),
            information_ratio=round(information_ratio, 3),
            equity_curve=curve,
            rolling_sharpe=rolling_sharpe,
            drawdown_series=drawdown_series,
            monthly_returns=monthly_returns,
        )
