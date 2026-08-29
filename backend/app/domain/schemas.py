"""Domain schemas — Pydantic models used across API, workflows, and DB."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from typing_extensions import TypedDict

from pydantic import BaseModel, Field, model_validator


# ── Enumerations ───────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    conservative = "conservative"
    balanced = "balanced"
    growth = "growth"


class MarketRegime(str, Enum):
    bull = "bull"
    bear = "bear"
    high_volatility = "high_volatility"
    neutral = "neutral"


class RebalanceFrequency(str, Enum):
    weekly = "weekly"
    monthly = "monthly"
    quarterly = "quarterly"


# ── Input Models ──────────────────────────────────────────────────────────────

class InvestorProfile(BaseModel):
    capital: float = Field(default=100_000_000, gt=0, description="Investment capital (VND)")
    risk_level: RiskLevel = RiskLevel.balanced
    horizon_months: int = Field(default=12, ge=1, le=120)
    max_positions: int = Field(default=5, ge=3, le=12)
    max_position_weight: float = Field(default=0.3, ge=0.1, le=0.5)
    preferred_sectors: list[str] = Field(default_factory=list)
    excluded_sectors: list[str] = Field(default_factory=list)
    esg_filter: bool = False


    @model_validator(mode="after")
    def validate_sectors(self) -> "InvestorProfile":
        overlap = set(self.preferred_sectors) & set(self.excluded_sectors)
        if overlap:
            raise ValueError(f"Sectors cannot be both preferred and excluded: {sorted(overlap)}")
        if self.max_position_weight * self.max_positions < 1:
            raise ValueError("max_position_weight is too small for max_positions")
        return self


# ── Market Data ────────────────────────────────────────────────────────────────

class MarketSnapshot(BaseModel):
    symbol: str
    company: str
    sector: str
    price: float
    expected_return: float
    volatility: float
    momentum: float
    quality: float
    value: float
    sentiment: float
    liquidity_score: float
    # Extended fundamental fields
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    roe: float = 0.0
    debt_to_equity: float = 0.0
    revenue_growth: float = 0.0
    # Technical
    rsi: float = 50.0
    macd_signal: float = 0.0
    beta: float = 1.0
    # Data quality
    data_age_hours: float = 0.0
    is_stale: bool = False


# ── Scores produced by individual nodes ───────────────────────────────────────

class FundamentalScore(BaseModel):
    symbol: str
    quality_score: float
    value_score: float
    growth_score: float
    profitability_score: float
    leverage_score: float
    composite: float


class ForecastScore(BaseModel):
    symbol: str
    expected_excess_return: float
    return_volatility: float
    outperform_probability: float
    confidence: float


class RankingScore(BaseModel):
    symbol: str
    final_score: float
    score_version: str
    signal_contributions: dict[str, float]


# ── Explanation ────────────────────────────────────────────────────────────────

class RecommendationExplanation(BaseModel):
    symbol: str
    summary: str
    key_drivers: list[str]
    risk_flags: list[str]
    shap_contributions: dict[str, float] = Field(default_factory=dict)


# ── Output Models ─────────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    rank: int
    symbol: str
    company: str
    sector: str
    price: float
    score: float
    expected_return: float
    volatility: float
    confidence: float
    weight: float
    allocation: float
    signals: dict[str, float]
    fundamental: dict[str, float] = Field(default_factory=dict)
    forecast: dict[str, float] = Field(default_factory=dict)
    reasons: list[str]
    risk_flags: list[str]
    shap_contributions: dict[str, float] = Field(default_factory=dict)
    explanation_text: str = ""


class PortfolioMetrics(BaseModel):
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    sortino_ratio: float = 0.0
    diversification_score: float
    max_drawdown_estimate: float = 0.0
    sector_concentration: dict[str, float] = Field(default_factory=dict)
    effective_n: float = 0.0    # 1 / HHI


class RecommendationResponse(BaseModel):
    recommendation_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    market_regime: str
    model_version: str
    data_as_of: str
    profile: InvestorProfile
    recommendations: list[RecommendationItem]
    portfolio: PortfolioMetrics
    pipeline_stages: list[dict[str, Any]] = Field(default_factory=list)
    disclaimer: str = "Model output for research purposes only; not financial advice."


# ── Portfolio ──────────────────────────────────────────────────────────────────

class PortfolioPosition(BaseModel):
    symbol: str
    company: str
    sector: str
    weight: float
    allocation: float
    expected_return: float
    volatility: float
    beta: float = 1.0


class PortfolioRiskReport(BaseModel):
    portfolio_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: PortfolioMetrics
    positions: list[PortfolioPosition]
    var_95: float = 0.0          # Value at Risk 95%
    cvar_95: float = 0.0         # Conditional VaR
    correlation_matrix: list[list[float]] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)




# ── Backtest ───────────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    profile: InvestorProfile = Field(default_factory=InvestorProfile)
    months: int = Field(default=24, ge=6, le=120)
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.monthly
    transaction_cost_bps: float = Field(default=10, ge=0, le=100)
    benchmark: str = "VNINDEX"


class BacktestResponse(BaseModel):
    period_months: int
    rebalance_frequency: str
    transaction_cost_bps: float
    annualized_return: float
    benchmark_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    hit_rate: float
    turnover: float
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    equity_curve: list[dict[str, float | str]]
    rolling_sharpe: list[dict[str, float | str]] = Field(default_factory=list)
    drawdown_series: list[dict[str, float | str]] = Field(default_factory=list)
    monthly_returns: list[dict[str, Any]] = Field(default_factory=list)


# ── Experiments ───────────────────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    run_id: str
    experiment_name: str
    model_name: str
    stage: str                  # "staging" | "production" | "archived"
    created_at: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    tags: dict[str, str]


class ExperimentListResponse(BaseModel):
    experiments: list[ModelMetrics]
    champion: ModelMetrics | None = None
    challenger: ModelMetrics | None = None


# ── LangGraph Workflow State (TypedDict) ──────────────────────────────────────

class RecommendationState(TypedDict, total=False):
    """Typed state passed between LangGraph nodes."""
    # Input
    profile: InvestorProfile
    universe: list[MarketSnapshot]
    # Intermediate
    regime: MarketRegime
    fundamental_scores: list[FundamentalScore]
    forecast_scores: list[ForecastScore]
    customer_adjusted_universe: list[MarketSnapshot]
    eligible_universe: list[MarketSnapshot]
    ranking_scores: list[RankingScore]
    selected: list[tuple[MarketSnapshot, RankingScore]]
    weights: list[float]
    # Output
    portfolio_metrics: PortfolioMetrics
    explanations: list[RecommendationExplanation]
    pipeline_stages: list[dict[str, Any]]
    errors: list[str]
