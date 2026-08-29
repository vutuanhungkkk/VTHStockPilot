"""Node 4 — Forecast Agent.

Estimates per-asset:
  - Expected excess return over benchmark
  - Return volatility (uncertainty)
  - Probability of outperforming the benchmark
  - Model confidence

Strategy (controlled by STOCK_USE_MLFLOW_MODEL in config / .env):
  - use_mlflow_model=True  → MLflow trained model predicts expected_excess_return
  - use_mlflow_model=False → Deterministic linear formula (original behaviour)

In both modes, vol, outperform_prob, and confidence are computed analytically
from market signals. The MLflow model only replaces the excess return estimate.

Fallback:
  If the MLflow model fails to load (not yet trained, registry empty, etc.),
  the node silently falls back to the linear formula and logs a warning.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from app.domain.schemas import ForecastScore, FundamentalScore, MarketSnapshot
from app.workflows.state import RecommendationState

logger = logging.getLogger(__name__)

# Benchmark excess return assumption (annualised)
_BENCHMARK_RETURN = 0.078

# Signal weights for the linear fallback forecast
_SIGNAL_WEIGHTS = {
    "momentum": 0.30,
    "quality_composite": 0.25,
    "expected_return": 0.25,
    "sentiment": 0.10,
    "value": 0.10,
}

# ── MLflow model cache (module-level, loaded once per process) ─────────────────

_mlflow_model = None          # Loaded sklearn/LightGBM model object
_mlflow_model_loaded = False  # True after first attempt (prevents repeated retries)
_mlflow_features: list[str] | None = None  # Feature column order the model expects


def _get_mlflow_model():
    """Lazy-load the MLflow ranked model (cached after first call).

    Returns (model, feature_list) or (None, None) on failure.
    """
    global _mlflow_model, _mlflow_model_loaded, _mlflow_features

    if _mlflow_model_loaded:
        return _mlflow_model, _mlflow_features

    _mlflow_model_loaded = True  # mark so we don't retry on every request

    try:
        from app.core.config import get_settings
        import mlflow
        import mlflow.sklearn
        import mlflow.pyfunc

        settings = get_settings()
        if not settings.use_mlflow_model:
            logger.debug("use_mlflow_model=False — skipping MLflow model load")
            return None, None

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        stage = settings.mlflow_model_stage  # "Production" | "Staging" | "None"

        if stage.lower() in ("none", ""):
            model_uri = f"models:/{settings.model_name}/latest"
        else:
            model_uri = f"models:/{settings.model_name}/{stage}"

        logger.info("Loading MLflow ranking model from %s ...", model_uri)
        _mlflow_model = mlflow.sklearn.load_model(model_uri)

        # Determine feature order from model metadata if available
        try:
            from app.etl.feature_eng import FEATURES, TRAIN_FEATURES
            # Prefer TRAIN_FEATURES if the model was trained on them
            _mlflow_features = TRAIN_FEATURES
        except ImportError:
            from app.etl.feature_eng import FEATURES
            _mlflow_features = FEATURES

        logger.info(
            "MLflow model loaded: %s (%s) | features=%d",
            settings.model_name, stage, len(_mlflow_features),
        )

    except Exception as exc:
        logger.warning(
            "MLflow model load failed (%s) — falling back to linear formula. "
            "Train a model first with: python scripts/train_model.py",
            exc,
        )
        _mlflow_model = None
        _mlflow_features = None

    return _mlflow_model, _mlflow_features


def reset_mlflow_model_cache() -> None:
    """Force reload on next inference call (useful for testing / model updates)."""
    global _mlflow_model, _mlflow_model_loaded, _mlflow_features
    _mlflow_model = None
    _mlflow_model_loaded = False
    _mlflow_features = None
    logger.info("MLflow model cache cleared")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _asset_to_feature_row(
    asset: MarketSnapshot,
    fund: FundamentalScore,
    features: list[str],
) -> list[float]:
    """Build a feature vector in the order the model was trained on."""
    feature_map: dict[str, float] = {
        "momentum": asset.momentum,
        "momentum_3m": asset.momentum,          # proxy: snapshot has only 1 momentum
        "quality": asset.quality,
        "value": asset.value,
        "sentiment": asset.sentiment,
        "volatility": asset.volatility,
        "liquidity_score": asset.liquidity_score,
        "pe_ratio": asset.pe_ratio,
        "pb_ratio": asset.pb_ratio,
        "roe": asset.roe,
        "debt_to_equity": asset.debt_to_equity,
        "revenue_growth": asset.revenue_growth,
        "rsi": asset.rsi,
        "beta": asset.beta,
        # fundamental composites
        "quality_composite": fund.composite,
        "value_score": fund.value_score,
        "growth_score": fund.growth_score,
    }
    return [feature_map.get(f, 0.0) for f in features]


# ── Forecast implementations ───────────────────────────────────────────────────

def _forecast_linear(
    asset: MarketSnapshot,
    fund: FundamentalScore,
    regime_weight_momentum: float,
) -> float:
    """Original deterministic linear forecast (fallback when no model is available)."""
    return (
        _SIGNAL_WEIGHTS["momentum"] * asset.momentum * regime_weight_momentum
        + _SIGNAL_WEIGHTS["quality_composite"] * fund.composite
        + _SIGNAL_WEIGHTS["expected_return"] * asset.expected_return
        + _SIGNAL_WEIGHTS["sentiment"] * asset.sentiment
        + _SIGNAL_WEIGHTS["value"] * asset.value
    ) - _BENCHMARK_RETURN * 0.5


def _forecast_with_model(
    assets: list[MarketSnapshot],
    funds: list[FundamentalScore],
    model,
    features: list[str],
) -> list[float]:
    """Batch predict excess returns using the MLflow model.

    Returns a list of raw (unclipped) excess return predictions,
    one per asset in the same order as `assets`.
    """
    import pandas as pd

    rows = [
        _asset_to_feature_row(a, f, features)
        for a, f in zip(assets, funds)
    ]
    X = pd.DataFrame(rows, columns=features)
    preds = model.predict(X)
    return preds.tolist()


def _compute_vol_and_confidence(
    asset: MarketSnapshot,
    fund: FundamentalScore,
) -> tuple[float, float]:
    """Compute return volatility and model confidence from market signals."""
    vol_estimate = max(asset.volatility * min(asset.beta, 1.5) * 0.6, 0.05)
    rsi_confidence = 1.0 - abs(asset.rsi - 50.0) / 50.0
    confidence = min(
        0.95,
        0.45
        + 0.30 * fund.quality_score
        - 0.20 * (asset.volatility - 0.20)
        + 0.05 * rsi_confidence,
    )
    confidence = max(0.30, confidence)
    return vol_estimate, confidence


def _build_forecast_score(
    asset: MarketSnapshot,
    fund: FundamentalScore,
    excess: float,
) -> ForecastScore:
    """Assemble a ForecastScore from a precomputed excess return value."""
    vol_estimate, confidence = _compute_vol_and_confidence(asset, fund)
    z = excess / vol_estimate
    outperform_prob = _normal_cdf(z)

    return ForecastScore(
        symbol=asset.symbol,
        expected_excess_return=round(excess, 4),
        return_volatility=round(vol_estimate, 4),
        outperform_probability=round(outperform_prob, 4),
        confidence=round(confidence, 4),
    )


# ── LangGraph node ─────────────────────────────────────────────────────────────

def forecast_node(state: RecommendationState) -> dict[str, Any]:
    """Forecast node: estimates expected excess return for each asset.

    Uses the trained MLflow model when `use_mlflow_model=True` in config,
    otherwise falls back to the heuristic linear formula.
    """
    universe: list[MarketSnapshot] = state["universe"]
    fundamentals: list[FundamentalScore] = state.get("fundamental_scores", [])
    regime_weights: dict[str, float] = state.get("regime_weights", {})

    fund_map = {f.symbol: f for f in fundamentals}
    momentum_w = regime_weights.get("momentum", 1.0)

    _default_fund = FundamentalScore(
        symbol="", quality_score=0.5, value_score=0.5,
        growth_score=0.5, profitability_score=0.5, leverage_score=0.5, composite=0.5,
    )
    funds = [fund_map.get(a.symbol, _default_fund) for a in universe]

    # ── Try MLflow model ────────────────────────────────────────────────────────
    model, features = _get_mlflow_model()
    forecast_method = "linear"

    if model is not None and features is not None:
        try:
            excess_returns = _forecast_with_model(universe, funds, model, features)
            scores = [
                _build_forecast_score(asset, fund, excess)
                for asset, fund, excess in zip(universe, funds, excess_returns)
            ]
            forecast_method = "mlflow_model"
            logger.debug(
                "MLflow model forecast complete for %d assets (method=mlflow_model)",
                len(scores),
            )
        except Exception as exc:
            logger.warning(
                "MLflow model inference failed (%s) — falling back to linear formula",
                exc,
            )
            model = None  # trigger linear fallback below

    # ── Linear formula fallback ─────────────────────────────────────────────────
    if model is None:
        scores = [
            _build_forecast_score(
                asset,
                fund,
                _forecast_linear(asset, fund, momentum_w),
            )
            for asset, fund in zip(universe, funds)
        ]

    # ── Aggregate stats for pipeline stage ────────────────────────────────────
    n = len(scores)
    mean_excess = sum(s.expected_excess_return for s in scores) / n if n else 0.0
    mean_outperform = sum(s.outperform_probability for s in scores) / n if n else 0.0

    stages: list[dict] = state.get("pipeline_stages", [])
    stages.append({
        "node": "forecast",
        "forecast_method": forecast_method,
        "assets_forecast": n,
        "mean_excess_return": round(mean_excess, 4),
        "mean_outperform_prob": round(mean_outperform, 4),
    })

    return {"forecast_scores": scores, "pipeline_stages": stages}
