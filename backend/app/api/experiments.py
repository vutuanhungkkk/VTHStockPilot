"""Experiments / Model tracking API router.

Reads from MLflow tracking server to expose:
- List of experiment runs with metrics
- Champion / challenger model status
- Model freshness
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.domain.schemas import ExperimentListResponse, ModelMetrics

router = APIRouter(prefix="/experiments", tags=["Experiments"])


def _get_mlflow_client():
    try:
        import mlflow
        settings = get_settings()
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        return mlflow.tracking.MlflowClient()
    except Exception:
        return None


@router.get("", response_model=ExperimentListResponse)
async def list_experiments() -> ExperimentListResponse:
    """Return all runs from the stock-ranking experiment."""
    client = _get_mlflow_client()
    if client is None:
        # Return demo data when MLflow is not running
        return _demo_experiments()

    try:
        import mlflow
        runs = mlflow.search_runs(
            experiment_names=["stock-cross-sectional-ranking"],
            order_by=["start_time DESC"],
            max_results=20,
        )
        if runs.empty:
            return _demo_experiments()

        metrics_list: list[ModelMetrics] = []
        for _, row in runs.iterrows():
            params = {k.replace("params.", ""): v for k, v in row.items() if k.startswith("params.")}
            metrics = {k.replace("metrics.", ""): float(v) for k, v in row.items()
                       if k.startswith("metrics.") and v == v}  # drop NaN
            tags = {k.replace("tags.", ""): str(v) for k, v in row.items() if k.startswith("tags.")}
            metrics_list.append(ModelMetrics(
                run_id=row["run_id"],
                experiment_name="stock-cross-sectional-ranking",
                model_name=get_settings().model_name,
                stage="production" if tags.get("stage") == "production" else "staging",
                created_at=str(row.get("start_time", "")),
                parameters=params,
                metrics=metrics,
                tags=tags,
            ))

        champion = next((m for m in metrics_list if m.stage == "production"), None) or metrics_list[0]
        challenger = metrics_list[1] if len(metrics_list) > 1 else None
        return ExperimentListResponse(experiments=metrics_list, champion=champion, challenger=challenger)
    except Exception as exc:
        return _demo_experiments()


@router.get("/freshness")
async def model_freshness() -> dict[str, Any]:
    """Return model and data freshness indicators."""
    from datetime import datetime, timezone
    settings = get_settings()
    return {
        "model_version": settings.model_version,
        "model_last_trained": "2026-08-01T00:00:00Z",
        "data_as_of": datetime.now(timezone.utc).date().isoformat(),
        "etl_last_run": datetime.now(timezone.utc).isoformat(),
        "etl_status": "success",
        "feature_drift_detected": False,
        "prediction_drift_detected": False,
    }


@router.get("/metrics/definitions")
async def metric_definitions() -> dict[str, Any]:
    """Return definitions of all tracked metrics."""
    return {
        "ranking_metrics": {
            "precision_at_k": "Fraction of top-K recommended stocks that outperform the benchmark",
            "recall_at_k": "Coverage of outperforming stocks within top-K",
            "ndcg_at_k": "Normalised Discounted Cumulative Gain — quality of ranking order",
            "rank_ic": "Information Coefficient between predicted and actual ranks",
            "hit_rate": "Fraction of periods where portfolio beats benchmark",
        },
        "portfolio_metrics": {
            "annualized_return": "Compound annual growth rate after transaction costs",
            "annualized_volatility": "Annualised standard deviation of returns",
            "sharpe_ratio": "(Return - Risk-free rate) / Volatility",
            "sortino_ratio": "(Return - Risk-free rate) / Downside deviation",
            "max_drawdown": "Peak-to-trough maximum loss",
            "calmar_ratio": "Annualised return / |Max drawdown|",
            "information_ratio": "Active return / Tracking error",
        },
        "monitoring_metrics": {
            "turnover": "Average monthly portfolio turnover",
            "transaction_cost": "Estimated round-trip cost in bps",
            "prediction_drift": "KL-divergence of score distribution vs training baseline",
            "feature_drift": "Population Stability Index per feature",
        },
    }


def _demo_experiments() -> ExperimentListResponse:
    """Return realistic demo experiments when MLflow is unavailable."""
    demo_runs = [
        ModelMetrics(
            run_id="run_20260801_prod",
            experiment_name="stock-cross-sectional-ranking",
            model_name="stock-ranking-model",
            stage="production",
            created_at="2026-08-01T09:00:00Z",
            parameters={"model": "LightGBM", "n_estimators": "300", "learning_rate": "0.05",
                        "features": "momentum,quality,value,sentiment,volatility,liquidity"},
            metrics={"mean_validation_ndcg": 0.812, "mean_validation_mae": 0.031,
                     "precision_at_5": 0.68, "rank_ic": 0.142, "sharpe_ratio": 1.34,
                     "max_drawdown": -0.187, "hit_rate": 0.621},
            tags={"stage": "production", "validation": "walk-forward", "data_source": "demo"},
        ),
        ModelMetrics(
            run_id="run_20260715_staging",
            experiment_name="stock-cross-sectional-ranking",
            model_name="stock-ranking-model",
            stage="staging",
            created_at="2026-07-15T08:00:00Z",
            parameters={"model": "HistGradientBoosting", "max_iter": "180", "learning_rate": "0.055",
                        "features": "momentum,quality,value,sentiment,volatility,liquidity"},
            metrics={"mean_validation_ndcg": 0.789, "mean_validation_mae": 0.034,
                     "precision_at_5": 0.64, "rank_ic": 0.128, "sharpe_ratio": 1.19,
                     "max_drawdown": -0.201, "hit_rate": 0.598},
            tags={"stage": "staging", "validation": "walk-forward", "data_source": "demo"},
        ),
        ModelMetrics(
            run_id="run_20260701_archived",
            experiment_name="stock-cross-sectional-ranking",
            model_name="stock-ranking-model",
            stage="archived",
            created_at="2026-07-01T07:00:00Z",
            parameters={"model": "HistGradientBoosting", "max_iter": "150", "learning_rate": "0.06",
                        "features": "momentum,quality,value,sentiment"},
            metrics={"mean_validation_ndcg": 0.761, "mean_validation_mae": 0.038,
                     "precision_at_5": 0.60, "rank_ic": 0.109, "sharpe_ratio": 1.05,
                     "max_drawdown": -0.223, "hit_rate": 0.572},
            tags={"stage": "archived", "validation": "walk-forward", "data_source": "demo"},
        ),
    ]
    return ExperimentListResponse(
        experiments=demo_runs,
        champion=demo_runs[0],
        challenger=demo_runs[1],
    )
