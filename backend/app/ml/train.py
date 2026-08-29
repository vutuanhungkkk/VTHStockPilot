"""Train and register stock ranking models with MLflow.

Trains two models on REAL Vietnamese market data:
  1. HistGradientBoostingRegressor (scikit-learn baseline / champion)
  2. LightGBM (challenger — registered as production if NDCG is better)

Data sources:
  - Primary  : data/processed/training_dataset.parquet (pre-built via label_builder)
  - Fallback : demo_training_data() (random synthetic — only if real data unavailable)

Walk-forward time-series split is used to prevent look-ahead bias.
Metrics logged: MAE, NDCG@K, Precision@K, RankIC, Sharpe (pseudo).

To train:
    python backend/scripts/train_model.py
    # or directly:
    python -m app.ml.train
"""
from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, ndcg_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

from app.core.config import get_settings
from app.etl.feature_eng import FEATURES, TRAIN_FEATURES

logger = logging.getLogger(__name__)

TARGET = "future_excess_return"

# Use the extended training feature set when the column is available,
# otherwise fall back to the inference-time feature set.
_CANDIDATE_FEATURES = TRAIN_FEATURES  # includes momentum_3m etc.


# ── Data loading ───────────────────────────────────────────────────────────────

def load_real_training_data(
    processed_dir: Path = Path("data/processed"),
) -> tuple[pd.DataFrame, dict]:
    """Load the pre-built training dataset.

    Returns:
        (dataframe, metadata_dict) where metadata_dict carries provenance info
        that will be logged as MLflow tags.

    Raises:
        FileNotFoundError: if no processed dataset exists — run
            scripts/collect_training_data.py first.
    """
    path = processed_dir / "training_dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at '{path}'. "
            "Run:  python backend/scripts/collect_training_data.py"
        )

    df = pd.read_parquet(path)
    df = df.dropna(subset=[TARGET])

    # Determine which features are actually present in the dataset
    avail_features = [f for f in _CANDIDATE_FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]

    if missing:
        logger.warning("Missing base features in training data: %s", missing)

    meta = {
        "data_source": "vnstock_real",
        "rows": len(df),
        "tickers": ",".join(sorted(df["ticker"].unique())) if "ticker" in df.columns else "unknown",
        "date_min": str(df["date"].min().date()) if "date" in df.columns else "unknown",
        "date_max": str(df["date"].max().date()) if "date" in df.columns else "unknown",
        "features_used": ",".join(avail_features),
        "n_features": len(avail_features),
        "positive_label_rate": float((df[TARGET] > 0).mean()),
        "label_mean": float(df[TARGET].mean()),
        "label_std": float(df[TARGET].std()),
    }

    logger.info(
        "Real training data loaded: %d rows | %d tickers | %s → %s",
        meta["rows"], df["ticker"].nunique() if "ticker" in df.columns else "?",
        meta["date_min"], meta["date_max"],
    )
    return df, meta


def _select_features(data: pd.DataFrame) -> list[str]:
    """Return the feature columns available in this dataset."""
    avail = [f for f in _CANDIDATE_FEATURES if f in data.columns]
    if not avail:
        raise ValueError(
            f"None of the expected features found in dataset. "
            f"Expected: {_CANDIDATE_FEATURES}"
        )
    return avail


# ── Demo dataset (fallback only) ───────────────────────────────────────────────

def demo_training_data(rows: int = 900) -> pd.DataFrame:
    """Synthetic demo dataset — used only when real data is unavailable."""
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({f: rng.uniform(0, 1, rows) for f in FEATURES})
    frame["momentum_3m"] = rng.uniform(-0.3, 0.5, rows)
    frame["volatility"] = rng.uniform(0.12, 0.48, rows)
    frame["pe_ratio"] = rng.uniform(5, 50, rows)
    frame["pb_ratio"] = rng.uniform(0.5, 6, rows)
    frame["roe"] = rng.uniform(0.03, 0.35, rows)
    frame["beta"] = rng.uniform(0.4, 1.8, rows)
    noise = rng.normal(0, 0.025, rows)
    frame[TARGET] = (
        0.040 * frame["momentum"] + 0.030 * frame["quality"] + 0.020 * frame["value"]
        + 0.015 * frame["sentiment"] - 0.040 * frame["volatility"]
        + 0.010 * frame["roe"] - 0.008 * frame["debt_to_equity"]
        + noise
    )
    frame["as_of"] = pd.date_range("2022-01-01", periods=rows, freq="D")
    return frame


# ── Metric helpers ─────────────────────────────────────────────────────────────

def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    top_k_pred = set(np.argsort(y_pred)[-k:])
    top_k_true = set(np.argsort(y_true)[-k:])
    return len(top_k_pred & top_k_true) / k


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between predicted and actual ranks."""
    from scipy.stats import spearmanr
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if not np.isnan(corr) else 0.0


def pseudo_sharpe(y_true: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Simplified Sharpe: mean return of top-K predicted stocks / std."""
    if len(y_pred) < k:
        return 0.0
    top_k_idx = np.argsort(y_pred)[-k:]
    top_k_returns = y_true[top_k_idx]
    std = top_k_returns.std()
    return float(top_k_returns.mean() / std) if std > 1e-8 else 0.0


def _compute_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    k = min(5, max(1, len(y_true) // 4))
    
    # ndcg_score requires positive relevance scores (y_true)
    y_true_non_negative = y_true - y_true.min() if len(y_true) > 0 else y_true
    
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "ndcg": ndcg_score([y_true_non_negative], [y_pred]) if len(y_true) > 1 else 0.0,
        "precision_at_k": precision_at_k(y_true, y_pred, k),
        "rank_ic": rank_ic(y_true, y_pred),
        "pseudo_sharpe": pseudo_sharpe(y_true, y_pred, k),
    }


# ── Training ───────────────────────────────────────────────────────────────────

def _train_model(
    model_name: str,
    model,
    data: pd.DataFrame,
    features: list[str],
    splitter: TimeSeriesSplit,
) -> tuple[dict, object]:
    """Walk-forward training with per-fold metric logging."""
    fold_metrics: list[dict] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(data), start=1):
        train_set, valid_set = data.iloc[train_idx], data.iloc[valid_idx]
        model.fit(train_set[features], train_set[TARGET])
        pred = model.predict(valid_set[features])
        metrics = _compute_fold_metrics(valid_set[TARGET].to_numpy(), pred)
        fold_metrics.append(metrics)
        mlflow.log_metrics(
            {f"{model_name}_fold{fold}_{k}": v for k, v in metrics.items()},
            step=fold,
        )
        logger.info(
            "Fold %d | MAE=%.4f NDCG=%.4f RankIC=%.4f Sharpe=%.3f",
            fold, metrics["mae"], metrics["ndcg"], metrics["rank_ic"], metrics["pseudo_sharpe"],
        )

    # Final fit on entire dataset
    model.fit(data[features], data[TARGET])
    avg = {k: float(np.mean([f[k] for f in fold_metrics])) for k in fold_metrics[0]}
    return avg, model


def _log_feature_importance(model, model_name: str, features: list[str]) -> None:
    """Log feature importance as an MLflow artifact (LightGBM only)."""
    try:
        if not _LGB_AVAILABLE or not hasattr(model, "feature_importances_"):
            return
        importance = pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        Path("artifacts").mkdir(exist_ok=True)
        imp_path = f"artifacts/{model_name}_feature_importance.csv"
        importance.to_csv(imp_path, index=False)
        mlflow.log_artifact(imp_path)
        logger.info("Feature importance logged → %s", imp_path)
    except Exception as exc:
        logger.debug("Feature importance logging skipped: %s", exc)


# ── Main train() function ──────────────────────────────────────────────────────

def train(
    data: pd.DataFrame | None = None,
    processed_dir: Path = Path("data/processed"),
    use_demo_fallback: bool = True,
) -> str:
    """Train ranking models on real (or demo) data and register in MLflow.

    Args:
        data:              Pre-loaded DataFrame. If None, load from parquet or demo.
        processed_dir:     Directory with training_dataset.parquet.
        use_demo_fallback: If True, use demo data when real data is unavailable.

    Returns:
        MLflow run_id string.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("stock-cross-sectional-ranking")

    data_meta: dict = {}
    data_source_tag = "demo"

    # ── Load data ─────────────────────────────────────────────────────────────
    if data is None:
        try:
            data, data_meta = load_real_training_data(processed_dir)
            data_source_tag = "vnstock_real"
        except FileNotFoundError as exc:
            if not use_demo_fallback:
                raise
            logger.warning(
                "Real training data not found (%s). "
                "Using demo data. Run scripts/collect_training_data.py for real data.",
                exc,
            )
            data = demo_training_data()
            data_source_tag = "demo"
            data_meta = {
                "data_source": "demo",
                "rows": len(data),
                "features_used": ",".join(FEATURES),
            }

    features = _select_features(data)
    splitter = TimeSeriesSplit(n_splits=5)

    logger.info(
        "Starting training | source=%s | rows=%d | features=%d",
        data_source_tag, len(data), len(features),
    )

    with mlflow.start_run(
        tags={
            "task": "ranking",
            "validation": "walk-forward",
            "data_source": data_source_tag,
            "market": "VN-HOSE",
        }
    ) as run:

        # ── Champion: HistGBM ─────────────────────────────────────────────────
        hist_model = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=20, random_state=42,
        )
        hist_metrics, hist_model = _train_model(
            "hist_gbm", hist_model, data, features, splitter,
        )
        mlflow.log_params({
            "champion_model": "HistGradientBoostingRegressor",
            "hist_max_iter": 200,
            "hist_learning_rate": 0.05,
            "features": ",".join(features),
            "n_features": len(features),
            "horizon_days": data_meta.get("horizon_days", 21),
        })
        mlflow.log_metrics({f"hist_gbm_{k}": v for k, v in hist_metrics.items()})
        logger.info(
            "HistGBM avg | MAE=%.4f NDCG=%.4f RankIC=%.4f Sharpe=%.3f",
            hist_metrics["mae"], hist_metrics["ndcg"],
            hist_metrics["rank_ic"], hist_metrics["pseudo_sharpe"],
        )

        # ── Challenger: LightGBM ──────────────────────────────────────────────
        if _LGB_AVAILABLE:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                min_child_samples=max(10, len(data) // 200),
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1,
            )
            lgb_metrics, lgb_model = _train_model(
                "lgb", lgb_model, data, features, splitter,
            )
            mlflow.log_params({
                "challenger_model": "LightGBM",
                "lgb_n_estimators": 300,
                "lgb_num_leaves": 31,
            })
            mlflow.log_metrics({f"lgb_{k}": v for k, v in lgb_metrics.items()})
            _log_feature_importance(lgb_model, "lgb", features)
            logger.info(
                "LightGBM avg | MAE=%.4f NDCG=%.4f RankIC=%.4f Sharpe=%.3f",
                lgb_metrics["mae"], lgb_metrics["ndcg"],
                lgb_metrics["rank_ic"], lgb_metrics["pseudo_sharpe"],
            )

            # Elect production model: prefer LightGBM if NDCG is better
            if lgb_metrics["ndcg"] >= hist_metrics["ndcg"]:
                production_model = lgb_model
                prod_metrics = lgb_metrics
                model_tag = "LightGBM"
            else:
                production_model = hist_model
                prod_metrics = hist_metrics
                model_tag = "HistGradientBoostingRegressor"
        else:
            production_model = hist_model
            prod_metrics = hist_metrics
            model_tag = "HistGradientBoostingRegressor"

        # ── Log dataset ───────────────────────────────────────────────────────
        cols_to_log = features + [TARGET]
        if "ticker" in data.columns:
            cols_to_log.append("ticker")
        mlflow.log_input(
            mlflow.data.from_pandas(
                data[cols_to_log].head(500),  # sample to keep artifact small
                name="training-set-sample",
            ),
            context="training",
        )

        # Log dataset provenance as tags
        if data_meta:
            mlflow.set_tags({
                f"data_{k}": str(v) for k, v in data_meta.items()
                if k in ("tickers", "date_min", "date_max", "rows", "positive_label_rate")
            })

        # ── Register production model ─────────────────────────────────────────
        input_example = data[features].head(3)
        model_info = mlflow.sklearn.log_model(
            production_model,
            name="ranking_model",
            input_example=input_example,
        )
        registered = mlflow.register_model(model_info.model_uri, settings.model_name)
        mlflow.set_tags({
            "model_type": model_tag,
            "stage": "staging",
            "prod_ndcg": f"{prod_metrics['ndcg']:.4f}",
            "prod_rank_ic": f"{prod_metrics['rank_ic']:.4f}",
        })

        Path("artifacts").mkdir(exist_ok=True)
        print(f"\n{'='*60}")
        print(f"MLflow Run ID : {run.info.run_id}")
        print(f"Model         : {model_tag}")
        print(f"Data source   : {data_source_tag}")
        print(f"NDCG (avg)    : {prod_metrics['ndcg']:.4f}")
        print(f"RankIC (avg)  : {prod_metrics['rank_ic']:.4f}")
        print(f"Sharpe (avg)  : {prod_metrics['pseudo_sharpe']:.4f}")
        print(f"Registered    : {settings.model_name} v{registered.version}")
        print(f"{'='*60}\n")
        return run.info.run_id


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_id = train()
    print(f"MLflow run: {run_id}")
