"""Feature engineering pipeline.

Provides two distinct interfaces:
1. **Inference** (snapshot-based): `compute_features(snapshots)` converts a list of
   MarketSnapshot objects into a cross-sectionally normalised DataFrame for
   real-time recommendation scoring.

2. **Training** (panel-based): `build_training_dataframe(tickers, ...)` fetches
   multi-year OHLCV + fundamental data via `label_builder`, computes time-series
   features, attaches `future_excess_return` labels, and returns a panel
   DataFrame ready for MLflow-tracked model training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.domain.schemas import MarketSnapshot

# ── Feature sets ──────────────────────────────────────────────────────────────

# Features used at inference time (must match MarketSnapshot attributes)
FEATURES = [
    "momentum", "quality", "value", "sentiment", "volatility", "liquidity_score",
    "pe_ratio", "pb_ratio", "roe", "debt_to_equity", "revenue_growth", "rsi", "beta",
]

# Extended feature set used during training (includes time-series signals)
TRAIN_FEATURES = FEATURES + ["momentum_3m"]


# ── Inference interface ────────────────────────────────────────────────────────

def compute_features(snapshots: list[MarketSnapshot]) -> pd.DataFrame:
    """Convert MarketSnapshot list to a cross-sectionally normalised feature DataFrame.

    Used at inference time inside the recommendation pipeline.
    """
    rows = []
    for s in snapshots:
        rows.append({f: getattr(s, f, 0.0) for f in FEATURES} | {"symbol": s.symbol})
    df = pd.DataFrame(rows).set_index("symbol")

    # Cross-sectional z-score normalisation
    for col in FEATURES:
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            df[f"{col}_z"] = (df[col] - mean) / std

    # Momentum composite: price momentum + RSI z-score
    df["momentum_composite"] = (
        0.6 * df.get("momentum_z", df["momentum"])
        + 0.4 * df.get("rsi_z", 0)
    )

    # Value composite: z-score of pe & pb (lower = better → flip)
    df["value_composite"] = -(
        0.5 * df.get("pe_ratio_z", 0)
        + 0.5 * df.get("pb_ratio_z", 0)
    )

    return df


# ── Training interface ─────────────────────────────────────────────────────────

def build_training_dataframe(
    tickers: list[str] | None = None,
    horizon_days: int = 21,
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
    use_cached: bool = True,
) -> pd.DataFrame:
    """Build or load a panel training DataFrame with real labels.

    This is a convenience wrapper over `app.etl.label_builder.build_training_dataset`.
    If `use_cached=True` and a processed parquet exists, it is loaded directly.

    Args:
        tickers:       List of HOSE ticker symbols. Defaults to all configured tickers.
        horizon_days:  Forward return horizon in trading days (default 21 ≈ 1 month).
        raw_dir:       Directory with raw OHLCV/ratio parquets from historical_data.py.
        processed_dir: Where to save/load the processed training dataset.
        use_cached:    Return cached parquet if it exists (skip rebuild).

    Returns:
        DataFrame with columns matching TRAIN_FEATURES + ["future_excess_return",
        "date", "ticker"] plus cross-sectional z-score columns.
    """
    from app.etl.label_builder import build_training_dataset, load_training_dataset

    cached_path = processed_dir / "training_dataset.parquet"
    if use_cached and cached_path.exists():
        return load_training_dataset(processed_dir)

    if tickers is None:
        from app.core.config import get_settings
        tickers = [t.strip() for t in get_settings().vnstock_tickers.split(",") if t.strip()]

    return build_training_dataset(
        tickers=tickers,
        horizon_days=horizon_days,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )
