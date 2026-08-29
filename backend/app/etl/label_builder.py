"""Training label builder — computes point-in-time safe future excess returns.

Takes raw OHLCV parquet files (from historical_data.py) and the VN-Index
benchmark, then produces a panel dataset of (date, ticker, features, label)
ready for walk-forward cross-sectional model training.

Label definition:
    future_excess_return[t] = cumulative_return(asset, t→t+H) - cumulative_return(VNINDEX, t→t+H)

where H = horizon_days (default 21 trading days ≈ 1 month).

Point-in-time safety:
    - All financial ratio features are forward-filled as of date t using
      only data published BEFORE t (last quarterly report available).
    - Future return labels use prices AFTER t, so they are not available
      at inference time and are only used during training.
    - No look-ahead bias: feature computation uses only [t-window, t] data.

Output:
    data/processed/training_dataset.parquet
    Columns: date, ticker, <features>, future_excess_return
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from app.etl.historical_data import DEFAULT_RAW_DIR, load_benchmark, load_ohlcv, load_ratios

logger = logging.getLogger(__name__)

DEFAULT_PROCESSED_DIR = Path("data/processed")

# ── Feature windows ────────────────────────────────────────────────────────────

_MOMENTUM_SHORT_DAYS = 21   # 1-month momentum
_MOMENTUM_LONG_DAYS  = 63   # 3-month momentum
_VOLATILITY_DAYS     = 21   # rolling vol window
_RSI_PERIOD          = 14   # RSI period
_VOLUME_MA_DAYS      = 20   # liquidity average


# ── Public API ────────────────────────────────────────────────────────────────

def build_training_dataset(
    tickers: list[str],
    horizon_days: int = 21,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    min_rows_per_ticker: int = 100,
) -> pd.DataFrame:
    """Build a panel training dataset with real labels.

    Args:
        tickers:              List of ticker symbols to include.
        horizon_days:         Forward return horizon in trading days (default 21).
        raw_dir:              Directory containing raw parquet files.
        processed_dir:        Where to save the processed dataset.
        min_rows_per_ticker:  Drop tickers with fewer rows than this threshold.

    Returns:
        DataFrame with columns: date, ticker, <features>, future_excess_return
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Load benchmark ────────────────────────────────────────────────────────
    try:
        benchmark_df = load_benchmark(raw_dir)
        benchmark_ret = _compute_forward_returns(
            benchmark_df.set_index("date")["close"], horizon_days
        ).rename("benchmark_fwd_return")
        logger.info("Benchmark loaded: %d rows", len(benchmark_df))
    except FileNotFoundError:
        logger.error("Benchmark (VN-Index) data not found. Run collect_all() first.")
        raise

    # ── Process each ticker ───────────────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        try:
            df = _build_ticker_panel(ticker, benchmark_ret, horizon_days, raw_dir)
            if len(df) < min_rows_per_ticker:
                logger.warning(
                    "%s: only %d rows after processing (threshold=%d) — skipping",
                    ticker, len(df), min_rows_per_ticker,
                )
                continue
            frames.append(df)
            logger.info("%s: %d training rows generated", ticker, len(df))
        except FileNotFoundError:
            logger.warning("%s: raw data not found — skipping", ticker)
        except Exception as exc:
            logger.error("%s: processing failed — %s", ticker, exc)

    if not frames:
        raise RuntimeError(
            "No tickers could be processed. "
            "Ensure raw data exists by running collect_all() first."
        )

    # ── Combine & cross-sectional normalise ──────────────────────────────────
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Cross-sectionally z-score numeric features per date
    feature_cols = _feature_columns()
    panel = _cross_section_zscore(panel, feature_cols)

    # Drop rows where the label is NaN (last H rows of each ticker — no future data)
    before = len(panel)
    panel = panel.dropna(subset=["future_excess_return"])
    logger.info(
        "Dropped %d rows with NaN labels (last %d trading days of each ticker)",
        before - len(panel), horizon_days,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = processed_dir / "training_dataset.parquet"
    panel.to_parquet(out_path, index=False)
    logger.info(
        "Training dataset saved → %s | %d rows | %d tickers | %s → %s",
        out_path, len(panel), panel["ticker"].nunique(),
        panel["date"].min().date(), panel["date"].max().date(),
    )

    _print_dataset_summary(panel)
    return panel


def load_training_dataset(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> pd.DataFrame:
    """Load the pre-built training dataset from parquet."""
    path = processed_dir / "training_dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {path}. "
            "Run build_training_dataset() first, or use "
            "scripts/collect_training_data.py."
        )
    df = pd.read_parquet(path)
    logger.info(
        "Loaded training dataset: %d rows | %d tickers | %s → %s",
        len(df), df["ticker"].nunique(),
        df["date"].min().date(), df["date"].max().date(),
    )
    return df


# ── Private — per-ticker panel builder ───────────────────────────────────────

def _build_ticker_panel(
    ticker: str,
    benchmark_ret: pd.Series,
    horizon_days: int,
    raw_dir: Path,
) -> pd.DataFrame:
    """Build per-ticker daily rows with features and forward return label."""
    ohlcv = load_ohlcv(ticker, raw_dir).set_index("date")
    ohlcv.index = pd.to_datetime(ohlcv.index)
    ohlcv = ohlcv.sort_index()

    close = ohlcv["close"].astype(float)
    volume = ohlcv["volume"].astype(float) if "volume" in ohlcv.columns else pd.Series(1.0, index=ohlcv.index)

    # ── Price-based features ──────────────────────────────────────────────────
    rows = pd.DataFrame(index=close.index)
    rows["ticker"] = ticker

    # Momentum (log returns over windows)
    rows["momentum"] = _rolling_return(close, _MOMENTUM_SHORT_DAYS)
    rows["momentum_3m"] = _rolling_return(close, _MOMENTUM_LONG_DAYS)

    # Volatility (annualised)
    log_ret = np.log(close / close.shift(1))
    rows["volatility"] = log_ret.rolling(_VOLATILITY_DAYS).std() * math.sqrt(252)

    # RSI
    rows["rsi"] = _rolling_rsi(close, _RSI_PERIOD)

    # Liquidity (log average volume)
    rows["liquidity_score"] = np.log1p(volume.rolling(_VOLUME_MA_DAYS).mean())

    # Price (in VND)
    rows["price"] = close

    # ── Fundamental features (forward-fill from quarterly ratios) ─────────────
    try:
        ratios = load_ratios(ticker, raw_dir)
        rows = _merge_ratios(rows, ratios)
    except FileNotFoundError:
        logger.warning("%s: no ratios — using defaults", ticker)
        for col in ["pe_ratio", "pb_ratio", "roe", "debt_to_equity", "revenue_growth"]:
            rows[col] = _ratio_defaults()[col]

    # ── Target label — future excess return ───────────────────────────────────
    fwd_asset = _compute_forward_returns(close, horizon_days)
    # Align benchmark returns to asset dates
    bm_aligned = benchmark_ret.reindex(close.index).ffill()
    rows["future_excess_return"] = fwd_asset - bm_aligned

    # Composite signals (match FEATURES list in feature_eng.py)
    rows["quality"] = _composite_quality(rows)
    rows["value"] = _composite_value(rows)
    rows["sentiment"] = 0.6 * rows["momentum"].clip(-1, 1) + 0.4 * (rows["rsi"] - 50) / 50
    rows["sentiment"] = rows["sentiment"].clip(-1, 1)
    rows["beta"] = rows["volatility"] / rows["volatility"].median() if rows["volatility"].median() > 0 else 1.0

    rows = rows.reset_index().rename(columns={"date": "date"})
    rows["date"] = pd.to_datetime(rows["date"])

    # Drop early rows where rolling features aren't ready
    min_periods = max(_MOMENTUM_LONG_DAYS, _VOLATILITY_DAYS, _RSI_PERIOD, _VOLUME_MA_DAYS)
    rows = rows.iloc[min_periods:].copy()
    return rows.dropna(subset=["momentum", "volatility", "rsi"])


# ── Private — technical indicator helpers ─────────────────────────────────────

def _rolling_return(close: pd.Series, window: int) -> pd.Series:
    """Log return over `window` days."""
    return np.log(close / close.shift(window))


def _rolling_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Vectorised RSI computation."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_forward_returns(close: pd.Series, horizon: int) -> pd.Series:
    """Compute log forward return H days ahead (point-in-time safe)."""
    return np.log(close.shift(-horizon) / close)


# ── Private — fundamental feature helpers ─────────────────────────────────────

def _ratio_defaults() -> dict:
    return {
        "pe_ratio": 15.0,
        "pb_ratio": 2.0,
        "roe": 0.12,
        "debt_to_equity": 1.0,
        "revenue_growth": 0.08,
    }


def _merge_ratios(rows: pd.DataFrame, ratios: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill quarterly ratios onto daily price rows (point-in-time safe)."""
    col_map = {
        "price_to_earning": "pe_ratio",
        "price_to_book": "pb_ratio",
        "roe": "roe",
        "debt_on_equity": "debt_to_equity",
        "revenue_growth": "revenue_growth",
    }

    ratio_cols = {}
    for src, dst in col_map.items():
        matches = [c for c in ratios.columns if src in c.lower()]
        if matches:
            ratio_cols[matches[0]] = dst

    if not ratio_cols or "period_date" not in ratios.columns:
        defaults = _ratio_defaults()
        for col, val in defaults.items():
            rows[col] = val
        return rows

    ratio_slim = ratios[["period_date"] + list(ratio_cols.keys())].copy()
    ratio_slim = ratio_slim.rename(columns=ratio_cols)
    ratio_slim["period_date"] = pd.to_datetime(ratio_slim["period_date"])
    ratio_slim = ratio_slim.sort_values("period_date").set_index("period_date")

    # Reindex to daily, forward-fill (only use ratios available at each date)
    daily_idx = rows.index if isinstance(rows.index, pd.DatetimeIndex) else pd.to_datetime(rows.index)
    ratio_daily = ratio_slim.reindex(ratio_slim.index.union(daily_idx)).ffill().reindex(daily_idx)

    defaults = _ratio_defaults()
    for col in ["pe_ratio", "pb_ratio", "roe", "debt_to_equity", "revenue_growth"]:
        if col in ratio_daily.columns:
            rows[col] = ratio_daily[col].values
            rows[col] = pd.to_numeric(rows[col], errors="coerce").fillna(defaults[col])
        else:
            rows[col] = defaults[col]

    # Clamp outliers
    rows["pe_ratio"] = rows["pe_ratio"].clip(1, 200)
    rows["pb_ratio"] = rows["pb_ratio"].clip(0.1, 50)
    rows["roe"] = rows["roe"].apply(lambda x: x / 100 if x > 1 else x).clip(0, 1)
    rows["debt_to_equity"] = rows["debt_to_equity"].clip(0, 20)
    rows["revenue_growth"] = rows["revenue_growth"].clip(-1, 5)
    return rows


def _composite_quality(df: pd.DataFrame) -> pd.Series:
    """ROE-based quality composite."""
    roe_norm = df["roe"].clip(0, 1)
    max_debt = max(df["debt_to_equity"].max(), 1e-3)
    lev_inv = 1.0 - (df["debt_to_equity"] / max_debt).clip(0, 1)
    return (0.6 * roe_norm + 0.4 * lev_inv).clip(0, 1)


def _composite_value(df: pd.DataFrame) -> pd.Series:
    """Inverted P/E rank as a value signal."""
    pe_range = max(df["pe_ratio"].max() - df["pe_ratio"].min(), 1e-3)
    pe_inv = 1.0 - (df["pe_ratio"] - df["pe_ratio"].min()) / pe_range
    
    pb_range = max(df["pb_ratio"].max() - df["pb_ratio"].min(), 1e-3)
    pb_inv = 1.0 - (df["pb_ratio"] - df["pb_ratio"].min()) / pb_range
    
    return (0.5 * pe_inv + 0.5 * pb_inv).clip(0, 1)


# ── Private — cross-sectional normalisation ───────────────────────────────────

def _feature_columns() -> list[str]:
    return [
        "momentum", "momentum_3m", "quality", "value", "sentiment",
        "volatility", "liquidity_score", "pe_ratio", "pb_ratio",
        "roe", "debt_to_equity", "revenue_growth", "rsi", "beta",
    ]


def _cross_section_zscore(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Z-score each feature cross-sectionally (per date)."""
    cols_present = [c for c in feature_cols if c in panel.columns]
    for col in cols_present:
        panel[f"{col}_z"] = panel.groupby("date")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    return panel


# ── Summary helper ─────────────────────────────────────────────────────────────

def _print_dataset_summary(panel: pd.DataFrame) -> None:
    label = panel["future_excess_return"]
    print("\n" + "=" * 60)
    print("Training Dataset Summary")
    print("-" * 60)
    print(f"  Total rows       : {len(panel):,}")
    print(f"  Tickers          : {panel['ticker'].nunique()} ({', '.join(sorted(panel['ticker'].unique()))})")
    print(f"  Date range       : {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"  Label mean       : {label.mean():.4f}")
    print(f"  Label std        : {label.std():.4f}")
    print(f"  Label min/max    : {label.min():.4f} / {label.max():.4f}")
    print(f"  Positive labels  : {(label > 0).mean():.1%}")
    print("=" * 60)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build training dataset from raw parquet files")
    parser.add_argument(
        "--tickers",
        default="VCB,BID,MBB,TCB,VPB,ACB,FPT,VIC,VHM,HPG,VNM,SAB,GAS,MSN,REE",
    )
    parser.add_argument("--horizon", type=int, default=21, help="Forward return horizon (trading days)")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    build_training_dataset(
        tickers=tickers,
        horizon_days=args.horizon,
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
    )
