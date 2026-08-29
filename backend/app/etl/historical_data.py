"""Historical data collector — migrated to vnstock 4.x new API.

Uses vnstock.api (not the deprecated Vnstock() class) to fetch:
  - OHLCV price history via  vnstock.api.quote.Quote
  - Financial ratios via     vnstock.api.financial.Finance

Rate-limit aware: Guest plan = 20 req/min → auto-retry with exponential
backoff between requests. Configurable delay between tickers.

Usage (CLI):
    python -m app.etl.historical_data --tickers VCB,BID,FPT --years 3
    python scripts/collect_training_data.py  # calls collect_all() below

Output layout:
    data/raw/
        {TICKER}_ohlcv.parquet   — daily OHLCV rows
        {TICKER}_ratios.parquet  — quarterly ratios (pivoted to long form)
        vnindex_ohlcv.parquet    — VN-Index benchmark prices
        _manifest.json           — fetch summary (tickers, dates, counts)
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_RAW_DIR = Path("data/raw")
BENCHMARK_TICKER = "VNINDEX"

# ── Rate limit constants (Guest plan = 20 req/min) ────────────────────────────
# Each ticker needs 2 API calls (OHLCV + ratio).
# Each vnstock call internally makes multiple HTTP requests.
# To stay safe: 5s between every individual API call = ~12 "logical" calls/min.
_INTRA_TICKER_DELAY_S = 5.0   # delay between OHLCV fetch and ratio fetch (same ticker)
_INTER_TICKER_DELAY_S = 5.0   # delay after ratio fetch (before next ticker)
_RETRY_DELAYS = [70, 70, 70]  # wait 70s per retry — past the 60s rate-limit window

# Rate-limit signal words to detect from exception messages
_RATE_LIMIT_KEYWORDS = ("rate limit", "429", "too many", "exceeded", "quota", "giới hạn")


# ── vnstock 4.x API wrappers ──────────────────────────────────────────────────

def _get_quote(symbol: str, source: str = "VCI"):
    """Return a Quote object using the new vnstock.api."""
    from vnstock.api.quote import Quote
    return Quote(symbol=symbol, source=source)


def _get_finance(symbol: str, source: str = "VCI"):
    """Return a Finance object using the new vnstock.api."""
    from vnstock.api.financial import Finance
    return Finance(symbol=symbol, source=source)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception is a rate-limit error from vnstock."""
    err_str = str(exc).lower()
    return any(kw in err_str for kw in _RATE_LIMIT_KEYWORDS)


def _fetch_with_retry(fn, *args, retries: list[int] = _RETRY_DELAYS, **kwargs):
    """Call fn(*args, **kwargs), retrying on rate-limit or transient errors.

    On rate-limit errors the function waits `retries[i]` seconds before each
    retry attempt. Non-rate-limit exceptions propagate immediately.
    """
    last_exc = None
    total_attempts = 1 + len(retries)
    for attempt, wait in enumerate([0] + list(retries), start=1):
        if wait > 0:
            logger.warning(
                "Rate limit hit — waiting %ds before retry %d/%d ...",
                wait, attempt, total_attempts,
            )
            time.sleep(wait)
        try:
            result = fn(*args, **kwargs)
            return result
        except KeyboardInterrupt:
            raise  # always let Ctrl+C through
        except Exception as exc:
            if _is_rate_limit_error(exc):
                logger.warning("Rate limit detected (attempt %d/%d): %s", attempt, total_attempts, exc)
                last_exc = exc
            else:
                raise  # non-rate-limit errors propagate immediately
    raise RuntimeError(
        f"Rate limit exceeded after {total_attempts} attempts ({sum(retries)}s total wait). "
        "Consider registering a free API key at https://vnstocks.com/login"
    ) from last_exc


# ── Ratio parsing helpers ─────────────────────────────────────────────────────

# Map from vnstock 4.x item_en names → our field names
_RATIO_FIELD_MAP = {
    "P/E":                  "pe_ratio",
    "P/B":                  "pb_ratio",
    "ROE (%)":              "roe_pct",       # divide by 100 later
    "ROA (%)":              "roa_pct",
    "Debt to Equity":       "debt_to_equity",
    "Debt/Equity":          "debt_to_equity_alt",
    "After-tax Profit Margin (%)": "net_margin_pct",
    "Revenue Growth":       "revenue_growth",
    "Gross Margin (%)":     "gross_margin_pct",
}


def _parse_ratio_wide(ratio_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert the wide-format ratio DataFrame (item_en × period) to long format.

    Input:  rows = financial metrics, cols = ['item', 'item_en', 'item_id', '2018-Q1', ...]
    Output: rows = quarters, cols = [period_date, ticker, pe_ratio, pb_ratio, roe, ...]
    """
    if ratio_df is None or ratio_df.empty:
        return pd.DataFrame()

    # Identify period columns (format: YYYY-Qn)
    period_cols = [c for c in ratio_df.columns if isinstance(c, str) and "-Q" in c]
    if not period_cols:
        logger.warning("%s: no period columns found in ratio data", ticker)
        return pd.DataFrame()

    # Set item_en as index, keep only period columns
    try:
        ratio_indexed = ratio_df.set_index("item_en")[period_cols]
    except KeyError:
        logger.warning("%s: 'item_en' column not found in ratio data", ticker)
        return pd.DataFrame()

    # Transpose: rows = quarters, cols = metric names
    ratio_T = ratio_indexed.T.copy()
    ratio_T.index.name = "period_str"   # e.g. "2023-Q4"

    # Parse period → approximate date (last month of the quarter)
    def _quarter_to_date(period_str: str) -> pd.Timestamp:
        try:
            year, q = period_str.split("-Q")
            quarter = int(q)
            month = quarter * 3   # Q1→3, Q2→6, Q3→9, Q4→12
            return pd.Timestamp(year=int(year), month=month, day=28)
        except Exception:
            return pd.NaT

    ratio_T["period_date"] = [_quarter_to_date(p) for p in ratio_T.index]
    ratio_T["ticker"] = ticker

    # Rename metric columns to our standard names
    rename_map = {}
    for item_en, our_name in _RATIO_FIELD_MAP.items():
        if item_en in ratio_T.columns:
            rename_map[item_en] = our_name
    ratio_T = ratio_T.rename(columns=rename_map)

    # Keep only known columns + period_date + ticker
    keep_cols = ["period_date", "ticker"] + list(rename_map.values())
    present = [c for c in keep_cols if c in ratio_T.columns]
    ratio_long = ratio_T[present].copy()

    # ROE/ROA/margins: the new API returns them in decimal form already
    # (e.g. 0.2311 = 23.11% ROE) — no division needed
    # Rename pct fields to their final names
    if "roe_pct" in ratio_long.columns:
        ratio_long["roe"] = pd.to_numeric(ratio_long["roe_pct"], errors="coerce")
        ratio_long.drop(columns=["roe_pct"], inplace=True, errors="ignore")

    if "roa_pct" in ratio_long.columns:
        ratio_long["roa"] = pd.to_numeric(ratio_long["roa_pct"], errors="coerce")
        ratio_long.drop(columns=["roa_pct"], inplace=True, errors="ignore")

    if "net_margin_pct" in ratio_long.columns:
        ratio_long["net_margin"] = pd.to_numeric(ratio_long["net_margin_pct"], errors="coerce")
        ratio_long.drop(columns=["net_margin_pct"], inplace=True, errors="ignore")

    if "gross_margin_pct" in ratio_long.columns:
        ratio_long["gross_margin"] = pd.to_numeric(ratio_long["gross_margin_pct"], errors="coerce")
        ratio_long.drop(columns=["gross_margin_pct"], inplace=True, errors="ignore")

    # Use the better Debt/Equity column (prefer 'Debt to Equity')
    if "debt_to_equity" not in ratio_long.columns and "debt_to_equity_alt" in ratio_long.columns:
        ratio_long["debt_to_equity"] = ratio_long["debt_to_equity_alt"]
    ratio_long.drop(columns=["debt_to_equity_alt"], inplace=True, errors="ignore")

    # Convert all numeric columns
    for col in ratio_long.columns:
        if col not in ("period_date", "ticker"):
            ratio_long[col] = pd.to_numeric(ratio_long[col], errors="coerce")

    ratio_long = ratio_long.dropna(subset=["period_date"])
    ratio_long = ratio_long.sort_values("period_date").reset_index(drop=True)
    return ratio_long


# ── Public API ────────────────────────────────────────────────────────────────

def collect_all(
    tickers: list[str],
    years: int = 3,
    raw_dir: Path = DEFAULT_RAW_DIR,
    skip_existing: bool = False,
) -> dict:
    """Collect OHLCV + ratios for all tickers and the VN-Index benchmark.

    Args:
        tickers:       List of HOSE ticker symbols (e.g. ['VCB', 'BID']).
        years:         Number of years of history to fetch.
        raw_dir:       Directory to save parquet files.
        skip_existing: If True, skip tickers whose parquet file already exists.

    Returns:
        Manifest dict with summary statistics.
    """
    # Verify new API is available
    try:
        from vnstock.api.quote import Quote
        from vnstock.api.financial import Finance
    except ImportError as exc:
        raise ImportError(
            "vnstock >= 4.0 is required. "
            "Install/upgrade with:  pip install -U vnstock"
        ) from exc

    raw_dir.mkdir(parents=True, exist_ok=True)
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=int(years * 365.25))
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    logger.info(
        "Collecting %d tickers | %s → %s | output: %s",
        len(tickers), start_str, end_str, raw_dir,
    )

    manifest: dict = {
        "tickers": tickers,
        "start": start_str,
        "end": end_str,
        "years": years,
        "fetched_at": date.today().isoformat(),
        "api": "vnstock.api (v4.x)",
        "results": {},
    }

    # ── Benchmark (VN-Index) ──────────────────────────────────────────────────
    logger.info("Fetching benchmark (VNINDEX)...")
    manifest["benchmark"] = _collect_ohlcv(
        BENCHMARK_TICKER, start_str, end_str,
        raw_dir / "vnindex_ohlcv.parquet", skip_existing,
        is_benchmark=True,
    )
    time.sleep(_INTER_TICKER_DELAY_S)

    # ── Per-ticker data ───────────────────────────────────────────────────────
    for i, ticker in enumerate(tickers, start=1):
        logger.info("[%d/%d] Fetching %s ...", i, len(tickers), ticker)
        ohlcv_path = raw_dir / f"{ticker}_ohlcv.parquet"
        ratios_path = raw_dir / f"{ticker}_ratios.parquet"
        result: dict = {}

        result["ohlcv"] = _collect_ohlcv(
            ticker, start_str, end_str, ohlcv_path, skip_existing,
        )
        time.sleep(_INTRA_TICKER_DELAY_S)   # gap between OHLCV and ratio

        result["ratios"] = _collect_ratios(ticker, ratios_path, skip_existing)
        time.sleep(_INTER_TICKER_DELAY_S)   # gap before next ticker

        manifest["results"][ticker] = result
        _log_ticker_summary(ticker, result)

        # ETA hint
        remaining = len(tickers) - i
        if remaining > 0:
            eta_s = remaining * (_INTRA_TICKER_DELAY_S + _INTER_TICKER_DELAY_S + 6)
            logger.info(
                "  Progress: %d/%d done. ETA ~%.0fmin for remaining %d tickers.",
                i, len(tickers), eta_s / 60, remaining,
            )

    # ── Save manifest ─────────────────────────────────────────────────────────
    manifest_path = raw_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Manifest saved → %s", manifest_path)
    return manifest


def load_ohlcv(ticker: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load cached OHLCV parquet for a given ticker."""
    path = raw_dir / f"{ticker}_ohlcv.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"OHLCV data for {ticker} not found at {path}. "
            "Run collect_all() first."
        )
    return pd.read_parquet(path)


def load_ratios(ticker: str, raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load cached financial ratios parquet for a given ticker."""
    path = raw_dir / f"{ticker}_ratios.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Ratio data for {ticker} not found at {path}. "
            "Run collect_all() first."
        )
    return pd.read_parquet(path)


def load_benchmark(raw_dir: Path = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """Load the VN-Index benchmark OHLCV parquet."""
    path = raw_dir / "vnindex_ohlcv.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark data not found at {path}. Run collect_all() first."
        )
    return pd.read_parquet(path)


# ── Private helpers ───────────────────────────────────────────────────────────

def _collect_ohlcv(
    ticker: str,
    start: str,
    end: str,
    save_path: Path,
    skip_existing: bool,
    is_benchmark: bool = False,
) -> dict:
    """Fetch daily OHLCV for one ticker and save to parquet."""
    if skip_existing and save_path.exists():
        df = pd.read_parquet(save_path)
        return {"rows": len(df), "skipped": True, "path": str(save_path)}

    try:
        def _fetch():
            q = _get_quote(ticker)
            return q.history(start=start, end=end, interval="1D")

        hist = _fetch_with_retry(_fetch)

        if hist is None or hist.empty:
            logger.warning("%s: OHLCV returned empty", ticker)
            return {"rows": 0, "error": "empty response", "path": str(save_path)}

        hist = hist.copy()
        hist.columns = [c.lower() for c in hist.columns]

        # Rename 'time' → 'date'
        if "time" in hist.columns:
            hist.rename(columns={"time": "date"}, inplace=True)

        hist["date"] = pd.to_datetime(hist["date"])
        hist["ticker"] = ticker

        # vnstock 4.x returns prices in thousands VND (same as v3)
        # FPT close=70.8 means 70,800 VND actual price
        for col in ["open", "high", "low", "close"]:
            if col in hist.columns:
                hist[col] = hist[col].astype(float) * 1_000

        hist = hist.sort_values("date").reset_index(drop=True)
        hist.to_parquet(save_path, index=False)
        logger.info("%s: %d OHLCV rows saved → %s", ticker, len(hist), save_path)
        return {"rows": len(hist), "path": str(save_path)}

    except Exception as exc:
        logger.error("%s: OHLCV fetch failed — %s", ticker, exc)
        return {"rows": 0, "error": str(exc), "path": str(save_path)}


def _collect_ratios(
    ticker: str,
    save_path: Path,
    skip_existing: bool,
) -> dict:
    """Fetch quarterly financial ratios for one ticker and save to parquet."""
    if skip_existing and save_path.exists():
        df = pd.read_parquet(save_path)
        return {"rows": len(df), "skipped": True, "path": str(save_path)}

    try:
        def _fetch():
            f = _get_finance(ticker)
            return f.ratio(period="quarter", lang="en")

        ratio_wide = _fetch_with_retry(_fetch)

        if ratio_wide is None or ratio_wide.empty:
            logger.warning("%s: ratios returned empty", ticker)
            return {"rows": 0, "error": "empty response", "path": str(save_path)}

        # Parse wide format → long format with period_date
        ratio_long = _parse_ratio_wide(ratio_wide, ticker)

        if ratio_long.empty:
            logger.warning("%s: ratio parsing produced empty result", ticker)
            return {"rows": 0, "error": "parsing produced empty", "path": str(save_path)}

        ratio_long.to_parquet(save_path, index=False)
        logger.info(
            "%s: %d ratio rows saved → %s (periods: %s → %s)",
            ticker, len(ratio_long), save_path,
            ratio_long["period_date"].min().date() if not ratio_long.empty else "?",
            ratio_long["period_date"].max().date() if not ratio_long.empty else "?",
        )
        return {"rows": len(ratio_long), "path": str(save_path)}

    except Exception as exc:
        logger.error("%s: ratio fetch failed — %s", ticker, exc)
        return {"rows": 0, "error": str(exc), "path": str(save_path)}


def _log_ticker_summary(ticker: str, result: dict) -> None:
    ohlcv_rows = result.get("ohlcv", {}).get("rows", 0)
    ratio_rows = result.get("ratios", {}).get("rows", 0)
    ohlcv_err = result.get("ohlcv", {}).get("error", "")
    ratio_err = result.get("ratios", {}).get("error", "")
    status = "✓" if ohlcv_rows > 0 and ratio_rows > 0 else "✗"
    logger.info(
        "  %s %s | OHLCV=%d rows%s | Ratios=%d rows%s",
        status, ticker, ohlcv_rows,
        f" (ERR: {ohlcv_err})" if ohlcv_err else "",
        ratio_rows,
        f" (ERR: {ratio_err})" if ratio_err else "",
    )


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Collect historical HOSE data (vnstock 4.x API)")
    parser.add_argument(
        "--tickers",
        default="VCB,BID,MBB,TCB,VPB,ACB,FPT,VIC,VHM,HPG,VNM,SAB,GAS,MSN,REE",
        help="Comma-separated ticker list",
    )
    parser.add_argument("--years", type=float, default=3.0, help="Years of history")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    manifest = collect_all(
        tickers=tickers,
        years=args.years,
        raw_dir=Path(args.raw_dir),
        skip_existing=args.skip_existing,
    )

    print("\n" + "=" * 60)
    print(f"{'Ticker':<10} {'OHLCV rows':>12} {'Ratio rows':>12} {'Status':>10}")
    print("-" * 60)
    for ticker, res in manifest["results"].items():
        ohlcv_rows = res.get("ohlcv", {}).get("rows", 0)
        ratio_rows = res.get("ratios", {}).get("rows", 0)
        ok = "✓" if ohlcv_rows > 0 and ratio_rows > 0 else "✗"
        print(f"{ticker:<10} {ohlcv_rows:>12,} {ratio_rows:>12,} {ok:>10}")
    bm = manifest.get("benchmark", {})
    print(f"{'VNINDEX':<10} {bm.get('rows', 0):>12,} {'—':>12} {'✓' if bm.get('rows', 0) > 0 else '✗':>10}")
    print("=" * 60)
