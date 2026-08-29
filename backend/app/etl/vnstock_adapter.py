"""vnstock adapter — fetches live Vietnamese market data and returns MarketSnapshot objects.

Migrated to vnstock >= 4.0 API (vnstock.api):
    - vnstock.api.quote.Quote      → OHLCV price history
    - vnstock.api.financial.Finance → P/E, P/B, ROE, D/E, revenue growth

Signal derivation from raw data:
    momentum    = 30-day price return, winsorised → normalised [0, 1]
    volatility  = annualised std of daily log-returns (30d window)
    rsi         = 14-period RSI from close prices
    macd_signal = EMA(12) - EMA(26) of close, normalised
    quality     = composite of ROE, low leverage, profit margin
    value       = sector-relative P/E rank (lower P/E = higher value)
    sentiment   = momentum-RSI composite proxy
    liquidity   = log(avg_volume_30d) cross-sectionally normalised

Fallback: if vnstock is unavailable or a ticker fails, the ticker is silently
skipped; the caller decides whether to use demo data.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Sector mapping (HOSE — changes infrequently) ──────────────────────────────
SECTOR_MAP: dict[str, str] = {
    "VCB": "Banking",  "BID": "Banking",  "MBB": "Banking",
    "TCB": "Banking",  "VPB": "Banking",  "ACB": "Banking",
    "CTG": "Banking",  "HDB": "Banking",  "STB": "Banking",
    "FPT": "Technology",
    "VIC": "Real Estate", "VHM": "Real Estate", "NVL": "Real Estate",
    "PDR": "Real Estate",
    "HPG": "Materials",  "HSG": "Materials",
    "VNM": "Consumer",   "SAB": "Consumer",   "MSN": "Consumer",
    "MWG": "Consumer",
    "GAS": "Energy",     "PLX": "Energy",
    "REE": "Industrials","PVT": "Industrials",
    "SSI": "Finance",    "VND": "Finance",
    "VJC": "Industrials",
}

COMPANY_MAP: dict[str, str] = {
    "VCB": "Vietcombank",        "BID": "BIDV",
    "MBB": "MB Bank",            "TCB": "Techcombank",
    "VPB": "VPBank",             "ACB": "Asia Commercial Bank",
    "CTG": "VietinBank",         "HDB": "HDBank",
    "STB": "Sacombank",          "FPT": "FPT Corporation",
    "VIC": "Vingroup",           "VHM": "Vinhomes",
    "NVL": "Novaland",           "PDR": "Phat Dat Real Estate",
    "HPG": "Hoa Phat Group",     "HSG": "Hoa Sen Group",
    "VNM": "Vinamilk",           "SAB": "Sabeco",
    "MSN": "Masan Group",        "MWG": "Mobile World",
    "GAS": "PetroVietnam Gas",   "PLX": "Petrolimex",
    "REE": "REE Corporation",    "PVT": "PV Trans",
    "SSI": "SSI Securities",     "VND": "VNDirect",
    "VJC": "Vietjet Air",
}


# ── Technical indicator helpers ───────────────────────────────────────────────

def _rsi(close: np.ndarray, period: int = 14) -> float:
    """Compute RSI (14-period) from close price array."""
    if len(close) < period + 1:
        return 50.0
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _ema(series: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    ema = np.empty_like(series)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = alpha * series[i] + (1 - alpha) * ema[i - 1]
    return ema


def _macd_signal(close: np.ndarray) -> float:
    """MACD line = EMA(12) - EMA(26). Normalised by price."""
    if len(close) < 26:
        return 0.0
    macd_line = _ema(close, 12) - _ema(close, 26)
    last_price = close[-1] if close[-1] != 0 else 1.0
    return round(float(macd_line[-1]) / last_price, 6)


def _normalise_cross_section(values: list[float], invert: bool = False) -> list[float]:
    """Rank-based normalisation → [0, 1]. invert=True for lower-is-better metrics."""
    arr = np.array(values, dtype=float)
    if len(arr) <= 1:
        return [0.5] * len(arr)
    ranks = arr.argsort().argsort().astype(float)
    normed = ranks / (len(arr) - 1)
    if invert:
        normed = 1.0 - normed
    return normed.tolist()


def _winsorise(values: list[float], pct: float = 0.05) -> list[float]:
    """Clip extreme values at pct and 1-pct percentiles."""
    if not values:
        return values
    lo, hi = np.percentile(values, pct * 100), np.percentile(values, (1 - pct) * 100)
    return [float(np.clip(v, lo, hi)) for v in values]


# ── Main adapter ──────────────────────────────────────────────────────────────

class VnstockAdapter:
    """Fetch live Vietnamese market data via vnstock 4.x and derive MarketSnapshot fields."""

    def __init__(self, tickers: list[str] | None = None, history_days: int = 90) -> None:
        from app.core.config import get_settings
        settings = get_settings()
        if tickers is None:
            tickers = [t.strip() for t in settings.vnstock_tickers.split(",") if t.strip()]
        self.tickers = tickers
        self.history_days = history_days

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch(self) -> list[Any]:  # list[MarketSnapshot]
        """Fetch and return MarketSnapshot objects for all tickers.

        Returns an empty list if vnstock >= 4.0 is not installed.
        Individual ticker failures are logged and skipped.
        """
        try:
            from vnstock.api.quote import Quote
            from vnstock.api.financial import Finance
        except ImportError:
            logger.warning("vnstock >= 4.0 not installed — pip install -U vnstock")
            return []

        today = date.today().strftime("%Y-%m-%d")
        start = (date.today() - timedelta(days=self.history_days + 10)).strftime("%Y-%m-%d")

        raw_records: list[dict] = []
        for ticker in self.tickers:
            try:
                record = self._fetch_ticker(Quote, Finance, ticker, start, today)
                if record:
                    raw_records.append(record)
                time.sleep(0.5)  # basic rate limiting
            except Exception as exc:
                logger.debug("Skipping %s: %s", ticker, exc)

        if not raw_records:
            logger.warning("vnstock returned no usable records")
            return []

        return self._build_snapshots(raw_records)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fetch_ticker(self, Quote, Finance, ticker: str, start: str, end: str) -> dict | None:
        """Fetch OHLCV history and financial ratios for a single ticker (new API)."""
        q = Quote(symbol=ticker, source="VCI")

        # ── Price history ──────────────────────────────────────────────────
        hist = q.history(start=start, end=end, interval="1D")
        if hist is None or hist.empty or len(hist) < 20:
            logger.debug("%s: insufficient price history (%d rows)", ticker,
                         0 if hist is None or hist.empty else len(hist))
            return None

        # vnstock 4.x 'time' column, prices in thousands VND (same as v3)
        hist = hist.copy()
        if "time" in hist.columns:
            hist = hist.rename(columns={"time": "date"})

        close = hist["close"].astype(float).to_numpy()
        volume = hist["volume"].astype(float).to_numpy() if "volume" in hist.columns else np.ones(len(hist))
        # Prices are in thousands VND: close=70.8 → 70,800 VND
        price_vnd = float(close[-1]) * 1_000

        # ── Technical signals ──────────────────────────────────────────────
        returns_30d = self._log_returns(close[-31:])
        momentum_raw = float(close[-1] / close[-31] - 1) if len(close) >= 31 else 0.0
        volatility_ann = float(np.std(returns_30d, ddof=1) * math.sqrt(252)) if len(returns_30d) > 1 else 0.25
        rsi = _rsi(close)
        macd = _macd_signal(close)
        avg_vol_30d = float(np.mean(volume[-30:])) if len(volume) >= 30 else float(np.mean(volume))

        # ── Financial ratios ───────────────────────────────────────────────
        pe, pb, roe, debt_to_equity, rev_growth = self._fetch_ratios(Finance, ticker)

        return {
            "ticker": ticker,
            "price_vnd": price_vnd,
            "momentum_raw": momentum_raw,
            "volatility_ann": max(volatility_ann, 0.05),  # floor at 5%
            "rsi": rsi,
            "macd": macd,
            "avg_vol_30d": avg_vol_30d,
            "pe": pe,
            "pb": pb,
            "roe": roe,
            "debt_to_equity": debt_to_equity,
            "rev_growth": rev_growth,
        }

    def _fetch_ratios(self, Finance, ticker: str) -> tuple[float, float, float, float, float]:
        """Return (pe, pb, roe, debt_to_equity, rev_growth). Defaults to neutral values on error.

        Uses the new Finance class from vnstock.api.financial.
        The ratio DataFrame is in wide format (item_en × period) and must be
        transposed to get the latest quarter values.
        """
        try:
            f = Finance(symbol=ticker, source="VCI")
            ratio_wide = f.ratio(period="quarter", lang="en")
            if ratio_wide is None or ratio_wide.empty:
                raise ValueError("empty ratio response")

            # Pivot wide → latest quarter row
            period_cols = [c for c in ratio_wide.columns if isinstance(c, str) and "-Q" in c]
            if not period_cols:
                raise ValueError("no period columns found")

            # Use the most recent quarter (last column)
            latest_col = period_cols[-1]
            ratio_indexed = ratio_wide.set_index("item_en")[latest_col]

            def _safe(item_en: str, default: float) -> float:
                try:
                    val = ratio_indexed.get(item_en, default)
                    v = float(val)
                    import math as _math
                    return v if not _math.isnan(v) else default
                except (TypeError, ValueError):
                    return default

            pe = _safe("P/E", 15.0)
            pb = _safe("P/B", 2.0)
            # ROE (%): new API returns decimal already (0.2311 = 23.11% ROE)
            roe = _safe("ROE (%)", 0.12)
            debt_to_equity = _safe("Debt to Equity", 1.0)
            if debt_to_equity == 1.0:  # try alternate column name
                debt_to_equity = _safe("Debt/Equity", 1.0)
            rev_growth = _safe("Revenue Growth", 0.08)

            return (
                max(pe, 1.0),
                max(pb, 0.1),
                max(min(roe, 1.0), 0.0),
                max(debt_to_equity, 0.0),
                rev_growth,
            )
        except Exception as exc:
            logger.debug("%s: ratio fetch failed (%s), using defaults", ticker, exc)
            return 15.0, 2.0, 0.12, 1.0, 0.08

    @staticmethod
    def _log_returns(close: np.ndarray) -> np.ndarray:
        if len(close) < 2:
            return np.array([0.0])
        return np.log(close[1:] / np.where(close[:-1] == 0, 1.0, close[:-1]))

    def _build_snapshots(self, records: list[dict]) -> list[Any]:
        """Cross-sectionally normalise all signals and build MarketSnapshot list."""
        from app.domain.schemas import MarketSnapshot

        n = len(records)
        if n == 0:
            return []

        # ── Cross-sectional normalisation ──────────────────────────────────
        momentum_norm = _normalise_cross_section(
            _winsorise([r["momentum_raw"] for r in records])
        )
        # volatility kept as raw annualised vol (not normalised — used directly)
        rsi_norm = [r["rsi"] / 100.0 for r in records]  # RSI already 0-100

        # quality: ROE (higher better) + low leverage (lower better) → normalise
        roe_norm = _normalise_cross_section([r["roe"] for r in records])
        lev_norm = _normalise_cross_section([r["debt_to_equity"] for r in records], invert=True)
        quality_norm = [0.6 * roe_norm[i] + 0.4 * lev_norm[i] for i in range(n)]

        # value: low P/E = better value (invert rank)
        pe_norm = _normalise_cross_section([r["pe"] for r in records], invert=True)
        pb_norm = _normalise_cross_section([r["pb"] for r in records], invert=True)
        value_norm = [0.5 * pe_norm[i] + 0.5 * pb_norm[i] for i in range(n)]

        # sentiment: momentum + RSI composite
        sentiment_norm = [
            0.6 * momentum_norm[i] + 0.4 * rsi_norm[i]
            for i in range(n)
        ]

        # liquidity: log volume, cross-sectionally normalised
        log_vols = [math.log(max(r["avg_vol_30d"], 1.0)) for r in records]
        liquidity_norm = _normalise_cross_section(log_vols)

        # expected_return: simple cross-sectional from momentum + ROE
        exp_ret_raw = [0.5 * momentum_norm[i] + 0.5 * roe_norm[i] for i in range(n)]
        # Scale to a realistic return range: 5% - 25%
        exp_ret = [0.05 + 0.20 * v for v in exp_ret_raw]

        # beta: proxy from volatility relative to universe median
        vols = [r["volatility_ann"] for r in records]
        median_vol = float(np.median(vols)) if vols else 0.25
        betas = [round(v / max(median_vol, 0.01), 2) for v in vols]

        now_str = datetime.now(timezone.utc).isoformat()
        snapshots: list[MarketSnapshot] = []
        for i, r in enumerate(records):
            ticker = r["ticker"]
            snapshots.append(MarketSnapshot(
                symbol=ticker,
                company=COMPANY_MAP.get(ticker, ticker),
                sector=SECTOR_MAP.get(ticker, "Other"),
                price=round(r["price_vnd"], 0),
                expected_return=round(exp_ret[i], 4),
                volatility=round(r["volatility_ann"], 4),
                momentum=round(momentum_norm[i], 4),
                quality=round(quality_norm[i], 4),
                value=round(value_norm[i], 4),
                sentiment=round(sentiment_norm[i], 4),
                liquidity_score=round(liquidity_norm[i], 4),
                pe_ratio=round(r["pe"], 2),
                pb_ratio=round(r["pb"], 2),
                roe=round(r["roe"], 4),
                debt_to_equity=round(r["debt_to_equity"], 4),
                revenue_growth=round(r["rev_growth"], 4),
                rsi=round(r["rsi"], 2),
                macd_signal=round(r["macd"], 6),
                beta=betas[i],
                data_age_hours=0.0,
                is_stale=False,
            ))

        logger.info(
            "VnstockAdapter: fetched %d/%d tickers successfully (as of %s)",
            len(snapshots), len(self.tickers), now_str[:10],
        )
        return snapshots
