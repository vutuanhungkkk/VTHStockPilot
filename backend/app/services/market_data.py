"""Market data service — live Vietnamese HOSE data via vnstock3 with demo fallback.

Priority order for get_universe():
    1. In-memory cache (TTL = settings.market_data_cache_ttl seconds)
    2. Live fetch via VnstockAdapter (vnstock3)
    3. Static demo universe (hardcoded blue-cap HOSE stocks)

The cache is a class-level attribute so it is shared across all instances
within a single process (singleton-like behaviour without DI overhead).

Setting STOCK_USE_REAL_MARKET_DATA=true in the environment enables live fetch.
Alternatively set STOCK_MARKET_DATA_SOURCE=vnstock.
"""
from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timezone
from typing import ClassVar

from app.domain.schemas import MarketSnapshot

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()


class MarketDataService:
    """Market data adapter with live/demo hybrid strategy and in-memory cache."""

    # ── Class-level cache (shared across instances in the same process) ────────
    _cache: ClassVar[list[MarketSnapshot] | None] = None
    _cache_time: ClassVar[datetime | None] = None
    _cache_source: ClassVar[str] = "demo"   # "live" | "demo"

    # ── Demo universe fallback ─────────────────────────────────────────────────
    # Vietnamese HOSE blue-cap universe
    # Fields: symbol, company, sector, price(VND), exp_ret, vol,
    #         momentum, quality, value, sentiment, liquidity,
    #         pe, pb, roe, d/e, rev_growth, rsi, macd, beta
    _UNIVERSE = [
        # ── Banking ────────────────────────────────────────────────────────
        ("VCB", "Vietcombank",               "Banking",         91500, .128, .198, .71, .92, .72, .68, .97, 14.2, 2.8, .20,  9.80, .11, 56, 0.3, 0.88),
        ("BID", "BIDV",                      "Banking",         48200, .114, .213, .64, .85, .78, .61, .95,  9.8, 1.9, .15, 10.10, .09, 51, 0.1, 0.82),
        ("MBB", "MB Bank",                   "Banking",         26800, .141, .237, .76, .83, .69, .65, .94, 10.5, 1.8, .18,  7.20, .14, 60, 0.5, 0.91),
        ("TCB", "Techcombank",               "Banking",         54300, .136, .244, .74, .87, .65, .63, .93, 12.1, 2.1, .19,  5.40, .13, 58, 0.4, 0.94),
        ("VPB", "VPBank",                    "Banking",         23700, .152, .281, .81, .78, .61, .70, .91, 11.3, 1.6, .16,  8.30, .17, 63, 0.7, 1.08),
        ("ACB", "Asia Commercial Bank",      "Banking",         29400, .122, .218, .68, .84, .74, .60, .92, 10.2, 2.0, .17,  6.10, .10, 53, 0.2, 0.85),
        # ── Technology ─────────────────────────────────────────────────────
        ("FPT", "FPT Corporation",           "Technology",     138000, .189, .254, .87, .91, .58, .79, .96, 22.3, 4.1, .22,  0.45, .24, 67, 1.1, 1.02),
        # ── Real Estate ────────────────────────────────────────────────────
        ("VIC", "Vingroup",                  "Real Estate",     54200, .098, .318, .55, .74, .62, .53, .89, 38.1, 2.9, .06,  2.10, .05, 43, -0.3, 1.22),
        ("VHM", "Vinhomes",                  "Real Estate",     41800, .112, .289, .61, .79, .67, .57, .90, 18.4, 2.4, .13,  0.92, .08, 48, -0.1, 1.15),
        # ── Steel & Materials ──────────────────────────────────────────────
        ("HPG", "Hoa Phat Group",            "Materials",       26500, .143, .301, .78, .81, .73, .66, .93, 12.8, 1.7, .14,  0.72, .16, 61,  0.6, 1.18),
        # ── Consumer Staples ───────────────────────────────────────────────
        ("VNM", "Vinamilk",                  "Consumer",        74200, .094, .178, .59, .89, .82, .62, .95, 16.5, 3.8, .25,  0.12, .06, 50,  0.0, 0.68),
        ("SAB", "Sabeco",                    "Consumer",       178000, .088, .192, .54, .86, .79, .58, .91, 20.2, 4.3, .21,  0.08, .04, 48, -0.1, 0.72),
        # ── Energy ─────────────────────────────────────────────────────────
        ("GAS", "PetroVietnam Gas",          "Energy",         104200, .107, .214, .63, .84, .76, .56, .92, 13.7, 2.6, .19,  0.18, .07, 52,  0.2, 0.78),
        # ── Diversified / Conglomerate ─────────────────────────────────────
        ("MSN", "Masan Group",               "Consumer",        79500, .118, .267, .67, .77, .64, .61, .88, 24.6, 2.3, .10,  1.45, .12, 55,  0.3, 1.05),
        # ── Industrials ────────────────────────────────────────────────────
        ("REE", "REE Corporation",           "Industrials",     72800, .104, .231, .62, .80, .77, .55, .87, 11.4, 1.6, .15,  0.38, .08, 51,  0.1, 0.80),
    ]

    # ── Public interface ───────────────────────────────────────────────────────

    def get_universe(self) -> list[MarketSnapshot]:
        """Return the current market universe.

        Tries live data first (if enabled), falls back to demo.
        Thread-safe via a module-level lock.
        """
        with _LOCK:
            if self._is_cache_valid():
                return MarketDataService._cache  # type: ignore[return-value]

            live = self._try_fetch_live()
            if live:
                MarketDataService._cache = live
                MarketDataService._cache_time = datetime.now(timezone.utc)
                MarketDataService._cache_source = "live"
                logger.info("MarketDataService: live universe loaded (%d symbols)", len(live))
                return live

            # Demo fallback — do NOT cache so we retry live on next call
            logger.info("MarketDataService: using demo universe")
            MarketDataService._cache_source = "demo"
            return self._demo_universe()

    @classmethod
    def data_source(cls) -> str:
        """Return 'live' or 'demo'."""
        return cls._cache_source

    @classmethod
    def cache_expires_at(cls) -> datetime | None:
        """Return the UTC datetime when the current cache expires, or None."""
        if cls._cache_time is None:
            return None
        from app.core.config import get_settings
        ttl = get_settings().market_data_cache_ttl
        from datetime import timedelta
        return cls._cache_time + timedelta(seconds=ttl)

    @classmethod
    def invalidate_cache(cls) -> None:
        """Force cache invalidation — next call will re-fetch live data."""
        with _LOCK:
            cls._cache = None
            cls._cache_time = None
            cls._cache_source = "demo"
        logger.info("MarketDataService: cache invalidated")

    @staticmethod
    def data_as_of() -> str:
        """Return ISO date string representing when data was last fetched."""
        if MarketDataService._cache_time is not None:
            return MarketDataService._cache_time.date().isoformat()
        return date.today().isoformat()

    # ── Private helpers ────────────────────────────────────────────────────────

    @classmethod
    def _is_cache_valid(cls) -> bool:
        if cls._cache is None or cls._cache_time is None:
            return False
        from app.core.config import get_settings
        from datetime import timedelta
        ttl = get_settings().market_data_cache_ttl
        age = (datetime.now(timezone.utc) - cls._cache_time).total_seconds()
        return age < ttl

    @staticmethod
    def _try_fetch_live() -> list[MarketSnapshot] | None:
        """Attempt to fetch live data. Returns None on any failure."""
        from app.core.config import get_settings
        settings = get_settings()
        if not settings.use_real_market_data and settings.market_data_source != "vnstock":
            return None  # live fetch not enabled
        try:
            from app.etl.vnstock_adapter import VnstockAdapter
            tickers = [t.strip() for t in settings.vnstock_tickers.split(",") if t.strip()]
            adapter = VnstockAdapter(tickers=tickers, history_days=settings.price_history_days)
            result = adapter.fetch()
            return result if result else None
        except Exception as exc:
            logger.warning("Live market data fetch failed: %s — falling back to demo", exc)
            return None

    def _demo_universe(self) -> list[MarketSnapshot]:
        return [
            MarketSnapshot(
                symbol=r[0], company=r[1], sector=r[2], price=r[3],
                expected_return=r[4], volatility=r[5], momentum=r[6],
                quality=r[7], value=r[8], sentiment=r[9], liquidity_score=r[10],
                pe_ratio=r[11], pb_ratio=r[12], roe=r[13],
                debt_to_equity=r[14], revenue_growth=r[15],
                rsi=r[16], macd_signal=r[17], beta=r[18],
                data_age_hours=0.5, is_stale=False,
            )
            for r in self._UNIVERSE
        ]
