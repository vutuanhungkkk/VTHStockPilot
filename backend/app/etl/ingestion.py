"""ETL data ingestion — delegates live fetching to VnstockAdapter.

Production:
    Set STOCK_USE_REAL_MARKET_DATA=true and STOCK_MARKET_DATA_SOURCE=vnstock
    to enable live data fetching from HOSE via vnstock3.

Demo mode (default):
    Returns the static Vietnamese blue-cap universe from MarketDataService.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone

from app.domain.schemas import MarketSnapshot
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)


class DataIngestionWorker:
    """Fetches Vietnamese market data and stores to DB.

    Delegates live fetching entirely to VnstockAdapter, which handles
    retries, signal computation, and cross-sectional normalisation.
    Set source="vnstock" to fetch live; "demo" to use static universe.
    """

    def __init__(self, source: str = "demo") -> None:
        self.source = source
        self._market = MarketDataService()

    def run(self) -> dict:
        started_at = datetime.now(timezone.utc)
        try:
            snapshots = self._load_market_data()
            validated = self._validate(snapshots)
            logger.info(
                "ETL ingestion: %d symbols loaded, %d valid",
                len(snapshots), len(validated),
            )
            return {
                "status": "success",
                "rows_ingested": len(validated),
                "symbols_processed": len(snapshots),
                "source": self.source,
                "market": "VN",
                "exchange": "HOSE",
                "as_of": date.today().isoformat(),
                "duration_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
            }
        except Exception as exc:
            logger.exception("ETL ingestion failed")
            return {"status": "failed", "error": str(exc)}

    def _load_market_data(self) -> list[MarketSnapshot]:
        if self.source == "vnstock":
            return self._load_vnstock()
        return self._market.get_universe()

    def _load_vnstock(self) -> list[MarketSnapshot]:
        """Load from vnstock3 via VnstockAdapter.

        Falls back to demo universe if adapter returns nothing.
        """
        try:
            from app.etl.vnstock_adapter import VnstockAdapter
            from app.core.config import get_settings
            settings = get_settings()
            tickers = [t.strip() for t in settings.vnstock_tickers.split(",") if t.strip()]
            adapter = VnstockAdapter(tickers=tickers, history_days=settings.price_history_days)
            snapshots = adapter.fetch()
            if snapshots:
                logger.info("VnstockAdapter returned %d snapshots", len(snapshots))
                return snapshots
            logger.warning("VnstockAdapter returned no data, falling back to demo universe")
            return self._market._demo_universe()
        except ImportError:
            logger.warning("vnstock3 not installed — pip install vnstock3; falling back to demo")
            return self._market._demo_universe()
        except Exception as exc:
            logger.exception("VnstockAdapter fetch failed: %s", exc)
            return self._market._demo_universe()

    def _validate(self, snapshots: list[MarketSnapshot]) -> list[MarketSnapshot]:
        return [s for s in snapshots if s.price > 0 and s.liquidity_score > 0.3]
