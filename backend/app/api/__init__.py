"""Updated API router (backward-compatible aggregation of sub-routers)."""
from app.api.recommendations import router as reco_router  # noqa: F401
from app.api.portfolio import router as portfolio_router  # noqa: F401
from app.api.experiments import router as experiments_router  # noqa: F401

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from app.core.config import get_settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {"status": "healthy", "service": settings.app_name, "version": settings.app_version}


@router.get("/metadata")
async def metadata() -> dict:
    from app.services.market_data import MarketDataService
    universe = MarketDataService().get_universe()
    sectors = sorted({a.sector for a in universe})
    return {
        "sectors": sectors,
        "risk_levels": ["conservative", "balanced", "growth"],
        "model_version": get_settings().model_version,
        "universe_size": len(universe),
    }


@router.get("/market/status")
async def market_status() -> dict[str, Any]:
    """Return current market data source status.

    Response fields:
        data_source     — "live" | "demo"
        data_as_of      — ISO date of the data
        universe_size   — number of symbols available
        cache_valid     — whether an in-memory cache is active
        cache_expires_at — UTC ISO datetime when cache expires (null if no cache)
        cache_age_seconds — seconds since last cache refresh (null if no cache)
        vnstock_enabled — True if live data is configured
        vnstock_available — True if vnstock3 package is installed
    """
    from app.services.market_data import MarketDataService
    svc = MarketDataService()
    settings = get_settings()

    universe = svc.get_universe()
    expires_at = MarketDataService.cache_expires_at()
    cache_time = MarketDataService._cache_time

    vnstock_available = False
    try:
        import vnstock  # noqa: F401
        vnstock_available = True
    except ImportError:
        pass

    vnstock_enabled = settings.use_real_market_data or settings.market_data_source == "vnstock"

    cache_age: float | None = None
    if cache_time is not None:
        cache_age = round((datetime.now(timezone.utc) - cache_time).total_seconds(), 1)

    return {
        "data_source": MarketDataService.data_source(),
        "data_as_of": MarketDataService.data_as_of(),
        "universe_size": len(universe),
        "cache_valid": MarketDataService._cache is not None,
        "cache_expires_at": expires_at.isoformat() if expires_at else None,
        "cache_age_seconds": cache_age,
        "cache_ttl_seconds": settings.market_data_cache_ttl,
        "vnstock_enabled": vnstock_enabled,
        "vnstock_available": vnstock_available,
        "tickers_configured": [t.strip() for t in settings.vnstock_tickers.split(",")],
    }


@router.post("/market/refresh")
async def market_refresh() -> dict[str, Any]:
    """Force invalidate the market data cache and re-fetch live data.

    Useful after market close or when you suspect stale data.
    Returns the new market/status response.
    """
    from app.services.market_data import MarketDataService
    MarketDataService.invalidate_cache()
    # Trigger fresh load
    svc = MarketDataService()
    svc.get_universe()
    return await market_status()

