from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application
    app_name: str = "VTH-StockPilot"
    app_version: str = "2.0.0"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    frontend_dir: Path = Path(__file__).resolve().parents[3] / "frontend"

    # Database
    database_url: str = "sqlite+aiosqlite:///./vth_stockpilot.db"
    database_url_sync: str = "sqlite:///./vth_stockpilot.db"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "stock-ranking-model"
    model_version: str = "baseline-vn-2026-01"
    # Set to True to use the trained MLflow model inside the forecast node.
    # When False, the heuristic linear formula is used instead.
    use_mlflow_model: bool = False
    # Model Registry stage to load: "Production", "Staging", or "None" (latest)
    mlflow_model_stage: str = "Production"

    # LLM (optional — used only for explanation node)
    llm_provider: str = "template"          # "template" | "openai" | "gemini"
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Market configuration — Vietnamese stock market
    market: str = "VN"                      # "VN" = Vietnam (HOSE/HNX/UPCOM)
    market_data_source: str = "demo"        # "demo" | "vnstock"
    vnstock_market: str = "HOSE"            # Primary exchange: HOSE | HNX | UPCOM
    risk_free_rate: float = 0.045           # 4.5% — Vietnamese government bond yield (5Y)
    currency: str = "VND"

    # vnstock tickers — comma-separated list of HOSE symbols to fetch
    vnstock_tickers: str = (
        "VCB,BID,MBB,TCB,VPB,ACB,FPT,VIC,VHM,HPG,VNM,SAB,GAS,MSN,REE"
    )
    # Market data cache TTL (seconds). 14400 = 4 hours.
    # VN market closes at 15:00 ICT; data won't change until next session.
    market_data_cache_ttl: int = 14400
    # Number of calendar days of price history to fetch for signal computation
    price_history_days: int = 90

    # ETL — schedule in ICT (UTC+7); 09:00 ICT = 02:00 UTC
    etl_schedule_cron: str = "0 2 * * *"   # 02:00 UTC = 09:00 ICT daily
    use_real_market_data: bool = False      # True → vnstock live; False → demo universe

    # Training — forward return horizon for label_builder (trading days)
    training_horizon_days: int = 21         # 21 trading days ≈ 1 calendar month

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="STOCK_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
