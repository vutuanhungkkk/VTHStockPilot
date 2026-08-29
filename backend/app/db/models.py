"""SQLAlchemy ORM models."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Stock(Base):
    """Point-in-time market snapshot (one row per symbol per ETL run)."""
    __tablename__ = "stocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    company: Mapped[str] = mapped_column(String(100))
    sector: Mapped[str] = mapped_column(String(50))
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    price: Mapped[float] = mapped_column(Float)
    expected_return: Mapped[float] = mapped_column(Float)
    volatility: Mapped[float] = mapped_column(Float)
    momentum: Mapped[float] = mapped_column(Float)
    quality: Mapped[float] = mapped_column(Float)
    value: Mapped[float] = mapped_column(Float)
    sentiment: Mapped[float] = mapped_column(Float)
    liquidity_score: Mapped[float] = mapped_column(Float)
    pe_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    pb_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    roe: Mapped[float] = mapped_column(Float, default=0.0)
    debt_to_equity: Mapped[float] = mapped_column(Float, default=0.0)
    revenue_growth: Mapped[float] = mapped_column(Float, default=0.0)
    rsi: Mapped[float] = mapped_column(Float, default=50.0)
    beta: Mapped[float] = mapped_column(Float, default=1.0)
    is_stale: Mapped[bool] = mapped_column(Boolean, default=False)


class RecommendationRun(Base):
    """Each call to the recommendation workflow."""
    __tablename__ = "recommendation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    model_version: Mapped[str] = mapped_column(String(50))
    market_regime: Mapped[str] = mapped_column(String(30))
    risk_level: Mapped[str] = mapped_column(String(20))
    capital: Mapped[float] = mapped_column(Float)
    profile_json: Mapped[dict] = mapped_column(JSON)
    portfolio_metrics_json: Mapped[dict] = mapped_column(JSON)
    items: Mapped[list["RecommendationRunItem"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class RecommendationRunItem(Base):
    """Individual stock recommendation within a run."""
    __tablename__ = "recommendation_run_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("recommendation_runs.id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    symbol: Mapped[str] = mapped_column(String(20))
    score: Mapped[float] = mapped_column(Float)
    weight: Mapped[float] = mapped_column(Float)
    signals_json: Mapped[dict] = mapped_column(JSON)
    shap_json: Mapped[dict] = mapped_column(JSON, default=dict)
    explanation_text: Mapped[str] = mapped_column(Text, default="")
    run: Mapped["RecommendationRun"] = relationship(back_populates="items")


class SavedPortfolio(Base):
    """User-saved portfolio snapshots."""
    __tablename__ = "saved_portfolios"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    recommendation_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    positions_json: Mapped[dict] = mapped_column(JSON)
    metrics_json: Mapped[dict] = mapped_column(JSON)
    notes: Mapped[str] = mapped_column(Text, default="")



class ETLRun(Base):
    """Audit log for each ETL ingestion."""
    __tablename__ = "etl_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="running")  # running | success | failed
    rows_ingested: Mapped[int] = mapped_column(Integer, default=0)
    symbols_processed: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(50), default="demo")
