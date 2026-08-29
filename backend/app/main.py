"""FastAPI application entry point."""
from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fastapi import WebSocket

from app.api import router
from app.api.recommendations import router as reco_router, recommendation_stream
from app.api.portfolio import router as portfolio_router
from app.api.experiments import router as experiments_router

from app.api.routes import router as legacy_backtest_router
from app.core.config import get_settings
from app.db.session import create_all_tables

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Explainable stock recommendation and portfolio intelligence platform. "
        "Powered by a 9-node LangGraph workflow, cross-sectional ML ranking, "
        "and MLflow experiment tracking."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prefix = settings.api_prefix

# Core utility routes
app.include_router(router, prefix=prefix)

# Feature routers
app.include_router(reco_router, prefix=prefix)
app.include_router(portfolio_router, prefix=prefix)
app.include_router(experiments_router, prefix=prefix)


# Backtest (legacy route kept for backward compat)
app.include_router(legacy_backtest_router, prefix=prefix)


@app.on_event("startup")
async def startup() -> None:
    await create_all_tables()


# ── WebSocket alias routes ─────────────────────────────────────────────────
# The frontend (older cached JS) calls /api/v1/ws/recommendations/{id}
# The canonical backend route is    /api/v1/recommendations/ws/{id}
# Both are registered so the app works regardless of browser cache state.
@app.websocket("/api/v1/ws/recommendations/{channel_id}")
async def reco_ws_alias(websocket: WebSocket, channel_id: str) -> None:
    await recommendation_stream(websocket, channel_id)


# Serve frontend
if settings.frontend_dir.exists():
    for sub in ("css", "js", "assets", "public"):
        p = settings.frontend_dir / sub
        if p.exists():
            app.mount(f"/{sub}", StaticFiles(directory=p), name=sub)

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(settings.frontend_dir / "index.html")
