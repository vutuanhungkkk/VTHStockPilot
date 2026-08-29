"""Recommendation API router."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.api.websocket_manager import ws_manager
from app.core.config import get_settings
from app.domain.schemas import InvestorProfile, RecommendationResponse
from app.services.market_data import MarketDataService
from app.workflows.recommendation_graph import RecommendationGraph, build_response

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])

_graph = RecommendationGraph()
_market = MarketDataService()


@router.post("", response_model=RecommendationResponse)
async def create_recommendations(profile: InvestorProfile) -> RecommendationResponse:
    """Run the full 9-node recommendation workflow synchronously."""
    settings = get_settings()
    try:
        universe = _market.get_universe()
        state = await _graph.run(profile, universe)
        errors = state.get("errors", [])
        if errors:
            raise ValueError(errors[0])
        rec_id = str(uuid.uuid4())
        return build_response(state, profile, settings.model_version, _market.data_as_of(), rec_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.websocket("/ws/{channel_id}")
async def recommendation_stream(websocket: WebSocket, channel_id: str) -> None:
    """
    WebSocket endpoint for recommendation pipeline with progress events.

    Message flow:
    1. Client → sends InvestorProfile JSON
    2. Server → sends {type: "progress", stage, percent} per pipeline node
    3. Server → sends {type: "result", data: RecommendationResponse}
    """
    settings = get_settings()
    channel = f"recommendations:{channel_id}"
    await ws_manager.connect(websocket, channel)
    try:
        raw = await websocket.receive_json()
        profile = InvestorProfile.model_validate(raw)
        universe = _market.get_universe()

        stage_labels = [
            ("Validating market data", 12),
            ("Detecting market regime", 25),
            ("Computing fundamental scores", 38),
            ("Forecasting returns", 50),
            ("Applying investor preferences", 62),
            ("Enforcing risk constraints", 73),
            ("Ranking opportunities", 83),
            ("Optimising portfolio weights", 92),
            ("Generating explanations", 98),
        ]

        # Emit progress for each node upfront (approximated)
        # Real node progress requires instrumenting the graph — out of scope here
        import asyncio
        async def progress_emitter():
            for stage, pct in stage_labels:
                await ws_manager.progress(channel, stage, pct)
                await asyncio.sleep(0.12)

        import asyncio as _asyncio
        progress_task = _asyncio.create_task(progress_emitter())
        state = await _graph.run(profile, universe)
        await progress_task

        errors = state.get("errors", [])
        if errors:
            await ws_manager.send_error(channel, errors[0])
            return

        rec_id = str(uuid.uuid4())
        response = build_response(state, profile, settings.model_version, _market.data_as_of(), rec_id)
        await ws_manager.send_result(channel, response.model_dump(mode="json"))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        await ws_manager.send_error(channel, str(exc))
    finally:
        ws_manager.disconnect(websocket, channel)
        try:
            await websocket.close()
        except Exception:
            pass
