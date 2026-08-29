"""Legacy backtest route — kept for backward compatibility with existing frontend."""
import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.api.websocket_manager import ws_manager
from app.core.config import get_settings
from app.domain.schemas import BacktestRequest, BacktestResponse
from app.services.backtest import BacktestService

router = APIRouter(tags=["Backtest"])
backtests = BacktestService()


@router.post("/backtests", response_model=BacktestResponse)
async def create_backtest(request: BacktestRequest) -> BacktestResponse:
    return await asyncio.to_thread(backtests.run, request)


@router.websocket("/ws/backtest/{job_id}")
async def backtest_progress_stream(websocket: WebSocket, job_id: str) -> None:
    """WebSocket endpoint for streaming backtest progress."""
    channel = f"backtest:{job_id}"
    await ws_manager.connect(websocket, channel)
    try:
        raw = await websocket.receive_json()
        request = BacktestRequest.model_validate(raw)
        await ws_manager.progress(channel, "Initialising backtest engine", 10)
        await asyncio.sleep(0.1)
        await ws_manager.progress(channel, "Simulating portfolio returns", 40)
        result = await asyncio.to_thread(backtests.run, request)
        await ws_manager.progress(channel, "Computing risk metrics", 80)
        await asyncio.sleep(0.05)
        await ws_manager.progress(channel, "Generating charts data", 95)
        await ws_manager.send_result(channel, result.model_dump(mode="json"))
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
