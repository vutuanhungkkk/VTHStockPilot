"""WebSocket connection manager with pub/sub support.

Channels:
- backtest:{job_id}     — progress of a backtest job
- market               — market regime updates (broadcast)
- recommendations:{id} — recommendation pipeline progress
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi import WebSocket


class WebSocketManager:
    def __init__(self) -> None:
        # channel_id → set of active WebSocket connections
        self._channels: dict[str, set[WebSocket]] = defaultdict(set)

    # ── Connection lifecycle ───────────────────────────────────────────────────

    async def connect(self, ws: WebSocket, channel: str) -> None:
        await ws.accept()
        self._channels[channel].add(ws)

    def disconnect(self, ws: WebSocket, channel: str) -> None:
        self._channels[channel].discard(ws)
        if not self._channels[channel]:
            del self._channels[channel]

    # ── Messaging ──────────────────────────────────────────────────────────────

    async def send(self, ws: WebSocket, message: dict) -> None:
        """Send a message to a single WebSocket connection."""
        try:
            await ws.send_json(message)
        except Exception:
            pass

    async def broadcast(self, channel: str, message: dict) -> None:
        """Send to all connections subscribed to `channel`."""
        dead: list[WebSocket] = []
        for ws in list(self._channels.get(channel, [])):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._channels[channel].discard(ws)

    async def progress(self, channel: str, stage: str, percent: int, meta: dict | None = None) -> None:
        """Helper for pipeline progress updates."""
        msg: dict = {"type": "progress", "stage": stage, "percent": percent}
        if meta:
            msg["meta"] = meta
        await self.broadcast(channel, msg)

    async def send_result(self, channel: str, data: dict) -> None:
        await self.broadcast(channel, {"type": "result", "data": data})

    async def send_error(self, channel: str, message: str) -> None:
        await self.broadcast(channel, {"type": "error", "message": message})

    def active_channels(self) -> list[str]:
        return list(self._channels.keys())

    def subscriber_count(self, channel: str) -> int:
        return len(self._channels.get(channel, set()))


# Singleton — shared across all routers via dependency injection or direct import
ws_manager = WebSocketManager()
