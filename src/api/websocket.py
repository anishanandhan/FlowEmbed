"""
WebSocket — Real-time connections for live classification and drift alerts.

The dashboard connects via WebSocket to receive:
1. Live classification results as flows are processed
2. Drift detection alerts
3. Updated embedding coordinates
"""

import asyncio
import json
import logging
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

ws_router = APIRouter()

# Connected clients
live_clients: Set[WebSocket] = set()
drift_clients: Set[WebSocket] = set()


@ws_router.websocket("/live")
async def ws_live_classification(websocket: WebSocket):
    """
    WebSocket for live classification feed.
    Clients receive real-time flow classification results.
    """
    await websocket.accept()
    live_clients.add(websocket)
    logger.info(f"Live WebSocket client connected. Total: {len(live_clients)}")

    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Client can send control messages if needed
            logger.debug(f"Received from client: {data}")
    except WebSocketDisconnect:
        live_clients.discard(websocket)
        logger.info(f"Live WebSocket client disconnected. Total: {len(live_clients)}")


@ws_router.websocket("/drift")
async def ws_drift_alerts(websocket: WebSocket):
    """
    WebSocket for drift detection alerts.
    Clients receive alerts when concept drift is detected.
    """
    await websocket.accept()
    drift_clients.add(websocket)
    logger.info(f"Drift WebSocket client connected. Total: {len(drift_clients)}")

    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received from drift client: {data}")
    except WebSocketDisconnect:
        drift_clients.discard(websocket)
        logger.info(f"Drift WebSocket client disconnected. Total: {len(drift_clients)}")


async def broadcast_classification(result: dict):
    """
    Broadcast a classification result to all connected live clients.

    Args:
        result: Dict with classification info (class, confidence, etc.)
    """
    if not live_clients:
        return

    message = json.dumps(result)
    disconnected = set()

    for client in live_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)

    live_clients.difference_update(disconnected)


async def broadcast_drift_alert(alert: dict):
    """
    Broadcast a drift alert to all connected drift clients.
    """
    if not drift_clients:
        return

    message = json.dumps(alert)
    disconnected = set()

    for client in drift_clients:
        try:
            await client.send_text(message)
        except Exception:
            disconnected.add(client)

    drift_clients.difference_update(disconnected)
