# app/routes/websocket.py

from fastapi import APIRouter, WebSocket, Depends, WebSocketDisconnect
from starlette.websockets import WebSocketState
from ..dependencies import get_yolo_model, get_deepsort_model, get_deepface_model
from ..processing.video_processing import process_video_frame
import logging
import asyncio

logging.basicConfig(level=logging.DEBUG)

router = APIRouter()

@router.websocket("/ws/video")
async def websocket_endpoint(
    websocket: WebSocket,
    yolo_model=Depends(get_yolo_model),
    deepsort_model=Depends(get_deepsort_model),
    deepface_model=Depends(get_deepface_model)
):
    await websocket.accept()  # Ensure accept is called

    async def send_keepalive():
        while True:
            try:
                await asyncio.sleep(10)
                await websocket.send_text('{"type": "ping"}')
            except Exception as e:
                logging.error(f"Error sending keepalive: {e}")
                break

    keepalive_task = asyncio.create_task(send_keepalive())

    try:
        while True:
            try:
                data = await websocket.receive_bytes()
                # logging.debug(f"Received data of length: {len(data)}")
                response = await process_video_frame(data, yolo_model, deepsort_model, deepface_model)
                await websocket.send_bytes(response)
            except WebSocketDisconnect:
                logging.info("Client disconnected")
                break
            except Exception as e:
                logging.error(f"Error processing frame: {e}")

    except WebSocketDisconnect:
        logging.info("Client disconnected")
    finally:
        keepalive_task.cancel()
        if websocket.application_state != WebSocketState.DISCONNECTED:
            try:
                await websocket.close()
            except RuntimeError as e:
                logging.error(f"Error closing websocket: {e}")
