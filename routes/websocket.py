from fastapi import APIRouter, WebSocket, Depends, WebSocketDisconnect
from starlette.websockets import WebSocketState
from ..dependencies import get_yolo_model, get_deepsort_model, get_deepface_model, get_lstm_model
from ..processing.video_processing import process_video_frame
import logging
import asyncio
import json

logging.basicConfig(level=logging.INFO)

router = APIRouter()

@router.websocket("/ws/video")
async def websocket_endpoint(
    websocket: WebSocket,
    yolo_model=Depends(get_yolo_model),
    deepsort_model=Depends(get_deepsort_model),
    deepface_model=Depends(get_deepface_model),
    lstm_model=Depends(get_lstm_model),
):
    await websocket.accept()
    logging.info("WebSocket connection accepted")
    use_lstm = False

    async def send_keepalive():
        while True:
            try:
                await asyncio.sleep(10)
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text('{"type": "ping"}')
            except Exception as e:
                logging.error(f"Error sending keepalive: {e}")
                break

    keepalive_task = asyncio.create_task(send_keepalive())

    try:
        while True:
            try:
                message = await websocket.receive_text()
                try:
                    # Attempt to parse as JSON control message
                    control_data = json.loads(message)
                    use_lstm = control_data.get("useLSTM", False)
                except json.JSONDecodeError:
                    # If not JSON, then it is likely binary frame data
                    pass

                # If we have received use_lstm and frame data, process the frame
                data = await websocket.receive_bytes()
                logging.debug(f"Received data of length: {len(data)}")
                response = await process_video_frame(data, yolo_model, deepsort_model, deepface_model, lstm_model, use_lstm)
                await websocket.send_bytes(response)
                await websocket.send_text('{"type": "ack"}')
            except WebSocketDisconnect:
                logging.info("Client disconnected")
                break
            except Exception as e:
                logging.error(f"Error processing frame: {e}")
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    finally:
        keepalive_task.cancel()
        if websocket.application_state != WebSocketState.DISCONNECTED:
            await websocket.close()
            logging.info("WebSocket closed properly")
