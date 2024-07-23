# main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import websocket, api

app = FastAPI()

# Include routes
app.include_router(websocket.router)
app.include_router(api.router, prefix="/api")

# Serve static files
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
