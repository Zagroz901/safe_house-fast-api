# main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import websocket, api
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # URL of the React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routes
app.include_router(websocket.router)
app.include_router(api.router, prefix="/api")

# Serve static files
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
