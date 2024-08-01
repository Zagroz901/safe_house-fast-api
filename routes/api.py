# app/routes/api.py

import base64
import io
from fastapi import APIRouter, WebSocket, UploadFile, File, Form, HTTPException, Depends
from ..dependencies import get_yolo_model, get_deepsort_model, get_deepface_model,get_lstm_model
from ..processing.video_processing import process_video_frame
from fastapi import Depends
from typing import List
from fastapi import FastAPI, Depends, HTTPException, status, Form
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from PIL import Image
from  ..models.user_model import Base, engine, SessionLocal, User , EmergencyContact
from ..schema.user_schema import *
import logging
from ..processing.utils import ALGORITHM, SECRET_KEY, get_password_hash, verify_password, create_access_token
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect  # Import the exception
import json
import os
import shutil

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
@router.post("/upload_video")
async def upload_video(
    file: UploadFile = File(...),
    use_lstm: bool = Form(False),
    yolo_model=Depends(get_yolo_model),
    deepsort_model=Depends(get_deepsort_model),
    deepface_model=Depends(get_deepface_model),
    lstm_model=Depends(get_lstm_model),
):
    if file.content_type != "video/mp4":
        raise HTTPException(status_code=400, detail="Invalid file type. Only mp4 is allowed.")
    
    video_data = await file.read()
    if not video_data:
        raise HTTPException(status_code=400, detail="Failed to read video file.")
    
    try:
        result = await process_video_frame(video_data, yolo_model, deepsort_model, deepface_model, lstm_model, use_lstm)
        return JSONResponse(content={"message": "Video processed successfully", "result": result})
    except ValueError as e:
        return JSONResponse(content={"message": "Failed to process video"}, status_code=500)

@router.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket, 
                             yolo_model=Depends(get_yolo_model),
                             deepsort_model=Depends(get_deepsort_model),
                             deepface_model=Depends(get_deepface_model),
                             lstm_model=Depends(get_lstm_model)):
    await websocket.accept()
    use_lstm = False  # Default to False
    try:
        # First, receive JSON control data
        control_data = await websocket.receive_text()
        settings = json.loads(control_data)
        use_lstm = settings.get("useLSTM", False)
        print(use_lstm)
        # Then, receive binary frame data
        while True:
            try:
                frame_data = await websocket.receive_bytes()
                response = await process_video_frame(frame_data, yolo_model, deepsort_model, deepface_model, lstm_model, use_lstm)
                await websocket.send_bytes(response)
                await websocket.send_text('{"type": "ack"}')
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                break
    except WebSocketDisconnect:
        await websocket.close()
    except Exception as e:
        print(f"Error processing frame or control data: {e}")
        await websocket.close()

@router.post("/register", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter((User.username == user.username) | (User.email == user.email)).first()
    if db_user:
        if db_user.email == user.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/login", response_model=dict)
def login_for_access_token(cred:UserLog , db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == cred.email).first()
    if not user or not verify_password(cred.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me/", response_model=UserResponse)
def read_users_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = db.query(User).filter(User.email == payload.get("sub")).first()
        if not user:
            raise HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})


@router.post("/users/emergency-contact/{user_id}")
def user_info(data: InfoCreate, user_id: int , db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
         
    for email in data.email:
        db_contact= EmergencyContact(user_id= user_id, email= email)
        db.add(db_contact)

    db.commit()
    db.refresh(db_contact)
    return data

if not os.path.exists('photos'):
    os.makedirs('photos')


@router.post("/users/upload-photos/{user_id}")
async def upload_photos(photo_paths: PhotoPaths, user_id:int , db: Session = Depends(get_db)):
    user=db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not os.path.exists(f'photos/{user.username}'):
        os.makedirs(f'photos/{user.username}')
    for path in photo_paths.paths:
        try:
            filename = os.path.basename(path)
            destination = f"photos/{user.username}/{filename}"
            shutil.copyfile(path, destination)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error saving the file: {str(e)}")
    return {"message": "uploaded successfully"}



