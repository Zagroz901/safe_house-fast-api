# app/routes/api.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from ..dependencies import get_yolo_model, get_deepsort_model, get_deepface_model
from ..processing.video_processing import process_video_frame
from fastapi import Depends
from typing import List
from fastapi import FastAPI, Depends, HTTPException, status, Form
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

from  ..models.user_model import Base, engine, SessionLocal, User ,EmergencyContact
from ..schema.user_schema import UserResponse ,EmergencyContactCreate, EmergencyContactResponse

from ..processing.utils import ALGORITHM, SECRET_KEY, get_password_hash, verify_password, create_access_token


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
    yolo_model=Depends(get_yolo_model),
    deepsort_model=Depends(get_deepsort_model),
    deepface_model=Depends(get_deepface_model)
):
    if file.content_type != "video/mp4":
        raise HTTPException(status_code=400, detail="Invalid file type. Only mp4 is allowed.")
    
    # Process the uploaded video
    result = await process_video_frame(file, yolo_model, deepsort_model, deepface_model)
    return {"message": "Video processed successfully", "result": result}

@router.post("/register", response_model=UserResponse)
def create_user(username: str = Form(...), email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    db_user = db.query(User).filter((User.username == username) | (User.email == email)).first()
    if db_user:
        if db_user.email == email:
            raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(password)
    db_user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/login", response_model=dict)
def login_for_access_token(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
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


@router.post("/users/{user_id}/emergency-contact", response_model=EmergencyContactResponse)
def add_emergency_contact(user_id: int, contact: EmergencyContactCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    new_contact = EmergencyContact(**contact.dict(), user_id=user_id)
    db.add(new_contact)
    db.commit()
    db.refresh(new_contact)
    return new_contact

@router.get("/users/{user_id}/emergency-contact", response_model=List[EmergencyContactResponse])
def get_emergency_contacts(user_id: int, db: Session = Depends(get_db)):
    contacts = db.query(EmergencyContact).filter(EmergencyContact.user_id == user_id).all()
    return contacts
