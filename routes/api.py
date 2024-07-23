# app/routes/api.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from ..dependencies import get_yolo_model, get_deepsort_model, get_deepface_model
from ..processing.video_processing import process_video_frame
from fastapi import Depends

from fastapi import FastAPI, Depends, HTTPException, status, Form
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

from  ..models.user_model import Base, engine, SessionLocal, User
from ..schema.user_schema import UserResponse
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

@router.post("/users/", response_model=UserResponse)
def create_user(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(password)
    db_user = User(username=username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/token", response_model=dict)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

def authenticate_user(db, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

@router.get("/users/me/", response_model=UserResponse)
def read_users_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user
