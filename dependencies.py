# app/dependencies.py

from fastapi import Depends
from .models.yolo_model import load_yolo_model
from .models.deepsort_model import load_deepsort_model
from .models.deepface_model import load_deepface_model
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

def get_yolo_model():
    return load_yolo_model()

def get_deepsort_model():
    return load_deepsort_model()

def get_deepface_model():
    return load_deepface_model()
