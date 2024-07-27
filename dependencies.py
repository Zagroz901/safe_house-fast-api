# app/dependencies.py

from fastapi import Depends
from .models.yolo_model import load_yolo_model
from .models.deepsort_model import load_deepsort_model
from .models.deepface_model import load_deepface_model
from .models.lstm_model import load_lstm_model

from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import logging

logger = logging.getLogger(__name__)


yolo_model_instance = None
deepsort_model_instance = None
deepface_model_instance = None
lstm_model_instance = None


def get_yolo_model():
    global yolo_model_instance
    if yolo_model_instance is None:
        yolo_model_instance = load_yolo_model()
    return yolo_model_instance

def get_deepsort_model():
    global deepsort_model_instance
    if deepsort_model_instance is None:
        deepsort_model_instance = load_deepsort_model()
    return deepsort_model_instance

def get_deepface_model():
    global deepface_model_instance
    if deepface_model_instance is None:
        deepface_model_instance = load_deepface_model()
    return deepface_model_instance

def get_lstm_model():
    global lstm_model_instance
    if lstm_model_instance is None:
        lstm_model_instance = load_lstm_model()
    return lstm_model_instance
