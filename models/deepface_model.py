# app/models/deepface_model.py

from deepface import DeepFace

def load_deepface_model():
    # In DeepFace, we generally don't need to load a model as it loads internally, but we can configure it here.
    return DeepFace
