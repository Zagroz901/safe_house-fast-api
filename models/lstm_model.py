# app/models/lstm_model.py

from keras.models import load_model

def load_lstm_model():
    model_path = "weights_models/lstm_30_13.h5"
    model = load_model(model_path)
    return model
