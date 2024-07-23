from ultralytics import YOLO


def load_yolo_model():
    model_path = "weights_models/yolov8m-pose.pt"
    model = YOLO(model_path)
    return model