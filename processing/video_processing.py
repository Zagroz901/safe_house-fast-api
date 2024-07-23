# app/processing/video_processing.py

import cv2
import numpy as np
from .utils import detect_person, track_person, verify_faces, process_verified_people
import logging
import asyncio

logging.basicConfig(level=logging.DEBUG)

previous_frame_data = {}

async def process_video_frame(data, yolo_model, deepsort_model, deepface_model):
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        logging.error("Failed to decode image")
        raise ValueError("Failed to decode image")

    # Apply YOLO for human detection
    person_detected, results = detect_person(frame, yolo_model)
    
    if person_detected:
        # DeepSort tracking
        data = track_person(results, frame, deepsort_model, previous_frame_data)
        
        # Face verification with DeepFace
        faces_detected = is_face_detected(data)
        verification_results = verify_faces(frame, faces_detected, data, deepface_model)
        
        # # Display verification results on frame
        process_verified_people(verification_results, data, frame)
        
    # Encode the processed frame back to bytes
    _, buffer = cv2.imencode('.jpg', frame)
    await asyncio.sleep(0)  # Yield control to the event loop
    return buffer.tobytes()

def is_face_detected(data):
    result = {}
    for key in data.keys():
        face_KP = data[key]["keyPoints"][0:5]
        face_KP_conf = face_KP[:, 2]
        count_of_detected_kp = np.sum(face_KP_conf > 0.8)
        result[key] = True if count_of_detected_kp >= 3 else False
    return result
