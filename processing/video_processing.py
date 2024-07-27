# video_processing.py

import cv2
import numpy as np
from .utils import *
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)
previous_frame_data = {}

logging.basicConfig(level=logging.DEBUG, format='%(threadName)s: %(message)s')

async def async_detect_person(frame, yolo_model):
    return detect_person(frame, yolo_model)

async def async_track_person(results, frame, deepsort_model):
    return track_person(results, frame, deepsort_model, previous_frame_data)

async def async_verify_faces(frame, faces_detected, data, deepface_model):
    return verify_faces(frame, faces_detected, data, deepface_model)

async def async_verify_violence(frame, data, lstm_model):
    return process_frame_for_violence(frame, data , lstm_model)

async def process_frame(frame, yolo_model, deepsort_model, deepface_model,lstm_model):
    logging.debug("Starting asynchronous frame processing")
    inactive_frame_count = 0 
    MAX_INACTIVE_FRAMES = 300  

    person_detected, results = await async_detect_person(frame, yolo_model)
    
    if person_detected:
        inactive_frame_count = 0 
        data = await async_track_person(results, frame, deepsort_model)
        if len(data) >=2:
            result_of_violence =  await async_verify_violence(frame,data,lstm_model)
        faces_detected = is_face_detected(data)
        verification_results = await async_verify_faces(frame, faces_detected, data, deepface_model)
        process_verified_people(verification_results, data, frame)
    else:
        inactive_frame_count += 1 
        inactive_frame_count = clear_chase(inactive_frame_count, MAX_INACTIVE_FRAMES, sequence, Q)
    logging.debug("Completed asynchronous frame processing")
    return frame



async def process_video_frame(data, yolo_model, deepsort_model, deepface_model,lstm_model):
    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        logging.error("Failed to decode image")
        raise ValueError("Failed to decode image")

    logging.debug("Submitting frame to asynchronous processing")
    processed_frame = await process_frame(frame, yolo_model, deepsort_model, deepface_model,lstm_model)
    
    _, buffer = cv2.imencode('.jpeg', processed_frame)
    return buffer.tobytes()


# def process_frame(frame, yolo_model, deepsort_model, deepface_model):
#     logging.debug("Starting frame processing")

#     # Downscale frame for processing
#     # original_size = (frame.shape[1], frame.shape[0])
#     # downscale_size = (640, 360)  # Example size, adjust based on your needs
#     # frame = cv2.resize(frame, downscale_size)

#     # Apply YOLO for human detection
#     person_detected, results = detect_person(frame, yolo_model)
    
#     if person_detected:
#         # DeepSort tracking
#         data = track_person(results, frame, deepsort_model, previous_frame_data)
        
#         # Face verification with DeepFace
#         faces_detected = is_face_detected(data)
#         verification_results = verify_faces(frame, faces_detected, data, deepface_model)
        
#         # Display verification results on frame
#         process_verified_people(verification_results, data, frame)
    
#     # Optionally, resize back to the original size if needed for display
#     # frame = cv2.resize(frame, original_size)
#     logging.debug("Completed frame processing")
#     return frame

# async def process_video_frame(data, yolo_model, deepsort_model, deepface_model):
#     nparr = np.frombuffer(data, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     if frame is None:
#         logging.error("Failed to decode image")
#         raise ValueError("Failed to decode image")

#     logging.debug("Submitting frame to thread pool")
#     loop = asyncio.get_event_loop()
#     processed_frame = await loop.run_in_executor(executor, process_frame, frame, yolo_model, deepsort_model, deepface_model)
    
#     _, buffer = cv2.imencode('.jpeg', processed_frame)
#     return buffer.tobytes()

def is_face_detected(data):
    result = {}
    for key in data.keys():
        face_KP = data[key]["keyPoints"][0:5]
        face_KP_conf = face_KP[:, 2]
        count_of_detected_kp = np.sum(face_KP_conf > 0.8)
        result[key] = True if count_of_detected_kp >= 3 else False
    return result
