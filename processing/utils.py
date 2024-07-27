# app/processing/utils.py

import cv2
import numpy as np
import logging
from passlib.context import CryptContext
import jwt
from datetime import datetime, timedelta
from collections import deque

sequence = deque(maxlen=128)  
Q = deque(maxlen=128)
violence_mode_active = False 
violence_duration = 0 
cooldown_counter = 0  
cooldown_threshold = 5

def detect_person(image, model):
    results = model.predict(image, classes=[0], conf=0.6, iou=0.5)
    if len(results[0].boxes): 
        return True, results
    else: 
        return False, []

def track_person(results, frame, deep_sort, previous_data):
    data = {}
    detections = []
    if results:  # Check if results is not empty
        for result in results[0].boxes:
            xywh = result.xywh.cpu().numpy().flatten()
            conf = np.array([result.conf.item()])
            detection = np.concatenate((xywh, conf), axis=0)
            detections.append(detection)
    formatted_detections = np.array(detections)
    if formatted_detections.size == 0:
        return data  # Prevent further processing if there are no detections

    deep_sort.update(formatted_detections[:, :4], formatted_detections[:, 4], frame)

    tracks = deep_sort.tracker.tracks  # Access Track objects directly
    keypoints = results[0].keypoints.data.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        best_match = None
        best_iou = 0  
        track_box = np.array([x1, y1, x2, y2])
        for idx in range(len(keypoints)):
            x11, y11, x22, y22 = boxes[idx]
            detection_box = np.array([x11, y11, x22, y22])
            iou = compute_iou(track_box, detection_box)
            if iou > best_iou:
                best_iou = iou
                best_match = idx
        if best_match is not None:
            x11, y11, x22, y22 = boxes[best_match]
            track_id = track.track_id
            data[track_id] = {
                'location': [x1, y1, x2, y2],
                'keyPoints': keypoints[best_match]
            }

    for id in data.keys():
        if id in previous_data.keys():
            if 'ver_res' in previous_data[id].keys(): data[id]['ver_res'] = previous_data[id]['ver_res']
            if 'verified' in previous_data[id].keys(): data[id]['verified'] = previous_data[id]['verified']
    return data


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1, yi1 = max(x1, x1_), max(y1, y1_)
    xi2, yi2 = min(x2, x2_), min(y2, y2_)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0  # Prevent division by zero
    
    return inter_area / union_area



def verify_faces(frame, faces_detected, data, deepface_model):
    results = {}
    frame_height, frame_width = frame.shape[:2]

    for id in data.keys():
        if not 'verified' in data[id].keys():
            if faces_detected[id]:
                x1, y1, x2, y2 = data[id]['location']
                # logging.debug(f"Original coordinates for ID {id}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                # Ensure coordinates are within frame boundaries
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(frame_width, int(x2)), min(frame_height, int(y2))
                # logging.debug(f"Adjusted coordinates for ID {id}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                # Extract face region and verify its size
                face_region = frame[y1:y2, x1:x2]
                if face_region.size == 0:
                    # logging.error(f"Face region for ID {id} is empty. Skipping verification.")
                    results[id] = False
                    continue

                # logging.debug(f"Face region for ID {id} size: {face_region.shape}")

                try:
                    verification = deepface_model.find(
                        face_region,
                        'data_photo/',
                        detector_backend="yolov8",
                        enforce_detection=False,
                        threshold=0.3,
                        model_name='Facenet512'
                    )
                    results[id] = True if len(verification) and len(verification[0]['identity']) else False
                except Exception as e:
                    logging.error(f"Error verifying face for ID {id}: {e}")
                    results[id] = False
            else:
                results[id] = 'kk'
    return results



def process_verified_people(results, data, frame):
    for id in data.keys():
        x1, y1, x2, y2 = data[id]['location']
        if 'verified' in data[id].keys():
            color = (0, 255, 0) if data[id]['verified'] else (0, 0, 255)
        else:
            color = (255, 0, 0) if results[id] == 'kk' else (0, 255, 0) if results[id] else (0, 0, 255)
            if 'ver_res' in data[id].keys():
                if results[id] != 'kk': data[id]['ver_res'].append(results[id])
            else: 
                if results[id] != 'kk': data[id]['ver_res'] = [results[id]]
            
            if 'ver_res' in data[id].keys() and len(data[id]['ver_res']) >= 5:
                true_count = data[id]['ver_res'].count(True)
                false_count = data[id]['ver_res'].count(False)
                print(f'{true_count},{false_count}')
                data[id]['verified'] = true_count > false_count
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, str(id), (int(x1) - 10, int(y1) - 10), 1, 2, color, 2)


def detect_violence(frame,model):
    print("Loading model ...")

    frame_resized = cv2.resize(frame, (128, 128)).astype("float32") / 255
    sequence.append(frame_resized)
    print(f"the size of sequesse : {len(sequence)}")

    if len(sequence) == 13:  # Check if we have collected 16 frames
        print("frame has been collected")
        input_sequence = np.expand_dims(np.array(sequence), axis=0)  # Shape (1, 16, 64, 64, 3)
        preds = model.predict(input_sequence)[0]
        Q.append(preds)
        sequence.popleft()  # Remove the oldest frame to maintain a sliding window of 16 frames
        results = np.array(Q).mean(axis=0)
        i = np.argmax(results)
        label = "Violence" if i == 1 else "No Violence"
        confidence = results[i] * 100

        return label, confidence

    return "No Violence", 0.0 


def process_frame_for_violence(frame, data, model):
    global violence_mode_active, violence_duration, cooldown_counter
    threshold = 0.01
    ids = list(data.keys())
    n = len(ids)
    iou_matrix = np.zeros((n, n)) if not violence_mode_active else None
    violence_detected = False
    detection_confidence = 0.0

    if not violence_mode_active:
        for i in range(n):
            for j in range(i + 1, n):
                iou_value = compute_iou(data[ids[i]]['location'], data[ids[j]]['location'])
                iou_matrix[i, j] = iou_value
                iou_matrix[j, i] = iou_value
                if iou_value > threshold:
                    label, confidence = detect_violence(frame, model)
                    if label == "Violence":
                        violence_mode_active = True
                        violence_duration = 1
                        display_violence_alert(frame, label, confidence)
                        violence_detected = True
                        detection_confidence = confidence
                        break
            if violence_mode_active:
                break

    if violence_mode_active:
        label, confidence = detect_violence(frame, model)
        display_violence_alert(frame, label, confidence)
        if label == "Violence":
            violence_duration += 1
            violence_detected = True
            detection_confidence = confidence
        else:
            if label == "No Violence":
                if violence_duration >= threshold:
                    if cooldown_counter < cooldown_threshold:
                        cooldown_counter += 1
                    else:
                        violence_mode_active = False
                        violence_duration = 0
                        cooldown_counter = 0
                else:
                    violence_mode_active = False
                    violence_duration = 0

    return {
        "violence_detected": violence_detected,
        "detection_confidence": detection_confidence,
        "violence_duration": violence_duration if violence_detected else 0
    }


def check_violence_duration(duration):
    threshold = 10
    return duration >= threshold

def display_violence_alert(frame, label, confidence):
    text = f"Violence Detected: {label} ({confidence:.2f}%)"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

   
   
def clear_chase(inactive_frame_count, MAX_INACTIVE_FRAMES, sequence, Q):
    if inactive_frame_count >= MAX_INACTIVE_FRAMES:
        sequence.clear()  
        Q.clear()         
        inactive_frame_count = 0 
    return inactive_frame_count

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "mysecretkey"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

