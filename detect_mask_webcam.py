import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys
import time

MODEL_PATH = os.path.join('model', 'mask_detector_model.h5')
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
IMG_SIZE = 224

# Load face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
# Load mask detector model
model = load_model(MODEL_PATH)

# Allow user to specify camera index via command-line argument
if len(sys.argv) > 1:
    try:
        user_cam_idx = int(sys.argv[1])
        cam_indices = [user_cam_idx]
        print(f"Camera index specified by user: {user_cam_idx}")
    except ValueError:
        print("Invalid camera index argument. Using default indices 0-4.")
        cam_indices = list(range(5))
else:
    cam_indices = list(range(5))

# Try multiple camera indices for robustness
camera_found = False
for cam_idx in cam_indices:
    cap = cv2.VideoCapture(cam_idx)
    if cap.isOpened():
        print(f"Using camera index {cam_idx}.")
        camera_found = True
        break
    else:
        cap.release()
if not camera_found:
    print("Error: Could not open any webcam. Please check if the webcam is connected, not used by another application, and drivers are up to date.")
    print("You can specify a camera index as an argument, e.g. python detect_mask_webcam.py 1")
    sys.exit(1)

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame from webcam. Attempting to reinitialize the camera after short delay...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(cam_idx)
            ret, frame = cap.read()
            if not ret:
                print("Error: Still could not read frame from webcam after reinitialization.\nPossible causes:\n- The webcam is in use by another application.\n- The webcam drivers are outdated or malfunctioning.\n- The camera index is incorrect.\n- Hardware connection issue.\nSuggestions:\n- Close other applications that may use the webcam.\n- Try a different USB port.\n- Update or reinstall webcam drivers.\n- Try running this script on another machine.\n- Try changing the camera index in the script or specify it as an argument.")
                break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_input = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_input = face_input.astype('float32') / 255.0
            # Enhanced normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            face_input = (face_input - mean) / std
            face_input = np.expand_dims(face_input, axis=0)
            pred = model.predict(face_input)[0][0]
            # Adjusted threshold for better accuracy
            threshold = 0.4
            label = "Mask" if pred < threshold else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.imshow('Face Mask Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"Exception occurred during frame capture or processing: {e}")
        break
cap.release()
cv2.destroyAllWindows()