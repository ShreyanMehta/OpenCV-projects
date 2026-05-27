import os
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import numpy as np

# We only need this one clean import for audio now!
from pycaw.pycaw import AudioUtilities

# 1. Download the Hand Landmarker model if it doesn't exist
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe Hand Landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Initialize Pycaw (Windows Volume Control) - NEW API
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume
volRange = volume.GetVolumeRange() 
minVol = volRange[0]
maxVol = volRange[1]

# 3. Setup MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7)
detector = vision.HandLandmarker.create_from_options(options)

# 4. Initialize Camera
cap = cv2.VideoCapture(0)

# Set default values for the UI
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    if not success:
        break
        
    # Convert BGR to RGB (MediaPipe requires RGB)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    
    # Process the image with the HandLandmarker task
    detection_result = detector.detect(mp_image)
    
    # Check if hands are detected
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            h, w, c = img.shape
            
            # Get coordinates of Thumb tip (Index 4) and Index finger tip (Index 8)
            x1, y1 = int(hand_landmarks[4].x * w), int(hand_landmarks[4].y * h)
            x2, y2 = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
            
            # Draw primary visual markers on the tracking fingers
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # Draw the rest of the hand joints as small green dots
            for landmark in hand_landmarks:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            
            # Calculate the physical distance between the fingers
            length = math.hypot(x2 - x1, y2 - y1)
            
            # Map hand distance to system volume and UI bar ranges
            vol = np.interp(length, [20, 200], [minVol, maxVol])
            volBar = np.interp(length, [20, 200], [400, 150])
            volPer = np.interp(length, [20, 200], [0, 100])
            
            # Change the system volume
            volume.SetMasterVolumeLevel(vol, None)
            
    # --- DRAWING THE UI ON THE SCREEN ---
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 
                1, (0, 255, 0), 3)
            
    cv2.imshow("Gesture Volume Control", img)
    
    # Press 'q' to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()