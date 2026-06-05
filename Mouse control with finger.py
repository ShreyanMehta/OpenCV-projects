import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import urllib.request
import os
import time
import math
import numpy as np

# --- 1. MODEL SETUP ---
# Download the Hand Landmarker model if it doesn't exist
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Hand Landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete!")

# --- 2. INITIALIZE MEDIA PIPE TASKS ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1 # We only want to track one hand at a time
)
detector = vision.HandLandmarker.create_from_options(options)

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Get your monitor's screen resolution
screen_w, screen_h = pyautogui.size()

# Variables for smoothing and clicking
smooth_x, smooth_y = 0, 0
smoothing_factor = 0.5  # 0.0 to 1.0 (Lower = smoother but more delay)
last_click_time = 0

print("Script running! Move your index finger to move the mouse. Pinch thumb and index to click.")
print("FAILSAFE: Slam your physical mouse to any corner of your screen to emergency stop.")

while True:
    success, frame = cam.read()
    if not success:
        break
        
    # Flip the frame horizontally so it acts like a mirror
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    
    # Define the "Active Box" (reduces the area you have to move your hand in)
    box_margin = 100
    cv2.rectangle(frame, (box_margin, box_margin), (frame_w - box_margin, frame_h - box_margin), (255, 0, 0), 2)
    
    # Convert and prepare the image for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect hands
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        # Get the landmarks for the first detected hand
        landmarks = detection_result.hand_landmarks[0]
        
        # Landmark 8 is the tip of the Index Finger
        # Landmark 4 is the tip of the Thumb
        index_tip = landmarks[8]
        thumb_tip = landmarks[4]
        
        # Get pixel coordinates
        index_x = int(index_tip.x * frame_w)
        index_y = int(index_tip.y * frame_h)
        thumb_x = int(thumb_tip.x * frame_w)
        thumb_y = int(thumb_tip.y * frame_h)
        
        # Draw circles on the tips for visual feedback
        cv2.circle(frame, (index_x, index_y), 8, (0, 255, 0), -1)
        cv2.circle(frame, (thumb_x, thumb_y), 8, (0, 255, 255), -1)
        
        # --- MOUSE MOVEMENT ---
        # Map the webcam coordinates (constrained to our active box) to the screen resolution
        # Using np.interp prevents the cursor from going off-screen if you leave the box
        target_x = np.interp(index_x, (box_margin, frame_w - box_margin), (0, screen_w))
        target_y = np.interp(index_y, (box_margin, frame_h - box_margin), (0, screen_h))
        
        # Apply smoothing
        smooth_x = smooth_x + (target_x - smooth_x) * smoothing_factor
        smooth_y = smooth_y + (target_y - smooth_y) * smoothing_factor
        
        pyautogui.moveTo(smooth_x, smooth_y)
        
        # --- MOUSE CLICKING (PINCH) ---
        # Calculate the Euclidean distance (hypotenuse) between the thumb and index finger
        pinch_distance = math.hypot(index_x - thumb_x, index_y - thumb_y)
        
        # If the fingers are close together, register a click
        click_threshold = 30 # Adjust this if it clicks too early or too late
        
        if pinch_distance < click_threshold:
            # Draw a line connecting them to show a successful pinch
            cv2.line(frame, (index_x, index_y), (thumb_x, thumb_y), (0, 0, 255), 3)
            
            current_time = time.time()
            if current_time - last_click_time > 0.5: # Half-second delay between clicks
                print("Pinch detected! Clicking...")
                pyautogui.click()
                last_click_time = current_time

    # Show the camera window
    cv2.imshow('Hand Controlled Mouse', frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cam.release()
cv2.destroyAllWindows()