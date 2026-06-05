import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import urllib.request
import os
import time


# --- 1. MODEL SETUP (New in Tasks API) ---
# The Tasks API requires a specific model file to run. 
# This checks if you have it, and downloads it if you don't.
MODEL_PATH = 'face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Face Landmarker model (approx 2MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Download complete!")

# --- 2. INITIALIZE MEDIA PIPE TASKS ---
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
    # Note: Iris landmarks are included by default in this model.
)
detector = vision.FaceLandmarker.create_from_options(options)

# Initialize the webcam
cam = cv2.VideoCapture(0)

# Get your monitor's screen resolution
screen_w, screen_h = pyautogui.size()

print("Script running! Look around to move. Blink your left eye to click.")
print("Press 'q' in the camera window to quit.")
print("FAILSAFE: Slam your physical mouse to any corner of your screen to emergency stop.")
last_click_time = 0

while True:
    # Capture and prepare the frame
    success, frame = cam.read()
    if not success:
        break
        
    # Flip the frame horizontally so it acts like a mirror
    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape
    
    # Convert the frame to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- 3. CREATE MP.IMAGE (New in Tasks API) ---
    # The Tasks API requires frames to be wrapped in its own Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Process the frame to find face landmarks
    detection_result = detector.detect(mp_image)

    # Check if any faces were detected
    if detection_result.face_landmarks:
        # Get the landmarks for the first face detected
        landmarks = detection_result.face_landmarks[0]
        
        # --- MOUSE MOVEMENT ---
        # Landmarks 474-477 map to the right iris.
        right_eye_iris = [landmarks[474], landmarks[475], landmarks[476], landmarks[477]]
        
        for id, landmark in enumerate(right_eye_iris):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            
            # Draw a green circle on the iris for visual feedback
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            if id == 1:
                # Map the camera coordinates to your screen resolution
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)
                
       # --- IMPROVED MOUSE CLICKING ---
        # Landmarks 145 (bottom) and 159 (top) map to the left eyelid
        left_eye_bottom = landmarks[145]
        left_eye_top = landmarks[159]
        
        cv2.circle(frame, (int(left_eye_top.x * frame_w), int(left_eye_top.y * frame_h)), 3, (0, 255, 255), -1)
        cv2.circle(frame, (int(left_eye_bottom.x * frame_w), int(left_eye_bottom.y * frame_h)), 3, (0, 255, 255), -1)

        # 1. Calculate distance in actual PIXELS instead of raw fractions
        vertical_dist = (left_eye_bottom.y - left_eye_top.y) * frame_h
        
        # 2. Print the distance to the terminal so you can see your personal baseline
        print(f"Eye Gap: {vertical_dist:.2f} pixels")
        
        # 3. New threshold (adjust this based on what prints in your terminal)
        blink_threshold = 8.0 
        
        if vertical_dist < blink_threshold:
            current_time = time.time()
            
            # 4. Only click if 1 second has passed since the last click (Non-blocking)
            if current_time - last_click_time > 1.0:
                print("BLINK DETECTED! CLICKING!")
                pyautogui.click()
                last_click_time = current_time
    # Show the camera window
    cv2.imshow('Eye Controlled Mouse', frame)
    
    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cam.release()
cv2.destroyAllWindows()