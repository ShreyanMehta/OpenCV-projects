import cv2
import mediapipe as mp
import math
import time
import urllib.request
import os

# 1. Download the required model bundle if it's missing
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand_landmarker.task...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)

# 2. Setup Tasks API (Notice num_hands is now 2)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2, 
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = HandLandmarker.create_from_options(options)

# 3. Setup Virtual Object
obj_x, obj_y = 100, 100
obj_w, obj_h = 100, 100
obj_color = (255, 0, 0)

# Zoom state tracking
prev_hands_dist = None

# 4. Webcam Loop
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
start_time = time.time()

while cap.isOpened():
    success, img = cap.read()
    if not success: break
        
    img = cv2.flip(img, 1) 
    h, w, c = img.shape
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    timestamp_ms = int((time.time() - start_time) * 1000)
    
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    obj_color = (255, 0, 0) # Default Blue
    pinches = [] # We will store the coordinates of any active pinches here
    
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]
            
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            cx, cy = (tx + ix) // 2, (ty + iy) // 2
            
            # If fingers are close, register it as a pinch
            if math.hypot(tx - ix, ty - iy) < 40:
                pinches.append((cx, cy))
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            else:
                cv2.circle(img, (cx, cy), 15, (255, 0, 0), cv2.FILLED)

    # 5. Interaction Logic
    if len(pinches) == 2:
        # TWO HANDS PINCHING = ZOOM MODE
        cx1, cy1 = pinches[0]
        cx2, cy2 = pinches[1]
        
        # Draw a visual line connecting the two hands
        cv2.line(img, (cx1, cy1), (cx2, cy2), (0, 255, 255), 3)
        obj_color = (0, 255, 255) # Turn yellow while scaling
        
        # Calculate current distance between the two hands
        current_hands_dist = math.hypot(cx2 - cx1, cy2 - cy1)
        
        if prev_hands_dist is None:
            prev_hands_dist = current_hands_dist
        else:
            # Calculate how much the hands moved since the last frame
            delta = int(current_hands_dist - prev_hands_dist)
            prev_hands_dist = current_hands_dist
            
            # Apply that change to the square's width and height
            new_w = max(30, obj_w + delta) # max() prevents it from shrinking below 30px
            new_h = max(30, obj_h + delta)
            
            # Shift the x and y coordinates so the square scales from its center
            obj_x -= (new_w - obj_w) // 2
            obj_y -= (new_h - obj_h) // 2
            
            obj_w, obj_h = new_w, new_h

    elif len(pinches) == 1:
        # ONE HAND PINCHING = MOVE MODE
        prev_hands_dist = None # Reset zoom tracker
        cx, cy = pinches[0]
        
        if obj_x < cx < obj_x + obj_w and obj_y < cy < obj_y + obj_h:
            obj_x = cx - obj_w // 2
            obj_y = cy - obj_h // 2
            obj_color = (0, 255, 0) # Turn green while moving
            
    else:
        # NO HANDS PINCHING
        prev_hands_dist = None

    # 6. Render the square
    cv2.rectangle(img, (obj_x, obj_y), (obj_x + obj_w, obj_y + obj_h), obj_color, cv2.FILLED)
    cv2.imshow("Virtual Object Control", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
landmarker.close()
cap.release()
cv2.destroyAllWindows()