import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- 1. Download the Hand Landmarker model if it doesn't exist ---
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe Hand Landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# --- 2. Setup MediaPipe Tasks API ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7)
detector = vision.HandLandmarker.create_from_options(options)

# --- 3. Setup Webcam (1280x720) ---
cap = cv2.VideoCapture(0)
frame_width = 1280
frame_height = 720
cap.set(3, frame_width)
cap.set(4, frame_height)

# --- 4. Drawing Settings & UI Setup ---
canvas = None
px, py = 0, 0 # Previous coords for smooth drawing

# Brush/Eraser settings
brushThickness = 15
eraserThickness = 80
drawColor = (255, 0, 0) # Default Blue (BGR)
currentThickness = brushThickness

# Define Sidebar Area (Left side)
sidebar_boundary_x = 150 # The menu takes up the first 150 pixels on the left

# Load UI Images
icon_size = (110, 110)
gap = 20

# Create a list of dictionaries for colors/eraser
tools = [
    {"type": "color", "value": (0, 0, 255),   "thick": brushThickness,  "img": "paint_red.png"},    
    {"type": "color", "value": (0, 255, 0),   "thick": brushThickness,  "img": "paint_green.png"},  
    {"type": "color", "value": (255, 0, 0),   "thick": brushThickness,  "img": "paint_blue.png"},   
    {"type": "eraser", "value": (0, 0, 0),    "thick": eraserThickness, "img": "eraser.png"}      
]

# Pre-load and resize images, calculate their bounding boxes for the LEFT sidebar
for i, tool in enumerate(tools):
    if os.path.exists(tool["img"]):
        img = cv2.imread(tool["img"])
        tool["loaded_img"] = cv2.resize(img, icon_size)
    else:
        # Fallback if image not found 
        fallback = np.zeros((icon_size[1], icon_size[0], 3), dtype=np.uint8)
        if tool["type"] == "eraser":
            fallback[:] = (255, 255, 255) 
        else:
            fallback[:] = tool["value"]
        tool["loaded_img"] = fallback
        print(f"Warning: {tool['img']} not found, using fallback.")

    # Calculate X offset relative to the left side (0)
    x_offset = (sidebar_boundary_x - icon_size[0]) // 2
    y_offset = gap + i * (icon_size[1] + gap)
    tool["bbox"] = (x_offset, y_offset, x_offset + icon_size[0], y_offset + icon_size[1])

print(f"AI Painter initialized ({frame_width}x{frame_height})!")
print("- One finger (Index) up: DRAW")
print("- Two fingers (Index + Middle) up: HOVER (Use this to select tools from the left sidebar)")
print("- Press 'c' to clear canvas, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    # Flip horizontally for natural mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Initialize canvas
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Draw semi-transparent background for the LEFT sidebar
    sub_img = frame[0:h, 0:sidebar_boundary_x]
    white_rect = np.full(sub_img.shape, 230, dtype=np.uint8) 
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    frame[0:h, 0:sidebar_boundary_x] = res

    # Draw Tool Icons on Sidebar
    for tool in tools:
        x1, y1, x2, y2 = tool["bbox"]
        frame[y1:y2, x1:x2] = tool["loaded_img"]
        # Draw border around currently selected tool
        if drawColor == tool["value"] and currentThickness == tool["thick"]:
             cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 0), 3)


    # --- 5. Process Hand Landmarks (Tasks API) ---
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # Get coords of Index Tip (8)
            x1, y1 = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h)
            
            # Gesture detection
            index_is_up = hand_landmarks[8].y < hand_landmarks[6].y
            middle_is_up = hand_landmarks[12].y < hand_landmarks[10].y

            # Condition A: Hover/Selection Mode (Two fingers up)
            if index_is_up and middle_is_up:
                px, py = 0, 0 
                selection_cursor_color = (255, 0, 255)
                cv2.circle(frame, (x1, y1), 15, selection_cursor_color, cv2.FILLED) 

                # Check if hovering over the left sidebar tools
                if x1 < sidebar_boundary_x:
                    for tool in tools:
                        bx1, by1, bx2, by2 = tool["bbox"]
                        if bx1 < x1 < bx2 and by1 < y1 < by2:
                            drawColor = tool["value"]
                            currentThickness = tool["thick"]
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), selection_cursor_color, 5)

            # Condition B: Draw Mode (Only Index finger up)
            elif index_is_up and not middle_is_up:
                # Disallow drawing inside the left sidebar area
                if x1 > sidebar_boundary_x:
                    cv2.circle(frame, (x1, y1), currentThickness // 2, drawColor, cv2.FILLED)
                    
                    if px == 0 and py == 0:
                        px, py = x1, y1
                    
                    cv2.line(canvas, (px, py), (x1, y1), drawColor, currentThickness)
                    px, py = x1, y1
                else:
                    px, py = 0, 0

    # --- 6. Merge Canvas onto WebCam Frame ---
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inverse_mask = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY_INV)
    frame_bg = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    final_output = cv2.add(frame_bg, canvas)

    # Draw a vertical separator line at the boundary
    cv2.line(final_output, (sidebar_boundary_x, 0), (sidebar_boundary_x, h), (100,100,100), 2)

    cv2.imshow("AI Painter Deluxe", final_output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('c'): 
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()