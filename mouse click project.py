import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Pyautogui security settings
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# -----------------------------
# Load Hand Landmarker Model (Tasks API)
# -----------------------------
base_option = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_option,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

landmarker = vision.HandLandmarker.create_from_options(options)

# Webcam Setup
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

frameR = 100  # Frame boundary reduction

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror effect ke liye flip karein
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # MediaPipe ke liye RGB format mein convert karein
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Hand Landmarker detect karein
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # Index Finger Tip (Landmark 8) aur Thumb Tip (Landmark 4)
            x1 = int(hand_landmarks[8].x * w)
            y1 = int(hand_landmarks[8].y * h)
            x2 = int(hand_landmarks[4].x * w)
            y2 = int(hand_landmarks[4].y * h)

            # Visual circles draw karein fingertips par
            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 1. Mouse Movement (Index Finger mapping)
            if x1 > frameR and x1 < w - frameR and y1 > frameR and y1 < h - frameR:
                x_screen = np.interp(x1, (frameR, w - frameR), (0, screen_w))
                y_screen = np.interp(y1, (frameR, h - frameR), (0, screen_h))
                pyautogui.moveTo(x_screen, y_screen)

            # 2. Left Click (Distance between Thumb and Index)
            distance = math.hypot(x2 - x1, y2 - y1)
            if distance < 35:
                cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 12, (0, 255, 0), -1)
                pyautogui.click()
                pyautogui.sleep(0.2)  # Double click roknay ke liye chhota delay

    cv2.imshow("Virtual Mouse (Tasks API)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()