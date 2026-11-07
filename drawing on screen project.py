import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

# Drawing variables
draw_color = (255, 0, 255)  # Default Purple/Pink
brush_thickness = 15
eraser_thickness = 50
xp, yp = 0, 0  # Previous coordinates

# Blank canvas for drawing
canvas = np.zeros((hCam, wCam, 3), np.uint8)


# Finger counting helper function
def count_fingers(landmarks):
    fingers = []
    # Thumb (x-axis comparison)
    fingers.append(landmarks[4].x < landmarks[3].x)
    # 4 Fingers (y-axis comparison)
    for tip in [8, 12, 16, 20]:
        fingers.append(landmarks[tip].y < landmarks[tip - 2].y)
    return fingers.count(True)


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # MediaPipe preprocessing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    # --- Draw Header / UI on Frame ---
    # Colors & Eraser boxes at the top
    cv2.rectangle(frame, (40, 10), (140, 70), (255, 0, 0), -1)  # Blue
    cv2.putText(frame, "BLUE", (55, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(frame, (180, 10), (280, 70), (0, 255, 0), -1)  # Green
    cv2.putText(frame, "GREEN", (190, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(frame, (320, 10), (420, 70), (0, 0, 255), -1)  # Red
    cv2.putText(frame, "RED", (345, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.rectangle(frame, (460, 10), (580, 70), (50, 50, 50), -1)  # Eraser
    cv2.putText(frame, "ERASE", (480, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            landmarks = hand_landmarks

            # Index finger tip (ID: 8)
            x1, y1 = int(landmarks[8].x * w), int(landmarks[8].y * h)

            # Check how many fingers are open
            fingers_up = count_fingers(landmarks)

            # 1. Selection Mode (Jab Index aur Middle dono fingers up hon ya haath top area mein ho)
            if y1 < 80:
                xp, yp = 0, 0  # Reset previous point so lines don't connect across selections
                if 40 < x1 < 140:
                    draw_color = (255, 0, 0)  # Blue
                elif 180 < x1 < 280:
                    draw_color = (0, 255, 0)  # Green
                elif 320 < x1 < 420:
                    draw_color = (0, 0, 255)  # Red
                elif 460 < x1 < 580:
                    draw_color = (0, 0, 0)  # Eraser (Black color on canvas)

                cv2.circle(frame, (x1, y1), 15, draw_color, -1)

            # 2. Erasing Mode (Full Hand / Open Palm - 5 fingers up)
            elif fingers_up >= 5:
                cv2.circle(frame, (x1, y1), eraser_thickness, (0, 0, 0), cv2.FILLED)
                cv2.putText(frame, "ERASING...", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                # Erase on canvas
                cv2.line(canvas, (xp, yp), (x1, y1), (0, 0, 0), eraser_thickness)
                xp, yp = x1, y1

            # 3. Drawing Mode (Sirf Index finger up ho)
            elif fingers_up == 1:
                cv2.circle(frame, (x1, y1), 10, draw_color, -1)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                # Draw line on canvas
                if draw_color == (0, 0, 0):
                    cv2.line(canvas, (xp, yp), (x1, y1), (0, 0, 0), eraser_thickness)
                else:
                    cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)

                xp, yp = x1, y1
            else:
                xp, yp = 0, 0

    # --- Merge Canvas with Video Frame ---
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv_thresh = cv2.cvtColor(inv_thresh, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv_thresh)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Iron Man Virtual Air Canvas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()