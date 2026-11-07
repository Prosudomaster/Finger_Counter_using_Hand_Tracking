import cv2
import mediapipe as mp      # mediapipe is Framework Developed by Google for Machine Learning Tasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------
# Load Hand Landmarker Model
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

# -----------------------------
# Finger Counting Function
# -----------------------------

def count_fingers(landmarks):
    fingers=[]

    fingers.append(landmarks[4].x < landmarks[3].x) # Thumb
    # other fingers
    for tip in [8,12,16,20]:
        fingers.append(landmarks[tip].y < landmarks[tip - 2].y)

    return fingers.count(True)

# -----------------------------
# Open Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            finger_count = count_fingers(hand_landmarks)

            h,w,c = frame.shape
            # Draw landmarks
            for Im in hand_landmarks:
                cx, cy = int(Im.x * w), int(Im.y * h) # 0.5 * 640 = 320 pixels
                cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
            # Display Finger Count
            cv2.putText(frame, f"Fingers: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

    cv2.imshow("Finger Counter (Task API)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()