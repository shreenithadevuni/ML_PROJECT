import os
import cv2
import joblib
import numpy as np
import mediapipe as mp
from collections import deque, Counter

MODEL_PATH = "models/random_forest.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

CONFIDENCE_THRESHOLD = 0.40
HISTORY_SIZE = 7
VOTES_NEEDED = 4

# Verify files existence
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(
        f"Model files missing! Ensure '{MODEL_PATH}' and '{ENCODER_PATH}' exist by running main.py first."
    )

print("Loading SignVision Models...")
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

classes = encoder.classes_.tolist()
print(f"Loaded model successfully. Classes ({len(classes)}): {classes}")

def extract_landmark_features(landmarks):
    """Extracts, translates to wrist origin, and scale-normalizes 21 3D landmarks."""
    coords = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
    coords -= coords[0]
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam with fallback
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(1)

if not camera.isOpened():
    hands.close()
    raise RuntimeError("Could not access webcam (Device 0 or 1). Check camera permissions.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

history = deque(maxlen=HISTORY_SIZE)
sentence = ""
stable_label = None

print("\n" + "=" * 50)
print(" SIGNVISION WEBCAM INTERFACE ")
print("=" * 50)
print(" Controls:")
print("  ENTER     : Append current stabilized sign")
print("  SPACE     : Insert space character")
print("  BACKSPACE : Delete last character")
print("  C         : Clear sentence")
print("  Q         : Quit application")
print("=" * 50 + "\n")

try:
    while True:
        ok, frame = camera.read()
        if not ok:
            print("Failed to capture frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        display_label = "Show Hand"
        confidence = 0.0
        current_stable = None

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]

            # Bounding Box for visual feedback
            xs = [p.x * w for p in landmarks.landmark]
            ys = [p.y * h for p in landmarks.landmark]
            pad = 20
            x1 = max(0, int(min(xs)) - pad)
            y1 = max(0, int(min(ys)) - pad)
            x2 = min(w, int(max(xs)) + pad)
            y2 = min(h, int(max(ys)) + pad)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            mp_draw.draw_landmarks(
                frame,
                landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # Feature extraction and prediction
            features = extract_landmark_features(landmarks).reshape(1, -1)
            probabilities = model.predict_proba(features)[0]
            best_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[best_idx])
            best_label = classes[best_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                history.append(best_label)
            else:
                history.append(None)

            valid_votes = [lbl for lbl in history if lbl is not None]
            if valid_votes:
                most_common, count = Counter(valid_votes).most_common(1)[0]
                if count >= VOTES_NEEDED:
                    current_stable = most_common
                    display_label = current_stable
                else:
                    display_label = "Stabilizing..."
            else:
                display_label = "Uncertain"
        else:
            history.clear()

        stable_label = current_stable

        # UI Overlay
        status_color = (0, 255, 0) if current_stable else (0, 165, 255)
        
        cv2.putText(frame, f"Sign: {display_label}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.0%}", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"Sentence: {sentence[-30:]}", (15, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "ENTER: Add | SPACE: Space | BACKSPACE: Del | C: Clear | Q: Quit",
                    (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cv2.imshow("SignVision - ASL Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (10, 13):  # Enter key
            if stable_label:
                if stable_label.upper() in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
                    sentence += stable_label.upper()
                elif stable_label.lower() == "space":
                    sentence += " "
                elif stable_label.lower() in ("del", "delete"):
                    sentence = sentence[:-1]
                history.clear()
                stable_label = None

        elif key == ord(" "):
            sentence += " "
        elif key in (8, 127):  # Backspace key
            sentence = sentence[:-1]
        elif key in (ord("c"), ord("C")):
            sentence = ""
        elif key in (ord("q"), ord("Q")):
            break

finally:
    camera.release()
    hands.close()
    cv2.destroyAllWindows()

print(f"\nFinal Constructed Sentence: '{sentence}'")
print("SignVision session closed.")