import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)

# Configuration
TRAIN_PATH = r"C:\Users\srini\OneDrive\Desktop\SignVision\dataset\asl_alphabet_train\asl_alphabet_train"
MAX_IMAGES_PER_CLASS = 300
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("SIGNVISION - ASL LANDMARK TRAINING PIPELINE")
print("=" * 60)

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(f"Dataset directory not found: {TRAIN_PATH}")

classes = sorted([
    folder for folder in os.listdir(TRAIN_PATH)
    if os.path.isdir(os.path.join(TRAIN_PATH, folder))
])

print(f"Found {len(classes)} classes: {classes}")

def extract_landmark_features(landmarks):
    """Extracts, translates to wrist origin, and scale-normalizes 21 3D landmarks."""
    coords = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
    # Translate wrist (landmark 0) to origin
    coords -= coords[0]
    # Scale normalize by maximum Euclidean distance from wrist
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

X_features = []
y_labels = []
skipped_count = 0

print("\nExtracting MediaPipe landmarks from dataset images...")

for label in classes:
    folder = os.path.join(TRAIN_PATH, label)
    files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])[:MAX_IMAGES_PER_CLASS]

    extracted_class_count = 0
    for file in files:
        path = os.path.join(folder, file)
        image = cv2.imread(path)

        if image is None:
            skipped_count += 1
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_image)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            feature_vector = extract_landmark_features(landmarks)
            X_features.append(feature_vector)
            y_labels.append(label)
            extracted_class_count += 1
        else:
            skipped_count += 1

    print(f"  Class '{label}': Extracted landmarks from {extracted_class_count}/{len(files)} images")

hands_detector.close()

X_features = np.array(X_features, dtype=np.float32)
y_labels = np.array(y_labels)

print(f"\nFeature extraction complete.")
print(f"Total samples extracted: {X_features.shape[0]}")
print(f"Skipped images (no hand detected or unreadable): {skipped_count}")

if len(X_features) == 0:
    raise RuntimeError("No hand landmarks were detected in the dataset images. Check dataset directory.")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)

encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
joblib.dump(label_encoder, encoder_path)
print(f"Label encoder saved to: {encoder_path}")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_encoded,
    test_size=0.20, random_state=42,
    stratify=y_encoded
)

# Logistic Regression
print("\nTraining Logistic Regression Classifier...")
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
logistic_preds = logistic_model.predict(X_test)
logistic_acc = accuracy_score(y_test, logistic_preds)
print(f"Logistic Regression Accuracy: {logistic_acc:.4f}")

joblib.dump(logistic_model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

# Random Forest
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Random Forest Accuracy: {rf_acc:.4f}")

rf_model_path = os.path.join(MODEL_DIR, "random_forest.pkl")
joblib.dump(rf_model, rf_model_path)
print(f"Random Forest model saved to: {rf_model_path}")

print("\nClassification Report (Random Forest):")
print(classification_report(
    y_test, rf_preds,
    target_names=label_encoder.classes_,
    zero_division=0
))

# Save Confusion Matrix
cm = confusion_matrix(y_test, rf_preds, labels=np.arange(len(label_encoder.classes_)))
fig, ax = plt.subplots(figsize=(12, 10))
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
display.plot(ax=ax, xticks_rotation=90, cmap="Blues", values_format="d")
plt.title("SignVision - Landmark Random Forest Confusion Matrix")
plt.tight_layout()
matrix_path = "landmark_confusion_matrix.png"
plt.savefig(matrix_path, dpi=300)
plt.close()
print(f"Confusion matrix plot saved to: {matrix_path}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE - SIGNVISION LANDMARK MODEL READY")
print("=" * 60) 