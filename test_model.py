
import os
import cv2
import joblib
import numpy as np

DATASET = r"C:\Users\srini\OneDrive\Desktop\SignVision\dataset\asl_alphabet_train\asl_alphabet_train"

model = joblib.load("models/random_forest.pkl")
encoder = joblib.load("models/label_encoder.pkl")

print("Enter a letter to test (A-Z):")
letter = input("> ").strip().upper()

if letter not in encoder.classes_:
    raise ValueError("Enter a valid letter from A to Z.")

folder = os.path.join(DATASET, letter)
images = [
    f for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

if not images:
    raise FileNotFoundError("No images found for " + letter)

path = os.path.join(folder, images[0])
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (32, 32))
features = image.astype(np.float32).reshape(1, -1) / 255.0

prediction = model.predict(features)[0]
predicted_label = encoder.inverse_transform([prediction])[0]

print("Actual label:", letter)
print("Predicted label:", predicted_label)

cv2.imshow("Dataset Image", cv2.imread(path))
cv2.waitKey(0)
cv2.destroyAllWindows()
