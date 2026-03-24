import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("dyslexia_cnn_model.h5")

IMG_SIZE = 64

# Load image
img_path = "Gambo/Test/Normal/A-41.png"
image = cv2.imread(img_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

# Normalize
data = gray / 255.0

# Reshape for CNN
data = data.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# Predict
prediction = model.predict(data)
pred_idx = np.argmax(prediction)

labels = ["Corrected", "Normal", "Reversal"]

print("Prediction:", labels[pred_idx])