import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Image size
IMG_SIZE = 64

# Dataset path
data_dir = "Gambo/Train"
categories = ["Corrected", "Normal", "Reversal"]

data = []
labels = []

LIMIT_PER_CLASS = 6000

# Load images
for category in categories:
    path = os.path.join(data_dir, category)
    label = categories.index(category)

    count = 0
    for img in os.listdir(path):
        if count >= LIMIT_PER_CLASS:
            break
        try:
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            data.append(image)
            labels.append(label)

            count += 1
        except Exception as e:
            print("Error:", e)

# Convert to numpy
data = np.array(data, dtype=np.float32) / 255.0
labels = np.array(labels)

# Reshape for CNN
data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# One-hot encoding
labels = to_categorical(labels, 3)

print("Dataset shape:", data.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(3, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("dyslexia_cnn_model_advance.h5")

print("✅ CNN Model Saved")