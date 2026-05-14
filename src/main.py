# =========================
# Brain Tumor Detection FYP
# =========================

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------
# 1. LOAD DATASET
# -------------------------
DATASET_PATH = "dataset"
CATEGORIES = ["non_tumor", "tumor"]
IMG_SIZE = 150

data = []

for category in CATEGORIES:
    path = os.path.join(DATASET_PATH, category)
    label = CATEGORIES.index(category)

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)

            # read image
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # resize image
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

            data.append([img_array, label])

        except Exception as e:
            pass

print("Dataset loaded:", len(data))

# -------------------------
# 2. SPLIT DATA
# -------------------------
X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X / 255.0
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 3. BUILD CNN MODEL
# -------------------------
model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

# binary output
model.add(layers.Dense(1, activation='sigmoid'))

# -------------------------
# 4. COMPILE MODEL
# -------------------------
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------
# 5. TRAIN MODEL
# -------------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# -------------------------
# 6. SAVE MODEL
# -------------------------
model.save("brain_tumor_model.h5")

# -------------------------
# 7. ACCURACY GRAPH
# -------------------------
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Model Accuracy")
plt.legend()
plt.show()

# -------------------------
# 8. PREDICTION FUNCTION
# -------------------------
def predict(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    img = img / 255.0

    result = model.predict(img)[0][0]

    if result > 0.5:
        print("🧠 Tumor Detected")
    else:
        print("✅ Non Tumor Detected")


# example:
# predict("test.jpg")
