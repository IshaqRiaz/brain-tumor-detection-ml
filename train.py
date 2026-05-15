import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# -------------------------
# CONFIG
# -------------------------
DATASET_PATH = "dataset"
CATEGORIES = ["non_tumor", "tumor"]
IMG_SIZE = 150

data = []

# -------------------------
# LOAD DATASET
# -------------------------
for category in CATEGORIES:
    path = os.path.join(DATASET_PATH, category)
    print("Checking folder:", path)
    if not os.path.exists(path):
        print("❌ Folder NOT found:", path)
    else:
        print("✅ Files:", os.listdir(path))
        label = CATEGORIES.index(category)

    for img in os.listdir(path):
        try:
            img_path = os.path.join(path, img)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append([img_array, label])
        except:
            pass

print("Images loaded:", len(data))

# -------------------------
# PREPROCESS
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
# CNN MODEL
# -------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# -------------------------
# TRAIN MODEL
# -------------------------
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# -------------------------
# SAVE MODEL
# -------------------------
os.makedirs("model", exist_ok=True)
model.save("model/brain_tumor_model.h5")

# -------------------------
# GRAPH
# -------------------------
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("Model Accuracy")
plt.legend()
plt.show()
