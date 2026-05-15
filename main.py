from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model/brain_tumor_model.h5")
IMG_SIZE = 150

# -------------------------
# HOME PAGE
# -------------------------


@app.route('/')
def home():
    return render_template('index.html')

# -------------------------
# PREDICT
# -------------------------


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    path = os.path.join("static/uploads", file.filename)
    file.save(path)

    # preprocess image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    img = img / 255.0

    prediction = model.predict(img)[0][0]
    confidence = float(prediction) * 100

    if prediction > 0.5:
        result = "🧠 Tumor Detected"
        confidence_text = f"{confidence:.2f}% Tumor Probability"
    else:
        result = "✅ Non Tumor Detected"
        confidence_text = f"{100 - confidence:.2f}% Non-Tumor Probability"

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence_text,
        image=file.filename
    )


# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)
