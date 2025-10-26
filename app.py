from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import json
from recycle_tips import get_recycling_tips

app = Flask(__name__)
CORS(app)

# Load model
model = tf.keras.models.load_model('waste_classifier.h5')

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {v: k for k, v in class_indices.items()}

# Recycling tips dictionary
TIPS = {
    "cardboard": ("recyclable", "Flatten boxes and keep them dry before recycling."),
    "glass": ("recyclable", "Rinse bottles, remove caps, and avoid broken glass."),
    "metal": ("recyclable", "Clean cans, remove labels if possible, and recycle."),
    "paper": ("recyclable", "Keep it clean and dry. Don’t recycle greasy paper."),
    "plastic": ("recyclable", "Rinse and reuse this plastic bottle."),
    "trash": ("trash", "Not recyclable. Try to reduce or reuse if possible.")
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    # Read image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img / 255.0, axis=0)

    # Predict
    pred = model.predict(img)
    pred_index = np.argmax(pred)
    category = index_to_class[pred_index]

    # Recyclable/trash logic
    recyclable_classes = ["plastic", "metal", "paper", "glass", "cardboard"]
    if category in recyclable_classes:
        status = "recyclable"
        suggestion = TIPS.get(category, ("recyclable", "Handle carefully."))[1]
    else:
        status = "trash"
        suggestion = TIPS.get(category, ("trash", "Not recyclable."))[1]

    tip = get_recycling_tips(category)

    return jsonify({
        "prediction": status,
        "suggestion": suggestion,
        "category": category,
        "tip": tip
    })

if __name__ == "__main__":
    app.run(debug=True)
