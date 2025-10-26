import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- App Title ---
st.title("♻️ AI Waste Classifier")
st.write("Upload a waste image and let the AI classify it as recyclable, organic, etc.")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("waste_classifier.h5")
    return model

model = load_model()

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # --- Preprocess Image ---
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # --- Predict ---
    st.write("🔍 Analyzing image...")
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # --- Display Result ---
    st.success(f"🧾 Predicted Class: {predicted_class}")
