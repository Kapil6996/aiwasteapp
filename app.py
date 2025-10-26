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
    
    # --- Class Info ---
    class_info = {
        0: {"name": "Cardboard", "recyclable": True, "tip": "Rinse and sort plastics by type before recycling."},
        1: {"name": "Glass", "recyclable": True, "tip": "Keep paper clean and dry. Avoid mixing with food waste."},
        2: {"name": "Metal", "recyclable": True, "tip": "Crush cans to save space and remove any labels if possible."},
        3: {"name": "Paper", "recyclable": True, "tip": "Rinse glass containers and separate by color if required."},
        4: {"name": "Plastic", "recyclable": False, "tip": "Check the recycling number. #1 and #2 are most recyclable.",
           "Plastic Bottles: Hanging Planters / Herb Garden: Cut large soda bottles horizontally or vertically. Puncture drainage holes, decorate the outside, and hang them by string or wire to create a space-saving wall garden."},
        5: {"name": "Trash", "non-recyclable": True, "tip": "Take electronics to a certified e-waste recycling facility."},
    }

    info = class_info.get(predicted_class, {"name": "Unknown", "recyclable": "N/A", "tip": "Try uploading a clearer image."})

    # --- Display Result ---
    st.success(f"🧾 Predicted Class: {predicted_class} ({info['name']})")
    st.write(f"♻️ Recyclable: **{info['recyclable']}**")
    st.info(f"💡 Tip: {info['tip']}")
