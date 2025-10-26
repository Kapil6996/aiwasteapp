import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("waste_classifier_portable.keras")

# Define category mapping (edit these as per your training classes)
categories = {
    0: {"name": "Plastic", "recyclable": True, "tips": "Rinse and dry plastic bottles before recycling. Avoid black plastic — many centers can't process it."},
    1: {"name": "Glass", "recyclable": True, "tips": "Separate by color if required. Remove lids and rinse to avoid contamination."},
    2: {"name": "Paper", "recyclable": True, "tips": "Keep paper clean and dry. Avoid greasy food wrappers or wax-coated paper."},
    3: {"name": "Metal", "recyclable": True, "tips": "Crush cans to save space. Make sure they’re clean and dry before recycling."},
    4: {"name": "Organic Waste", "recyclable": False, "tips": "Compost it! Organic waste can turn into nutrient-rich soil through composting."},
    5: {"name": "E-Waste", "recyclable": False, "tips": "Take electronics to an authorized e-waste collection center. Never throw them in regular bins."}
}

# Streamlit UI
st.title("♻️ AI Waste Classifier")
st.write("Upload an image of waste, and the AI will tell you its category, recyclability, and how to handle it responsibly.")

uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]

    # Get class info
    info = categories.get(predicted_class, {"name": "Unknown", "recyclable": None, "tips": "No data available."})

    # Display results
    st.subheader(f"🧩 Predicted Category: **{info['name']}**")

    if info["recyclable"]:
        st.success("♻️ This item **is recyclable!**")
    elif info["recyclable"] is False:
        st.error("🚫 This item **is not recyclable!**")
    else:
        st.warning("⚠️ Recyclability information not available.")

    st.info(f"💡 **Recycling Tip:** {info['tips']}")
