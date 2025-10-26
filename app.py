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
        3: {"name": "Paper", "recyclable": True, "tip": "Rinse glass containers and separate by color if required.","tips": "1.Newspaper/Magazine Pages:	Woven Baskets or Coasters:	Roll the paper tightly into long, thin tubes or strips, then weave them together, securing them with glue, to create durable baskets, placemats, or coasters.

"2.Junk Mail/Scraps:	Shredded Packing Material:	Run unwanted paper (excluding glossy or photo paper) through a shredder to create simple, effective cushioning for shipping parcels."

"3. Magazine Pages:	Decoupage:	Cut out colourful images or text and glue them onto furniture, picture frames, or boxes for a unique, patterned finish."

"4. Paper Bags:	DIY Book Covers:	Cut large paper bags open and use them to cover school books or notebooks, protecting them from damage"},

        4: {"name": "Plastic", "recyclable": True, "tip": "Check the recycling number. #1 and #2 are most recyclable.","tips": "1. Plastic Bottles: Hanging Planters / Herb Garden: Cut large soda bottles horizontally or vertically. Puncture drainage holes, decorate the outside, and hang them by string or wire to create a space-saving wall garden.    "
           "2.    Bottle Bottoms: Flower Decorations:	Cut off the rigid, patterned bottoms of large soda bottles, paint them bright colours, and string them together to create garlands, wind chimes, or faux flowers.    "
           "3.    Lotion/Shampoo Bottles:	Charging Phone Holder:	Cut a slot in the back of an empty, clean bottle (near the neck) to hang over an electrical adapter while the phone rests in the body of the bottle during charging.    "
           "4.    Plastic Lids/Caps:	Mosaic Art/Play Mat:	Glue colourful plastic caps onto a piece of plywood or cardboard to create a textured mosaic for decoration or a sensory mat for kids."},
        5: {"name": "Trash", "non-recyclable": False, "tip": "Take electronics to a certified e-waste recycling facility."},
    }

    info = class_info.get(predicted_class, {"name": "Unknown", "recyclable": "N/A", "tip": "Try uploading a clearer image."})

    # --- Display Result ---
    st.success(f"🧾 Predicted Class: {predicted_class} ({info['name']})")
    st.write(f"♻️ Recyclable: **{info['recyclable']}**")
    st.info(f"💡 Tip: {info['tip']}")
    st.info(f"💡 Tips: {info['tips']}")

