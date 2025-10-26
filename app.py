import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model("waste_classifier_portable.keras")

# Define category mapping (edit these as per your training classes)
categories = {
    0: {"name": "Plastic", "recyclable": True, "tips": '''Rinse and dry plastic bottles before recycling. Avoid black plastic — many centers can't process it.
    Plastic Bottles: Hanging Planters / Herb Garden: Cut large soda bottles horizontally or vertically. Puncture drainage holes, decorate the outside, and hang them by string or wire to create a space-saving wall garden.

Bottle Bottoms: Flower Decorations:	Cut off the rigid, patterned bottoms of large soda bottles, paint them bright colours, and string them together to create garlands, wind chimes, or faux flowers.

Lotion/Shampoo Bottles:	Charging Phone Holder:	Cut a slot in the back of an empty, clean bottle (near the neck) to hang over an electrical adapter while the phone rests in the body of the bottle during charging.

Plastic Lids/Caps:	Mosaic Art/Play Mat:	Glue colourful plastic caps onto a piece of plywood or cardboard to create a textured mosaic for decoration or a sensory mat for kids.''',

        "cardboard": '''Flatten boxes before recycling.

Cereal Boxes: Magazine or File Holders	: Cut the box diagonally from one corner to the opposite side to create a stylish slanted holder. Cover with decorative paper, fabric, or paint.

Toilet Paper / Paper Towel Rolls:	Cord Organizers:	Insert coiled power cords or extension cords into the tubes to keep them untangled in a drawer. Label the outside of the tube with the cord's use or length.

Shoeboxes:	Decorative Storage Boxes:	Cover the boxes with wallpaper, contact paper, or paint. Use them to store photos, craft supplies, office clutter, or socks.

Egg Cartons (Cardboard):	Seed Starter Trays:	Fill the cups with soil and seeds. Once the seedlings are ready, you can often tear off the individual cup and plant it directly into the ground, as the cardboard will biodegrade''',
'''},
    1: {"name": "Glass", "recyclable": True, "tips": '''Separate by color if required. Remove lids and rinse to avoid contamination.'''},
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
