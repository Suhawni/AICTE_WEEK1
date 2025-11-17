import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import cv2

# ---------------------------
# Load model & labels
# ---------------------------
model = load_model("waste_classification_model.h5")
classes = {0: "Organic Waste (O)", 1: "Inorganic Waste (R)"}

# ---------------------------
# CSS / Header
# ---------------------------
st.markdown(
    """
    <style>
    body {
        background: rgba(255, 255, 255, 0.8);
        font-family: Arial, sans-serif;
        color: #fff;
    }
    .main {
        background: linear-gradient(to right, #764BA2, #667EEA);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        cursor: pointer;
        font-size: 14px;
    }
    .stButton button:hover { background-color: #45a049; }
    .header { font-family: 'Arial Black', sans-serif; font-size: 35px; color: #4CAF50; text-align: center; }
    .subheader { font-family: 'Arial', sans-serif; font-size: 22px; color: #ffffff; text-align: center; }
    hr { border: none; height: 1px; background: #ddd; margin: 20px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="header">🌿 Waste Classification App</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Classify waste as Organic or Inorganic</div>', unsafe_allow_html=True)

# Sidebar for mode selection
st.sidebar.title("⚙️ Input Mode")
input_mode = st.sidebar.radio("Choose input mode:", ["Upload Image", "Camera Capture"])

# ---------------------------
# Image preprocessing (fixes RGBA -> RGB)
# ---------------------------
def preprocess_image(pil_image, target_size=(150, 150)):
    """Ensures image is RGB, resizes, converts to numpy array and scales to [0,1]."""
    if pil_image.mode == "RGBA":
        background = Image.new("RGB", pil_image.size, (255, 255, 255))
        background.paste(pil_image, mask=pil_image.split()[3])
        pil_image = background
    else:
        pil_image = pil_image.convert("RGB")

    pil_image = pil_image.resize(target_size)
    arr = img_to_array(pil_image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------------------
# Prediction
# ---------------------------
def predict(pil_image):
    processed = preprocess_image(pil_image, target_size=(150, 150))
    prediction = model.predict(processed, verbose=0)
    predicted_class = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction))
    return classes[predicted_class], confidence

# ---------------------------
# Draw bounding box & label
# ---------------------------
def draw_bounding_box(pil_image, label, confidence):
    pil_rgb = pil_image.convert("RGB")
    image_np = np.array(pil_rgb)
    h, w, _ = image_np.shape

    # Draw bounding box
    x1, y1, x2, y2 = int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)
    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Add label
    label_text = f"{label} ({confidence:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(label_text, font, 0.7, 2)[0]
    text_x = x1
    text_y = y1 - 10 if y1 - 10 > text_size[1] + 5 else y1 + text_size[1] + 15
    cv2.rectangle(
        image_np,
        (text_x, text_y - text_size[1] - 6),
        (text_x + text_size[0] + 6, text_y + 2),
        (0, 255, 0),
        -1,
    )
    cv2.putText(image_np, label_text, (text_x + 3, text_y - 2), font, 0.7, (0, 0, 0), 2)
    return Image.fromarray(image_np)

# ---------------------------
# Main UI logic
# ---------------------------
if input_mode == "Upload Image":
    st.write("📤 **Upload Image**")
    uploaded_file = st.file_uploader("Upload an image for classification", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Classify Image"):
            try:
                label, confidence = predict(image)
                result_image = draw_bounding_box(image, label, confidence)
                st.image(result_image, caption=f"Result — {label} ({confidence:.2f})", use_column_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")

elif input_mode == "Camera Capture":
    st.write("📸 **Capture Image**")
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture)
        st.image(image, caption="Captured Image", use_column_width=True)

        if st.button("🔍 Classify Image"):
            try:
                label, confidence = predict(image)
                result_image = draw_bounding_box(image, label, confidence)
                st.image(result_image, caption=f"Result — {label} ({confidence:.2f})", use_column_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")
