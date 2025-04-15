import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Parkinson's Detection", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-size: 16px;
    }
    .stFileUploader>div>div {
        background-color: #ffffff;
        border: 2px dashed #aaa;
        border-radius: 10px;
        padding: 1em;
    }
    .footer {
        font-size: 12px;
        text-align: center;
        color: gray;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://i.imgur.com/zbw3mXb.png", use_column_width=True)
st.sidebar.title("🧠 Parkinson's Detector")
st.sidebar.markdown("""
Upload a **spiral** or **wave** image to classify it as:
- spiral_healthy  
- spiral_parkinson  
- wave_healthy  
- wave_parkinson

Built with 🧠 MobileNetV2 + Streamlit.
""")

# Load model
@st.cache_resource
def load_parkinson_model():
    return load_model("parkinson_model1.h5")

model = load_parkinson_model()
class_labels = ['spiral_healthy', 'spiral_parkinson', 'wave_healthy', 'wave_parkinson']

# Main content
st.title("🎯 Parkinson's Disease Detection App")
st.write("Detect Parkinson’s from hand-drawn spiral/wave images using a deep learning model.")

uploaded_file = st.file_uploader("📁 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence = predictions[0][predicted_index]

    with col2:
        st.subheader("🔍 Prediction")
        st.success(f"**{predicted_class}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")

        st.subheader("📊 Probabilities")
        prob_cols = st.columns(2)
        for i, (label, prob) in enumerate(zip(class_labels, predictions[0])):
            with prob_cols[i % 2]:
                st.info(f"{label}: {prob * 100:.2f}%")

# Footer
st.markdown("---")
st.markdown('<p class="footer">Created with ❤️ by Yashasri | Powered by TensorFlow & Streamlit</p>', unsafe_allow_html=True)
