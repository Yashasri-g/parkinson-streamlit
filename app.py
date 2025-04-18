import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
import librosa
import tempfile
import os

# Load models
image_model = tf.keras.models.load_model("parkinson_imagemodel.h5")
audio_model = joblib.load("parkinson_audio_model.pkl")

# Feature extraction
def extract_audio_features(audio_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=None)
        if len(y) < sr * 1:
            raise ValueError("Audio file is too short")

        # Extract 30 MFCCs to be safe; pipeline will handle what it needs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        os.unlink(tmp_path)  # Cleanup
        return mfccs_mean.reshape(1, -1)

    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None

# Predictions
def predict_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = image_model.predict(img_array)[0][0]
    return "🧠 Parkinson Detected" if pred >= 0.5 else "✅ Healthy"

def predict_audio(audio_file):
    features = extract_audio_features(audio_file)
    if features is not None:
        try:
            pred = audio_model.predict(features)[0]
            return "🧠 Parkinson Detected" if pred == 1 else "✅ Healthy"
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return None
    return None

# UI Styling
st.set_page_config(page_title="Parkinson Detection App", layout="centered", page_icon="🧬")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e3f2fd, #fce4ec);
        padding: 20px;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 3em;
        color: #4a148c;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3em;
        color: #6a1b9a;
        margin-bottom: 1.5em;
    }
    .stRadio > div {
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Parkinson's Disease Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image or audio sample for prediction</div>", unsafe_allow_html=True)

# App navigation
mode = st.radio("Choose input type", ["🖼️ Image", "🎧 Audio"], horizontal=True)

# Image mode
if mode == "🖼️ Image":
    st.header("Upload a Spiral/Wave Image")
    img_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if img_file:
        st.image(Image.open(img_file), caption="Uploaded Image", use_column_width=True)
        if st.button("Predict Image"):
            result = predict_image(img_file)
            if result:
                st.success(f"Prediction: {result}")

# Audio mode
elif mode == "🎧 Audio":
    st.header("Upload a .wav Audio File")
    audio_file = st.file_uploader("Upload a .wav file", type=['wav'])
    if audio_file:
        st.audio(audio_file)
        if st.button("Predict Audio"):
            result = predict_audio(audio_file)
            if result:
                st.success(f"Prediction: {result}")
