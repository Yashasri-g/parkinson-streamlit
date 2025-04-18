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
audio_model = joblib.load("parkinson_audiomodel.pkl")

# Extract audio features
def extract_audio_features(audio_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=None)
        if len(y) < sr * 1:
            raise ValueError("Audio file is too short")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        os.unlink(tmp_path)
        return mfccs_mean.reshape(1, -1)

    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None

# Predict image
def predict_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred_prob = image_model.predict(img_array)[0][0]
    label = "🧠 Parkinson Detected" if pred_prob >= 0.5 else "✅ Healthy"
    return label, pred_prob

# Predict audio
def predict_audio(audio_file):
    features = extract_audio_features(audio_file)
    if features is not None:
        try:
            pred = audio_model.predict(features)[0]
            prob = audio_model.predict_proba(features)[0][1]
            label = "🧠 Parkinson Detected" if pred == 1 else "✅ Healthy"
            return label, prob
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return None, None
    return None, None

# UI Setup
st.set_page_config(page_title="Parkinson Detection App", layout="centered", page_icon="🧬")
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f3e5f5, #e1f5fe);
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
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Parkinson's Disease Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image or audio sample for prediction</div>", unsafe_allow_html=True)

# Input type
mode = st.radio("Choose input type", ["🖼️ Image", "🎧 Audio"], horizontal=True)

# Image mode
if mode == "🖼️ Image":
    st.header("Upload an Image (Spiral/Wave)")
    img_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
    if img_file:
        st.image(Image.open(img_file), caption="Uploaded Image", use_column_width=True)
        if st.button("Predict Image"):
            with st.spinner("Predicting..."):
                label, prob = predict_image(img_file)
                st.markdown(f"<h4 style='color:#4a148c;'>Prediction: {label}</h4>", unsafe_allow_html=True)
                st.markdown(f"<small>Model Confidence: <b>{prob:.2f}</b></small>", unsafe_allow_html=True)

# Audio mode
elif mode == "🎧 Audio":
    st.header("Upload a .wav Audio File")
    audio_file = st.file_uploader("Upload audio", type=['wav'])
    if audio_file:
        st.audio(audio_file)
        if st.button("Predict Audio"):
            with st.spinner("Predicting..."):
                label, prob = predict_audio(audio_file)
                if label:
                    st.markdown(f"<h4 style='color:#1565c0;'>Prediction: {label}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<small>Model Confidence: <b>{prob:.2f}</b></small>", unsafe_allow_html=True)
