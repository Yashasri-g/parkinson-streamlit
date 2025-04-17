import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
import librosa
import io

# Load models
image_model = tf.keras.models.load_model("parkinson_imagemodel.h5")
audio_model = joblib.load("parkinson_audio_model.pkl")

def predict_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = image_model.predict(img_array)[0][0]
    return "Parkinson Detected" if pred >= 0.5 else "Healthy"

def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

def predict_audio(audio_file):
    features = extract_audio_features(audio_file)
    pred = audio_model.predict(features)[0]
    return "Parkinson Detected" if pred == 1 else "Healthy"

# UI Styling
st.set_page_config(page_title="Parkinson Detection App", layout="centered", page_icon="🌿")
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stApp {background: linear-gradient(120deg, #e0f7fa 0%, #fce4ec 100%);}
    .title {text-align: center; font-size: 3em; color: #4a148c; margin-bottom: 0.2em;}
    .subtitle {text-align: center; font-size: 1.3em; color: #6a1b9a; margin-top: 0em;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Parkinson's Disease Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Choose a detection method below</div>", unsafe_allow_html=True)

# Navigation
app_mode = st.radio("Select Input Type", ["Image", "Audio"], horizontal=True)

if app_mode == "Image":
    st.header("🖼️ Upload Image (Spiral/Wave)")
    image_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        st.image(Image.open(image_file), caption='Uploaded Image', use_column_width=True)
        if st.button("Predict"):
            result = predict_image(image_file)
            st.success(f"Prediction: {result}")

elif app_mode == "Audio":
    st.header("🎧 Upload Audio (.wav)")
    audio_file = st.file_uploader("Upload a .wav file", type=['wav'])
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("Predict"):
            result = predict_audio(audio_file)
            st.success(f"Prediction: {result}")
