import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
import librosa
import io
import tempfile

# Load models
image_model = tf.keras.models.load_model("parkinson_imagemodel.h5")
audio_model = joblib.load("parkinson_audio_model.pkl")

# AUDIO FEATURE EXTRACTOR
def extract_features_advanced(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y) < sr * 1:
            raise ValueError("Audio too short")

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

        feature_vector = np.hstack([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(zcr),
            np.std(zcr),
            np.mean(spec_centroid),
            np.std(spec_centroid)
        ])
        return feature_vector
    except Exception as e:
        raise e

# IMAGE PREDICTION
def predict_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = image_model.predict(img_array)[0][0]
    return "🧠 Parkinson Detected" if pred >= 0.5 else "✅ Healthy"

# AUDIO PREDICTION
def extract_audio_features(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    features = extract_features_advanced(tmp_path)
    return features.reshape(1, -1)

def predict_audio(audio_file):
    features = extract_audio_features(audio_file)
    pred = audio_model.predict(features)[0]
    return "🧠 Parkinson Detected" if pred == 1 else "✅ Healthy"

# UI STYLING
st.set_page_config(page_title="Parkinson Detection App", layout="centered", page_icon="🌿")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f7fa 0%, #fce4ec 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 3em;
        color: #4a148c;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        font-size: 1.3em;
        color: #6a1b9a;
        margin-top: 0em;
        margin-bottom: 2em;
    }
    .stRadio > div {
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Parkinson's Disease Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image or audio sample to detect Parkinson's</div>", unsafe_allow_html=True)

# SELECT INPUT TYPE
app_mode = st.radio("Choose Detection Mode", ["🖼️ Image", "🎧 Audio"], horizontal=True)

# IMAGE MODE
if app_mode == "🖼️ Image":
    st.header("Upload Spiral or Wave Image")
    image_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if image_file is not None:
        st.image(Image.open(image_file), caption='Uploaded Image', use_column_width=True)
        if st.button("🔍 Predict from Image"):
            result = predict_image(image_file)
            st.success(f"Prediction: {result}")

# AUDIO MODE
elif app_mode == "🎧 Audio":
    st.header("Upload .wav Audio File")
    audio_file = st.file_uploader("Upload a .wav file", type=['wav'])
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("🔍 Predict from Audio"):
            try:
                result = predict_audio(audio_file)
                st.success(f"Prediction: {result}")
            except Exception as e:
                st.error(f"Error: {e}")
