Parkinson’s Disease Detection Using Handwriting and Voice Analysis
This project presents an AI-based system for the early, non-invasive detection of Parkinson’s Disease (PD) by analyzing vocal impairments and handwriting patterns. It features two independently trained models — one based on voice features using ensemble machine learning, and the other using handwriting images with deep learning — integrated into a unified web interface using Streamlit.

Project Objective
To build and deploy a dual-modality AI system that allows real-time PD screening using:

Voice feature data (jitter, shimmer, MFCCs, etc.)

Handwriting images (spiral or wave patterns)

Models
Voice Model:

Features: Jitter, Shimmer, HNR, MFCCs

Algorithms: XGBoost, LightGBM, Random Forest (stacked with Logistic Regression)

Accuracy: 93.33%

Handwriting Model:

Architecture: MobileNetV2 with Transfer Learning

Input: Preprocessed handwriting images

Accuracy: ~90%

Features
Real-time predictions with uploaded voice/image files

Confidence scores displayed with result

Separate models for each input type

Fully deployed online via Streamlit

Tech Stack
Python, scikit-learn, XGBoost, LightGBM, TensorFlow, Keras

Streamlit (for UI)

joblib & JSON (for model and threshold saving)

Deployed on Streamlit Cloud

How to Use
Upload either a .csv file (voice features) or an image file (handwriting)

Click the Predict button

View result: Healthy or Parkinson’s with confidence score

Deployment
Live app: https://parkinson-app.streamlit.app
GitHub Repo: https://github.com/Yashasri-g/parkinson-streamlit
