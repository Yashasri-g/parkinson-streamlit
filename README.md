**Parkinson’s Disease Detection Using Handwriting and Voice Analysis**

This project presents an AI-based system for the early, non-invasive detection of Parkinson’s Disease (PD) by analyzing vocal impairments and handwriting patterns. It features two independently trained models — one based on voice features using ensemble machine learning, and the other using handwriting images with deep learning — integrated into a unified web interface using Streamlit.

**Project Objective**

To build and deploy a dual-modality AI system that allows real-time PD screening using:
1. Voice feature data (jitter, shimmer, MFCCs, etc.)
2. Handwriting images (spiral or wave patterns)
   
**Models**

Voice Model:
1. Features: Jitter, Shimmer, HNR, MFCCs
2. Algorithms: XGBoost, LightGBM, Random Forest (stacked with Logistic Regression)
3. Accuracy: 93.33%
   
Handwriting Model:
1. Architecture: MobileNetV2 with Transfer Learning
2. Input: Preprocessed handwriting images
3. Accuracy: ~90%
   
**Features**

1. Real-time predictions with uploaded voice/image files
2. Confidence scores displayed with result
3. Separate models for each input type
4. Fully deployed online via Streamlit

**Tech Stack**

1. Python, scikit-learn, XGBoost, LightGBM, TensorFlow, Keras
2. Streamlit (for UI)
3. joblib & JSON (for model and threshold saving)
4. Deployed on Streamlit Cloud

**How to Use**

1. Upload either a .csv file (voice features) or an image file (handwriting)
2. Click the Predict button
3. View result: Healthy or Parkinson’s with confidence score
**Deployment**

Live app: https://parkinson-app.streamlit.app

GitHub Repo: https://github.com/Yashasri-g/parkinson-streamlit
