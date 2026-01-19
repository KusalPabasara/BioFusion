"""
Live Prediction Page - Pneumonia Detection
Upload chest X-ray images for AI-powered analysis with Grad-CAM visualization.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_model, predict, get_prediction_label, preprocess_image, load_image
from utils import create_gradcam_visualization

# Page config
st.set_page_config(
    page_title="Live Prediction | Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="auto"
)

# Minimal CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)

# ============ TOP NAVIGATION BAR ============
nav_cols = st.columns(4)
with nav_cols[0]:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
with nav_cols[1]:
    if st.button("🔬 Predict", use_container_width=True, type="primary"):
        st.switch_page("pages/1_🔬_Live_Prediction.py")
with nav_cols[2]:
    if st.button("📊 Metrics", use_container_width=True):
        st.switch_page("pages/2_📊_Model_Insights.py")
with nav_cols[3]:
    if st.button("🗂️ Dataset", use_container_width=True):
        st.switch_page("pages/3_🗂️_Dataset_Explorer.py")

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.5rem 0;">
        <span class="material-symbols-outlined" style="font-size: 28px; color: #3b82f6;">pulmonology</span>
        <div>
            <div style="font-weight: 600; font-size: 0.95rem;">BioFusion</div>
            <div style="font-size: 0.7rem; opacity: 0.7;">Hackathon 2026</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

# Page Header
st.markdown("""
<div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
    <span class="material-symbols-outlined" style="font-size: 28px; color: #3b82f6;">radiology</span>
    <span style="font-size: 1.5rem; font-weight: 700;">Live Analysis</span>
</div>
""", unsafe_allow_html=True)

st.markdown("Upload a chest X-ray image for real-time pneumonia detection with AI-powered visualization.")

st.divider()

# Initialize model
@st.cache_resource
def get_model():
    return load_model()

with st.spinner("Loading AI model..."):
    model, device = get_model()

device_name = "GPU (CUDA)" if "cuda" in str(device) else "CPU"
st.success(f"✅ Model loaded • {device_name}")

# Main content
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("#### 📤 Upload Chest X-Ray")
    uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")

with col_info:
    with st.container(border=True):
        st.markdown("#### 💡 Guidelines")
        st.markdown("""
        - ✅ Use frontal chest X-rays
        - ✅ Ensure clear exposure
        - ✅ Optimized for pediatric
        - ✅ JPG, PNG, BMP formats
        """)

# Process uploaded image
if uploaded_file is not None:
    st.divider()
    
    image = load_image(uploaded_file)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("#### 🖼️ Original")
        st.image(image, use_container_width=True)
    
    with st.spinner("Analyzing..."):
        input_tensor = preprocess_image(image)
        pred_class, confidence, probabilities = predict(model, input_tensor, device)
        pred_label = get_prediction_label(pred_class)
        
        try:
            heatmap, overlay = create_gradcam_visualization(model, input_tensor, image, device, pred_class)
            gradcam_success = True
        except:
            gradcam_success = False
    
    with col2:
        st.markdown("#### 👁️ Heatmap")
        if gradcam_success:
            st.image(overlay, use_container_width=True)
        else:
            st.info("Heatmap unavailable")
    
    with col3:
        st.markdown("#### 🔍 Result")
        
        is_pneumonia = pred_label == "PNEUMONIA"
        
        if is_pneumonia:
            st.error(f"### ⚠️ {pred_label}")
            st.metric("Confidence", f"{confidence*100:.1f}%")
        else:
            st.success(f"### ✅ {pred_label}")
            st.metric("Confidence", f"{confidence*100:.1f}%")
    
    st.divider()
    
    st.markdown("#### 📊 Probability Distribution")
    
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        normal_prob = float(probabilities[0] * 100)
        st.progress(normal_prob / 100, text=f"Normal: {normal_prob:.1f}%")
    
    with prob_col2:
        pneumonia_prob = float(probabilities[1] * 100)
        st.progress(pneumonia_prob / 100, text=f"Pneumonia: {pneumonia_prob:.1f}%")
    
    st.warning("⚠️ **Disclaimer:** For research/demo only. Consult healthcare professionals for diagnosis.")

else:
    st.info("👆 Upload a chest X-ray image to begin analysis")
