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
    page_title="Live Analysis | Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="auto"
)

# Industry-Level CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

<style>
    /* Global Variables */
    :root {
        --primary-color: #2563eb;
        --success-color: #10b981;
        --warning-color: #f59e0b; /* Amber replacing Red */
        --neutral-dark: #0f172a;
        --neutral-gray: #64748b;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    
    /* Hiding Elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Components */
    .icon-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: #eff6ff;
        color: #2563eb;
        margin-right: 1rem;
        border: 1px solid rgba(37, 99, 235, 0.1);
    }
    
    .material-symbols-rounded {
        font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# ============ TOP NAVIGATION BAR ============
nav_cols = st.columns([1, 1, 1, 1])
with nav_cols[0]:
    if st.button("Home", use_container_width=True):
        st.switch_page("app.py")
with nav_cols[1]:
    if st.button("Analysis", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Live_Prediction.py")
with nav_cols[2]:
    if st.button("Metrics", use_container_width=True):
        st.switch_page("pages/2_Model_Insights.py")
with nav_cols[3]:
    if st.button("Dataset", use_container_width=True):
        st.switch_page("pages/3_Dataset_Explorer.py")

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 1rem; padding: 1rem 0;">
        <div style="background: rgba(37, 99, 235, 0.1); padding: 8px; border-radius: 8px;">
            <span class="material-symbols-rounded" style="color: #2563eb; font-size: 24px;">pulmonology</span>
        </div>
        <div>
            <div style="font-weight: 600; font-size: 1rem;">BioFusion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

# Page Header
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 2rem;">
    <div class="icon-box">
        <span class="material-symbols-rounded" style="font-size: 28px;">radiology</span>
    </div>
    <div>
        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 600;">Live Analysis</h2>
        <p style="margin: 0; opacity: 0.6; font-size: 0.95rem;">Real-time diagnostic inference engine</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Function to get model
@st.cache_resource
def get_model():
    return load_model()

with st.spinner("Initializing ResNet50 inference engine..."):
    model, device = get_model()

device_name = "CUDA Accelerated" if "cuda" in str(device) else "CPU Mode"
st.caption(f"Engine Status: Online • {device_name}")

# Main content
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("##### New Study")
    uploaded_file = st.file_uploader("Upload Radiograph", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")

with col_info:
    with st.container(border=True):
        st.markdown("##### Protocol")
        st.markdown("""
        <div style="font-size: 0.85rem; opacity: 0.8; line-height: 1.8;">
        1. Ensure AP/PA view<br>
        2. Verify patient ID hidden<br>
        3. Upload high-res JPEG/PNG<br>
        4. Confirm result with radiologist
        </div>
        """, unsafe_allow_html=True)

# Process uploaded image
if uploaded_file is not None:
    st.divider()
    
    image = load_image(uploaded_file)
    
    # Process
    with st.spinner("Processing image tensors..."):
        input_tensor = preprocess_image(image)
        pred_class, confidence, probabilities = predict(model, input_tensor, device)
        pred_label = get_prediction_label(pred_class)
        
        try:
            heatmap, overlay = create_gradcam_visualization(model, input_tensor, image, device, pred_class)
            gradcam_success = True
        except:
            gradcam_success = False

    # Displays
    col_viz1, col_viz2, col_result = st.columns([1, 1, 1.2])
    
    with col_viz1:
        st.markdown("**Source Image**")
        st.image(image, use_container_width=True)
    
    with col_viz2:
        st.markdown("**Grad-CAM Heatmap**")
        if gradcam_success:
            st.image(overlay, use_container_width=True)
        else:
            st.warning("Visualization unavailable")
    
    with col_result:
        st.markdown("**Diagnostic Output**")
        
        is_pneumonia = pred_label == "PNEUMONIA"
        
        # Clinical Coloring: Amber for Pneumonia (Warning), Emerald for Normal (Success)
        # Red is completely removed per user request
        
        icon = "priority_high" if is_pneumonia else "check_circle"
        
        # Professional Amber (#f59e0b) vs Emerald (#10b981)
        color = "#f59e0b" if is_pneumonia else "#10b981" 
        
        # Very subtle backgrounds
        bg_color = "rgba(245, 158, 11, 0.05)" if is_pneumonia else "rgba(16, 185, 129, 0.05)"
        
        st.markdown(f"""
        <div style="padding: 1.5rem; background: {bg_color}; border-radius: 12px; border: 1px solid {color}40; margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem; color: {color}; margin-bottom: 0.5rem;">
                <span class="material-symbols-rounded">{icon}</span>
                <span style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; font-size: 0.9rem;">{pred_label}</span>
            </div>
            <div style="font-size: 3rem; font-weight: 700; color: {color}; line-height: 1; letter-spacing: -0.03em;">
                {confidence*100:.1f}%
            </div>
            <div style="font-size: 0.8rem; opacity: 0.6; margin-top: 0.5rem; font-weight: 500;">CONFIDENCE SCORE</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probabilities
        st.markdown("###### Class Probability")
        
        normal_prob = float(probabilities[0] * 100)
        pneumonia_prob = float(probabilities[1] * 100)
        
        # Custom progress bars with new palette
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.25rem;">
                <span style="font-weight: 500;">Normal</span>
                <span style="font-family: monospace;">{normal_prob:.1f}%</span>
            </div>
            <div style="width: 100%; height: 6px; background: rgba(100,116,139,0.1); border-radius: 3px; overflow: hidden;">
                <div style="width: {normal_prob}%; height: 100%; background: #10b981;"></div>
            </div>
        </div>

        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.25rem;">
                <span style="font-weight: 500;">Pneumonia</span>
                <span style="font-family: monospace;">{pneumonia_prob:.1f}%</span>
            </div>
            <div style="width: 100%; height: 6px; background: rgba(100,116,139,0.1); border-radius: 3px; overflow: hidden;">
                <div style="width: {pneumonia_prob}%; height: 100%; background: #f59e0b;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.info("ℹ️ **MD Review Required:** AI results are supportive only. Final diagnosis must be made by a qualified physician.")

else:
    st.markdown("""
    <div style="padding: 4rem 2rem; text-align: center; opacity: 0.4; border: 2px dashed #94a3b8; border-radius: 12px; background: rgba(248,250,252, 0.5);">
        <span class="material-symbols-rounded" style="font-size: 48px; margin-bottom: 1rem;">cloud_upload</span>
        <p style="font-weight: 500;">Awaiting DICOM/JPEG Input</p>
    </div>
    """, unsafe_allow_html=True)
