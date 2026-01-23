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

# Professional CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

<style>
    /* Global Styles */
    :root { --primary-color: #3b82f6; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    
    /* Decoration Hiding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Icon Box */
    .icon-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: rgba(59, 130, 246, 0.1);
        color: #3b82f6;
        margin-right: 1rem;
    }
    
    /* Result Cards */
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.1);
        background: var(--background-color);
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
        <div style="background: rgba(59, 130, 246, 0.1); padding: 8px; border-radius: 8px;">
            <span class="material-symbols-rounded" style="color: #3b82f6; font-size: 24px;">pulmonology</span>
        </div>
        <div>
            <div style="font-weight: 600; font-size: 1rem;">BioFusion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

# Page Header
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
    <div class="icon-box">
        <span class="material-symbols-rounded" style="font-size: 28px;">radiology</span>
    </div>
    <div>
        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 600;">Live Analysis</h2>
        <p style="margin: 0; opacity: 0.6; font-size: 0.95rem;">Real-time AI diagnostics with explainability</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize model
@st.cache_resource
def get_model():
    return load_model()

with st.spinner("Initializing diagnostic engine..."):
    model, device = get_model()

device_name = "GPU Accelerated" if "cuda" in str(device) else "CPU Mode"
st.caption(f"System Ready • {device_name}")

# Main content
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("##### Upload Radiograph")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")

with col_info:
    with st.container(border=True):
        st.markdown("##### Input Guidelines")
        st.markdown("""
        <div style="font-size: 0.85rem; opacity: 0.8; line-height: 1.8;">
        ✓ Frontal Chest X-Ray<br>
        ✓ Pediatric or Adult<br>
        ✓ High Resolution JPEG/PNG<br>
        ✓ Ensure clear lung field visibility
        </div>
        """, unsafe_allow_html=True)

# Process uploaded image
if uploaded_file is not None:
    st.divider()
    
    image = load_image(uploaded_file)
    
    # Process
    with st.spinner("Analyzing lung fields..."):
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
        st.markdown("**Original Source**")
        st.image(image, use_container_width=True)
    
    with col_viz2:
        st.markdown("**Attention Map (Grad-CAM)**")
        if gradcam_success:
            st.image(overlay, use_container_width=True)
        else:
            st.warning("Visualization unavailable")
    
    with col_result:
        st.markdown("**Diagnostic Result**")
        
        is_pneumonia = pred_label == "PNEUMONIA"
        icon = "warning" if is_pneumonia else "check_circle"
        color = "#ef4444" if is_pneumonia else "#22c55e"
        bg_color = "rgba(239, 68, 68, 0.1)" if is_pneumonia else "rgba(34, 197, 94, 0.1)"
        
        st.markdown(f"""
        <div style="padding: 1.5rem; background: {bg_color}; border-radius: 12px; border: 1px solid {color}30; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem; color: {color}; margin-bottom: 0.5rem;">
                <span class="material-symbols-rounded">{icon}</span>
                <span style="font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">{pred_label}</span>
            </div>
            <div style="font-size: 2.5rem; font-weight: 700; color: {color}; line-height: 1;">
                {confidence*100:.1f}%
            </div>
            <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.5rem;">Confidence Score</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probabilities
        st.markdown("###### Probability Distribution")
        
        normal_prob = float(probabilities[0] * 100)
        pneumonia_prob = float(probabilities[1] * 100)
        
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
            <span>Normal</span>
            <span>{normal_prob:.1f}%</span>
        </div>
        <div style="width: 100%; height: 6px; background: rgba(128,128,128,0.2); border-radius: 3px; overflow: hidden; margin-bottom: 0.75rem;">
            <div style="width: {normal_prob}%; height: 100%; background: #22c55e;"></div>
        </div>

        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.2rem;">
            <span>Pneumonia</span>
            <span>{pneumonia_prob:.1f}%</span>
        </div>
        <div style="width: 100%; height: 6px; background: rgba(128,128,128,0.2); border-radius: 3px; overflow: hidden;">
            <div style="width: {pneumonia_prob}%; height: 100%; background: #ef4444;"></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.info("ℹ️ **Clinical Note:** This result is generated by an AI model and should be verified by a qualified radiologist.")

else:
    st.markdown("""
    <div style="padding: 3rem; text-align: center; opacity: 0.5; border: 2px dashed gray; border-radius: 12px;">
        <span class="material-symbols-rounded" style="font-size: 48px; margin-bottom: 1rem;">cloud_upload</span>
        <p>Awaiting Image Upload</p>
    </div>
    """, unsafe_allow_html=True)
