"""
Live Prediction Page - Pneumonia Detection
Upload chest X-ray images for AI-powered analysis with Grad-CAM visualization.
"""

import streamlit as st
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_model, predict, get_prediction_label, preprocess_image, load_image
from utils import create_gradcam_visualization

import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Live Prediction | Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

# Professional CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

<style>
    :root {
        --bg-primary: #1a1a2e;
        --bg-card: #1f2937;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-primary: #3b82f6;
        --accent-success: #22c55e;
        --accent-danger: #ef4444;
        --border-color: #334155;
    }
    
    .main { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    
    .page-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    .page-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .page-subtitle {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
    }
    
    .card-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .upload-zone {
        background: var(--bg-card);
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        transition: border-color 0.2s ease;
    }
    
    .upload-zone:hover {
        border-color: var(--accent-primary);
    }
    
    .result-card {
        background: var(--bg-card);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    
    .result-card.normal {
        border-color: var(--accent-success);
    }
    
    .result-card.pneumonia {
        border-color: var(--accent-danger);
    }
    
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1rem 0 0.5rem;
    }
    
    .result-normal { color: var(--accent-success); }
    .result-pneumonia { color: var(--accent-danger); }
    
    .confidence-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .progress-bar {
        background: var(--border-color);
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 4px;
    }
    
    .alert {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-top: 1.5rem;
    }
    
    .alert-icon {
        color: #f59e0b;
        flex-shrink: 0;
    }
    
    .alert-content {
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    .divider {
        height: 1px;
        background: var(--border-color);
        margin: 1.5rem 0;
    }
    
    .info-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .info-list li {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.375rem 0;
        font-size: 0.875rem;
        color: var(--text-secondary);
    }
    
    .info-list .material-symbols-outlined {
        font-size: 16px;
        color: var(--accent-primary);
    }
    
    .material-symbols-outlined {
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# Page Header
st.markdown("""
<div class="page-header">
    <span class="material-symbols-outlined" style="font-size: 32px; color: #3b82f6;">radiology</span>
    <span class="page-title">Live Analysis</span>
</div>
<p class="page-subtitle">Upload a chest X-ray image for real-time pneumonia detection with AI-powered visualization.</p>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Initialize model (cached)
@st.cache_resource
def get_model():
    """Load model once and cache it."""
    return load_model()

# Load model with spinner
with st.spinner("Loading AI model..."):
    model, device = get_model()

# Model status
device_name = "GPU (CUDA)" if "cuda" in str(device) else "CPU"
st.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.75rem 1rem; background: rgba(34, 197, 94, 0.1); border-radius: 8px; border: 1px solid rgba(34, 197, 94, 0.2); margin-bottom: 1.5rem;">
    <span class="material-symbols-outlined" style="color: #22c55e; font-size: 20px;">check_circle</span>
    <span style="color: #22c55e; font-size: 0.875rem; font-weight: 500;">Model loaded successfully</span>
    <span style="color: #64748b; font-size: 0.875rem; margin-left: 0.5rem;">Running on {device_name}</span>
</div>
""", unsafe_allow_html=True)

# Main content
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("""
    <div class="card-title">
        <span class="material-symbols-outlined">upload_file</span>
        Upload Chest X-Ray
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a frontal chest X-ray image (JPEG or PNG format)",
        label_visibility="collapsed"
    )

with col_info:
    st.markdown("""
    <div class="card">
        <div class="card-title">
            <span class="material-symbols-outlined">tips_and_updates</span>
            Guidelines
        </div>
        <ul class="info-list">
            <li>
                <span class="material-symbols-outlined">check</span>
                Use frontal (PA/AP) chest X-rays
            </li>
            <li>
                <span class="material-symbols-outlined">check</span>
                Ensure image is clear and well-exposed
            </li>
            <li>
                <span class="material-symbols-outlined">check</span>
                Optimized for pediatric X-rays
            </li>
            <li>
                <span class="material-symbols-outlined">check</span>
                Supported: JPG, PNG, BMP
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Process uploaded image
if uploaded_file is not None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Load and display original image
    image = load_image(uploaded_file)
    
    # Create columns for display
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        <div class="card-title">
            <span class="material-symbols-outlined">image</span>
            Original X-Ray
        </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    # Make prediction
    with st.spinner("Analyzing X-ray..."):
        # Preprocess
        input_tensor = preprocess_image(image)
        
        # Predict
        pred_class, confidence, probabilities = predict(model, input_tensor, device)
        pred_label = get_prediction_label(pred_class)
        
        # Generate Grad-CAM
        try:
            heatmap, overlay = create_gradcam_visualization(
                model, input_tensor, image, device, pred_class
            )
            gradcam_success = True
        except Exception as e:
            gradcam_success = False
    
    # Display Grad-CAM
    with col2:
        st.markdown("""
        <div class="card-title">
            <span class="material-symbols-outlined">visibility</span>
            Attention Heatmap
        </div>
        """, unsafe_allow_html=True)
        if gradcam_success:
            st.image(overlay, use_container_width=True)
        else:
            st.info("Heatmap not available")
    
    # Display prediction result
    with col3:
        st.markdown("""
        <div class="card-title">
            <span class="material-symbols-outlined">diagnosis</span>
            Analysis Result
        </div>
        """, unsafe_allow_html=True)
        
        is_pneumonia = pred_label == "PNEUMONIA"
        result_class = "pneumonia" if is_pneumonia else "normal"
        label_class = "result-pneumonia" if is_pneumonia else "result-normal"
        icon = "error" if is_pneumonia else "check_circle"
        icon_color = "#ef4444" if is_pneumonia else "#22c55e"
        conf_color = "#ef4444" if is_pneumonia else "#22c55e"
        
        st.markdown(f"""
        <div class="result-card {result_class}">
            <span class="material-symbols-outlined" style="font-size: 48px; color: {icon_color};">{icon}</span>
            <div class="result-label {label_class}">{pred_label}</div>
            <div style="color: #64748b; font-size: 0.875rem; margin-bottom: 1rem;">Detection Result</div>
            <div class="confidence-value" style="color: {conf_color};">{confidence*100:.1f}%</div>
            <div style="color: #64748b; font-size: 0.75rem;">Confidence Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Probability distribution
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card-title">
        <span class="material-symbols-outlined">bar_chart</span>
        Probability Distribution
    </div>
    """, unsafe_allow_html=True)
    
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        normal_prob = probabilities[0] * 100
        st.markdown(f"""
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #22c55e; font-weight: 500; display: flex; align-items: center; gap: 0.5rem;">
                    <span class="material-symbols-outlined" style="font-size: 18px;">check_circle</span>
                    Normal
                </span>
                <span style="color: #22c55e; font-weight: 600;">{normal_prob:.1f}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="background: #22c55e; width: {normal_prob}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with prob_col2:
        pneumonia_prob = probabilities[1] * 100
        st.markdown(f"""
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #ef4444; font-weight: 500; display: flex; align-items: center; gap: 0.5rem;">
                    <span class="material-symbols-outlined" style="font-size: 18px;">error</span>
                    Pneumonia
                </span>
                <span style="color: #ef4444; font-weight: 600;">{pneumonia_prob:.1f}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="background: #ef4444; width: {pneumonia_prob}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Clinical disclaimer
    st.markdown("""
    <div class="alert">
        <span class="material-symbols-outlined alert-icon">warning</span>
        <div class="alert-content">
            <strong>Clinical Disclaimer:</strong> This AI tool is for research and demonstration purposes only. 
            It should not be used as the sole basis for clinical decisions. 
            Always consult a qualified healthcare professional for medical diagnosis.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Show placeholder when no image uploaded
    st.markdown("""
    <div class="upload-zone">
        <span class="material-symbols-outlined" style="font-size: 48px; color: #64748b;">cloud_upload</span>
        <div style="color: #94a3b8; font-size: 1rem; margin-top: 1rem;">
            Drop your chest X-ray image here or click to browse
        </div>
        <div style="color: #64748b; font-size: 0.875rem; margin-top: 0.5rem;">
            Supported formats: JPG, PNG, BMP
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar info
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
            <span class="material-symbols-outlined" style="font-size: 32px; color: #3b82f6;">pulmonology</span>
            <div>
                <div style="font-weight: 600; color: #f8fafc;">BioFusion</div>
                <div style="font-size: 0.75rem; color: #64748b;">Hackathon 2026</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="card">
        <div class="card-title">
            <span class="material-symbols-outlined">memory</span>
            Model Information
        </div>
        <ul class="info-list">
            <li>
                <span class="material-symbols-outlined">architecture</span>
                ResNet50
            </li>
            <li>
                <span class="material-symbols-outlined">aspect_ratio</span>
                224 x 224 input
            </li>
            <li>
                <span class="material-symbols-outlined">developer_board</span>
                {device_name}
            </li>
            <li>
                <span class="material-symbols-outlined">category</span>
                Normal / Pneumonia
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
