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

# CSS with System Theme Detection
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

<style>
    :root {
        --bg-primary: #f8fafc;
        --bg-card: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --accent-primary: #2563eb;
        --accent-success: #16a34a;
        --accent-danger: #dc2626;
        --border-color: #e2e8f0;
        --shadow: rgba(0, 0, 0, 0.08);
    }
    
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-primary: #3b82f6;
            --accent-success: #22c55e;
            --accent-danger: #ef4444;
            --border-color: #334155;
            --shadow: rgba(0, 0, 0, 0.3);
        }
    }
    
    .main, .stApp { font-family: 'Inter', sans-serif !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    
    .page-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; flex-wrap: wrap; }
    .page-title { font-size: 1.5rem; font-weight: 700; color: var(--text-primary); }
    .page-subtitle { color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem; }
    
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 8px var(--shadow);
    }
    
    .card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    
    .upload-zone {
        background: var(--bg-card);
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 2rem 1rem;
        text-align: center;
    }
    
    .result-card {
        background: var(--bg-card);
        border: 2px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .result-card.normal { border-color: var(--accent-success); }
    .result-card.pneumonia { border-color: var(--accent-danger); }
    
    .result-label { font-size: 1.25rem; font-weight: 700; margin: 0.75rem 0 0.25rem; }
    .result-normal { color: var(--accent-success); }
    .result-pneumonia { color: var(--accent-danger); }
    .confidence-value { font-size: 1.75rem; font-weight: 700; }
    
    .progress-bar { background: var(--border-color); border-radius: 4px; height: 6px; overflow: hidden; margin-top: 0.5rem; }
    .progress-fill { height: 100%; border-radius: 4px; }
    
    .alert {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin-top: 1rem;
    }
    
    .divider { height: 1px; background: var(--border-color); margin: 1.25rem 0; }
    
    .info-list { list-style: none; padding: 0; margin: 0; }
    .info-list li { display: flex; align-items: center; gap: 0.5rem; padding: 0.3rem 0; font-size: 0.8rem; color: var(--text-secondary); }
    .info-list .material-symbols-outlined { font-size: 14px; color: var(--accent-primary); }
    
    .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; vertical-align: middle; }
    
    @media (max-width: 768px) {
        .page-title { font-size: 1.25rem; }
        .card { padding: 1rem; }
        .result-label { font-size: 1.1rem; }
        .confidence-value { font-size: 1.5rem; }
        .upload-zone { padding: 1.5rem 1rem; }
    }
    
    @media (max-width: 480px) {
        .page-title { font-size: 1.1rem; }
        .result-card { padding: 1rem; }
        .confidence-value { font-size: 1.25rem; }
    }
</style>
""", unsafe_allow_html=True)

# Page Header
st.markdown("""
<div class="page-header">
    <span class="material-symbols-outlined" style="font-size: 28px; color: var(--accent-primary);">radiology</span>
    <span class="page-title">Live Analysis</span>
</div>
<p class="page-subtitle">Upload a chest X-ray image for real-time pneumonia detection with AI-powered visualization.</p>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Initialize model
@st.cache_resource
def get_model():
    return load_model()

with st.spinner("Loading AI model..."):
    model, device = get_model()

device_name = "GPU (CUDA)" if "cuda" in str(device) else "CPU"
st.markdown(f"""
<div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.6rem 1rem; background: rgba(34, 197, 94, 0.1); border-radius: 8px; border: 1px solid rgba(34, 197, 94, 0.2); margin-bottom: 1rem; flex-wrap: wrap;">
    <span class="material-symbols-outlined" style="color: #22c55e; font-size: 18px;">check_circle</span>
    <span style="color: #22c55e; font-size: 0.8rem; font-weight: 500;">Model loaded</span>
    <span style="color: var(--text-muted); font-size: 0.8rem;">• {device_name}</span>
</div>
""", unsafe_allow_html=True)

# Main content
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown('<div class="card-title"><span class="material-symbols-outlined">upload_file</span>Upload Chest X-Ray</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")

with col_info:
    st.markdown("""
    <div class="card">
        <div class="card-title"><span class="material-symbols-outlined">tips_and_updates</span>Guidelines</div>
        <ul class="info-list">
            <li><span class="material-symbols-outlined">check</span>Use frontal chest X-rays</li>
            <li><span class="material-symbols-outlined">check</span>Ensure clear exposure</li>
            <li><span class="material-symbols-outlined">check</span>Optimized for pediatric</li>
            <li><span class="material-symbols-outlined">check</span>JPG, PNG, BMP</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Process uploaded image
if uploaded_file is not None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    image = load_image(uploaded_file)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown('<div class="card-title"><span class="material-symbols-outlined">image</span>Original</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="card-title"><span class="material-symbols-outlined">visibility</span>Heatmap</div>', unsafe_allow_html=True)
        if gradcam_success:
            st.image(overlay, use_container_width=True)
        else:
            st.info("Heatmap unavailable")
    
    with col3:
        st.markdown('<div class="card-title"><span class="material-symbols-outlined">diagnosis</span>Result</div>', unsafe_allow_html=True)
        
        is_pneumonia = pred_label == "PNEUMONIA"
        result_class = "pneumonia" if is_pneumonia else "normal"
        label_class = "result-pneumonia" if is_pneumonia else "result-normal"
        icon = "error" if is_pneumonia else "check_circle"
        icon_color = "#ef4444" if is_pneumonia else "#22c55e"
        
        st.markdown(f"""
        <div class="result-card {result_class}">
            <span class="material-symbols-outlined" style="font-size: 40px; color: {icon_color};">{icon}</span>
            <div class="result-label {label_class}">{pred_label}</div>
            <div style="color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.75rem;">Detection Result</div>
            <div class="confidence-value" style="color: {icon_color};">{confidence*100:.1f}%</div>
            <div style="color: var(--text-muted); font-size: 0.7rem;">Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="material-symbols-outlined">bar_chart</span>Probability</div>', unsafe_allow_html=True)
    
    prob_col1, prob_col2 = st.columns(2)
    
    with prob_col1:
        normal_prob = probabilities[0] * 100
        st.markdown(f"""
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #22c55e; font-weight: 500; font-size: 0.85rem;">Normal</span>
                <span style="color: #22c55e; font-weight: 600; font-size: 0.85rem;">{normal_prob:.1f}%</span>
            </div>
            <div class="progress-bar"><div class="progress-fill" style="background: #22c55e; width: {normal_prob}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)
    
    with prob_col2:
        pneumonia_prob = probabilities[1] * 100
        st.markdown(f"""
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #ef4444; font-weight: 500; font-size: 0.85rem;">Pneumonia</span>
                <span style="color: #ef4444; font-weight: 600; font-size: 0.85rem;">{pneumonia_prob:.1f}%</span>
            </div>
            <div class="progress-bar"><div class="progress-fill" style="background: #ef4444; width: {pneumonia_prob}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert">
        <span class="material-symbols-outlined" style="color: #f59e0b;">warning</span>
        <div style="font-size: 0.8rem; color: var(--text-secondary);"><strong>Disclaimer:</strong> For research/demo only. Consult healthcare professionals for diagnosis.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-zone">
        <span class="material-symbols-outlined" style="font-size: 40px; color: var(--text-muted);">cloud_upload</span>
        <div style="color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.75rem;">Drop X-ray image here or click to browse</div>
        <div style="color: var(--text-muted); font-size: 0.8rem; margin-top: 0.25rem;">JPG, PNG, BMP</div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.75rem 0;">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <span class="material-symbols-outlined" style="font-size: 28px; color: var(--accent-primary);">pulmonology</span>
            <div>
                <div style="font-weight: 600; color: var(--text-primary); font-size: 0.95rem;">BioFusion</div>
                <div style="font-size: 0.7rem; color: var(--text-muted);">Hackathon 2026</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="card">
        <div class="card-title"><span class="material-symbols-outlined">memory</span>Model Info</div>
        <ul class="info-list">
            <li><span class="material-symbols-outlined">architecture</span>ResNet50</li>
            <li><span class="material-symbols-outlined">aspect_ratio</span>224×224</li>
            <li><span class="material-symbols-outlined">developer_board</span>{device_name}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
