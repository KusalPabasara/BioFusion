"""
Pneumonia Detection - Clinical Decision Support System
BioFusion Hackathon 2026 | Team GMora
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection | BioFusion",
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
    :root {
        --primary-color: #3b82f6;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Hide Streamlit Decorations */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Typography */
    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    /* Custom Components */
    .icon-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: rgba(59, 130, 246, 0.1);
        color: #3b82f6;
        margin-bottom: 1rem;
    }
    
    .feature-card {
        padding: 1.5rem;
        border-radius: 16px;
        background-color: var(--background-color);
        border: 1px solid rgba(128, 128, 128, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    
    .badge-primary {
        background: rgba(59, 130, 246, 0.1);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .badge-success {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }

    /* Material Icons */
    .material-symbols-rounded {
        font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# ============ TOP NAVIGATION BAR ============
# Clean text-based navigation for mobile compatibility
nav_cols = st.columns([1, 1, 1, 1])
with nav_cols[0]:
    if st.button("Home", use_container_width=True, type="primary"):
        st.switch_page("app.py")
with nav_cols[1]:
    if st.button("Analysis", use_container_width=True):
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
            <div style="font-weight: 600; font-size: 1rem; letter-spacing: -0.01em;">BioFusion</div>
            <div style="font-size: 0.75rem; opacity: 0.6;">Hackathon 2026</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("###### Status")
    st.info("System Online • v1.0.0")

# Main Hero Section
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <div class="icon-box">
                <span class="material-symbols-rounded" style="font-size: 28px;">medical_services</span>
            </div>
            <span style="font-size: 0.9rem; font-weight: 600; color: #3b82f6; letter-spacing: 0.05em; text-transform: uppercase;">
                Clinical Decision Support
            </span>
        </div>
        <h1 style="font-size: 3rem; line-height: 1.1; margin-bottom: 1rem;">
            Pneumonia Detection <br>
            <span style="opacity: 0.5;">Assistant</span>
        </h1>
        <p style="font-size: 1.1rem; opacity: 0.8; line-height: 1.6; max-width: 600px;">
            Advanced AI diagnostics leveraging ResNet50 architecture to detect pediatric pneumonia with clinical-grade accuracy.
            Designed for rapid triage in high-volume environments.
        </p>
    </div>
    
    <div style="display: flex; gap: 1rem; margin-top: 2rem; flex-wrap: wrap;">
        <span class="badge badge-primary">
            <span class="material-symbols-rounded" style="font-size: 16px;">psychology</span>
            ResNet50 Architecture
        </span>
        <span class="badge badge-success">
            <span class="material-symbols-rounded" style="font-size: 16px;">bolt</span>
            Real-time Inference
        </span>
         <span class="badge badge-primary">
            <span class="material-symbols-rounded" style="font-size: 16px;">visibility</span>
            Grad-CAM Analysis
        </span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Stylized Hero Visual
    st.markdown("""
    <div style="background: radial-gradient(circle at center, rgba(59, 130, 246, 0.1) 0%, transparent 70%); padding: 3rem; text-align: center;">
        <span class="material-symbols-rounded" style="font-size: 120px; color: #3b82f6; opacity: 0.9;">radiology</span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Key Performance Indicators
st.markdown("### System Performance")

metric_cols = st.columns(4)
metrics = [
    {"label": "Sensitivity (Recall)", "value": "96.67%", "desc": "Detection Rate", "icon": "check_circle"},
    {"label": "Global Accuracy", "value": "87.18%", "desc": "Test Set Evaluation", "icon": "analytics"},
    {"label": "AUC-ROC Score", "value": "0.9428", "desc": "Model Robustness", "icon": "area_chart"},
    {"label": "Inference Time", "value": "< 200ms", "desc": "Per X-Ray Image", "icon": "timer"}
]

for col, metric in zip(metric_cols, metrics):
    with col:
        st.markdown(f"""
        <div style="padding: 1rem; border-radius: 12px; background: rgba(59, 130, 246, 0.03); border: 1px solid rgba(59, 130, 246, 0.1);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 0.8rem; font-weight: 500; opacity: 0.7;">{metric['label']}</span>
                <span class="material-symbols-rounded" style="font-size: 18px; opacity: 0.5;">{metric['icon']}</span>
            </div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #3b82f6;">{metric['value']}</div>
            <div style="font-size: 0.75rem; opacity: 0.5; margin-top: 0.25rem;">{metric['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Feature Grid
st.markdown("### Core Capabilities")

feat_cols = st.columns(3)

features = [
    {
        "title": "Deep Learning Engine",
        "icon": "neurology",
        "desc": "Powered by a fine-tuned ResNet50 model pretrained on ImageNet, optimized for medical imaging features."
    },
    {
        "title": "Explainable AI",
        "icon": "layers",
        "desc": "Integrated Grad-CAM visualization extracts activation maps to highlight suspicious regions."
    },
    {
        "title": "Clinical Workflow",
        "icon": "clinical_notes",
        "desc": "Seamless integration ready for PACS systems with standardized DICOM/JPEG compatibility."
    }
]

for col, feature in zip(feat_cols, features):
    with col:
        st.markdown(f"""
        <div style="padding: 1.5rem; height: 100%;">
            <div class="icon-box">
                <span class="material-symbols-rounded" style="font-size: 24px;">{feature['icon']}</span>
            </div>
            <h4 style="margin: 0.5rem 0; font-size: 1.1rem;">{feature['title']}</h4>
            <p style="font-size: 0.9rem; opacity: 0.7; line-height: 1.5;">{feature['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.caption("BioFusion Hackathon 2026 | Team GMora")
