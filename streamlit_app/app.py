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

# Industry-Level Professional CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

<style>
    /* ==================================
       1. Global Variables & Reset
       ================================== */
    :root {
        --primary-color: #2563eb;       /* Sapphire Blue */
        --primary-light: #eff6ff;
        --success-color: #10b981;       /* Emerald */
        --success-light: #ecfdf5;
        --warning-color: #f59e0b;       /* Amber - replacing Red */
        --warning-light: #fffbeb;
        --neutral-dark: #0f172a;        /* Slate 900 */
        --neutral-gray: #64748b;        /* Slate 500 */
        --neutral-light: #f8fafc;       /* Slate 50 */
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }
    
    /* Decoration Hiding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* ==================================
       2. Typography
       ================================== */
    h1, h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: var(--neutral-dark);
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: var(--neutral-gray);
        line-height: 1.6;
    }
    
    /* ==================================
       3. Component Library
       ================================== */
    
    /* Icon Box */
    .icon-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: var(--primary-light);
        color: var(--primary-color);
        margin-bottom: 1rem;
        border: 1px solid rgba(37, 99, 235, 0.1);
    }
    
    /* Cards */
    .feature-card {
        padding: 1.5rem;
        background: var(--neutral-light);
        border: 1px solid rgba(100, 116, 139, 0.1);
        border-radius: 12px;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-2px);
        border-color: rgba(37, 99, 235, 0.3);
    }
    
    /* Metric Card */
    .metric-container {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(100, 116, 139, 0.1);
        background: white;
    }
    
    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.85rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }
    
    .badge-primary {
        background: var(--primary-light);
        color: var(--primary-color);
        border: 1px solid rgba(37, 99, 235, 0.15);
    }
    
    .badge-success {
        background: var(--success-light);
        color: var(--success-color);
        border: 1px solid rgba(16, 185, 129, 0.15);
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: var(--primary-color) !important;
        border: none !important;
        transition: opacity 0.2s;
    }
    
    button[kind="primary"]:hover {
        opacity: 0.9;
    }

    /* Material Icons */
    .material-symbols-rounded {
        font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }
    
    /* Dark Mode Adjustments provided by Streamlit naturally, 
       but we ensure text visibility */
    @media (prefers-color-scheme: dark) {
        h1, h2, h3 { color: #f8fafc; }
        .subtitle { color: #94a3b8; }
        .feature-card { background: #1e293b; border-color: #334155; }
        .metric-container { background: #0f172a; border-color: #334155; }
    }
</style>
""", unsafe_allow_html=True)

# ============ TOP NAVIGATION BAR ============
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
        <div style="background: rgba(37, 99, 235, 0.1); padding: 8px; border-radius: 8px;">
            <span class="material-symbols-rounded" style="color: #2563eb; font-size: 24px;">pulmonology</span>
        </div>
        <div>
            <div style="font-weight: 600; font-size: 1rem; letter-spacing: -0.01em;">BioFusion</div>
            <div style="font-size: 0.75rem; opacity: 0.6;">Clinical AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("###### System Status")
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; opacity: 0.8;">
        <span class="material-symbols-rounded" style="font-size: 16px; color: #10b981;">check_circle</span>
        <span>Online v2.4</span>
    </div>
    """, unsafe_allow_html=True)

# Main Hero Section
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
            <span class="badge badge-primary">Clinical Decision Support System</span>
        </div>
        <h1 style="font-size: 3.5rem; line-height: 1.1; margin-bottom: 1.5rem;">
            Pneumonia Detection <br>
            <span style="opacity: 0.4;">Assistance Suite</span>
        </h1>
        <p class="subtitle" style="max-width: 600px;">
            Enterprise-grade diagnostic support leverage ResNet50 architecture.
            Designed for high-throughput clinical environments with explainable AI integration.
        </p>
    </div>
    
    <div style="display: flex; gap: 1rem; margin-top: 2rem; flex-wrap: wrap;">
        <span class="badge badge-success">
            <span class="material-symbols-rounded" style="font-size: 16px;">verified</span>
            ResNet50 Architecture
        </span>
        <span class="badge badge-success">
            <span class="material-symbols-rounded" style="font-size: 16px;">bolt</span>
            Real-time < 200ms
        </span>
         <span class="badge badge-success">
            <span class="material-symbols-rounded" style="font-size: 16px;">visibility</span>
            Explainable AI
        </span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Minimalist graphical element instead of complex gradient
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; height: 100%; min-height: 300px;">
        <div style="position: relative;">
            <span class="material-symbols-rounded" style="font-size: 160px; color: #f8fafc; text-shadow: 0 0 1px rgba(0,0,0,0.1);">radiology</span>
            <span class="material-symbols-rounded" style="font-size: 80px; color: #2563eb; position: absolute; bottom: -20px; right: -20px; background: white; border-radius: 50%; padding: 10px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">medical_services</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Key Performance Indicators
st.markdown("### Performance Metrics")

metric_cols = st.columns(4)
metrics = [
    {"label": "Sensitivity", "value": "96.67%", "desc": "Detection Rate"},
    {"label": "Accuracy", "value": "87.18%", "desc": "Global Test Set"},
    {"label": "AUC-ROC", "value": "0.942", "desc": "Discriminability"},
    {"label": "False Negatives", "value": "13", "desc": "Total Missed Cases"}
]

for col, metric in zip(metric_cols, metrics):
    with col:
        st.markdown(f"""
        <div class="metric-container">
            <div style="font-size: 0.85rem; font-weight: 500; opacity: 0.6; margin-bottom: 0.25rem;">{metric['label']}</div>
            <div style="font-size: 2rem; font-weight: 700; color: #2563eb; letter-spacing: -0.02em;">{metric['value']}</div>
            <div style="font-size: 0.75rem; color: #10b981; font-weight: 500;">{metric['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Core Capabilities
st.markdown("### Platform Features")

feat_cols = st.columns(3)

features = [
    {
        "title": "Deep Learning Core",
        "icon": "neurology",
        "desc": "ResNet50 model fine-tuned on 5,863 pediatric chest X-rays. Optimized for feature extraction in medical imaging contexts."
    },
    {
        "title": "Visual Explainability",
        "icon": "layers",
        "desc": "Integrated Gradient-weighted Class Activation Mapping (Grad-CAM) highlights regions of interest for validation."
    },
    {
        "title": "Clinical Integration",
        "icon": "integration_instructions",
        "desc": "Standardized input processing for DICOM-converted JPEG images, ready for PACS system interoperability."
    }
]

for col, feature in zip(feat_cols, features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div class="icon-box">
                <span class="material-symbols-rounded" style="font-size: 24px;">{feature['icon']}</span>
            </div>
            <h4 style="margin: 0.5rem 0; font-size: 1.1rem;">{feature['title']}</h4>
            <p style="font-size: 0.9rem; color: var(--neutral-gray); line-height: 1.5;">{feature['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()
st.caption("BioFusion 2026 | Team GMora • Clinical AI Research • Built with PyTorch")
