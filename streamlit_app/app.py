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

# Minimal CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    .main { font-family: 'Inter', sans-serif !important; }
    .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; vertical-align: middle; }
    
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-primary { background: rgba(59, 130, 246, 0.15); color: #3b82f6; }
    .badge-success { background: rgba(34, 197, 94, 0.15); color: #22c55e; }
</style>
""", unsafe_allow_html=True)

# ============ TOP NAVIGATION BAR ============
nav_cols = st.columns(4)
with nav_cols[0]:
    if st.button("🏠 Home", use_container_width=True, type="primary"):
        st.switch_page("app.py")
with nav_cols[1]:
    if st.button("🔬 Predict", use_container_width=True):
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
    st.caption("Team GMora • Clinical AI Research")

# Main Content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 36px; color: #3b82f6;">pulmonology</span>
        <span style="font-size: 2rem; font-weight: 700;">Pneumonia Detection</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**AI-Powered Chest X-Ray Analysis System**")
    
    st.markdown("""
    A clinical decision support tool leveraging **ResNet50** deep learning 
    to detect pneumonia from pediatric chest X-rays with **96.67% sensitivity**. 
    Built for rapid screening and triage in clinical environments.
    """)
    
    st.markdown("""
    <div style="display: flex; gap: 0.5rem; margin-top: 1rem;">
        <span class="badge badge-primary">
            <span class="material-symbols-outlined" style="font-size: 14px;">verified</span>
            ResNet50
        </span>
        <span class="badge badge-success">
            <span class="material-symbols-outlined" style="font-size: 14px;">speed</span>
            Real-time
        </span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem;">
        <span class="material-symbols-outlined" style="font-size: 48px; color: #3b82f6;">radiology</span>
        <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 0.75rem;">Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Key Metrics Section
st.subheader("📈 Model Performance")

metric_cols = st.columns(4)

with metric_cols[0]:
    st.metric("Recall", "96.67%", help="Sensitivity - ability to detect pneumonia cases")

with metric_cols[1]:
    st.metric("Accuracy", "87.18%", help="Overall accuracy on test set")

with metric_cols[2]:
    st.metric("AUC-ROC", "0.9428", help="Area under ROC curve")

with metric_cols[3]:
    st.metric("False Negatives", "13", help="Missed cases out of 390")

st.divider()

# Features Section
st.subheader("⭐ Key Features")

feature_cols = st.columns(3)

with feature_cols[0]:
    st.markdown("#### 🧠 Deep Learning")
    st.markdown("ResNet50 with transfer learning from ImageNet for medical imaging.")

with feature_cols[1]:
    st.markdown("#### 👁️ Explainable AI")
    st.markdown("Grad-CAM visualizations highlight lung regions for clinical validation.")

with feature_cols[2]:
    st.markdown("#### ⚡ Real-Time")
    st.markdown("Sub-second inference for rapid screening and triage.")

st.divider()

# Problem Statement
st.subheader("ℹ️ Clinical Context")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ⚠️ Global Burden")
    st.markdown("""
    - 450 million cases annually worldwide
    - 4 million deaths globally each year
    - 15% of deaths in children under 5
    """)

with col2:
    st.markdown("#### ✅ Our Solution")
    st.markdown("""
    - Automated pre-screening for urgent cases
    - Decision support for physicians
    - Scalable for resource-limited settings
    """)

# Footer
st.divider()
st.caption("🏆 BioFusion Hackathon 2026 | Team GMora • Built with PyTorch and Streamlit")
