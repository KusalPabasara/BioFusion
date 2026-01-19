"""
Pneumonia Detection - Clinical Decision Support System
BioFusion Hackathon 2026 | Team GMora
Professional Streamlit Demo with DaisyUI Styling
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection | BioFusion",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with DaisyUI-inspired styling and Google Material Icons
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

<style>
    /* Root Variables - Professional Color Palette */
    :root {
        --bg-primary: #1a1a2e;
        --bg-secondary: #16213e;
        --bg-card: #1f2937;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-primary: #3b82f6;
        --accent-success: #22c55e;
        --accent-warning: #f59e0b;
        --accent-danger: #ef4444;
        --border-color: #334155;
    }
    
    /* Global Styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Card Component - DaisyUI Style */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }
    
    .card:hover {
        border-color: var(--accent-primary);
    }
    
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Stat Component */
    .stat {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-primary);
        line-height: 1.2;
    }
    
    .stat-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    .stat-desc {
        font-size: 0.7rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    
    /* Hero Section */
    .hero {
        padding: 2rem 0;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .hero-subtitle {
        font-size: 1.125rem;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
    }
    
    .hero-description {
        font-size: 0.95rem;
        color: var(--text-secondary);
        line-height: 1.6;
        max-width: 600px;
    }
    
    /* Badge Component */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .badge-primary {
        background: rgba(59, 130, 246, 0.15);
        color: var(--accent-primary);
    }
    
    .badge-success {
        background: rgba(34, 197, 94, 0.15);
        color: var(--accent-success);
    }
    
    /* Feature Card */
    .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
    }
    
    .feature-icon {
        width: 48px;
        height: 48px;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    
    .feature-icon .material-symbols-outlined {
        font-size: 24px;
        color: var(--accent-primary);
    }
    
    .feature-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.875rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Alert Component */
    .alert {
        border-radius: 10px;
        padding: 1rem 1.25rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .alert-info {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .alert-info .material-symbols-outlined {
        color: var(--accent-primary);
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: var(--border-color);
        margin: 2rem 0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    
    /* Navigation Link */
    .nav-link {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        color: var(--text-secondary);
        text-decoration: none;
        font-size: 0.875rem;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    
    .nav-link:hover {
        background: rgba(59, 130, 246, 0.1);
        color: var(--text-primary);
    }
    
    .nav-link.active {
        background: rgba(59, 130, 246, 0.15);
        color: var(--accent-primary);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-muted);
        border-top: 1px solid var(--border-color);
        margin-top: 3rem;
        font-size: 0.875rem;
    }
    
    /* Material Icons */
    .material-symbols-outlined {
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }
    
    /* List Styles */
    .check-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .check-list li {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0;
        color: var(--text-secondary);
        font-size: 0.9rem;
    }
    
    .check-list .material-symbols-outlined {
        color: var(--accent-success);
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
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
    
    st.markdown('<div class="divider" style="margin: 1rem 0;"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin-bottom: 0.75rem; padding-left: 1rem;">
        Navigation
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 0.25rem;">
        <div class="nav-link active">
            <span class="material-symbols-outlined">home</span>
            Home
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider" style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 0 1rem;">
        <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.5rem;">Team GMora</div>
        <div style="font-size: 0.7rem; color: #475569;">Clinical AI Research</div>
    </div>
    """, unsafe_allow_html=True)

# Main Content
# Hero Section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            <span class="material-symbols-outlined" style="font-size: 40px; color: #3b82f6;">pulmonology</span>
            Pneumonia Detection
        </div>
        <div class="hero-subtitle">AI-Powered Chest X-Ray Analysis System</div>
        <p class="hero-description">
            A clinical decision support tool leveraging <strong>ResNet50</strong> deep learning 
            to detect pneumonia from pediatric chest X-rays with <strong>96.67% sensitivity</strong>. 
            Built for rapid screening and triage in clinical environments.
        </p>
        <div style="display: flex; gap: 0.5rem; margin-top: 1.25rem;">
            <span class="badge badge-primary">
                <span class="material-symbols-outlined" style="font-size: 14px;">verified</span>
                ResNet50
            </span>
            <span class="badge badge-success">
                <span class="material-symbols-outlined" style="font-size: 14px;">speed</span>
                Real-time
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card" style="text-align: center; padding: 2rem;">
        <span class="material-symbols-outlined" style="font-size: 64px; color: #3b82f6;">radiology</span>
        <div style="font-size: 0.875rem; color: #94a3b8; margin-top: 1rem;">Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

# Action Buttons
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

with col_btn1:
    if st.button("Start Analysis", type="primary", use_container_width=True):
        st.switch_page("pages/1_🔬_Live_Prediction.py")

with col_btn2:
    if st.button("View Metrics", use_container_width=True):
        st.switch_page("pages/2_📊_Model_Insights.py")

with col_btn3:
    if st.button("Dataset Info", use_container_width=True):
        st.switch_page("pages/3_🗂️_Dataset_Explorer.py")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Key Metrics Section
st.markdown("""
<div class="section-header">
    <span class="material-symbols-outlined">analytics</span>
    Model Performance
</div>
""", unsafe_allow_html=True)

metric_cols = st.columns(4)

metrics = [
    ("96.67%", "Recall", "Sensitivity", "#22c55e"),
    ("87.18%", "Accuracy", "Overall", "#3b82f6"),
    ("0.9428", "AUC-ROC", "Discrimination", "#8b5cf6"),
    ("13", "False Negatives", "Out of 390", "#f59e0b"),
]

for col, (value, label, desc, color) in zip(metric_cols, metrics):
    with col:
        st.markdown(f"""
        <div class="stat">
            <div class="stat-value" style="color: {color};">{value}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Features Section
st.markdown("""
<div class="section-header">
    <span class="material-symbols-outlined">star</span>
    Key Features
</div>
""", unsafe_allow_html=True)

feature_cols = st.columns(3)

features = [
    ("neurology", "Deep Learning", "ResNet50 architecture with transfer learning from ImageNet, achieving state-of-the-art performance on medical imaging."),
    ("visibility", "Explainable AI", "Grad-CAM visualizations highlight lung regions influencing predictions, enabling clinical validation."),
    ("bolt", "Real-Time Analysis", "Sub-second inference enables rapid screening and triage in time-critical clinical settings."),
]

for col, (icon, title, desc) in zip(feature_cols, features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">
                <span class="material-symbols-outlined">{icon}</span>
            </div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Problem Statement
st.markdown("""
<div class="section-header">
    <span class="material-symbols-outlined">info</span>
    Clinical Context
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">
            <span class="material-symbols-outlined" style="color: #ef4444;">warning</span>
            Global Burden
        </div>
        <ul class="check-list" style="margin-top: 1rem;">
            <li>
                <span class="material-symbols-outlined">arrow_forward</span>
                450 million cases annually worldwide
            </li>
            <li>
                <span class="material-symbols-outlined">arrow_forward</span>
                4 million deaths globally each year
            </li>
            <li>
                <span class="material-symbols-outlined">arrow_forward</span>
                15% of all deaths in children under 5
            </li>
            <li>
                <span class="material-symbols-outlined">arrow_forward</span>
                Leading infectious cause of death in children
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">
            <span class="material-symbols-outlined" style="color: #22c55e;">check_circle</span>
            Our Solution
        </div>
        <ul class="check-list" style="margin-top: 1rem;">
            <li>
                <span class="material-symbols-outlined">check</span>
                Automated pre-screening for urgent cases
            </li>
            <li>
                <span class="material-symbols-outlined">check</span>
                Decision support for non-specialist physicians
            </li>
            <li>
                <span class="material-symbols-outlined">check</span>
                Scalable deployment in resource-limited settings
            </li>
            <li>
                <span class="material-symbols-outlined">check</span>
                Consistent, fatigue-free diagnostic assistance
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 0.5rem;">
        <span class="material-symbols-outlined" style="font-size: 18px;">trophy</span>
        BioFusion Hackathon 2026 | Team GMora
    </div>
    <div>Built with PyTorch and Streamlit</div>
</div>
""", unsafe_allow_html=True)
