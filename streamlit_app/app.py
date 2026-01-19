"""
Pneumonia Detection - Clinical Decision Support System
BioFusion Hackathon 2026 | Team GMora
Professional Streamlit Demo with System Theme Detection and Mobile Responsive Design
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection | BioFusion",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="auto"
)

# Professional CSS with System Theme Detection (prefers-color-scheme) and Mobile Responsive
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

<style>
    /* ============ LIGHT MODE (Default) ============ */
    :root {
        --bg-primary: #f8fafc;
        --bg-secondary: #ffffff;
        --bg-card: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --accent-primary: #2563eb;
        --accent-success: #16a34a;
        --accent-warning: #d97706;
        --accent-danger: #dc2626;
        --border-color: #e2e8f0;
        --shadow: rgba(0, 0, 0, 0.08);
    }
    
    /* ============ DARK MODE (System Preference) ============ */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-primary: #3b82f6;
            --accent-success: #22c55e;
            --accent-warning: #f59e0b;
            --accent-danger: #ef4444;
            --border-color: #334155;
            --shadow: rgba(0, 0, 0, 0.3);
        }
    }
    
    /* ============ GLOBAL STYLES ============ */
    .main, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* ============ CARD COMPONENT ============ */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px var(--shadow);
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
    
    /* ============ STAT COMPONENT ============ */
    .stat {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 8px var(--shadow);
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--accent-primary);
        line-height: 1.2;
    }
    
    .stat-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    .stat-desc {
        font-size: 0.65rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    
    /* ============ HERO SECTION ============ */
    .hero { padding: 1.5rem 0; }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
    }
    
    .hero-description {
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.6;
        max-width: 600px;
    }
    
    /* ============ BADGE COMPONENT ============ */
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
    
    /* ============ FEATURE CARD ============ */
    .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        height: 100%;
        box-shadow: 0 2px 8px var(--shadow);
    }
    
    .feature-icon {
        width: 44px;
        height: 44px;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.75rem;
    }
    
    .feature-icon .material-symbols-outlined {
        font-size: 22px;
        color: var(--accent-primary);
    }
    
    .feature-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }
    
    /* ============ SECTION HEADERS ============ */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* ============ DIVIDER ============ */
    .divider {
        height: 1px;
        background: var(--border-color);
        margin: 1.5rem 0;
    }
    
    /* ============ LIST STYLES ============ */
    .check-list {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .check-list li {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0;
        color: var(--text-secondary);
        font-size: 0.85rem;
    }
    
    .check-list .material-symbols-outlined {
        color: var(--accent-success);
        font-size: 16px;
    }
    
    /* ============ FOOTER ============ */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: var(--text-muted);
        border-top: 1px solid var(--border-color);
        margin-top: 2rem;
        font-size: 0.8rem;
    }
    
    /* ============ MATERIAL ICONS ============ */
    .material-symbols-outlined {
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }
    
    /* ============ MOBILE RESPONSIVE STYLES ============ */
    
    /* Tablets */
    @media (max-width: 992px) {
        .hero-title { font-size: 1.75rem; }
        .stat-value { font-size: 1.5rem; }
        .feature-card { padding: 1rem; }
    }
    
    /* Mobile phones */
    @media (max-width: 768px) {
        .hero { padding: 1rem 0; }
        
        .hero-title {
            font-size: 1.4rem;
            gap: 0.5rem;
        }
        
        .hero-title .material-symbols-outlined {
            font-size: 26px !important;
        }
        
        .hero-subtitle { font-size: 0.9rem; }
        .hero-description { font-size: 0.85rem; }
        
        .card, .stat, .feature-card {
            padding: 1rem;
            border-radius: 10px;
        }
        
        .stat-value { font-size: 1.35rem; }
        .stat-label { font-size: 0.6rem; }
        .section-header { font-size: 1rem; }
        
        .feature-icon {
            width: 38px;
            height: 38px;
        }
        
        .feature-icon .material-symbols-outlined {
            font-size: 18px;
        }
        
        .feature-title { font-size: 0.9rem; }
        .feature-desc { font-size: 0.75rem; }
        .divider { margin: 1rem 0; }
        .footer { padding: 1rem; font-size: 0.75rem; }
        .badge { font-size: 0.7rem; padding: 0.2rem 0.5rem; }
    }
    
    /* Small mobile phones */
    @media (max-width: 480px) {
        .hero-title { font-size: 1.2rem; }
        .stat-value { font-size: 1.15rem; }
        .card-title { font-size: 0.9rem; }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.75rem 0;">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;">
            <span class="material-symbols-outlined" style="font-size: 28px; color: var(--accent-primary);">pulmonology</span>
            <div>
                <div style="font-weight: 600; color: var(--text-primary); font-size: 0.95rem;">BioFusion</div>
                <div style="font-size: 0.7rem; color: var(--text-muted);">Hackathon 2026</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="divider" style="margin: 0.75rem 0;"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 0.5rem; padding-left: 0.25rem;">
        Navigation
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 0.5rem 0; margin-top: 1rem;">
        <div style="font-size: 0.7rem; color: var(--text-muted);">Team GMora</div>
        <div style="font-size: 0.65rem; color: var(--text-muted);">Clinical AI Research</div>
    </div>
    """, unsafe_allow_html=True)

# Main Content
# Hero Section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">
            <span class="material-symbols-outlined" style="font-size: 36px; color: var(--accent-primary);">pulmonology</span>
            Pneumonia Detection
        </div>
        <div class="hero-subtitle">AI-Powered Chest X-Ray Analysis System</div>
        <p class="hero-description">
            A clinical decision support tool leveraging <strong>ResNet50</strong> deep learning 
            to detect pneumonia from pediatric chest X-rays with <strong>96.67% sensitivity</strong>. 
            Built for rapid screening and triage in clinical environments.
        </p>
        <div style="display: flex; gap: 0.5rem; margin-top: 1rem; flex-wrap: wrap;">
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
    <div class="card" style="text-align: center; padding: 1.5rem;">
        <span class="material-symbols-outlined" style="font-size: 48px; color: var(--accent-primary);">radiology</span>
        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.75rem;">Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

# Action Buttons
col_btn1, col_btn2, col_btn3 = st.columns(3)

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
    ("13", "False Neg", "Out of 390", "#f59e0b"),
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
    ("neurology", "Deep Learning", "ResNet50 with transfer learning from ImageNet for medical imaging."),
    ("visibility", "Explainable AI", "Grad-CAM visualizations highlight lung regions for clinical validation."),
    ("bolt", "Real-Time", "Sub-second inference for rapid screening and triage."),
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
            <span class="material-symbols-outlined" style="color: var(--accent-danger);">warning</span>
            Global Burden
        </div>
        <ul class="check-list" style="margin-top: 0.75rem;">
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
                15% of deaths in children under 5
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">
            <span class="material-symbols-outlined" style="color: var(--accent-success);">check_circle</span>
            Our Solution
        </div>
        <ul class="check-list" style="margin-top: 0.75rem;">
            <li>
                <span class="material-symbols-outlined">check</span>
                Automated pre-screening for urgent cases
            </li>
            <li>
                <span class="material-symbols-outlined">check</span>
                Decision support for physicians
            </li>
            <li>
                <span class="material-symbols-outlined">check</span>
                Scalable for resource-limited settings
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div style="display: flex; align-items: center; justify-content: center; gap: 0.5rem; margin-bottom: 0.25rem;">
        <span class="material-symbols-outlined" style="font-size: 16px;">trophy</span>
        BioFusion Hackathon 2026 | Team GMora
    </div>
    <div>Built with PyTorch and Streamlit</div>
</div>
""", unsafe_allow_html=True)
