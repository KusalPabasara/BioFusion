"""
Dataset Explorer Page - Pneumonia Detection
Explore the chest X-ray dataset used for training the model.
"""

import streamlit as st
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Dataset Explorer | Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

# Initialize theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# CSS with Light/Dark Mode and Mobile Responsiveness
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

<style>
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
    
    [data-theme="light"] {
        --bg-primary: #f8fafc;
        --bg-card: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --accent-primary: #2563eb;
        --accent-success: #16a34a;
        --accent-danger: #dc2626;
        --border-color: #e2e8f0;
        --shadow: rgba(0, 0, 0, 0.1);
    }
    
    .main, .stApp { font-family: 'Inter', sans-serif; background-color: var(--bg-primary) !important; }
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stSidebar"] { background: var(--bg-card) !important; }
    
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
    
    .stat-large {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 8px var(--shadow);
    }
    
    .stat-large-value { font-size: 2rem; font-weight: 700; color: var(--accent-primary); }
    .stat-large-label { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem; }
    
    .section-header {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .divider { height: 1px; background: var(--border-color); margin: 1.25rem 0; }
    
    .alert-warning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .quality-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        height: 100%;
        box-shadow: 0 2px 8px var(--shadow);
    }
    
    .quality-icon {
        width: 36px;
        height: 36px;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.5rem;
    }
    
    .quality-title { font-size: 0.8rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem; }
    .quality-desc { font-size: 0.7rem; color: var(--text-secondary); line-height: 1.4; }
    
    .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; vertical-align: middle; }
    
    @media (max-width: 768px) {
        .page-title { font-size: 1.25rem; }
        .card, .quality-card { padding: 1rem; }
        .stat-large { padding: 1rem; }
        .stat-large-value { font-size: 1.5rem; }
        .quality-icon { width: 32px; height: 32px; }
        .section-header { font-size: 0.9rem; }
    }
    
    @media (max-width: 480px) {
        .page-title { font-size: 1.1rem; }
        .stat-large-value { font-size: 1.3rem; }
    }
</style>
""", unsafe_allow_html=True)

# Apply theme
theme_class = "light" if st.session_state.theme == "light" else "dark"
st.markdown(f'<script>document.documentElement.setAttribute("data-theme", "{theme_class}");</script>', unsafe_allow_html=True)

# Page Header
st.markdown("""
<div class="page-header">
    <span class="material-symbols-outlined" style="font-size: 28px; color: var(--accent-primary);">database</span>
    <span class="page-title">Dataset Explorer</span>
</div>
<p class="page-subtitle">Explore the chest X-ray dataset used to train the Pneumonia Detection model.</p>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Dataset Overview
st.markdown('<div class="section-header"><span class="material-symbols-outlined">info</span>Dataset Overview</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    | Attribute | Details |
    |-----------|---------|
    | **Name** | Chest X-Ray Images (Pneumonia) |
    | **Source** | Kaggle / Kermany et al., 2018 |
    | **Total Images** | 5,863 chest X-rays (JPEG) |
    | **Demographics** | Pediatric patients, ages 1-5 |
    | **Institution** | Guangzhou Women and Children's Medical Center |
    | **Classes** | Normal, Pneumonia |
    """)

with col2:
    st.markdown("""
    <div class="stat-large">
        <span class="material-symbols-outlined" style="font-size: 32px; color: var(--accent-primary);">image</span>
        <div class="stat-large-value">5,863</div>
        <div class="stat-large-label">Total X-Ray Images</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Class Distribution
st.markdown('<div class="section-header"><span class="material-symbols-outlined">bar_chart</span>Class Distribution</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

is_light = st.session_state.theme == "light"
text_color = "#0f172a" if is_light else "#f8fafc"
bg_color = "rgba(0,0,0,0)"

with col1:
    split_data = {"Split": ["Train", "Validation", "Test"], "Normal": [1341, 8, 234], "Pneumonia": [3875, 8, 390]}
    
    fig_split = go.Figure()
    fig_split.add_trace(go.Bar(name='Normal', x=split_data["Split"], y=split_data["Normal"], marker_color='#22c55e', text=split_data["Normal"], textposition='inside', textfont=dict(color='white')))
    fig_split.add_trace(go.Bar(name='Pneumonia', x=split_data["Split"], y=split_data["Pneumonia"], marker_color='#ef4444', text=split_data["Pneumonia"], textposition='inside', textfont=dict(color='white')))
    fig_split.update_layout(barmode='stack', title=dict(text='Distribution by Split', font=dict(size=12, color=text_color)), xaxis_title="Split", yaxis_title="Count", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color, family='Inter', size=10), height=280, margin=dict(l=10, r=10, t=35, b=10), legend=dict(x=0.7, y=0.95, bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_split, use_container_width=True)

with col2:
    fig_pie = go.Figure(data=[go.Pie(labels=['Normal', 'Pneumonia'], values=[1583, 4273], hole=.45, marker_colors=['#22c55e', '#ef4444'], textinfo='label+percent', textfont_size=11, textfont_color='white')])
    fig_pie.update_layout(title=dict(text='Overall Distribution', font=dict(size=12, color=text_color)), paper_bgcolor=bg_color, font=dict(color=text_color, family='Inter', size=10), height=280, margin=dict(l=10, r=10, t=35, b=10), showlegend=False, annotations=[dict(text='5,856', x=0.5, y=0.5, font_size=14, font_color=text_color, showarrow=False)])
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("""
<div class="alert-warning">
    <span class="material-symbols-outlined" style="color: #f59e0b; flex-shrink: 0;">warning</span>
    <div style="font-size: 0.8rem; color: var(--text-secondary);">
        <strong style="color: #f59e0b;">Class Imbalance (27:73)</strong> — Addressed using <strong>weighted cross-entropy loss</strong> during training.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Data Quality
st.markdown('<div class="section-header"><span class="material-symbols-outlined">verified</span>Data Quality</div>', unsafe_allow_html=True)

quality_cols = st.columns(3)
quality_items = [("clinical_notes", "Expert Annotation", "Screened by expert physicians, graded by two additional experts"), ("local_hospital", "Clinical Source", "Real patient records from Guangzhou Medical Center"), ("fact_check", "Quality Control", "Third expert reviewed any disagreements")]

for col, (icon, title, desc) in zip(quality_cols, quality_items):
    with col:
        st.markdown(f'<div class="quality-card"><div class="quality-icon"><span class="material-symbols-outlined" style="color: var(--accent-primary); font-size: 18px;">{icon}</span></div><div class="quality-title">{title}</div><div class="quality-desc">{desc}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Citation
st.markdown('<div class="section-header"><span class="material-symbols-outlined">menu_book</span>Citation</div>', unsafe_allow_html=True)
st.code('@article{kermany2018identifying,\n  title={Identifying medical diagnoses and treatable diseases by image-based deep learning},\n  author={Kermany et al.},\n  journal={Cell}, volume={172}, pages={1122--1131}, year={2018}\n}', language="bibtex")

# Sidebar
with st.sidebar:
    st.markdown('<div style="padding: 0.75rem 0;"><div style="display: flex; align-items: center; gap: 0.75rem;"><span class="material-symbols-outlined" style="font-size: 28px; color: var(--accent-primary);">pulmonology</span><div><div style="font-weight: 600; color: var(--text-primary); font-size: 0.95rem;">BioFusion</div><div style="font-size: 0.7rem; color: var(--text-muted);">Hackathon 2026</div></div></div></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("☀️ Light", use_container_width=True, type="secondary" if st.session_state.theme == "dark" else "primary"):
            st.session_state.theme = "light"
            st.rerun()
    with col2:
        if st.button("🌙 Dark", use_container_width=True, type="secondary" if st.session_state.theme == "light" else "primary"):
            st.session_state.theme = "dark"
            st.rerun()
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card"><div class="card-title"><span class="material-symbols-outlined">summarize</span>Quick Stats</div></div>', unsafe_allow_html=True)
    st.metric("Total Images", "5,863")
    st.metric("Normal", "1,583 (27%)")
    st.metric("Pneumonia", "4,273 (73%)")
