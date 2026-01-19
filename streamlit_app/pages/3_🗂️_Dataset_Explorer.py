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
    layout="wide",
    initial_sidebar_state="auto"
)

# Minimal CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)

# ============ TOP NAVIGATION BAR ============
nav_cols = st.columns(4)
with nav_cols[0]:
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("app.py")
with nav_cols[1]:
    if st.button("🔬 Predict", use_container_width=True):
        st.switch_page("pages/1_🔬_Live_Prediction.py")
with nav_cols[2]:
    if st.button("📊 Metrics", use_container_width=True):
        st.switch_page("pages/2_📊_Model_Insights.py")
with nav_cols[3]:
    if st.button("🗂️ Dataset", use_container_width=True, type="primary"):
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
    
    with st.container(border=True):
        st.markdown("#### 📋 Quick Stats")
        st.metric("Total Images", "5,863")
        st.metric("Normal", "1,583 (27%)")
        st.metric("Pneumonia", "4,273 (73%)")

# Page Header
st.markdown("""
<div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
    <span class="material-symbols-outlined" style="font-size: 28px; color: #3b82f6;">database</span>
    <span style="font-size: 1.5rem; font-weight: 700;">Dataset Explorer</span>
</div>
""", unsafe_allow_html=True)

st.markdown("Explore the chest X-ray dataset used to train the Pneumonia Detection model.")

st.divider()

# Dataset Overview
st.subheader("ℹ️ Dataset Overview")

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
    with st.container(border=True):
        st.markdown("""
        <div style="text-align: center;">
            <span class="material-symbols-outlined" style="font-size: 48px; color: #3b82f6;">image</span>
            <div style="font-size: 2rem; font-weight: 700; color: #3b82f6;">5,863</div>
            <div style="opacity: 0.7;">Total X-Ray Images</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Class Distribution
st.subheader("📊 Class Distribution")

col1, col2 = st.columns(2)

with col1:
    split_data = {"Split": ["Train", "Validation", "Test"], "Normal": [1341, 8, 234], "Pneumonia": [3875, 8, 390]}
    
    fig_split = go.Figure()
    fig_split.add_trace(go.Bar(name='Normal', x=split_data["Split"], y=split_data["Normal"], marker_color='#22c55e', text=split_data["Normal"], textposition='inside'))
    fig_split.add_trace(go.Bar(name='Pneumonia', x=split_data["Split"], y=split_data["Pneumonia"], marker_color='#ef4444', text=split_data["Pneumonia"], textposition='inside'))
    fig_split.update_layout(barmode='stack', title="Distribution by Split", xaxis_title="Split", yaxis_title="Count", height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_split, use_container_width=True)

with col2:
    fig_pie = go.Figure(data=[go.Pie(labels=['Normal', 'Pneumonia'], values=[1583, 4273], hole=.45, marker_colors=['#22c55e', '#ef4444'], textinfo='label+percent')])
    fig_pie.update_layout(title="Overall Distribution", height=320, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

st.warning("⚠️ **Class Imbalance (27:73)** — Addressed using weighted cross-entropy loss during training.")

st.divider()

# Data Quality
st.subheader("✅ Data Quality")

quality_cols = st.columns(3)

with quality_cols[0]:
    with st.container(border=True):
        st.markdown("#### 📝 Expert Annotation")
        st.markdown("Screened by expert physicians, graded by two additional experts.")

with quality_cols[1]:
    with st.container(border=True):
        st.markdown("#### 🏥 Clinical Source")
        st.markdown("Real patient records from Guangzhou Women and Children's Medical Center.")

with quality_cols[2]:
    with st.container(border=True):
        st.markdown("#### ✔️ Quality Control")
        st.markdown("Third expert reviewed any disagreements between graders.")

st.divider()

# Citation
st.subheader("📚 Citation")
st.code('''@article{kermany2018identifying,
  title={Identifying medical diagnoses and treatable diseases by image-based deep learning},
  author={Kermany et al.},
  journal={Cell}, volume={172}, pages={1122--1131}, year={2018}
}''', language="bibtex")
