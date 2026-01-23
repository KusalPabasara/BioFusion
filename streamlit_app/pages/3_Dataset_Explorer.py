"""
Dataset Explorer Page - Pneumonia Detection
Explore the chest X-ray dataset used for training the model.
"""

import streamlit as st
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Dataset Analytics | Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="auto"
)

# Industry-Level CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

<style>
    /* Global Variables */
    :root {
        --primary-color: #2563eb;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --neutral-dark: #0f172a;
        --neutral-gray: #64748b;
    }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    
    /* Hiding Elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Components */
    .icon-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: #eff6ff;
        color: #2563eb;
        margin-right: 1rem;
        border: 1px solid rgba(37, 99, 235, 0.1);
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
    if st.button("Analysis", use_container_width=True):
        st.switch_page("pages/1_Live_Prediction.py")
with nav_cols[2]:
    if st.button("Metrics", use_container_width=True):
        st.switch_page("pages/2_Model_Insights.py")
with nav_cols[3]:
    if st.button("Dataset", use_container_width=True, type="primary"):
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
            <div style="font-weight: 600; font-size: 1rem;">BioFusion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

# Page Header
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 2rem;">
    <div class="icon-box">
        <span class="material-symbols-rounded" style="font-size: 28px;">database</span>
    </div>
    <div>
        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 600;">Dataset Analytics</h2>
        <p style="margin: 0; opacity: 0.6; font-size: 0.95rem;">Cohort demographics and quality assurance</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Dataset Overview
st.markdown("##### Cohort Summary")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    | Attribute | Details |
    |-----------|---------|
    | **Name** | Chest X-Ray Images (Pneumonia) |
    | **Source Warning** | Class Imbalance Present |
    | **Total Volume** | 5,863 JPEG Images |
    | **Demographics** | Pediatric (1-5 years) |
    | **Site** | Guangzhou Medical Center |
    | **Labels** | Binary: Normal / Pneumonia |
    """)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 2rem; border: 1px solid rgba(100,116,139,0.1); border-radius: 12px; background: rgba(37, 99, 235, 0.03);">
        <span class="material-symbols-rounded" style="font-size: 48px; color: #2563eb;">image</span>
        <div style="font-size: 3rem; font-weight: 700; color: #2563eb; line-height: 1.1; margin-top: 0.5rem; letter-spacing: -0.02em;">5,863</div>
        <div style="opacity: 0.6; font-size: 0.8rem; font-weight: 500;">TOTAL RADIOGRAPHS</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Class Distribution
st.markdown("##### Distribution Analysis")

col1, col2 = st.columns(2)

with col1:
    split_data = {"Split": ["Train", "Validation", "Test"], "Normal": [1341, 8, 234], "Pneumonia": [3875, 8, 390]}
    
    fig_split = go.Figure()
    # Sapphire for Normal, Amber for Pneumonia (Warning state concepts)
    fig_split.add_trace(go.Bar(name='Normal', x=split_data["Split"], y=split_data["Normal"], marker_color='#10b981', text=split_data["Normal"], textposition='inside'))
    fig_split.add_trace(go.Bar(name='Pneumonia', x=split_data["Split"], y=split_data["Pneumonia"], marker_color='#f59e0b', text=split_data["Pneumonia"], textposition='inside'))
    fig_split.update_layout(
        barmode='stack',
        title="Sample Distribution by Split",
        xaxis_title="Split",
        yaxis_title="Image Count",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    st.plotly_chart(fig_split, use_container_width=True)

with col2:
    # Emerald and Amber palette
    fig_pie = go.Figure(data=[go.Pie(labels=['Normal (Healthy)', 'Pneumonia (Pathology)'], values=[1583, 4273], hole=.55, marker_colors=['#10b981', '#f59e0b'], textinfo='percent')])
    fig_pie.update_layout(
        title="Global Class Balance",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        annotations=[dict(text='27:73', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.info("ℹ️ **Data Engineering:** Weighted Cross-Entropy Loss (weights=[0.74, 0.26]) applied during training to counteract class imbalance.")

st.divider()

# Data Quality
st.markdown("##### Quality Assurance Protocol")

quality_cols = st.columns(3)

quality_metrics = [
    ("Triple Grading", "Primary screening by expert physician, followed by two independent validator reviews.", "verified_user"),
    ("Clinical Origin", "Sourced from real-world pediatric inflows at major metropolitan medical center.", "local_hospital"),
    ("Adjudication", "Disagreements resolved by third senior expert review to ground truth.", "fact_check")
]

for col, (title, desc, icon) in zip(quality_cols, quality_metrics):
    with col:
        st.markdown(f"""
        <div style="padding: 1.5rem; height: 100%; border: 1px solid rgba(100,116,139,0.1); border-radius: 12px; background: #fff;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span class="material-symbols-rounded" style="color: #2563eb;">{icon}</span>
                <span style="font-weight: 600; color: #0f172a;">{title}</span>
            </div>
            <div style="font-size: 0.85rem; color: #64748b; line-height: 1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
