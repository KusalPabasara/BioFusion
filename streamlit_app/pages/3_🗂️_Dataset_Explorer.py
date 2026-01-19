"""
Dataset Explorer Page - Pneumonia Detection
Explore the chest X-ray dataset used for training the model.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page config
st.set_page_config(
    page_title="Dataset Explorer | Pneumonia Detection",
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
    
    .stat-large {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
    }
    
    .stat-large-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--accent-primary);
    }
    
    .stat-large-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }
    
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .divider {
        height: 1px;
        background: var(--border-color);
        margin: 1.5rem 0;
    }
    
    .alert-warning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 10px;
        padding: 1rem 1.25rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .quality-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        height: 100%;
    }
    
    .quality-icon {
        width: 40px;
        height: 40px;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.75rem;
    }
    
    .quality-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .quality-desc {
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.4;
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
    <span class="material-symbols-outlined" style="font-size: 32px; color: #3b82f6;">database</span>
    <span class="page-title">Dataset Explorer</span>
</div>
<p class="page-subtitle">Explore the chest X-ray dataset used to train the Pneumonia Detection model.</p>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Dataset Overview
st.markdown("""
<div class="section-header">
    <span class="material-symbols-outlined">info</span>
    Dataset Overview
</div>
""", unsafe_allow_html=True)

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
    | **Image Type** | Anterior-Posterior (AP) X-rays |
    """)

with col2:
    st.markdown("""
    <div class="stat-large">
        <span class="material-symbols-outlined" style="font-size: 36px; color: #3b82f6;">image</span>
        <div class="stat-large-value">5,863</div>
        <div class="stat-large-label">Total X-Ray Images</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Class Distribution
st.markdown("""
<div class="section-header">
    <span class="material-symbols-outlined">bar_chart</span>
    Class Distribution
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Dataset split breakdown
    split_data = {
        "Split": ["Train", "Validation", "Test"],
        "Normal": [1341, 8, 234],
        "Pneumonia": [3875, 8, 390],
    }
    
    fig_split = go.Figure()
    
    fig_split.add_trace(go.Bar(
        name='Normal',
        x=split_data["Split"],
        y=split_data["Normal"],
        marker_color='#22c55e',
        text=split_data["Normal"],
        textposition='inside',
        textfont=dict(color='white')
    ))
    
    fig_split.add_trace(go.Bar(
        name='Pneumonia',
        x=split_data["Split"],
        y=split_data["Pneumonia"],
        marker_color='#ef4444',
        text=split_data["Pneumonia"],
        textposition='inside',
        textfont=dict(color='white')
    ))
    
    fig_split.update_layout(
        barmode='stack',
        title=dict(text='Distribution by Split', font=dict(size=14, color='#f8fafc')),
        xaxis_title="Split",
        yaxis_title="Count",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc', family='Inter'),
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.8, y=0.95, bgcolor='rgba(0,0,0,0)')
    )
    
    st.plotly_chart(fig_split, use_container_width=True)

with col2:
    # Pie chart
    total_normal = 1583
    total_pneumonia = 4273
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Normal', 'Pneumonia'],
        values=[total_normal, total_pneumonia],
        hole=.45,
        marker_colors=['#22c55e', '#ef4444'],
        textinfo='label+percent',
        textfont_size=12,
        textfont_color='white'
    )])
    
    fig_pie.update_layout(
        title=dict(text='Overall Distribution', font=dict(size=14, color='#f8fafc')),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#f8fafc', family='Inter'),
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        annotations=[dict(
            text=f'{total_normal + total_pneumonia}',
            x=0.5, y=0.5,
            font_size=18,
            font_color='#f8fafc',
            showarrow=False
        )]
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)

# Class Imbalance Warning
st.markdown("""
<div class="alert-warning">
    <span class="material-symbols-outlined" style="color: #f59e0b; flex-shrink: 0;">warning</span>
    <div style="font-size: 0.875rem; color: #94a3b8;">
        <strong style="color: #f59e0b;">Class Imbalance (27:73 ratio)</strong><br>
        The dataset has more pneumonia cases than normal. We address this using 
        <strong>weighted cross-entropy loss</strong> during training:<br>
        Normal weight: ~1.94 (upweighted) | Pneumonia weight: ~0.67 (downweighted)
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Data Quality Section
st.markdown("""
<div class="section-header">
    <span class="material-symbols-outlined">verified</span>
    Data Quality Assurance
</div>
""", unsafe_allow_html=True)

quality_cols = st.columns(3)

quality_items = [
    ("clinical_notes", "Expert Annotation", "All images screened by expert physicians and graded by two additional experts"),
    ("local_hospital", "Clinical Source", "Real patient records from Guangzhou Women and Children's Medical Center"),
    ("fact_check", "Quality Control", "Third expert reviewed any disagreements in the evaluation set"),
]

for col, (icon, title, desc) in zip(quality_cols, quality_items):
    with col:
        st.markdown(f"""
        <div class="quality-card">
            <div class="quality-icon">
                <span class="material-symbols-outlined" style="color: #3b82f6;">{icon}</span>
            </div>
            <div class="quality-title">{title}</div>
            <div class="quality-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Citation
st.markdown("""
<div class="section-header">
    <span class="material-symbols-outlined">menu_book</span>
    Citation
</div>
""", unsafe_allow_html=True)

st.code("""
@article{kermany2018identifying,
  title={Identifying medical diagnoses and treatable diseases by image-based deep learning},
  author={Kermany, Daniel S and Goldbaum, Michael and Cai, Wenjia and others},
  journal={Cell},
  volume={172},
  number={5},
  pages={1122--1131},
  year={2018},
  publisher={Elsevier}
}
""", language="bibtex")

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
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <div class="card-title">
            <span class="material-symbols-outlined">summarize</span>
            Quick Stats
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.metric("Total Images", "5,863")
    st.metric("Normal", "1,583 (27%)")
    st.metric("Pneumonia", "4,273 (73%)")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="padding: 0 0.5rem;">
        <div style="font-size: 0.75rem; color: #64748b; margin-bottom: 0.25rem;">Publication</div>
        <div style="font-size: 0.85rem; color: #94a3b8;">Cell, Vol 172, Issue 5</div>
        <div style="font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">February 2018</div>
    </div>
    """, unsafe_allow_html=True)
