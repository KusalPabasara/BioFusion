"""
Model Insights Page - Pneumonia Detection
Display model performance metrics, confusion matrix, ROC curve, and training history.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Page config
st.set_page_config(
    page_title="Model Insights | Pneumonia Detection",
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
    
    .stat {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px var(--shadow);
    }
    
    .stat-value { font-size: 1.5rem; font-weight: 700; line-height: 1.2; }
    .stat-label { font-size: 0.65rem; font-weight: 500; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }
    
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
    
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; vertical-align: middle; }
    
    @media (max-width: 768px) {
        .page-title { font-size: 1.25rem; }
        .stat { padding: 0.75rem; }
        .stat-value { font-size: 1.25rem; }
        .stat-label { font-size: 0.6rem; }
        .card { padding: 1rem; }
        .section-header { font-size: 0.9rem; }
    }
    
    @media (max-width: 480px) {
        .page-title { font-size: 1.1rem; }
        .stat-value { font-size: 1.1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Apply theme
theme_class = "light" if st.session_state.theme == "light" else "dark"
st.markdown(f'<script>document.documentElement.setAttribute("data-theme", "{theme_class}");</script>', unsafe_allow_html=True)

# Page Header
st.markdown("""
<div class="page-header">
    <span class="material-symbols-outlined" style="font-size: 28px; color: var(--accent-primary);">analytics</span>
    <span class="page-title">Model Insights</span>
</div>
<p class="page-subtitle">Performance metrics and visualizations for the Pneumonia Detection model.</p>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Metrics
st.markdown('<div class="section-header"><span class="material-symbols-outlined">monitoring</span>Performance Metrics</div>', unsafe_allow_html=True)

metrics_data = [
    ("87.18%", "Accuracy", "#3b82f6"),
    ("96.67%", "Recall", "#22c55e"),
    ("84.38%", "Precision", "#8b5cf6"),
    ("90.11%", "F1-Score", "#f59e0b"),
    ("94.28%", "AUC-ROC", "#ec4899"),
    ("70.09%", "Specificity", "#06b6d4"),
]

cols = st.columns(6)
for col, (value, label, color) in zip(cols, metrics_data):
    with col:
        st.markdown(f'<div class="stat"><div class="stat-value" style="color: {color};">{value}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Charts
col1, col2 = st.columns(2)

# Get colors based on theme
is_light = st.session_state.theme == "light"
text_color = "#0f172a" if is_light else "#f8fafc"
bg_color = "rgba(0,0,0,0)"

with col1:
    st.markdown('<div class="section-header"><span class="material-symbols-outlined">grid_on</span>Confusion Matrix</div>', unsafe_allow_html=True)
    
    cm_data = np.array([[164, 70], [13, 377]])
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_data,
        x=['Pred Normal', 'Pred Pneumonia'],
        y=['Actual Normal', 'Actual Pneumonia'],
        text=cm_data,
        texttemplate="%{text}",
        textfont={"size": 16, "color": "white"},
        colorscale=[[0, '#1e3a5f'], [0.5, '#3b82f6'], [1, '#22c55e']],
        showscale=False
    ))
    
    fig_cm.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, family='Inter', size=11),
        height=300,
        margin=dict(l=10, r=10, t=20, b=10)
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    st.markdown('<div class="section-header"><span class="material-symbols-outlined">show_chart</span>ROC Curve</div>', unsafe_allow_html=True)
    
    fpr = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    tpr = np.array([0, 0.55, 0.72, 0.82, 0.88, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1.0])
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='AUC=0.94', line=dict(color='#3b82f6', width=2), fill='tozeroy', fillcolor='rgba(59,130,246,0.15)'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='#64748b', width=1, dash='dash')))
    
    fig_roc.update_layout(
        xaxis_title="FPR",
        yaxis_title="TPR",
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color, family='Inter', size=11),
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(x=0.6, y=0.2, bgcolor='rgba(0,0,0,0)')
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Training History
st.markdown('<div class="section-header"><span class="material-symbols-outlined">timeline</span>Training History</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

epochs = list(range(1, 7))
train_loss = [0.45, 0.32, 0.24, 0.19, 0.15, 0.12]
val_loss = [0.42, 0.30, 0.25, 0.22, 0.20, 0.19]
train_acc = [78, 84, 87, 89, 91, 93]
val_acc = [80, 84, 86, 87, 88, 88]

with col1:
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Train', line=dict(color='#3b82f6', width=2), marker=dict(size=5)))
    fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Val', line=dict(color='#22c55e', width=2), marker=dict(size=5)))
    fig_loss.update_layout(title=dict(text='Loss', font=dict(size=12, color=text_color)), xaxis_title="Epoch", yaxis_title="Loss", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color, family='Inter', size=10), height=250, margin=dict(l=10, r=10, t=30, b=10), legend=dict(x=0.7, y=0.95, bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_loss, use_container_width=True)

with col2:
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Train', line=dict(color='#3b82f6', width=2), marker=dict(size=5)))
    fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Val', line=dict(color='#22c55e', width=2), marker=dict(size=5)))
    fig_acc.update_layout(title=dict(text='Accuracy', font=dict(size=12, color=text_color)), xaxis_title="Epoch", yaxis_title="Accuracy (%)", paper_bgcolor=bg_color, plot_bgcolor=bg_color, font=dict(color=text_color, family='Inter', size=10), height=250, margin=dict(l=10, r=10, t=30, b=10), legend=dict(x=0.7, y=0.2, bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_acc, use_container_width=True)

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
    
    st.markdown('<div class="card"><div class="card-title"><span class="material-symbols-outlined">target</span>Priority</div><p style="font-size: 0.8rem; color: var(--text-secondary); margin: 0;">High <strong style="color: #22c55e;">Recall</strong> to minimize missed cases.</p></div>', unsafe_allow_html=True)
