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
    layout="wide",
    initial_sidebar_state="auto"
)

# Professional CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">

<style>
    /* Global Styles */
    :root { --primary-color: #3b82f6; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    
    /* Decoration Hiding */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Icon Box */
    .icon-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        background: rgba(59, 130, 246, 0.1);
        color: #3b82f6;
        margin-right: 1rem;
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
    if st.button("Metrics", use_container_width=True, type="primary"):
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
            <div style="font-weight: 600; font-size: 1rem;">BioFusion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    with st.container(border=True):
        st.markdown("**Optimization Goal**")
        st.markdown("Maximize sensitivity (Recall) to ensure no pneumonia cases are missed.")

# Page Header
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 2rem;">
    <div class="icon-box">
        <span class="material-symbols-rounded" style="font-size: 28px;">analytics</span>
    </div>
    <div>
        <h2 style="margin: 0; font-size: 1.8rem; font-weight: 600;">Model Analytics</h2>
        <p style="margin: 0; opacity: 0.6; font-size: 0.95rem;">Performance evaluation and training diagnostics</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Metrics
cols = st.columns(6)

metrics_data = [
    ("Accuracy", "87.18%", "Test Accuracy"),
    ("Recall", "96.67%", "Sensitivity"),
    ("Precision", "84.38%", "PPV"),
    ("F1-Score", "90.11%", "Harmonic Mean"),
    ("AUC-ROC", "94.28%", "Discriminability"),
    ("Specificity", "70.09%", "TNR"),
]

for col, (label, value, desc) in zip(cols, metrics_data):
    with col:
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="font-size: 0.8rem; opacity: 0.7;">{label}</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #3b82f6;">{value}</div>
            <div style="font-size: 0.7rem; opacity: 0.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("##### Confusion Matrix")
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
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    st.markdown("##### ROC Curve")
    fpr = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    tpr = np.array([0, 0.55, 0.72, 0.82, 0.88, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1.0])
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='AUC=0.94', line=dict(color='#3b82f6', width=2), fill='tozeroy', fillcolor='rgba(59,130,246,0.15)'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(color='#888', width=1, dash='dash')))
    
    fig_roc.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        legend=dict(x=0.6, y=0.1)
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)

st.divider()
st.markdown("##### Training History")

col1, col2 = st.columns(2)

epochs = list(range(1, 7))
train_loss, val_loss = [0.45, 0.32, 0.24, 0.19, 0.15, 0.12], [0.42, 0.30, 0.25, 0.22, 0.20, 0.19]
train_acc, val_acc = [78, 84, 87, 89, 91, 93], [80, 84, 86, 87, 88, 88]

with col1:
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Train', line=dict(color='#3b82f6', width=2)))
    fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Val', line=dict(color='#22c55e', width=2)))
    fig_loss.update_layout(title="Loss Curve", height=280, margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Inter, sans-serif"), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_loss, use_container_width=True)

with col2:
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Train', line=dict(color='#3b82f6', width=2)))
    fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Val', line=dict(color='#22c55e', width=2)))
    fig_acc.update_layout(title="Accuracy Curve", height=280, margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Inter, sans-serif"), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_acc, use_container_width=True)
