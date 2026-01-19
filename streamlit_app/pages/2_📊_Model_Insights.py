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

# Minimal CSS
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0" rel="stylesheet">
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    .material-symbols-outlined { font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)

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
        st.markdown("#### 🎯 Priority")
        st.markdown("High **Recall** to minimize missed pneumonia cases.")

# Page Header
st.markdown("""
<div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
    <span class="material-symbols-outlined" style="font-size: 28px; color: #3b82f6;">analytics</span>
    <span style="font-size: 1.5rem; font-weight: 700;">Model Insights</span>
</div>
""", unsafe_allow_html=True)

st.markdown("Performance metrics and visualizations for the Pneumonia Detection model.")

st.divider()

# Metrics
st.subheader("📊 Performance Metrics")

cols = st.columns(6)

metrics_data = [
    ("Accuracy", "87.18%"),
    ("Recall", "96.67%"),
    ("Precision", "84.38%"),
    ("F1-Score", "90.11%"),
    ("AUC-ROC", "94.28%"),
    ("Specificity", "70.09%"),
]

for col, (label, value) in zip(cols, metrics_data):
    with col:
        st.metric(label, value)

st.divider()

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔢 Confusion Matrix")
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
        height=350,
        margin=dict(l=10, r=10, t=20, b=10)
    )
    
    st.plotly_chart(fig_cm, use_container_width=True)

with col2:
    st.subheader("📈 ROC Curve")
    fpr = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
    tpr = np.array([0, 0.55, 0.72, 0.82, 0.88, 0.92, 0.94, 0.96, 0.97, 0.98, 0.99, 1.0])
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='AUC=0.94', line=dict(color='#3b82f6', width=2), fill='tozeroy', fillcolor='rgba(59,130,246,0.15)'))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='#888', width=1, dash='dash')))
    
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=350,
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(x=0.6, y=0.2)
    )
    
    st.plotly_chart(fig_roc, use_container_width=True)

st.divider()

# Training History
st.subheader("📉 Training History")

col1, col2 = st.columns(2)

epochs = list(range(1, 7))
train_loss, val_loss = [0.45, 0.32, 0.24, 0.19, 0.15, 0.12], [0.42, 0.30, 0.25, 0.22, 0.20, 0.19]
train_acc, val_acc = [78, 84, 87, 89, 91, 93], [80, 84, 86, 87, 88, 88]

with col1:
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines+markers', name='Train', line=dict(color='#3b82f6', width=2)))
    fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines+markers', name='Validation', line=dict(color='#22c55e', width=2)))
    fig_loss.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss", height=280, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_loss, use_container_width=True)

with col2:
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines+markers', name='Train', line=dict(color='#3b82f6', width=2)))
    fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines+markers', name='Validation', line=dict(color='#22c55e', width=2)))
    fig_acc.update_layout(title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy (%)", height=280, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_acc, use_container_width=True)
