"""
Screening — the core BioFusion flow.

Patients/parents and clinicians use the same page and the same model; the role
selector tunes how the result is shown. Input can be a live camera capture (for
phones) or an uploaded file, and either a digital X-ray or a photo of a film.
Every image passes a quality gate before inference; photos are perspective- and
contrast-corrected first. Results are shown as a triage band with escalation
guidance, never as a diagnosis.
"""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (load_model, predict, preprocess_image, load_image,
                   create_gradcam_visualization, rectify_mobile_xray,
                   assess_quality, triage, ui)

st.set_page_config(page_title="Screening | BioFusion", page_icon="🫁",
                   layout="wide", initial_sidebar_state="collapsed")

ui.inject_theme()
ui.top_nav(active="Screening")
st.divider()

# --- model + fail-safe threshold ------------------------------------------- #
WEIGHTS_PATH = Path(__file__).parent.parent.parent / "pneumonia_resnet50_best.pth"
METRICS_PATH = Path(__file__).parent.parent.parent / "training_metrics.json"


@st.cache_resource
def get_model():
    weights = str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else None
    model, device = load_model(weights)
    return model, device, weights is not None


@st.cache_data
def get_threshold():
    """Fail-safe decision threshold from training; default 0.5 if unavailable."""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH) as f:
                return float(json.load(f)["operating_point"]["threshold"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            pass
    return 0.5


with st.spinner("Starting the screening engine…"):
    model, device, using_trained_weights = get_model()
threshold = get_threshold()

# --- header + role --------------------------------------------------------- #
ui.page_header("Chest X-ray screening",
               "Take a photo or upload an X-ray to check for signs of pneumonia.")

role = ui.role_selector()
clinician = ui.is_clinician()

if not using_trained_weights:
    st.warning(
        "⚠️ **Demo mode:** trained weights not found — running on raw ImageNet "
        "weights, so predictions are **not** clinically meaningful."
    )

# --- input ----------------------------------------------------------------- #
st.markdown("##### 1 · Provide the X-ray")

source = st.radio("Source", ["Take a photo", "Upload a file"], horizontal=True,
                  label_visibility="collapsed")
is_photo = source == "Take a photo"

captured = None
if is_photo:
    st.caption("Point your camera straight at the X-ray film, fill the frame, and avoid glare.")
    captured = st.camera_input("Capture X-ray", label_visibility="collapsed")
else:
    file_is_photo = st.checkbox(
        "This is a phone photo of a film (not a digital X-ray)",
        help="We'll correct perspective and contrast before screening.",
    )
    captured = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png", "bmp"],
                                label_visibility="collapsed")
    is_photo = file_is_photo

# --- process --------------------------------------------------------------- #
if captured is not None:
    st.divider()
    original_image = load_image(captured)

    # 2 · Quality gate — refuse to run on unsuitable input (before the animation).
    #     Relaxed for phone photos (soft focus + warm cast are expected there).
    quality = assess_quality(original_image, phone_mode=is_photo)
    for w in quality.warnings:
        st.warning(f"⚠️ {w}")
    if not quality.ok:
        st.error("**This image can't be screened reliably:**")
        for issue in quality.issues:
            st.markdown(f"- {issue}")
        st.stop()

    # 3 · Staged analysis animation over the real pipeline steps. Each stage's
    #     work runs inside its own step so the walkthrough is honest, not faked.
    state = {}

    def _prepare():
        if is_photo:
            rect = rectify_mobile_xray(original_image)   # run once, reuse below
            state["rect"] = rect
            state["image"] = rect.image
        else:
            state["image"] = original_image

    def _infer():
        img = state["image"]
        tensor = preprocess_image(img)
        state["tensor"] = tensor
        pred_class, confidence, probabilities = predict(model, tensor, device)
        state["pred_class"] = pred_class
        state["probs"] = probabilities

    def _triage():
        state["triage"] = triage(float(state["probs"][1]), threshold)

    stages = [("Checking image quality", None)]
    if is_photo:
        stages.append(("Correcting perspective & contrast", _prepare))
    else:
        stages.append(("Preparing the radiograph", _prepare))
    stages += [
        ("Running the screening model", _infer),
        ("Preparing your result", _triage),
    ]
    ui.run_analysis_animation(stages)

    image = state["image"]
    input_tensor = state["tensor"]
    pred_class = state["pred_class"]
    probabilities = state["probs"]
    t = state["triage"]

    if is_photo:
        rect = state["rect"]
        (st.success if rect.rectified else st.info)(f"📱 {rect.detail}")

    st.markdown("##### 2 · Result")
    result_col, image_col = st.columns([1.1, 1])

    with result_col:
        ui.render_result_card(t, clinician=clinician)

    with image_col:
        st.image(image, caption="Screened image", use_container_width=True)
        if is_photo:
            with st.expander("Original photo"):
                st.image(original_image, use_container_width=True)

    # 5 · Clinicians get Grad-CAM + probability detail.
    if clinician:
        st.divider()
        st.markdown("##### Explainability (Grad-CAM)")
        try:
            with st.spinner("Generating heatmap…"):
                heatmap, overlay = create_gradcam_visualization(
                    model, input_tensor, image, device, pred_class)
            gc1, gc2 = st.columns(2)
            with gc1:
                st.image(image, caption="Input", use_container_width=True)
            with gc2:
                st.image(overlay, caption="Grad-CAM — regions driving the prediction",
                         use_container_width=True)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            st.warning("Grad-CAM unavailable for this image.")
            st.caption(f"Reason: {exc}")

        st.markdown("**Class probabilities**")
        pc1, pc2 = st.columns(2)
        pc1.metric("Normal", f"{probabilities[0]*100:.1f}%")
        pc2.metric("Pneumonia", f"{probabilities[1]*100:.1f}%")

else:
    st.markdown("""
    <div style="padding:3rem 2rem; text-align:center; color:var(--ink-60);
         border:2px dashed #d7dee8; border-radius:12px; background:#FAFBFD; margin-top:1rem;">
        <p style="font-weight:500; margin:0;">Take a photo or upload an X-ray to begin</p>
    </div>
    """, unsafe_allow_html=True)
