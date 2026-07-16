"""
Image-quality gatekeeper.

Before the model ever runs, we screen the input for problems that would make a
prediction untrustworthy: too small, out of focus, or clearly not a chest
radiograph. Catching these up front prevents confident-looking predictions on
garbage input — a core safety requirement for a screening tool.

The checks are deliberately simple, fast, and explainable (each returns a reason
the UI can show the user), not a second ML model.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

# Thresholds tuned to be permissive (avoid rejecting real X-rays) while still
# catching obvious problems. They can be adjusted without touching call sites.
MIN_SIDE = 128            # px; smaller than this loses diagnostic detail
BLUR_VAR_THRESHOLD = 40.0  # variance of Laplacian; lower = blurrier
GRAY_SAT_THRESHOLD = 0.12  # mean HSV saturation above this => too colourful for CXR


@dataclass
class QualityResult:
    ok: bool                       # passed all hard checks
    score: float                   # 0..1 rough overall quality score
    issues: list = field(default_factory=list)   # hard failures (block inference)
    warnings: list = field(default_factory=list)  # soft flags (allow, but caution)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Focus measure: variance of the Laplacian. Higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_saturation(bgr: np.ndarray) -> float:
    """Mean saturation in [0,1]. Chest X-rays are near-grayscale (low saturation)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean()) / 255.0


def assess_quality(pil_image: Image.Image, phone_mode: bool = False) -> QualityResult:
    """Screen an image for suitability as a chest-X-ray model input.

    Returns a QualityResult; when `ok` is False the caller should refuse to run
    inference and show `issues` to the user instead.

    When `phone_mode` is True the checks are relaxed for phone photos of films:
    soft focus and a warm colour cast are *expected* (and the model is trained
    to handle them), so we only reject genuinely unusable images — far too
    blurry, too small, or with almost no detail — and drop the "colourful"
    warning entirely.
    """
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Phone photos are softer and colourful by nature; use lenient thresholds.
    blur_thresh = 8.0 if phone_mode else BLUR_VAR_THRESHOLD
    contrast_thresh = 8 if phone_mode else 15

    issues, warnings = [], []

    # 1) Resolution — a hard requirement.
    if min(h, w) < MIN_SIDE:
        issues.append(
            f"Image is too small ({w}×{h}px). Please use an image at least "
            f"{MIN_SIDE}px on the shorter side."
        )

    # 2) Focus / blur (only reject when genuinely unusable).
    blur = _laplacian_variance(gray)
    if blur < blur_thresh:
        issues.append(
            "Image looks very out of focus. Retake the photo holding the camera "
            "steady, with the X-ray well lit and filling the frame."
        )

    # 3) Colourfulness — real CXRs are near-grayscale, so a very colourful image
    #    may be a regular photo. Skipped in phone mode (warm cast is expected).
    saturation = _mean_saturation(bgr)
    if not phone_mode and saturation > GRAY_SAT_THRESHOLD:
        warnings.append(
            "This image is quite colourful for a chest X-ray — double-check you "
            "uploaded a radiograph, not a regular photo."
        )

    # 4) Dynamic range — a nearly-flat image (all one shade) carries no signal.
    if gray.std() < contrast_thresh:
        issues.append(
            "Image has very little contrast/detail. Please upload a clearer "
            "chest X-ray."
        )

    # Rough composite score for display (not a medical metric).
    focus_score = min(blur / (BLUR_VAR_THRESHOLD * 4), 1.0)
    gray_score = max(0.0, 1.0 - saturation / GRAY_SAT_THRESHOLD)
    res_score = min(min(h, w) / (MIN_SIDE * 3), 1.0)
    score = round(0.5 * focus_score + 0.3 * gray_score + 0.2 * res_score, 2)

    return QualityResult(ok=len(issues) == 0, score=score,
                         issues=issues, warnings=warnings)
