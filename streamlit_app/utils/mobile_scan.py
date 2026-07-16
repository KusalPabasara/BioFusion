"""
Mobile X-ray scan rectification.

Turns a smartphone *photo of a chest X-ray film* into a clean, front-facing
input suitable for the model, using only OpenCV (no extra ML dependency and no
external model weights). The pipeline:

  1. Detect the X-ray film / lightbox region as the largest 4-corner contour.
  2. Perspective-warp that quad to a straight rectangle (removes camera-angle
     distortion, analogous in spirit to PTRN but far lighter-weight).
  3. Convert to grayscale and apply CLAHE (Contrast Limited Adaptive Histogram
     Equalisation) to normalise brightness/contrast across phone cameras.

If a confident film boundary cannot be found, we fall back to a centre-based
contrast-normalised crop so the app still produces a usable image rather than
failing. Every stage is reported so the UI can tell the user what happened.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class RectifyResult:
    image: Image.Image        # rectified RGB image ready for preprocess_image()
    rectified: bool           # True if a 4-corner film boundary was found & warped
    detail: str               # human-readable explanation for the UI


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left has smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right has largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest y-x
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest y-x
    return rect


def _find_film_quad(gray: np.ndarray, frac_threshold: float = 0.25):
    """Return the ordered 4 corners of the largest plausible quad, or None.

    A candidate must be convex, roughly quadrilateral, and cover a meaningful
    fraction of the frame (so we don't lock onto a small bright artifact).
    """
    img_area = gray.shape[0] * gray.shape[1]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < frac_threshold * img_area:
            break  # remaining contours are only smaller
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_corners(approx.reshape(4, 2).astype("float32"))
    return None


def _warp(image_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-warp the quad to an axis-aligned rectangle."""
    (tl, tr, br, bl) = quad
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    width, height = max(width, 1), max(height, 1)
    dst = np.array([[0, 0], [width - 1, 0],
                    [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image_bgr, M, (width, height))


def _clahe_gray(bgr: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE contrast normalisation, returned as 3-channel RGB."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)


def rectify_mobile_xray(pil_image: Image.Image) -> RectifyResult:
    """Rectify a phone photo of a chest X-ray into a clean model input.

    Args:
        pil_image: the uploaded/captured PIL image (any mode).

    Returns:
        RectifyResult with the processed RGB image and a status detail.
    """
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # NOTE: the model is now trained on phone-photo-simulated images and reads
    # raw phone captures best, so we no longer apply CLAHE contrast enhancement
    # (it was pushing borderline-normal images into "uncertain"). We only apply
    # the genuinely useful perspective correction when a film boundary is found.
    quad = _find_film_quad(gray)
    if quad is not None:
        warped = _warp(bgr, quad)
        out = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        return RectifyResult(
            image=Image.fromarray(out),
            rectified=True,
            detail="Detected the X-ray film and corrected its perspective.",
        )

    # No reliable film boundary — use the photo as-is (the model handles phone
    # captures directly). Guide the user to a better shot for next time.
    return RectifyResult(
        image=pil_image.convert("RGB"),
        rectified=False,
        detail="Using your photo as captured. For best results, photograph the "
               "X-ray straight-on, filling the frame, against a plain background.",
    )
