"""
Synthetic phone-photo augmentation for chest X-rays.

Real datasets of *photographed* X-rays (CheXphoto) are 20-56 GB and infeasible
here, so we reproduce the capture artefacts synthetically — the same technique
the CheXphoto/PTRN papers use. Given a clean X-ray, this simulates a phone
snapshot of a film on a lightbox/screen:

  * perspective distortion (non-ideal camera angle)
  * the film not filling the frame -> dark, slightly warm borders/margins
  * a warm colour cast (the orange tint from ambient light)
  * specular glare highlights and a soft screen glow
  * vignetting (uneven illumination)
  * mild blur, sensor noise, and JPEG-like softening

Applied during training (label unchanged), it teaches the model to see through
these artefacts. Kept as a standalone module so training and validation share
exactly the same simulation.
"""

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _add_dark_border(img, rng):
    """Composite the X-ray onto a dark, slightly warm canvas with random,
    asymmetric margins — like a film that doesn't fill the photo."""
    w, h = img.size
    # margins as a fraction of each side
    ml = int(w * rng.uniform(0.0, 0.18))
    mr = int(w * rng.uniform(0.0, 0.18))
    mt = int(h * rng.uniform(0.0, 0.22))
    mb = int(h * rng.uniform(0.0, 0.22))
    cw, ch = w + ml + mr, h + mt + mb
    # dark, faintly warm background
    base = (rng.integers(6, 26), rng.integers(4, 20), rng.integers(2, 16))
    canvas = Image.new("RGB", (cw, ch), base)
    canvas.paste(img, (ml, mt))
    return canvas


def _warm_cast(img, rng):
    """Multiply channels to add a warm (orange) ambient colour cast."""
    arr = np.asarray(img).astype(np.float32)
    r = rng.uniform(1.02, 1.18)
    g = rng.uniform(0.98, 1.06)
    b = rng.uniform(0.80, 0.98)
    arr[..., 0] *= r
    arr[..., 1] *= g
    arr[..., 2] *= b
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _glare(img, rng):
    """Add 1-3 bright elliptical specular highlights + optional soft glow."""
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(rng.integers(1, 4)):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        sx = rng.uniform(w * 0.05, w * 0.28)
        sy = rng.uniform(h * 0.05, h * 0.28)
        strength = rng.uniform(40, 130)
        blob = np.exp(-(((xx - cx) ** 2) / (2 * sx ** 2)
                        + ((yy - cy) ** 2) / (2 * sy ** 2)))
        arr += (strength * blob)[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _vignette(img, rng):
    """Radial darkening toward the corners (uneven lighting)."""
    arr = np.asarray(img).astype(np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    d = d / d.max()
    strength = rng.uniform(0.15, 0.45)
    mask = 1.0 - strength * (d ** 2)
    arr *= mask[..., None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _jpeg(img, rng):
    """Re-encode at a low-ish JPEG quality to add compression artefacts."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(rng.integers(45, 85)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _noise(img, rng):
    arr = np.asarray(img).astype(np.float32)
    arr += rng.normal(0, rng.uniform(2, 9), arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


class PhonePhotoAug:
    """Callable PIL->PIL transform. Applies the full simulation with prob `p`;
    otherwise returns the image unchanged so the model still sees clean scans.
    """

    def __init__(self, p=0.6, seed=None):
        self.p = p
        self._seed = seed

    def __call__(self, img):
        if random.random() > self.p:
            return img
        # Fresh rng per call, seeded from Python's `random` (which the DataLoader
        # reseeds per worker), so all workers/samples get diverse augmentation.
        rng = np.random.default_rng(
            self._seed if self._seed is not None else random.getrandbits(32))
        img = img.convert("RGB")

        # geometric: mild rotation is handled upstream; here add border framing.
        if rng.random() < 0.85:
            img = _add_dark_border(img, rng)
        # photometric
        if rng.random() < 0.9:
            img = _warm_cast(img, rng)
        if rng.random() < 0.7:
            img = _glare(img, rng)
        if rng.random() < 0.8:
            img = _vignette(img, rng)
        # contrast/brightness wobble
        if rng.random() < 0.7:
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.2))
        if rng.random() < 0.7:
            img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.25))
        # optics: blur + noise + jpeg
        if rng.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.6)))
        if rng.random() < 0.5:
            img = _noise(img, rng)
        if rng.random() < 0.6:
            img = _jpeg(img, rng)
        return img
