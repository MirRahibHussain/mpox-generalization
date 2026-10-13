"""Approximate skin tone via the Individual Typology Angle (ITA).

ITA = arctan((L* - 50) / b*) * 180 / pi, computed on skin pixels in CIELAB
(Kinyanjui et al., MICCAI 2020, apply ITA to dermatology images).

The public mpox datasets have no lesion masks, so we approximate "healthy
skin" as: pixels inside a YCrCb skin-colour range, minus the reddest 30%
(by a*), which are most likely lesion or inflamed skin. This is a proxy and
is reported as such in the paper.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab

# Coarse bands (degrees). Finer categories exist (very light > 55, light 41-55,
# intermediate 28-41, tan 10-28, brown -30-10, dark < -30); with a few
# hundred images we merge them into three bands to keep groups usable.
BANDS = [("light", 41.0, np.inf), ("intermediate", 10.0, 41.0), ("dark", -np.inf, 10.0)]


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    img = Image.fromarray(rgb).convert("YCbCr")
    ycc = np.asarray(img, dtype=np.int16)
    cb, cr = ycc[..., 1], ycc[..., 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)


def estimate(path: str, max_side: int = 256, red_drop: float = 0.30, min_pixels: int = 500) -> dict:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side))
        rgb = np.asarray(im)
    mask = skin_mask(rgb)
    lab = rgb2lab(rgb)
    L, a, b = lab[..., 0][mask], lab[..., 1][mask], lab[..., 2][mask]
    if len(L) < min_pixels:
        return {"ita": np.nan, "skin_pixels": int(len(L)), "ita_band": "unknown"}
    keep = a <= np.quantile(a, 1 - red_drop)
    L, b = L[keep], b[keep]
    ita = float(np.degrees(np.arctan2(np.median(L) - 50.0, np.median(b))))
    return {"ita": ita, "skin_pixels": int(mask.sum()), "ita_band": band(ita)}


def band(ita: float) -> str:
    if np.isnan(ita):
        return "unknown"
    for name, lo, hi in BANDS:
        if lo <= ita < hi:
            return name
    return "unknown"
