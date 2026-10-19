"""Training methods compared in the paper.

The proposed method, TALE (Tone-Aware Look-alike Exposure), has three parts,
each tied to a failure found in the evaluation study:

* look-alike exposure - auxiliary non-mpox images of other papulovesicular
  skin diseases (``external.py``) added to the training set only, against the
  specificity collapse on unseen look-alike diseases;
* tone-balanced sampling - every (class, skin-tone band) group is drawn equally
  often, against the confound "dark skin = mpox";
* tone-shift augmentation - the healthy skin around a lesion is moved to a
  random ITA band while the lesion itself is left unchanged.

Baselines: plain fine-tuning (ERM), hue/saturation colour jitter, mixup and
Group DRO (Sagawa et al., ICLR 2020) over the same (class, tone) groups.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.color import lab2rgb, rgb2lab

from .ita import skin_mask

BAND_RANGES = {"dark": (-30.0, 10.0), "intermediate": (10.0, 41.0), "light": (41.0, 60.0)}
BAND_ORDER = ["dark", "intermediate", "light", "unknown"]

# name -> settings. ``extra`` = which auxiliary negatives to add (False, "lookalike"
# or "control"); ``sampler`` = "group" for tone-balanced sampling; ``aug`` in
# {"base", "hsv", "tone"}. "rand" is the control for "la": the same number of
# Fitzpatrick17k images, matched on Fitzpatrick type, from conditions that are
# NOT look-alikes (external.build_control_pool).
CONFIGS = {
    "erm":   {"aug": "base", "sampler": None, "loss": "ce", "mixup": 0.0, "extra": False},
    "hsv":   {"aug": "hsv", "sampler": None, "loss": "ce", "mixup": 0.0, "extra": False},
    "mixup": {"aug": "base", "sampler": None, "loss": "ce", "mixup": 0.2, "extra": False},
    "gdro":  {"aug": "base", "sampler": None, "loss": "gdro", "mixup": 0.0, "extra": False},
    "la":    {"aug": "base", "sampler": None, "loss": "ce", "mixup": 0.0, "extra": "lookalike"},
    "rand":  {"aug": "base", "sampler": None, "loss": "ce", "mixup": 0.0, "extra": "control"},
    "tone":  {"aug": "tone", "sampler": "group", "loss": "ce", "mixup": 0.0, "extra": False},
    "tale":  {"aug": "tone", "sampler": "group", "loss": "ce", "mixup": 0.0, "extra": "lookalike"},
}


class ToneShift:
    """Move the healthy skin of an image to a random ITA band; keep the lesion.

    Healthy skin = YCrCb skin pixels that are not clearly redder than the image's
    typical skin (smoothed a* above median + 2 MAD, capped so at least the
    reddest 10% count as lesion). Only L* of healthy skin changes, through a
    blurred mask so there is no hard edge. Applied with probability ``p``.
    """

    def __init__(self, p: float = 0.5, red_drop: float = 0.30, blur: float = 3.0,
                 max_delta: float = 35.0):
        self.p, self.red_drop, self.blur, self.max_delta = p, red_drop, blur, max_delta
        self._pid, self.rng = None, None

    def __call__(self, img: Image.Image) -> Image.Image:
        if self._pid != os.getpid():  # one generator per DataLoader worker, seeded by torch
            self._pid = os.getpid()
            self.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        if self.rng.random() >= self.p:
            return img
        out = shift_tone(img, self._target(), self.red_drop, self.blur, self.max_delta)
        return img if out is None else out

    def _target(self) -> float:
        lo, hi = BAND_RANGES[self.rng.choice(list(BAND_RANGES))]
        return float(self.rng.uniform(lo, hi))


def shift_tone(img: Image.Image, target_ita: float, red_drop: float = 0.30, blur: float = 3.0,
               max_delta: float = 35.0, min_pixels: int = 300, return_info: bool = False):
    """Return a copy of ``img`` whose healthy skin has ITA ~ ``target_ita``, or None.

    With ``return_info`` a (image or None, info) pair is returned; ``info`` holds
    the skin and healthy-skin masks and the applied shift, for checking the
    augmentation (scripts/12_tone_shift_check.py). The image is the same either way.
    """
    info = {"reason": None}
    out = _shift_tone(img, target_ita, red_drop, blur, max_delta, min_pixels, info)
    return (out, info) if return_info else out


def _shift_tone(img, target_ita, red_drop, blur, max_delta, min_pixels, info):
    rgb = np.asarray(img.convert("RGB"))
    skin = skin_mask(rgb)
    info["skin"] = skin
    if skin.sum() < min_pixels:
        info["reason"] = "too few skin pixels"
        return None
    lab = rgb2lab(rgb)
    # Lesion / inflamed skin = skin pixels clearly redder than the image's typical
    # skin (a* smoothed first, so the mask is a few coherent blobs, not noise).
    a = gaussian_filter(lab[..., 1], sigma=2.0)
    med = np.median(a[skin])
    mad = 1.4826 * np.median(np.abs(a[skin] - med))
    thr = min(med + 2.0 * max(mad, 1.0), np.quantile(a[skin], 1 - red_drop / 3))
    healthy = skin & (a <= thr)
    info["healthy"] = healthy
    if healthy.sum() < min_pixels:
        info["reason"] = "too few healthy-skin pixels"
        return None
    L_med = float(np.median(lab[..., 0][healthy]))
    b_med = float(np.median(lab[..., 2][healthy]))
    if b_med < 3.0:  # ITA is unstable for near-neutral colour
        info["reason"] = "near-neutral colour"
        return None
    L_target = 50.0 + b_med * np.tan(np.radians(target_ita))
    delta = float(np.clip(np.clip(L_target, 20.0, 90.0) - L_med, -max_delta, max_delta))
    info.update(delta=delta, wanted=float(L_target - L_med), L_med=L_med, b_med=b_med)
    w = gaussian_filter(healthy.astype(np.float32), sigma=blur)
    lab[..., 0] = np.clip(lab[..., 0] + w * delta, 0.0, 100.0)
    out = np.clip(lab2rgb(lab) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def group_ids(targets, bands) -> np.ndarray:
    """(class, tone band) group index for each sample."""
    b = np.array([BAND_ORDER.index(x) if x in BAND_ORDER else 3 for x in bands])
    return np.asarray(targets) * len(BAND_ORDER) + b


def balanced_weights(groups: np.ndarray) -> np.ndarray:
    """Sampling weight per sample so that every group is drawn equally often."""
    uniq, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


class GroupDRO:
    """Online Group DRO (Sagawa et al., 2020): minimise the worst group's loss."""

    def __init__(self, n_groups: int, step: float = 0.01, device="cpu"):
        self.q = torch.ones(n_groups, device=device) / n_groups
        self.step = step

    def __call__(self, logits, y, g):
        losses = F.cross_entropy(logits, y, reduction="none")
        n = self.q.numel()
        onehot = F.one_hot(g, n).float()
        counts = onehot.sum(0)
        group_loss = (onehot * losses[:, None]).sum(0) / counts.clamp(min=1)
        present = counts > 0
        with torch.no_grad():
            self.q[present] *= torch.exp(self.step * group_loss[present].detach())
            self.q /= self.q.sum()
        return (self.q * group_loss).sum()


def mixup(x, y, alpha: float, gen: torch.Generator | None = None):
    """Return mixed inputs, the two target vectors and the mixing weight."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam
