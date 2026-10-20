"""FACTOR: Factorized, Atlas-Conditioned Training for Open-set Robustness.

The control study showed that adding images of *any* other skin disease as
negatives helps as much as adding curated look-alikes: what a model trained on
mostly healthy negatives lacks is the knowledge that "visible skin disease" is
not the same as "mpox". FACTOR builds this into the model:

* Factorized decision. One network, two logits. ``s_d`` answers "is any skin
  disease visible?" and is trained on every image (healthy study images are the
  negatives). ``s_m`` answers "is the disease mpox?" and is trained only on
  diseased images (mpox against every other disease), so lesion presence and
  photo style, which separate healthy stock photos from clinical photos, carry
  no information for it. A case is called mpox when the geometric mean of
  sigmoid(s_d) and sigmoid(s_m) is at least 0.5, i.e. when both class-balanced
  factors agree on average.
* Condition-diverse atlas exposure. Auxiliary diseased negatives come from the
  whole Fitzpatrick17k atlas. Every epoch a new subset of ``k`` images is drawn
  with weights uniform over conditions, so over training the model sees the
  full range of conditions at the cost of a small fixed pool.
* Tone matching. Draw weights are re-scaled so that the ITA-band mix of the
  atlas negatives equals that of the mpox training images, which makes skin
  tone uninformative for ``s_m``.

HOD baseline (Roy et al., Med. Image Anal. 2022, "Does your dermatology
classifier know what it doesn't know?"): one fine-grained class per inlier
class and per training outlier condition, trained with a fine-grained
cross-entropy plus a coarse (mpox vs rest) loss on the summed probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Sampler

BANDS = ["dark", "intermediate", "light", "unknown"]


def atlas_weights(conditions, bands, mode: str = "cond", mpox_bands=None,
                  n_conditions: int | None = None, seed: int = 0) -> np.ndarray:
    """Draw probability for every atlas image.

    mode "cond": uniform over conditions (each condition gets the same total
    weight). "cond_tone": as "cond", then re-weighted per ITA band so that the
    expected band mix equals the band mix of ``mpox_bands``. With
    ``n_conditions``, only that many randomly chosen conditions get weight
    (used to test how much condition diversity matters).
    """
    cond = np.asarray(conditions).astype(str)
    keep = np.ones(len(cond), bool)
    if n_conditions:
        uniq = np.array(sorted(set(cond)))
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(uniq, size=min(int(n_conditions), len(uniq)), replace=False))
        keep = np.array([c in chosen for c in cond])
    w = np.zeros(len(cond))
    counts = pd.Series(cond[keep]).value_counts()
    w[keep] = 1.0 / counts.reindex(cond[keep]).to_numpy(dtype=float)
    if mode == "cond_tone":
        if mpox_bands is None or len(mpox_bands) == 0:
            raise ValueError("cond_tone needs the ITA bands of the mpox training images")
        b = np.array([x if x in BANDS else "unknown" for x in np.asarray(bands).astype(str)])
        target = pd.Series([x if x in BANDS else "unknown" for x in mpox_bands]).value_counts(normalize=True)
        tot = w.sum()
        for band in BANDS:
            have = w[b == band].sum() / tot
            if have > 0:
                w[b == band] *= target.get(band, 0.0) / have
    if w.sum() <= 0:
        raise ValueError("no atlas image has positive weight")
    return w / w.sum()


class EpochSampler(Sampler):
    """Every study index once per epoch plus ``k`` atlas indices drawn anew each epoch.

    Study images occupy dataset indices [0, n_study); atlas images start at
    ``atlas_offset``. Atlas indices are drawn without replacement when enough
    images have positive weight.
    """

    def __init__(self, n_study: int, atlas_offset: int, atlas_w, k: int, seed: int = 0):
        self.n_study, self.off, self.k = int(n_study), int(atlas_offset), int(k)
        self.w = np.asarray(atlas_w, dtype=float)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_study + self.k

    def __iter__(self):
        nz = int((self.w > 0).sum())
        a = self.rng.choice(len(self.w), size=self.k, replace=self.k > nz, p=self.w)
        idx = np.concatenate([np.arange(self.n_study), self.off + a])
        self.rng.shuffle(idx)
        return iter(idx.tolist())


def factor_loss(logits, y, d, pw_d, pw_m):
    """BCE on the disease factor (all images) + BCE on the mpox factor (diseased images)."""
    s_d, s_m = logits[:, 0].float(), logits[:, 1].float()
    loss = F.binary_cross_entropy_with_logits(s_d, d.float(), pos_weight=pw_d)
    m = d > 0
    if bool(m.any()):
        loss = loss + F.binary_cross_entropy_with_logits(s_m[m], y[m].float(), pos_weight=pw_m)
    return loss


def factor_scores(logits) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(p_mpox, p_disease, p_mpox_given_disease); p_mpox is the geometric mean."""
    lg = logits.float()
    pd_, pm = torch.sigmoid(lg[:, 0]), torch.sigmoid(lg[:, 1])
    return (torch.sqrt(pd_ * pm).cpu().numpy(), pd_.cpu().numpy(), pm.cpu().numpy())


def hod_loss(logits, fine, y, w_coarse):
    """Fine-grained CE over all classes + class-weighted coarse loss (class 0 = mpox vs rest)."""
    lg = logits.float()
    loss_f = F.cross_entropy(lg, fine)
    logp = F.log_softmax(lg, 1)
    lp_m = logp[:, 0]
    lp_r = torch.logsumexp(logp[:, 1:], 1)
    yf = y.float()
    wt = w_coarse[1] * yf + w_coarse[0] * (1 - yf)
    loss_c = -(w_coarse[1] * yf * lp_m + w_coarse[0] * (1 - yf) * lp_r).sum() / wt.sum()
    return loss_f + loss_c


def hod_scores(logits) -> np.ndarray:
    return torch.softmax(logits.float(), 1)[:, 0].cpu().numpy()
