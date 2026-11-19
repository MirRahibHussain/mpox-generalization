"""Metrics with bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score)


def _sens_spec(y, pred, pos):
    yp, pp = (y == pos), (pred == pos)
    tp = np.sum(yp & pp)
    fn = np.sum(yp & ~pp)
    tn = np.sum(~yp & ~pp)
    fp = np.sum(~yp & pp)
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return sens, spec


def compute(y: np.ndarray, prob: np.ndarray, pos: int) -> dict[str, float]:
    """y: (N,) int targets; prob: (N, C) class probabilities; pos: mpox index."""
    y = np.asarray(y)
    prob = np.asarray(prob)
    pred = prob.argmax(1)
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
    }
    out["mpox_sensitivity"], out["mpox_specificity"] = _sens_spec(y, pred, pos)
    try:
        if prob.shape[1] == 2:
            out["auroc"] = roc_auc_score(y, prob[:, 1])
        else:
            out["auroc"] = roc_auc_score(y, prob, multi_class="ovr", average="macro",
                                         labels=list(range(prob.shape[1])))
    except ValueError:  # a class missing from y
        out["auroc"] = np.nan
    return {k: float(v) for k, v in out.items()}


def bootstrap(y, prob, pos, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, tuple]:
    """Percentile bootstrap CI for every metric in ``compute``."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    prob = np.asarray(prob)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        stats.append(compute(y[idx], prob[idx], pos))
    out = {}
    for k in stats[0]:
        v = np.array([s[k] for s in stats], dtype=float)
        v = v[~np.isnan(v)]
        out[k] = (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))) if len(v) else (np.nan, np.nan)
    return out
