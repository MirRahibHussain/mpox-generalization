"""Step 13 (CPU) - paired comparisons of training methods, per direction, with Holm's correction.

For every requested pair of methods and every cross-dataset direction, the
runs are matched on (backbone, seed). For each pair we report the mean
difference, a 95% percentile-bootstrap interval of that mean (resampling the
matched pairs), the number of wins, the exact two-sided Wilcoxon signed-rank
p-value and the Holm-adjusted p-value. Holm's correction is applied within
each (direction, metric) family.

    python scripts/13_paired_stats.py --runs results_methods/all_runs.csv \
        --pairs tale:erm tale:hsv tale:mixup tale:gdro tale:la tale:tone la:erm tone:erm
"""

import argparse
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd

DIRS = {"msid->msld_v2": "MSID → MSLD v2.0", "msld_v2->msid": "MSLD v2.0 → MSID"}
DEFAULT_PAIRS = ["tale:erm", "tale:hsv", "tale:mixup", "tale:gdro", "tale:la", "tale:tone", "la:erm", "tone:erm"]


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def wilcoxon(d):
    from scipy.stats import wilcoxon as w
    d = np.asarray(d, float)
    if np.allclose(d, 0):
        return 1.0
    return float(w(d, zero_method="wilcox", method="exact" if len(d) <= 25 else "auto").pvalue)


def boot_ci(d, n_boot=10000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results_methods/all_runs.csv")
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS)
    ap.add_argument("--metrics", nargs="*", default=["balanced_accuracy"])
    ap.add_argument("--out", default=None, help="output prefix (default: next to --runs)")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    runs = pd.read_csv(a.runs)
    p3 = runs[runs["protocol"] == "P3"]
    rows = []
    for d in [x for x in DIRS if x in set(p3["direction"])] + sorted(set(p3["direction"]) - set(DIRS)):
        sub = p3[p3["direction"] == d]
        for metric in a.metrics:
            piv = sub.pivot_table(index=["model", "rep"], columns="method", values=metric)
            fam = []
            for pr in a.pairs:
                x, y = pr.split(":")
                if x not in piv or y not in piv:
                    continue
                pp = piv[[x, y]].dropna()
                diff = (pp[x] - pp[y]).to_numpy()
                lo, hi = boot_ci(diff, a.n_boot)
                fam.append({"direction": d, "metric": metric, "comparison": f"{x} vs {y}", "pairs": len(diff),
                            "mean_diff": diff.mean(), "ci_low": lo, "ci_high": hi,
                            "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "p": wilcoxon(diff)})
            for f, q in zip(fam, holm([f["p"] for f in fam])):
                f["p_holm"] = float(q)
            rows += fam
    tt = pd.DataFrame(rows)
    out = Path(a.out) if a.out else Path(a.runs).with_name("paired_stats")
    tt.to_csv(f"{out}.csv", index=False)
    lines = ["## Paired comparisons per direction (95% bootstrap CI of the mean difference; "
             "Holm-adjusted within direction and metric)", "",
             "| direction | metric | comparison | pairs | mean diff [95% CI] | wins/losses | p | p (Holm) |",
             "| --- " * 8 + "|"]
    for r in tt.itertuples():
        lines.append(f"| {DIRS.get(r.direction, r.direction)} | {r.metric} | {r.comparison} | {r.pairs} | "
                     f"{r.mean_diff:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.wins}/{r.losses} | "
                     f"{r.p:.4f} | {r.p_holm:.4f} |")
    Path(f"{out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
