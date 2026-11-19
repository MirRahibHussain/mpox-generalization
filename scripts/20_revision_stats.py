"""Recompute the statistics added in the TUJNS revision from the released files (CPU, no images).

Usage: python scripts/20_revision_stats.py [repo_root]   
       python scripts/20_revision_stats.py > paper_results/revision_stats.txt

Covers Sections 7.5 (chi-square tests, 36-run Wilcoxon, bootstrap CI), Table 9 (Wilson and Newcombe
intervals), Table 11 (closed-set balanced accuracy and unseen-class specificity, TALE study), the
evaluation-study Tables 6-8, and a robustness check of the main paired comparisons (hierarchical
bootstrap over backbones and seeds, Holm adjustment across all sessions).
Bootstrap intervals can differ in the third decimal with the random seed.
"""
import sys, glob, os, math
import numpy as np
import pandas as pd
from scipy import stats

R = sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[1])
im = pd.read_csv(f"{R}/release/v2_evaluation/index/images.csv")
lab = dict(zip(im.image_id, im.label))
band = dict(zip(im.image_id, im.ita_band))

print("== 1. ITA bands: mpox vs non-mpox (Section 7.5)")
for ds in ["msid", "msld_v2"]:
    d = im[(im.dataset == ds) & (im.ita_band != "unknown")]
    d = d.assign(mp=(d.label == "mpox"))
    ct = pd.crosstab(d.mp, d.ita_band)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = math.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    a = im[im.dataset == ds]  # shares over all images, unknown band included (as in the paper)
    dark_m = (a[a.label == "mpox"].ita_band == "dark").mean(); dark_o = (a[a.label != "mpox"].ita_band == "dark").mean()
    print(f"{ds}: chi2({dof})={chi2:.1f} p={p:.2g} V={v:.2f}  dark: mpox {dark_m:.1%} other {dark_o:.1%}")

print("\n== 2. 36 ERM runs, sensitivity dark minus light, MSLD v2.0 -> MSID (Section 7.5)")
gaps, darks, lights = [], [], []
files = {"tale": "tale/tone_sensitivity_runs.csv", "control": "control/tone_sensitivity_runs.csv",
         "factor": "factor/tone_sensitivity_runs.csv", "external": "external/tone_sensitivity_runs.csv"}
for s, f in files.items():
    t = pd.read_csv(f"{R}/paper_results/{f}")
    mcol = "method" if "method" in t else "me"
    e = t[t[mcol] == "erm"]
    print(f"  {s}: {len(e)} ERM runs")
    gaps += list(e.dark - e.light); darks += list(e.dark); lights += list(e.light)
g = np.array(gaps)
rng = np.random.default_rng(0)
boot = [rng.choice(g, len(g)).mean() for _ in range(10000)]
w = stats.wilcoxon(g)
print(f"  n={len(g)} positive={int((g>0).sum())} mean dark {np.mean(darks):.3f} light {np.mean(lights):.3f} "
      f"gap {g.mean():.3f} CI {np.percentile(boot,2.5):.3f} to {np.percentile(boot,97.5):.3f} Wilcoxon p={w.pvalue:.2g}")

print("\n== 3. Table 9 Wilson / Newcombe, per-run n (mpox: 100 dark, 74 light)")
def wilson(p, n, z=1.96):
    c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return c - h, c + h
t = pd.read_csv(f"{R}/paper_results/p3_skin_tone.csv")
for m in t.model.unique():
    a = t[(t.model == m) & (t.test_dataset == "msid")].set_index("band")
    pd_, pl = a.loc["dark", "sensitivity"], a.loc["light", "sensitivity"]
    nd, nl = int(a.loc["dark", "n_mpox"] // 3), int(a.loc["light", "n_mpox"] // 3)
    ld, ud = wilson(pd_, nd); ll, ul = wilson(pl, nl)
    diff = pd_ - pl
    lo = diff - math.sqrt((pd_-ld)**2 + (ul-pl)**2); hi = diff + math.sqrt((ud-pd_)**2 + (pl-ll)**2)
    print(f"  {m:24s} dark {pd_:.3f} ({ld:.3f}-{ud:.3f}) light {pl:.3f} ({ll:.3f}-{ul:.3f}) diff {diff:+.3f} ({lo:+.3f} to {hi:+.3f})")

print("\n== 4. Table 11 closed-set bal. acc. and unseen-class spec., TALE study, MSID -> MSLD v2.0")
shared_neg = {"chickenpox", "measles", "healthy"}; unseen = {"cowpox", "hfmd"}
rows = []
for f in glob.glob(f"{R}/release/v3_tale/results_methods/binary_P3_msid_to_msld_v2_rep*/*/*/predictions.csv"):
    method = f.split("/")[-2]
    p = pd.read_csv(f); p["label"] = p.image_id.map(lab)
    sens = (p[p.label == "mpox"].y_pred == 1).mean()
    spec_sh = (p[p.label.isin(shared_neg)].y_pred == 0).mean()
    # closed-set balanced accuracy: mean of mpox sensitivity and specificity on shared classes
    rows.append(dict(method=method, closed=(sens + spec_sh) / 2,
                     unseen=(p[p.label.isin(unseen)].y_pred == 0).mean(),
                     hfmd=(p[p.label == "hfmd"].y_pred == 1).mean()*100,
                     cowpox=(p[p.label == "cowpox"].y_pred == 1).mean()*100))
print(pd.DataFrame(rows).groupby("method").agg(["mean", "count"]).round(4).to_string())

print("\n== 5. Evaluation study (Kaggle Version 2): Tables 6-8 from per-image predictions")
from sklearn.metrics import balanced_accuracy_score
ev = []
for f in glob.glob(f"{R}/release/v2_evaluation/results/binary_*/*/predictions.csv"):
    split, model = f.split("/")[-3], f.split("/")[-2]
    p = pd.read_csv(f); p["label"] = p.image_id.map(lab)
    proto = split.split("_")[1]; tr = split.split("_")[2]
    ev.append(dict(split=split, model=model, protocol=proto, train=tr,
                   ba=balanced_accuracy_score(p.y_true, p.y_pred), p=p))
evd = pd.DataFrame([{k: v for k, v in e.items() if k != "p"} for e in ev])
t6 = evd.groupby(["model", "train", "protocol"]).ba.agg(["mean", "std", "count"]).round(3)
print("Table 6 (balanced accuracy, mean / sd / runs):"); print(t6.unstack(["train", "protocol"])["mean"].to_string())
m = evd.groupby(["train", "protocol"]).ba.mean()
for tr in ["msid", "msld"]:
    print(f"  trained on {tr}: P1 {m[(tr,'P1')]:.3f}  P2 {m[(tr,'P2')]:.3f}  P3 {m[(tr,'P3')]:.3f}  "
          f"inflation P1-P2 {100*(m[(tr,'P1')]-m[(tr,'P2')]):+.1f} points")
p3 = [e for e in ev if e["protocol"] == "P3" and e["train"] == "msid"]
allp = pd.concat([e["p"] for e in p3])
print("Table 7, trained on MSID, % called mpox:", (allp.groupby("label").y_pred.mean() * 100).round(1).to_dict())
p3b = [e for e in ev if e["protocol"] == "P3" and e["train"] == "msld"]
allb = pd.concat([e["p"] for e in p3b])
print("Table 7, trained on MSLD v2.0, % called mpox:", (allb.groupby("label").y_pred.mean() * 100).round(1).to_dict())
cs, uns, hs, ds = [], [], [], []
for e in p3:
    p = e["p"]
    sens = (p[p.label == "mpox"].y_pred == 1).mean()
    cs.append((sens + (p[p.label.isin(shared_neg)].y_pred == 0).mean()) / 2)
    uns.append((p[p.label.isin(unseen)].y_pred == 0).mean())
    hs.append((p[p.label == "healthy"].y_pred == 0).mean())
    ds.append((p[p.label.isin({"chickenpox", "measles"})].y_pred == 0).mean())
print(f"Table 8, evaluation study (18 runs): closed-set bal. acc. {np.mean(cs):.3f}, all classes {np.mean([e['ba'] for e in p3]):.3f}, "
      f"shared-disease spec. {np.mean(ds):.2f}, healthy spec. {np.mean(hs):.2f}, unseen-class spec. {np.mean(uns):.2f}")

print("\n== 6. Robustness of the main paired comparisons, trained on MSID (cross-dataset)")
print("Pairs are (backbone, seed); seeds are nested in backbones, so the 9 pairs are not fully independent.")
print("Hierarchical bootstrap: resample the 3 backbones with replacement, then 3 seeds within each.")
print("Holm adjustment here is across all comparisons listed below (all sessions together).")
def load(session):
    if session in ("tale", "control"):
        t = pd.read_csv(f"{R}/paper_results/{session}/runs.csv")
        t = t[(t.p == "P3") & (t.tr == "m")]
        return t.rename(columns={"mo": "bb", "me": "method", "ba": "y"})[["bb", "rep", "method", "y"]]
    if session == "factor":
        t = pd.read_csv(f"{R}/paper_results/factor/all_runs.csv")
        t = t[(t.protocol == "P3") & (t.direction == "msid->msld_v2")]
        return t.rename(columns={"model": "bb", "balanced_accuracy": "y"})[["bb", "rep", "method", "y"]]
    t = pd.read_csv(f"{R}/paper_results/external/external_runs.csv")
    t = t[t.direction == "msid->mcsi"]
    return t.rename(columns={"model": "bb", "balanced_accuracy": "y"})[["bb", "rep", "method", "y"]]
comps = [("tale", "tale", "erm"), ("tale", "la", "erm"), ("control", "tale", "erm"), ("control", "rand", "erm"),
         ("control", "la", "rand"), ("factor", "rand", "erm"), ("factor", "atlas", "erm"), ("factor", "hod", "erm"),
         ("external", "rand", "erm"), ("external", "atlas", "erm"), ("external", "hod", "erm")]
rng = np.random.default_rng(0); out = []
for s, a, b in comps:
    t = load(s).pivot_table(index=["bb", "rep"], columns="method", values="y")
    d = (t[a] - t[b]).dropna()
    bbs = d.index.get_level_values(0).unique()
    groups = [d.xs(k).values for k in bbs]
    boot = []
    for _ in range(10000):
        pick = rng.integers(0, len(groups), len(groups))
        boot.append(np.mean([rng.choice(groups[i], len(groups[i])).mean() for i in pick]))
    per_bb = [g.mean() for g in groups]
    p = stats.wilcoxon(d.values, method="exact").pvalue
    out.append(dict(session=s, comparison=f"{a} - {b}", n=len(d), mean=100*d.mean(),
                    lo=100*np.percentile(boot, 2.5), hi=100*np.percentile(boot, 97.5),
                    backbones_positive=f"{sum(x > 0 for x in per_bb)}/{len(per_bb)}", p=p))
o = pd.DataFrame(out)
order = np.argsort(o.p.values); adj = np.empty(len(o)); run = 0
for k, i in enumerate(order):
    run = max(run, min(1, (len(o) - k) * o.p.values[i])); adj[i] = run
o["p_holm_all"] = adj
print(o.to_string(index=False, float_format=lambda v: f"{v:.3f}" if abs(v) < 1 else f"{v:.1f}"))
