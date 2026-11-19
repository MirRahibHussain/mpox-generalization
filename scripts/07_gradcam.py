"""Step 7 (GPU optional) - Grad-CAM sheets for the explainability check.

Needs runs made with --save-ckpt. For each chosen run, draws Grad-CAM on up to
--n test images and writes gradcam_review.csv with an empty column
`attention` for you to fill by eye: lesion / background / mixed.

    pip install grad-cam
    python scripts/07_gradcam.py --run results/binary_P3_msld_v2_to_msid_rep0/resnet50
"""

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from mpoxgen.labels import TASKS, positive_index
from mpoxgen.train import IMAGENET_MEAN, IMAGENET_STD


def target_layer(model, name):
    if name.startswith("resnet"):
        return model.layer4[-1]
    if name.startswith("densenet"):
        return model.features.denseblock4
    if name.startswith(("efficientnet", "mobilenetv3")):
        return model.blocks[-1]
    if name.startswith("convnext"):
        return model.stages[-1]
    raise SystemExit(f"Grad-CAM layer not defined for {name} (CNNs only).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/<split>/<model> with model.pt")
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    run = Path(a.run)
    rec = json.loads((run / "metrics.json").read_text())
    name, task = rec["model"], rec["task"]
    images = pd.read_csv(Path(a.index) / "images.csv").set_index("image_id")
    pred = pd.read_csv(run / "predictions.csv")
    pos = positive_index(task)

    # Balanced sample: true positives, false negatives, false positives, true negatives.
    pred["kind"] = np.select(
        [(pred.y_true == pos) & (pred.y_pred == pos), (pred.y_true == pos) & (pred.y_pred != pos),
         (pred.y_true != pos) & (pred.y_pred == pos)], ["TP", "FN", "FP"], "TN")
    pick = pred.groupby("kind", group_keys=False).apply(lambda g: g.sample(min(len(g), a.n // 4), random_state=0))

    model = timm.create_model(name, pretrained=False, num_classes=len(TASKS[task]))
    model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
    model.eval()
    cam = GradCAM(model=model, target_layers=[target_layer(model, name)])
    out = run / "gradcam"
    out.mkdir(exist_ok=True)
    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)
    rows = []
    for r in pick.itertuples():
        with Image.open(images.loc[r.image_id, "path"]) as im:
            im = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (224, 224))
        rgb = np.asarray(im, dtype=np.float32) / 255.0
        x = torch.tensor(((rgb - mean) / std).transpose(2, 0, 1), dtype=torch.float32)[None]
        heat = cam(input_tensor=x, targets=None)[0]  # explains the predicted class
        over = show_cam_on_image(rgb, heat, use_rgb=True)
        fn = f"{r.kind}_{r.image_id.replace(':', '_')}.png"
        Image.fromarray(np.hstack([(rgb * 255).astype(np.uint8), over])).save(out / fn)
        rows.append({"image_id": r.image_id, "kind": r.kind, "file": fn, "attention": ""})
    pd.DataFrame(rows).to_csv(out / "gradcam_review.csv", index=False)
    print(f"Wrote {len(rows)} Grad-CAM images to {out}/ - fill the 'attention' column by eye.")


if __name__ == "__main__":
    main()
