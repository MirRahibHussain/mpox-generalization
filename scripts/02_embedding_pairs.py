"""Step 2 (GPU recommended, optional) - find edited copies with CNN embeddings.

Pairs with cosine similarity >= --auto are accepted automatically (accept=1).
Pairs between --review and --auto get accept=0 and are drawn on a contact
sheet: open it, and set accept=1 in embedding_pairs.csv for true duplicates
before running step 3.
"""

import argparse
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset

from mpoxgen.dedup import pairs_frame
from mpoxgen.viz import pair_sheet


class Imgs(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(ImageOps.exif_transpose(im).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--auto", type=float, default=0.97)
    ap.add_argument("--review", type=float, default=0.92)
    a = ap.parse_args()
    idx = Path(a.index)
    images = pd.read_csv(idx / "images.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(a.model, pretrained=True, num_classes=0).to(device).eval()
    cfg = resolve_data_config({}, model=model)
    tf = create_transform(**cfg)

    feats = []
    with torch.no_grad():
        for x in DataLoader(Imgs(images["path"], tf), batch_size=64, num_workers=2):
            f = model(x.to(device)).float()
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy())
    feats = np.concatenate(feats)
    sim = feats @ feats.T
    iu = np.triu_indices(len(images), k=1)
    mask = sim[iu] >= a.review
    i, j = iu[0][mask], iu[1][mask]
    pairs = pairs_frame(images, i, j, "embed", sim[i, j])
    pairs["accept"] = (pairs["score"] >= a.auto).astype(int)
    pairs = pairs.sort_values("score", ascending=False)
    pairs.to_csv(idx / "embedding_pairs.csv", index=False)
    np.save(idx / "embeddings.npy", feats)

    review = pairs[pairs["accept"] == 0]
    n = pair_sheet(review, images, str(idx / "embedding_review_sheet.png"), max_pairs=90)
    print(f"{len(pairs)} candidate pairs; {int(pairs['accept'].sum())} auto-accepted "
          f"(cos >= {a.auto}); {len(review)} need review ({n} drawn on embedding_review_sheet.png)")


if __name__ == "__main__":
    main()
