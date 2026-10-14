"""Resized image cache for fast training.

Training crops are taken at 224 px, so decoding multi-megapixel originals on
every epoch wastes time. We resize each image once so its shorter side is
`short_side` px (default 288) and train from the copies. RandomResizedCrop
works on relative scale and the test transform resizes to 256 px first, so
the inputs the models see are the same up to interpolation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


def build(images: pd.DataFrame, cache_dir: str | Path, short_side: int = 288) -> pd.DataFrame:
    """Return a copy of `images` whose `path` points at resized copies."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    new_paths, made = [], 0
    for iid, src in zip(images["image_id"], images["path"]):
        dst = cache_dir / (iid.replace(":", "_") + ".jpg")
        if not dst.exists():
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                w, h = im.size
                s = min(w, h)
                if s > short_side:
                    f = short_side / s
                    im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.Resampling.BICUBIC)
                im.save(dst, quality=95)
            made += 1
        new_paths.append(str(dst))
    out = images.copy()
    out["path"] = new_paths
    print(f"Image cache: {made} new, {len(out) - made} reused -> {cache_dir}")
    return out
