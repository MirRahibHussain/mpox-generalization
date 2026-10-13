"""Contact sheets for manual review of duplicate candidates."""

from __future__ import annotations

import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def _thumb(path: str, side: int) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return ImageOps.pad(im, (side, side), color=(255, 255, 255))


def pair_sheet(pairs: pd.DataFrame, images: pd.DataFrame, out_png: str,
               max_pairs: int = 60, side: int = 160, cols: int = 3) -> int:
    """Save a grid of image pairs (a | b) with captions. Returns pairs drawn."""
    if pairs.empty:
        return 0
    path = images.set_index("image_id")["path"]
    lab = images.set_index("image_id")["label"]
    pairs = pairs.head(max_pairs)
    cap = 28
    cell_w, cell_h = side * 2 + 10, side + cap
    rows = (len(pairs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * 10, rows * cell_h + (rows + 1) * 10), "white")
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(pairs.itertuples(index=False)):
        x0 = 10 + (k % cols) * (cell_w + 10)
        y0 = 10 + (k // cols) * (cell_h + 10)
        sheet.paste(_thumb(path[r.a], side), (x0, y0))
        sheet.paste(_thumb(path[r.b], side), (x0 + side + 10, y0))
        txt = f"#{k} {r.method}={r.score:.2f}  {lab[r.a]} | {lab[r.b]}"
        draw.text((x0, y0 + side + 6), txt, fill=(0, 0, 0))
    sheet.save(out_png)
    return len(pairs)
