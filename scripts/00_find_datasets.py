"""Locate the MSLD v2.0 and MSID folders under a Kaggle input directory.

Prints shell assignments, e.g.
    MSLD=/kaggle/input/mpox-skin-lesion-dataset-version-20-msld-v20
    MSID=/kaggle/input/monkeypoxskinimagedataset
"""

import argparse
import os
import shlex
import sys

KEYS = {
    "MSLD": ("msld", "mpox-skin-lesion"),
    "MSID": ("monkeypoxskinimage", "msid", "monkeypox-skin-image"),
}
OPTIONAL = {"FITZ": ("fitzpatrick17k",)}  # look-alike pool for the method study


def find(top: str, keywords, max_depth: int = 4):
    hits = []
    for root, dirs, _ in os.walk(top):
        depth = root[len(top):].count(os.sep)
        if depth >= max_depth:
            dirs[:] = []
        if root != top and any(k in os.path.basename(root).lower() for k in keywords):
            hits.append(root)
    return sorted(hits, key=lambda p: (p.count(os.sep), len(p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/kaggle/input")
    a = ap.parse_args()
    for var, kws in KEYS.items():
        hits = find(a.input, kws)
        if not hits:
            sys.exit(f"Could not find the {var} dataset under {a.input}. Add it as notebook input.")
        print(f"{var}={shlex.quote(hits[0])}")
    for var, kws in OPTIONAL.items():
        hits = find(a.input, kws)
        print(f"{var}={shlex.quote(hits[0]) if hits else ''}")


if __name__ == "__main__":
    main()
