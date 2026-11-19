"""Perceptual hashes (pHash, dHash) implemented with NumPy/SciPy only.

Each hash is a 64-bit boolean vector. We also hash the horizontally flipped
image, so mirrored copies are caught.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dct


def _gray(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    g = ImageOps.exif_transpose(img).convert("L").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(g, dtype=np.float64)


def dhash(img: Image.Image, hash_size: int = 8) -> np.ndarray:
    px = _gray(img, (hash_size + 1, hash_size))
    return (px[:, 1:] > px[:, :-1]).flatten()


def phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    n = hash_size * highfreq_factor
    px = _gray(img, (n, n))
    coeffs = dct(dct(px, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:hash_size, :hash_size]
    return (low > np.median(low)).flatten()


def hash_image(path: str) -> dict[str, np.ndarray]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        flip = ImageOps.mirror(im)
        return {
            "phash": phash(im), "dhash": dhash(im),
            "phash_flip": phash(flip), "dhash_flip": dhash(flip),
        }


def pack(bits: np.ndarray) -> np.ndarray:
    """(N, 64) bool -> (N, 8) uint8."""
    return np.packbits(bits.astype(np.uint8), axis=1)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_matrix(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Pairwise Hamming distances between packed hash sets a (N,8) and b (M,8)."""
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.uint8)
    for s in range(0, a.shape[0], chunk):
        x = a[s:s + chunk, None, :] ^ b[None, :, :]
        out[s:s + chunk] = _POP[x].sum(axis=2)
    return out
