"""Class-name harmonization across public mpox datasets."""

from __future__ import annotations

import re

# Canonical class -> folder-name aliases seen in public datasets.
ALIASES: dict[str, list[str]] = {
    "mpox": ["mpox", "monkeypox", "monkey pox", "mkp", "monkeypox virus"],
    "chickenpox": ["chickenpox", "chicken pox", "varicella"],
    "measles": ["measles"],
    "cowpox": ["cowpox", "cow pox"],
    "hfmd": ["hfmd", "hand foot mouth", "hand foot and mouth", "hand foot mouth disease",
             "hand foot and mouth disease"],
    "healthy": ["healthy", "normal"],
}

# Classes shared by MSLD v2.0 and MSID (used by the 4-class task).
SHARED_4CLASS = ["mpox", "chickenpox", "measles", "healthy"]

TASKS = {
    "binary": ["non_mpox", "mpox"],
    "4class": SHARED_4CLASS,
}


def _norm(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[_\-,]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


_LOOKUP = {_norm(a): canon for canon, al in ALIASES.items() for a in al}


def canonical_class(folder_name: str) -> str | None:
    """Map a folder name to a canonical class, or None if it is not a class folder."""
    n = _norm(folder_name)
    if n in _LOOKUP:
        return _LOOKUP[n]
    # Allow names like "Monkeypox (279)" or "mpox images".
    for alias, canon in sorted(_LOOKUP.items(), key=lambda kv: -len(kv[0])):
        if n.startswith(alias + " ") or n.startswith(alias + "("):
            return canon
    return None


def task_target(label: str, task: str) -> int | None:
    """Integer target for a canonical label under a task, or None if excluded."""
    if task == "binary":
        return 1 if label == "mpox" else 0
    if task == "4class":
        return SHARED_4CLASS.index(label) if label in SHARED_4CLASS else None
    raise ValueError(f"unknown task: {task}")


def positive_index(task: str) -> int:
    """Index of the mpox class for sensitivity/specificity."""
    return 1 if task == "binary" else SHARED_4CLASS.index("mpox")
