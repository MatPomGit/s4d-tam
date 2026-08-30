from __future__ import annotations

import tools.update_paper_0830 as updater


def _normalize(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").replace("\u2011", "-").split())


def robust_replace_contains(doc, needle: str, new: str) -> None:
    wanted = _normalize(needle)
    for paragraph in doc.paragraphs:
        if wanted in _normalize(paragraph.text):
            paragraph.text = new
            return
    raise ValueError(f"Text fragment not found after normalization: {needle}")


updater.replace_contains = robust_replace_contains
updater.main()
