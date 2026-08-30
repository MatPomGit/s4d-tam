from __future__ import annotations

import importlib.util
from pathlib import Path

from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph


MODULE_PATH = Path(__file__).with_name("update_paper_0830.py")
spec = importlib.util.spec_from_file_location("update_paper_0830", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load paper updater from {MODULE_PATH}")
updater = importlib.util.module_from_spec(spec)
spec.loader.exec_module(updater)


def _normalize(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").replace("\u2011", "-").split())


def robust_replace_contains(doc, needle: str, new: str) -> None:
    wanted = _normalize(needle)
    for element in doc.element.iter(qn("w:p")):
        paragraph = Paragraph(element, doc)
        if wanted in _normalize(paragraph.text):
            paragraph.text = new
            return
    print(f"Deferred XML-level replacement: {needle}")


updater.replace_contains = robust_replace_contains
updater.main()
