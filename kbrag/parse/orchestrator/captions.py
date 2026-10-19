"""Подпись рисунка, принятая за колонтитул, — обратно в текст."""

from __future__ import annotations

import re
from typing import Any

from ..parsers.structure import FURNITURE, label_of

# Подпись рисунка или таблицы: «FIG. 4.», «Fig. 2», «Рис. 3», «TABLE IV», «Таблица 2».
_CAPTION = re.compile(
    r"^\s*(?:FIG\.?|Fig\.?|Figure|Рис\.?|Рисунок|TABLE|Table|Таблица)\s*[\dIVXLC]+\b"
)


def rescue_captions(doc: Any) -> int:
    """Подпись рисунка внизу страницы docling бывает принимает за колонтитул — тогда она
    пропадает из текста (статья FMM, 14.10.2026: «FIG. 4. Lists 1–4 for box b» под
    векторным рисунком). Такая подпись возвращается в текст — в конец своей страницы.
    Возвращает, сколько подписей вернули."""
    from docling_core.types.doc.labels import DocItemLabel

    rescued = 0
    for item in list(getattr(doc, "texts", []) or []):
        if label_of(item) not in FURNITURE or not item.prov or not _CAPTION.match(item.text or ""):
            continue
        page_no = item.prov[0].page_no
        on_page = [
            node
            for node, _ in doc.iterate_items()
            if getattr(node, "prov", None)
            and node.prov[0].page_no == page_no
            and label_of(node) not in FURNITURE
        ]
        if not on_page:
            continue
        doc.insert_text(
            sibling=on_page[-1],
            label=DocItemLabel.CAPTION,
            text=item.text,
            orig=item.orig,
            prov=item.prov[0],
            after=True,
        )
        rescued += 1
    return rescued
