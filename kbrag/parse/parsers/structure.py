"""Разметка docling: разделы, текст по страницам, колонтитулы.

Отдельно от docling.py: здесь нет самого docling с его моделями — функции нужны и в
основном процессе (оркестратор), куда тяжёлый docling не тянем.
"""

from __future__ import annotations

from typing import Any

FURNITURE = {"page_header", "page_footer"}  # колонтитулы: в Markdown и фрагменты не идут


def label_of(item: Any) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label)).lower()


def extract_structure(doc: Any) -> tuple[list[str], dict[int, str], int]:
    """Заголовки разделов, текст по страницам (без колонтитулов — как в Markdown) и
    число страниц."""
    sections: list[str] = []
    pages_text: dict[int, str] = {}

    for item in getattr(doc, "texts", []) or []:
        text = (getattr(item, "text", "") or "").strip()
        label = label_of(item)
        if not text or label in FURNITURE:
            continue
        if label in {"section_header", "title"} and len(text) < 300:
            sections.append(text)
        for prov in getattr(item, "prov", None) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is None:
                continue
            pages_text[page_no] = (pages_text.get(page_no, "") + "\n" + text).strip()

    pages = getattr(doc, "pages", None) or {}
    page_count = len(pages) if pages else (max(pages_text) if pages_text else 0)
    return sections, pages_text, page_count
