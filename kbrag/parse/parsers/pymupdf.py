"""Запасной парсер — PyMuPDF4LLM.

Docling — основной путь. PyMuPDF4LLM включается, только когда основной упал, завис
или не прошёл проверку: он заметно быстрее и проще, но хуже держит сложную вёрстку и
таблицы, а разметки для нарезки по структуре не даёт (фрагменты режутся по абзацам).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from ..diag import describe, full_traceback
from ..models import FAILED, OK, PYMUPDF, ParsedDoc, ParseInput

_HEADING_RE = re.compile(r"^#{1,4}\s+(.+?)\s*$", re.MULTILINE)


def _sections(markdown: str) -> list[str]:
    """Заголовки разделов из Markdown («# …» — «#### …»)."""
    return [m.strip() for m in _HEADING_RE.findall(markdown) if len(m) < 300]


def parse_with_pymupdf(source: Path | str | ParseInput) -> ParsedDoc:
    inp = ParseInput.of(source)
    started = time.monotonic()
    result = ParsedDoc.failed(inp, PYMUPDF)

    try:
        import pymupdf
        import pymupdf4llm
    except ImportError as exc:
        result.errors.append(f"pymupdf4llm не установлен: {exc}")
        return result

    doc = None
    try:
        if inp.data is not None:
            doc = pymupdf.open(stream=inp.data, filetype="pdf")
        else:
            doc = pymupdf.open(inp.path)
        kwargs: dict[str, Any] = {"page_chunks": True, "show_progress": False}
        # Сам по себе pymupdf4llm распознаёт страницы с картинками (OCR) — выключаем,
        # как и в docling. В старых версиях своего OCR нет — нет и параметра.
        try:
            from pymupdf4llm.ocr import OCRMode

            kwargs["use_ocr"] = OCRMode.NEVER
        except ImportError:
            pass
        try:
            pages = pymupdf4llm.to_markdown(doc, **kwargs)
        except TypeError as exc:
            # Путь без разметки страницы параметра use_ocr не принимает (там и OCR нет);
            # любая другая TypeError — настоящая ошибка, её не прячем.
            if "use_ocr" not in str(exc):
                raise
            kwargs.pop("use_ocr")
            pages = pymupdf4llm.to_markdown(doc, **kwargs)

        pages_text: dict[int, str] = {}
        tables = 0
        for i, page in enumerate(pages, start=1):
            meta = page.get("metadata") or {}
            page_no = int(meta.get("page_number") or i)
            pages_text[page_no] = (page.get("text") or "").strip()  # чистка — у оркестратора
            tables += len(page.get("tables") or [])

        markdown = "\n\n".join(pages_text[p] for p in sorted(pages_text))
        result.status = OK
        result.markdown = markdown
        result.pages_text = pages_text
        result.page_count = len(pages_text) or int(doc.page_count)
        result.table_count = tables
        result.sections = _sections(markdown)
    except Exception as exc:  # noqa: BLE001 — сбой разбора — статус FAILED с причиной
        result.status = FAILED
        result.errors += [describe(exc), full_traceback(exc)]
    finally:
        result.duration_sec = round(time.monotonic() - started, 2)
        if doc is not None:
            try:
                doc.close()
            except Exception:  # noqa: BLE001, S110
                pass

    return result
