"""Валидация после парсинга.

Две группы проверок:

1. Эвристики, которые работают без эталона (есть ли вообще текст, не битая ли
   кодировка, нет ли патологических повторов). Они ловят самые частые провалы:
   парсер вытащил почти пустой текст или PDF с поломанными шрифтами отдал мусор
   вида (cid:34). Не прошли — документ разбирает запасной парсер.
2. Сверка с эталоном (golden standard) из <папка эталонов>/<имя>.json: разделы, ключевые
   сущности, число таблиц и страниц. Это и есть метрика «точность парсинга».

Сопоставление сущностей — с поправкой на русскую морфологию: эталонное
«Анабарский щит» находится и в «Анабарского щита».
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..text import normalize, phrase_pattern
from .config import Settings
from .models import ParsedDoc, ValidationReport

# Символы, которые мы считаем нормальными для научной статьи ru/en.
_ALLOWED_RE = re.compile(
    r"[0-9a-zA-Zа-яА-ЯёЁ\s\.,;:!\?\-–—‑'\"«»()\[\]{}/\\%°±×·′″†‡&@#\$€₽\+=<>\|~\^_\*"
    r"Ͱ-Ͽ"  # греческие буквы — формулы, минералогия
    r"‐-‧‰-⁞"  # типографика
    r"]"
)
_CID_RE = re.compile(r"\(cid:\d+\)")


def load_golden(doc: Path | str, settings: Settings) -> dict[str, Any] | None:
    """Эталон для документа: <Settings.golden_dir>/<doc_id>.json."""
    stem = doc.stem if isinstance(doc, Path) else str(doc)
    candidate = settings.golden_dir / f"{stem}.json"
    if not candidate.exists():
        return None
    try:
        data: dict[str, Any] | None = json.loads(candidate.read_text(encoding="utf-8"))
        return data
    except json.JSONDecodeError as exc:
        raise ValueError(f"битый эталон {candidate}: {exc}") from exc


def garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    cid_hits = len(_CID_RE.findall(text))
    cleaned = _CID_RE.sub("", text)
    allowed = len(_ALLOWED_RE.findall(cleaned))
    total = max(len(cleaned), 1)
    # каждый (cid:NN) считаем как один «мусорный» символ сверху
    return round(1 - allowed / total + cid_hits / total, 4)


def thin_pages(parsed: ParsedDoc, min_chars: int) -> list[int]:
    """Страницы, с которых почти ничего не извлеклось."""
    return [p for p, t in sorted(parsed.pages_text.items()) if len(t.strip()) < min_chars // 2]


def validate(
    parsed: ParsedDoc, settings: Settings, golden: dict[str, Any] | None = None
) -> ValidationReport:
    report = ValidationReport()
    _text_checks(report, parsed, settings)
    if golden:
        _golden_checks(report, parsed, golden, settings)
    else:
        report.add(
            "golden_standard", True, critical=False, detail="эталон не задан, сверка пропущена"
        )
    return report


def _text_checks(report: ValidationReport, parsed: ParsedDoc, settings: Settings) -> None:
    """Проверки без эталона: есть ли текст, не мусор ли, нет ли повторов и пустых страниц."""
    text = parsed.text
    pages = max(parsed.page_count, 1)

    # --- 1. Есть ли вообще текст --------------------------------------------
    chars_per_page = len(text.strip()) / pages
    has_text = chars_per_page >= settings.min_chars_per_page
    report.add(
        "has_text",
        has_text,
        critical=True,
        detail=f"{chars_per_page:.0f} символов на страницу (порог {settings.min_chars_per_page})",
    )

    # --- 2. Кодировка и мусор ------------------------------------------------
    ratio = garbage_ratio(text)
    report.add(
        "encoding_sane",
        ratio <= settings.max_garbage_ratio,
        critical=True,
        detail=f"доля мусорных символов {ratio:.1%} (порог {settings.max_garbage_ratio:.0%})",
    )

    # --- 3. Патологические повторы ------------------------------------------
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
    repeats = Counter(lines).most_common(1)
    worst = repeats[0][1] if repeats else 0
    report.add(
        "no_pathological_repeats",
        worst <= settings.max_line_repeats,
        critical=False,
        detail=f"самая частая строка встречается {worst} раз",
    )

    # --- 4. Пустые страницы (информационно) ----------------------------------
    thin = thin_pages(parsed, settings.min_chars_per_page)
    report.add(
        "pages_covered",
        len(thin) <= pages * 0.2,
        critical=False,
        detail=f"почти пустых страниц: {len(thin)} из {pages}"
        + (f" → {thin[:10]}" if thin else ""),
    )


def _golden_checks(
    report: ValidationReport, parsed: ParsedDoc, golden: dict[str, Any], settings: Settings
) -> None:
    """Сверка с эталоном: разделы, ключевые сущности, число таблиц и страниц."""
    norm = normalize(parsed.text)
    _found_share(report, "sections", golden.get("sections"), norm, settings.min_sections_ratio)
    _found_share(report, "entities", golden.get("entities"), norm, settings.min_entities_ratio)

    if golden.get("tables") is not None:
        expected_tables = int(golden["tables"])
        delta = abs(parsed.table_count - expected_tables)
        report.add(
            "tables",
            delta <= settings.table_tolerance,
            critical=False,
            detail=f"найдено {parsed.table_count}, ожидалось {expected_tables}",
        )

    if golden.get("pages") is not None:
        expected_pages = int(golden["pages"])
        report.add(
            "page_count",
            parsed.page_count == expected_pages,
            critical=False,
            detail=f"страниц {parsed.page_count}, ожидалось {expected_pages}",
        )


def _found_share(
    report: ValidationReport, name: str, expected: Any, norm: str, threshold: float
) -> None:
    """Доля эталонных фраз, что нашлись в тексте (с поправкой на падежи), от порога."""
    wanted = [x for x in (expected or []) if x.strip()]
    if not wanted:
        return
    found = [x for x in wanted if phrase_pattern(x).search(norm)]
    missing = [x for x in wanted if x not in found]
    report.add(
        name,
        len(found) / len(wanted) >= threshold,
        critical=True,
        detail=f"найдено {len(found)}/{len(wanted)}"
        + (f", нет: {', '.join(missing[:5])}" if missing else ""),
    )


def accuracy(report: ValidationReport) -> float:
    """Доля пройденных проверок — грубая метрика «точность парсинга» из плана."""
    if not report.checks:
        return 0.0
    return round(sum(1 for c in report.checks if c.ok) / len(report.checks), 3)
