"""Зонд: что за документ — до разбора, чтобы обрабатывать его своим вариантом.

Документы разные, и одно правило на всех ошибается. Самое наглядное — дефис в конце
строки: в одних PDF это перенос внутри слова («госу- дарственных» — склеить), в
других — настоящий дефис («кварц- серицитовых»), а сам PDF этого не говорит. Зонд
за доли секунды (PyMuPDF, без моделей) смотрит на текстовый слой и выбирает вариант:

    скан                 текстового слоя нет            не разбирается
    переносы размечены   в тексте есть мягкий перенос ¬ по ¬ склеивается (words), простой дефис — правила языка
    с переносами         строк, кончающихся дефисом, ≥ 1 % и ≥ 5  правила языка
    без переносов        таких строк почти нет          дефис на конце строки настоящий: «геолого-геофизический»

Кто сделал PDF (Word, LaTeX, InDesign), пишется в журнал, но решения не меняет:
учебник из Word с включёнными переносами кончает переносом 11 % строк.

Правила дефиса на конце строки по варианту — в hyphens.py.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from ..diag import describe
from ..models import ParseInput

SCAN = "скан"
MARKED = "переносы размечены"
HYPHENATED = "с переносами"
PLAIN = "без переносов"
UNKNOWN = "неизвестно"  # PDF не открылся зондом — дефисы не трогаем

SAMPLE_PAGES = 12  # страниц для зонда — с начала документа
SCAN_CHARS = 50  # знаков текстового слоя на страницу — меньше значит скан
HYPHEN_RATE = 0.01  # доля строк, кончающихся переносом, от которой вёрстка «с переносами»
# …и не меньше стольких строк: в коротком документе без переносов (Google Docs) три
# настоящих дефиса на конце строки («ИИ-/решений», «Стартап-/полигон») — уже 1.6 %.
MIN_LINE_BREAKS = 5
MIN_SOFT_HYPHENS = 5  # мягких переносов, чтобы считать их разметкой, а не случайностью

_LETTERS = "A-Za-zА-Яа-яЁё"
_LINE_BREAK = re.compile(rf"[{_LETTERS}]-$")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")


@dataclass
class Profile:
    """Что за документ и почему решено так — для журнала."""

    variant: str
    reason: str
    language: str = ""
    producer: str = ""
    pages: int = 0
    chars_per_page: float = 0.0
    soft_hyphens: int = 0
    line_breaks: int = 0  # строк, кончающихся дефисом перед строчной буквой
    lines: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe(inp: ParseInput) -> Profile:
    """Вариант документа по его текстовому слою (первые SAMPLE_PAGES страниц).
    Не прочитался — вариант «неизвестно» с причиной: дефисы не трогаем, а разобрать
    документ попробуют парсеры, у них свои способы."""
    started = time.monotonic()
    try:
        p = _measure(inp)
    except Exception as exc:  # noqa: BLE001
        p = Profile(UNKNOWN, f"зонд не прочитал PDF: {describe(exc)}")
    p.seconds = round(time.monotonic() - started, 3)
    return p


def _measure(inp: ParseInput) -> Profile:
    import pymupdf

    doc = (
        pymupdf.open(stream=inp.data, filetype="pdf")
        if inp.data is not None
        else pymupdf.open(inp.path)
    )
    with doc:
        meta = doc.metadata or {}
        pages = doc.page_count
        sample = min(pages, SAMPLE_PAGES)
        text = "".join(doc[i].get_text() for i in range(sample))
    lines = text.splitlines()
    breaks = sum(
        1
        for line, following in zip(lines, lines[1:], strict=False)
        if _LINE_BREAK.search(line.rstrip()) and following[:1].islower()
    )
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    p = Profile(
        variant=UNKNOWN,
        reason="",
        language="ru" if cyrillic > latin else "en",
        producer=" / ".join(x for x in (meta.get("creator"), meta.get("producer")) if x)[:120],
        pages=pages,
        chars_per_page=round(len(text) / max(1, sample), 1),
        soft_hyphens=text.count("\u00ad"),
        line_breaks=breaks,
        lines=len(lines),
    )
    return decide(p)


def decide(p: Profile) -> Profile:
    """Вариант по признакам зонда (отдельно от чтения PDF — проверяется на числах)."""
    rate = p.line_breaks / max(1, p.lines)
    breaks = f"строк с переносом {p.line_breaks} из {p.lines} ({rate:.1%})"
    if p.chars_per_page < SCAN_CHARS:
        p.variant = SCAN
        p.reason = f"текстового слоя почти нет: {p.chars_per_page:.0f} знаков на страницу"
    elif p.soft_hyphens >= MIN_SOFT_HYPHENS:
        p.variant, p.reason = MARKED, f"мягких переносов {p.soft_hyphens}"
    elif rate >= HYPHEN_RATE and p.line_breaks >= MIN_LINE_BREAKS:
        p.variant, p.reason = HYPHENATED, breaks
    else:
        p.variant, p.reason = PLAIN, breaks
    return p
