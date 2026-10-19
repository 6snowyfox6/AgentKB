"""Дефис на конце строки: «слово- слово» — перенос внутри слова или настоящий дефис.

PDF этого не говорит, а одно правило на все документы ошибается: решение — по
варианту документа, который выбрал зонд (profile.py). Без переносов — дефис
настоящий; с переносами и с размеченными переносами — правила языка.

Правила языка — без словаря, проверены на ручной разметке 160 мест из базы
(13.10.2026): заглавная после дефиса или сокращение/элемент перед ним — дефис
настоящий; русское составное (первая часть ≥ 5 букв на -о/-е, вторая ≥ 5 букв) —
дефис; иначе русское — склеить. Русские 74 места: верно 68, ошибок 6 (составные с
минералом: «кварц- серицитовые»). Латиница, обе части со строчной («hy- drothermal»
/ «ore- forming»): без словаря не решить (18 переносов на 22 настоящих) — не трогается.

Правила — для русского и английского (words.own_language): текст другого языка не
трогается вовсе — в немецком «Erz- und Kupferlagerstätten» дефис настоящий, а
правило склейки о таком не знает.
"""

from __future__ import annotations

import re

from .profile import PLAIN, SCAN, UNKNOWN
from .words import own_language

_LETTERS = "A-Za-zА-Яа-яЁё"
_LINE_HYPHEN = re.compile(rf"\b([{_LETTERS}]{{2,}})- ([{_LETTERS}]{{2,}})\b")
_HANGING = {"и", "или", "либо", "а", "and", "or", "nor", "to"}  # «северо- и юго-западный»
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_ELEMENT = re.compile(r"[A-Z][a-z]?")


def line_hyphen(first: str, second: str) -> str | None:
    """Дефис на конце строки по правилам языка: склеить («ab»), оставить дефис («a-b»)
    или None — не трогать."""
    if second.lower() in _HANGING:
        return None
    if second[0].isupper():
        return f"{first}-{second}"  # «Albite- Muscovite», «Mongol- Okhotsk»
    if sum(ch.isupper() for ch in first) >= 2 or _ELEMENT.fullmatch(first):
        return f"{first}-{second}"  # «NW- high», «Pb- изотопного», «PGE- Mo»
    if _CYRILLIC.search(first + second):
        if len(first) >= 5 and first[-1] in "оеОЕ" and len(second) >= 5:
            return f"{first}-{second}"  # «сульфидно- кварцевым», «северо- западного»
        return first + second  # «метасома- тические», «труд- нодоступных»
    return None  # латиница, обе со строчной: «hy- drothermal» или «ore- forming» — не решить


def join_line_hyphens(text: str, variant: str) -> str:
    """«слово- слово» (дефис на конце строки, строки склеены пробелом) — по варианту."""
    return fix_line_hyphens(text, variant)[0]


def fix_line_hyphens(text: str, variant: str) -> tuple[str, int]:
    """То же и сколько мест исправлено — для отчёта оркестратора."""
    if not text or variant in (UNKNOWN, SCAN) or "- " not in text:
        return text, 0
    if not own_language(text):
        return text, 0
    fixed = 0

    def fix(m: re.Match[str]) -> str:
        nonlocal fixed
        first, second = m.group(1), m.group(2)
        if variant == PLAIN:  # переносов нет — дефис настоящий
            new = m.group(0) if second.lower() in _HANGING else f"{first}-{second}"
        else:
            new = line_hyphen(first, second) or m.group(0)
        fixed += new != m.group(0)
        return new

    return _LINE_HYPHEN.sub(fix, text), fixed
