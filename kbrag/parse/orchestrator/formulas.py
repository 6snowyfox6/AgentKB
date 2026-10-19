"""Формулы: чистка LaTeX, который выдала модель формул docling.

Отдельно от docling.py: здесь только регулярные выражения — чистка работает в основном
процессе (оркестратор), куда тяжёлый docling не тянем.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
# Номер формулы, который модель дописывает в LaTeX («\quad \ \ ( 8 )», «& & ( 1 3 )»),
# и всё после него: там бывает мусор («\quad p a r { x }»).
_EQ_NUMBER = re.compile(r"(?:\s*(?:\\q?quad|\\ |&))+\s*\(\s*\d(?:\s*\d)*\s*\)[\s\S]*$")
# Модель дочитывает после формулы соседнюю строку текста: «\\ \intertext { f l e t h r
# o s …}» (бывает до предела длины ответа) или «\\ \text{simplified data through …}».
# Это не формула: в статье Nature (13.10.2026) так в 5 формулах из 13. Между переносом
# и \text бывает выравнивание: «\\ & \text {derivation} …» (статья FMM, 14.10.2026).
_INTERTEXT = re.compile(
    r"\s*(?:\\\\\s*)?(?:&\s*)*\\intertext\b[\s\S]*$|\s*\\\\\s*(?:&\s*)*\\text\b[\s\S]*$"
)
_ALIGN = re.compile(r"\s*(?<!\\)&\s*")  # выравнивание «a & = b» — в одной формуле не нужно
_TRAILING_BREAK = re.compile(r"(?:\s*\\\\)+\s*$")
_ORIG_NUMBER = re.compile(r"\s*\(\s*\d+\s*\)\s*$")
# Номер формулы в тексте PDF: у одной строки — в конце, у системы строк — посередине
# («… Southlist(c); (31) c ∈ Eastlist(b) …»). «f(2)» — не номер: перед скобкой буква.
_ORIG_TAG = re.compile(r"(?<![\w.])\(\s*(\d{1,3})\s*\)(?![\w(])")
# Что дописывает сама чистка: текст из PDF и номер. Повторная чистка (перечанкинг)
# снимает их и ставит заново — иначе «(2) (2)». Старый вид номера (\tag) — тоже.
_OURS = re.compile(
    r"(?:\s*\\quad \\text\{\[.*\]\})?(?:\s*(?:\\tag\{\d{1,3}\}|\\quad \(\d{1,3}\)))?\s*$"
)


@dataclass
class FormulaFix:
    """Чистая формула и что с ней сделано — для отчёта оркестратора."""

    text: str
    tail: bool = False  # убрана соседняя строка, которую модель дописала к формуле
    number: str = ""  # номер из текста PDF — «(N)» в конце формулы
    russian: bool = False  # приписан текст из PDF с русскими обозначениями


def fix_formula(latex: str, orig: str) -> FormulaFix:
    """LaTeX формулы без выравнивания и мусора после формулы. Номер — «\\quad (N)» из
    текста PDF: модель его не видит (рамка узкая), а текст на него ссылается («(55)»).
    Русские обозначения модель читает латиницей («V_{обр}» → «V_{op6}»), а в тексте PDF
    они верные, — тогда он приписывается к формуле: \\text{[K п = V пор V обр · 100 %]}."""
    latex = _OURS.sub("", latex or "")
    tail = bool(_INTERTEXT.search(latex))
    latex = _EQ_NUMBER.sub("", _INTERTEXT.sub("", latex))
    latex = _TRAILING_BREAK.sub("", _ALIGN.sub(" ", latex)).strip()
    if not latex:
        return FormulaFix("", tail=tail)
    orig = " ".join((orig or "").split())
    numbers = _ORIG_TAG.findall(orig)
    number = numbers[-1] if numbers else ""
    # \tag{N} просмотрщик без его поддержки показывает буквами «tag» — похоже на тангенс;
    # «(N)» после отступа виден одинаково везде (так же пишет номер Marker).
    tag = f" \\quad ({number})" if number else ""
    orig = _ORIG_NUMBER.sub("", orig)
    if _CYRILLIC.search(orig):
        return FormulaFix(f"{latex} \\quad \\text{{[{orig}]}}{tag}", tail, number, russian=True)
    return FormulaFix(latex + tag, tail, number)


def tidy_formula(latex: str, orig: str) -> str:
    """Только чистый LaTeX (см. fix_formula)."""
    return fix_formula(latex, orig).text
