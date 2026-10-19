"""Чистка текста после разбора: то, из-за чего слово не находит поиск по словам.

Обе беды — от PDF, а не от парсера, и чинятся одинаково для любого парсера:

1. Мягкий перенос (U+00AD) — место, где слово можно разорвать в конце строки. В
   части PDF (CyberLeninka и др.) им размечены все переносы, и при склейке строк
   слово остаётся разорванным: «ра¬ - бот», «рай¬ онов», «недоста¬ то¬ чно». Мягкий
   перенос всегда внутри слова — по нему слово склеивается вместе с пробелами и
   видимым дефисом. Обычный дефис в конце строки так не починить: «госу- дарственных»
   склеивать, а «рудно- магматических» — нет, без словаря не различить.
2. Буквы-двойники: в слове из кириллицы латинская «p» или «x» («веpxняя»), в формуле
   из латиницы кириллическая «О» («TiО»). Глазом не видно, поиск по словам не
   находит. Буквы меньшинства, у которых есть двойник в алфавите большинства,
   заменяются; если хоть одной двойника нет («Pbизотопного», «Feместорождение») —
   это не подмена, а пропущенный пробел, слово не трогается.

Замер 13.10.2026 по корпусу (104 статьи): разорванных слов — 255 в двух статьях
(мягкие переносы — в 8), слов из смеси алфавитов — 60 в 15 статьях.

Мягкий перенос — знак самого Юникода, он внутри слова в любом языке: склеивается
всегда. Буквы-двойники проверены только на русском и английском (OWN_LANGUAGES):
в украинском, сербском, немецком свои буквы и свои слова из двух алфавитов, и правило
их портило бы. Текст другого языка (lang.detect) не трогается; текста мало, чтобы
понять язык, — правило работает, как раньше.
"""

from __future__ import annotations

import re

from ...lang import detect

# Языки, на которых проверены правила чистки (буквы-двойники здесь, дефисы на концах
# строк — hyphens.py). Пустой — язык не понять (мало текста): правила работают.
OWN_LANGUAGES = frozenset({"ru", "en", ""})


def own_language(text: str) -> bool:
    """Текст — на языке, для которого правила чистки проверены."""
    return detect(text) in OWN_LANGUAGES


# После мягкого переноса — пробелы, перевод строки (у PyMuPDF строки не склеены) и
# видимый дефис; абзац мягким переносом не кончается, поэтому строка — одна.
_SOFT_HYPHEN = re.compile("\u00ad[ \t]*\n?[ \t]*(?:-[ \t]*\n?[ \t]*)?")

_LATIN_TO_CYRILLIC = dict(zip("ABCEHKMOPTXaceopxy", "АВСЕНКМОРТХасеорху", strict=True))
_CYRILLIC_TO_LATIN = {c: lat for lat, c in _LATIN_TO_CYRILLIC.items()}
_MIXED_WORD = re.compile(
    r"\b(?=[A-Za-zА-Яа-яЁё]*[А-Яа-яЁё])(?=[A-Za-zА-Яа-яЁё]*[A-Za-z])[A-Za-zА-Яа-яЁё]+"
)
_CYRILLIC = set("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя")
# Символ элемента, приклеенный к слову («Moпорфировых» — Mo-порфировых): две
# латинские буквы-двойника, но это не подмена, а пропущенный дефис.
_ELEMENT_PREFIX = re.compile(r"[A-Z][a-z][А-Яа-яЁё]")


def join_soft_hyphens(text: str) -> str:
    """Слово, разорванное мягким переносом, — целиком: «ра¬ - бот» → «работ»."""
    return _SOFT_HYPHEN.sub("", text) if "\u00ad" in text else text


def _one_script(match: re.Match[str]) -> str:
    word = match.group()
    cyrillic = sum(ch in _CYRILLIC for ch in word)
    latin = len(word) - cyrillic
    if cyrillic == latin:
        return word  # не ясно, какой алфавит свой
    to_cyrillic = cyrillic > latin
    if to_cyrillic and _ELEMENT_PREFIX.match(word):
        return word
    swap = _LATIN_TO_CYRILLIC if to_cyrillic else _CYRILLIC_TO_LATIN
    minority = [ch for ch in word if (ch in _CYRILLIC) != to_cyrillic]
    if not all(ch in swap for ch in minority):
        return word
    return "".join(swap[ch] if (ch in _CYRILLIC) != to_cyrillic else ch for ch in word)


def fix_homoglyphs(text: str) -> str:
    """Слово из смеси алфавитов — в одном алфавите, если чужие буквы — двойники.
    Только в русском и английском тексте (own_language)."""
    if not _MIXED_WORD.search(text) or not own_language(text):
        return text
    return _MIXED_WORD.sub(_one_script, text)


def tidy_text(text: str) -> str:
    """Всё вместе: мягкие переносы, затем буквы-двойники."""
    return fix_homoglyphs(join_soft_hyphens(text)) if text else text
