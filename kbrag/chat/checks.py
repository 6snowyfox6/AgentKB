"""Проверка ответа модели: номера фрагментов, отказ, утверждения без ссылки."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

NO_ANSWER = "В базе знаний нет ответа"  # начало фразы-отказа из правила 3

_CITE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
CITE_MARK = re.compile(r"[ \t]*\[\d+(?:\s*[,;]\s*\d+)*\]")
# Фразы о самих фрагментах, а не о предмете базы: «по данным статей базы», «во
# фрагментах нет», «в базе знаний об этом ничего».
_ABOUT_SOURCES = re.compile(
    r"фрагмент|в базе|статей базы|статьях базы|по данным статей|" r"в найденных", re.IGNORECASE
)


def _norm(text: str) -> str:
    return re.sub(r"[«»\"'*_.\s]+", " ", text.lower().replace("ё", "е")).strip()


def is_refusal(start: str) -> bool:
    """Начинается ли ответ с фразы «в базе знаний нет ответа» (правило 3)."""
    return _norm(start).startswith(_norm(NO_ANSWER))


def citations(answer: str, count: int) -> tuple[list[int], list[int]]:
    """Какие номера фрагментов в ответе есть и какие из них выдуманы."""
    found: set[int] = set()
    for group in _CITE.findall(answer):
        for part in re.split(r"[,;]", group):
            if part.strip().isdigit():
                found.add(int(part))
    used = sorted(n for n in found if 1 <= n <= count)
    unknown = sorted(n for n in found if not 1 <= n <= count)
    return used, unknown


# Строка-заголовок: «### Методы», «**1) Признаки рудных узлов**». Считалась
# утверждением без ссылки — у ответов из трёх разделов треть «утверждений» были
# заголовками.
_HEADING = re.compile(r"(?m)^\s*(?:#{1,6}\s+[^\n]*|\*\*[^*\n]+\*\*:?|__[^_\n]+__:?)\s*$")


def coverage(answer: str) -> tuple[int, int]:
    """Сколько утверждений ответа подкреплены ссылкой: (со ссылкой, всего).

    Утверждение — предложение или пункт списка длиннее 40 знаков. Не считаются
    фразы о самих фрагментах («По данным статей базы…», «Во фрагментах нет…»):
    это не факт из статьи, ссылка к ним не нужна. Ссылка может стоять и сразу
    после точки — она относится к предыдущему предложению.
    """
    text = _HEADING.sub("", answer)  # заголовок раздела — не утверждение
    text = re.sub(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", "", text)
    text = re.sub(r"([.!?…])\s*((?:\[\d+(?:\s*[,;]\s*\d+)*\]\s*)+)", r" \2\1", text)
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    claims = [
        p
        for p in parts
        if len(_CITE.sub("", p).strip()) > 40
        and not (_ABOUT_SOURCES.search(p) and not _CITE.search(p))
    ]
    cited = [p for p in claims if _CITE.search(p)]
    return len(cited), len(claims)


# Конец предложения вместе со ссылками после точки: «…разломов. [2][3] Далее…».
_SENTENCE_END = re.compile(r"([.!?…])((?:\s*\[\d+(?:\s*[,;]\s*\d+)*\])*)\s+")
_BULLET = re.compile(r"^(\s*(?:[-*•]|\d+[.)])\s+)")


def cited_numbers(part: str) -> list[int]:
    """Номера фрагментов, на которые ссылается кусок ответа."""
    out: list[int] = []
    for group in _CITE.findall(part):
        for n in re.split(r"[,;]", group):
            if n.strip().isdigit() and int(n) not in out:
                out.append(int(n))
    return out


def about_sources(part: str) -> bool:
    """Фраза о самих фрагментах, а не о предмете базы («во фрагментах нет…»)."""
    return bool(_ABOUT_SOURCES.search(part))


def is_claim(part: str) -> bool:
    """Утверждение (как в coverage): длиннее 40 знаков без ссылок, не подводка к списку
    («используют следующие методы:») и не фраза о самих фрагментах без ссылки."""
    bare = _CITE.sub("", part).strip()
    if len(bare) <= 40 or bare.endswith(":"):
        return False
    return bool(_CITE.search(part)) or not _ABOUT_SOURCES.search(part)


def rewrite(answer: str, change: Callable[[str], str | None]) -> str:
    """Ответ, в котором каждое предложение (и пункт списка) прошло через change: строка —
    новый вид, None — убрать. Заголовок, под которым ничего не осталось, уходит тоже.
    Так же режут ответ drop_uncited, cite_by_numbers и сверка ссылок (support.py)."""
    lines: list[str] = []
    for line in answer.split("\n"):
        if not line.strip() or _HEADING.fullmatch(line):
            lines.append(line)
            continue
        bullet = _BULLET.match(line)
        prefix = bullet.group(1) if bullet else ""
        body = line[len(prefix) :]
        parts = _SENTENCE_END.sub(lambda m: m.group(1) + m.group(2) + "\0", body).split("\0")
        kept = [new for p in parts if (new := change(p.strip())) is not None]
        rest = " ".join(p for p in kept if p)
        if rest:
            lines.append(prefix + rest)
    # Заголовок без текста под ним и лишние пустые строки.
    out: list[str] = []
    for i, line in enumerate(lines):
        if line.strip() and _HEADING.fullmatch(line):
            after = next((x for x in lines[i + 1 :] if x.strip()), "")
            if not after or _HEADING.fullmatch(after):
                continue
        if not line.strip() and (not out or not out[-1].strip()):
            continue
        out.append(line)
    return "\n".join(out).strip()


def drop_uncited(answer: str) -> tuple[str, list[str]]:
    """Ответ без утверждений, у которых нет ссылки на фрагмент.

    Правило 2 просит модель писать только то, что подкреплено фрагментом, но
    ручная проверка 190 утверждений (.local/logs/quality-20261005) нашла 9 % фраз без
    ссылки — общие слова модели, а не статей. Здесь они убираются: утверждение
    (is_claim) без номера фрагмента. Возвращает (новый текст, убранные фразы)."""
    removed: list[str] = []

    def change(part: str) -> str | None:
        if is_claim(part) and not _CITE.search(part):
            removed.append(part)
            return None
        return part

    return rewrite(answer, change), removed


# Число в тексте: 1.660, 50–500 (две штуки), 1:50 000 (50000). Значимое — из двух цифр
# и больше или с дробью: одиночные 1–9 есть почти в любом фрагменте.
_NUMBER = re.compile(r"\d{1,3}(?:[   ]\d{3})+(?![.,]?\d)|\d+(?:[.,]\d+)?")


def numbers(text: str) -> set[str]:
    """Значимые числа текста в одном виде: без пробелов в разрядах, с точкой в дроби."""
    out = set()
    for raw in _NUMBER.findall(_CITE.sub(" ", text)):
        value = re.sub(r"[   ]", "", raw).replace(",", ".")
        if len(value.replace(".", "")) >= 2:
            out.add(value)
    return out


def cite_by_numbers(answer: str, sources: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Утверждению без ссылки, все значимые числа которого есть в одном фрагменте, —
    ссылка на этот фрагмент, а не вычёркивание: верная цифра по радону («максимум
    5343,33 в точке TTN-07») убиралась как фраза без ссылки, и ответ говорил, что
    значений нет (стресс-тест 20.10.2026). Возвращает (текст, получившие ссылку)."""
    known = [(s["n"], numbers(s.get("text") or "")) for s in sources]
    cited: list[str] = []

    def change(part: str) -> str | None:
        if not is_claim(part) or _CITE.search(part):
            return part
        need = numbers(part)
        if not need:
            return part
        match = next((n for n, have in known if need <= have), None)
        if match is None:
            return part
        cited.append(part)
        end = re.search(r"[.!?…]$", part)
        return (part[:-1] + f" [{match}]" + part[-1]) if end else f"{part} [{match}]"

    return rewrite(answer, change), cited
