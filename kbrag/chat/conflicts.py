"""Расхождения между статьями: отвечают ли фрагменты разных статей на вопрос по-разному."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .. import acl
from .messages import fragment_block
from .settings import Settings

# Две статьи говорят об одном разное — модель выбирала одну и молчала о другой: на 8
# вопросах с нарочно противоречащими фрагментами обе позиции не показала ни разу, и
# правило «покажи обе» в инструкции этого не изменило. Работает другое: модель коротко
# отвечает за каждый фрагмент (это она умеет), а противоречат ли ответы, решает модель
# одним словом по двум коротким ответам (числа, что сходятся, сверяет код — без
# модели). Найдено — в задание идёт пометка с
# номерами и словами фрагментов. Проверка 2026-10-11 на 8 + 8 вопросах: расхождение
# найдено в 6 из 8, в 8 из 8 согласных случаев лишнего нет. Спрашивать «одно или
# разное» нельзя: на 19 настоящих вопросах оценки «расхождения» находились в 17,
# всего 84 пары (в 9 — все шесть сверенных): разные стороны ответа модель считала
# разными ответами. Строгое «противоречат» (одно и то же, но вместе неверно) —
# 4 пары в 3 вопросах.
CONFLICT_PAIRS = 6  # сколько пар фрагментов из разных статей сверять самое большее
CONFLICT_CHARS = 1200  # сколько текста фрагмента читать для короткого ответа
SHORT_SYSTEM = "Ты внимательно читаешь фрагменты статей. Отвечай только JSON."
SHORT_USER = """Вопрос: «{question}»

Фрагменты:

{fragments}

Что отвечает на вопрос КАЖДЫЙ фрагмент? Одной короткой фразой по-русски, словами \
фрагмента, без своих добавлений; называет фрагмент район или объект — с ним. Фрагмент \
не отвечает на вопрос — «не отвечает».

Верни JSON: {{{keys}}}"""
SAME_SYSTEM = "Ты сравниваешь ответы двух источников. Отвечай только JSON."
SAME_USER = """Вопрос: «{question}»
Ответ по одному источнику: «{a}»
Ответ по другому источнику: «{b}»

Противоречат ли ответы друг другу? Противоречие — только когда оба говорят об одном \
и том же (тот же объект, район, величина) и не могут быть верны вместе: разные \
значения одной величины, «да» и «нет», разные причины одного и того же. Ответы о \
разных объектах или районах, разные стороны, примеры или методы, один подробнее \
другого — не противоречие.
Верни JSON: {{"ответ": "противоречат"}} или {{"ответ": "не противоречат"}}"""
_NUMBER_RANGE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:[-–—]\s*(\d+(?:[.,]\d+)?))?")


def _ranges(text: str) -> list[tuple[float, float]]:
    out = []
    for a, b in _NUMBER_RANGE.findall(text):
        x, y = float(a.replace(",", ".")), float((b or a).replace(",", "."))
        out.append((min(x, y), max(x, y)))
    return out


def numbers_clash(a: str, b: str) -> bool | None:
    """Числа в обоих ответах: диапазоны не пересекаются — разное («850–990 ‰» и
    «620–700 ‰»), пересекаются — одно («850–990» и «860–980»). Чисел нет — None."""
    ra, rb = _ranges(a), _ranges(b)
    if not ra or not rb:
        return None
    for lo1, hi1 in ra:
        for lo2, hi2 in rb:
            slack = 0.03 * max(hi1, hi2)
            if lo1 <= hi2 + slack and lo2 <= hi1 + slack:
                return False
    return True


def _silent(answer: str) -> bool:
    return not answer.strip() or "не отвечает" in answer.lower()


def find_conflicts(
    question: str, sources: list[dict[str, Any]], settings: Settings
) -> list[dict[str, Any]]:
    """Пары фрагментов из разных статей, которые отвечают на вопрос по-разному."""
    frags = [s for s in sources if s.get("kind") != "датасет"]
    if len({s["doc_id"] for s in frags}) < 2:
        return []
    client = settings.client()
    blocks = "\n\n".join(fragment_block(dict(s, text=s["text"][:CONFLICT_CHARS])) for s in frags)
    keys = ", ".join(f'"{s["n"]}": "..."' for s in frags)
    try:
        data = client.chat_json(
            SHORT_SYSTEM,
            SHORT_USER.format(question=question, fragments=blocks, keys=keys),
            temperature=0,
            max_tokens=60 * len(frags) + 50,
        )
    except Exception:  # noqa: BLE001 — не вышло сверить: отвечаем как обычно
        return []
    said = {s["n"]: str(data.get(str(s["n"])) or "") for s in frags}
    pairs = [
        (a, b)
        for i, a in enumerate(frags)
        for b in frags[i + 1 :]
        if a["doc_id"] != b["doc_id"] and not _silent(said[a["n"]]) and not _silent(said[b["n"]])
    ][:CONFLICT_PAIRS]

    def differ(pair: tuple[dict[str, Any], dict[str, Any]]) -> bool:
        a, b = said[pair[0]["n"]], said[pair[1]["n"]]
        if numbers_clash(a, b) is False:
            return False  # числа сходятся — одно и то же, модель не нужна
        try:
            got = client.chat_json(
                SAME_SYSTEM, SAME_USER.format(question=question, a=a, b=b), temperature=0
            )
        except Exception:  # noqa: BLE001
            return False
        verdict = str(got.get("ответ", "")).lower()
        return "противореч" in verdict and "не противореч" not in verdict

    if not pairs:
        return []
    with ThreadPoolExecutor(min(4, len(pairs))) as pool:
        verdicts = list(pool.map(acl.carry(differ), pairs))
    return [
        {"a": a["n"], "b": b["n"], "said_a": said[a["n"]], "said_b": said[b["n"]]}
        for (a, b), clash in zip(pairs, verdicts, strict=True)
        if clash
    ]
