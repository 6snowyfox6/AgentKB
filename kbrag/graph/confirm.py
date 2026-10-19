"""Подтверждение фактов: модель отвечает на вопросы по цитате, код сверяет ответ с фактом.

Спросить модель «верен ли факт?» не работает: на 100 размеченных вручную фактах
Qwen3-14B поймала 5 ошибок из 24 и отбросила бы 10 верных. Отвечать на вопрос по
предложению модель умеет лучше, а сравнивает ответ с фактом уже код. По каждому
факту четыре проверки — два вопроса и два пропуска, вперёд и назад:

    «Согласно предложению, „X“ — приурочено к — что именно?»   → должно быть Y
    «Согласно предложению, что именно — приурочено к — „Y“?»   → должно быть X
    «X приурочено к ___»                                        → Y
    «___ приурочено к Y»                                        → X

Все четыре — одним запросом к модели (раньше — двумя: вопросы и пропуски
отдельно). На независимой сотне размеченных фактов (.local/logs/quality-20261005)
одним запросом: среди фактов с ≥ 3 подтверждениями верных 83 % при 81 %
сохранённых верных — как двумя запросами на той же видеокарте; с 4 — 92 % при
63 % (двумя — 92 % при 49 %). Без проверки верных 73 %.

Число совпавших ответов (0–4) пишется в facts.votes. Граф берёт факты с
≥ MIN_VOTES; остальные остаются в базе (порог можно поменять, ничего не
прогоняя заново).

Перепутанное направление ловится само: на вопрос «что приурочено к X?» модель
отвечает тем, что в предложении действительно приурочено к X.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..kb import KB
from ..kb import get as get_kb
from . import store

MIN_VOTES = store.MIN_VOTES
# Ответ на четыре вопроса — несколько десятков токенов; потолок — от зацикливания.
ANSWER_TOKENS = 400

# {область} — из настроек базы (kb.py); шаблон целиком — «подтверждение.система».
SYSTEM = "Ты внимательно читаешь предложения из статей по {область}. Отвечай только JSON."
_RULE = (
    "Отвечай только по этому предложению, словами из него. Если предложение этого прямо "
    "не утверждает (только перечисляет рядом, говорит о другом, говорит обратное) — "
    "ответь «не сказано»."
)
QUESTIONS = (
    "Предложение: «{quote}»\n\n" + _RULE + "\n\n"
    "1. Согласно предложению, «{src}» — {rel} — что именно?\n"
    "2. Согласно предложению, что именно — {rel} — «{dst}»?\n"
    "3. Заполни пропуск так, как утверждает предложение: {src} {rel} ___\n"
    "4. Заполни пропуск так, как утверждает предложение: ___ {rel} {dst}\n\n"
    'Верни JSON: {{"1": "...", "2": "...", "3": "...", "4": "..."}}'
)
NOT_SAID = "не сказано"


def _stems(text: str) -> set[str]:
    return {w.lower()[:5] for w in re.findall(r"[^\W\d_]{3,}", text)}


def matches(answer: Any, name: str) -> bool:
    """Ответ называет то же, что имя в факте: половина основ имени — или ответ уже, но из него."""
    text = str(answer or "")
    found, want = _stems(text), _stems(name)
    if not found or not want or NOT_SAID in text.lower():
        return False
    common = found & want
    return len(common) / len(want) >= 0.5 or bool(common and found <= want)


def votes(llm: Any, src: str, relation: str, dst: str, quote: str, kb: KB | None = None) -> int:
    """Сколько из четырёх проверок факт прошёл (0–4). Один запрос к модели."""
    prompt = QUESTIONS.format(quote=quote, src=src, rel=relation, dst=dst)
    system = (kb or get_kb()).prompt("подтверждение.система", SYSTEM)
    try:
        answer = llm.chat_json(system, prompt, temperature=0, max_tokens=ANSWER_TOKENS)
    except ValueError:  # ответ оборван или не JSON — факт не подтверждён
        return 0
    if not isinstance(answer, dict):
        return 0
    return (
        matches(answer.get("1"), dst)
        + matches(answer.get("2"), src)
        + matches(answer.get("3"), dst)
        + matches(answer.get("4"), src)
    )


def fact_votes(llm: Any, fact: Any, kb: KB | None = None) -> int:
    """votes() для факта из facts.check — чтобы подтверждать сразу при проходе."""
    return votes(llm, fact.src, fact.relation, fact.dst, fact.quote, kb)


def run(
    conn: Any,
    llm: Any,
    limit: int | None = None,
    log: Any = print,
    stop_after_errors: int = 3,
    workers: int = 1,
    kb: KB | None = None,
) -> dict[str, int]:
    """Проверить факты, которых ещё не проверяли. Возобновляемо: прервали — продолжит.

    workers — сколько фактов у модели одновременно (см. facts.run); в базу пишет
    только этот поток."""
    kb = kb or get_kb()
    rows = store.unconfirmed(conn, limit=limit, kb=kb.path)
    stats = {"todo": len(rows), "done": 0, "kept": 0, "errors": 0}
    errors_in_row = 0
    started = time.monotonic()

    def ask(row: tuple[Any, ...]) -> int:
        return votes(llm, row[1], row[2], row[3], row[4], kb)

    pool = ThreadPoolExecutor(max(1, workers))
    try:
        # Пачками по workers: модель занята целиком, а остановиться можно в любой момент.
        for start in range(0, len(rows), max(1, workers)):
            batch = rows[start : start + max(1, workers)]
            futures = [pool.submit(ask, row) for row in batch]
            stop = False
            for i, (row, future) in enumerate(zip(batch, futures, strict=True), start=start + 1):
                try:
                    got = future.result()
                    errors_in_row = 0
                except Exception as exc:  # noqa: BLE001 — модель молчит: этот факт потом
                    stats["errors"] += 1
                    errors_in_row += 1
                    if log:
                        log(f"  модель не ответила: {type(exc).__name__}: {exc}")
                    if errors_in_row >= stop_after_errors:
                        if log:
                            log(
                                "  Модель молчит несколько раз подряд — останавливаюсь; "
                                "продолжится отсюда."
                            )
                        stop = True
                        break
                    continue
                store.set_votes(conn, row[0], got)
                stats["done"] += 1
                stats["kept"] += got >= MIN_VOTES
                if i % 50 == 0 or i == len(rows):
                    conn.commit()
                    if log:
                        pace = (time.monotonic() - started) / i
                        log(
                            f"  [{i}/{len(rows)}] подтверждено ≥{MIN_VOTES} из 4: {stats['kept']} "
                            f"из {stats['done']} · {pace:.1f} с на факт"
                        )
            if stop:
                break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    conn.commit()
    return stats
