"""Поиск с моделью: широкий набор → модель отбирает → ищет недостающее.

Модель читает всех кандидатов пачками, называет подходящие, говорит, чего не
хватает, и составляет новые запросы — до MAX_ROUNDS кругов. Молчит — отбор по
порогу близости (search.find_sources)."""

from __future__ import annotations

import re
import time
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ..index.search import cap_per_doc
from ..kb import KB, languages
from ..kb import get as get_kb
from .plan import Plan, clean_queries
from .search import (
    PART_MIN,
    add_neighbors,
    find_sources,
    fragment_key,
    is_close,
    merged_search,
    source_dict,
)
from .settings import Settings

# Поиск с моделью: широкий набор кандидатов, модель отбирает и ищет ещё.
WIDE = 16  # кандидатов на простой вопрос (и на запросы нового круга)
WIDE_PART = 8  # на каждую часть составного вопроса и на вопрос целиком
LOOSE_SIMILARITY = 0.35  # мягкий порог для кандидатов: отбирает уже модель
# Модель читает кандидатов пачками по JUDGE_MAX — все, а не первые сколько-то:
# раньше при трёх частях третьей доставалось 4 кандидата из 10, а найденное
# по вопросу целиком модель не видела вовсе. Текста — CANDIDATE_CHARS: фрагмент
# около 2000 знаков, и по первым 450 нужное отбрасывалось, если суть дальше.
JUDGE_MAX = 12
# Пачки одного круга уходят в модель разом: vLLM сам собирает одновременные запросы в
# одну пачку (--max-num-seqs 8 в docker-compose.yml), и круг стоит одной пачки, а не
# суммы. Кандидаты у пачек разные, поэтому «уже отобрано» каждой — на начало круга.
JUDGE_WORKERS = 4
CANDIDATE_PER_DOC = 3  # кандидатов из одной статьи в наборе (пока есть другие)
JUDGE_SLACK = 4  # короткий хвост пачки читается вместе с предыдущей
CANDIDATE_CHARS = 1000
MAX_ROUNDS = 3  # кругов поиска самое большее
EXTRA_QUERIES = 3  # новых запросов за круг
# Реранкер (index/rerank.py) вместо модели на первом круге: оценка пары «вопрос —
# фрагмент» от 0 до 1. Подходит — от RERANK_MIN; у bge-reranker-v2-m3 фрагмент по делу
# обычно выше 0,5, похожий словами, но о другом — ниже 0,1. Модель зовётся, только
# если подходящего мало: на часть составного вопроса — меньше PART_MIN, на простой
# вопрос — меньше RERANK_ENOUGH; тогда она говорит, чего не хватает, и даёт запросы.
RERANK_MIN = 0.2
RERANK_ENOUGH = 3
AUTO = "auto"  # реранкер — общий на процесс, если включён в настройках (Settings.rerank)
# Новый запрос, почти совпадающий по основам слов с уже сделанным, не ищется:
# он приносит те же фрагменты, а круг стоит вызова модели.
REPEAT_OVERLAP = 0.6

# {пользователь} — из настроек базы (kb.py); узел может заменить шаблоны целиком:
# «отбор.система» и «отбор».
JUDGE_SYSTEM = (
    "Ты отбираешь фрагменты научных статей, из которых можно ответить на вопрос "
    "{пользователь}, и решаешь, что ещё поискать в базе. Отвечай только JSON."
)

JUDGE_USER = """Вопрос: «{question}»
{parts}{taken}
Кандидаты из базы — нашёл поиск по словам и смыслу, среди них бывают случайные:
{candidates}

1. «подходят» — номера кандидатов, в которых есть сведения для ответа (хотя бы на одну
   часть вопроса), от самых полезных к менее полезным. Кандидат, где только похожие
   слова, а по сути речь о другом, — не бери.
2. «не_хватает» — каких сведений для полного ответа нет ни в одном подходящем
   фрагменте, коротко. Всего хватает — пустой список.
3. «запросы» — до {n} новых поисковых запросов, чтобы найти недостающее: 2–6 слов,
   терминами предметной области, {languages}, не повторяя сделанные
   и не переставляя в них слова: {done}. Всего хватает или искать больше нечего —
   пустой список.

Верни JSON: {{"подходят": [1, 4], "не_хватает": ["..."], "запросы": ["..."]}}"""


@dataclass
class Retrieval:
    """Что нашёл поиск с моделью."""

    sources: list[dict[str, Any]]
    rounds: int = 1  # сколько кругов поиска было
    checked: int = 0  # сколько кандидатов модель прочла
    judged: bool = True  # отбирала модель (False — отбор по порогу, модель молчит)
    judge_sec: float = 0.0  # сколько секунд из поиска ушло на отбор (модель или реранкер)
    reranked: bool = False  # отбирал реранкер (модель — только «чего не хватает»)

    def info(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "checked": self.checked,
            "judged": self.judged,
            "reranked": self.reranked,
        }


def is_candidate(hit: Any) -> bool:
    """Годится ли в кандидаты: все слова запроса в тексте или похожесть от мягкого порога."""
    strict = hit.fts_rank is not None and getattr(hit, "fts_strict", True)
    return strict or (hit.similarity or 0.0) >= LOOSE_SIMILARITY


def candidates(
    conn: Any,
    embedder: Any,
    queries: list[str],
    cache: dict[str, Any],
    limit: int = WIDE,
    kb: str = "",
) -> list[Any]:
    """Кандидаты для модели — не больше CANDIDATE_PER_DOC из одной статьи (лишние — в
    конец): одна подробная статья занимала 9 мест из 15, и модели не из чего было
    выбирать — ответ на составной вопрос шёл по одной статье. kb — узел дерева баз."""
    close = [h for h in merged_search(conn, embedder, queries, cache, kb) if is_candidate(h)]
    return cap_per_doc(close, CANDIDATE_PER_DOC)[:limit]


TOP_DOWN = 10  # кандидатов из поиска сверху вниз


def top_down_candidates(conn: Any, embedder: Any, query: str, kb: str = "") -> list[Any]:
    """Кандидаты поиска сверху вниз — те, что проходят мягкий порог; сбой — без них."""
    from ..index import db, summaries

    if not db.is_real(conn):
        return []  # заглушка в проверках — резюме нет
    try:
        found = summaries.top_down(conn, embedder, query, kb, limit=3 * TOP_DOWN)
    except Exception:  # noqa: BLE001 — нет таблицы резюме, заглушка в проверках
        return []
    return cap_per_doc([h for h in found if is_candidate(h)], CANDIDATE_PER_DOC)[:TOP_DOWN]


def judge_prompt(
    question: str,
    plan: Plan,
    batch: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    kb: KB | None = None,
) -> str:
    parts = ""
    if len(plan.parts) > 1:
        parts = (
            "Части вопроса:\n"
            + "\n".join(f"{i}) {p.question}" for i, p in enumerate(plan.parts, start=1))
            + "\n"
        )
    taken = ""
    if accepted:
        taken = (
            "\nУже отобрано раньше (снова не называй):\n"
            + "\n".join(
                f"- «{c['hit'].title[:90]}»: {' '.join(c['hit'].text.split())[:160]}"
                for c in accepted[:12]
            )
            + "\n"
        )
    lines = []
    for c in batch:
        h = c["hit"]
        year = f", {h.year}" if h.year else ""
        lines.append(
            f"[{c['id']}] «{(h.title or h.doc_id)[:100]}»{year}: "
            f"{' '.join(h.text.split())[:CANDIDATE_CHARS]}"
        )
    shown = "\n\n".join(lines) or (
        "— новых кандидатов нет: подходящее уже отобрано (выше). Скажи, чего не хватает."
        if accepted
        else "— кандидатов нет: по запросам ничего близкого не нашлось."
    )
    done = ", ".join(f"«{q}»" for q in plan.queries + plan.extra) or "—"
    kb = kb or get_kb()
    return kb.prompt(
        "отбор",
        JUDGE_USER,
        question=question.strip(),
        parts=parts,
        taken=taken,
        candidates=shown,
        n=EXTRA_QUERIES,
        done=done,
        languages=languages(kb, " или "),
    )


def judge_fragments(
    question: str,
    plan: Plan,
    batch: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    settings: Settings,
) -> dict[str, Any] | None:
    """Модель читает кандидатов: {"подходят": [id…], "не_хватает": […], "запросы": […]}.
    Не ответила — None: тогда отбор по порогу, как у обычного поиска."""
    kb = get_kb(settings.kb)
    try:
        data = settings.client().chat_json(
            kb.prompt("отбор.система", JUDGE_SYSTEM),
            judge_prompt(question, plan, batch, accepted, kb),
            temperature=0.1,
            timeout=(10, 180),
        )
    except Exception:  # noqa: BLE001
        return None
    return data if isinstance(data, dict) else None


def _query_words(query: str) -> set[str]:
    from ..text import name_key

    return set(name_key(query).split())


def is_repeat(query: str, done: list[str]) -> bool:
    """Тот же запрос, что уже был, — по основам слов, с перестановкой и падежами."""
    words = _query_words(query)
    if not words:
        return True
    for other in map(_query_words, done):
        if other and len(words & other) / len(words | other) >= REPEAT_OVERLAP:
            return True
    return False


def _ids(values: Any) -> list[int]:
    out = []
    for v in values if isinstance(values, list) else []:
        m = re.search(r"\d+", str(v))
        if m and int(m.group()) not in out:
            out.append(int(m.group()))
    return out


def _select(
    accepted: list[dict[str, Any]],
    plan: Plan,
    settings: Settings,
    top_k: int,
    taken: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Отобранное моделью → фрагменты ответа: у каждой части свои места,
    не больше per_doc из статьи (пока есть другие), порядок — как решила модель."""
    per_doc: dict[str, int] = {}
    for s in taken:
        per_doc[s["doc_id"]] = per_doc.get(s["doc_id"], 0) + 1
    chosen: list[dict[str, Any]] = []

    def take(c: Any, strict: Any = True) -> bool:
        if c in chosen or len(chosen) >= top_k:
            return False
        doc = c["hit"].doc_id
        if strict and settings.per_doc and per_doc.get(doc, 0) >= settings.per_doc:
            return False
        per_doc[doc] = per_doc.get(doc, 0) + 1
        chosen.append(c)
        return True

    if len(plan.parts) > 1:
        quota = max(PART_MIN, (top_k - 2) // len(plan.parts))
        for i in range(1, len(plan.parts) + 1):
            got = 0
            for c in accepted:
                if got >= quota:
                    break
                if c["part"] == i and take(c):
                    got += 1
    for c in accepted:
        take(c)
    for c in accepted:  # статей мало — лишние куски тех же статей, в конце
        take(c, strict=False)
    if len(plan.parts) > 1:
        for i, part in enumerate(plan.parts, start=1):
            part.found = sum(c["part"] == i for c in chosen)
    elif plan.parts:
        plan.parts[0].found = len(chosen)
    return chosen


def retrieve(
    conn: Any,
    embedder: Any,
    question: str,
    plan: Plan,
    settings: Settings,
    judge: Any = judge_fragments,
    taken: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    max_rounds: int = MAX_ROUNDS,
    extra: list[Any] | None = None,
    reranker: Any = AUTO,
) -> Generator[Any, Any, Any]:
    """Поиск с моделью. Генератор: отдаёт события status, в конце возвращает Retrieval
    (`result = yield from retrieve(…)`).

    Модель читает всех кандидатов круга — пачками по JUDGE_MAX, все пачки разом. Новый круг
    ищется, только если модель говорит, чего не хватает, и предлагает запросы,
    которых ещё не было; круг, не добавивший ни одного фрагмента, — последний.

    judge=None или молчащая модель — отбор по порогу близости (find_sources).

    С реранкером (reranker — Reranker, None — без него, AUTO — общий, если включён)
    отбирает он, а модель (judge) — только говорит, чего не хватает (reranked_rounds)."""
    taken = list(taken or [])
    if reranker == AUTO:
        from ..index.rerank import shared

        reranker = shared() if settings.rerank else None
    if judge is None and reranker is None:  # ни модели, ни реранкера — отбор по порогу
        sources = find_sources(
            conn, embedder, plan, settings, start=len(taken) + 1, taken=taken, limit=limit
        )
        return Retrieval(sources, rounds=1, checked=0, judged=False)
    top_k = limit or settings.top_k
    cache: dict[str, Any] = {}
    pool: list[dict[str, Any]] = []
    seen = {fragment_key(s) for s in taken}

    spare: list[tuple[Any, ...]] = []  # сверх CANDIDATE_PER_DOC из статьи — если других мало

    def add(hits: Any, part: Any, rnd: Any, capped: bool = True) -> list[dict[str, Any]]:
        new = []
        for h in hits:
            if fragment_key(h) in seen:
                continue
            if capped and sum(c["hit"].doc_id == h.doc_id for c in pool) >= CANDIDATE_PER_DOC:
                spare.append((h, part, rnd))
                continue
            seen.add(fragment_key(h))
            c = {"id": len(pool) + 1, "hit": h, "part": part, "round": rnd}
            pool.append(c)
            new.append(c)
        return new

    multi = len(plan.parts) > 1
    if multi:
        # Все запросы вопроса — одной пачкой, одновременно (search_all): по частям
        # они шли бы по три, а каждая пачка ждала бы предыдущую.
        everything = [q for p in plan.parts for q in (p.queries or [p.question])]
        merged_search(conn, embedder, everything + list(plan.queries), cache, settings.kb)
    # Фрагменты о предмете вопроса из графа — первыми: порога близости у них нет,
    # они найдены не по похожести, а по фактам.
    add(extra or [], None if multi else 1, 1, capped=False)
    kb = settings.kb
    if multi:
        for i, part in enumerate(plan.parts, start=1):
            asked = part.queries or [part.question]
            add(candidates(conn, embedder, asked, cache, WIDE_PART, kb), i, 1)
        add(candidates(conn, embedder, plan.queries, cache, WIDE_PART, kb), None, 1)
    else:
        add(candidates(conn, embedder, plan.queries, cache, WIDE, kb), 1, 1)
    # Сверху вниз (index/summaries.py): статьи по резюме → разделы → фрагменты. Резюме
    # нет — пусто; есть — ещё кандидаты из статей, которые про вопрос целиком.
    add(top_down_candidates(conn, embedder, plan.main or question, kb), None if multi else 1, 1)
    # Ограничение — на весь набор: поиски по частям вопроса складываются, и одна статья
    # набирала 7 мест из 16. Других статей мало — лишние куски той же добирают места.
    for h, part, rnd in spare[: max(0, WIDE - len(pool))]:
        add([h], part, rnd, capped=False)

    if reranker is not None:
        found = yield from reranked_rounds(
            conn, embedder, question, plan, settings, judge, reranker, pool, add, cache, max_rounds
        )
        return _finish(conn, found, plan, settings, top_k, taken, multi)

    fresh: list[dict[str, Any]] = list(pool)
    accepted: list[dict[str, Any]] = []
    rounds, checked, judged, judge_sec = 1, 0, False, 0.0
    while True:
        before = len(accepted)
        batches = [fresh[i : i + JUDGE_MAX] for i in range(0, len(fresh), JUDGE_MAX)] or [[]]
        if len(batches) > 1 and len(batches[-1]) <= JUDGE_SLACK:
            batches[-2:] = [batches[-2] + batches[-1]]  # 12 + 1 — лишний вызов модели
        missing, proposed, silent, answered = [], [], [], False
        packs = f" ({len(batches)} пачки разом)" if len(batches) > 1 else ""
        yield {
            "type": "status",
            "text": (
                f"модель читает найденное: {len(fresh)} фрагментов{packs}…"
                if fresh
                else "в базе ничего близкого — модель подбирает другие запросы…"
            ),
        }
        snapshot = list(accepted)
        started = time.monotonic()
        with ThreadPoolExecutor(min(JUDGE_WORKERS, len(batches))) as executor:
            answers = list(
                executor.map(
                    lambda b, seen=snapshot: judge(question, plan, b, seen, settings), batches
                )
            )
        judge_sec += time.monotonic() - started
        for batch, data in zip(batches, answers, strict=True):
            if data is None:  # модель не ответила на эту пачку
                silent += batch
                continue
            answered = True
            checked += len(batch)
            by_id = {c["id"]: c for c in batch}
            for i in _ids(data.get("подходят") or data.get("relevant")):
                if i in by_id and by_id[i] not in accepted:
                    accepted.append(by_id[i])
            missing += [
                " ".join(str(m).split()) for m in data.get("не_хватает") or [] if str(m).strip()
            ]
            proposed += clean_queries(data.get("запросы") or data.get("queries"), EXTRA_QUERIES)
        if not answered:
            break  # модель молчит — отбор уже сделанный
        judged = True
        if rounds == 1:
            # Модель ответила не на все пачки первого круга — из молчаливых
            # берётся близкое по порогу, как у обычного поиска.
            accepted += [c for c in silent if is_close(c["hit"])]
        if missing or rounds == 1:
            plan.missing = list(dict.fromkeys(missing))[:5]
        if rounds >= max_rounds:
            break
        if rounds > 1 and len(accepted) == before:
            break  # новый круг ничего не добавил
        if accepted and not missing:
            break  # всего хватает
        done = plan.queries + plan.extra
        queries: list[str] = []
        for q in proposed:
            if q.lower() not in {d.lower() for d in done + queries} and not is_repeat(q, done):
                queries.append(q)
        queries = queries[:EXTRA_QUERIES]
        if not queries:
            break
        plan.extra += queries
        rounds += 1
        yield {"type": "status", "text": "ищу недостающее: " + " · ".join(queries)}
        fresh = add(candidates(conn, embedder, queries, cache, WIDE, kb), None, rounds)
        if not fresh and not missing:
            break

    if not judged:  # модель не ответила — по порогу
        sources = find_sources(
            conn, embedder, plan, settings, start=len(taken) + 1, taken=taken, limit=limit
        )
        return Retrieval(sources, rounds=1, checked=0, judged=False, judge_sec=judge_sec)
    result = Retrieval([], rounds=rounds, checked=checked, judged=True, judge_sec=judge_sec)
    return _finish(conn, (result, accepted), plan, settings, top_k, taken, multi)


def _finish(
    conn: Any,
    found: tuple[Retrieval, list[dict[str, Any]]],
    plan: Plan,
    settings: Settings,
    top_k: int,
    taken: list[dict[str, Any]],
    multi: bool,
) -> Retrieval:
    """Отобранное → фрагменты ответа с номерами и соседними кусками."""
    result, accepted = found
    chosen = _select(accepted, plan, settings, top_k, taken)
    for n, c in enumerate(chosen, start=len(taken) + 1):
        s = source_dict(n, c["hit"], c["part"] if multi else None)
        if c["round"] > 1:
            s["round"] = c["round"]
        result.sources.append(s)
    if conn is not None:
        add_neighbors(conn, result.sources)
    return result


def rerank_batch(
    reranker: Any, question: str, plan: Plan, batch: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Кандидаты, которые реранкер счёл подходящими, от лучших. Кандидат части
    составного вопроса оценивается по её вопросу, остальные — по вопросу целиком."""
    whole = (plan.standalone or question).strip()
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in batch:
        part = c["part"]
        asked = (
            plan.parts[part - 1].question
            if len(plan.parts) > 1 and isinstance(part, int) and plan.parts[part - 1].question
            else whole
        )
        groups.setdefault(asked, []).append(c)
    for asked, items in groups.items():
        texts = [
            f"{c['hit'].title}\n{' '.join(c['hit'].text.split())[:CANDIDATE_CHARS]}" for c in items
        ]
        for c, score in zip(items, reranker.score(asked, texts), strict=True):
            c["rerank"] = round(score, 4)
    good = [c for c in batch if c.get("rerank", 0.0) >= RERANK_MIN]
    return sorted(good, key=lambda c: -c["rerank"])


def gaps(plan: Plan, accepted: list[dict[str, Any]]) -> bool:
    """Мало ли подходящего: у части составного вопроса — меньше PART_MIN, у простого
    вопроса — меньше RERANK_ENOUGH."""
    if len(plan.parts) > 1:
        return any(
            sum(c["part"] == i for c in accepted) < PART_MIN for i in range(1, len(plan.parts) + 1)
        )
    return len(accepted) < RERANK_ENOUGH


def reranked_rounds(
    conn: Any,
    embedder: Any,
    question: str,
    plan: Plan,
    settings: Settings,
    judge: Any,
    reranker: Any,
    pool: list[dict[str, Any]],
    add: Any,
    cache: dict[str, Any],
    max_rounds: int,
) -> Generator[Any, Any, tuple[Retrieval, list[dict[str, Any]]]]:
    """Отбор реранкером; модель — только «чего не хватает», и только если мало."""
    fresh, accepted = list(pool), []
    rounds, checked, seconds, judged = 1, 0, 0.0, False
    while True:
        yield {"type": "status", "text": f"оцениваю найденное: {len(fresh)} фрагментов…"}
        started = time.monotonic()
        new = [c for c in rerank_batch(reranker, question, plan, fresh) if c not in accepted]
        accepted += new
        checked += len(fresh)
        seconds += time.monotonic() - started
        if judge is None or rounds >= max_rounds or not gaps(plan, accepted):
            break
        if rounds > 1 and not new:
            break  # новый круг ничего подходящего не дал — модель звать незачем
        yield {"type": "status", "text": "подходящего мало — модель подбирает запросы…"}
        started = time.monotonic()
        data = judge(question, plan, [], list(accepted), settings)
        seconds += time.monotonic() - started
        if data is None:
            break  # модель молчит — что отобрал реранкер, то и есть
        judged = True
        plan.missing = list(
            dict.fromkeys(
                " ".join(str(m).split()) for m in data.get("не_хватает") or [] if str(m).strip()
            )
        )[:5]
        done = plan.queries + plan.extra
        queries = [
            q
            for q in clean_queries(data.get("запросы") or data.get("queries"), EXTRA_QUERIES)
            if q.lower() not in {d.lower() for d in done} and not is_repeat(q, done)
        ][:EXTRA_QUERIES]
        if not queries:
            break
        plan.extra += queries
        rounds += 1
        yield {"type": "status", "text": "ищу недостающее: " + " · ".join(queries)}
        fresh = add(candidates(conn, embedder, queries, cache, WIDE, settings.kb), None, rounds)
        if not fresh:
            break
    found = Retrieval(
        [], rounds=rounds, checked=checked, judged=judged, judge_sec=seconds, reranked=True
    )
    return found, accepted


def run(gen: Any) -> Any:
    """Прогнать генератор-поиск без событий и взять результат (для оценки и проверок)."""
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value
