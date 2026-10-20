"""Гибридный поиск: вектор + полнотекст, объединение через RRF.

Зачем два поиска, а не один. Вектор понимает смысл и переводит между языками:
запрос «выделение рудных узлов» найдёт статью про ore cluster delineation. Но он
плохо держит точные строки — «Анабарский щит», «Куонамский», номер листа
Q-49-XXI. Полнотекстовый поиск наоборот: точные слова находит, смысл — нет.

RRF (Reciprocal Rank Fusion) складывает не оценки, а места в списках:

    score(d) = Σ 1 / (k + rank(d))

Оценки двух поисков несопоставимы (косинус против ts_rank), а места —
сопоставимы. k = 60 — значение из исходной статьи Cormack et al.; оно гасит
вклад хвоста, но не даёт первому месту одного списка задавить всё остальное.
"""

from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

from ..machine import integer
from . import textlang

RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    ord: int
    text: str
    headings: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    title: str = ""
    year: int | None = None
    journal: str | None = None
    url: str = ""
    authors: list[str] = field(default_factory=list)
    score: float = 0.0
    found_by: str = ""  # вектор | текст | оба
    vec_rank: int | None = None
    fts_rank: int | None = None
    # Нашёлся ли по словам «строго» — все слова запроса есть в тексте. Мягкий
    # поиск (хотя бы часть слов) сам по себе ничего не доказывает.
    fts_strict: bool = False
    # Косинусная близость к запросу, от 0 до 1. У BGE-M3 текст про то же самое
    # даёт примерно 0.6 и выше, случайный сосед — около 0.3-0.45. По ней и
    # отсекается «ближайшее, но не по делу».
    similarity: float | None = None


SELECT_FIELDS = """
    c.id, c.doc_id, c.ord, c.text, c.headings, c.pages,
    d.title, d.year, d.journal, d.url, d.authors
"""


def _filters(
    year_from: int | None, source: str | None, kb: str = "", docs: list[str] | None = None
) -> tuple[str, dict[str, Any]]:
    """Условия на статью: год, источник, узел дерева баз (kb — его поддерево) и, для
    поиска сверху вниз (summaries.top_down), только эти статьи (docs)."""
    from .db import in_kb

    where: list[str] = []
    params: dict[str, Any] = {}
    if docs is not None:
        where.append("c.doc_id = ANY(%(docs)s)")
        params["docs"] = list(docs)
    if year_from:
        where.append("d.year >= %(year_from)s")
        params["year_from"] = year_from
    if source:
        where.append("d.source = %(source)s")
        params["source"] = source
    branch, branch_params = in_kb(kb)
    params.update(branch_params)
    return (" AND " + " AND ".join(where) if where else "") + branch, params


# Сколько соседей HNSW смотрит за один поиск. По умолчанию в pgvector 40 — и
# больше 40 строк он не вернёт, сколько ни проси в LIMIT: веб-поиск просил 50
# и молча получал 40. Берём с запасом от запрошенного.
EF_SEARCH_MIN = 100


def _tune_hnsw(cur: Any, limit: int, filtered: bool) -> None:
    """Настройки векторного индекса на время одного поиска (до конца транзакции)."""
    cur.execute(f"SET LOCAL hnsw.ef_search = {min(max(EF_SEARCH_MIN, 2 * int(limit)), 1000)}")
    if not filtered:
        return
    # С фильтром по году или источнику HNSW сначала берёт ближайших, потом
    # отсеивает — и может вернуть почти пусто. pgvector с 0.8 умеет добирать
    # (iterative scan). Старая версия этой настройки не знает — тогда без неё.
    cur.execute("SAVEPOINT kbrag_hnsw")
    try:
        cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
    except Exception:  # noqa: BLE001 — pgvector старше 0.8
        cur.execute("ROLLBACK TO SAVEPOINT kbrag_hnsw")
    else:
        cur.execute("RELEASE SAVEPOINT kbrag_hnsw")


def _transaction(conn: Any) -> Any:
    """SET LOCAL действует только внутри транзакции. У соединения чат-бота
    autocommit (оно живёт, пока модель пишет ответ, и не должно держать
    транзакцию открытой минутами) — поэтому векторный поиск идёт в своей."""
    begin = getattr(conn, "transaction", None)
    return begin() if callable(begin) else nullcontext()


# Маленькая ветка (или узкий доступ, год, источник) — перебором, без HNSW. Индекс один
# на все фрагменты: с фильтром он берёт ближайших по всей базе и отсеивает чужих, и чем
# меньше доля ветки, тем дольше добирает — ветка в 32 статьи из 120 искалась 220 мс
# против 28 без фильтра, а на полной базе доля в десятки раз меньше. Перебор 20 тыс.
# фрагментов — около 100 мс, и он точный. Порог — KBRAG_EXACT_MAX.
EXACT_MAX = integer("KBRAG_EXACT_MAX", 50_000, low=0)


def _small(cur: Any, clause: str, params: dict[str, Any]) -> bool:
    """В отобранном не больше EXACT_MAX фрагментов. Считает не дальше порога."""
    if not clause or not EXACT_MAX:
        return False
    cur.execute(
        f"""
        SELECT count(*) FROM (
            SELECT 1 FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.embedding IS NOT NULL {clause}
            LIMIT %(exact_max)s
        ) t
        """,
        {**params, "exact_max": EXACT_MAX + 1},
    )
    row = cur.fetchone()
    return row is not None and int(row[0]) <= EXACT_MAX


def vector_search(
    conn: Any,
    vector: list[float],
    limit: int,
    year_from: Any = None,
    source: Any = None,
    kb: str = "",
    docs: list[str] | None = None,
) -> list[tuple[Any, ...]]:
    from .db import vector_literal

    clause, params = _filters(year_from, source, kb, docs)
    params.update({"q": vector_literal(vector), "limit": limit})
    with _transaction(conn), conn.cursor() as cur:
        exact = _small(cur, clause, params)
        if not exact:
            _tune_hnsw(cur, limit, bool(clause))
        # «+ 0» — порядок не по выражению индекса: Postgres его не берёт и перебирает
        # отобранные фрагменты (по индексу chunks.doc_id).
        order = "(c.embedding <=> %(q)s::vector) + 0" if exact else "c.embedding <=> %(q)s::vector"
        cur.execute(
            f"""
            SELECT {SELECT_FIELDS}, 1 - (c.embedding <=> %(q)s::vector) AS score
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.embedding IS NOT NULL {clause}
            ORDER BY {order}
            LIMIT %(limit)s
            """,
            params,
        )
        rows: list[tuple[Any, ...]] = cur.fetchall()
        return rows


# Словари запроса — по языкам фрагментов базы (textlang.py): у каждого фрагмента
# tsv построен словарём его языка. Русская конфигурация снимает окончания у
# кириллицы, а латиницу отдаёт английскому стеммеру — но беглую гласную она не
# знает: «рудный узел» даёт основу «узел», а «рудного узла» — «узл», и одно по
# другому не находилось. Поэтому запрос собирается здесь: у слова с беглой
# гласной — оба варианта через «или». Кавычки и минус — синтаксис строки поиска
# («точная фраза», -исключить): такой запрос отдаётся Postgres как есть
# (websearch_to_tsquery).

_WORD = re.compile(r"[\w\-]{3,}", re.UNICODE)
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
# Только окончания, где беглая гласная и бывает: узел, угол, песок, конец. Слова
# на -ом, -ем, -ов — это падежи, их не трогаем («золотом» — не «золотм»).
_FLEETING = re.compile(r"^(.*[бвгджзклмнпрстфхцчшщ])[еоё]([лкцнр])$")


# Существительное на -ом, -ов, -ем, -ев: стеммер Postgres срезает это как окончание.
# «разлом» → 'разл', а «разлома», «разломы» → 'разлом' — и запрос «Персияновский
# разлом» не находил ни одного фрагмента со словом «Персияновского разлома». Вариант
# с «-а» стеммится в ту же основу, что и косвенные падежи.
_NOUN_TAIL = re.compile(r"^[а-яё]{3,}(?:ом|ов|ем|ев)$")


# Обратно — из косвенной формы к начальной: «рудного узла» → «узел». Стеммер даёт
# «узла» основу «узл», а «узел» — «узел», и запрос в косвенной форме не находил текст
# в начальной. С окончания существительного снимается падеж; основа на согласную с
# л, к, ц, н, р — вариант со вставленной «е» и «о» (узел, песок, конец, угол).
_CASE_ENDINGS = (
    "ами",
    "ями",
    "ах",
    "ях",
    "ам",
    "ям",
    "ом",
    "ем",
    "ов",
    "ей",
    "а",
    "у",
    "ы",
    "е",
    "и",
    "ю",
)
_CLUSTER = re.compile(r"^(.+[бвгджзклмнпрстфхцчшщ])([лкцнр])$")


def _with_vowel(word: str) -> list[str]:
    """«узла» → [«узел», «узол»]: начальная форма слова с беглой гласной."""
    for ending in _CASE_ENDINGS:
        stem = word[: -len(ending)]
        if word.endswith(ending) and len(stem) >= 3:
            match = _CLUSTER.match(stem)
            if match:
                return [f"{match.group(1)}{vowel}{match.group(2)}" for vowel in "ео"]
            return []
    return []


def _variants(word: str) -> str:
    low = word.lower()
    options = [low]
    match = _FLEETING.match(low)
    if match:
        options.append(f"{match.group(1)}{match.group(2)}")
    options += [v for v in _with_vowel(low) if v not in options]
    if _NOUN_TAIL.match(low):
        options.append(low + "а")
    return f"({' | '.join(options)})" if len(options) > 1 else low


def build_tsquery(query: str, any_word: bool = False) -> str:
    """Текст запроса → выражение для to_tsquery: все слова (или хотя бы одно)."""
    words = [w for w in _TOKEN.findall(query) if w.lower() not in {"or", "and", "not"}]
    return (" | " if any_word else " & ").join(_variants(w) for w in words)


def _is_web_syntax(query: str) -> bool:
    return '"' in query or bool(re.search(r"(^|\s)-\w", query))


def text_search(
    conn: Any,
    query: str,
    limit: int,
    year_from: Any = None,
    source: Any = None,
    any_word: bool = False,
    alternatives: list[str] | None = None,
    kb: str = "",
    docs: list[str] | None = None,
) -> list[tuple[Any, ...]]:
    """alternatives — тот же запрос с другими написаниями из словаря синонимов:
    находится фрагмент, где есть все слова хотя бы одного из вариантов."""
    clause, params = _filters(year_from, source, kb, docs)
    configs = textlang.query_configs(conn, kb)
    if _is_web_syntax(query) and not any_word:
        tsq, q = textlang.tsquery_sql(configs, web=True), query
    else:
        parts = [
            p
            for p in (build_tsquery(x, any_word) for x in [query, *(alternatives or [])])
            if p.strip()
        ]
        tsq = textlang.tsquery_sql(configs)
        q = " | ".join(f"({p})" for p in parts) if len(parts) > 1 else (parts[0] if parts else "")
    if not q.strip():
        return []
    params.update({"q": q, "limit": limit})
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {SELECT_FIELDS},
                   ts_rank_cd(c.tsv, {tsq}) AS score
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.tsv @@ {tsq} {clause}
            ORDER BY score DESC
            LIMIT %(limit)s
            """,
            params,
        )
        found: list[tuple[Any, ...]] = cur.fetchall()
        return found


def _to_hit(row: tuple[Any, ...]) -> Hit:
    return Hit(
        chunk_id=row[0],
        doc_id=row[1],
        ord=row[2],
        text=row[3],
        headings=list(row[4] or []),
        pages=list(row[5] or []),
        title=row[6] or "",
        year=row[7],
        journal=row[8],
        url=row[9] or "",
        authors=list(row[10] or []),
    )


def rrf_merge(ranked_lists: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Места в списках → общая оценка. Вход: списки id в порядке релевантности."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for position, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)
    return scores


def hybrid_search(
    conn: Any,
    embedder: Any,
    query: str,
    limit: int = 10,
    candidates: int = 50,
    rrf_k: int = RRF_K,
    year_from: int | None = None,
    source: str | None = None,
    max_per_doc: int = 2,
    min_similarity: float = 0.0,
    require_words: bool = False,
    alternatives: list[str] | None = None,
    vector: list[float] | None = None,
    kb: str = "",
    docs: list[str] | None = None,
) -> list[Hit]:
    """alternatives — написания запроса из словаря синонимов (synonyms.synonym_variants):
    идут только в полнотекст, вектор и так понимает перевод. vector — уже посчитанный
    вектор запроса (search_many считает их пачкой). kb — узел дерева баз: только
    статьи его поддерева; пустой — вся база."""
    if vector is None:
        vector = embedder.encode_one(query)
    vec_rows = vector_search(conn, vector, candidates, year_from, source, kb, docs)
    fts_rows = text_search(
        conn, query, candidates, year_from, source, alternatives=alternatives, kb=kb, docs=docs
    )
    strict = True
    # Китайский, японский, корейский: слова не разделены — по парам знаков (textlang).
    cjk_rows: list[tuple[Any, ...]] = []
    if textlang.has_cjk(query):
        clause, params = _filters(year_from, source, kb, docs)
        cjk_rows = textlang.cjk_search(conn, query, candidates, clause, params, SELECT_FIELDS)
    # Полнотекст требует все слова сразу, и длинный вопрос словами целиком
    # почти никогда не находится. Тогда — мягкий заход: хотя бы часть слов.
    # Такое совпадение само по себе не доказательство, его проверяет порог близости.
    if not fts_rows and len(_WORD.findall(query)) >= 3:
        fts_rows = text_search(
            conn,
            query,
            candidates,
            year_from,
            source,
            any_word=True,
            alternatives=alternatives,
            kb=kb,
            docs=docs,
        )
        strict = False

    hits: dict[int, Hit] = {}
    for row in vec_rows + fts_rows + cjk_rows:
        hits.setdefault(row[0], _to_hit(row))
    for row in vec_rows:
        hits[row[0]].similarity = round(float(row[11]), 4)

    vec_ids = [row[0] for row in vec_rows]
    fts_ids = [row[0] for row in fts_rows]
    cjk_ids = [row[0] for row in cjk_rows]
    scores = rrf_merge([vec_ids, fts_ids, cjk_ids], k=rrf_k)

    for position, chunk_id in enumerate(vec_ids, start=1):
        hits[chunk_id].vec_rank = position
    for position, chunk_id in enumerate(fts_ids, start=1):
        hits[chunk_id].fts_rank = position
        hits[chunk_id].fts_strict = strict
    for position, row in enumerate(cjk_rows, start=1):
        # Все пары знаков запроса нашлись — доказательство, как все слова в полнотексте.
        hit = hits[row[0]]
        if hit.fts_rank is None:
            hit.fts_rank = position
            hit.fts_strict = float(row[11]) >= 1.0

    for chunk_id, score in scores.items():
        hit = hits[chunk_id]
        hit.score = score
        hit.found_by = (
            "оба" if hit.vec_rank and hit.fts_rank else ("вектор" if hit.vec_rank else "текст")
        )

    kept = [h for h in hits.values() if _passes(h, min_similarity, require_words)]
    ordered = sorted(kept, key=lambda h: h.score, reverse=True)
    return cap_per_doc(ordered, max_per_doc)[:limit]


def hit_dict(hit: Hit) -> dict[str, Any]:
    """Найденный фрагмент для страницы, --json и MCP: где он (doc_id, ord, chunk_id —
    по нему get_fragment), откуда, текст и чем найден."""
    return {
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "ord": hit.ord,
        "title": hit.title,
        "year": hit.year,
        "journal": hit.journal,
        "url": hit.url,
        "authors": hit.authors[:3],
        "headings": hit.headings[-2:],
        "pages": hit.pages,
        "text": " ".join(hit.text.split()),
        "found_by": hit.found_by,
        "score": round(hit.score, 4),
        "similarity": hit.similarity,
    }


def _passes(hit: Hit, min_similarity: float, require_words: bool) -> bool:
    """Достаточно ли доказательств, что кусок относится к запросу.

    Совпадение по словам — само по себе доказательство: полнотекстовый поиск
    Postgres требует, чтобы в тексте нашлись все слова запроса. Поэтому такой
    кусок проходит порог близости без разговоров. А найденный только вектором
    обязан быть действительно близким, иначе это просто наименее непохожее.
    """
    if require_words:
        return hit.fts_rank is not None and hit.fts_strict
    if hit.fts_rank is not None and hit.fts_strict:
        return True
    return (hit.similarity or 0.0) >= min_similarity


def cap_per_doc(hits: list[Hit], max_per_doc: int) -> list[Hit]:
    """Не больше N кусков одной статьи в выдаче.

    Без этого одна подробная статья занимает всю первую страницу: её куски
    похожи друг на друга и на запрос одинаково. Человеку нужен обзор того,
    что вообще есть в базе, а не десять абзацев одного текста.
    """
    if max_per_doc <= 0:
        return hits
    seen: dict[str, int] = {}
    kept, extra = [], []
    for hit in hits:
        count = seen.get(hit.doc_id, 0)
        if count < max_per_doc:
            kept.append(hit)
            seen[hit.doc_id] = count + 1
        else:
            extra.append(hit)
    # Лишние куски не выбрасываем совсем: если статей в базе мало, пусть
    # выдача добирается ими, но уже после всех остальных документов.
    return kept + extra
