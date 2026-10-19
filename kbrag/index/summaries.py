"""Резюме статей и разделов (Д6) и поиск сверху вниз (Д7).

**Резюме.** При загрузке статьи (или потом: python kbrag.py summaries) модель пишет
резюме каждого раздела — 3–5 предложений о том, что в нём есть: объекты, методы,
выводы, числа, — и по ним резюме всей статьи. Раздел — верхний заголовок фрагментов
(headings[0]); фрагменты «Рисунки» и разделы короче SECTION_MIN знаков не резюмируются.
Каждое резюме — отдельная запись таблицы summaries со своим вектором (BGE-M3).
Отпечаток текста раздела (source_hash) — чтобы резюме писалось заново, только когда
текст правда изменился: пометка формул $$…$$ не в счёт.

**Сверху вниз.** Сначала статьи — по резюме статей (TOP_DOCS), в них — разделы по
резюме разделов (TOP_SECTIONS), и уже среди фрагментов этих статей — обычный поиск,
где фрагменты найденных разделов идут первыми (top_down). Это ещё одна выдача
кандидатов чат-бота рядом с обычным поиском по всей базе: в большой базе обычный
поиск тонет в похожих кусках чужих статей, а резюме держат в фокусе статьи, которые
про вопрос целиком.

**Обзорный вопрос** («обзор методов…», «какие работы есть о…», «что известно о…»)
отвечается по резюме статей (overview_sources): фрагмент на каждую статью вместо
восьми случайных кусков из двух-трёх статей.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..common import kb_from_args
from . import db

ARTICLE = "article"
SECTION = "section"
SECTION_MIN = 600  # знаков: раздел короче — без резюме (благодарности, сведения об авторах)
SECTION_CHARS = 7000  # знаков раздела модели: около 2 500 токенов
ARTICLE_CHARS = 9000  # знаков резюме разделов для резюме статьи
MAX_TOKENS = 400  # длина резюме — потолок ответа модели
WORKERS = 4  # разделов одной статьи у модели одновременно (vLLM считает их пачкой)
TOP_DOCS = 6
TOP_SECTIONS = 12
FIGURES_HEADING = "Рисунки"  # chunks/figures.HEADING — без импорта модели рисунков

SECTION_PROMPT = """Ты — помощник {пользователь}, специалист по {область}. Перед тобой \
раздел научной статьи «{title}», раздел «{section}». Напиши его резюме: 3–5 предложений \
— о чём раздел, какие {объекты}, методы, данные и выводы в нём, главные числа. Только \
то, что в тексте есть; язык резюме — {язык_ответа}. Верни JSON: {{"резюме": "..."}}"""

ARTICLE_PROMPT = """Ты — помощник {пользователь}, специалист по {область}. Перед тобой \
резюме разделов научной статьи «{title}». Напиши резюме всей статьи: 4–6 предложений — \
предмет, {объекты}, методы, главные результаты и выводы. Только то, что есть в резюме \
разделов; язык — {язык_ответа}. Верни JSON: {{"резюме": "..."}}"""


def source_hash(texts: list[str]) -> str:
    """Отпечаток текста раздела: пометка формул $$ и пробелы — не изменение."""
    joined = " ".join(" ".join(t.replace("$$", "").split()) for t in texts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def sections_of(chunks: list[tuple[int, str, list[str]]]) -> list[dict[str, Any]]:
    """Фрагменты статьи (ord, текст, заголовки) по разделам — по верхнему заголовку,
    в порядке статьи. Фрагменты «Рисунки» — не раздел статьи."""
    out: list[dict[str, Any]] = []
    for ord_, text, headings in sorted(chunks, key=lambda c: c[0]):
        name = (headings or [""])[0].strip()
        if name == FIGURES_HEADING:
            continue
        if not out or out[-1]["section"] != name:
            out.append({"section": name, "ords": [], "texts": []})
        out[-1]["ords"].append(int(ord_))
        out[-1]["texts"].append(text)
    # Один заголовок, встретившийся дважды (разорван таблицей), — один раздел.
    merged: dict[str, dict[str, Any]] = {}
    for part in out:
        known = merged.setdefault(
            part["section"], {"section": part["section"], "ords": [], "texts": []}
        )
        known["ords"] += part["ords"]
        known["texts"] += part["texts"]
    return [s for s in merged.values() if sum(len(t) for t in s["texts"]) >= SECTION_MIN]


def _clip(texts: list[str], limit: int) -> str:
    out, size = [], 0
    for text in texts:
        piece = " ".join(text.split())
        if size + len(piece) > limit:
            out.append(piece[: max(0, limit - size)].rsplit(" ", 1)[0] + " …")
            break
        out.append(piece)
        size += len(piece) + 1
    return "\n".join(out)


def _ask(client: Any, prompt: str, text: str) -> str:
    data = client.chat_json(prompt, text, temperature=0, max_tokens=MAX_TOKENS)
    value = data.get("резюме") or data.get("summary") or ""
    return " ".join(str(value).split())


def _fill(kb: Any, name: str, default: str, **values: str) -> str:
    if kb is None:
        from ..kb import get as get_kb

        kb = get_kb()
    text: str = kb.prompt(name, default, **values)
    return text


# --------------------------------------------------------------------------- #
#  Запись
# --------------------------------------------------------------------------- #
def _known(conn: Any, doc_id: str) -> dict[tuple[str, str], str]:
    with conn.cursor() as cur:
        cur.execute("SELECT kind, section, source_hash FROM summaries WHERE doc_id = %s", (doc_id,))
        return {(r[0], r[1]): r[2] for r in cur.fetchall()}


def _chunks(conn: Any, doc_id: str) -> tuple[str, list[tuple[int, str, list[str]]]]:
    with conn.cursor() as cur:
        cur.execute("SELECT coalesce(title, '') FROM documents WHERE doc_id = %s", (doc_id,))
        row = cur.fetchone()
        cur.execute("SELECT ord, text, headings FROM chunks WHERE doc_id = %s", (doc_id,))
        rows = [(int(r[0]), str(r[1] or ""), list(r[2] or [])) for r in cur.fetchall()]
    return (str(row[0]) if row else ""), rows


def _save(
    conn: Any,
    doc_id: str,
    kind: str,
    section: str,
    ords: list[int],
    text: str,
    digest: str,
    model: str,
    vector: list[float],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO summaries (doc_id, kind, section, ords, text, source_hash, model,
                                   embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (doc_id, kind, section) DO UPDATE SET
                ords = EXCLUDED.ords, text = EXCLUDED.text, source_hash = EXCLUDED.source_hash,
                model = EXCLUDED.model, embedding = EXCLUDED.embedding, created_at = now()
            """,
            (doc_id, kind, section, ords, text, digest, model, db.vector_literal(vector)),
        )


def summarize_document(
    conn: Any,
    embedder: Any,
    client: Any,
    doc_id: str,
    kb: Any = None,
    redo: bool = False,
) -> int:
    """Резюме разделов статьи и её самой — те, которых нет или чей текст изменился.
    Сколько резюме написано."""
    title, rows = _chunks(conn, doc_id)
    parts = sections_of(rows)
    if not parts:
        return 0
    known = _known(conn, doc_id)
    for part in parts:
        part["hash"] = source_hash(part["texts"])
    todo = [p for p in parts if redo or known.get((SECTION, p["section"])) != p["hash"]]

    def one(part: dict[str, Any]) -> str:
        prompt = _fill(
            kb, "резюме_раздела", SECTION_PROMPT, title=title, section=part["section"] or "—"
        )
        return _ask(client, prompt, _clip(part["texts"], SECTION_CHARS))

    written = 0
    if todo:
        with ThreadPoolExecutor(min(WORKERS, len(todo))) as pool:
            texts = list(pool.map(one, todo))
        done = [(p, t) for p, t in zip(todo, texts, strict=True) if t]
        vectors = embedder.encode([t for _, t in done]) if done else []
        for (part, text), vector in zip(done, vectors, strict=True):
            _save(
                conn,
                doc_id,
                SECTION,
                part["section"],
                part["ords"],
                text,
                part["hash"],
                client.model,
                vector,
            )
            written += 1
    # Резюме статьи — по резюме разделов; переписывается, если изменился хоть один.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT section, text, source_hash FROM summaries WHERE doc_id = %s AND kind = %s "
            "ORDER BY ords[1] NULLS LAST",
            (doc_id, SECTION),
        )
        sections = cur.fetchall()
        # Разделов, которых в статье больше нет (переразбор), — убрать.
        alive = {p["section"] for p in parts}
        for gone in [s[0] for s in sections if s[0] not in alive]:
            cur.execute(
                "DELETE FROM summaries WHERE doc_id = %s AND kind = %s AND section = %s",
                (doc_id, SECTION, gone),
            )
    sections = [s for s in sections if s[0] in alive]
    if not sections:
        return written
    digest = source_hash([s[2] for s in sections])
    if not redo and known.get((ARTICLE, "")) == digest:
        return written
    lines = [f"{s[0] or 'Раздел'}: {s[1]}" for s in sections]
    prompt = _fill(kb, "резюме_статьи", ARTICLE_PROMPT, title=title, section="")
    text = _ask(client, prompt, _clip(lines, ARTICLE_CHARS))
    if text:
        ords = sorted({o for p in parts for o in p["ords"]})
        _save(
            conn, doc_id, ARTICLE, "", ords, text, digest, client.model, embedder.encode([text])[0]
        )
        written += 1
    return written


def summarize(
    conn: Any,
    embedder: Any,
    client: Any,
    doc_ids: list[str],
    kb: Any = None,
    redo: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[int, list[str]]:
    """Резюме статей по очереди; одна не вышла — остальные идут. (написано, ошибки)."""
    total, errors = 0, []
    for i, doc_id in enumerate(doc_ids, start=1):
        started = time.monotonic()
        try:
            n = summarize_document(conn, embedder, client, doc_id, kb, redo)
            conn.commit()
        except Exception as exc:  # noqa: BLE001 — модель не ответила на статью
            conn.rollback()
            errors.append(f"{doc_id}: {type(exc).__name__}: {exc}")
            continue
        total += n
        if n:
            log(f"  [{i}/{len(doc_ids)}] {doc_id}: резюме {n} ({time.monotonic() - started:.0f} с)")
    return total, errors


def missing(conn: Any, kb: str = "") -> list[str]:
    """Статьи поддерева kb без резюме статьи."""
    clause, params = db.in_kb(kb, "d.doc_id")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT d.doc_id FROM documents d WHERE NOT EXISTS (SELECT 1 FROM summaries s "
            f"WHERE s.doc_id = d.doc_id AND s.kind = 'article'){clause} ORDER BY d.doc_id",
            params,
        )
        return [str(r[0]) for r in cur.fetchall()]


def after_ingest(
    conn: Any, embedder: Any, doc_ids: list[str], log: Callable[[str], None] = print
) -> int:
    """Резюме только что загруженных статей — если модель отвечает; не отвечает —
    подсказка, как дописать потом. Сколько резюме написано."""
    from ..chat.settings import Settings

    if not doc_ids:
        return 0
    client = Settings().client()
    if not client.status(timeout=3).get("ok"):
        log(
            "Резюме статей не написаны: модель не отвечает. Потом — python kbrag.py "
            "summaries (допишет недостающие)."
        )
        return 0
    log(f"Резюме: {len(doc_ids)} статей — пишет модель {client.model}")
    total, errors = summarize(conn, embedder, client, doc_ids, log=log)
    for error in errors:
        log(f"  ! резюме {error}")
    return total


# --------------------------------------------------------------------------- #
#  Поиск
# --------------------------------------------------------------------------- #
_CACHE: dict[str, tuple[float, bool]] = {}
CACHE_SEC = 300.0


def available(conn: Any, kb: str = "") -> bool:
    """Есть ли в поддереве kb резюме статей (иначе поиск сверху вниз не нужен)."""
    now = time.monotonic()
    cached = _CACHE.get(kb)
    if cached and now - cached[0] < CACHE_SEC:
        return cached[1]
    clause, params = db.in_kb(kb, "s.doc_id")
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM summaries s WHERE s.kind = 'article'{clause} LIMIT 1", params
            )
            found = cur.fetchone() is not None
    except Exception:  # noqa: BLE001 — таблицы нет (база до резюме)
        found = False
    _CACHE[kb] = (now, found)
    return found


def search_summaries(
    conn: Any,
    vector: list[float],
    kind: str,
    kb: str = "",
    limit: int = TOP_DOCS,
    docs: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Резюме, ближайшие к вектору запроса: kind — статьи или разделы; docs — только
    в этих статьях."""
    clause, params = db.in_kb(kb, "s.doc_id")
    if docs is not None:
        clause += " AND s.doc_id = ANY(%(docs)s)"
        params["docs"] = list(docs)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT s.doc_id, s.section, s.ords, s.text, d.title, d.year, d.journal, d.url,
                   d.authors, 1 - (s.embedding <=> %(q)s::vector) AS similarity
            FROM summaries s JOIN documents d ON d.doc_id = s.doc_id
            WHERE s.kind = %(kind)s AND s.embedding IS NOT NULL{clause}
            ORDER BY s.embedding <=> %(q)s::vector
            LIMIT %(limit)s
            """,
            {**params, "q": db.vector_literal(vector), "kind": kind, "limit": limit},
        )
        keys = "doc_id section ords text title year journal url authors similarity".split()
        return [dict(zip(keys, r, strict=True)) for r in cur.fetchall()]


def top_down(
    conn: Any,
    embedder: Any,
    query: str,
    kb: str = "",
    limit: int = 30,
    vector: list[float] | None = None,
) -> list[Any]:
    """Статьи по резюме → разделы по резюме → фрагменты: обычный поиск, но только в
    найденных статьях, и фрагменты найденных разделов — первыми. Резюме нет — []."""
    from .search import hybrid_search

    if not available(conn, kb):
        return []
    vector = vector if vector is not None else embedder.encode_one(query)
    articles = search_summaries(conn, vector, ARTICLE, kb, TOP_DOCS)
    if not articles:
        return []
    docs = [a["doc_id"] for a in articles]
    sections = search_summaries(conn, vector, SECTION, kb, TOP_SECTIONS, docs)
    focus = {(s["doc_id"], int(o)) for s in sections for o in (s["ords"] or [])}
    hits = hybrid_search(
        conn, embedder, query, limit=limit, max_per_doc=0, vector=vector, kb=kb, docs=docs
    )
    inside = [h for h in hits if (h.doc_id, h.ord) in focus]
    rest = [h for h in hits if (h.doc_id, h.ord) not in focus]
    return inside + rest


# Обзорный вопрос: обзор, сравнение подходов, какие работы, что известно в целом.
_OVERVIEW = re.compile(
    r"\bобзор|\bкакие\s+(?:есть\s+)?(?:работ|стат|исследован|публикац)|"
    r"\bчто\s+(?:известно|пишут|написано)\s+(?:в\s+(?:базе|статьях|литературе)|о\b|об\b|про\b)|"
    r"\bв\s+целом\b|\bсостояни\w*\s+(?:изученност|вопроса)|\breview\b|\boverview\b",
    re.IGNORECASE,
)
OVERVIEW_DOCS = 8


def is_overview(question: str) -> bool:
    return bool(_OVERVIEW.search(question or ""))


def overview_sources(
    conn: Any, embedder: Any, question: str, kb: str = "", start: int = 1
) -> list[dict[str, Any]]:
    """Обзорный вопрос — по резюме статей: источник на статью («Резюме статьи»)."""
    if not available(conn, kb):
        return []
    found = search_summaries(conn, embedder.encode_one(question), ARTICLE, kb, OVERVIEW_DOCS)
    return [
        {
            "n": start + i,
            "doc_id": s["doc_id"],
            "chunk_id": None,
            "ord": -1,
            "title": s["title"] or s["doc_id"],
            "year": s["year"],
            "journal": s["journal"],
            "authors": list(s["authors"] or [])[:3],
            "pages": [],
            "headings": ["Резюме статьи"],
            "url": s["url"] or "",
            "text": s["text"],
            "similarity": round(float(s["similarity"]), 4),
            "found_by": "резюме",
            "part": None,
            "kind": "резюме",
        }
        for i, s in enumerate(found)
    ]


# --------------------------------------------------------------------------- #
#  Команда: python kbrag.py summaries [doc_id] [--redo] [--show]
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from ..chat.settings import Settings
    from ..common import add_db_args
    from .embed import build_embedder

    p = argparse.ArgumentParser(prog="python kbrag.py summaries", description=__doc__)
    p.add_argument("doc", nargs="?", help="только эта статья")
    p.add_argument("--redo", action="store_true", help="написать заново и готовые")
    p.add_argument("--show", action="store_true", help="показать резюме статьи, не писать")
    p.add_argument("--json", action="store_true")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    add_db_args(p)
    args = p.parse_args(argv)
    base = kb_from_args(args.kb)
    if base is None:
        return 1
    with db.connect(args.dsn) as conn:
        db.init_db(conn, with_hnsw=False)
        if args.show:
            if not args.doc:
                print("Чью статью показать: summaries <doc_id> --show", file=sys.stderr)
                return 1
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT kind, section, text FROM summaries WHERE doc_id = %s "
                    "ORDER BY kind, ords[1]",
                    (args.doc,),
                )
                rows = [{"kind": r[0], "section": r[1], "text": r[2]} for r in cur.fetchall()]
            if args.json:
                print(json.dumps({"doc_id": args.doc, "summaries": rows}, ensure_ascii=False))
            for r in rows if not args.json else []:
                print(f"{'Статья' if r['kind'] == ARTICLE else r['section']}:\n  {r['text']}\n")
            return 0 if rows else 1
        ids = (
            [args.doc]
            if args.doc
            else (
                [d["doc_id"] for d in db.documents(conn, limit=1_000_000, kb=base.path)]
                if args.redo
                else missing(conn, base.path)
            )
        )
        if not ids:
            print("Резюме есть у всех статей.")
            return 0
        settings = Settings()
        client = settings.client()
        if not client.status(timeout=5).get("ok"):
            print(f"Модель на {client.host} не отвечает: python kbrag.py start", file=sys.stderr)
            return 1
        print(f"Резюме: {len(ids)} статей ({base.path or 'всё дерево'}), модель {client.model}")
        embedder = build_embedder(device=args.device)
        total, errors = summarize(conn, embedder, client, ids, kb=base, redo=args.redo)
    print(f"Написано резюме: {total}")
    for error in errors:
        print(f"  ! {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
