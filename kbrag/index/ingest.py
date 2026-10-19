"""Загрузка разобранных статей в базу: чанки → векторы → строки в Postgres.

Источник данных — папка статей (.local/acquired: туда кладёт import, там же лежит
добытое прежней добычей). Рядом лежат <doc_id>.json (метаданные и статус) и
<doc_id>.chunks.json (нарезка). Сам PDF для этого не нужен: всё, что надо для
индекса, снято при разборе.

Повторный запуск не плодит дубли: чанки документа переписываются целиком,
а документы, у которых число чанков не изменилось, пропускаются.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..lang import detect, majority
from . import db, figures, formulas, noise

# Список литературы — это чужие названия и ссылки, а не содержание статьи.
# В поиске такие чанки выглядят убедительно (там все нужные слова!), но читать
# в них нечего: на живом прогоне половину выдачи занимали именно библиографии.
REFS_HEADING = re.compile(
    r"^\s*(список\s+(литератур\w*|источник\w*)|литератур\w*|библиограф\w*|"
    r"references?|bibliography|works\s+cited)\s*:?\s*$",
    re.IGNORECASE,
)
# Даже без заголовка библиографию выдаёт плотность ссылок и DOI.
_CITATION_MARK = re.compile(r"https?://|doi:|doi\.org|//\s*[A-ZА-Я]|\bpp\.\s*\d", re.IGNORECASE)
MIN_MARKS_PER_KB = 4


def is_reference_chunk(chunk: dict[str, Any]) -> bool:
    """Похож ли чанк на список литературы, а не на текст статьи."""
    for heading in chunk.get("headings") or []:
        if REFS_HEADING.match(str(heading)):
            return True

    text = chunk.get("text") or ""
    if len(text) < 200:
        return False
    marks = len(_CITATION_MARK.findall(text))
    return marks / (len(text) / 1000) >= MIN_MARKS_PER_KB


def embed_text(meta: dict[str, Any], chunk: dict[str, Any]) -> str:
    """Что уходит в модель эмбеддинга: название статьи, разделы, текст.

    embed_text от нарезки уже несёт заголовки разделов — модель видит, из
    какой части статьи кусок. Название статьи добавляется сверху: иначе
    фрагмент статьи про Анабарский щит, где сам щит не назван, ничем не
    отличался от такого же фрагмента про любую другую территорию. Человеку
    показывается всё равно text.
    """
    body = chunk.get("embed_text") or chunk.get("text") or ""
    title = " ".join(str(meta.get("title") or "").split())
    if title and not body.startswith(title):
        return f"{title}\n{body}"
    return body


@dataclass
class IngestReport:
    seen: int = 0
    indexed: int = 0
    skipped: int = 0
    empty: int = 0
    chunks: int = 0
    refs_dropped: int = 0
    noise_dropped: int = 0  # служебные разделы и пустышки (noise.py)
    noise_trimmed: int = 0  # фрагменты, из которых вырезан служебный текст
    duplicates: int = 0
    forgotten: int = 0
    errors: list[str] = field(default_factory=list)
    indexed_ids: list[str] = field(default_factory=list)  # что записано в этот раз
    formulas: int = 0  # формул записано (index/formulas.py)
    figures: int = 0  # рисунков записано (index/figures.py)


def _document_row(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": meta.get("doc_id") or "",
        "title": _clean(meta.get("title")),
        "authors": [_clean(a) for a in meta.get("authors") or []],
        "year": meta.get("year"),
        "journal": meta.get("journal"),
        "doi": meta.get("doi"),
        "url": meta.get("url") or "",
        "source": meta.get("source") or "",
        "language": meta.get("language"),
        "status": meta.get("status") or "",
        "sha256": meta.get("sha256") or "",
        "pages": int(meta.get("pages") or 0),
        "parser": meta.get("parser") or "",
        "accuracy": float(meta.get("accuracy") or 0.0),
        "fetched_at": meta.get("fetched_at") or "",
    }


def _clean(text: Any) -> str:
    """Нулевой символ из PDF: Postgres не принимает его в тексте и роняет всю статью."""
    return str(text or "").replace("\x00", "")


def chunk_languages(chunks: list[dict[str, Any]], document: str = "") -> list[str]:
    """Язык каждого фрагмента (lang.py); не понять — язык статьи: из метаданных, иначе
    по большинству её фрагментов."""
    found = [detect(str(c.get("text") or "")) for c in chunks]
    fallback = document or majority(found)
    return [lang or fallback for lang in found]


def _chunk_rows(
    doc_id: str,
    chunks: list[dict[str, Any]],
    vectors: list[list[float]],
    languages: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    languages = languages or chunk_languages(chunks)
    for chunk, vector, language in zip(chunks, vectors, languages, strict=True):
        rows.append(
            {
                "doc_id": doc_id,
                "ord": int(chunk.get("index", 0)),
                "text": _clean(chunk.get("text")),
                "headings": [_clean(h) for h in chunk.get("headings") or []],
                "pages": [int(p) for p in (chunk.get("pages") or [])],
                "n_tokens": int(chunk.get("n_tokens") or 0),
                "has_table": bool(chunk.get("has_table")),
                "embedding": db.vector_literal(vector),
                "language": language or None,
            }
        )
    return rows


def ingest_dir(
    conn: Any,
    embedder: Any,
    acquired_dir: Path,
    force: bool = False,
    verbose: bool = True,
    only: set[str] | None = None,
) -> IngestReport:
    """only — загрузить только эти статьи (import, reparse), остальные не трогать."""
    report = IngestReport()
    # Статьи, которые прежняя команда clean убрала в _отсев, база тоже забывает: иначе
    # они остаются в поиске и в чат-боте, хотя в папке их уже нет.
    moved = sorted(
        p.name[: -len(".json")]
        for p in (acquired_dir / "_отсев").glob("*.json")
        if not p.name.endswith((".chunks.json", ".docling.json"))
    )
    report.forgotten = db.delete_documents(conn, moved) if moved else 0
    conn.commit()

    known = {} if force else db.indexed_docs(conn)
    hashes = db.document_hashes(conn)

    for chunks_path in sorted(acquired_dir.glob("*.chunks.json")):
        doc_id = chunks_path.name[: -len(".chunks.json")]
        if only is not None and doc_id not in only:
            continue
        meta_path = acquired_dir / f"{doc_id}.json"
        report.seen += 1

        try:
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except (OSError, json.JSONDecodeError) as exc:
            report.errors.append(f"{doc_id}: {exc}")
            continue

        if not chunks:
            report.empty += 1
            continue

        # Тот же файл уже лежит под другим идентификатором — второй раз не кладём:
        # иначе поиск и чат-бот показывают одну статью дважды.
        sha = meta.get("sha256") or ""
        if sha and hashes.get(sha, doc_id) != doc_id:
            report.duplicates += 1
            if verbose:
                print(f"  = {(meta.get('title') or doc_id)[:60]} — уже есть как {hashes[sha]}")
            continue
        if sha:
            hashes[sha] = doc_id

        # Библиографию в базу не кладём: места занимает много, читать нечего.
        kept = [c for c in chunks if not is_reference_chunk(c)]
        report.refs_dropped += len(chunks) - len(kept)
        # Служебный текст журнала (благодарности, колонтитулы, e-mail) — тоже.
        cleaned = noise.clean_document(kept, str(meta.get("title") or ""))
        report.noise_dropped += (
            cleaned.dropped_sections + cleaned.dropped_empty + cleaned.dropped_authors
        )
        report.noise_trimmed += cleaned.trimmed
        kept = cleaned.chunks
        if not kept:
            report.empty += 1
            continue
        # Формулы статьи (index/formulas.py): статья с формулами, которых в базе ещё нет
        # (загружена до таблицы формул), загружается заново — с пометкой и описаниями.
        found = formulas.prepare(doc_id, acquired_dir)
        pictures = figures.prepare(meta)
        fresh = (bool(found) and not formulas.has_rows(conn, doc_id)) or (
            bool(pictures) and not figures.has_rows(conn, doc_id)
        )
        if not force and not fresh and known.get(doc_id) == len(kept):
            report.skipped += 1
            continue
        chunks = kept
        where = formulas.mark_chunks(chunks, found)

        meta.setdefault("doc_id", doc_id)
        texts = [embed_text(meta, c) for c in chunks]
        languages = chunk_languages(chunks, str(meta.get("language") or ""))
        meta["language"] = meta.get("language") or majority(languages) or None

        try:
            vectors = embedder.encode(texts, progress=verbose)
            db.upsert_document(conn, _document_row(meta))
            rows = _chunk_rows(doc_id, chunks, vectors, languages)
            written = db.replace_chunks(conn, doc_id, rows)
            described = embedder.encode([f["description"] for f in found]) if found else []
            report.formulas += formulas.store(conn, doc_id, found, where, described)
            seen = embedder.encode([f["search_text"] for f in pictures]) if pictures else []
            report.figures += figures.store(conn, doc_id, pictures, seen)
            conn.commit()
        except Exception as exc:  # noqa: BLE001 — одна статья не должна ронять загрузку
            conn.rollback()
            report.errors.append(f"{doc_id}: {type(exc).__name__}: {exc}")
            continue

        report.indexed += 1
        report.indexed_ids.append(doc_id)
        report.chunks += written
        if verbose:
            title = (meta.get("title") or doc_id)[:60]
            print(f"  + {title} — {written} чанков")

    # Фрагменты, загруженные до того, как у них появился язык, — дописать.
    db.fill_languages(conn)
    conn.commit()
    return report


@dataclass
class TidyReport:
    documents: int = 0
    dropped: int = 0  # фрагментов убрано целиком
    trimmed: int = 0  # фрагментов подчищено
    facts_dropped: int = 0  # фактов, чья цитата ушла вместе со служебным текстом


def tidy_indexed(conn: Any, embedder: Any, verbose: bool = True) -> TidyReport:
    """Убрать служебный текст из уже загруженной базы — без папки статей.

    Нужна для статей, загруженных до noise.py: чистка та же, что при загрузке.
    Подчищенному фрагменту пересчитывается вектор. Факты графа у него остаются,
    если их цитата по-прежнему есть в тексте; убранный фрагмент уносит свои
    факты каскадом."""
    from ..text import locate_quote

    report = TidyReport()
    for doc_id, title in db.documents_with_titles(conn):
        rows = db.chunk_rows(conn, doc_id)
        chunks = [
            {"index": r["ord"], "text": r["text"], "headings": r["headings"], "id": r["id"]}
            for r in rows
        ]
        cleaned = noise.clean_document(chunks, title)
        before = {r["id"]: r["text"] for r in rows}
        kept = {c["id"] for c in cleaned.chunks}
        drop = [c["id"] for c in chunks if c["id"] not in kept]
        changed = [c for c in cleaned.chunks if c["text"] != before[c["id"]]]
        if not drop and not changed:
            continue
        vectors = (
            embedder.encode([embed_text({"title": title}, c) for c in changed], progress=False)
            if changed
            else []
        )
        updates = [
            (c["id"], c["text"], db.vector_literal(v))
            for c, v in zip(changed, vectors, strict=True)
        ]
        stale = [
            fact_id
            for c in changed
            for fact_id, quote in db.facts_of_chunk(conn, c["id"])
            if locate_quote(quote, c["text"]) is None
        ]
        db.tidy_chunks(conn, drop, updates, stale)
        conn.commit()
        report.documents += 1
        report.dropped += len(drop)
        report.trimmed += len(changed)
        report.facts_dropped += len(stale)
        if verbose:
            print(f"  ~ {title[:60] or doc_id} — убрано {len(drop)}, подчищено {len(changed)}")
    return report
