"""Формулы в базе: таблица формул, ссылки на них из текста, поиск по описанию.

Откуда: разметка docling статьи (library/objects.py, formulas) — LaTeX уже чистый и
с номером «(N)». При загрузке статьи (ingest):

1. формулы во фрагментах помечаются $$…$$ — страница рисует их (KaTeX), модель ответа
   видит, где формула, и цитирует её дословно с номером;
2. в текст для вектора фрагмента вместо голого LaTeX идёт формула словами — знаки,
   фраза перед ней, «где ρ — …» (objects.describe): запрос по смыслу её находит;
3. в таблице formulas — каждая формула: номер, LaTeX, текст из PDF, страница, рамка на
   странице, абзацы до и после, фрагмент, где она стоит, описание и его вектор;
4. в formula_refs — какие фрагменты на неё ссылаются: «из формулы (55)», «уравнение 3»,
   «(2)» (если формула с таким номером в статье есть).

Команды: python kbrag.py formula <doc_id> [номер], python kbrag.py formula --find
«описание»; адреса /api/formula и /api/formulas, инструменты MCP get_formula и
search_formulas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..library import objects
from ..paths import ARTICLES
from . import db, objtable


def prepare(doc_id: str, articles: Path = ARTICLES) -> list[dict[str, Any]]:
    """Формулы статьи для загрузки — с описанием словами (objects.describe)."""
    items = objects.formulas(doc_id, articles)
    for item in items:
        item["description"] = objects.describe(item)
    return items


def mark_chunks(chunks: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[int, list[int]]:
    """Формулы во фрагментах: в тексте — $$…$$, в тексте для вектора — описание словами.
    Возвращает: номер фрагмента (ord) → index формул, которые в нём стоят."""
    where: dict[int, list[int]] = {}
    if not items:
        return where
    for chunk in chunks:
        text, found = objects.mark_formulas(str(chunk.get("text") or ""), items)
        if not found:
            continue
        chunk["text"] = text
        embed = str(chunk.get("embed_text") or chunk.get("text") or "")
        chunk["embed_text"], _ = objects.mark_formulas(
            embed, items, replace=lambda item: f"[{item['description']}]"
        )
        where[int(chunk.get("index", 0))] = found
    return where


def store(
    conn: Any,
    doc_id: str,
    items: list[dict[str, Any]],
    where: dict[int, list[int]],
    vectors: list[list[float]],
) -> int:
    """Формулы статьи — заново (прежние уходят вместе со ссылками), и ссылки на них из
    её фрагментов. Сколько формул записано."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM formulas WHERE doc_id = %s", (doc_id,))
        if not items:
            return 0
        cur.execute("SELECT ord, id, text FROM chunks WHERE doc_id = %s", (doc_id,))
        chunk_rows = cur.fetchall()
        chunk_by_ord = {int(r[0]): int(r[1]) for r in chunk_rows}
        chunk_of = {
            index: chunk_by_ord[ord_]
            for ord_, indexes in where.items()
            if ord_ in chunk_by_ord
            for index in indexes
        }
        ids: dict[str, int] = {}
        for item, vector in zip(items, vectors, strict=True):
            cur.execute(
                """
                INSERT INTO formulas (doc_id, idx, number, latex, pdf_text, page, bbox,
                                      before, after, description, chunk_id, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                RETURNING id
                """,
                (
                    doc_id,
                    item["index"],
                    item.get("number"),
                    item["latex"],
                    item.get("pdf_text") or "",
                    item.get("page"),
                    json.dumps(item["bbox"]) if item.get("bbox") else None,
                    item.get("before") or "",
                    item.get("after") or "",
                    item["description"],
                    chunk_of.get(int(item["index"])),
                    db.vector_literal(vector),
                ),
            )
            row = cur.fetchone()
            if row and item.get("number"):
                ids.setdefault(str(item["number"]), int(row[0]))
        numbers = set(ids)
        for _ord, chunk_id, text in chunk_rows:
            for number in objects.references(str(text or ""), numbers):
                cur.execute(
                    "INSERT INTO formula_refs (chunk_id, formula_id) VALUES (%s, %s) "
                    "ON CONFLICT DO NOTHING",
                    (chunk_id, ids[number]),
                )
    return len(items)


TABLE = objtable.Table(
    name="formulas",
    alias="f",
    columns=(
        "id doc_id idx number latex pdf_text page bbox before after description chunk_id"
    ).split(),
    keys=(
        "id doc_id index number latex pdf_text page bbox before after description chunk_id"
    ).split(),
    order="idx",
    refs="formula_refs",
    ref_column="formula_id",
)


def has_rows(conn: Any, doc_id: str) -> bool:
    return TABLE.has_rows(conn, doc_id)


def by_id(conn: Any, formula_id: int) -> dict[str, Any] | None:
    return TABLE.by_id(conn, formula_id)


def of_document(conn: Any, doc_id: str, number: str | None = None) -> list[dict[str, Any]]:
    """Формулы статьи по порядку (или одна — с номером «(number)»), с теми фрагментами,
    что на неё ссылаются."""
    if number is None:
        return TABLE.of_document(conn, doc_id)
    return TABLE.of_document(conn, doc_id, " AND f.number = %s", (str(number).strip().strip("()"),))


def in_chunks(conn: Any, chunk_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    """Формулы, стоящие в этих фрагментах: фрагмент → его формулы (для кнопок и показа)."""
    if not chunk_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {TABLE.fields} FROM {TABLE.source} "
            "WHERE f.chunk_id = ANY(%s) ORDER BY f.chunk_id, f.idx",
            (list(chunk_ids),),
        )
        out: dict[int, list[dict[str, Any]]] = {}
        for r in cur.fetchall():
            item = TABLE.row(r)
            out.setdefault(int(item["chunk_id"]), []).append(item)
    return out


def search(
    conn: Any, embedder: Any, query: str, kb: str = "", limit: int = 10
) -> list[dict[str, Any]]:
    """Формулы по смыслу — по описанию (формула словами с контекстом), в поддереве kb."""
    return TABLE.search(conn, embedder, query, kb, limit)


# --------------------------------------------------------------------------- #
#  Команда: python kbrag.py formula <doc_id> [номер] | formula --find «описание»
# --------------------------------------------------------------------------- #
def _show(item: dict[str, Any]) -> str:
    number = f" ({item['number']})" if item.get("number") else ""
    page = f", с. {item['page']}" if item.get("page") else ""
    lines = [
        f"Формула{number} — {item.get('title') or item.get('doc_id')}{page}",
        f"  LaTeX: {item.get('latex')}",
        f"  словами: {item.get('description')}",
    ]
    return "\n".join(lines + objtable.refs_line(item))


COMMAND = objtable.Command(
    prog="python kbrag.py formula",
    title="Формулы статей",
    number_help="номер формулы — как «(N)» в статье",
    find_help="найти формулы по описанию словами",
    json_key="formulas",
    not_found="Формул не нашлось.",
    lookup=of_document,
    search=search,
    show=_show,
)


def main(argv: list[str] | None = None) -> int:
    return objtable.run(COMMAND, argv)


if __name__ == "__main__":
    raise SystemExit(main())
