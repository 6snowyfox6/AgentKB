"""Рисунки в базе: таблица рисунков, ссылки «рис. N» на них из текста, поиск по подписи
и описанию.

Откуда: перечень рисунков статьи (поле figures её записи; chunks/figures.py,
save_images) — файл картинки, страница, рамка, подпись, номер в статье («Рис. 3» →
label «3») и, если рисунок описан моделью со зрением, описание. При загрузке статьи
(ingest):

1. в таблице figures — каждый рисунок, с фрагментом, где он стоит (фрагмент
   «Рисунок» с описанием, а нет описания — фрагмент с подписью), и вектором подписи
   с описанием: рисунок находится по смыслу;
2. в figure_refs — какие фрагменты на него ссылаются: «рис. 3», «на рисунке 2а»,
   «Fig. 4», «figs. 2 and 3», а также фрагмент с подписью рисунка.

По фрагментам ответа рисунки находятся (in_chunks) — их показывает страница под
ответом и Телеграм кнопками «Рис. N». Команды: python kbrag.py figures (описать
рисунки моделью со зрением), адреса /api/figure и /api/figures, инструменты MCP
get_figure и search_figures.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..chunks.figures import HEADING, MARK, label_of
from . import db, objtable

_NUMBER = r"\d{1,3}[а-яa-z]?"
# «рис. 3», «рисунке 2а», «рис. 2, 3 и 5», «Fig. 4», «figs. 2 and 3», «Figure 1b».
_REF = re.compile(
    rf"(?<![\w])(?:рис(?:ун\w*)?\.?|figs?\.|figures?)\s*({_NUMBER}"
    rf"(?:\s*(?:,|и|and|&)\s*{_NUMBER})*)(?![\w])",
    re.IGNORECASE,
)


def references(text: str, labels: set[str]) -> set[str]:
    """Номера рисунков статьи, на которые ссылается текст."""
    found: set[str] = set()
    for match in _REF.finditer(text or ""):
        found |= {n.lower() for n in re.findall(_NUMBER, match.group(1))}
    return found & labels


def prepare(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Рисунки статьи из её записи — с номером в статье и текстом для вектора."""
    items = []
    for raw in meta.get("figures") or []:
        if not isinstance(raw, dict) or not raw.get("n"):
            continue
        item = dict(raw)
        caption = " ".join(str(item.get("caption") or "").split())
        item["caption"] = caption
        item["label"] = item.get("label") or label_of(caption)
        item["description"] = str(item.get("description") or "")
        name = f"Рисунок {item['label']}" if item["label"] else "Рисунок"
        item["search_text"] = ". ".join(p for p in (name, caption, item["description"]) if p)
        items.append(item)
    return items


def _is_own_description(text: str, n: int) -> bool:
    return MARK in text and re.match(rf"Рисунок {n}\b", text) is not None


def store(conn: Any, doc_id: str, items: list[dict[str, Any]], vectors: list[list[float]]) -> int:
    """Рисунки статьи — заново (прежние уходят вместе со ссылками), и ссылки на них из
    её фрагментов. Сколько рисунков записано."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM figures WHERE doc_id = %s", (doc_id,))
        if not items:
            return 0
        cur.execute("SELECT id, text, headings FROM chunks WHERE doc_id = %s", (doc_id,))
        chunk_rows = [(int(r[0]), str(r[1] or ""), list(r[2] or [])) for r in cur.fetchall()]
        text_rows = [r for r in chunk_rows if (r[2] or [None])[0] != HEADING]
        ids: dict[str, int] = {}
        for item, vector in zip(items, vectors, strict=True):
            n = int(item["n"])
            caption = item["caption"]
            described = next((r[0] for r in chunk_rows if _is_own_description(r[1], n)), None)
            captioned = (
                next((r[0] for r in text_rows if caption[:80] in " ".join(r[1].split())), None)
                if len(caption) >= 12
                else None
            )
            cur.execute(
                """
                INSERT INTO figures (doc_id, n, label, file, page, bbox, caption, description,
                                     width, height, chunk_id, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                RETURNING id
                """,
                (
                    doc_id,
                    n,
                    item.get("label"),
                    item.get("file") or "",
                    item.get("page"),
                    json.dumps(item["bbox"]) if item.get("bbox") else None,
                    caption,
                    item["description"],
                    item.get("width"),
                    item.get("height"),
                    described or captioned,
                    db.vector_literal(vector),
                ),
            )
            row = cur.fetchone()
            if row is None:
                continue
            figure_id = int(row[0])
            if item.get("label"):
                ids.setdefault(str(item["label"]), figure_id)
            if captioned is not None:
                _ref(cur, captioned, figure_id)
        labels = set(ids)
        for chunk_id, text, _headings in text_rows:
            for label in references(text, labels):
                _ref(cur, chunk_id, ids[label])
    return len(items)


def _ref(cur: Any, chunk_id: int, figure_id: int) -> None:
    cur.execute(
        "INSERT INTO figure_refs (chunk_id, figure_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (chunk_id, figure_id),
    )


TABLE = objtable.Table(
    name="figures",
    alias="g",
    columns=("id doc_id n label file page bbox caption description width height chunk_id").split(),
    keys=("id doc_id n label file page bbox caption description width height chunk_id").split(),
    order="n",
    refs="figure_refs",
    ref_column="figure_id",
)


def has_rows(conn: Any, doc_id: str) -> bool:
    return TABLE.has_rows(conn, doc_id)


def by_id(conn: Any, figure_id: int) -> dict[str, Any] | None:
    return TABLE.by_id(conn, figure_id)


def of_document(conn: Any, doc_id: str, label: str | None = None) -> list[dict[str, Any]]:
    """Рисунки статьи по порядку (или один — «Рис. label»), с фрагментами, что на него
    ссылаются."""
    if label is None:
        return TABLE.of_document(conn, doc_id)
    return TABLE.of_document(
        conn, doc_id, " AND lower(g.label) = %s", (str(label).strip().lower(),)
    )


def in_chunks(conn: Any, chunk_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    """Рисунки фрагментов: тот, что во фрагменте стоит (описание, подпись), и те, на
    которые фрагмент ссылается. Фрагмент → его рисунки по порядку."""
    if not chunk_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT {TABLE.fields}, links.c
            FROM (
                SELECT chunk_id AS c, id AS fid FROM figures WHERE chunk_id = ANY(%(ids)s)
                UNION
                SELECT chunk_id AS c, figure_id AS fid FROM figure_refs
                WHERE chunk_id = ANY(%(ids)s)
            ) links
            JOIN figures g ON g.id = links.fid
            JOIN documents d ON d.doc_id = g.doc_id
            ORDER BY c, g.n, g.id
            """,
            {"ids": list(chunk_ids)},
        )
        out: dict[int, list[dict[str, Any]]] = {}
        for r in cur.fetchall():
            out.setdefault(int(r[-1]), []).append(TABLE.row(r))
    return out


def search(
    conn: Any, embedder: Any, query: str, kb: str = "", limit: int = 10
) -> list[dict[str, Any]]:
    """Рисунки по смыслу — по подписи и описанию, в поддереве kb."""
    return TABLE.search(conn, embedder, query, kb, limit)


FIGURES_IN_ANSWER = 4  # рисунков под ответом: больше — не смотрят


def for_answer(
    conn: Any,
    sources: list[dict[str, Any]],
    used: list[int] | None,
    limit: int = FIGURES_IN_ANSWER,
) -> list[dict[str, Any]]:
    """Рисунки фрагментов, на которые сослался ответ ([n] из used; не сослался ни на
    один — ни одного): по порядку ссылок, у каждого — n фрагмента ответа. Сначала тот,
    что во фрагменте стоит, потом те, на что фрагмент ссылается."""
    chosen = set(used or [])
    cited = [s for s in sources if s.get("chunk_id") and s.get("n") in chosen]
    if not cited:
        return []
    by_chunk = in_chunks(conn, [int(s["chunk_id"]) for s in cited])
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for src in cited:
        for item in by_chunk.get(int(src["chunk_id"]), []):
            if item["id"] in seen or len(out) >= limit:
                continue
            seen.add(item["id"])
            out.append({**item, "source": src["n"]})
    return out


# --------------------------------------------------------------------------- #
#  Команда: python kbrag.py figure <doc_id> [номер] | figure --find «описание»
# --------------------------------------------------------------------------- #
def _show(item: dict[str, Any]) -> str:
    name = f"Рис. {item['label']}" if item.get("label") else f"Рисунок {item.get('n')}"
    page = f", с. {item['page']}" if item.get("page") else ""
    lines = [
        f"{name} — {item.get('title') or item.get('doc_id')}{page}",
        f"  файл: {item.get('file')}",
    ]
    if item.get("caption"):
        lines.append(f"  подпись: {item['caption'][:300]}")
    if item.get("description"):
        lines.append(f"  описание: {item['description'][:300]}")
    return "\n".join(lines + objtable.refs_line(item))


COMMAND = objtable.Command(
    prog="python kbrag.py figure",
    title="Рисунки статей",
    number_help="номер рисунка — как «Рис. 3» в статье",
    find_help="найти рисунки по подписи и описанию",
    json_key="figures",
    not_found="Рисунков не нашлось.",
    lookup=of_document,
    search=search,
    show=_show,
)


def main(argv: list[str] | None = None) -> int:
    return objtable.run(COMMAND, argv)


if __name__ == "__main__":
    raise SystemExit(main())
