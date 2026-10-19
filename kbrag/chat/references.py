"""Статьи, на которые опирается ответ: список литературы под ответом чат-бота.

Ответ ссылается на фрагменты — [1], [3], — а человеку нужны статьи: кто, когда, где
напечатано и как открыть. Здесь процитированные фрагменты (used) собираются по статьям
в порядке первой ссылки: у каждой — библиографическая строка, номера её фрагментов,
страницы и ссылки: DOI, страница издателя и свой PDF на диске (сервер страницы отдаёт
его по /api/pdf, открывается на процитированной странице; Телеграм — файлом).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

AUTHORS_SHOWN = 3  # больше — «и др.»


def group(sources: list[dict[str, Any]], used: list[int] | None) -> list[dict[str, Any]]:
    """Процитированные фрагменты → статьи, по порядку первой ссылки. Сводка фактов графа
    (kind «датасет») — не статья; её фрагменты-доказательства — статьи."""
    chosen = set(used or [])
    out: dict[str, dict[str, Any]] = {}
    for src in sorted(sources, key=lambda s: int(s.get("n") or 0)):
        doc_id = str(src.get("doc_id") or "")
        if src.get("n") not in chosen or not doc_id or src.get("kind") == "датасет":
            continue
        item = out.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": src.get("title") or doc_id,
                "authors": list(src.get("authors") or []),
                "year": src.get("year"),
                "journal": src.get("journal"),
                "url": src.get("url") or "",
                "doi": "",
                "fragments": [],
                "pages": [],
                "pdf": False,
            },
        )
        item["fragments"].append(int(src["n"]))
        for page in src.get("pages") or []:
            if page not in item["pages"]:
                item["pages"].append(page)
    for item in out.values():
        item["pages"].sort()
    return list(out.values())


def enrich(conn: Any, articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DOI из базы и есть ли свой PDF (путь из записи статьи, файл на месте)."""
    from ..index import db
    from ..library import objects

    if not articles:
        return articles
    if db.is_real(conn):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, doi FROM documents WHERE doc_id = ANY(%s)",
                ([a["doc_id"] for a in articles],),
            )
            dois = {str(r[0]): str(r[1] or "") for r in cur.fetchall()}
        for item in articles:
            item["doi"] = dois.get(item["doc_id"], "")
    for item in articles:
        item["pdf"] = pdf_path(item["doc_id"], objects.record) is not None
    return articles


def pdf_path(doc_id: str, record: Any = None) -> Path | None:
    """Свой PDF статьи: путь из её записи (import), если файл на месте и это PDF."""
    if record is None:
        from ..library.objects import record
    raw = (record(doc_id) or {}).get("pdf_path")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.suffix.lower() == ".pdf" and path.is_file() else None


def articles(
    conn: Any, sources: list[dict[str, Any]], used: list[int] | None
) -> list[dict[str, Any]]:
    """Статьи ответа с DOI и PDF; база не ответила — без них, но список есть."""
    found = group(sources, used)
    try:
        return enrich(conn, found)
    except Exception:  # noqa: BLE001 — без DOI и PDF список всё равно полезен
        return found


def doi_link(doi: str) -> str:
    doi = (doi or "").strip()
    if not doi:
        return ""
    return doi if doi.startswith("http") else f"https://doi.org/{doi}"


def citation(item: dict[str, Any]) -> str:
    """«Иванов И. И., Петров П. П. (2021). Название. Журнал» — без ссылок и номеров."""
    authors = [str(a) for a in item.get("authors") or [] if str(a).strip()]
    who = ", ".join(authors[:AUTHORS_SHOWN]) + (" и др." if len(authors) > AUTHORS_SHOWN else "")
    year = f" ({item['year']})" if item.get("year") else ""
    head = f"{who}{year}. " if who else (f"{item['year']}. " if item.get("year") else "")
    journal = f". {item['journal']}" if item.get("journal") else ""
    return f"{head}{item.get('title') or item.get('doc_id')}{journal}"
