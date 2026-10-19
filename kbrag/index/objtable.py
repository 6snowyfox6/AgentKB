"""Объекты статей в базе — формулы, рисунки: общее чтение, поиск и команда.

У формул (formulas.py) и рисунков (figures.py) одно устройство: строка на объект
статьи, рамка на странице (bbox), фрагмент, где объект стоит (chunk_id), вектор
описания и таблица ссылок на него из текста (formula_refs, figure_refs). Как такая
таблица читается — здесь, одним кодом; что в ней своё — в описании таблицы (Table):
столбцы, порядок в статье, имя таблицы ссылок.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from . import db

REF_TEXT = 300  # знаков текста ссылающегося фрагмента — в ответе


@dataclass(frozen=True)
class Table:
    name: str  # formulas
    alias: str  # f — в SQL
    columns: Sequence[str]  # свои столбцы, по порядку ключей keys
    keys: Sequence[str]  # ключи словаря объекта (index — вместо idx и т. п.)
    order: str  # столбец порядка в статье
    refs: str  # таблица ссылок из текста
    ref_column: str  # её столбец с id объекта

    @property
    def fields(self) -> str:
        own = ", ".join(f"{self.alias}.{c}" for c in self.columns)
        return f"{own}, d.title, d.year, d.url"

    @property
    def source(self) -> str:
        """FROM … JOIN documents — объект вместе с названием статьи."""
        return f"{self.name} {self.alias} JOIN documents d ON d.doc_id = {self.alias}.doc_id"

    def row(self, r: tuple[Any, ...]) -> dict[str, Any]:
        keys = (*self.keys, "title", "year", "url")
        out = dict(zip(keys, r[: len(keys)], strict=True))
        if isinstance(out.get("bbox"), str):
            out["bbox"] = json.loads(out["bbox"])
        return out

    # -- чтение ----------------------------------------------------------------- #
    def has_rows(self, conn: Any, doc_id: str) -> bool:
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {self.name} WHERE doc_id = %s LIMIT 1", (doc_id,))
            return cur.fetchone() is not None

    def by_id(self, conn: Any, item_id: int) -> dict[str, Any] | None:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {self.fields} FROM {self.source} WHERE {self.alias}.id = %s",
                (item_id,),
            )
            row = cur.fetchone()
        return self.row(row) if row else None

    def of_document(
        self, conn: Any, doc_id: str, clause: str = "", params: tuple[Any, ...] = ()
    ) -> list[dict[str, Any]]:
        """Объекты статьи по порядку (clause — ещё условие, с AND впереди) — у каждого
        фрагменты, которые на него ссылаются (referenced_by)."""
        a = self.alias
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {self.fields} FROM {self.source} WHERE {a}.doc_id = %s{clause} "
                f"ORDER BY {a}.{self.order}",
                (doc_id, *params),
            )
            found = [self.row(r) for r in cur.fetchall()]
            for item in found:
                cur.execute(
                    f"SELECT c.id, c.ord, c.pages, c.text FROM {self.refs} r "
                    f"JOIN chunks c ON c.id = r.chunk_id WHERE r.{self.ref_column} = %s "
                    "ORDER BY c.ord",
                    (item["id"],),
                )
                item["referenced_by"] = [
                    {
                        "chunk_id": c[0],
                        "ord": c[1],
                        "pages": list(c[2] or []),
                        "text": c[3][:REF_TEXT],
                    }
                    for c in cur.fetchall()
                ]
        return found

    def search(
        self, conn: Any, embedder: Any, query: str, kb: str = "", limit: int = 10
    ) -> list[dict[str, Any]]:
        """Объекты по смыслу — по вектору описания, в поддереве kb."""
        a = self.alias
        where, params = db.in_kb(kb, f"{a}.doc_id")
        vector = db.vector_literal(embedder.encode_one(query))
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT {self.fields}, 1 - ({a}.embedding <=> %(q)s::vector) AS similarity
                FROM {self.source}
                WHERE {a}.embedding IS NOT NULL{where}
                ORDER BY {a}.embedding <=> %(q)s::vector
                LIMIT %(limit)s
                """,
                {**params, "q": vector, "limit": limit},
            )
            rows = cur.fetchall()
        return [{**self.row(r), "similarity": round(float(r[-1]), 4)} for r in rows]


def refs_line(item: dict[str, Any]) -> list[str]:
    """«ссылаются фрагменты: 3, 7» — для показа в терминале; ссылок нет — ничего."""
    refs = item.get("referenced_by") or []
    return ["  ссылаются фрагменты: " + ", ".join(str(r["ord"]) for r in refs)] if refs else []


# --------------------------------------------------------------------------- #
#  Команда: python kbrag.py formula|figure <doc_id> [номер] | --find «описание»
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Command:
    prog: str  # python kbrag.py formula
    title: str  # Формулы статей
    number_help: str  # что за второй аргумент
    find_help: str
    json_key: str  # formulas
    not_found: str  # Формул не нашлось.
    lookup: Callable[[Any, str, str | None], list[dict[str, Any]]]  # (conn, doc_id, номер)
    search: Callable[..., list[dict[str, Any]]]  # (conn, embedder, запрос, kb, limit)
    show: Callable[[dict[str, Any]], str]


def run(command: Command, argv: list[str] | None = None) -> int:
    import argparse

    from ..common import add_db_args, kb_from_args

    parser = argparse.ArgumentParser(prog=command.prog, description=command.title)
    parser.add_argument("doc_id", nargs="?", help="статья (doc_id)")
    parser.add_argument("number", nargs="?", help=command.number_help)
    parser.add_argument("--find", default="", help=command.find_help)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="одним объектом JSON в stdout")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    add_db_args(parser)
    args = parser.parse_args(argv)
    if not args.doc_id and not args.find:
        parser.print_usage(sys.stderr)
        return 1
    base = kb_from_args(args.kb)
    if base is None:
        return 1
    with db.connect(args.dsn) as conn:
        db.init_db(conn, with_hnsw=False)
        if args.find:
            from .embed import build_embedder

            embedder = build_embedder(device=args.device)
            found = command.search(conn, embedder, args.find, base.path, args.limit)
        else:
            found = command.lookup(conn, args.doc_id, args.number)
    if args.json:
        payload = {"kb": base.path, command.json_key: found}
        print(json.dumps(payload, ensure_ascii=False, indent=1))
        return 0 if found else 1
    if not found:
        print(command.not_found, file=sys.stderr)
        return 1
    for item in found:
        print(command.show(item) + "\n")
    return 0
