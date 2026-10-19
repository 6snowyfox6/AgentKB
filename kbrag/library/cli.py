"""Свои PDF в базу знаний: import и reparse.

    python -m kbrag.library.cli import D:/статьи --kb physics.hadron.elsm
    python -m kbrag.library.cli import статья.pdf --ask         спросить, чего нет в метаданных
    python -m kbrag.library.cli reparse --kb geology.ore.anabar  все статьи базы с PDF
    python -m kbrag.library.cli reparse --doc 10.1234_abcd

После разбора статьи загружаются в базу и кладутся в ветку --kb (import) —
отдельный ingest не нужен; --no-ingest — только разобрать.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..common import add_db_args, kb_from_args
from ..parse.config import Settings
from ..parse.models import ParserSetupError
from ..paths import ARTICLES
from . import importer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Свои PDF в базу знаний")
    p.add_argument("command", choices=["import", "reparse"])
    p.add_argument("paths", nargs="*", type=Path, help="import: PDF или папки с ними")
    p.add_argument("--doc", action="append", default=[], help="reparse: только эта статья")
    p.add_argument("--ask", action="store_true", help="спросить в терминале, чего нет")
    p.add_argument("--force", action="store_true", help="разобрать и тот файл, что уже был")
    p.add_argument(
        "--figures", action="store_true", help="описать рисунки моделью со зрением (qwen3-vl)"
    )
    p.add_argument("--no-images", action="store_true", help="не сохранять рисунки файлами")
    p.add_argument("--ocr", action="store_true", help="сканы — распознать текст (медленно)")
    p.add_argument("--no-ingest", action="store_true", help="только разобрать, в базу не грузить")
    p.add_argument(
        "--no-summaries", action="store_true", help="без резюме статей и разделов (пишет модель)"
    )
    p.add_argument("--articles", type=Path, default=ARTICLES, help="папка статей")
    p.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    add_db_args(p)
    return p


def _settings(args: Any) -> Settings:
    s = Settings()
    s.device = args.device
    s.figures = not args.no_images  # рисунки docling вырезает, только если попросить
    s.ocr = args.ocr  # сканы: распознать текст страниц (docling + OCR)
    return s


def _ids_in_kb(args: Any, kb_path: str) -> list[str]:
    """Статьи поддерева базы (для reparse без --doc)."""
    from ..index import db

    with db.connect(args.dsn) as conn:
        return [d["doc_id"] for d in db.documents(conn, limit=1_000_000, kb=kb_path)]


def _load(args: Any, items: list[importer.Item], kb_path: str, force: bool) -> int:
    """Загрузить разобранное в базу и положить статьи в ветку kb."""
    from ..index import db
    from ..index.embed import build_embedder
    from ..index.ingest import ingest_dir

    ids = {i.doc_id for i in items if i.indexed}
    if not ids:
        return 0
    print(f"\nЗагружаю в базу: {len(ids)} статей. Векторы считает BGE-M3 ({args.device})")
    embedder = build_embedder(device=args.device)
    with db.connect(args.dsn) as conn:
        db.init_db(conn)
        report = ingest_dir(conn, embedder, args.articles, force=force, only=ids)
        placed = db.assign(conn, sorted(ids), kb_path) if kb_path else 0
        conn.commit()
        if not args.no_summaries:
            from ..index.summaries import after_ingest

            after_ingest(conn, embedder, report.indexed_ids)
    print(
        f"В базе: записано {report.indexed}, без изменений {report.skipped}, повторов "
        f"{report.duplicates}; фрагментов {report.chunks}"
        + (f"; в ветку «{kb_path}» добавлено {placed}" if kb_path else "")
    )
    for error in report.errors:
        print(f"  ! {error}")
    return 1 if report.errors else 0


def _report(items: list[importer.Item]) -> None:
    print("\n" + "=" * 78)
    for item in items:
        name = Path(item.file).name if item.file else item.doc_id
        line = f"{item.status:<14}{name[:44]:<46}{item.chunks:>5} фр. {item.figures:>3} рис."
        print(line + (f"\n{'':14}{item.note}" if item.note else ""))
    counts: dict[str, int] = {}
    for item in items:
        counts[item.status] = counts.get(item.status, 0) + 1
    print("-" * 78 + "\n" + ", ".join(f"{k}: {v}" for k, v in counts.items()))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base = kb_from_args(args.kb)
    if base is None:
        return 1
    settings = _settings(args)
    try:
        if args.command == "import":
            if not args.paths:
                print("Что импортировать: python kbrag.py import <папка или PDF>", file=sys.stderr)
                return 1
            missing = [str(p) for p in args.paths if not p.exists()]
            if missing:
                print(f"Нет такого: {', '.join(missing)}", file=sys.stderr)
                return 1
            if not base.path:
                print(
                    "База не выбрана (--kb): статьи не лягут ни в одну ветку дерева — их "
                    "увидит только всё дерево (--kb .)."
                )
            print(f"База: {base.title} ({base.path or 'всё дерево'})")
            items = importer.import_pdfs(
                args.paths,
                settings,
                args.articles,
                ask=args.ask,
                force=args.force,
                describe=args.figures,
                kb=base,
            )
            force = args.force
        else:
            ids = args.doc or _ids_in_kb(args, base.path)
            if not ids:
                print(f"В базе «{base.path or 'всё дерево'}» статей нет.", file=sys.stderr)
                return 1
            print(f"Переразбор: {len(ids)} статей ({base.path or 'всё дерево'})")
            items = importer.reparse(ids, settings, args.articles, describe=args.figures, kb=base)
            force = True  # фрагментов столько же, а текст другой — переписать
    except ParserSetupError as exc:
        print(f"\nРазбор остановлен: {exc}", file=sys.stderr)
        return 2
    _report(items)
    if args.no_ingest:
        return 0
    return _load(args, items, base.path if args.command == "import" else "", force)


if __name__ == "__main__":
    raise SystemExit(main())
