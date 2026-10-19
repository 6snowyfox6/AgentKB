"""CLI базы знаний.

    python -m kbrag.index.cli init                     # создать таблицы и индексы
    python -m kbrag.index.cli ingest                   # разобранное → векторы → база
    python -m kbrag.index.cli ingest --force           # переиндексировать всё заново
    python -m kbrag.index.cli search "методы измерения вязкости"
    python -m kbrag.index.cli stats                    # что лежит в базе

Адрес базы берётся из переменной KBRAG_DSN, иначе локальный контейнер
из docker-compose.yml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..common import add_db_args, kb_from_args
from ..paths import ARTICLES
from . import db
from .embed import build_embedder
from .ingest import ingest_dir, tidy_indexed
from .search import hit_dict, hybrid_search


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="База знаний KBRAG: индексация и поиск")
    p.add_argument("command", choices=["init", "ingest", "tidy", "search", "stats"])
    p.add_argument("query", nargs="?", help="поисковый запрос для команды search")
    add_db_args(p)
    p.add_argument("--acquired", type=Path, default=ARTICLES)
    p.add_argument("--model", default=None, help="модель векторов (по умолчанию BAAI/bge-m3)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--batch", type=int, default=8, help="сколько чанков кодировать за раз")
    p.add_argument("--force", action="store_true", help="переиндексировать, даже если уже есть")
    p.add_argument(
        "--no-summaries", action="store_true", help="ingest: без резюме статей (их пишет модель)"
    )
    p.add_argument("--limit", type=int, default=10, help="сколько результатов показать")
    p.add_argument("--candidates", type=int, default=50, help="сколько брать из каждого поиска")
    p.add_argument("--year-from", type=int, default=None)
    p.add_argument("--source", default=None, help="искать только по одному источнику")
    p.add_argument(
        "--per-doc",
        type=int,
        default=2,
        help="сколько кусков одной статьи показывать (0 — без ограничения)",
    )
    p.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="отсечь непохожее: 0.45 — обычно, 0.55 — строго, 0 — показывать всё",
    )
    p.add_argument(
        "--require-words",
        action="store_true",
        help="только куски, где есть все слова запроса",
    )
    p.add_argument("--no-hnsw", action="store_true", help="не строить векторный индекс")
    p.add_argument(
        "--json", action="store_true", help="search: выдача одним объектом JSON в stdout"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "search" and not args.query:
        print("Для search нужен запрос: python -m kbrag.index.cli search «текст»", file=sys.stderr)
        return 1
    base = kb_from_args(args.kb)
    if base is None:
        return 1

    try:
        with db.connect(args.dsn) as conn:
            if args.command == "init":
                db.init_db(conn, with_hnsw=not args.no_hnsw)
                print("Таблицы и индексы на месте.")
                # Статьи, загруженные до дерева баз, — в ветку --kb (или базу по
                # умолчанию): иначе поиск и граф по узлу их не видят. Ветка не названа —
                # остаются без ветки (их видит всё дерево), и подсказка, как положить.
                home = base.path
                placed = db.assign_unplaced(conn, home)
                conn.commit()
                if placed:
                    print(f"Статей без ветки положено в «{home}»: {placed}.")
                elif not home and (left := db.unplaced(conn)):
                    print(
                        f"Статей без ветки: {left} — их видит только всё дерево. Положить в "
                        "ветку: python kbrag.py init --kb <ветка> (например physics.optics)."
                    )
                # Язык фрагментов, загруженных до того, как его стали определять (lang.py).
                filled = db.fill_languages(conn)
                conn.commit()
                if filled:
                    print(f"Язык определён у фрагментов: {filled}.")
                return 0

            if args.command == "stats":
                info = db.stats(conn, base.path)
                print(f"База: {base.title} ({base.path or 'всё дерево'})")
                lo, hi = info["years"]
                print(f"Документов: {info['documents']}")
                print(f"Чанков: {info['chunks']} (с вектором: {info['embedded']})")
                print(f"Годы: {lo or '—'}–{hi or '—'}")
                return 0

            embedder = build_embedder(
                model_id=args.model, device=args.device, batch_size=args.batch
            )

            if args.command == "ingest":
                if not args.acquired.exists():
                    print(f"Папки {args.acquired} нет — сначала import", file=sys.stderr)
                    return 1
                print(f"Векторы считает BGE-M3 на {args.device}")
                report = ingest_dir(conn, embedder, args.acquired, force=args.force)
                # Новые статьи — в выбранную базу (--kb). Уже загруженные остаются в
                # своих ветках: папка статей одна на все базы.
                if base.path:
                    db.assign(conn, report.indexed_ids, base.path)
                    db.assign_unplaced(conn, base.path)
                    conn.commit()
                print(
                    f"\nДокументов просмотрено {report.seen}, проиндексировано {report.indexed}, "
                    f"пропущено {report.skipped}, без чанков {report.empty}; "
                    f"чанков записано {report.chunks}"
                    + (f"; формул {report.formulas}" if report.formulas else "")
                    + (f"; рисунков {report.figures}" if report.figures else "")
                    + (
                        f"; библиографии отброшено {report.refs_dropped}"
                        if report.refs_dropped
                        else ""
                    )
                    + (
                        f"; служебного текста: фрагментов убрано {report.noise_dropped}, "
                        f"подчищено {report.noise_trimmed}"
                        if report.noise_dropped or report.noise_trimmed
                        else ""
                    )
                    + (
                        f"; повторов одной статьи пропущено {report.duplicates}"
                        if report.duplicates
                        else ""
                    )
                    + (
                        f"; убранных через clean забыто {report.forgotten}"
                        if report.forgotten
                        else ""
                    )
                )
                for error in report.errors:
                    print(f"  ! {error}")
                if not args.no_summaries:
                    from .summaries import after_ingest

                    after_ingest(conn, embedder, report.indexed_ids)
                return 0

            if args.command == "tidy":
                print(f"Векторы считает BGE-M3 на {args.device}")
                tidy = tidy_indexed(conn, embedder)
                print(
                    f"\nСлужебный текст убран в {tidy.documents} статьях: фрагментов убрано "
                    f"{tidy.dropped}, подчищено {tidy.trimmed}; фактов графа ушло вместе "
                    f"с ним {tidy.facts_dropped}"
                )
                return 0

            hits = hybrid_search(
                conn,
                embedder,
                args.query,
                limit=args.limit,
                candidates=args.candidates,
                year_from=args.year_from,
                source=args.source,
                max_per_doc=args.per_doc,
                min_similarity=args.min_similarity,
                require_words=args.require_words,
                kb=base.path,
            )
            if args.json:
                found = [hit_dict(h) for h in hits]
                print(
                    json.dumps(
                        {"query": args.query, "kb": base.path, "hits": found},
                        ensure_ascii=False,
                        indent=1,
                    )
                )
                return 0
            if not hits:
                print("Ничего подходящего не нашлось.")
                if args.min_similarity or args.require_words:
                    print("Возможно, порог слишком строгий — попробуйте без --min-similarity.")
                else:
                    print("Проверьте, что в базе есть статьи: ... cli stats")
                return 0

            print(f"\nНайдено {len(hits)} по запросу «{args.query}»\n" + "=" * 78)
            for i, hit in enumerate(hits, start=1):
                where = " / ".join(hit.headings[-2:]) if hit.headings else ""
                print(f"\n[{i}] {hit.title[:70]}")
                near = f" · похожесть {hit.similarity:.0%}" if hit.similarity is not None else ""
                print(
                    f"    {hit.year or '—'} · {hit.journal or 'источник не указан'} · "
                    f"нашёл: {hit.found_by}{near}"
                )
                if where:
                    print(f"    раздел: {where}")
                if hit.pages:
                    print(f"    страницы: {', '.join(str(p) for p in hit.pages)}")
                print(f"    {hit.url}")
                snippet = " ".join(hit.text.split())[:400]
                print(f"    {snippet}…")
            return 0

    except ImportError as exc:
        print(f"Не хватает библиотеки: {exc}", file=sys.stderr)
        print("Поставьте: pip install -r requirements.txt", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"База недоступна или ответила ошибкой: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Проверьте, что контейнер запущен: docker compose ps", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
