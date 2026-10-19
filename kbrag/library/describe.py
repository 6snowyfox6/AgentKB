"""Рисунки статей — моделью со зрением, по сохранённым файлам (без переразбора PDF).

    python kbrag.py figures                       все неописанные рисунки базы (--kb)
    python kbrag.py figures --doc 10.1234_abcd    одной статьи
    python kbrag.py figures --redo                описать заново и описанные

Описание пишется в запись рисунка (поле figures записи статьи), во фрагменты статьи
добавляются фрагменты «Рисунок» (прежние такие — заменяются), и статья загружается в
базу заново: описание находит поиск, у рисунка в таблице рисунков — оно же.

Модель со зрением — сервис vision в docker-compose.yml (или любой сервер с протоколом
OpenAI: KBRAG_FIGURE_LLM, KBRAG_FIGURE_MODEL). Как она делит видеокарту с основной
моделью — KBRAG_VISION_MODE (chunks/figures.py, vision_mode); запускатель
(python kbrag.py figures) сам поднимает и гасит, что нужно.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..chunks.figures import (
    FIGURE_HOST,
    FIGURE_MODEL,
    describe_saved,
    description_chunks,
    is_description,
)
from ..common import add_db_args, kb_from_args
from ..kb import KB
from ..paths import ARTICLES, LOCAL
from . import store


def refresh(doc_id: str, record: dict[str, Any], articles: Path) -> int:
    """Фрагменты «Рисунок» статьи — по описаниям в её записи: прежние убираются, новые
    идут в конец. Записывает и запись, и фрагменты. Сколько фрагментов «Рисунок»."""
    path = articles / f"{doc_id}.chunks.json"
    chunks = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    text = [c for c in chunks if not is_description(c)]
    extra = [
        c.to_dict() for c in description_chunks(doc_id, record.get("figures") or [], len(text))
    ]
    path.write_text(json.dumps([*text, *extra], ensure_ascii=False, indent=2), encoding="utf-8")
    store.save_record(articles, record)
    return len(extra)


def describe_articles(
    doc_ids: list[str],
    articles: Path,
    client: Any,
    kb: KB | None = None,
    redo: bool = False,
    log: Any = print,
) -> list[str]:
    """Описать рисунки статей; какие статьи изменились (их надо загрузить заново)."""
    records = store.records(articles)
    changed = []
    for i, doc_id in enumerate(doc_ids, start=1):
        record = records.get(doc_id)
        pictures = (record or {}).get("figures") or []
        if record is None or not pictures:
            continue
        todo = sum(1 for f in pictures if redo or not f.get("description"))
        if not todo:
            continue
        log(f"[{i}/{len(doc_ids)}] {doc_id}: рисунков {todo}")
        done = describe_saved(pictures, articles, client, kb=kb, log=log, redo=redo)
        if done:
            refresh(doc_id, record, articles)
            changed.append(doc_id)
            log(f"  описано {done}")
    return changed


PENDING = LOCAL / "figures-pending.json"  # описаны, но ещё не загружены в базу


def _pending(add: list[str] | None = None) -> list[str]:
    known = json.loads(PENDING.read_text(encoding="utf-8")) if PENDING.is_file() else []
    if add is None:
        return [str(x) for x in known]
    ids = sorted({*known, *add})
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    PENDING.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    return ids


def _ingest(args: Any, ids: list[str]) -> int:
    """Загрузить статьи с новыми описаниями в базу заново."""
    from ..index import db
    from ..index.embed import build_embedder
    from ..index.ingest import ingest_dir

    embedder = build_embedder(device=args.device)
    with db.connect(args.dsn) as conn:
        db.init_db(conn)
        report = ingest_dir(conn, embedder, args.articles, force=True, only=set(ids))
        conn.commit()
    print(f"В базе: записано {report.indexed}; рисунков {report.figures}")
    for error in report.errors:
        print(f"  ! {error}")
    if not report.errors:
        PENDING.unlink(missing_ok=True)
    return 1 if report.errors else 0


def main(argv: list[str] | None = None) -> int:
    from ..llm import LLM

    p = argparse.ArgumentParser(prog="python kbrag.py figures", description=__doc__)
    p.add_argument("--doc", action="append", default=[], help="только эта статья")
    p.add_argument("--redo", action="store_true", help="описать заново и уже описанные")
    p.add_argument(
        "--no-ingest",
        action="store_true",
        help="только описать; статьи запомнятся, загрузит их --ingest-pending",
    )
    p.add_argument(
        "--ingest-pending", action="store_true", help="загрузить описанное прежде с --no-ingest"
    )
    p.add_argument("--articles", type=Path, default=ARTICLES, help="папка статей")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    add_db_args(p)
    args = p.parse_args(argv)
    base = kb_from_args(args.kb)
    if base is None:
        return 1
    if args.ingest_pending:
        pending = _pending()
        return _ingest(args, pending) if pending else 0
    ids = args.doc
    if not ids:
        from ..index import db

        with db.connect(args.dsn) as conn:
            ids = [d["doc_id"] for d in db.documents(conn, limit=1_000_000, kb=base.path)]
    print(
        f"Рисунки: {len(ids)} статей ({base.path or 'всё дерево'}); модель {FIGURE_MODEL} на {FIGURE_HOST}"
    )
    client = LLM(model=FIGURE_MODEL, host=FIGURE_HOST, temperature=0, timeout=300)
    changed = describe_articles(ids, args.articles, client, kb=base, redo=args.redo)
    print(f"Статей с новыми описаниями: {len(changed)}")
    if not changed:
        return 0
    if args.no_ingest:
        _pending(changed)  # загрузит --ingest-pending: когда видеокарта освободится
        return 0
    return _ingest(args, changed)


if __name__ == "__main__":
    raise SystemExit(main())
