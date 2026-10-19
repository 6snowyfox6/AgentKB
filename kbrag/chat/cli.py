"""Спросить чат-бота из терминала.

    python -m kbrag.chat.cli "какие методы описаны в статьях базы?"

То же, что вкладка «Чат-бот» в браузере, только один вопрос. Модель
эмбеддингов грузится на каждый запуск, поэтому для разговора удобнее
браузер: python kbrag.py web.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ..common import add_db_args, add_embedder_args, add_llm_args, kb_from_args
from ..index import db
from ..index.embed import build_embedder
from .answer import answer, describe_timings, llm_status
from .settings import Settings

SOURCE_FIELDS = ("n", "doc_id", "chunk_id", "title", "year", "journal", "authors", "pages", "url")


def collect(events: Any, kb: str = "") -> dict[str, Any]:
    """События ответа (chat.answer) → один объект: текст (уже без фраз без ссылки на
    фрагмент), источники, какие процитированы, режим, время по этапам, как искал."""
    out: dict[str, Any] = {"kb": kb, "answer": "", "sources": [], "used": [], "mode": None}
    text = ""
    for event in events:
        kind = event["type"]
        if kind == "status":
            print(event["text"], file=sys.stderr)
        elif kind == "sources":
            out["sources"] = [
                {k: s.get(k) for k in SOURCE_FIELDS} | {"text": s.get("text", "")}
                for s in event["sources"]
            ]
            out["search"] = {
                k: event.get(k)
                for k in ("queries", "parts", "extra", "missing", "dataset", "rounds", "reranked")
            }
        elif kind == "general":
            out["note"] = event["text"]
        elif kind == "token":
            text += event["text"]
        elif kind == "revised":
            text = event["text"]
            out["removed"] = event["removed"]
        elif kind == "error":
            out["error"] = event["error"]
        elif kind == "done":
            out.update(
                {
                    k: event.get(k)
                    for k in ("mode", "used", "unknown", "coverage", "model", "articles")
                }
                | {"seconds": event.get("seconds"), "timings": event.get("timings")}
            )
    out["answer"] = text.strip()
    return out


def print_articles(articles: list[dict[str, Any]]) -> None:
    """Статьи, на которые опирается ответ: строка статьи, её фрагменты, ссылки."""
    from .references import citation, doi_link

    if not articles:
        return
    print("\nСтатьи, на которые опирается ответ:")
    for i, item in enumerate(articles, start=1):
        cited = ", ".join(f"[{n}]" for n in item["fragments"])
        print(f"{i}. {citation(item)} — {cited}")
        for link in dict.fromkeys(x for x in (doi_link(item.get("doi", "")), item.get("url")) if x):
            print(f"   {link}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Вопрос к чат-боту базы знаний")
    parser.add_argument("question")
    parser.add_argument(
        "--json",
        action="store_true",
        help="ответ одним объектом JSON в stdout (для скриптов и Claude Code), остальное — в stderr",
    )
    add_llm_args(parser)
    add_db_args(parser)
    add_embedder_args(parser)
    args = parser.parse_args(argv)

    base = kb_from_args(args.kb)
    if base is None:
        return 1
    print(f"База: {base.title} ({base.path or 'всё дерево'})", file=sys.stderr)
    settings = Settings(model=args.model, host=args.llm_host, kb=base.path)
    status = llm_status(settings.host, settings.model)
    if not status["ok"]:
        print(status["error"], file=sys.stderr)
        return 1

    print("Загружаю модель эмбеддингов…", file=sys.stderr)
    embedder = build_embedder(device=args.device)
    sources: list[dict[str, Any]] = []
    code = 0
    if args.json:
        with db.connect(args.dsn) as conn:
            result = collect(answer(conn, embedder, args.question, settings=settings), base.path)
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 1 if result.get("error") else 0
    with db.connect(args.dsn) as conn:
        for event in answer(conn, embedder, args.question, settings=settings):
            if event["type"] == "status":
                print(event["text"], file=sys.stderr)
            elif event["type"] == "sources":
                sources = event["sources"]
                if event.get("dataset"):
                    print(
                        f"Собрано из фактов графа о «{event['dataset']}» — по всем статьям базы",
                        file=sys.stderr,
                    )
                for i, part in enumerate(event.get("parts") or [], start=1):
                    print(
                        f"Часть {i}: {part['question']} — фрагментов {part['found']}",
                        file=sys.stderr,
                    )
                print("Искал: " + " · ".join(f"«{q}»" for q in event["queries"]), file=sys.stderr)
                if event.get("judged") is False:
                    print(
                        "Модель при отборе не ответила — фрагменты взяты по близости",
                        file=sys.stderr,
                    )
                elif event.get("checked"):
                    print(
                        f"Модель прочла найденное (фрагментов: {event['checked']}), взяла "
                        f"{len(sources)}; кругов поиска: {event.get('rounds', 1)}",
                        file=sys.stderr,
                    )
                if event.get("extra"):
                    missing = event.get("missing") or []
                    print(
                        "Искал ещё"
                        + (f" (не хватало: {'; '.join(missing)})" if missing else "")
                        + ": "
                        + " · ".join(f"«{q}»" for q in event["extra"]),
                        file=sys.stderr,
                    )
                print(file=sys.stderr)
            elif event["type"] == "general":
                print("Искал: " + " · ".join(f"«{q}»" for q in event["queries"]), file=sys.stderr)
                print(f"\n!!! {event['text']}\n")
            elif event["type"] == "token":
                print(event["text"], end="", flush=True)
            elif event["type"] == "revised":
                # В терминале напечатанное не стереть — показываем, что убрано.
                print("\n\nУбрано как фразы без ссылки на фрагмент (слова модели, не статей):")
                for phrase in event["removed"]:
                    print(f"  − {phrase}")
            elif event["type"] == "error":
                print(f"\nОшибка: {event['error']}", file=sys.stderr)
                code = 1
            elif event["type"] == "done":
                print("\n")
                used = set(event["used"])
                for s in sources:
                    mark = "→" if s["n"] in used else " "
                    pages = f", с. {s['pages'][0]}" if s.get("pages") else ""
                    year = f", {s['year']}" if s.get("year") else ""
                    print(f"{mark} [{s['n']}] {s['title'][:80]}{year}{pages}")
                    if s.get("url"):
                        print(f"       {s['url']}")
                print_articles(event.get("articles") or [])
                if event["unknown"]:
                    print(
                        f"\nВнимание: модель сослалась на несуществующие фрагменты "
                        f"{event['unknown']} — эти утверждения не подтверждены."
                    )
                cited, claims = event.get("coverage") or (0, 0)
                if claims:
                    print(f"\nСсылка на фрагмент — у {cited} из {claims} утверждений.")
                if event.get("mode") == "без базы":
                    print("[Ответ модели из общих знаний — не из базы знаний]")
                if event.get("model"):
                    print(f"{event['model']}, {event['seconds']} с")
                if event.get("timings"):
                    print("Время: " + describe_timings(event["timings"]))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
