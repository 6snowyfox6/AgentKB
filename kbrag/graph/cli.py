"""Граф знаний: факты из статей, которые выписывает Qwen3 и проверяет код.

    python -m kbrag.graph.cli build             Qwen3 проходит новые фрагменты
    python -m kbrag.graph.cli build --redo      всё заново (после правки правил)
    python -m kbrag.graph.cli build --limit 30  только 30 фрагментов — проба
    python -m kbrag.graph.cli report            что нашла модель и где ошибалась
    python -m kbrag.graph.cli dataset           сущности, у которых больше всего фактов
    python -m kbrag.graph.cli dataset --name "Анабарский щит"   датасет и файл CSV
    python -m kbrag.graph.cli synonyms          предложения в словарь синонимов
    python -m kbrag.graph.cli names             автосинонимы: новые имена — к известным (идёт и после build)
    python -m kbrag.graph.cli names --redo      автосинонимы заново, по всем именам
    python -m kbrag.graph.cli questions         вопросы оценки из фактов (.local/kbs/<база>/eval-graph.yaml)

Факт — «от — связь — к» с дословной цитатой (подробно — в facts.py). Разные
написания одного и того же сводит словарь synonyms.yaml базы (synonyms.py):
его правка действует сразу, модель заново проходить не нужно.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..common import add_db_args, add_embedder_args, add_llm_args, kb_from_args
from ..index import db
from ..paths import ROOT
from . import api, autosyn, confirm, facts, store
from .synonyms import (
    FILE,
    PROPOSALS,
    SynonymsError,
    judge_pairs,
    load,
    proposals_text,
    suggest,
)

# Сколько запросов держать у модели одновременно. vLLM считает их одной пачкой:
# на RTX 5070 Ti (16 ГБ) выписать и подтвердить — 1,3 с на фрагмент при 8 сразу
# против 2,1 с при 4 (40 фрагментов, 2026-10-12). Больше 8 vLLM на 16 ГБ всё равно
# не держит (--max-num-seqs в docker-compose.yml). Другое число — --workers или
# переменная KBRAG_WORKERS.
WORKERS = 8


def _shown(path: Path) -> str:
    """Путь для человека — от папки проекта, если файл в ней."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _llm(args: Any) -> Any:
    from ..llm import LLM

    return LLM(model=args.model, host=args.llm_host, timeout=args.timeout)


def _confirm(conn: Any, args: Any) -> int:
    """Подтвердить вопросами факты, которых ещё не проверяли (confirm.py)."""
    left = len(store.unconfirmed(conn, kb=args.base.path))
    if not left:
        print("Все факты уже проверены вопросами.")
        return 0
    todo = min(left, args.limit) if args.limit else left
    print(
        f"Проверяю факты вопросами по цитате: {todo}. В граф идут подтверждённые не меньше "
        f"чем {confirm.MIN_VOTES} проверками из 4. Можно прервать Ctrl+C — продолжится."
    )
    stats = confirm.run(conn, _llm(args), limit=args.limit, workers=args.workers, kb=args.base)
    print(
        f"Проверено фактов: {stats['done']}; в граф идут {stats['kept']}, остальные "
        "остаются в базе." + (f" Модель не ответила: {stats['errors']}." if stats["errors"] else "")
    )
    return 0


def _build(conn: Any, args: Any) -> int:
    llm = _llm(args)
    status = llm.status()
    if not status["ok"]:
        print(
            f"Qwen3 недоступна: {status['error']}\nГраф строит модель — запустите её (python kbrag.py start) и "
            "снова python kbrag.py graph.",
            file=sys.stderr,
        )
        return 1
    try:
        rules = facts.find_rules(args.rules, args.base)
        facts.load_rules(rules)
    except (OSError, ValueError) as exc:
        print(f"Правила для модели не читаются: {exc}", file=sys.stderr)
        return 1
    if args.redo:
        store.clear(conn, args.base.path)
        conn.commit()
        print("Прошлые факты убраны — модель проходит всё заново.")
    left = len(store.todo(conn, kb=args.base.path))
    if not left:
        print(
            "Новых фрагментов нет — модель уже прошла всё "
            "(заново: python kbrag.py graph --redo)."
        )
        return _confirm(conn, args)
    todo = min(left, args.limit) if args.limit else left
    print(
        f"Факты выписывает {args.model} по правилам {_shown(rules)}, каждый проверяет "
        f"код и тут же — вопросы по цитате. Фрагментов: {todo}, одновременно у модели "
        f"{args.workers}. "
        "Можно прервать Ctrl+C — продолжится с этого места."
    )
    stats = facts.run(
        conn,
        llm,
        limit=args.limit,
        rules_path=rules,
        workers=args.workers,
        confirm=lambda llm, fact: confirm.fact_votes(llm, fact, args.base),
        kb=args.base,
    )
    print(
        f"\nПройдено фрагментов: {stats['done']} из {stats['todo']}. Фактов принято: "
        f"{stats['facts']}; отброшено проверкой: {stats['rejected']}."
        + (f" Модель не ответила: {stats['errors']}." if stats["errors"] else "")
    )
    if not stats["stopped"]:
        _confirm(conn, args)
        _names(conn, args)
    print(
        "Что нашла модель и где ошибалась: python kbrag.py graph --new\n"
        "Свести разные написания одного и того же: python kbrag.py synonyms"
    )
    return 2 if stats["stopped"] else 0


def _names(conn: Any, args: Any) -> int:
    """Автосинонимы: новые имена графа сводятся к уже известным (autosyn.py)."""
    from ..index.embed import build_embedder

    try:
        syn = load(args.synonyms, args.base)
    except SynonymsError as exc:
        print(f"Словарь синонимов: {exc}", file=sys.stderr)
        return 1
    print(
        "Свожу написания одного и того же: имя сравнивается с уже известными по смыслу "
        f"(похожесть от {autosyn.AUTO})" + (" — всё заново." if args.redo else ".")
    )
    stats = autosyn.update(conn, build_embedder(device=args.device), syn, args.redo)
    print(
        f"Новых имён: {stats['new']}; сведено к известным: {stats['merged']}"
        + (
            f"; похожих, но разных по числам, сторонам или названиям: {stats['stopped_by_rules']}"
            if stats["stopped_by_rules"]
            else ""
        )
        + ". Переводы и сокращения («золото» — «gold») сводит словарь: python kbrag.py synonyms"
    )
    return 0


def _report(conn: Any, syn: Any, args: Any) -> int:
    data = api.report(conn, syn, kb=args.base.path)
    p = data["pass"]
    if not p["passed"]:
        print("Модель ещё не выписывала факты: python kbrag.py graph (нужна модель).")
        return 0
    print(
        f"\nМодель прошла фрагментов: {p['passed']} из {p['chunks']}; фактов: {p['facts']}; "
        f"отброшено проверкой: {p['rejected']}. Сущностей: {data['entities_total']}."
    )
    print(f"\n{'сущность':<46}{'статей':>7}{'фактов':>8}")
    for name, docs, count, known, spellings in data["entities"]:
        mark = "  (словарь)" if known else ""
        extra = f"  ← {', '.join(spellings)}" if spellings else ""
        print(f"  {name[:44]:<44}{docs:>7}{count:>8}{mark}{extra}"[:160])
    print("\nСвязи (сколько фактов):")
    for relation, count in data["relations"]:
        print(f"  {count:>5}  {relation}")
    if data["rejected"]:
        print("\nПочему код отбрасывал факты (чаще всего):")
        for reason, count in data["rejected"]:
            print(f"  {count:>5}  {reason[:90]}")
    print(
        f"\nРазные написания одного и того же сводит {FILE} базы; предложения "
        "для него: python kbrag.py synonyms"
    )
    return 0


def _dataset(conn: Any, syn: Any, args: Any) -> int:
    if args.json:
        try:
            data = (
                api.dataset_response(conn, syn, {"name": args.name}, kb=args.base.path)
                if args.name
                else api.datasets_response(conn, syn, limit=args.limit or 300, kb=args.base.path)
            )
        except api.RequestError as exc:
            data = exc.payload()
        print(json.dumps({"kb": args.base.path, **data}, ensure_ascii=False, indent=1, default=str))
        return 1 if "error" in data else 0
    if not args.name:
        data = api.datasets_response(conn, syn, limit=40, kb=args.base.path)
        if not data["datasets"]:
            print("Фактов пока нет: python kbrag.py graph (нужна модель).")
            return 0
        print(f"\nСущностей с фактами: {data['total']}. Больше всего фактов у:\n")
        print(f"{'сущность':<50}{'статей':>7}{'фактов':>8}")
        for r in data["datasets"]:
            print(f"{r['name'][:48]:<50}{r['documents']:>7}{r['facts']:>8}")
        print('\nФакты о любой из них: python kbrag.py facts "Анабарский щит"')
        return 0
    try:
        data = api.dataset_response(conn, syn, {"name": args.name}, kb=args.base.path)
    except api.RequestError as exc:
        near = ", ".join(exc.known[:15])
        print(f"{exc.message}: «{exc.detail}». Больше всего фактов у: {near}", file=sys.stderr)
        return 1
    inside = f" (вместе с: {', '.join(data['includes'])})" if data["includes"] else ""
    print(
        f"\nФакты: {data['name']}{inside} — статей {data['documents']}, "
        f"фактов {len(data['facts'])}\n"
    )
    for r in data["facts"]:
        line = (
            f"  {r['about']} — {r['relation']} — {r['other']}"
            if r["direction"] == "→"
            else f"  {r['other']} — {r['relation']} — {r['about']}"
        )
        print(f"{line[:110]:<112}статей {r['documents']}")
    out = Path(args.out) if args.out else args.base.local / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in " -_." else "_" for ch in data["name"])[:80]
    path = out / f"факты-{safe}.csv"
    path.write_text(api.dataset_csv(data)[1:], encoding="utf-8-sig")
    print(f"\nТаблица для Excel: {path}")
    return 0


def _synonyms(conn: Any, syn: Any, args: Any) -> int:
    from ..index.embed import build_embedder

    g = api.graph(conn, syn, args.base.path)
    if not g.entities:
        print("Фактов пока нет: python kbrag.py graph (нужна модель).")
        return 0
    llm = _llm(args)
    status = llm.status()
    if not status["ok"]:
        print(f"Qwen3 недоступна: {status['error']}", file=sys.stderr)
        return 1
    print("Загружаю модель векторов…")
    embedder = build_embedder(device=args.device)
    names = {name: (name, e["facts"]) for name, e in g.entities.items()}
    relations = Counter_relations(g)
    print(
        f"Ищу похожие среди {len(names)} имён и {len(relations)} связей; Qwen3 решает, "
        "одно ли это…"
    )
    name_groups = suggest(
        names,
        syn,
        embedder.encode,
        lambda p: judge_pairs(llm, p, args.base),
        threshold=args.threshold,
    )
    rel_groups = suggest(
        relations,
        _relations_as_names(syn),
        embedder.encode,
        lambda p: judge_pairs(llm, p, args.base),
        threshold=max(args.threshold, 0.85),
    )
    target = _shown(args.base.own_dir / FILE)
    out = args.base.local / PROPOSALS
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(proposals_text(name_groups, rel_groups, target), encoding="utf-8")
    print(
        f"\nПредложений: имён {len(name_groups)}, связей {len(rel_groups)}.\n"
        f"Файл: {out}\nПросмотрите Блокнотом и перенесите нужное в "
        f"{target} (или в словарь ветки выше) — действует сразу, модель заново не нужна."
    )
    return 0


def _questions(conn: Any, syn: Any, args: Any) -> int:
    from . import questions

    g = api.graph(conn, syn, args.base.path)
    if not g.links:
        print("Фактов пока нет: python kbrag.py graph (нужна модель).")
        return 0
    llm = None
    if not args.no_llm:
        llm = _llm(args)
        status = llm.status()
        if not status["ok"]:
            print(f"Qwen3 недоступна ({status['error']}) — вопросы по шаблону.")
            llm = None
    print(
        f"Составляю вопросы из фактов графа ({'Qwen3 переформулирует' if llm else 'по шаблону'})…"
    )
    items = questions.build(g, llm, limit=args.limit or 30, kb=args.base)
    out = Path(args.out_questions) if args.out_questions else args.base.local / questions.OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(questions.to_yaml(items), encoding="utf-8")
    docs = len({d for item in items for d in item["статьи"]})
    print(
        f"\nВопросов: {len(items)} по {docs} статьям. Файл: {out}\n"
        f"Оценка по ним: python kbrag.py eval --questions {out.relative_to(ROOT)}"
    )
    return 0


def Counter_relations(g: Any) -> dict[str, tuple[str, int]]:  # noqa: N802 — читается как таблица
    count: dict[str, int] = {}
    for link in g.links.values():
        count[link.relation] = count.get(link.relation, 0) + len(link.facts)
    return {r: (r, n) for r, n in count.items()}


def _relations_as_names(syn: Any) -> Any:
    """Для предложений по связям — словарь, где «имена» это связи."""
    from .synonyms import Synonyms

    groups: dict[str, list[str]] = {canon: [] for canon in set(syn.relations.values())}
    return Synonyms(names={}, groups=groups)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Граф знаний KBRAG: факты из статей")
    parser.add_argument(
        "command",
        choices=[
            "build",
            "confirm",
            "recheck",
            "names",
            "report",
            "dataset",
            "synonyms",
            "questions",
        ],
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="вопросы по шаблону, без модели (questions)"
    )
    parser.add_argument("--out-questions", default=None, help="иначе — в папку данных базы")
    parser.add_argument("--name", default=None, help="чей датасет (dataset)")
    parser.add_argument(
        "--json", action="store_true", help="dataset: факты одним объектом JSON в stdout"
    )
    parser.add_argument("--territory", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--out", default=None, help="куда класть таблицу датасета (иначе — в папку данных базы)"
    )
    parser.add_argument("--redo", action="store_true", help="модель проходит всё заново")
    parser.add_argument("--limit", type=int, default=None, help="сколько фрагментов пройти")
    parser.add_argument(
        "--rules", default=None, help="другой файл правил для модели (иначе — правила базы)"
    )
    parser.add_argument(
        "--synonyms", default=None, help="другой словарь синонимов (иначе — словари базы)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="насколько похожи имена, чтобы спросить модель (synonyms)",
    )
    parser.add_argument("--timeout", type=int, default=300, help="сколько ждать модель, с")
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("KBRAG_WORKERS") or WORKERS),
        help="сколько фрагментов у модели одновременно (build, confirm)",
    )
    add_db_args(parser)
    add_embedder_args(parser)
    add_llm_args(parser)
    args = parser.parse_args(argv)
    args.name = args.name or args.territory

    base = kb_from_args(args.kb)
    if base is None:
        return 1
    args.base = base
    print(
        f"База: {args.base.title} ({args.base.path or 'всё дерево'})",
        file=sys.stderr if args.json else sys.stdout,
    )
    try:
        syn = load(args.synonyms, args.base)
    except SynonymsError as exc:
        print(f"Словарь синонимов: {exc}", file=sys.stderr)
        return 1
    for warning in syn.warnings:
        print(f"Словарь синонимов, предупреждение: {warning}", file=sys.stderr)

    try:
        with db.connect(args.dsn) as conn:
            store.init(conn)
            if args.command == "build":
                return _build(conn, args)
            if args.command == "confirm":
                return _confirm(conn, args)
            if args.command == "recheck":
                facts.recheck(conn)
                return 0
            if args.command == "names":
                return _names(conn, args)
            if args.command == "report":
                return _report(conn, syn, args)
            if args.command == "dataset":
                return _dataset(conn, syn, args)
            if args.command == "questions":
                return _questions(conn, syn, args)
            return _synonyms(conn, syn, args)
    except KeyboardInterrupt:
        print("\nПрервано. Сделанное сохранено; та же команда продолжит с этого места.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
