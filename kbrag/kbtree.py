"""Управление деревом баз: kb list, create, move, assign, unassign, remove.

    python kbrag.py kb list
    python kbrag.py kb create physics.hadron.elsm --title "Модель ЭЛСМ"
    python kbrag.py kb move geology.ore.anabar geology.shields.anabar
    python kbrag.py kb assign physics.hadron 10.1103_physrevc.99.000001
    python kbrag.py kb assign physics.hadron --from physics.old
    python kbrag.py kb unassign physics.hadron 10.1103_physrevc.99.000001
    python kbrag.py kb remove physics.old            показать, что уберётся
    python kbrag.py kb remove physics.old --yes      убрать

Узел — и строка в базе (таблица kbs: к нему привязаны статьи), и папка настроек в
config/kbs (kb.yaml, синонимы, правила графа, вопросы оценки), и папка данных в
.local/kbs. Команды держат все три вместе: create делает папку с kb.yaml, move
переносит и папки, и статьи, remove убирает узел из базы, а папки — только с --config.
Настройки наследуются от родителя к детям (kb.py): у новой базы без своих слов —
слова ветки выше.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from . import kb
from .common import add_db_args

TEMPLATE = """# НАСТРОЙКИ БАЗЫ {path} — {title}.
#
# Чего здесь нет, берётся у ветки выше (config/kbs/…), до config/kbs/kb.yaml — там же
# сказано, что значит каждое слово темы и в каком оно падеже. Достаточно написать
# только своё. Правится Блокнотом, сохранять в UTF-8; действует сразу.

название: {title}

# тема:
#   пользователь: исследователя     # кого бот слушает, в родительном падеже
#   область: теме этой базы знаний   # предмет базы, в дательном падеже после «по»
#   по_теме: о предмете базы или о методах его изучения
#   отказ_не_по_теме: Вопрос не относится к теме этой базы знаний.
"""


def _db(args: Any) -> Any:
    from .index import db

    return db.connect(args.dsn)


def _title(path: str) -> str:
    return kb.get(path).title


# --------------------------------------------------------------------------- #
#  list
# --------------------------------------------------------------------------- #
def nodes(conn: Any = None) -> dict[str, Any]:
    """Дерево баз для показа (kb list, страница, Телеграм): узлы из базы и из папок
    настроек, по порядку пути, с названием и числом статей. conn=None — только папки."""
    from .index import db

    rows = {r["path"]: r for r in db.kb_nodes(conn)} if conn is not None else {}
    unplaced = db.unplaced(conn) if conn is not None else None
    configured = set(kb.configured())
    out = []
    for path in sorted(set(rows) | configured):
        row = rows.get(path)
        out.append(
            {
                "path": path,
                "title": _title(path),
                "depth": path.count("."),
                "own": row["own"] if row else None,
                "subtree": row["subtree"] if row else None,
                "configured": path in configured,
                "in_db": row is not None,
            }
        )
    return {"default": kb.default_path(), "unplaced": unplaced, "nodes": out}


def cmd_list(args: Any) -> int:
    from .index import db

    tree = None
    try:
        with _db(args) as conn:
            db.init_db(conn, with_hnsw=False)
            tree = nodes(conn)
    except Exception as exc:  # noqa: BLE001 — без базы покажем хотя бы папки
        print(f"База недоступна ({type(exc).__name__}) — только папки настроек.\n")
        tree = nodes()
    if not tree["nodes"]:
        print("Баз пока нет: python kbrag.py kb create <путь> --title «название»")
        return 0
    width = max(len(n["path"].split(".")[-1]) + 2 * n["depth"] for n in tree["nodes"]) + 2
    print(f"{'база':<{width}}{'название':<28}{'своих':>7}{'всего':>7}  ")
    for n in tree["nodes"]:
        name = f"{'  ' * n['depth']}{n['path'].split('.')[-1]}"
        own = "—" if n["own"] is None else str(n["own"])
        total = "—" if n["subtree"] is None else str(n["subtree"])
        notes = []
        if n["path"] == tree["default"]:
            notes.append("по умолчанию")
        if not n["configured"]:
            notes.append("нет папки настроек")
        elif not n["in_db"] and tree["unplaced"] is not None:
            notes.append("в базе ещё нет")
        line = f"{name:<{width}}{n['title'][:26]:<28}{own:>7}{total:>7}"
        print(line + (f"  ← {', '.join(notes)}" if notes else ""))
    if tree["unplaced"]:
        print(f"\nСтатей без ветки: {tree['unplaced']} — их видит только всё дерево (--kb .).")
    print("\nСвоих — статей в самой базе, всего — вместе с базами под ней.")
    return 0


# --------------------------------------------------------------------------- #
#  create
# --------------------------------------------------------------------------- #
def cmd_create(args: Any) -> int:
    from .index import db

    path = kb.check_path(args.path)
    if not path:
        print("Корень создавать не нужно — он есть всегда.", file=sys.stderr)
        return 1
    title = (args.title or path.split(".")[-1]).strip()
    folder = kb.folder(path)
    settings = folder / kb.SETTINGS
    if settings.exists():
        print(f"Папка настроек уже есть: {settings}")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        settings.write_text(TEMPLATE.format(path=path, title=title), encoding="utf-8")
        print(f"Настройки базы: {settings}")
    try:
        with _db(args) as conn:
            db.init_db(conn, with_hnsw=False)
            db.ensure_kb(conn, path, title)
    except Exception as exc:  # noqa: BLE001
        print(f"В базу не записано ({type(exc).__name__}: {exc}) — запишется при import.")
        return 0
    parent = path.rsplit(".", 1)[0] if "." in path else ""
    print(
        f"База «{title}» ({path}) создана"
        + (f", настройки — от «{parent}» и выше." if parent else ".")
        + f"\nСтатьи в неё: python kbrag.py import <папка> --kb {path}"
    )
    return 0


# --------------------------------------------------------------------------- #
#  move
# --------------------------------------------------------------------------- #
def _move_folder(old: Path, new: Path) -> str:
    if not old.exists():
        return ""
    if new.exists():
        raise ValueError(f"папка {new} уже есть")
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(new))
    return f"{old} → {new}"


def _retarget_default(old: str, new: str) -> bool:
    """База по умолчанию была в переносимом поддереве — строка в корневом kb.yaml
    правится на месте (комментарии остаются)."""
    current = kb.default_path()
    if not (current == old or current.startswith(old + ".")):
        return False
    root = kb.CONFIG_DIR / kb.SETTINGS
    text = root.read_text(encoding="utf-8-sig")
    moved = new + current[len(old) :]
    # Значение — до конца строки или до комментария после него.
    fixed = re.sub(
        rf"(?m)^({re.escape(kb.DEFAULT_KEY)}:\s*){re.escape(current)}(?=\s*(?:#.*)?$)",
        rf"\g<1>{moved}",
        text,
    )
    if fixed == text:
        return False
    root.write_text(fixed, encoding="utf-8")
    return True


def cmd_move(args: Any) -> int:
    from .index import db

    old, new = kb.check_path(args.path), kb.check_path(args.to)
    pairs = [
        (kb.folder(old), kb.folder(new)),
        (kb.LOCAL_DIR.joinpath(*old.split(".")), kb.LOCAL_DIR.joinpath(*new.split("."))),
    ]
    try:
        # Папки назначения проверяются до базы: иначе база перенесла бы узел, а папка — нет.
        for source, target in pairs:
            if source.exists() and target.exists():
                raise ValueError(f"папка {target} уже есть")
        with _db(args) as conn:
            nodes = db.move_kb(conn, old, new)
            moved = [_move_folder(source, target) for source, target in pairs]
    except ValueError as exc:
        print(f"Не перенесено: {exc}", file=sys.stderr)
        return 1
    kb._CACHE.clear()
    print(f"Перенесено узлов: {nodes} ({old} → {new}); статьи — вместе с ними.")
    for line in filter(None, moved):
        print(f"  папка: {line}")
    if _retarget_default(old, new):
        print(f"  база по умолчанию в config/kbs/kb.yaml — теперь в «{new}»")
    if new.split(".")[:-1] != old.split(".")[:-1]:
        print("Настройки теперь наследуются от новой ветки — проверьте: python kbrag.py kb list")
    return 0


# --------------------------------------------------------------------------- #
#  assign / unassign
# --------------------------------------------------------------------------- #
def _doc_ids(conn: Any, args: Any) -> list[str]:
    from .index import db

    ids = list(args.docs)
    if args.source is not None:
        source = kb.check_path(args.source)
        ids += [d["doc_id"] for d in db.documents(conn, limit=10_000_000, kb=source)]
    if args.unplaced:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT d.doc_id FROM documents d WHERE NOT EXISTS "
                "(SELECT 1 FROM document_kbs dk WHERE dk.doc_id = d.doc_id)"
            )
            ids += [r[0] for r in cur.fetchall()]
    return list(dict.fromkeys(ids))


def cmd_assign(args: Any, remove: bool = False) -> int:
    from .index import db

    path = kb.check_path(args.path)
    if not path:
        print("Корень — не ветка: в него статьи не кладутся.", file=sys.stderr)
        return 1
    with _db(args) as conn:
        ids = _doc_ids(conn, args)
        if not ids:
            print(
                "Каких статей? doc_id через пробел, --from <база> или --unplaced", file=sys.stderr
            )
            return 1
        with conn.cursor() as cur:
            cur.execute("SELECT doc_id FROM documents WHERE doc_id = ANY(%s)", (ids,))
            known = {r[0] for r in cur.fetchall()}
        done = db.unassign(conn, ids, path) if remove else db.assign(conn, ids, path)
    missing = [i for i in ids if i not in known]
    verb = "убрано из" if remove else "положено в"
    print(
        f"Статей {verb} «{path}»: {done} из {len(ids)}"
        + ("" if remove else " (уже лежавшие там не считаются)")
    )
    if missing:
        print(f"Нет в базе: {', '.join(missing[:10])}" + (" …" if len(missing) > 10 else ""))
    return 0


# --------------------------------------------------------------------------- #
#  remove
# --------------------------------------------------------------------------- #
def cmd_remove(args: Any) -> int:
    from .index import db

    path = kb.check_path(args.path)
    if not path:
        print("Корень не убирается.", file=sys.stderr)
        return 1
    folders = [kb.folder(path), kb.LOCAL_DIR.joinpath(*path.split("."))]
    with _db(args) as conn:
        nodes = [
            r for r in db.kb_nodes(conn) if r["path"] == path or r["path"].startswith(path + ".")
        ]
        if not args.yes:
            total = max((r["subtree"] for r in nodes if r["path"] == path), default=0)
            print(
                f"Уберётся узлов: {len(nodes)} ({path} и всё под ним); статей в поддереве: "
                f"{total}. Статьи из других веток останутся там; лежащие только здесь — "
                + (
                    "будут удалены из базы с фрагментами и фактами."
                    if args.docs
                    else "останутся без ветки (их видит только --kb .)."
                )
            )
            if args.config:
                print(
                    "Папки настроек и данных удалятся: "
                    + ", ".join(str(f) for f in folders if f.exists())
                )
            else:
                print(f"Папка настроек остаётся: {folders[0]} (удалить — ключ --config).")
            print("Убрать — та же команда с --yes")
            return 0
        if path == kb.default_path() or kb.default_path().startswith(path + "."):
            print(
                "Это база по умолчанию (config/kbs/kb.yaml) — сначала поменяйте её там.",
                file=sys.stderr,
            )
            return 1
        stats = db.remove_kb(conn, path, drop_orphans=args.docs)
    if args.config:
        for f in folders:
            if f.exists():
                shutil.rmtree(f)
    kb._CACHE.clear()
    print(
        f"Убрано узлов: {stats['nodes']}. Статей, лежавших только здесь: {stats['orphans']}"
        + (f" — удалено из базы {stats['deleted']}." if args.docs else " — теперь без ветки.")
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python kbrag.py kb", description="Дерево баз")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="дерево баз и сколько в них статей")
    p = sub.add_parser("create", help="новая база: папка настроек и узел в базе")
    p.add_argument("path")
    p.add_argument("--title", default="", help="название (иначе — последняя метка пути)")
    p = sub.add_parser("move", help="перенести базу со всем, что под ней")
    p.add_argument("path")
    p.add_argument("to")
    for name, text in (("assign", "положить статьи в базу"), ("unassign", "убрать статьи из базы")):
        p = sub.add_parser(name, help=text)
        p.add_argument("path")
        p.add_argument("docs", nargs="*", help="doc_id статей")
        p.add_argument("--from", dest="source", default=None, help="все статьи этой базы")
        p.add_argument("--unplaced", action="store_true", help="все статьи без ветки")
    p = sub.add_parser("remove", help="убрать базу (без --yes — только показать)")
    p.add_argument("path")
    p.add_argument("--yes", action="store_true", help="убрать на самом деле")
    p.add_argument("--docs", action="store_true", help="и удалить статьи, что лежат только здесь")
    p.add_argument("--config", action="store_true", help="и удалить папки настроек и данных")
    for p in sub.choices.values():
        add_db_args(p)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            return cmd_list(args)
        if args.command == "create":
            return cmd_create(args)
        if args.command == "move":
            return cmd_move(args)
        if args.command in ("assign", "unassign"):
            return cmd_assign(args, remove=args.command == "unassign")
        return cmd_remove(args)
    except kb.KBError as exc:
        print(f"База: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
