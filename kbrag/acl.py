"""Кому какие базы: доступ людей к веткам дерева баз.

Файл .local/access.yaml (только для этой машины, в git не идёт; правится командой
python kbrag.py access или Блокнотом):

    люди:
      Анна:
        базы: [physics.optics, chemistry]   # ветки — каждая со всем, что под ней
        telegram: 123456789                  # id в Телеграме (бот пришлёт его сам)
        токен_sha256: 5e88…                  # вход на страницу и для MCP (access add)
      Борис:
        базы: ["."]                          # «.» — всё дерево
    гости: []        # что видит гость открытого Телеграм-бота («*» в telegram_users.txt)

Нет файла — как раньше: всё видно всем, кого пускают (страница — с этого компьютера,
бот — по списку telegram_users.txt). Есть файл — страница и MCP пускают только по
токену, а каждый человек видит только свои ветки: поиск, чат-бот, граф, статьи,
формулы, рисунки и само дерево баз. Владельцы бота (telegram_users.txt) и команды в
терминале видят всё: у кого терминал, у того и база.

Как это работает: доступ запроса — «область» (scope), список веток человека, или None —
без ограничений. Она ставится на запрос (страница, бот) и лежит в contextvar, а
db.in_kb дописывает её к каждому запросу о статьях: статья видна, если она лежит хотя
бы в одной ветке человека. Статья, не разложенная по веткам, видна только тем, у кого
доступ ко всему. Потоки, которые делают запросы за запрос (поиск по нескольким
запросам сразу, ответ бота), берут область с собой — carry.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import LOCAL

ACCESS_FILE = LOCAL / "access.yaml"

# Область доступа: ветки человека или None — без ограничений. Пустой кортеж — ничего.
Scope = tuple[str, ...] | None
_SCOPE: ContextVar[Scope] = ContextVar("kbrag_scope", default=None)


@dataclass(frozen=True)
class Person:
    name: str
    bases: Scope  # None — всё дерево
    telegram: frozenset[int]
    token_sha: str = ""


@dataclass(frozen=True)
class Access:
    people: tuple[Person, ...] = ()
    guests: Scope = None  # гость открытого бота; без файла — всё
    configured: bool = False  # файл есть: страница и MCP — только по токену

    def by_token(self, token: str | None) -> Person | None:
        if not token:
            return None
        digest = token_hash(token)
        return next(
            (p for p in self.people if p.token_sha and hmac.compare_digest(p.token_sha, digest)),
            None,
        )

    def by_telegram(self, user_id: Any) -> Person | None:
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            return None
        return next((p for p in self.people if uid in p.telegram), None)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.strip().encode("utf-8")).hexdigest()


def _bases(raw: Any) -> Scope:
    """Список веток из файла → область. «.» или пусто в списке — всё дерево."""
    from .kb import check_path

    items = raw if isinstance(raw, list) else ([] if raw is None else [raw])
    paths = []
    for item in items:
        path = check_path(str(item))  # «.» → «»
        if not path:
            return None
        paths.append(path)
    return tuple(sorted(set(paths)))


def _telegram(raw: Any) -> frozenset[int]:
    items = raw if isinstance(raw, list) else ([] if raw is None else [raw])
    return frozenset(int(x) for x in items if str(x).lstrip("-").isdigit())


_CACHE: dict[Path, tuple[float, Access]] = {}


def load(path: Path | None = None) -> Access:
    """Доступ из файла; файл перечитывается, если его поправили. Нет файла — Access()
    без ограничений. Ошибка в файле — AccessError: лучше не пустить, чем пустить всех."""
    import yaml

    path = path or ACCESS_FILE
    if not path.is_file():
        return Access()
    mtime = path.stat().st_mtime
    hit = _CACHE.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
        people = []
        for name, info in (data.get("люди") or {}).items():
            info = info or {}
            people.append(
                Person(
                    name=str(name),
                    bases=_bases(info.get("базы")) if "базы" in info else (),
                    telegram=_telegram(info.get("telegram")),
                    token_sha=str(info.get("токен_sha256") or ""),
                )
            )
        access = Access(tuple(people), _bases(data.get("гости") or []), configured=True)
    except Exception as exc:  # noqa: BLE001 — любой сбой файла: не пускать никого
        raise AccessError(f"{path}: {exc}") from exc
    _CACHE[path] = (mtime, access)
    return access


class AccessError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
#  Область запроса
# --------------------------------------------------------------------------- #
def current() -> Scope:
    return _SCOPE.get()


@contextmanager
def scoped(scope: Scope) -> Iterator[None]:
    token = _SCOPE.set(scope)
    try:
        yield
    finally:
        _SCOPE.reset(token)


def carry[T](fn: Callable[..., T]) -> Callable[..., T]:
    """fn с областью этого потока — для пула потоков и Thread: в новом потоке contextvar
    пуст, и запрос шёл бы без ограничений."""
    scope = current()

    def run(*args: Any, **kwargs: Any) -> T:
        with scoped(scope):
            return fn(*args, **kwargs)

    return run


def visible(path: str) -> bool:
    """visible_in для области этого запроса."""
    return visible_in(path, current())


def visible_in(path: str, scope: Scope) -> bool:
    """Можно ли работать с узлом path: он в ветке человека, под ней или над ней (тогда
    видна только часть поддерева — ветки человека). Корень «» — если хоть что-то есть."""
    if scope is None:
        return True
    if not path:
        return bool(scope)
    return any(
        path == own or path.startswith(own + ".") or own.startswith(path + ".") for own in scope
    )


def sql(doc: str = "d.doc_id") -> tuple[str, dict[str, Any]]:
    """Условие доступа к статье doc (с AND впереди) и его параметр — для db.in_kb."""
    scope = current()
    if scope is None:
        return "", {}
    if not scope:
        return " AND false", {}
    return (
        f" AND EXISTS (SELECT 1 FROM document_kbs da WHERE da.doc_id = {doc}"
        " AND da.path <@ ANY(%(acl)s::ltree[]))",
        {"acl": list(scope)},
    )


def cache_key(kb: str) -> tuple[str, Scope]:
    """Ключ кэша по базе и области: кэш одного человека не отдаётся другому."""
    return kb, current()


# --------------------------------------------------------------------------- #
#  Команда: python kbrag.py access list | add | remove | token
# --------------------------------------------------------------------------- #
HEADER = """# Кому какие базы — python kbrag.py access, описание — kbrag/acl.py.
# Ветка даёт доступ ко всему, что под ней; «.» — всё дерево. Токен хранится только
# отпечатком (sha256): потерян — python kbrag.py access token ИМЯ выдаст новый.
"""


def _read_raw(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        return {"люди": {}, "гости": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    data.setdefault("люди", {})
    data.setdefault("гости", [])
    return data


def _write_raw(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=None)
    path.write_text(HEADER + body, encoding="utf-8")
    _CACHE.pop(path, None)


def new_token() -> str:
    return secrets.token_urlsafe(24)


def main(argv: list[str] | None = None, path: Path | None = None) -> int:
    import argparse

    from .kb import KBError, check_path

    path = path or ACCESS_FILE
    p = argparse.ArgumentParser(prog="python kbrag.py access", description="Кому какие базы")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("list", help="кто к каким базам")
    add = sub.add_parser("add", help="человек и его базы (повтор — дописать)")
    add.add_argument("name")
    add.add_argument("--kb", action="append", default=[], help="ветка дерева; «.» — всё")
    add.add_argument("--telegram", type=int, action="append", default=[], help="id в Телеграме")
    add.add_argument("--no-token", action="store_true", help="без входа на страницу")
    rm = sub.add_parser("remove", help="убрать человека или его ветку (--kb)")
    rm.add_argument("name")
    rm.add_argument("--kb", action="append", default=[])
    tok = sub.add_parser("token", help="новый токен человеку (прежний перестаёт работать)")
    tok.add_argument("name")
    guests = sub.add_parser("guests", help="что видит гость открытого бота")
    guests.add_argument("--kb", action="append", default=[], help="ветка; без ключа — ничего")
    args = p.parse_args(argv or ["list"])

    data = _read_raw(path)
    people: dict[str, Any] = data["люди"]
    try:
        kbs = [check_path(x) or "." for x in getattr(args, "kb", [])]
    except KBError as exc:
        print(f"База: {exc}", file=sys.stderr)
        return 1

    if args.command in (None, "list"):
        if not path.is_file():
            print("Ограничений нет: файла .local/access.yaml нет — всё видно всем, кого пускают.")
            return 0
        for name, info in people.items():
            info = info or {}
            bases = ", ".join(map(str, info.get("базы") or [])) or "ничего"
            tg = ", ".join(map(str, info.get("telegram") or [])) if info.get("telegram") else "—"
            login = "есть" if info.get("токен_sha256") else "нет"
            print(f"{name}: базы {bases}; Телеграм {tg}; вход на страницу {login}")
        print(f"Гости открытого бота: {', '.join(map(str, data['гости'])) or 'ничего'}")
        return 0

    if args.command == "guests":
        data["гости"] = kbs
        _write_raw(path, data)
        print(f"Гости открытого бота видят: {', '.join(kbs) or 'ничего'}")
        return 0

    if args.command == "add":
        info = people.get(args.name) or {}
        info["базы"] = sorted(set(info.get("базы") or []) | set(kbs))
        if args.telegram:
            info["telegram"] = sorted(set(info.get("telegram") or []) | set(args.telegram))
        token = None
        if not args.no_token and not info.get("токен_sha256"):
            token = new_token()
            info["токен_sha256"] = token_hash(token)
        people[args.name] = info
        _write_raw(path, data)
        print(f"{args.name}: базы {', '.join(info['базы']) or 'пока никаких (--kb)'}")
        if token:
            print(f"Токен (показывается один раз): {token}")
            print(f"Вход на страницу: http://localhost:8000/?token={token}")
        return 0

    if args.name not in people:
        print(f"Нет такого: {args.name}", file=sys.stderr)
        return 1
    if args.command == "remove":
        if kbs:
            info = people[args.name]
            info["базы"] = [b for b in info.get("базы") or [] if b not in kbs]
            print(f"{args.name}: базы {', '.join(info['базы']) or 'ничего'}")
        else:
            people.pop(args.name)
            print(f"{args.name} убран(а): доступа больше нет.")
        _write_raw(path, data)
        return 0
    # token
    token = new_token()
    people[args.name]["токен_sha256"] = token_hash(token)
    _write_raw(path, data)
    print(f"Новый токен {args.name} (показывается один раз): {token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
