"""Доступ людей к базам (kbrag/acl.py): кто к каким веткам, фильтр в запросах, потоки,
вход на страницу, Телеграм и команда access."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fakedb import FakeConn

from kbrag import acl
from kbrag.index import db

ACCESS = """
люди:
  Анна:
    базы: [physics.optics, chemistry]
    telegram: 111
    токен_sha256: {anna}
  Борис:
    базы: ["."]
    telegram: [222, 333]
  Вера:
    telegram: 444
гости: [physics.optics]
"""


@pytest.fixture
def access_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "access.yaml"
    path.write_text(ACCESS.format(anna=acl.token_hash("секрет-анны")), encoding="utf-8")
    monkeypatch.setattr(acl, "ACCESS_FILE", path)
    acl._CACHE.clear()
    return path


# --------------------------------------------------------------------------- #
#  Файл прав
# --------------------------------------------------------------------------- #
def test_load(access_file: Path) -> None:
    access = acl.load()
    assert access.configured
    anna, boris, vera = access.people
    assert anna.bases == ("chemistry", "physics.optics") and anna.telegram == {111}
    assert boris.bases is None  # «.» — всё дерево
    assert vera.bases == ()  # баз не дали — ничего
    assert access.guests == ("physics.optics",)
    assert access.by_token("секрет-анны") is anna and access.by_token("чужой") is None
    assert access.by_token("") is None and access.by_telegram("333") is boris


def test_no_file_means_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acl, "ACCESS_FILE", tmp_path / "нет.yaml")
    access = acl.load()
    assert not access.configured and access.guests is None


def test_broken_file_lets_nobody_in(access_file: Path) -> None:
    access_file.write_text("люди: [незакрыто", encoding="utf-8")
    acl._CACHE.clear()
    with pytest.raises(acl.AccessError):
        acl.load()


def test_visible_in() -> None:
    scope = ("physics.optics",)
    assert acl.visible_in("physics.optics", scope)
    assert acl.visible_in("physics.optics.lasers", scope)  # под веткой
    assert acl.visible_in("physics", scope)  # над веткой: видна её часть
    assert acl.visible_in("", scope)
    assert not acl.visible_in("physics.hadron", scope)
    assert not acl.visible_in("physics.opticsx", scope)  # не префикс строки, а путь
    assert not acl.visible_in("", ()) and acl.visible_in("что-угодно", None)


# --------------------------------------------------------------------------- #
#  Фильтр в запросах
# --------------------------------------------------------------------------- #
def test_in_kb_adds_rights() -> None:
    assert db.in_kb("physics") == (
        " AND EXISTS (SELECT 1 FROM document_kbs dk WHERE dk.doc_id = d.doc_id"
        " AND dk.path <@ %(kb)s::ltree)",
        {"kb": "physics"},
    )
    with acl.scoped(("physics.optics",)):
        where, params = db.in_kb("physics")
        assert "da.path <@ ANY(%(acl)s::ltree[])" in where and params["acl"] == ["physics.optics"]
        root, root_params = db.in_kb("")  # всё дерево — тоже только свои ветки
        assert "ANY(%(acl)s" in root and root_params == {"acl": ["physics.optics"]}
    with acl.scoped(()):
        assert db.in_kb("") == (" AND false", {})


def test_queries_carry_rights() -> None:
    def answer(sql: str, params: Any) -> list[Any]:
        if "FROM kbs" in sql:
            return []
        if "min(year)" in sql or "count(c.embedding)" in sql:
            return [(None, None)]
        return [(0,)]

    conn = FakeConn(answer=answer)
    with acl.scoped(("chemistry",)):
        db.stats(conn, "")
        db.kb_nodes(conn)
        assert db.unplaced(conn) == 0  # без ветки — только тем, кому видно всё
    assert all("%(acl)s" in sql for sql, _ in conn.log)  # подстановки сверяет fakedb


def test_visible_doc() -> None:
    assert db.visible_doc(FakeConn(), "d1")  # без ограничений — без запроса
    with acl.scoped(("chemistry",)):
        assert not db.visible_doc(FakeConn(rows=[]), "d1")
        assert db.visible_doc(FakeConn(rows=[(1,)]), "d1")


def test_carry_into_threads() -> None:
    seen: list[Any] = []
    with acl.scoped(("chemistry",)):
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(acl.carry(lambda _: seen.append(acl.current())), range(3)))
            list(pool.map(lambda _: seen.append(acl.current()), range(1)))  # без carry
        thread = threading.Thread(target=acl.carry(lambda: seen.append(acl.current())))
        thread.start()
        thread.join()
    assert seen == [("chemistry",)] * 3 + [None] + [("chemistry",)]


def test_search_all_keeps_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag.chat import search

    seen: list[Any] = []

    @contextmanager
    def sibling(conn: Any) -> Any:
        yield conn

    def hybrid(conn: Any, embedder: Any, query: str, **kw: Any) -> list[Any]:
        seen.append(acl.current())
        return []

    class Embedder:
        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

    monkeypatch.setattr(search.db, "is_real", lambda conn: True)
    monkeypatch.setattr(search.db, "sibling", sibling)
    monkeypatch.setattr(search, "hybrid_search", hybrid)
    with acl.scoped(("chemistry",)):
        search.search_all(object(), Embedder(), ["a", "b", "c"])
    assert seen == [("chemistry",)] * 3


def test_cache_key_differs_by_scope() -> None:
    with acl.scoped(("chemistry",)):
        anna = acl.cache_key("")
    assert anna != acl.cache_key("") == ("", None)


# --------------------------------------------------------------------------- #
#  Страница
# --------------------------------------------------------------------------- #
class Headers(dict[str, str]):
    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return super().get(key, default)


def handler(headers: dict[str, str] | None = None) -> Any:
    from kbrag.web import server

    h = server.Handler.__new__(server.Handler)
    h.headers = Headers(headers or {})  # type: ignore[assignment]
    h.sent = []  # type: ignore[attr-defined]
    h._json = lambda payload, code=200: h.sent.append((code, payload))  # type: ignore[method-assign]
    return h


def test_page_needs_token(access_file: Path) -> None:
    assert handler()._scope({}) == (False, ())
    assert handler({"X-KBRAG-Token": "секрет-анны"})._scope({}) == (
        True,
        ("chemistry", "physics.optics"),
    )
    assert handler({"Cookie": "kbrag_token=секрет-анны"})._scope({})[0] is False  # не ASCII
    good = acl.new_token()
    access_file.write_text(ACCESS.format(anna=acl.token_hash(good)), encoding="utf-8")
    acl._CACHE.clear()
    assert handler({"Cookie": f"kbrag_token={good}"})._scope({})[0]
    assert handler()._scope({"token": [good]})[0]


def test_page_open_without_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acl, "ACCESS_FILE", tmp_path / "нет.yaml")
    assert handler()._scope({}) == (True, None)


def test_foreign_kb_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag.kb import KBError
    from kbrag.web import server

    monkeypatch.setattr(server.State.chat_settings, "kb", "geology")
    h = handler()
    with acl.scoped(("physics.optics",)):
        assert h._kb_for(None) == ""  # база сервера чужая — свои ветки
        assert h._kb_for("physics") == "physics"
        with pytest.raises(KBError, match="нет доступа"):
            h._kb_for("geology.ore")


def test_hidden_document(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag.web import server

    @contextmanager
    def connect(dsn: Any) -> Any:
        yield None

    monkeypatch.setattr(server.db, "connect", connect)
    monkeypatch.setattr(server.db, "doc_of_chunk", lambda conn, cid: "чужая")
    monkeypatch.setattr(server.db, "visible_doc", lambda conn, doc_id: doc_id == "своя")
    h = handler()
    assert not h._hidden("/api/article", {"doc_id": ["своя"]})  # без ограничений — не смотрит
    with acl.scoped(("chemistry",)):
        assert not h._hidden("/api/article", {"doc_id": ["своя"]})
        assert h._hidden("/api/article", {"doc_id": ["чужая"]})
        assert h._hidden("/api/fragment", {"chunk_id": ["5"]})
    assert h.sent == [(404, {"error": "нет такой статьи"})] * 2  # как будто её нет


# --------------------------------------------------------------------------- #
#  Телеграм
# --------------------------------------------------------------------------- #
@pytest.fixture
def bot(access_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from kbrag.chat.settings import Settings
    from kbrag.tg import bot as tgbot

    users = tmp_path / "users.txt"
    users.write_text("*\n42\n", encoding="utf-8")
    monkeypatch.delenv("KBRAG_TG_USERS", raising=False)

    class Telegram:
        def __init__(self) -> None:
            self.sent: list[str] = []

        def send(self, chat_id: int, text: str, markup: Any = None) -> None:
            self.sent.append(text)

    return tgbot.Bot(
        Telegram(),  # type: ignore[arg-type]
        None,
        None,
        Settings(kb="geology"),
        users_file=users,
        log=lambda _: None,
        kb_file=tmp_path / "kb.json",
    )


def test_bot_scopes(bot: Any) -> None:
    assert bot.scope_for(42) is None  # владелец бота — всё
    assert bot.scope_for(111) == ("chemistry", "physics.optics")
    assert bot.scope_for(222) is None
    assert bot.scope_for(999) == ("physics.optics",)  # гость открытого бота
    assert not bot.guest(111) and bot.guest(999) and not bot.guest(42)


def test_bot_kb_choice_respects_rights(bot: Any) -> None:
    session = bot.session(111)
    with acl.scoped(bot.scope_for(111)):
        assert bot.kb_of(111) == ""  # база бота (geology) чужая — свои ветки
        assert bot.set_kb(111, session, "geology") == "Нет доступа"
        assert bot.kb_of(111) == ""
        bot.set_kb(111, session, "chemistry")
        assert bot.kb_of(111) == "chemistry"


def test_closed_bot_lets_people_in(bot: Any) -> None:
    bot.users_file.write_text("42\n", encoding="utf-8")  # без «*»
    assert bot.allowed(42) and bot.allowed(111) and not bot.allowed(999)


# --------------------------------------------------------------------------- #
#  Команда access
# --------------------------------------------------------------------------- #
def test_access_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "access.yaml"
    assert acl.main(["add", "Анна", "--kb", "physics.optics", "--telegram", "111"], path) == 0
    token = capsys.readouterr().out.split("Токен (показывается один раз): ")[1].split()[0]
    text = path.read_text(encoding="utf-8")
    assert token not in text and acl.token_hash(token) in text  # только отпечаток
    acl.main(["add", "Анна", "--kb", "chemistry"], path)
    assert "Токен" not in capsys.readouterr().out  # второй раз — тот же токен
    person = acl.load(path).by_token(token)
    assert person is not None and person.bases == ("chemistry", "physics.optics")
    acl.main(["remove", "Анна", "--kb", "chemistry"], path)
    acl.main(["token", "Анна"], path)
    assert acl.load(path).by_token(token) is None  # старый больше не работает
    acl.main(["guests", "--kb", "."], path)
    assert acl.load(path).guests is None
    assert acl.main(["remove", "Борис"], path) == 1
    acl.main(["remove", "Анна"], path)
    assert acl.load(path).people == ()


def test_get_without_token(access_file: Path) -> None:
    h = handler()
    h.path = "/api/stats"
    h.do_GET()
    assert h.sent[0][0] == 401 and h.sent[0][1]["login"] is True
    pages: list[Any] = []
    h._send = lambda code, body, kind: pages.append(code)  # type: ignore[method-assign]
    h.path = "/"
    h.do_GET()
    assert pages == [200]  # страница — без входа: на ней окно входа, данных нет


def test_tree_shows_own_branches(tree: Any) -> None:
    from kbrag.kbtree import nodes

    for path in ("physics.optics", "physics.hadron", "geology"):
        tree(path, "kb.yaml", f"название: {path}\n")
    every = [n["path"] for n in nodes()["nodes"]]
    with acl.scoped(("physics.optics",)):
        own = [n["path"] for n in nodes()["nodes"]]
    assert "geology" in every and "physics.hadron" in every
    assert "physics.optics" in own and "geology" not in own and "physics.hadron" not in own
