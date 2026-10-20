"""Поддерево базы в запросах к Postgres и кэш графа на узел — без настоящей базы.

Подставной курсор записывает запросы: у узла в каждом запросе должно быть условие
по document_kbs, у корня — не должно, а параметры — сходиться с подстановками."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fakedb import FakeConn

from kbrag.graph import api, store
from kbrag.graph.synonyms import Synonyms
from kbrag.index import db, search

CALLS: dict[str, Callable[[Any, str], Any]] = {
    "db.stats": lambda c, kb: db.stats(c, kb),
    "db.documents": lambda c, kb: db.documents(c, kb=kb),
    "search.vector_search": lambda c, kb: search.vector_search(c, [0.1] * 4, 5, kb=kb),
    "search.text_search": lambda c, kb: search.text_search(c, "рудный узел", 5, kb=kb),
    "store.todo": lambda c, kb: store.todo(c, 10, kb),
    "store.unconfirmed": lambda c, kb: store.unconfirmed(c, 10, kb),
    "store.clear": lambda c, kb: store.clear(c, kb),
    "store.all_facts": lambda c, kb: store.all_facts(c, kb=kb),
    "store.summary": lambda c, kb: store.summary(c, kb),
    "store.rejected_reasons": lambda c, kb: store.rejected_reasons(c, kb),
    "api._only_added": lambda c, kb: api._only_added(c, (1, 5, 0, 0), kb),
}


@pytest.mark.parametrize("name", sorted(CALLS))
@pytest.mark.parametrize(("kb", "filtered"), [("", False), ("geology.ore", True)])
def test_subtree_filter(name: str, kb: str, filtered: bool) -> None:
    conn = FakeConn()
    try:
        CALLS[name](conn, kb)
    except (TypeError, ValueError):
        pass  # курсор строк не отдаёт — проверяются только запросы
    sql = " ".join(q for q, _ in conn.log)
    assert conn.log
    assert ("document_kbs" in sql) is filtered
    if filtered:
        assert all(p.get("kb") == kb for q, p in conn.log if "document_kbs" in q)


@pytest.mark.parametrize(
    ("kb", "found", "exact"), [("", 10, False), ("a.b", 10, True), ("a.b", 10**6, False)]
)
def test_small_branch_searched_exactly(kb: str, found: int, exact: bool) -> None:
    """Маленькая ветка — перебором, без HNSW; большая и вся база — по индексу."""

    def answer(sql: str, params: Any) -> list[Any]:
        return [(min(found, params["exact_max"]),)] if "count(*)" in sql else []

    conn = FakeConn(answer=answer)
    search.vector_search(conn, [0.1] * 4, 5, kb=kb)
    sql = " ".join(q for q, _ in conn.log)
    assert ("count(*)" in sql) is bool(kb)  # у корня без фильтра и считать нечего
    assert ("<=> %(q)s::vector) + 0" in sql) is exact
    assert ("hnsw." in sql) is not exact


def test_assign_creates_ancestors() -> None:
    conn = FakeConn()
    db.assign(conn, ["d1"], "geology.ore.anabar")
    nodes = [p[0] for q, p in conn.log if "INSERT INTO kbs" in q]
    assert nodes == ["geology", "geology.ore", "geology.ore.anabar"]
    assert any("INSERT INTO document_kbs" in q for q, _ in conn.log)


def test_assign_unplaced_only_without_branch() -> None:
    conn = FakeConn()
    db.assign_unplaced(conn, "geology.ore.anabar")
    assert "NOT EXISTS" in conn.log[-1][0]
    assert db.assign_unplaced(FakeConn(), "") == 0  # корень — не ветка


class FakeGraph:
    built: list[tuple[str, int]] = []

    def __init__(self, facts: list[Any], syn: Any) -> None:
        FakeGraph.built.append(("new", len(facts)))

    def extended(self, facts: list[Any]) -> FakeGraph:
        FakeGraph.built.append(("ext", len(facts)))
        return self


def test_graph_cache_per_node(monkeypatch: pytest.MonkeyPatch) -> None:
    stamps: dict[str, tuple[int, ...]] = {
        "": (10, 100, 0, 0),
        "geology": (4, 90, 0, 0, 3, 777),
        "physics": (2, 100, 0, 0, 1, 555),
    }
    FakeGraph.built = []
    monkeypatch.setattr(api, "Graph", FakeGraph)
    monkeypatch.setattr(api.autosyn, "mapping", lambda conn: ({}, None))
    monkeypatch.setattr(api.autosyn, "extend", lambda syn, auto, stamp: syn)
    monkeypatch.setattr(api, "_stamp", lambda conn, kb="": stamps[kb])
    monkeypatch.setattr(api, "_only_added", lambda conn, old, kb="": True)
    monkeypatch.setattr(api, "_CACHE", {})
    syn, conn = Synonyms(sha="s"), FakeConn()

    everything = api.graph(conn, syn, "")
    geology = api.graph(conn, syn, "geology")
    physics = api.graph(conn, syn, "physics")
    assert len({id(everything), id(geology), id(physics)}) == 3
    assert api.graph(conn, syn, "geology") is geology  # из кэша
    assert len(FakeGraph.built) == 3

    stamps["geology"] = (6, 95, 0, 0, 3, 777)  # дописаны факты
    api.graph(conn, syn, "geology")
    assert FakeGraph.built[-1][0] == "ext"

    stamps["geology"] = (6, 95, 0, 0, 4, 888)  # статью положили в ветку
    api.graph(conn, syn, "geology")
    assert FakeGraph.built[-1][0] == "new"

    api.graph(conn, syn, "physics")  # другой узел не тронут
    assert len(FakeGraph.built) == 5

    api.graph(conn, Synonyms(sha="другой"), "physics")  # словарь поменялся
    assert len(FakeGraph.built) == 6


def test_tree_queries_shape() -> None:
    """Новые запросы дерева собираются и сходятся с параметрами."""
    conn = FakeConn()
    assert db.kb_nodes(conn) == []
    assert db.unplaced(conn) == 0
    db.unassign(conn, ["d1"], "geo.ore")
    assert db.move_kb(conn, "geo.ore", "geo.metals") == 0
    update = next(q for q, _ in conn.log if q.lstrip().startswith("UPDATE kbs"))
    assert "subpath" in update
    stats = db.remove_kb(conn, "geo.ore")
    assert stats == {"nodes": 0, "orphans": 0, "deleted": 0}


@pytest.mark.parametrize(("old", "new"), [("", "a"), ("a", ""), ("a", "a"), ("a", "a.b")])
def test_move_refuses(old: str, new: str) -> None:
    with pytest.raises(ValueError):
        db.move_kb(FakeConn(), old, new)
