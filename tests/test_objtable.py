"""Объекты статей (index/objtable.py): чтение формул и рисунков одним кодом — запросы,
подстановки, ссылки из текста, рамка из JSON."""

from __future__ import annotations

from typing import Any

from fakedb import FakeConn

from kbrag.index import figures, formulas

FORMULA = (7, "d1", 2, "55", r"x = y", "x = y", 4, '{"l": 1}', "до", "после", "словами", 11)
FIGURE = (9, "d1", 1, "3", "d1.figures/fig-001.png", 2, None, "Рис. 3.", "", 400, 300, 12)
DOC = ("Статья", 2020, "https://x")


class Embedder:
    def encode_one(self, text: str) -> list[float]:
        return [0.5, 0.25]


def test_formula_by_id_and_bbox() -> None:
    conn = FakeConn(rows=[FORMULA + DOC])
    item = formulas.by_id(conn, 7)
    assert item is not None
    assert item["index"] == 2 and item["number"] == "55" and item["bbox"] == {"l": 1}
    assert item["title"] == "Статья"
    sql, params = conn.log[0]
    assert "FROM formulas f JOIN documents d ON d.doc_id = f.doc_id WHERE f.id = %s" in sql
    assert params == (7,)


def test_of_document_with_refs() -> None:
    def answer(sql: str, params: Any) -> list[Any]:
        if "FROM formula_refs" in sql:
            return [(31, 5, [4], "Из формулы (55) следует " * 30)]
        return [FORMULA + DOC]

    conn = FakeConn(answer=answer)
    [item] = formulas.of_document(conn, "d1", "(55)")
    assert conn.log[0][1] == ("d1", "55")  # номер — без скобок
    assert "AND f.number = %s ORDER BY f.idx" in conn.log[0][0]
    assert item["referenced_by"][0]["chunk_id"] == 31
    assert len(item["referenced_by"][0]["text"]) == 300
    assert conn.log[1][1] == (7,)


def test_figure_label_lookup() -> None:
    conn = FakeConn(answer=lambda sql, params: [] if "figure_refs" in sql else [FIGURE + DOC])
    [item] = figures.of_document(conn, "d1", " 3A ")
    assert conn.log[0][1] == ("d1", "3a") and "lower(g.label) = %s ORDER BY g.n" in conn.log[0][0]
    assert item["file"] == "d1.figures/fig-001.png" and item["referenced_by"] == []


def test_search_in_subtree() -> None:
    conn = FakeConn(rows=[FIGURE + DOC + (0.812345,)])
    [item] = figures.search(conn, Embedder(), "схема", kb="physics", limit=3)
    assert item["similarity"] == 0.8123
    sql, params = conn.log[0]
    assert "ORDER BY g.embedding <=> %(q)s::vector" in sql and "dk.path <@ %(kb)s::ltree" in sql
    assert params["limit"] == 3 and params["kb"] == "physics"


def test_in_chunks() -> None:
    conn = FakeConn(rows=[FIGURE + DOC + (12,), FIGURE + DOC + (40,)])
    found = figures.in_chunks(conn, [12, 40])
    assert sorted(found) == [12, 40]
    assert "JOIN figures g ON g.id = links.fid JOIN documents d" in conn.log[0][0]
    assert formulas.in_chunks(conn, []) == {}  # пусто — без запроса
    assert len(conn.log) == 1
