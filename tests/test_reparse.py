"""Переразбор не теряет граф: фрагменты сопоставляются с прежними по тексту, а не по
номеру (index/db.py: match_chunks, replace_chunks)."""

from __future__ import annotations

from typing import Any

from fakedb import FakeConn

from kbrag.index import db


def row(ord_: int, text: str) -> dict[str, Any]:
    return {
        "doc_id": "d1",
        "ord": ord_,
        "text": text,
        "headings": [],
        "pages": [],
        "n_tokens": 1,
        "has_table": False,
        "embedding": "[0]",
    }


OLD = [(11, 0, "Введение."), (12, 1, "Методы."), (13, 2, "Выводы.")]  # (id, ord, текст)


def test_new_paragraph_at_start_keeps_every_chunk() -> None:
    """Абзац в начале: номера всех фрагментов сдвинулись, но тексты те же."""
    new = [row(0, "Новый абзац."), row(1, "Введение."), row(2, "Методы."), row(3, "Выводы.")]
    assert db.match_chunks(OLD, new) == [(None, False), (11, True), (12, True), (13, True)]


def test_edited_text_keeps_place_but_not_facts() -> None:
    new = [row(0, "Введение."), row(1, "Методы, исправлено."), row(2, "Выводы.")]
    assert db.match_chunks(OLD, new) == [(11, True), (12, False), (13, True)]


def test_marks_and_spaces_are_not_changes() -> None:
    old = [(21, 0, r"Плотность: \rho = 1")]
    assert db.match_chunks(old, [row(0, "Плотность:  $$\\rho = 1$$")]) == [(21, True)]


def test_duplicate_texts_go_to_nearest() -> None:
    old = [(1, 0, "Таблица 1."), (2, 5, "Таблица 1.")]
    new = [row(4, "Таблица 1."), row(9, "Таблица 1."), row(10, "Таблица 1.")]
    assert db.match_chunks(old, new) == [(2, True), (1, True), (None, False)]


def test_reorder_and_removal() -> None:
    new = [row(0, "Выводы."), row(1, "Введение.")]  # «Методы» убраны, порядок другой
    assert db.match_chunks(OLD, new) == [(13, True), (11, True)]


def facts_conn(old: list[tuple[int, int, str]]) -> FakeConn:
    def answer(sql: str, params: Any) -> list[Any]:
        if sql.startswith("SELECT id, ord, text FROM chunks"):
            return old
        if "to_regclass('facts')" in sql:
            return [(True,)]
        return []

    return FakeConn(answer=answer)


def test_replace_chunks_after_shift() -> None:
    conn = facts_conn(OLD)
    new = [row(0, "Новый абзац."), row(1, "Введение."), row(2, "Методы."), row(3, "Выводы.")]
    assert db.replace_chunks(conn, "d1", new) == 4
    assert conn.queries("UPDATE chunks SET ord = -1 - ord") == [("d1",)]  # номера освобождены
    moved = [(p["chunk_id"], p["ord"]) for p in conn.queries("UPDATE chunks SET ord = %(ord)s")]
    assert moved == [(11, 1), (12, 2), (13, 3)]
    assert [p["text"] for p in conn.queries("INSERT INTO chunks")] == ["Новый абзац."]
    assert conn.queries("DELETE FROM facts") == []  # граф статьи цел
    assert conn.queries("DELETE FROM chunks") == []


def test_replace_chunks_drops_only_changed_and_gone() -> None:
    conn = facts_conn(OLD)
    db.replace_chunks(conn, "d1", [row(0, "Введение."), row(1, "Методы, исправлено.")])
    assert conn.queries("DELETE FROM chunks WHERE id = ANY") == [([13],)]  # «Выводы» ушли
    assert conn.queries("DELETE FROM facts WHERE chunk_id") == [([12],)]  # текст другой


def test_first_load_inserts_without_renumbering() -> None:
    conn = facts_conn([])
    db.replace_chunks(conn, "d1", [row(0, "Введение.")])
    assert conn.queries("UPDATE chunks") == [] and len(conn.queries("INSERT INTO chunks")) == 1
