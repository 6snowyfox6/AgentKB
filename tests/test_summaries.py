"""Резюме статей и разделов (Д6) и поиск сверху вниз (Д7)."""

from __future__ import annotations

from typing import Any

import pytest
from fakedb import FakeConn

from kbrag.index import search, summaries
from kbrag.index.search import Hit

LONG = "Текст раздела о разломах и рудных телах. " * 30


def test_sections_of_chunks() -> None:
    rows = [
        (3, LONG, ["Методы", "Отбор проб"]),
        (0, "Коротко.", ["Благодарности"]),
        (1, LONG, ["Геология"]),
        (2, LONG, ["Геология", "Разломы"]),
        (4, LONG, ["Рисунки"]),  # описания рисунков — не раздел статьи
        (5, LONG, ["Геология"]),  # тот же раздел после таблицы
    ]
    parts = summaries.sections_of(rows)
    assert [(p["section"], p["ords"]) for p in parts] == [("Геология", [1, 2, 5]), ("Методы", [3])]


def test_source_hash_ignores_formula_marks() -> None:
    plain = summaries.source_hash([r"x = \rho  y"])
    assert plain == summaries.source_hash([r"$$x = \rho y$$"])
    assert plain != summaries.source_hash(["x = y"])


def test_clip() -> None:
    assert summaries._clip(["один два", "три"], 100) == "один два\nтри"
    assert summaries._clip(["слово " * 50], 20).endswith(" …")


class Client:
    model = "qwen3-14b"

    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []

    def chat_json(self, system: str, user: str, **kw: Any) -> dict[str, Any]:
        self.asked.append((system, user))
        return {"резюме": f"резюме  {len(self.asked)}"}


class Embedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] for _ in texts]

    def encode_one(self, text: str) -> list[float]:
        return [0.1]


class Store(FakeConn):
    """Таблица резюме в памяти: (вид, раздел) → (текст, отпечаток, фрагменты)."""

    def __init__(self) -> None:
        super().__init__(answer=self.reply)
        self.store: dict[tuple[str, str], tuple[str, str, list[int]]] = {}

    def reply(self, sql: str, params: Any) -> list[Any]:
        if sql.startswith("SELECT section, text, source_hash FROM summaries"):
            found = [(k[1], v[0], v[1], v[2]) for k, v in self.store.items() if k[0] == "section"]
            return [(s, t, h) for s, t, h, _ in sorted(found, key=lambda r: r[3][0])]
        if sql.startswith("DELETE FROM summaries"):
            self.store.pop(("section", params[2]), None)
        return []


@pytest.fixture
def doc(monkeypatch: pytest.MonkeyPatch) -> Store:
    conn = Store()
    rows = [(0, LONG, ["Введение"]), (1, LONG, ["Геология"]), (2, LONG, ["Геология"])]
    monkeypatch.setattr(summaries, "_chunks", lambda c, d: ("Анабарский щит", list(rows)))
    monkeypatch.setattr(
        summaries, "_known", lambda c, d: {(k[0], k[1]): v[1] for k, v in c.store.items()}
    )

    def save(
        c: Store,
        doc_id: str,
        kind: str,
        section: str,
        ords: list[int],
        text: str,
        digest: str,
        model: str,
        vector: list[float],
    ) -> None:
        c.store[(kind, section)] = (text, digest, ords)

    monkeypatch.setattr(summaries, "_save", save)
    return conn


def test_summarize_document(doc: Store) -> None:
    client = Client()
    assert summaries.summarize_document(doc, Embedder(), client, "d1") == 3  # 2 раздела и статья
    assert doc.store[("section", "Геология")][2] == [1, 2]
    assert doc.store[("article", "")][0] == "резюме 3"
    system, user = client.asked[-1]
    assert "«Анабарский щит»" in system
    assert user.startswith("Введение: резюме") and "Геология: резюме" in user

    # Текст тот же — модель не спрашивается; --redo — всё заново.
    assert summaries.summarize_document(doc, Embedder(), client, "d1") == 0
    assert len(client.asked) == 3
    assert summaries.summarize_document(doc, Embedder(), client, "d1", redo=True) == 3


def test_section_gone_after_reparse(doc: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    summaries.summarize_document(doc, Embedder(), Client(), "d1")
    monkeypatch.setattr(summaries, "_chunks", lambda c, d: ("T", [(0, LONG, ["Введение"])]))
    summaries.summarize_document(doc, Embedder(), Client(), "d1")
    assert ("section", "Геология") not in doc.store  # раздела больше нет — резюме убрано


def hit(chunk_id: int, doc_id: str, ord_: int) -> Hit:
    return Hit(chunk_id, doc_id, ord_, "текст", [], [], "T", 2020, None, "", [])


def test_top_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(summaries, "available", lambda conn, kb: True)
    calls: list[tuple[str, Any]] = []

    def found(conn: Any, vector: Any, kind: str, kb: str, limit: int, docs: Any = None) -> Any:
        calls.append((kind, docs))
        if kind == summaries.ARTICLE:
            return [{"doc_id": "a"}, {"doc_id": "b"}]
        return [{"doc_id": "b", "ords": [7]}]

    monkeypatch.setattr(summaries, "search_summaries", found)

    def hybrid(conn: Any, embedder: Any, query: str, **kw: Any) -> list[Hit]:
        assert kw["docs"] == ["a", "b"] and kw["kb"] == "geo"
        return [hit(1, "a", 0), hit(2, "b", 7), hit(3, "b", 8)]

    monkeypatch.setattr(search, "hybrid_search", hybrid)
    hits = summaries.top_down(None, Embedder(), "вопрос", "geo")
    assert [h.chunk_id for h in hits] == [2, 1, 3]  # фрагмент найденного раздела — первым
    assert calls == [("article", None), ("section", ["a", "b"])]


def test_top_down_without_summaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(summaries, "available", lambda conn, kb: False)
    assert summaries.top_down(None, Embedder(), "вопрос") == []


def test_overview_question() -> None:
    assert summaries.is_overview("Сделай обзор методов датирования")
    assert summaries.is_overview("Какие работы есть о кимберлитах?")
    assert summaries.is_overview("Что известно об Анабарском щите в целом?")
    assert not summaries.is_overview("Какой возраст гранитов Далдынского массива?")


def test_overview_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(summaries, "available", lambda conn, kb: True)
    row = {
        "doc_id": "d1",
        "section": "",
        "ords": [],
        "text": "Статья о щите.",
        "title": "Щит",
        "year": 2019,
        "journal": None,
        "url": "",
        "authors": ["А", "Б", "В", "Г"],
        "similarity": 0.81234,
    }
    monkeypatch.setattr(summaries, "search_summaries", lambda *a, **k: [row])
    [src] = summaries.overview_sources(None, Embedder(), "обзор", start=2)
    assert src["n"] == 2 and src["kind"] == "резюме" and src["headings"] == ["Резюме статьи"]
    assert src["chunk_id"] is None and src["authors"] == ["А", "Б", "В"]
    assert src["similarity"] == 0.8123


def test_chat_skips_on_fake_connection() -> None:
    from kbrag.chat import answer, retrieval

    assert retrieval.top_down_candidates(object(), Embedder(), "вопрос") == []
    assert answer._overview_sources(object(), Embedder(), "обзор", "") == []
