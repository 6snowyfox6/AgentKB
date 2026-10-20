"""Каталожные вопросы (chat/catalog.py): разбор, запрос к базе, ответ — без модели и базы."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fakedb import FakeConn

from kbrag.chat import answer as chat
from kbrag.chat import catalog
from kbrag.chat.plan import Part, Plan, parse_plan
from kbrag.chat.settings import Settings

FOUND = [
    {
        "doc_id": "a",
        "title": "Lineaments of the Rif, Morocco",
        "year": 2024,
        "authors": ["A. B"],
        "journal": None,
        "url": "",
        "doi": "10.1/a",
        "by_title": True,
        "hits": 9,
    },
    {
        "doc_id": "b",
        "title": "Atlas faults",
        "year": 2019,
        "authors": [],
        "journal": None,
        "url": "",
        "doi": "",
        "by_title": False,
        "hits": 3,
    },
]


def test_parse_catalog_from_plan() -> None:
    data = {
        "части": [{"вопрос": "Сколько статей про Марокко?", "запросы": ["Марокко"]}],
        "каталог": {
            "тема": "Марокко",
            "основы": ["Morocc", "Maroc", "Марокк", "Ma", "Morocc"],
            "год_от": "2015",
            "год_до": None,
            "автор": "null",
        },
    }
    plan = parse_plan(data, "?", Settings())
    q = plan.catalog
    assert q.stems == ["Morocc", "Maroc", "Марокк"]  # короткие и повторы — вон
    assert (q.topic, q.year_from, q.year_to, q.author) == ("Марокко", 2015, None, "")
    assert parse_plan({"части": [], "каталог": None}, "?", Settings()).catalog is None


def test_parse_cleans_model_output() -> None:
    """Основы строкой, слова «статьи», годы и фамилия автора среди основ — убираются."""
    q = catalog.parse(
        {
            "тема": "статьи 2020–2023 годов про Sentinel-1",
            "основы": "Sentinel-1, радар, 2020-2023, Morocco articles",
            "год_от": 2020,
            "год_до": "2023",
        }
    )
    assert q.stems == ["Sentinel-1", "радар", "Morocco"] and (q.year_from, q.year_to) == (
        2020,
        2023,
    )
    q = catalog.parse(
        {
            "тема": "статьи Иваситы",
            "основы": ["ивасита", "Iwasita", "работы Иваситы"],
            "автор": "Iwasita",
        }
    )
    assert q.stems == [] and q.author == "Iwasita"
    assert catalog._author_key("Iwasita") == "Iwas"  # сходится с Iwashita в базе
    assert catalog._author_key("Inoue") == "Inou"


def test_find_builds_valid_query() -> None:
    conn = FakeConn(rows=[("a", "T", 2020, ["X"], None, "", "", True, 4)])
    query = catalog.CatalogQuery(stems=["Morocc", "SAR"], year_from=2015, author="Inoue")
    found = catalog.find(conn, query, "geology.lineaments")
    sql, params = conn.log[-1]
    assert found[0]["by_title"] and found[0]["hits"] == 4
    assert "document_kbs" in sql and params["kb"] == "geology.lineaments"
    assert params["s0"] == "%Morocc%" and params["s1"] == r"\mSAR\M"  # короткое — целым словом
    assert params["author"] == r"\mInou" and params["min_chunks"] == catalog.MIN_CHUNKS
    assert catalog.author_pattern("Ivasita") == r"\mI[vw]as"


def test_describe_counts_and_lists() -> None:
    query = catalog.CatalogQuery(stems=["Morocc"], topic="Марокко")
    head, listing = catalog.describe(query, FOUND, "Линеаменты")
    assert head.startswith("**В базе «Линеаменты» 2 статьи о теме «Марокко».**")
    assert "в названии — у 1, только в тексте" in head.lower()
    assert listing.splitlines()[0].endswith("— в названии")
    assert "в тексте, кусков: 3" in listing.splitlines()[1]
    head, listing = catalog.describe(catalog.CatalogQuery(author="Inoue"), [], "Линеаменты")
    assert "0 статей с автором «Inoue»" in head and listing == ""


def test_catalog_question_skips_fragment_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(chat, "LOGS", tmp_path)
    plan = Plan(
        [Part("Сколько статей про Марокко?", [])],
        catalog=catalog.CatalogQuery(stems=["Morocc"], topic="Марокко"),
    )
    asked: list[Any] = []

    def find(conn: Any, query: Any, kb: str) -> list[dict[str, Any]]:
        asked.append((query.stems, kb))
        return FOUND

    def no_search(*a: Any, **k: Any) -> Any:
        raise AssertionError("каталожный вопрос не ищет по фрагментам")

    monkeypatch.setattr(chat, "retrieve", no_search)
    monkeypatch.setattr(chat, "_catalog_articles", find)
    events = list(
        chat.answer(
            None,
            None,
            "Сколько статей про Марокко?",
            [],
            Settings(kb="geology"),
            planner=lambda q, h, s: plan,
        )
    )
    done = events[-1]
    assert asked == [(["Morocc"], "geology")]
    assert done["mode"] == "каталог" and done["catalog"]["count"] == 2
    assert [a["doc_id"] for a in done["articles"]] == ["a", "b"]
    assert done["articles"][0]["note"] == "в названии"
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert "2 статьи" in text and "Atlas faults" in text  # список — и в тексте (Телеграм, MCP)
