"""Статья для MCP — по страницам (ответ инструмента Claude Code не вмещает статью в 30
страниц), и потолки limit у поиска (web/server.py)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from kbrag import mcp_server
from kbrag.web import server

FRAGMENTS = [{"ord": i, "text": "x" * 2_000} for i in range(60)]  # ~120 тыс. знаков


@pytest.fixture
def article(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Обработчик /api/article без сети и базы: что он ответил — в sent."""
    from kbrag.library import objects

    @contextmanager
    def connect(dsn: Any) -> Any:
        yield None

    def chunks(conn: Any, doc_id: str, offset: int = 0, limit: int | None = None) -> Any:
        return FRAGMENTS[offset : None if limit is None else offset + limit]

    monkeypatch.setattr(server.db, "connect", connect)
    monkeypatch.setattr(server.db, "document", lambda conn, doc_id: {"doc_id": doc_id})
    monkeypatch.setattr(server.db, "document_chunks", chunks)
    monkeypatch.setattr(server.db, "chunk_count", lambda conn, doc_id: len(FRAGMENTS))
    monkeypatch.setattr(objects, "record", lambda doc_id: {})
    monkeypatch.setattr(objects, "figures", lambda doc_id: [])
    monkeypatch.setattr(objects, "formulas", lambda doc_id: [])
    sent: list[Any] = []
    handler = server.Handler.__new__(server.Handler)
    handler._json = lambda payload, code=200: sent.append(payload)  # type: ignore[method-assign]

    def ask(**params: str) -> dict[str, Any]:
        handler._article({k: [v] for k, v in {"doc_id": "d1", **params}.items()})
        result: dict[str, Any] = sent[-1]
        return result

    return ask


def test_default_page(article: Any) -> None:
    page = article(text="1")
    assert len(page["fragments"]) == server.ARTICLE_PAGE
    assert page["fragments_total"] == 60 and page["offset"] == 0
    assert page["next_offset"] == 20


def test_pages_cover_whole_article(article: Any) -> None:
    seen, offset = [], 0
    while offset is not None:
        page = article(text="1", offset=str(offset), limit="50")
        assert sum(len(f["text"]) for f in page["fragments"]) <= server.ARTICLE_CHARS
        seen += [f["ord"] for f in page["fragments"]]
        offset = page["next_offset"]
    assert seen == list(range(60))  # ни пропусков, ни повторов


def test_limit_capped_and_bad_values(article: Any) -> None:
    assert len(article(text="1", limit="10000")["fragments"]) <= server.ARTICLE_PAGE_MAX
    page = article(text="1", offset="-5", limit="abc")
    assert page["offset"] == 0 and len(page["fragments"]) == server.ARTICLE_PAGE
    assert "fragments" not in article()  # без text — только метаданные


def test_int_arg() -> None:
    q = {"limit": ["500"], "neg": ["-3"], "bad": ["x"]}
    assert server._int_arg(q, "limit", 10, 1, server.OBJECTS_MAX) == 50
    assert server._int_arg(q, "neg", 10, 1, 50) == 1
    assert server._int_arg(q, "bad", 10, 1, 50) == 10
    assert server._int_arg(q, "нет", 7, 1, 50) == 7


def test_fit_chars_keeps_first() -> None:
    big = [{"text": "y" * 100_000}, {"text": "z"}]
    assert server._fit_chars(big) == big[:1]  # один огромный фрагмент — всё равно отдаётся


def test_mcp_get_article_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(mcp_server, "get_json", lambda path, **p: calls.append((path, p)) or {})
    mcp_server.get_article("d1")
    mcp_server.get_article("d1", text=True, offset=40)
    assert calls == [
        ("/api/article", {"doc_id": "d1"}),
        ("/api/article", {"doc_id": "d1", "text": "1", "offset": 40, "limit": 20}),
    ]
