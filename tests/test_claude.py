"""Для Claude Code: вывод --json (ask) и MCP-сервер поверх serve (mcp_server.py).

Сервер базы подменён: проверяется, что MCP спрашивает у него и что отдаёт клиенту."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kbrag import mcp_server
from kbrag.chat.cli import collect
from kbrag.library import objects


def test_ask_collect(capsys: Any) -> None:
    events = [
        {"type": "status", "text": "ищу в базе…"},
        {
            "type": "sources",
            "sources": [{"n": 1, "doc_id": "d1", "chunk_id": 5, "title": "Статья", "text": "т"}],
            "queries": ["золото"],
            "reranked": True,
        },
        {"type": "token", "text": "Золото [1]. "},
        {"type": "token", "text": "Без ссылки."},
        {"type": "revised", "text": "Золото [1].", "removed": ["Без ссылки."]},
        {
            "type": "done",
            "mode": "база",
            "used": [1],
            "coverage": [1, 1],
            "seconds": 2.0,
            "timings": {"answer": 1.0},
        },
    ]
    out = collect(iter(events), "geo")
    assert out["answer"] == "Золото [1]." and out["removed"] == ["Без ссылки."]
    assert out["sources"][0]["chunk_id"] == 5 and out["used"] == [1] and out["kb"] == "geo"
    assert out["search"]["reranked"] is True and out["timings"] == {"answer": 1.0}
    assert capsys.readouterr().err.strip() == "ищу в базе…"  # статус — в stderr, не в JSON


# --------------------------------------------------------------------------- #
#  Формулы и рисунки из папки статей
# --------------------------------------------------------------------------- #
def write_docling(folder: Path, doc_id: str) -> None:
    doc = {
        "texts": [
            {"label": "text", "text": "Плотность пород выражается через пористость:"},
            {
                "label": "formula",
                "text": r"\rho = \rho_0 (1 - \varphi) \quad (1)",
                "orig": "ρ = ρ0 (1 − φ) (1)",
                "prov": [{"page_no": 3, "bbox": {"l": 1, "t": 2, "r": 3, "b": 4}}],
            },
            {"label": "text", "text": "где φ — пористость."},
            {"label": "formula", "text": r"x = y", "orig": "x = y (2)", "prov": [{"page_no": 4}]},
        ]
    }
    (folder / f"{doc_id}.docling.json").write_text(json.dumps(doc), encoding="utf-8")


def test_formulas_from_docling(tmp_path: Path) -> None:
    write_docling(tmp_path, "d1")
    found = objects.formulas("d1", tmp_path)
    assert [f["number"] for f in found] == ["1", "2"]  # второй — из текста PDF
    first = found[0]
    assert first["page"] == 3 and first["before"].startswith("Плотность")
    assert first["after"] == "где φ — пористость."
    assert found[1]["page"] == 4
    assert objects.formulas("нет", tmp_path) == []


def test_figure_file_stays_inside(tmp_path: Path) -> None:
    (tmp_path / "d1.figures").mkdir()
    (tmp_path / "d1.figures" / "fig-001.png").write_bytes(b"png")
    record = {
        "doc_id": "d1",
        "figures": [
            {"n": 1, "file": "d1.figures/fig-001.png", "page": 2, "caption": "Схема"},
            {"n": 2, "file": "../../секрет.txt"},  # мимо папки статей — не отдаётся
        ],
    }
    (tmp_path / "d1.json").write_text(json.dumps(record), encoding="utf-8")
    assert (
        objects.figure_file("d1", 1, tmp_path)
        == (tmp_path / "d1.figures" / "fig-001.png").resolve()
    )
    assert objects.figure_file("d1", 2, tmp_path) is None
    assert objects.figure_file("d1", 3, tmp_path) is None


# --------------------------------------------------------------------------- #
#  MCP: инструменты — запросы к серверу базы
# --------------------------------------------------------------------------- #
@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Сервер базы-заглушка: запоминает адреса, отвечает JSON или картинкой."""
    asked: list[str] = []

    def fetch(path: str, **params: Any) -> tuple[bytes, str]:
        asked.append(mcp_server._url(path, params).split("8000", 1)[1])
        if path == "/api/figure" and params.get("n"):
            return b"\x89PNG", "image/png"
        if path == "/api/figure":
            return (
                json.dumps({"figures": [{"n": 1, "page": 2, "caption": "Схема"}]}).encode(),
                "application/json",
            )
        return json.dumps({"ok": path}).encode(), "application/json"

    monkeypatch.setattr(mcp_server, "_fetch", fetch)
    return asked


def test_tools_ask_server(served: list[str]) -> None:
    mcp_server.search("рудные узлы", kb="geology.ore")
    mcp_server.search("x", kb="")  # всё дерево
    mcp_server.get_fragment(5)
    mcp_server.get_article("d1", text=True)
    mcp_server.facts("Анабарский щит")
    mcp_server.facts()
    mcp_server.get_formula("d1", "55")
    assert served == [
        "/api/search?q=%D1%80%D1%83%D0%B4%D0%BD%D1%8B%D0%B5+%D1%83%D0%B7%D0%BB%D1%8B&kb=geology.ore&limit=10&per_doc=2",
        "/api/search?q=x&kb=.&limit=10&per_doc=2",
        "/api/fragment?chunk_id=5",
        "/api/article?doc_id=d1&text=1",
        "/api/dataset?name=%D0%90%D0%BD%D0%B0%D0%B1%D0%B0%D1%80%D1%81%D0%BA%D0%B8%D0%B9+%D1%89%D0%B8%D1%82",
        "/api/datasets",
        "/api/formula?doc_id=d1&number=55",
    ]


def test_figure_tool(served: list[str]) -> None:
    pytest.importorskip("mcp")
    image, caption = mcp_server.get_figure("d1", 1)
    assert caption == "Рисунок 1, с. 2. Схема"
    assert image.data == b"\x89PNG"


def test_server_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server, "BASE", "http://127.0.0.1:9")  # никто не слушает
    answer = mcp_server.list_kbs()
    assert "python kbrag.py serve" in answer["error"]


def test_server_registers_tools() -> None:
    pytest.importorskip("mcp")
    import asyncio

    server = mcp_server.build()
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {
        "list_kbs",
        "search",
        "get_fragment",
        "get_article",
        "facts",
        "get_formula",
        "search_formulas",
        "search_figures",
        "get_figure",
    }
