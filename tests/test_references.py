"""Статьи, на которые опирается ответ (chat/references.py): список литературы под ответом
— на странице, в Телеграме, в терминале — и свой PDF статьи (/api/pdf, кнопка «PDF n»)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from kbrag.chat import references

SOURCES = [
    {
        "n": 1,
        "doc_id": "a",
        "title": "Строение щита",
        "year": 2021,
        "journal": "Геотектоника",
        "authors": ["Иванов И.", "Петров П.", "Сидоров С.", "Козлов К."],
        "pages": [5],
        "url": "",
    },
    {"n": 2, "doc_id": "b", "title": "Не процитирована", "pages": [1]},
    {"n": 3, "doc_id": "a", "title": "Строение щита", "pages": [7, 5]},
    {"n": 4, "doc_id": "c", "title": "Вторая", "year": 2019, "pages": [], "url": "https://x/y"},
    {"n": 5, "doc_id": "датасет:щит", "title": "Факты из графа", "kind": "датасет"},
]


def test_group_by_article_in_citation_order() -> None:
    found = references.group(SOURCES, [4, 3, 1, 5])
    assert [a["doc_id"] for a in found] == ["a", "c"]  # по номеру первой ссылки; сводка — не статья
    first = found[0]
    assert first["fragments"] == [1, 3] and first["pages"] == [5, 7]
    assert references.group(SOURCES, []) == []  # не сослался ни на что — списка нет


def test_citation_and_doi() -> None:
    first, second = references.group(SOURCES, [1, 4])
    assert references.citation(first) == (
        "Иванов И., Петров П., Сидоров С. и др. (2021). Строение щита. Геотектоника"
    )
    assert references.citation(second) == "2019. Вторая"
    assert references.doi_link("10.1/x") == "https://doi.org/10.1/x"
    assert references.doi_link("https://doi.org/10.1/x") == "https://doi.org/10.1/x"
    assert references.doi_link("") == ""


def test_pdf_path(tmp_path: Path) -> None:
    pdf = tmp_path / "статья.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    records = {
        "a": {"pdf_path": str(pdf)},
        "b": {"pdf_path": str(tmp_path / "нет.pdf")},
        "c": {"pdf_path": str(tmp_path)},
    }
    assert references.pdf_path("a", records.get) == pdf
    assert references.pdf_path("b", records.get) is None  # файла нет
    assert references.pdf_path("c", records.get) is None  # не PDF
    assert references.pdf_path("d", records.get) is None  # статья без своего файла


def test_articles_without_database(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag.library import objects

    monkeypatch.setattr(objects, "record", lambda doc_id: {})
    found = references.articles(object(), SOURCES, [1])  # подставное соединение — без DOI
    assert found[0]["doi"] == "" and found[0]["pdf"] is False


def test_answer_done_has_articles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import test_chat

    events, _ = test_chat.ask(monkeypatch, tmp_path, conflicts_on=False)
    done = events[-1]
    assert done["used"] == [1]
    assert [a["doc_id"] for a in done["articles"]] == [events_sources(events)[0]["doc_id"]]


def events_sources(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = next(e for e in events if e["type"] == "sources")["sources"]
    return sources


# --------------------------------------------------------------------------- #
#  Телеграм и терминал
# --------------------------------------------------------------------------- #
def test_telegram_article_lines_and_buttons() -> None:
    from kbrag.tg import ui

    first, second = references.group(SOURCES, [1, 3, 4])
    first.update(doi="10.1/x", pdf=True)
    line = ui.article_line(1, first)
    assert '<a href="https://doi.org/10.1/x">Строение щита</a>' in line
    assert line.endswith("— [1], [3] · PDF — кнопкой")
    assert '<a href="https://x/y">Вторая</a>' in ui.article_line(2, second)
    text = ui.answer_text("Ответ [1].", None, SOURCES, {"used": [1], "articles": [first, second]})
    assert "Статьи, на которые опирается ответ" in text and "Не процитирована" not in text
    markup = ui.answer_keyboard(7, SOURCES[:1], [first, second])
    assert markup is not None
    pdfs = [b for row in markup["inline_keyboard"] for b in row if b["text"].startswith("PDF")]
    assert [(b["text"], b["callback_data"]) for b in pdfs] == [("PDF 1", "pdf:7:1")]


def test_telegram_sends_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag.chat.settings import Settings
    from kbrag.tg import bot as tgbot

    pdf = tmp_path / "статья.pdf"
    pdf.write_bytes(b"%PDF-1.4 x")
    monkeypatch.setattr(
        references, "pdf_path", lambda doc_id, record=None: pdf if doc_id == "a" else None
    )

    @contextmanager
    def connect(dsn: Any) -> Any:
        yield None

    from kbrag.index import db

    monkeypatch.setattr(db, "connect", connect)
    monkeypatch.setattr(db, "visible_doc", lambda conn, doc_id: True)

    class Telegram:
        def __init__(self) -> None:
            self.files: list[Any] = []

        def document(self, chat_id: int, name: str, data: bytes, caption: str = "") -> None:
            self.files.append((name, data, caption))

    tg = Telegram()
    b = tgbot.Bot(tg, None, None, Settings(kb=""), log=lambda _: None, kb_file=tmp_path / "k.json")  # type: ignore[arg-type]
    session = b.session(5)
    item = {**references.group(SOURCES, [1, 3])[0], "pdf": True}
    session.answers[9] = {"sources": SOURCES, "articles": [item]}
    assert b.press(5, 1, session, "pdf:9:1") is None
    [(name, data, caption)] = tg.files
    assert name == "статья.pdf" and data == b"%PDF-1.4 x"
    assert caption.startswith("Иванов И.") and caption.endswith("Процитировано: с. 5, 7")
    assert b.press(5, 1, session, "pdf:9:2") == tgbot.STALE  # такой статьи в ответе нет
    other = {**item, "doc_id": "b"}
    session.answers[10] = {"sources": [], "articles": [other]}
    assert "нет" in str(b.press(5, 1, session, "pdf:10:1"))


def test_terminal_prints_articles(capsys: pytest.CaptureFixture[str]) -> None:
    from kbrag.chat import cli

    first, second = references.group(SOURCES, [1, 4])
    first["doi"] = "10.1/x"
    cli.print_articles([first, second])
    out = capsys.readouterr().out
    assert (
        "1. Иванов И., Петров П., Сидоров С. и др. (2021). Строение щита. Геотектоника — [1]" in out
    )
    assert "https://doi.org/10.1/x" in out and "https://x/y" in out


# --------------------------------------------------------------------------- #
#  Страница: /api/pdf
# --------------------------------------------------------------------------- #
def test_api_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag.web import server

    pdf = tmp_path / "статья 1.pdf"
    pdf.write_bytes(b"%PDF-1.4 y")
    monkeypatch.setattr(
        references, "pdf_path", lambda doc_id, record=None: pdf if doc_id == "a" else None
    )
    h = server.Handler.__new__(server.Handler)
    sent: list[Any] = []
    headers: dict[str, str] = {}
    h.send_response = lambda code: sent.append(code)  # type: ignore[method-assign]
    h.send_header = lambda k, v: headers.__setitem__(k, v)  # type: ignore[method-assign]
    h.end_headers = lambda: None  # type: ignore[method-assign]

    class Out:
        body = b""

        def write(self, data: bytes) -> None:
            Out.body += data

    h.wfile = Out()  # type: ignore[assignment]
    h._json = lambda payload, code=200: sent.append((code, payload))  # type: ignore[method-assign]
    h._pdf({"doc_id": ["a"]})
    assert sent == [200] and Out.body == b"%PDF-1.4 y"
    assert headers["Content-Type"] == "application/pdf"
    assert headers["Content-Disposition"].startswith("inline; filename*=UTF-8''")
    h._pdf({"doc_id": ["b"]})
    assert sent[-1][0] == 404
    assert "/api/pdf" in server.BY_DOCUMENT  # права — по самой статье


def test_page_renders_articles() -> None:
    page = (Path(__file__).resolve().parents[1] / "kbrag" / "web" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "function renderArticles" in page and "/api/pdf?doc_id=" in page and "#page=" in page
