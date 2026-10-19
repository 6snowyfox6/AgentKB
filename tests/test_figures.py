"""Рисунки (Д15–Д17): номер из подписи, ссылки «рис. N», таблица рисунков, описание
моделью со зрением по файлам, рисунки под ответом, модель со зрением в Docker."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pytest
from fakedb import FakeConn, returning_ids

from kbrag.chunks import figures as chunk_figures
from kbrag.index import figures
from kbrag.library import describe, importer, store


@pytest.fixture
def root() -> Path:
    return Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
#  Номер и ссылки
# --------------------------------------------------------------------------- #
def test_label_of_caption() -> None:
    assert chunk_figures.label_of("Рис. 3. Схема разломов") == "3"
    assert chunk_figures.label_of("Рисунок 2а — разрез") == "2а"
    assert chunk_figures.label_of("Fig. 4: map") == "4"
    assert chunk_figures.label_of("Figure 1b. Section") == "1b"
    assert chunk_figures.label_of("Схема без номера") is None


def test_references() -> None:
    text = (
        "Как видно на рис. 3 и на рисунке 2а, а также (рис. 5, 6 и 7); см. Fig. 4, figs. 8 and 9."
    )
    labels = {"2а", "3", "4", "5", "6", "7", "8", "9", "10"}
    assert figures.references(text, labels) == labels - {"10"}
    assert figures.references("Таблица 3, рисунков нет", {"3"}) == set()
    assert figures.references("рис. 12", {"1", "2"}) == set()  # не «1» и не «2»


def test_prepare_from_record() -> None:
    meta = {
        "figures": [
            {
                "n": 1,
                "caption": "Рис. 3.  Схема\nразломов",
                "page": 2,
                "file": "d.figures/fig-001.png",
            },
            {"n": 2, "caption": "", "page": 4, "description": "карта"},
            "мусор",
        ]
    }
    first, second = figures.prepare(meta)
    assert first["label"] == "3" and first["caption"] == "Рис. 3. Схема разломов"
    assert first["search_text"] == "Рисунок 3. Рис. 3. Схема разломов"
    assert second["label"] is None and second["search_text"] == "Рисунок. карта"


# --------------------------------------------------------------------------- #
#  Таблица рисунков (подставное соединение)
# --------------------------------------------------------------------------- #
def test_store_figures_and_refs() -> None:
    mark = chunk_figures.MARK
    chunks = [
        (11, "Текст. Рис. 1. Схема разломов Анабарского щита. Дальше…", ["Геология"]),
        (12, "Разломы (рис. 1) секут свиту, см. также рис. 2.", ["Геология"]),
        (13, f"Рисунок 2 (с. 5). Рис. 2. Разрез\n{mark} слои", [chunk_figures.HEADING]),
    ]
    conn = FakeConn(answer=returning_ids(chunks=chunks))
    items = figures.prepare(
        {
            "figures": [
                {"n": 1, "caption": "Рис. 1. Схема разломов Анабарского щита", "page": 3},
                {"n": 2, "caption": "Рис. 2. Разрез", "page": 5, "description": "слои"},
            ]
        }
    )
    assert figures.store(conn, "d1", items, [[0.1], [0.2]]) == 2
    inserts = conn.queries("INSERT INTO figures")
    assert [p[10] for p in inserts] == [11, 13]  # где стоит: подпись в тексте, фрагмент «Рисунок»
    refs = set(conn.queries("INSERT INTO figure_refs"))
    assert refs == {(11, 101), (12, 101), (12, 102)}  # повтор отсекает ON CONFLICT
    assert conn.log[0][0].startswith("DELETE FROM figures")


def test_for_answer_only_cited(monkeypatch: pytest.MonkeyPatch) -> None:
    def in_chunks(conn: Any, ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        assert ids == [21, 22]
        return {21: [{"id": 1}, {"id": 2}], 22: [{"id": 2}, {"id": 3}]}

    monkeypatch.setattr(figures, "in_chunks", in_chunks)
    sources = [{"n": 1, "chunk_id": 21}, {"n": 2, "chunk_id": 99}, {"n": 3, "chunk_id": 22}]
    found = figures.for_answer(None, sources, [1, 3], limit=3)
    assert [(f["id"], f["source"]) for f in found] == [(1, 1), (2, 1), (3, 3)]
    assert figures.for_answer(None, sources, []) == []  # ни на что не сослался — ни одного


# --------------------------------------------------------------------------- #
#  Описание по файлам
# --------------------------------------------------------------------------- #
def png(width: int, height: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, "PNG")
    return buf.getvalue()


class Client:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def stream(self, messages: list[dict[str, Any]], temperature: float = 0) -> Any:
        content = messages[0]["content"]
        self.seen.append(content[1]["image_url"]["url"][:22])
        yield {"text": "Карта  разломов"}
        yield {"text": " с легендой"}


def article(tmp_path: Path) -> Path:
    articles = tmp_path / "articles"
    pictures = articles / "d1.figures"
    pictures.mkdir(parents=True)
    (pictures / "fig-001.png").write_bytes(png(400, 300))
    (pictures / "fig-002.png").write_bytes(png(100, 100))  # значок — не описываем
    record = {
        "doc_id": "d1",
        "figures": [
            {"n": 1, "file": "d1.figures/fig-001.png", "page": 2, "caption": "Рис. 1. Карта"},
            {"n": 2, "file": "d1.figures/fig-002.png", "page": 3, "caption": "лого"},
        ],
    }
    store.save_record(articles, record)
    text = [{"doc_id": "d1", "index": 0, "text": "Текст", "headings": ["Введение"]}]
    (articles / "d1.chunks.json").write_text(json.dumps(text), encoding="utf-8")
    return articles


def test_describe_saved_files_and_refresh(tmp_path: Path) -> None:
    articles = article(tmp_path)
    client = Client()
    changed = describe.describe_articles(["d1", "нет"], articles, client, log=lambda _: None)
    assert changed == ["d1"] and client.seen == ["data:image/png;base64,"]
    record = store.records(articles)["d1"]
    assert record["figures"][0]["description"] == "Карта разломов с легендой"
    assert "description" not in record["figures"][1]
    chunks = json.loads((articles / "d1.chunks.json").read_text("utf-8"))
    assert [c["index"] for c in chunks] == [0, 1]
    assert chunks[1]["text"].startswith("Рисунок 1 (с. 2). Рис. 1. Карта\n")
    assert chunk_figures.is_description(chunks[1]) and not chunk_figures.is_description(chunks[0])

    # Ещё раз — описанное не трогается; --redo — описание заменяется, а не дублируется.
    assert describe.describe_articles(["d1"], articles, client, log=lambda _: None) == []
    describe.describe_articles(["d1"], articles, client, redo=True, log=lambda _: None)
    chunks = json.loads((articles / "d1.chunks.json").read_text("utf-8"))
    assert len(chunks) == 2


def test_reparse_keeps_descriptions() -> None:
    old = [{"n": 1, "page": 2, "caption": "Рис. 1. Карта", "description": "карта"}]
    new = [{"n": 1, "page": 2, "caption": "Рис. 1.  Карта"}, {"n": 2, "page": 2, "caption": "x"}]
    assert importer.keep_descriptions(old, new) == 1
    assert new[0]["description"] == "карта" and "description" not in new[1]


def test_pending_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(describe, "PENDING", tmp_path / "pending.json")
    describe._pending(["b", "a"])
    assert describe._pending(["a", "c"]) == ["a", "b", "c"]
    assert describe._pending() == ["a", "b", "c"]


# --------------------------------------------------------------------------- #
#  Модель со зрением: Docker и запускатель
# --------------------------------------------------------------------------- #
def test_compose_has_vision_service(root: Path) -> None:
    yaml = pytest.importorskip("yaml")
    compose = yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf-8"))
    vision = compose["services"]["vision"]
    assert vision["profiles"] == ["vision"]
    assert "--served-model-name=qwen3-vl" in vision["command"]
    assert vision["ports"] == ["127.0.0.1:8101:8000"]
    assert compose["services"]["llm"]["command"][0] == "${KBRAG_LLM_MODEL:-Qwen/Qwen3-14B-AWQ}"


def test_vision_turns(launcher: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """По очереди: модель ответов гаснет, vision поднимается, после — всё обратно."""
    calls: list[tuple[str, ...]] = []
    state = {"vision": False}

    def docker(*args: str, quiet: bool = False) -> Any:
        calls.append(args)
        if args[-2:] == ("-d", "vision"):
            state["vision"] = True

        class Done:
            returncode = 0

        return Done()

    monkeypatch.setattr(launcher, "_docker", docker)
    monkeypatch.setattr(launcher, "_answers", lambda url, timeout=5: state["vision"])
    monkeypatch.setattr(launcher, "_llm_up", lambda timeout=5: True)
    monkeypatch.setattr(launcher, "_ensure_llm", lambda wait=True: calls.append(("llm-up",)))
    assert launcher._with_vision(lambda: 0) == 0
    assert calls == [
        ("--profile", "llm", "stop", "llm"),
        ("--profile", "vision", "up", "-d", "vision"),
        ("--profile", "vision", "stop", "vision"),
        ("llm-up",),
    ]


def test_vision_single(launcher: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Одна модель со зрением: рисунки описывает основная — vision не поднимается."""
    monkeypatch.setattr(
        launcher, "_machine_value", lambda name, default: "single" if "MODE" in name else default
    )
    monkeypatch.setattr(launcher, "_docker", lambda *a, **k: pytest.fail("docker не нужен"))
    monkeypatch.setattr(launcher, "_ensure_llm", lambda wait=True: True)
    assert launcher._with_vision(lambda: 0) == 0


def test_import_figures_after_parse(launcher: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """turns: импорт без описания (docling и vision разом не помещаются), потом figures."""
    import argparse

    opts = argparse.Namespace(
        text="D:/статьи", ask=False, force=False, figures=True, redo=False, ocr=False
    )
    ran: list[Any] = []
    monkeypatch.setattr(launcher, "_ensure_database", lambda: True)
    monkeypatch.setattr(launcher, "_has_cuda", lambda: False)
    monkeypatch.setattr(launcher, "_py", lambda module, *args: ran.append((module, args)) or 0)
    monkeypatch.setattr(launcher, "_with_vision", lambda run: run())
    assert launcher.cmd_import(opts) == 0
    assert ran[0][0] == "kbrag.library.cli" and "--figures" not in ran[0][1]
    assert ran[1] == ("kbrag.library.describe", ("--no-ingest",))
    assert ran[2][1][0] == "--ingest-pending"


# --------------------------------------------------------------------------- #
#  Телеграм и страница
# --------------------------------------------------------------------------- #
def test_fragment_keyboard_figures() -> None:
    from kbrag.tg import ui

    markup = ui.fragment_keyboard(
        {"url": ""}, None, [], [{"id": 7, "label": "3", "n": 1}, {"id": 8, "label": None, "n": 2}]
    )
    assert markup is not None
    buttons = [b for row in markup["inline_keyboard"] for b in row]
    assert [(b["text"], b["callback_data"]) for b in buttons] == [
        ("Рис. 3", "fg:7"),
        ("Рисунок 2", "fg:8"),
    ]
    caption = ui.figure_caption({"label": "3", "title": "A & B", "page": 4, "caption": "<x>"}, 2)
    assert caption == "<b>Рис. 3</b> к [2] — A &amp; B, с. 4\n&lt;x&gt;"


def test_page_renders_figures(root: Path) -> None:
    page = (root / "kbrag" / "web" / "index.html").read_text(encoding="utf-8")
    assert 'ev.type === "figures"' in page and "function renderFigures" in page


def test_figure_png_is_data_url() -> None:
    figure = chunk_figures.Figure(1, "", png(400, 300), (400, 300))
    data = base64.b64encode(figure.png)
    assert data.startswith(b"iVBOR")
