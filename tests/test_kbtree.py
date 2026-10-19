"""Дерево баз: команды kb (kbtree.py), выбор базы в Телеграме, вопросы оценки по ветке.

Postgres подменён: команды проверяются на папках настроек и на том, что уходит в базу."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kbrag import evaluation, kb, kbtree

ROOT_YAML = "база_по_умолчанию: geo.ore  # комментарий остаётся\nназвание: Все\n"


class NoDB:
    """Соединение, которого нет: команды должны работать по папкам или честно сказать."""

    def __enter__(self) -> Any:
        raise ConnectionError("базы нет")

    def __exit__(self, *exc: Any) -> None:
        return None


class Args:
    def __init__(self, **kw: Any) -> None:
        self.dsn = None
        self.__dict__.update(kw)


def test_configured_and_folder(tree: Any) -> None:
    tree("geo", "kb.yaml", "название: Геология\n")
    tree("geo.ore.anabar", "kb.yaml", "название: Анабар\n")
    (tree.root / "_старое").mkdir()  # не метка пути — пропускается
    assert kb.configured() == ["geo", "geo.ore", "geo.ore.anabar"]
    assert kb.folder("geo.ore") == tree.root / "geo" / "ore"
    assert kb.folder("") == tree.root


def test_create_without_db(tree: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(kbtree, "_db", lambda args: NoDB())
    tree("", "kb.yaml", ROOT_YAML)
    assert kbtree.cmd_create(Args(path="phys.hadron", title="Адроны")) == 0
    settings = tree.root / "phys" / "hadron" / "kb.yaml"
    assert settings.is_file()
    kb._CACHE.clear()
    assert kb.get("phys.hadron").title == "Адроны"
    assert "запишется при import" in capsys.readouterr().out
    assert kbtree.cmd_create(Args(path="", title="")) == 1  # корень есть всегда


def test_list_without_db(tree: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    monkeypatch.setattr(kbtree, "_db", lambda args: NoDB())
    tree("", "kb.yaml", ROOT_YAML)
    tree("geo.ore", "kb.yaml", "название: Руды\n")
    assert kbtree.cmd_list(Args()) == 0
    out = capsys.readouterr().out
    assert "База недоступна" in out and "Руды" in out and "по умолчанию" in out


class FakeConn:
    def __enter__(self) -> FakeConn:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


def test_move_folders_and_default(tree: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag.index import db

    tree("", "kb.yaml", ROOT_YAML)
    tree("geo.ore", "kb.yaml", "название: Руды\n")
    local = kb.LOCAL_DIR / "geo" / "ore"
    local.mkdir(parents=True)
    (local / "датасет.csv").write_text("x", encoding="utf-8")
    moved: list[tuple[str, str]] = []
    monkeypatch.setattr(kbtree, "_db", lambda args: FakeConn())
    monkeypatch.setattr(db, "move_kb", lambda conn, old, new: moved.append((old, new)) or 2)
    assert kbtree.cmd_move(Args(path="geo.ore", to="geo.metals")) == 0
    assert moved == [("geo.ore", "geo.metals")]
    assert (tree.root / "geo" / "metals" / "kb.yaml").is_file()
    assert (kb.LOCAL_DIR / "geo" / "metals" / "датасет.csv").is_file()
    root = (tree.root / "kb.yaml").read_text(encoding="utf-8")
    assert "база_по_умолчанию: geo.metals  # комментарий остаётся" in root


def test_move_refuses_busy_folder(tree: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag.index import db

    tree("a", "kb.yaml", "название: A\n")
    tree("b", "kb.yaml", "название: B\n")
    monkeypatch.setattr(kbtree, "_db", lambda args: FakeConn())
    monkeypatch.setattr(db, "move_kb", lambda *a: pytest.fail("база не должна меняться"))
    assert kbtree.cmd_move(Args(path="a", to="b")) == 1
    assert (tree.root / "a" / "kb.yaml").is_file()


def test_remove_needs_yes(tree: Any, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    from kbrag.index import db

    tree("", "kb.yaml", ROOT_YAML)
    tree("old", "kb.yaml", "название: Старое\n")
    removed: list[Any] = []
    monkeypatch.setattr(kbtree, "_db", lambda args: FakeConn())
    monkeypatch.setattr(
        db, "kb_nodes", lambda conn: [{"path": "old", "title": "", "own": 3, "subtree": 3}]
    )
    monkeypatch.setattr(
        db,
        "remove_kb",
        lambda conn, path, drop_orphans: removed.append((path, drop_orphans))
        or {"nodes": 1, "orphans": 3, "deleted": 0},
    )
    plan = Args(path="old", yes=False, docs=False, config=False)
    assert kbtree.cmd_remove(plan) == 0 and not removed
    assert "--yes" in capsys.readouterr().out
    assert kbtree.cmd_remove(Args(path="old", yes=True, docs=False, config=True)) == 0
    assert removed == [("old", False)] and not (tree.root / "old").exists()
    # базу по умолчанию не убрать, пока она по умолчанию
    assert kbtree.cmd_remove(Args(path="geo", yes=True, docs=False, config=False)) == 1


# --------------------------------------------------------------------------- #
#  Телеграм: выбор базы у каждого чата
# --------------------------------------------------------------------------- #
class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str, Any]] = []

    def send(self, chat_id: int, text: str, markup: Any = None) -> dict[str, Any]:
        self.sent.append((chat_id, text, markup))
        return {"message_id": len(self.sent)}

    def edit(self, chat_id: int, mid: int, text: str, markup: Any = None) -> None:
        self.sent.append((chat_id, text, markup))


def test_bot_remembers_choice(tree: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag.chat.settings import Settings
    from kbrag.tg import bot as tgbot
    from kbrag.tg.base import STALE

    tree("geo", "kb.yaml", "название: Геология\n")
    tree("phys", "kb.yaml", "название: Физика\n")
    choice = tmp_path / "telegram_kb.json"
    tg = FakeTelegram()
    b = tgbot.Bot(tg, None, None, Settings(kb="geo"), kb_file=choice)  # type: ignore[arg-type]
    session = b.session(7)
    session.history.append({"role": "user", "content": "был вопрос"})
    assert b.kb_of(7) == "geo"
    shown = {
        "default": "geo",
        "unplaced": None,
        "nodes": [
            {"path": "geo", "title": "Геология", "depth": 0, "own": None, "subtree": None},
            {"path": "phys", "title": "Физика", "depth": 0, "own": None, "subtree": None},
        ],
    }
    monkeypatch.setattr(kbtree, "nodes", lambda conn=None: shown)
    from kbrag.index import db

    monkeypatch.setattr(db, "connect", lambda dsn=None: NoDB())
    b.route(7, session, "/kb")
    _, text, markup = tg.sent[-1]
    labels = [row[0]["text"] for row in markup["inline_keyboard"]]
    assert labels == ["Все базы — всё дерево", "• Геология", "Физика"]
    assert b.press(7, 1, session, "kb:2") == "Физика"
    assert b.kb_of(7) == "phys" and not session.history  # разговор — заново
    assert json.loads(choice.read_text(encoding="utf-8")) == {"7": "phys"}
    again = tgbot.Bot(tg, None, None, Settings(kb="geo"), kb_file=choice)  # type: ignore[arg-type]
    assert again.kb_of(7) == "phys" and again.kb_of(8) == "geo"  # после перезапуска
    b.route(7, session, "/kb .")
    assert b.kb_of(7) == ""  # всё дерево
    assert b.press(7, 1, b.session(9), "kb:5") == STALE  # кнопка от старого меню


def test_read_choice_skips_bad(tmp_path: Path) -> None:
    from kbrag.tg.access import read_choice

    path = tmp_path / "c.json"
    path.write_text('{"1": "geo.ore", "2": "плохой путь", "x": "geo"}', encoding="utf-8")
    assert read_choice(path) == {1: "geo.ore"}
    assert read_choice(tmp_path / "нет.json") == {}


# --------------------------------------------------------------------------- #
#  Оценка: вопросы своей ветки
# --------------------------------------------------------------------------- #
QUESTIONS = "вопросы:\n  - id: s01\n    вопрос: {q}\n"


def test_question_files_of_subtree(tree: Any) -> None:
    tree("geo.ore.anabar", "eval-questions.yaml", QUESTIONS.format(q="Что в Анабаре?"))
    tree("geo.ore.aldan", "eval-questions.yaml", QUESTIONS.format(q="Что в Алдане?"))
    tree("phys", "eval-questions.yaml", QUESTIONS.format(q="Что с адронами?"))
    files = evaluation.question_files(kb.get("geo.ore"))
    assert [node for node, _ in files] == ["geo.ore.aldan", "geo.ore.anabar"]
    questions = evaluation.load_tree(files)
    assert [(q.id, q.kb) for q in questions] == [
        ("s01", "geo.ore.aldan"),
        ("anabar:s01", "geo.ore.anabar"),  # тот же id в другой базе — с её меткой
    ]
    # в поддереве вопросов нет — ближайший файл выше, вопросы — в самой базе
    tree("geo", "eval-questions.yaml", QUESTIONS.format(q="Что в геологии?"))
    kb._CACHE.clear()
    [(node, file)] = evaluation.question_files(kb.get("geo.shields"))
    assert node == "geo.shields" and file.parent.name == "geo"


def test_previous_run_same_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(evaluation, "default_path", lambda: "geo.ore.anabar")
    runs = {
        "eval-20261001-1000.json": {"meta": {}, "summary": {"q": "до дерева"}},
        "eval-20261002-1000.json": {"meta": {"kb": "phys"}, "summary": {"q": "физика"}},
    }
    for name, data in runs.items():
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    assert evaluation.previous_run(tmp_path, "20261003-1000", "phys") == {"q": "физика"}
    assert evaluation.previous_run(tmp_path, "20261003-1000", "geo.ore.anabar") == {
        "q": "до дерева"
    }
    assert evaluation.previous_run(tmp_path, "20261003-1000", "geo") is None


def result(base: str, qid: str, share: float) -> dict[str, Any]:
    search = {
        "found": 3,
        "keywords": share,
        "missing": [],
        "ok": True,
        "docs": None,
        "docs_missing": [],
        "sources": [],
    }
    return {
        "kb": base,
        "id": qid,
        "question": "?",
        "expect": "база",
        "queries": [],
        "concept": "",
        "territory": "",
        "method": "",
        "search": search,
    }


def test_report_by_base() -> None:
    meta = {
        "when": "сейчас",
        "documents": 1,
        "chunks": 1,
        "model": "m",
        "with_answer": False,
        "planner": True,
        "kb": "",
        "kb_title": "Все базы",
    }
    both = [result("a", "1", 1.0), result("b", "2", 0.5)]
    two = evaluation.report(both, evaluation.summarize(both), None, meta)
    assert "# Оценка: Все базы" in two and "## По базам" in two and "| b | 1 |" in two
    single = [result("a", "1", 1.0)]
    one = evaluation.report(single, evaluation.summarize(single), None, meta)
    assert "## По базам" not in one  # одна база — без лишней таблицы
