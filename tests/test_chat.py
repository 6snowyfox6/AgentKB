"""Чат-бот на заглушках: пачки отбора разом, выключаемая сверка, время по этапам."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kbrag.chat import answer as chat
from kbrag.chat import retrieval
from kbrag.chat.settings import Settings
from kbrag.index.search import Hit

JUDGE_SECONDS = 0.3


def hit(i: int) -> Hit:
    return Hit(
        i,
        f"doc{i}",
        i,
        f"Фрагмент {i}. Золото связано с кварцем.",
        title=f"Статья {i}",
        similarity=0.6,
    )


class Judge:
    """Модель-отборщик: каждую пачку «читает» JUDGE_SECONDS, считает одновременные вызовы."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = self.peak = self.calls = 0

    def __call__(self, question: str, plan: Any, batch: list[Any], accepted: Any, s: Any) -> Any:
        with self.lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        time.sleep(JUDGE_SECONDS)
        with self.lock:
            self.active -= 1
        return {"подходят": [c["id"] for c in batch[:2]], "не_хватает": [], "запросы": []}


def llm(messages: list[dict[str, Any]], settings: Any) -> Any:
    for word in ["Золото ", "связано ", "с кварцем [1]."]:
        yield {"text": word}
    yield {"done": True, "tokens": 3}


def ask(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, conflicts_on: bool) -> tuple[Any, Judge]:
    hits = [hit(i) for i in range(30)]  # 30 кандидатов — три пачки по JUDGE_MAX
    monkeypatch.setattr(retrieval, "merged_search", lambda *a, **k: [])
    monkeypatch.setattr(retrieval, "candidates", lambda *a, **k: list(hits))
    monkeypatch.setattr(chat, "LOGS", tmp_path)
    checked: list[bool] = []

    def conflicts(question: str, sources: Any, settings: Any) -> list[Any]:
        checked.append(True)
        return []

    judge = Judge()
    settings = Settings(kb="geology.ore.anabar")
    settings.check_conflicts = conflicts_on
    events = list(
        chat.answer(
            None,
            None,
            "С чем связано золото?",
            [],
            settings,
            llm=llm,
            planner=lambda q, h, s: ["золото кварц"],
            judge=judge,
            datasets=None,
            conflicts=conflicts,
        )
    )
    assert bool(checked) is conflicts_on
    return events, judge


def test_batches_go_at_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events, judge = ask(monkeypatch, tmp_path, conflicts_on=False)
    done = events[-1]
    assert done["type"] == "done" and done["mode"] == "база"
    assert judge.calls == 3 and judge.peak == 3
    assert done["timings"]["judge"] < 2 * JUDGE_SECONDS  # не сумма трёх пачек
    status = [e["text"] for e in events if e["type"] == "status"]
    assert any("3 пачки разом" in s for s in status)


def test_conflicts_switch_and_timings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    on, _ = ask(monkeypatch, tmp_path, conflicts_on=True)
    off, _ = ask(monkeypatch, tmp_path, conflicts_on=False)
    assert set(on[-1]["timings"]) == {"plan", "search", "judge", "conflicts", "answer"}
    assert "conflicts" not in off[-1]["timings"]
    assert not any("сверяю" in e.get("text", "") for e in off if e["type"] == "status")
    log = next(tmp_path.glob("chat-*.log")).read_text(encoding="utf-8").splitlines()
    assert len(log) == 2 and "сверка" in log[0] and "сверка" not in log[1]
    assert "отбор" in chat.describe_timings(on[-1]["timings"])


def test_off_topic_refusal_from_kb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Вопрос не по теме и в базе пусто — отказ словами базы, без ответа из общих знаний."""
    from kbrag.chat.plan import Part, Plan

    monkeypatch.setattr(retrieval, "candidates", lambda *a, **k: [])
    monkeypatch.setattr(retrieval, "merged_search", lambda *a, **k: [])
    monkeypatch.setattr(chat, "LOGS", tmp_path)
    plan = Plan([Part("Как испечь хлеб?", ["хлеб"])], on_topic=False)
    events = list(
        chat.answer(
            None,
            None,
            "Как испечь хлеб?",
            [],
            Settings(kb="geology.ore.anabar"),
            llm=llm,
            planner=lambda q, h, s: plan,
            judge=lambda *a: {"подходят": [], "не_хватает": [], "запросы": []},
            datasets=None,
        )
    )
    general = next(e for e in events if e["type"] == "general")
    assert general.get("off_topic") and "геолог" in general["text"]
