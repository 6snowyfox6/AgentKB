"""Скорость: реранкер вместо модели в отборе, быстрый режим, пул соединений, настройки машины."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from kbrag import machine
from kbrag.chat import answer as chat
from kbrag.chat import retrieval
from kbrag.chat.plan import Part, Plan
from kbrag.chat.settings import Settings
from kbrag.index.search import Hit


def hit(i: int, text: str) -> Hit:
    return Hit(i, f"doc{i}", i, text, title=f"Статья {i}", similarity=0.6)


class Words:
    """Реранкер-заглушка: высокая оценка, если в тексте есть слово из вопроса."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append((query, len(texts)))
        words = {w for w in query.lower().replace("?", "").split() if len(w) > 3}
        return [0.9 if words & set(t.lower().split()) else 0.05 for t in texts]


class Judge:
    def __init__(self, queries: list[str]) -> None:
        self.queries, self.calls = queries, 0

    def __call__(self, question: str, plan: Any, batch: Any, accepted: Any, s: Any) -> Any:
        self.calls += 1
        assert batch == []  # с реранкером модель кандидатов не читает
        return {"не_хватает": ["методы"], "запросы": self.queries}


def run(gen: Any) -> Any:
    events = []
    try:
        while True:
            events.append(next(gen))
    except StopIteration as stop:
        return stop.value, events


def setup(monkeypatch: pytest.MonkeyPatch, rounds: dict[str, list[Hit]]) -> None:
    """Поиск-заглушка: по первому запросу — свои кандидаты."""
    monkeypatch.setattr(retrieval, "merged_search", lambda *a, **k: [])
    monkeypatch.setattr(
        retrieval,
        "candidates",
        lambda conn, emb, queries, *a, **k: list(rounds.get(queries[0], [])),
    )


def test_reranker_alone_when_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    setup(
        monkeypatch,
        {"золото": [hit(i, f"золото и кварц {i}") for i in range(4)] + [hit(9, "погода")]},
    )
    judge, words = Judge([]), Words()
    plan = Plan([Part("Где золото?", ["золото"])])
    found, events = run(
        retrieval.retrieve(None, None, "Где золото?", plan, Settings(), judge, reranker=words)
    )
    assert found.reranked and not found.judged and judge.calls == 0  # модель не звали
    assert [s["doc_id"] for s in found.sources] == ["doc0", "doc1", "doc2", "doc3"]
    assert any("оцениваю найденное" in e["text"] for e in events)


def test_model_only_for_what_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    setup(
        monkeypatch,
        {
            "золото": [hit(1, "золото в жилах"), hit(2, "погода")],
            "методы прогноза": [hit(3, "методы прогноза золото")],
        },
    )
    judge, words = Judge(["методы прогноза"]), Words()
    plan = Plan([Part("Где золото?", ["золото"])])
    found, events = run(
        retrieval.retrieve(None, None, "Где золото?", plan, Settings(), judge, reranker=words)
    )
    # Второй круг дал подходящее, но его всё ещё мало — модель спросили снова; она
    # повторила запрос, и поиск остановился.
    assert judge.calls == 2 and found.judged and found.rounds == 2
    assert {s["doc_id"] for s in found.sources} == {"doc1", "doc3"}
    assert plan.extra == ["методы прогноза"] and plan.missing == ["методы"]
    assert any("ищу недостающее" in e["text"] for e in events)


def test_parts_scored_by_own_question(monkeypatch: pytest.MonkeyPatch) -> None:
    setup(monkeypatch, {"a": [hit(1, "признаки узлов")], "b": [hit(2, "методы выделения")]})
    words = Words()
    plan = Plan([Part("Какие признаки?", ["a"]), Part("Какие методы?", ["b"])])
    run(
        retrieval.retrieve(None, None, "Признаки и методы?", plan, Settings(), None, reranker=words)
    )
    asked = {q for q, _ in words.calls}
    assert {"Какие признаки?", "Какие методы?"} <= asked


def test_fast_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Быстрый режим: ни разбора моделью, ни модели в отборе, ни сверки — один вызов на ответ."""
    setup(monkeypatch, {"Где золото?": [hit(i, f"золото {i}") for i in range(3)]})
    monkeypatch.setattr(chat, "LOGS", tmp_path)

    def never(*a: Any, **k: Any) -> Any:
        pytest.fail("в быстром режиме это не вызывается")

    def llm(messages: Any, settings: Any) -> Any:
        yield {"text": "Золото [1]."}
        yield {"done": True, "tokens": 2}

    settings = Settings(kb="geology.ore.anabar")
    settings.fast = True
    events = list(
        chat.answer(
            None,
            None,
            "Где золото?",
            [],
            settings,
            llm=llm,
            planner=never,
            judge=never,
            datasets=None,
            conflicts=never,
            reranker=Words(),
        )
    )
    done = events[-1]
    assert done["mode"] == "база" and done["fast"] and done["reranked"]
    assert "plan" in done["timings"] and "conflicts" not in done["timings"]
    assert not any("разбираю вопрос" in e.get("text", "") for e in events)


def test_reranker_off_by_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag.index import rerank

    monkeypatch.setenv("KBRAG_RERANKER", "off")
    assert rerank.shared() is None


# --------------------------------------------------------------------------- #
#  Настройки машины и пул соединений
# --------------------------------------------------------------------------- #
def test_machine_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file = tmp_path / "settings.env"
    file.write_text(
        '# окно побольше\nKBRAG_NUM_CTX=32768  # для 24 ГБ\nKBRAG_TOP_K = "12"\nKBRAG_X=abc\n',
        encoding="utf-8",
    )
    assert machine.read(file) == {
        "KBRAG_NUM_CTX": "32768",
        "KBRAG_TOP_K": "12",
        "KBRAG_X": "abc",
    }
    assert machine.integer("KBRAG_NUM_CTX", 12288, path=file) == 32768
    assert machine.integer("KBRAG_X", 5, path=file) == 5  # не число — по умолчанию
    assert machine.integer("KBRAG_NONE", 7, path=file) == 7
    monkeypatch.setenv("KBRAG_TOP_K", "20")
    assert machine.integer("KBRAG_TOP_K", 8, path=file) == 20  # окружение главнее файла


class FakePool:
    made: list[dict[str, Any]] = []

    def __init__(self, dsn: str, **kw: Any) -> None:
        FakePool.made.append({"dsn": dsn, **kw})
        self.given = 0

    def connection(self) -> Any:
        import contextlib

        @contextlib.contextmanager
        def borrowed() -> Any:
            self.given += 1
            yield f"соединение {self.given}"

        return borrowed()


def test_pool_shared(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag.index import db

    module = types.ModuleType("psycopg_pool")
    module.ConnectionPool = FakePool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg_pool", module)
    monkeypatch.setattr(db, "_POOLS", {})
    monkeypatch.setattr(db, "POOL_SIZE", 4)
    FakePool.made = []
    with db.connect("postgresql://x") as first:
        pass
    with db.connect("postgresql://x") as second:
        pass
    assert (first, second) == ("соединение 1", "соединение 2")
    assert len(FakePool.made) == 1 and FakePool.made[0]["max_size"] == 4  # пул один


def test_pool_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag.index import db

    monkeypatch.setattr(db, "POOL_SIZE", 1)
    assert db.pool("postgresql://x") is None


def test_no_model_after_empty_round(monkeypatch: pytest.MonkeyPatch) -> None:
    """Круг, не давший подходящего, — последний: модель второй раз не зовут."""
    setup(monkeypatch, {"золото": [hit(1, "золото")], "методы": [hit(2, "погода")]})
    judge = Judge(["методы"])
    plan = Plan([Part("Где золото?", ["золото"])])
    found, _ = run(
        retrieval.retrieve(None, None, "Где золото?", plan, Settings(), judge, reranker=Words())
    )
    assert judge.calls == 1 and found.rounds == 2
