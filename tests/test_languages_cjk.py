"""Статьи на китайском и японском: запросы в переводе (plan) и авторы в заголовке
фрагмента (messages) — без модели."""

from __future__ import annotations

from typing import Any

import pytest

from kbrag import kb as kbmod
from kbrag.chat import plan as planmod
from kbrag.chat.messages import fragment_block, surnames
from kbrag.chat.settings import Settings


def test_translations_searched_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    """Переводы — не в общем списке запросов, а своей добавкой кандидатов (TRANSLATED)."""
    from kbrag.chat import retrieval
    from kbrag.index.search import Hit

    data = {
        "вопрос_целиком": "Масштаб аэромагнитной съёмки Илийского бассейна",
        "части": [{"вопрос": "Масштаб съёмки", "запросы": ["аэромагнитная съёмка масштаб"]}],
        "перевод": ["伊犁盆地 航磁测量 比例尺", "イリ盆地 空中磁気探査"],
    }
    plan = planmod.make_plan("?", [], Settings(), lambda q, h, s: planmod.parse_plan(data, q, s))
    assert plan.translated == data["перевод"]
    assert not set(plan.translated) & set(plan.queries)

    asked: list[tuple[list[str], int]] = []

    def candidates(conn: Any, emb: Any, queries: list[str], cache: Any, limit: int, kb: str):
        asked.append((list(queries), limit))
        if queries == plan.translated:
            return [Hit(99, "zh-doc", 1, "伊犁盆地 1∶5万 高精度航磁测量", title="伊犁盆地")]
        return []

    monkeypatch.setattr(retrieval, "candidates", candidates)
    monkeypatch.setattr(retrieval, "merged_search", lambda *a, **k: [])
    monkeypatch.setattr(retrieval, "top_down_candidates", lambda *a, **k: [])

    class Reranker:
        def score(self, query: str, texts: list[str]) -> list[float]:
            return [0.9 for _ in texts]

    found = retrieval.run(
        retrieval.retrieve(None, None, "?", plan, Settings(), judge=None, reranker=Reranker())
    )
    assert (plan.translated, retrieval.TRANSLATED) in asked
    assert [s["doc_id"] for s in found.sources] == ["zh-doc"]


def test_translation_rule_only_for_kb_with_languages(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    class Client:
        def chat_json(self, system: str, user: str, **kw: Any) -> dict[str, Any]:
            asked.append(user)
            return {"части": [{"вопрос": "q", "запросы": ["q"]}]}

    monkeypatch.setattr(Settings, "client", lambda self: Client())
    real = kbmod.get("geology")
    for langs in ([], ["по-китайски", "по-японски"]):
        topic = {**real.topic, "языки_перевода": langs}
        node = kbmod.KB(real.path, {**real.settings, "тема": topic}, real.dirs, real.own)
        monkeypatch.setattr(planmod, "get_kb", lambda path=None, node=node: node)
        planmod.plan_question("Вопрос?", [], Settings(kb="geology"))
    without, with_langs = asked
    assert "перевод" not in without
    assert "по-китайски, по-японски" in with_langs and '"перевод"' in with_langs


def test_authors_in_fragment_head() -> None:
    assert surnames(["Daiei Inoue", "Yukihiro Mizuochi", "Hiroyuki SAKURADA", "Ещё Один"]) == (
        "Inoue, Mizuochi, SAKURADA"
    )
    assert surnames(["Г. Н. Иванченко"]) == "Иванченко"
    assert surnames(None) == ""
    head = fragment_block(
        {
            "n": 7,
            "title": "Geological Characteristics",
            "authors": ["Daiei Inoue"],
            "year": 1992,
            "text": "Many lineaments coincide with faults.",
        }
    ).splitlines()[0]
    assert head == "[7] «Geological Characteristics» (Inoue, 1992)"


def test_cjk_fragment_scored_by_translation() -> None:
    """Китайский кусок с русским вопросом реранкер оценивает низко — берётся оценка по
    вопросу в переводе; латинский кусок переводом не оценивается."""
    from kbrag.chat import retrieval
    from kbrag.chat.plan import Part, Plan
    from kbrag.index.search import Hit

    plan = Plan([Part("Масштаб съёмки?", [])], translated=["伊犁盆地 航磁 比例尺"])
    zh = Hit(1, "zh", 1, "伊犁盆地中部做了1∶5万高精度航磁测量", title="伊犁盆地")
    en = Hit(2, "en", 1, "Aeromagnetic survey of the basin", title="Survey")
    asked: list[tuple[str, int]] = []

    class Reranker:
        def score(self, query: str, texts: list[str]) -> list[float]:
            asked.append((query, len(texts)))
            return [0.9 if "伊犁" in query else 0.05 for _ in texts]

    batch = [{"id": 1, "hit": zh, "part": 1}, {"id": 2, "hit": en, "part": 1}]
    good = retrieval.rerank_batch(Reranker(), "Масштаб съёмки?", plan, batch)
    assert [c["hit"].doc_id for c in good] == ["zh"]
    assert ("伊犁盆地 航磁 比例尺", 1) in asked  # переводом — только китайский кусок


def test_reranker_gets_capped_fragments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Четвёртый кусок статьи (сверх CANDIDATE_PER_DOC) реранкер всё равно видит."""
    from kbrag.chat import retrieval
    from kbrag.chat.plan import Part, Plan
    from kbrag.index.search import Hit

    hits = [Hit(i, "drastic", i, f"Table {i}", title="DRASTIC", similarity=0.6) for i in range(3)]
    abstract = Hit(
        9,
        "drastic",
        9,
        "Lineaments were extracted from ETM+ imagery.",
        title="DRASTIC",
        similarity=0.6,
    )
    monkeypatch.setattr(retrieval, "candidates", lambda *a, **k: list(hits))
    monkeypatch.setattr(retrieval, "merged_search", lambda *a, **k: [*hits, abstract])
    monkeypatch.setattr(retrieval, "top_down_candidates", lambda *a, **k: [])

    class Reranker:
        def score(self, query: str, texts: list[str]) -> list[float]:
            return [0.93 if "ETM+" in t else 0.5 for t in texts]

    plan = Plan([Part("По каким снимкам?", ["снимки"])], main="По каким снимкам?")
    found = retrieval.run(
        retrieval.retrieve(None, None, "?", plan, Settings(), judge=None, reranker=Reranker())
    )
    assert found.sources[0]["text"].startswith("Lineaments were extracted from ETM+")
