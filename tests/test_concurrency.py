"""Несколько человек сразу: очередь ответов (turns.py) и бюджет заданий модели (llm.py).

Раньше страница пускала к модели всех сразу: два человека с двух устройств — и на 16 ГБ
запросы вытесняли друг друга в vLLM и падали по таймауту у обоих."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from contextlib import contextmanager
from typing import Any

import pytest

from kbrag import llm, turns


def _in_thread(fn: Any, *args: Any) -> threading.Thread:
    thread = threading.Thread(target=fn, args=args, daemon=True)
    thread.start()
    return thread


def _until(check: Any, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not check():
        if time.monotonic() > deadline:
            raise AssertionError("не дождались")
        time.sleep(0.01)


# --------------------------------------------------------------------------- #
#  Очередь ответов
# --------------------------------------------------------------------------- #
def test_turn_waits_and_reports_place() -> None:
    queue = turns.Turns(1)
    release = threading.Event()
    places: list[int] = []
    order: list[str] = []

    def first() -> None:
        with queue.turn():
            order.append("first")
            release.wait(5)

    def second() -> None:
        with queue.turn(on_wait=places.append, poll=0.05):
            order.append("second")

    a = _in_thread(first)
    _until(lambda: order == ["first"])
    b = _in_thread(second)
    _until(lambda: places)
    assert order == ["first"] and queue.waiting == 1  # второй ждёт, а не идёт к модели
    release.set()
    a.join(5)
    b.join(5)
    assert order == ["first", "second"]
    assert set(places) == {1} and queue.waiting == 0


def test_turn_left_queue_frees_nothing_it_did_not_take() -> None:
    """Ушёл со страницы, пока ждал (on_wait падает): с очереди снят, чужое место цело."""
    queue = turns.Turns(1)
    with queue.turn():

        def gone(place: int) -> None:
            raise BrokenPipeError

        with pytest.raises(BrokenPipeError):
            with queue.turn(on_wait=gone, poll=0.05):
                pass
        assert queue.waiting == 0
    with queue.turn():  # место одно и свободно — не утекло и не удвоилось
        with pytest.raises(BrokenPipeError):
            with queue.turn(on_wait=gone, poll=0.05):
                pass


# --------------------------------------------------------------------------- #
#  Бюджет заданий модели
# --------------------------------------------------------------------------- #
def test_budget_waits_for_room_and_lets_big_one_alone() -> None:
    budget = llm.Budget(100)
    entered = threading.Event()
    done = threading.Event()

    def second() -> None:
        with budget.hold(50):
            entered.set()
            done.wait(5)

    with budget.hold(70):
        thread = _in_thread(second)
        time.sleep(0.1)
        assert not entered.is_set()  # 70 + 50 > 100 — ждёт
    _until(entered.is_set)
    assert budget.used == 50
    done.set()
    thread.join(5)
    assert budget.used == 0
    with budget.hold(10_000):  # больше всего бюджета — одно, когда больше никого нет
        assert budget.used == 100
    assert budget.used == 0


class _Response:
    def __init__(self, lines: list[bytes] | None = None) -> None:
        self.status_code = 200
        self.lines = lines or []

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    def iter_lines(self) -> Any:
        yield from self.lines

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


@pytest.fixture
def small_budget(monkeypatch: pytest.MonkeyPatch) -> llm.Budget:
    host = "http://model.test"
    budget = llm.Budget(3000)
    monkeypatch.setitem(llm._BUDGETS, host, budget)
    return budget


def test_chat_json_requests_do_not_overflow_model(
    small_budget: llm.Budget, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Два задания по ~2 тыс. токенов в бюджет на 3 тыс. — к модели по одному."""
    import requests

    now = {"in": 0, "max": 0}
    lock = threading.Lock()

    def post(url: str, **kw: Any) -> _Response:
        with lock:
            now["in"] += 1
            now["max"] = max(now["max"], now["in"])
        time.sleep(0.1)
        with lock:
            now["in"] -= 1
        return _Response()

    monkeypatch.setattr(requests, "post", post)
    client = llm.LLM(host="http://model.test")
    text = "слово " * 1500  # ~3000 знаков → ~1000 токенов + 1024 на ответ
    threads = [_in_thread(client.chat_json, "система", text) for _ in range(3)]
    for thread in threads:
        thread.join(10)
    assert now["max"] == 1 and small_budget.used == 0


def test_stream_frees_budget_when_closed(
    small_budget: llm.Budget, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Нажали «Стоп» на середине ответа — место в бюджете освобождается сразу."""
    import requests

    chunk = {"choices": [{"delta": {"content": "слово "}}]}
    lines = [b"data: " + json.dumps(chunk).encode()] * 50
    monkeypatch.setattr(requests, "post", lambda url, **kw: _Response(lines))
    events = llm.LLM(host="http://model.test").stream([{"role": "user", "content": "вопрос"}])
    assert next(events) == {"text": "слово "}
    assert small_budget.used > 0
    events.close()
    assert small_budget.used == 0


def test_kv_size_from_vllm_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    import requests

    metrics = (
        "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n"
        'vllm:cache_config_info{block_size="16",cache_dtype="fp8",num_gpu_blocks="818"} 1.0\n'
    )

    class Got:
        text = metrics

    monkeypatch.setattr(requests, "get", lambda url, timeout: Got())
    assert llm._kv_from_metrics("http://model.test") == 818 * 16


# --------------------------------------------------------------------------- #
#  Страница: два человека сразу
# --------------------------------------------------------------------------- #
def test_page_second_chat_waits_in_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Два вопроса со страницы сразу при одном месте: второй видит «вы в очереди» и
    получает ответ, когда первый закончил, — никто не падает."""
    from kbrag.web import server

    monkeypatch.setattr(turns, "SHARED", turns.Turns(1))
    monkeypatch.setattr(turns, "POLL", 0.05)
    first_may_finish = threading.Event()
    inside: list[str] = []

    def answer(conn: Any, embedder: Any, question: str, history: Any, settings: Any) -> Any:
        inside.append(question)
        if question == "первый":
            first_may_finish.wait(5)
        yield {"type": "token", "text": f"ответ на {question}"}
        yield {"type": "done"}

    @contextmanager
    def connect(dsn: Any = None) -> Any:
        yield object()

    monkeypatch.setattr(server.chat, "answer", answer)
    monkeypatch.setattr(server.db, "connect", connect)
    monkeypatch.setattr(server.db, "autocommit", lambda conn: None)
    httpd = server.Server(("127.0.0.1", 0), server.Handler)
    _in_thread(httpd.serve_forever)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/api/chat"
    got: dict[str, list[dict[str, Any]]] = {}

    def ask(question: str) -> None:
        body = json.dumps({"question": question}).encode()
        request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=10) as response:
            got[question] = [json.loads(line) for line in response if line.strip()]

    try:
        a = _in_thread(ask, "первый")
        _until(lambda: inside == ["первый"])
        b = _in_thread(ask, "второй")
        time.sleep(0.3)
        assert inside == ["первый"]  # второй ждёт очереди, к модели не пошёл
        first_may_finish.set()
        a.join(10)
        b.join(10)
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert [e["text"] for e in got["первый"] if e["type"] == "token"] == ["ответ на первый"]
    second = got["второй"]
    assert any(e["type"] == "status" and "в очереди" in e["text"] for e in second)
    assert [e["text"] for e in second if e["type"] == "token"] == ["ответ на второй"]
    assert not [e for e in second + got["первый"] if e["type"] == "error"]
