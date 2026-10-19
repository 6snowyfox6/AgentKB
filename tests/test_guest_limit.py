"""П17: бот открыт всем («*») — у гостя ограничено число вопросов в час, и tg --check
об этом предупреждает."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kbrag import machine
from kbrag.tg import bot as tgbot
from kbrag.tg import check as tgcheck


class Telegram:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, chat_id: int, text: str, markup: Any = None) -> None:
        self.sent.append(text)


@pytest.fixture
def opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[tgbot.Bot, Telegram]:
    users = tmp_path / "users.txt"
    users.write_text("*\n42\n", encoding="utf-8")
    monkeypatch.delenv("KBRAG_TG_USERS", raising=False)
    monkeypatch.setattr(machine, "integer", lambda name, default, low=1, path=None: 2)
    tg = Telegram()
    b = tgbot.Bot(tg, None, None, None, users_file=users, log=lambda _: None)  # type: ignore[arg-type]
    asked: list[str] = []
    monkeypatch.setattr(b, "question", lambda chat_id, session, text: asked.append(text))
    monkeypatch.setattr(b, "search", lambda chat_id, session, text: asked.append("/s " + text))
    b.asked = asked  # type: ignore[attr-defined]
    return b, tg


def message(user: int, text: str) -> dict[str, Any]:
    return {"chat": {"id": user}, "from": {"id": user}, "text": text}


def test_guest_limited_owner_not(opened: tuple[tgbot.Bot, Telegram]) -> None:
    b, tg = opened
    for _ in range(3):
        b.on_message(message(7, "Что такое кимберлит?"))
    assert b.asked == ["Что такое кимберлит?"] * 2  # type: ignore[attr-defined]
    assert "у гостя — 2 вопросов в час" in tg.sent[-1]
    b.on_message(message(7, "/search кимберлит"))  # поиск — без ограничения
    for _ in range(3):
        b.on_message(message(42, "Вопрос владельца"))
    assert b.asked[2:] == ["/s кимберлит"] + ["Вопрос владельца"] * 3  # type: ignore[attr-defined]


def test_window_passes(opened: tuple[tgbot.Bot, Telegram]) -> None:
    b, _ = opened
    assert b.over_limit(7, now=0.0) == 0 and b.over_limit(7, now=10.0) == 0
    assert b.over_limit(7, now=20.0) == pytest.approx(3580.0)
    assert b.over_limit(7, now=3601.0) == 0  # первый вопрос вышел из окна


def test_check_warns_about_open_bot(
    opened: tuple[tgbot.Bot, Telegram], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Api:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def call(self, method: str, **kw: Any) -> Any:
            return {"getMe": {"username": "geo_bot"}, "getWebhookInfo": {}}.get(method, [])

    monkeypatch.setattr(tgcheck, "Telegram", Api)
    lines: list[str] = []
    tgcheck.check(lines.append, token="123456789:" + "A" * 35, users_file=opened[0].users_file)
    warning = next(line for line in lines if "открыт всем" in line)
    assert warning.startswith("[ !! ]") and "2 вопросов в час" in warning and "42" in warning
