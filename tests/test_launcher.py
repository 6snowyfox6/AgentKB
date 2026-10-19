"""Запускатель kbrag.py: сквозные команды (у модуля свои ключи) готовят окружение —
базу, модель — и передают аргументы как есть."""

from __future__ import annotations

import sys
from typing import Any

import pytest


@pytest.fixture
def calls(launcher: Any, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    log: list[Any] = []
    monkeypatch.setattr(launcher, "_need_venv", lambda: None)
    monkeypatch.setattr(launcher, "_ensure_database", lambda: log.append("db") or True)
    monkeypatch.setattr(launcher, "_ensure_llm", lambda wait=True: log.append("llm") or True)
    monkeypatch.setattr(launcher, "_py", lambda module, *args: log.append((module, args)) or 0)
    monkeypatch.setattr(sys, "stdout", sys.stdout)  # --json подменяет stdout — вернём сами
    return log


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["kb"], [("kbrag.kbtree", ("list",))]),  # без базы: список — из настроек
        (["kb", "list"], [("kbrag.kbtree", ("list",))]),
        (["kb", "create", "a.b"], ["db", ("kbrag.kbtree", ("create", "a.b"))]),
        (["summaries"], ["db", "llm", ("kbrag.index.summaries", ())]),
        (["summaries", "d1", "--show"], ["db", ("kbrag.index.summaries", ("d1", "--show"))]),
        (["formula", "d1", "55"], ["db", ("kbrag.index.formulas", ("d1", "55"))]),
        (["figure", "--find", "схема"], ["db", ("kbrag.index.figures", ("--find", "схема"))]),
        (["mcp"], [("kbrag.mcp_server", ())]),  # MCP говорит через stdout — запускатель молчит
    ],
)
def test_passthrough(launcher: Any, calls: list[Any], argv: list[str], expected: list[Any]) -> None:
    assert launcher.main(argv) == 0
    assert calls == expected


def test_passthrough_json_keeps_stdout_clean(launcher: Any, calls: list[Any]) -> None:
    out = sys.stdout
    try:
        launcher.main(["formula", "d1", "--json"])
        assert sys.stdout is sys.stderr  # сообщения запускателя — мимо JSON
    finally:
        sys.stdout = out


def test_no_database_no_command(launcher: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(launcher, "_need_venv", lambda: None)
    monkeypatch.setattr(launcher, "_ensure_database", lambda: False)
    monkeypatch.setattr(launcher, "_py", lambda *a: pytest.fail("без базы модуль не зовётся"))
    assert launcher.main(["formula", "d1"]) == 1
