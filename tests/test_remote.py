"""Удалённый доступ: MCP по HTTP за токеном (mcp_server.http_app) и адрес Tailscale
(remote.py). Каждый запрос идёт к серверу страницы с токеном того, кто спросил."""

from __future__ import annotations

import asyncio
import json
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kbrag import acl, mcp_server, remote

ACCESS = """
люди:
  Анна:
    базы: [physics.optics]
    токен_sha256: {anna}
  Борис:
    базы: ["."]
    токен_sha256: {boris}
"""


@pytest.fixture
def access_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "access.yaml"
    text = ACCESS.format(anna=acl.token_hash("anna-token"), boris=acl.token_hash("boris-token"))
    path.write_text(text, encoding="utf-8")
    monkeypatch.setattr(acl, "ACCESS_FILE", path)
    acl._CACHE.clear()
    return path


@pytest.fixture
def upstream(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Сервер страницы понарошку: запоминает, с каким токеном к нему пришли."""
    seen: list[str | None] = []

    class Answer:
        headers = type("H", (), {"get_content_type": staticmethod(lambda: "application/json")})

        def __init__(self, token: str | None) -> None:
            self.body = json.dumps({"kbs": [], "token": token}).encode()

        def read(self) -> bytes:
            return self.body

        def __enter__(self) -> Answer:
            return self

        def __exit__(self, *exc: Any) -> None:
            pass

    def urlopen(request: Any, timeout: float) -> Answer:
        token = request.get_header("X-kbrag-token")
        time.sleep(0.05)  # запросы двух людей идут вперемешку
        seen.append(token)
        return Answer(token)

    monkeypatch.setattr(mcp_server.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(mcp_server, "TOKEN", "")  # своего токена у сервера нет
    return seen


@pytest.fixture
def mcp_url(access_file: Path) -> Any:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = mcp_server.serve_http("127.0.0.1", port)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "MCP по HTTP не поднялся"
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(10)


async def _list_kbs(url: str, token: str) -> dict[str, Any]:
    import httpx
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as http:
        async with streamable_http_client(url, http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("list_kbs", {})
                data: dict[str, Any] = json.loads(result.content[0].text)
                return data


def test_http_mcp_needs_token(mcp_url: str, upstream: list[str | None]) -> None:
    import httpx

    body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    head = {"Accept": "application/json, text/event-stream"}
    for auth in ({}, {"Authorization": "Bearer stranger"}, {"Authorization": "Basic anna-token"}):
        got = httpx.post(mcp_url, json=body, headers={**head, **auth}, timeout=10)
        assert got.status_code == 401 and "токен" in got.json()["error"]
    assert upstream == []  # до сервера страницы не дошло ничего


def test_http_mcp_passes_each_persons_token(mcp_url: str, upstream: list[str | None]) -> None:
    async def both() -> list[dict[str, Any]]:
        return list(
            await asyncio.gather(
                _list_kbs(mcp_url, "anna-token"),
                _list_kbs(mcp_url, "boris-token"),
                _list_kbs(mcp_url, "anna-token"),
            )
        )

    anna, boris, anna_again = asyncio.run(both())
    assert anna["token"] == "anna-token" and anna_again["token"] == "anna-token"
    assert boris["token"] == "boris-token"  # токены разных людей не перепутались
    assert sorted(t or "" for t in upstream) == ["anna-token", "anna-token", "boris-token"]


def test_remote_refused_without_people(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acl, "ACCESS_FILE", tmp_path / "нет.yaml")
    acl._CACHE.clear()
    problem = mcp_server.check_remote()
    assert problem and "access add" in problem
    assert mcp_server.main(["--http"]) == 1


def test_remote_allowed_with_people(access_file: Path) -> None:
    assert mcp_server.check_remote() is None


def test_tailscale_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KBRAG_REMOTE_HOST", raising=False)
    monkeypatch.setattr(remote, "value", lambda name, default: "")
    monkeypatch.setattr(remote, "_tailscale", lambda: "tailscale")
    status = {
        "BackendState": "Running",
        "Self": {"TailscaleIPs": ["100.101.102.103", "fd7a::1"], "DNSName": "comp.tail1.ts.net."},
    }

    def run(*args: Any, **kw: Any) -> Any:
        return subprocess.CompletedProcess(args, 0, json.dumps(status), "")

    monkeypatch.setattr(remote.subprocess, "run", run)
    address, why = remote.find()
    assert address == remote.Address("100.101.102.103", ("comp.tail1.ts.net", "comp")) and not why

    status["BackendState"] = "NeedsLogin"
    address, why = remote.find()
    assert address is None and "NeedsLogin" in why


def test_tailscale_missing_and_own_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote, "value", lambda name, default: "")
    monkeypatch.setattr(remote, "_tailscale", lambda: None)
    address, why = remote.find()
    assert address is None and "не установлен" in why
    monkeypatch.setattr(remote, "value", lambda name, default: "10.8.0.5")  # своя VPN
    assert remote.find() == (remote.Address("10.8.0.5"), "")
