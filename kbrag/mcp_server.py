"""MCP-сервер базы знаний — для Claude Code и других клиентов MCP.

Работает поверх запущенного сервера страницы (python kbrag.py serve или web): ходит в
его адреса /api, поэтому модели векторов и реранкера загружены один раз — в том
сервере, — а этот процесс лёгкий и стартует мгновенно. Говорит с клиентом через
stdin/stdout (MCP stdio), ничего другого туда не пишет.

Подключить к Claude Code (один раз, из папки проекта):

    python kbrag.py serve                                      в своём окне
    claude mcp add kbrag -- python D:/projects/AgentBK/kbrag.py mcp

Инструменты:

    list_kbs       дерево баз: пути, названия, сколько статей
    search         фрагменты статей по смыслу и словам, в базе kb (её поддереве)
    get_fragment   фрагмент целиком по chunk_id из выдачи
    get_article    статья: метаданные, ветки, рисунки, число формул; с text — весь текст
    facts          факты графа о сущности с цитатами; без имени — у чего фактов больше всего
    get_formula    формулы статьи: LaTeX, номер «(N)», страница, текст до и после, кто ссылается
    search_formulas  формулы по смыслу — по описанию с контекстом
    get_figure     рисунок статьи картинкой и его подпись

Адрес сервера — KBRAG_SERVE (по умолчанию http://127.0.0.1:8000).
Доступ ограничен (.local/access.yaml) — токен человека в KBRAG_TOKEN.

Удалённо (с другого компьютера) — тот же MCP по HTTP, на удалённой машине ставить
ничего, кроме Claude Code, не нужно:

    python kbrag.py web --remote          страница и MCP по адресу Tailscale (kbrag/remote.py)
    python kbrag.py mcp --http --host 100.x.y.z    только MCP по HTTP (порт 8200)

    claude mcp add --transport http kbrag http://АДРЕС:8200/mcp \\
        --header "Authorization: Bearer ТОКЕН"

Токен — человека из .local/access.yaml (python kbrag.py access add): каждый запрос идёт
к серверу страницы с токеном того, кто спросил, и видит только его базы. Без
access.yaml MCP по HTTP не запускается: открыть базу всем, кто дотянулся до порта, нельзя.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from contextvars import ContextVar
from typing import Any

from .machine import value

BASE = value("KBRAG_SERVE", "http://127.0.0.1:8000")
# Токен входа (python kbrag.py access add): есть .local/access.yaml — без него сервер не
# ответит; инструменты видят только базы этого человека.
TOKEN = value("KBRAG_TOKEN", "")
# MCP по HTTP: токен того, кто прислал этот запрос (заголовок Authorization) — у каждого
# запроса свой, запросы разных людей идут вперемешку.
_REQUEST_TOKEN: ContextVar[str] = ContextVar("kbrag_mcp_token", default="")
HTTP_PORT = 8200
TIMEOUT = 120  # с: поиск с синонимами по большой базе — секунды, не минуты
NOT_RUNNING = (
    "Сервер базы знаний не отвечает на {base}. Запустите его в отдельном окне: "
    "python kbrag.py serve (из папки проекта) — и повторите."
)

INSTRUCTIONS = """База знаний по научным статьям (PDF пользователя), деревом баз на разные
темы. Чтобы ответить по статьям: search (в нужной базе kb — пути из list_kbs), затем
get_fragment или get_article для подробностей. Ссылайся на статьи: название, год,
страницы. facts — связи из графа знаний с дословными цитатами. get_formula и
get_figure — формулы (LaTeX с номером) и рисунки статьи."""


def _url(path: str, params: dict[str, Any]) -> str:
    clean = {k: v for k, v in params.items() if v is not None and v != ""}
    return f"{BASE.rstrip('/')}{path}" + (f"?{urllib.parse.urlencode(clean)}" if clean else "")


def _fetch(path: str, **params: Any) -> tuple[bytes, str]:
    """Ответ сервера: тело и тип. Сервер ответил ошибкой — её JSON (там объяснение)."""
    request = urllib.request.Request(_url(path, params))
    token = _REQUEST_TOKEN.get() or TOKEN
    if token:
        request.add_header("X-KBRAG-Token", token)  # доступ к базам — по людям (acl.py)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.read(), response.headers.get_content_type()
    except urllib.error.HTTPError as exc:
        return exc.read(), "application/json"
    except (urllib.error.URLError, OSError) as exc:
        message = NOT_RUNNING.format(base=BASE) + f" ({type(exc).__name__})"
        return json.dumps({"error": message}, ensure_ascii=False).encode(), "application/json"


def get_json(path: str, **params: Any) -> dict[str, Any]:
    body, _ = _fetch(path, **params)
    try:
        data = json.loads(body.decode("utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"error": f"сервер ответил не JSON: {body[:200]!r}"}
    return data if isinstance(data, dict) else {"result": data}


def _kb(kb: str | None) -> str | None:
    """Путь базы для адреса: «» и «.» — всё дерево (на сервере это «.»)."""
    if kb is None:
        return None
    return kb.strip() or "."


# --------------------------------------------------------------------------- #
#  Инструменты — обычные функции (их проверяют и без MCP), регистрирует их build()
# --------------------------------------------------------------------------- #
def list_kbs() -> dict[str, Any]:
    """Дерево баз знаний: путь (для параметра kb других инструментов), название, сколько
    статей в самой базе и вместе с базами под ней; default — база по умолчанию."""
    return get_json("/api/kbs")


def search(query: str, kb: str | None = None, limit: int = 10, per_doc: int = 2) -> dict[str, Any]:
    """Найти фрагменты статей по смыслу и по точным словам (с синонимами из словаря базы).

    kb — путь базы из list_kbs: ищется в ней и во всех базах под ней; «.» — во всём
    дереве; не задано — в базе по умолчанию. Запрос — словами по теме, можно по-русски и
    по-английски. У каждого фрагмента chunk_id (для get_fragment), doc_id (для
    get_article), название, год, страницы и текст."""
    return get_json("/api/search", q=query, kb=_kb(kb), limit=limit, per_doc=per_doc)


def get_fragment(chunk_id: int) -> dict[str, Any]:
    """Фрагмент статьи целиком по chunk_id из выдачи search: текст, разделы, страницы,
    статья (название, год, авторы, ссылка)."""
    return get_json("/api/fragment", chunk_id=chunk_id)


def get_article(
    doc_id: str, text: bool = False, offset: int = 0, limit: int = 20
) -> dict[str, Any]:
    """Статья по doc_id: название, авторы, год, журнал, DOI, ссылка, язык, в каких базах
    дерева лежит, рисунки (номер, страница, подпись), сколько формул.

    text=True — ещё и текст по фрагментам, страницей: limit фрагментов (до 50, и не
    больше ~60 тыс. знаков) начиная с offset. В ответе fragments_total — сколько их
    всего, next_offset — с какого брать следующую страницу (null — это последняя)."""
    if not text:
        return get_json("/api/article", doc_id=doc_id)
    return get_json("/api/article", doc_id=doc_id, text="1", offset=offset, limit=limit)


def facts(name: str | None = None, kb: str | None = None) -> dict[str, Any]:
    """Факты графа знаний (выписаны из статей, с дословными цитатами).

    name — сущность (место, метод, процесс…): все факты о ней со статьями и цитатами.
    Без name — сущности, у которых фактов больше всего. kb — база (её поддерево)."""
    if name:
        return get_json("/api/dataset", name=name, kb=_kb(kb))
    return get_json("/api/datasets", kb=_kb(kb))


def get_formula(doc_id: str, number: str | None = None) -> dict[str, Any]:
    """Формулы статьи: LaTeX, номер «(N)», страница, текст из PDF и абзацы до и после
    (по ним понятно, что формула значит). number — одна формула, «(N)» из текста статьи."""
    return get_json("/api/formula", doc_id=doc_id, number=number)


def search_formulas(query: str, kb: str | None = None, limit: int = 10) -> dict[str, Any]:
    """Найти формулы по смыслу: query — что за формула словами («плотность через
    пористость»). Ищется по описанию формулы с её контекстом в статье; у каждой — LaTeX,
    номер, статья, страница."""
    return get_json("/api/formulas", q=query, kb=_kb(kb), limit=limit)


def search_figures(query: str, kb: str | None = None, limit: int = 10) -> dict[str, Any]:
    """Найти рисунки по смыслу: query — что на рисунке («карта разломов», «разрез
    скважины»). Ищется по подписи и описанию моделью со зрением; у каждого — статья,
    n (для get_figure), номер в статье, страница, подпись."""
    return get_json("/api/figures", q=query, kb=_kb(kb), limit=limit)


def get_figure(doc_id: str, n: int) -> list[Any]:
    """Рисунок n статьи (номер из get_article: figures) — картинкой, и его подпись."""
    from mcp.server.mcpserver import Image

    body, kind = _fetch("/api/figure", doc_id=doc_id, n=n)
    if not kind.startswith("image/"):
        try:
            return [json.loads(body.decode("utf-8")).get("error") or "рисунка нет"]
        except (UnicodeDecodeError, json.JSONDecodeError):
            return ["рисунка нет"]
    known = get_json("/api/figure", doc_id=doc_id).get("figures") or []
    meta: dict[str, Any] = next((f for f in known if f.get("n") == n), {})
    caption = meta.get("caption") or "без подписи"
    where = f", с. {meta['page']}" if meta.get("page") else ""
    return [Image(data=body, format=kind.split("/", 1)[1]), f"Рисунок {n}{where}. {caption}"]


TOOLS: list[Callable[..., Any]] = [
    list_kbs,
    search,
    get_fragment,
    get_article,
    facts,
    get_formula,
    search_formulas,
    search_figures,
    get_figure,
]


def build() -> Any:
    from mcp.server.mcpserver import MCPServer

    from . import __version__

    server = MCPServer("kbrag", instructions=INSTRUCTIONS, version=__version__)
    for tool in TOOLS:
        # Картинка с подписью — не JSON: без схемы ответа.
        server.add_tool(tool, structured_output=False if tool is get_figure else None)
    return server


# --------------------------------------------------------------------------- #
#  MCP по HTTP — для удалённых машин
# --------------------------------------------------------------------------- #
def _bearer(scope: dict[str, Any]) -> str:
    for name, raw in scope.get("headers") or []:
        if name.lower() == b"authorization":
            kind, _, token = raw.decode("latin-1").partition(" ")
            return token.strip() if kind.lower() == "bearer" else ""
    return ""


def http_app(host_names: list[str]) -> Any:
    """Приложение ASGI: MCP по HTTP за проверкой токена. host_names — по каким именам и
    адресам к нему обращаются (Host): остальные отбиваются — защита от DNS rebinding."""
    from mcp.server.transport_security import TransportSecuritySettings

    from . import acl

    names = sorted({*host_names, "127.0.0.1", "localhost"})
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[f"{n}:*" for n in names] + names,
        allowed_origins=[f"http://{n}:*" for n in names],
    )
    # stateless: у каждого запроса — свой разбор, без памяти о сеансе; сервер можно
    # перезапускать, клиенты этого не заметят.
    inner = build().streamable_http_app(stateless_http=True, transport_security=security)

    async def deny(send: Any, why: str) -> None:
        body = json.dumps({"error": why}, ensure_ascii=False).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json; charset=utf-8"),
                    (b"www-authenticate", b'Bearer realm="kbrag"'),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await inner(scope, receive, send)  # lifespan — запуск и остановка
            return
        token = _bearer(scope)
        try:
            person = acl.load().by_token(token) if token else None
        except acl.AccessError:
            person = None  # файл доступа с ошибкой — никого (как и страница)
        if person is None:
            await deny(send, "нужен токен: Authorization: Bearer … (python kbrag.py access add)")
            return
        mark = _REQUEST_TOKEN.set(token)
        try:
            await inner(scope, receive, send)
        finally:
            _REQUEST_TOKEN.reset(mark)

    return app


def check_remote() -> str | None:
    """Почему MCP по HTTP запускать нельзя (None — можно): нет людей с токенами."""
    from . import acl

    try:
        access = acl.load()
    except acl.AccessError as exc:
        return f"файл доступа .local/access.yaml с ошибкой: {exc}"
    if not access.configured:
        return (
            "удалённый доступ — только по токенам, а людей с доступом нет. Добавьте: "
            "python kbrag.py access add ИМЯ --kb . (печатает токен)"
        )
    return None


def serve_http(host: str, port: int = HTTP_PORT, host_names: list[str] | None = None) -> Any:
    """Сервер uvicorn с MCP по HTTP (не запущен: .run() — здесь же, или в потоке)."""
    import uvicorn

    app = http_app([host, *(host_names or [])])
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="on")
    return uvicorn.Server(config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MCP-сервер базы знаний KBRAG")
    parser.add_argument("--http", action="store_true", help="MCP по HTTP, а не stdio")
    parser.add_argument("--host", default="127.0.0.1", help="адрес для --http")
    parser.add_argument("--port", type=int, default=HTTP_PORT, help="порт для --http")
    args = parser.parse_args(argv)
    if not args.http:
        build().run()
        return 0
    problem = check_remote()
    if problem:
        print(f"MCP по HTTP не запущен: {problem}", file=sys.stderr)
        return 1
    print(f"MCP по HTTP: http://{args.host}:{args.port}/mcp (сервер страницы — {BASE})")
    serve_http(args.host, args.port).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
