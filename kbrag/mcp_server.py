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
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from .machine import value

BASE = value("KBRAG_SERVE", "http://127.0.0.1:8000")
# Токен входа (python kbrag.py access add): есть .local/access.yaml — без него сервер не
# ответит; инструменты видят только базы этого человека.
TOKEN = value("KBRAG_TOKEN", "")
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
    if TOKEN:
        request.add_header("X-KBRAG-Token", TOKEN)  # доступ к базам — по людям (acl.py)
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


def main() -> int:
    build().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
