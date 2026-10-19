"""Веб-интерфейс к базе знаний: поиск, статьи, чат-бот и граф связей.

    python -m kbrag.web.server            открыть http://localhost:8000
    python -m kbrag.web.server --port 9000

Модель эмбеддингов грузится один раз при запуске, а не на каждый запрос:
в терминале каждый поиск стоит полминуты на загрузку весов, здесь — доли
секунды.

Никаких новых библиотек: сервер из стандартной поставки Python, страница —
один файл рядом. Наружу ничего не смотрит, слушает только localhost.

Для страницы:

    GET  /api/stats                    сколько статей и фрагментов
    GET  /api/search?q=...             гибридный поиск
    GET  /api/documents                список статей
    GET  /api/document?doc_id=...      разбор статьи по фрагментам
    GET  /api/chat/status              жив ли сервер модели и та ли на нём модель
    POST /api/chat                     вопрос чат-боту; ответ потоком, строка JSON на событие
    GET  /api/network                  граф: сущности, связи с цитатами, статьи
    GET  /api/facts?chunks=12,34       связи графа, выписанные из этих фрагментов (под ответом)
    GET  /api/datasets                 у каких сущностей больше всего фактов
    GET  /api/dataset?name=...         датасет сущности: все факты о ней с цитатами
                                       (POST — то же JSON в теле); &format=csv — для Excel

С другой машины к нему ходят через SSH-туннель: порт снаружи не открывается.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from ..chat import answer as chat
from ..chat.settings import Settings
from ..common import add_db_args, add_embedder_args, add_llm_args, kb_from_args
from ..graph import api as graph_api
from ..graph import store as graph
from ..graph.synonyms import SynonymsError, synonym_variants
from ..graph.synonyms import files as synonym_files
from ..graph.synonyms import load as load_synonyms
from ..index import db
from ..index.embed import build_embedder
from ..index.search import hit_dict, hybrid_search
from ..kb import KBError, check_path
from ..kb import get as get_kb

PAGE = Path(__file__).with_name("index.html")
STATIC = Path(__file__).with_name("static")  # KaTeX — копией: страница работает без интернета
STATIC_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".woff2": "font/woff2",
    ".png": "image/png",
}


class State:
    """Общее на весь сервер: адрес базы, загруженная модель, словари синонимов баз."""

    dsn: str | None = None
    embedder: Any = None
    chat_settings = Settings()  # .kb — узел дерева баз по умолчанию (ключ --kb)
    _syn: dict[str, tuple[Any, Any]] = {}

    @classmethod
    def kb(cls, value: Any = None) -> str:
        """Узел дерева баз запроса (параметр kb); нет — тот, с которым запущен сервер.
        Пустая строка или «.» — всё дерево. Негодный путь — KBError (ответ 400)."""
        return cls.chat_settings.kb if value is None else check_path(str(value))

    @classmethod
    def synonyms(cls, kb: str) -> Any:
        """Словарь базы kb перечитывается, если файлы поменяли, — сервер можно не
        перезапускать."""
        base = get_kb(kb)
        key = tuple((str(f), f.stat().st_mtime) for f in synonym_files(kb=base))
        hit = cls._syn.get(base.path)
        if hit is None or hit[0] != key:
            hit = (key, load_synonyms(kb=base))
            cls._syn[base.path] = hit
        return hit[1]


def _hit_to_dict(hit: Any) -> dict[str, Any]:
    # doc_id и ord — чтобы из выдачи открыть разбор статьи и показать, из какого места
    # взят кусок; chunk_id — по нему фрагмент целиком (/api/fragment, MCP get_fragment).
    return hit_dict(hit)


class Handler(BaseHTTPRequestHandler):
    server_version = "kbrag"

    # Стандартный лог пишет каждую картинку — в консоли от него только шум.
    def log_message(self, fmt: Any, *args: Any) -> None:
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict[str, Any], code: int = 200) -> None:
        self._send(
            code,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def do_GET(self) -> None:  # noqa: N802 — имя задано базовым классом
        route = urlparse(self.path)
        query = parse_qs(route.query)

        try:
            asked = query.get("kb")
            kb = State.kb(asked[0] if asked else None)
        except KBError as exc:
            self._json({"error": f"база: {exc}"}, 400)
            return
        try:
            if route.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif route.path.startswith("/static/"):
                self._static(route.path[len("/static/") :])
            elif route.path == "/api/stats":
                self._json(self._stats(kb))
            elif route.path == "/api/kbs":
                self._json(self._kbs(kb))
            elif route.path == "/api/search":
                self._json(self._search(query, kb))
            elif route.path == "/api/documents":
                with db.connect(State.dsn) as conn:
                    self._json({"documents": db.documents(conn, kb=kb)})
            elif route.path == "/api/datasets":
                self._graph(kb, lambda conn, syn: graph_api.datasets_response(conn, syn, kb=kb))
            elif route.path == "/api/dataset":
                self._dataset({k: v[0] for k, v in query.items() if v}, kb)
            elif route.path == "/api/network":
                self._graph(kb, lambda conn, syn: graph_api.network_response(conn, syn, kb=kb))
            elif route.path == "/api/facts":
                raw = (query.get("chunks") or [""])[0].split(",")
                ids = [int(x) for x in raw if x.strip().isdigit()][:40]
                self._graph(kb, lambda conn, syn: graph_api.facts_response(conn, syn, ids, kb))
            elif route.path == "/api/chat/status":
                status = chat.llm_status(State.chat_settings.host, State.chat_settings.model)
                # Сверка статей по умолчанию — для галочки на странице, пока её не трогали.
                self._json({**status, "conflicts": State.chat_settings.check_conflicts})
            elif route.path == "/api/document":
                doc_id = (query.get("doc_id") or [""])[0]
                with db.connect(State.dsn) as conn:
                    self._json({"doc_id": doc_id, "chunks": db.document_chunks(conn, doc_id)})
            elif route.path == "/api/article":
                self._article(query)
            elif route.path == "/api/fragment":
                self._fragment(query)
            elif route.path == "/api/formula":
                self._formula(query)
            elif route.path == "/api/formulas":
                self._formulas(query, kb)
            elif route.path == "/api/formula_image":
                self._formula_image(query)
            elif route.path == "/api/figure":
                self._figure(query)
            elif route.path == "/api/figures":
                self._figures(query, kb)
            else:
                self._json({"error": "нет такой страницы"}, 404)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # страницу закрыли, пока отвечали
        except Exception as exc:  # noqa: BLE001 — ошибка уходит на страницу, сервер живёт
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path)
        try:
            if route.path not in ("/api/chat", "/api/dataset"):
                self._json({"error": "нет такого адреса"}, 404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > 256 * 1024:
                self._json({"error": "запрос слишком большой"}, 413)
                return
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._json({"error": "тело запроса — не JSON", "detail": str(exc)}, 400)
                return
            try:
                kb = State.kb(payload.pop("kb", None))
            except KBError as exc:
                self._json({"error": f"база: {exc}"}, 400)
                return
            if route.path == "/api/chat":
                self._chat(payload, kb)
            else:
                self._dataset(payload, kb)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    # -- чат-бот ------------------------------------------------------------ #
    def _chat(self, payload: dict[str, Any], kb: str) -> None:
        """Ответ потоком: строка JSON на событие, страница рисует по мере прихода."""
        history = payload.get("history") if isinstance(payload.get("history"), list) else []
        settings = replace(State.chat_settings, kb=kb)
        if isinstance(payload.get("conflicts"), bool):  # галочка «сверять статьи» на странице
            settings = replace(settings, check_conflicts=payload["conflicts"])
        if isinstance(payload.get("fast"), bool):  # галочка «быстрый режим»
            settings = replace(settings, fast=payload["fast"])
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        events = None
        try:
            with db.connect(State.dsn) as conn:
                db.autocommit(conn)  # ответ пишется минутами — без открытой транзакции
                events = chat.answer(
                    conn, State.embedder, str(payload.get("question") or ""), history, settings
                )
                sources: list[dict[str, Any]] = []
                for event in events:
                    if event.get("type") == "sources":
                        sources = list(event.get("sources") or [])
                    self.wfile.write((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()
                    if event.get("type") == "done" and sources:
                        extra = _answer_figures(conn, sources, event.get("used"))
                        if extra:
                            line = {"type": "figures", "figures": extra}
                            self.wfile.write((json.dumps(line, ensure_ascii=False) + "\n").encode())
                            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # страницу закрыли или нажали «Стоп» — не страшно
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            try:
                line = {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
                self.wfile.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
            except OSError:
                pass
        finally:
            if events is not None:
                events.close()  # закрывает и поток от модели: она перестаёт писать

    # -- граф знаний: сеть, датасеты ------------------------------------------ #
    def _graph(self, kb: str, build: Any) -> None:
        """build(conn, синонимы базы kb) → ответ. Неизвестное имя — 400, словарь не
        читается — 500."""
        try:
            syn = State.synonyms(kb)
            with db.connect(State.dsn) as conn:
                self._json(build(conn, syn))
        except graph_api.RequestError as exc:
            self._json(exc.payload(), 400)
        except SynonymsError as exc:
            self._json({"error": f"словарь синонимов: {exc}"}, 500)

    def _dataset(self, payload: dict[str, Any], kb: str) -> None:
        """Датасет сущности: JSON, а с format=csv — таблица для Excel."""
        payload = dict(payload)
        payload.pop("kb", None)
        if str(payload.pop("format", "")).lower() != "csv":
            self._graph(kb, lambda conn, syn: graph_api.dataset_response(conn, syn, payload, kb))
            return
        try:
            with db.connect(State.dsn) as conn:
                data = graph_api.dataset_response(conn, State.synonyms(kb), payload, kb)
        except graph_api.RequestError as exc:
            self._json(exc.payload(), 400)
            return
        body = graph_api.dataset_csv(data).encode("utf-8")
        name = quote(f"датасет-{data['name']}.csv")
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header(
            "Content-Disposition", f"attachment; filename=\"dataset.csv\"; filename*=UTF-8''{name}"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- действия ---------------------------------------------------------- #
    def _kbs(self, kb: str) -> dict[str, Any]:
        """Дерево баз для выбора сверху страницы и то, что нужно странице о выбранной:
        название и примеры вопросов (kb.yaml, «примеры_вопросов»)."""
        from ..kbtree import nodes

        try:
            with db.connect(State.dsn) as conn:
                tree = nodes(conn)
        except Exception:  # noqa: BLE001 — база не отвечает: дерево по папкам настроек
            tree = nodes()
        base = get_kb(kb)
        examples = base.settings.get("примеры_вопросов") or []
        return {
            **tree,
            "server": State.chat_settings.kb,  # с какой базой запущен сервер (--kb)
            "current": kb,
            "title": base.title,
            "examples": [str(x) for x in examples if str(x).strip()][:8],
        }

    def _static(self, rel: str) -> None:
        """Файл из web/static — только изнутри неё и только известных типов."""
        path = (STATIC / rel).resolve()
        kind = STATIC_TYPES.get(path.suffix.lower())
        if kind is None or STATIC.resolve() not in path.parents or not path.is_file():
            self._json({"error": "нет такого файла"}, 404)
            return
        self._send(200, path.read_bytes(), kind)

    # -- статья, фрагмент, формулы, рисунки (их берёт и MCP-сервер, kbrag/mcp.py) -- #
    def _article(self, query: dict[str, Any]) -> None:
        """Статья: метаданные, ветки, рисунки, формулы; с text=1 — и фрагменты по порядку."""
        from ..library import objects

        doc_id = (query.get("doc_id") or [""])[0]
        with db.connect(State.dsn) as conn:
            found = db.document(conn, doc_id)
            chunks = db.document_chunks(conn, doc_id) if found and _flag(query, "text") else None
        if found is None:
            self._json({"error": f"статьи «{doc_id}» в базе нет"}, 404)
            return
        record = objects.record(doc_id) or {}
        found.update(
            pdf_path=record.get("pdf_path"),
            figures=objects.figures(doc_id),
            formulas=len(objects.formulas(doc_id)),
        )
        if chunks is not None:
            found["fragments"] = chunks
        self._json(found)

    def _fragment(self, query: dict[str, Any]) -> None:
        from ..graph import store

        raw = (query.get("chunk_id") or [""])[0]
        if not raw.isdigit():
            self._json({"error": "нужен chunk_id — число из выдачи поиска"}, 400)
            return
        with db.connect(State.dsn) as conn:
            found = store.fragments(conn, [int(raw)]).get(int(raw))
        if found is None:
            self._json({"error": f"фрагмента {raw} в базе нет"}, 404)
            return
        self._json(found)

    def _formula(self, query: dict[str, Any]) -> None:
        """Формулы статьи из базы (index/formulas.py) — с фрагментами, что на них
        ссылаются; с number=N — одна, «(N)». В базе формул статьи нет (загружена до
        таблицы формул) — из её разметки docling, без ссылок."""
        from ..index import formulas
        from ..library import objects

        doc_id = (query.get("doc_id") or [""])[0]
        number = (query.get("number") or [""])[0] or None
        with db.connect(State.dsn) as conn:
            found = formulas.of_document(conn, doc_id, number)
            stored = bool(found) or formulas.has_rows(conn, doc_id)
        if not stored:
            found = objects.formulas(doc_id)
            if number:
                found = [f for f in found if f["number"] == str(number).strip("()")]
        if number:
            if not found:
                every = formulas_numbers(doc_id)
                self._json({"error": f"формулы ({number}) в статье нет", "known": every}, 404)
                return
            self._json({"doc_id": doc_id, **found[0]})
            return
        self._json({"doc_id": doc_id, "formulas": found})

    def _formulas(self, query: dict[str, Any], kb: str) -> None:
        """Формулы по смыслу — по их описанию словами, в поддереве базы kb."""
        from ..index import formulas

        text = (query.get("q") or [""])[0].strip()
        if not text:
            self._json({"error": "нужен запрос q — что за формула"}, 400)
            return
        limit = (query.get("limit") or ["10"])[0]
        with db.connect(State.dsn) as conn:
            found = formulas.search(
                conn, State.embedder, text, kb, int(limit) if limit.isdigit() else 10
            )
        self._json({"query": text, "kb": kb, "formulas": found})

    def _formula_image(self, query: dict[str, Any]) -> None:
        """Формула картинкой: вырезка со страницы PDF, иначе нарисованный LaTeX."""
        from ..index import formulas
        from ..library import objects

        raw = (query.get("id") or [""])[0]
        if not raw.isdigit():
            self._json({"error": "нужен id формулы"}, 400)
            return
        with db.connect(State.dsn) as conn:
            item = formulas.by_id(conn, int(raw))
        png = objects.formula_png(item) if item else None
        if png is None:
            self._json({"error": "картинки формулы нет", "latex": (item or {}).get("latex")}, 404)
            return
        self._send(200, png, "image/png")

    def _figure(self, query: dict[str, Any]) -> None:
        """Картинка рисунка: n статьи (по порядку) или id в таблице рисунков. Без них —
        перечень рисунков статьи из базы, с теми, кто на них ссылается (label=3 — один,
        «Рис. 3»); в базе нет — из записи статьи."""
        from ..index import figures
        from ..library import objects

        doc_id = (query.get("doc_id") or [""])[0]
        raw = (query.get("n") or [""])[0]
        ident = (query.get("id") or [""])[0]
        if ident.isdigit():
            with db.connect(State.dsn) as conn:
                item = figures.by_id(conn, int(ident))
            doc_id, raw = (str(item["doc_id"]), str(item["n"])) if item else (doc_id, "")
            if item is None:
                self._json({"error": f"рисунка {ident} нет"}, 404)
                return
        if not raw.isdigit():
            label = (query.get("label") or [""])[0] or None
            with db.connect(State.dsn) as conn:
                found = figures.of_document(conn, doc_id, label)
                stored = bool(found) or figures.has_rows(conn, doc_id)
            if not stored:
                found = objects.figures(doc_id)
            if label and not found:
                self._json({"error": f"рисунка «{label}» в статье нет"}, 404)
                return
            self._json({"doc_id": doc_id, "figures": found})
            return
        path = objects.figure_file(doc_id, int(raw))
        if path is None:
            self._json({"error": f"рисунка {raw} у статьи «{doc_id}» нет"}, 404)
            return
        kind = {".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(path.suffix.lower(), "image/png")
        self._send(200, path.read_bytes(), kind)

    def _figures(self, query: dict[str, Any], kb: str) -> None:
        """Рисунки по смыслу — по подписи и описанию, в поддереве базы kb."""
        from ..index import figures

        text = (query.get("q") or [""])[0].strip()
        if not text:
            self._json({"error": "нужен запрос q — что на рисунке"}, 400)
            return
        limit = (query.get("limit") or ["10"])[0]
        with db.connect(State.dsn) as conn:
            found = figures.search(
                conn, State.embedder, text, kb, int(limit) if limit.isdigit() else 10
            )
        self._json({"query": text, "kb": kb, "figures": found})

    def _stats(self, kb: str) -> dict[str, Any]:
        with db.connect(State.dsn) as conn:
            info = db.stats(conn, kb)
        lo, hi = info["years"]
        return {
            "kb": kb,
            "kb_title": get_kb(kb).title,
            "documents": info["documents"],
            "chunks": info["chunks"],
            "year_from": lo,
            "year_to": hi,
        }

    def _search(self, query: dict[str, Any], kb: str) -> dict[str, Any]:
        text = (query.get("q") or [""])[0].strip()
        if not text:
            return {"hits": [], "query": ""}

        def number(name: Any, default: Any = None) -> Any:
            raw = (query.get(name) or [""])[0]
            return int(raw) if raw.isdigit() else default

        def fraction(name: Any, default: Any = 0.0) -> Any:
            raw = (query.get(name) or [""])[0]
            try:
                return float(raw)
            except ValueError:
                return default

        # Другие написания из словаря синонимов: «Донбасс» ищется и как «Donetsk basin».
        try:
            also = synonym_variants(text, State.synonyms(kb))
        except SynonymsError:
            also = []
        with db.connect(State.dsn) as conn:
            hits = hybrid_search(
                conn,
                State.embedder,
                text,
                limit=number("limit", 10),
                year_from=number("year_from"),
                max_per_doc=number("per_doc", 2),
                min_similarity=fraction("min_similarity"),
                require_words=(query.get("require_words") or [""])[0] == "1",
                alternatives=also,
                kb=kb,
            )
        return {"query": text, "also": also, "hits": [_hit_to_dict(h) for h in hits]}


def _answer_figures(conn: Any, sources: list[dict[str, Any]], used: Any) -> list[dict[str, Any]]:
    """Рисунки фрагментов ответа — для страницы (адрес картинки, подпись, к какому [n]).
    Не вышло (таблицы ещё нет) — без рисунков: ответ уже дошёл."""
    from ..index import figures

    try:
        chosen = figures.for_answer(conn, sources, used if isinstance(used, list) else [])
    except Exception:  # noqa: BLE001
        return []
    return [
        {
            "id": f["id"],
            "source": f["source"],
            "label": f.get("label"),
            "n": f["n"],
            "doc_id": f["doc_id"],
            "page": f.get("page"),
            "caption": f.get("caption") or "",
            "title": f.get("title") or "",
            "url": f"/api/figure?id={f['id']}",
        }
        for f in chosen
    ]


def formulas_numbers(doc_id: str) -> list[str]:
    """Какие номера формул у статьи есть — подсказка, если спросили не тот."""
    from ..index import formulas

    with db.connect(State.dsn) as conn:
        found = formulas.of_document(conn, doc_id)
    return [str(f["number"]) for f in found if f.get("number")]


def _flag(query: dict[str, Any], name: str) -> bool:
    return (query.get(name) or [""])[0].lower() in ("1", "true", "да", "yes")


class Server(ThreadingHTTPServer):
    """Порт — только наш. Стандартный сервер Python ставит SO_REUSEADDR, и на Windows
    второй запуск молча садился на тот же порт: страница попадала то в старый процесс
    со старым кодом, то в новый, а в каждом работал свой Телеграм-бот."""

    allow_reuse_address = False

    def server_bind(self) -> None:
        import socket

        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Страницу закрыли или обновили, пока сервер отвечал, — не ошибка, журнал не засоряем."""
        import sys as _sys

        if isinstance(
            _sys.exc_info()[1], (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)
        ):
            return
        super().handle_error(request, client_address)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Веб-интерфейс к базе знаний KBRAG")
    parser.add_argument("--port", type=int, default=8000)
    add_db_args(parser)
    add_embedder_args(parser)
    parser.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    parser.add_argument(
        "--tg", action="store_true", help="заодно запустить Телеграм-бота (модель векторов общая)"
    )
    add_llm_args(parser)
    args = parser.parse_args(argv)

    # Порт — первым делом: если сервер уже запущен в другом окне, второй не стартует
    # (и не запускает второго Телеграм-бота).
    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError:
        print(
            f"Порт {args.port} занят: сервер уже запущен в другом окне. Закройте его "
            f"(Ctrl+C) и запустите заново — или другой порт: --port {args.port + 1}",
            file=sys.stderr,
        )
        return 1

    State.dsn = args.dsn
    base = kb_from_args(args.kb)
    if base is None:
        server.server_close()
        return 1
    State.chat_settings = Settings(model=args.model, host=args.llm_host, kb=base.path)
    print(f"База: {base.title} ({base.path or 'всё дерево'})")
    try:
        syn = State.synonyms(base.path)
        print(f"Словарь синонимов: {len(syn.groups)} имён")
    except SynonymsError as exc:
        print(f"Словарь синонимов не читается: {exc}", file=sys.stderr)
        server.server_close()
        return 1

    # Проверяем базу до старта: лучше упасть здесь, чем показать пустую страницу.
    try:
        with db.connect(args.dsn) as conn:
            info = db.stats(conn, base.path)
            graph.init(conn)  # таблицы графа — один раз при старте, не на каждый запрос
    except Exception as exc:  # noqa: BLE001
        print(f"База недоступна: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Поднимите её: docker compose up -d", file=sys.stderr)
        server.server_close()
        return 1

    print(f"В базе: {info['documents']} статей, {info['chunks']} чанков")
    print("Загружаю модель эмбеддингов (один раз на весь запуск)...")
    State.embedder = build_embedder(device=args.device)
    State.embedder.encode_one("прогрев")  # чтобы первый поиск не ждал загрузку весов

    if args.tg:
        from ..tg.bot import start_in_thread

        try:
            start_in_thread(State.dsn, State.embedder, State.chat_settings)
        except Exception as exc:  # noqa: BLE001 — без бота страница всё равно работает
            print(f"Телеграм-бот не запустился: {exc}", file=sys.stderr)
    else:
        from ..tg.access import read_token

        if read_token():
            print(
                "Телеграм-бот НЕ запущен: токен есть, но нужен ключ --tg — "
                "python kbrag.py web --tg"
            )

    address = f"http://localhost:{args.port}"
    status = chat.llm_status(State.chat_settings.host, State.chat_settings.model)
    print(
        f"Чат-бот: {State.chat_settings.model} — " + ("готов" if status["ok"] else status["error"])
    )
    print(f"\nГотово: {address}")
    print("Остановить — Ctrl+C\n")

    if not args.no_browser:
        import webbrowser

        webbrowser.open(address)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Остановлен.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
