"""Общие ключи командной строки.

Адрес базы, устройство для векторов и сервер языковой модели нужны в пяти местах:
загрузка, разметка, чат-бот, веб-сервер, оценка. Раньше каждое объявляло их
само, и значения по умолчанию расходились (где-то был mps, где-то нет). Теперь
ключи объявляются здесь и везде одинаковы.
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from .llm import DEFAULT_MODEL, LLM_HOST

if TYPE_CHECKING:
    from .kb import KB

DEVICES = ["auto", "cuda", "cpu", "mps"]


def add_db_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dsn", default=None, help="адрес базы, иначе KBRAG_DSN")
    parser.add_argument(
        "--kb",
        default=None,
        help="узел дерева баз, например physics.optics; «.» — всё дерево; "
        "иначе KBRAG_KB или база_по_умолчанию в config/kbs/kb.yaml",
    )


def kb_from_args(path: str | None) -> KB | None:
    """База дерева из ключа --kb (None — по умолчанию). Нет такой или путь неверный —
    сообщение в stderr и None: команда выходит с кодом 1."""
    from .kb import KBError
    from .kb import get as get_kb

    try:
        return get_kb(path)
    except KBError as exc:
        print(f"База: {exc}", file=sys.stderr)
        return None


def add_embedder_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="auto",
        help="где считать векторы BGE-M3: auto — на видеокарте, если она есть",
    )


def add_llm_args(parser: argparse.ArgumentParser, model: str = DEFAULT_MODEL) -> None:
    parser.add_argument("--model", default=model, help="языковая модель (имя на сервере vLLM)")
    parser.add_argument("--llm-host", default=LLM_HOST, help="адрес сервера модели (vLLM)")
