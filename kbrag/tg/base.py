"""Бот: постоянные, разговор (Session) и общее для частей класса Bot (BotBase)."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import acl
from . import ui
from .access import KB_FILE, USERS_FILE, read_choice
from .api import Telegram
from .ui import esc

EDIT_EVERY = 1.5  # как часто обновлять сообщение, пока модель пишет, с
HISTORY_TURNS = 4  # сколько прошлых реплик помнит разговор
KEEP = 20  # сколько последних ответов и списков помнить для кнопок
SEARCH_HITS = 20
HELP = ui.HELP

# Ответов модели одновременно. vLLM считает одновременные запросы одной пачкой;
# сверх этого числа вопросы ждут здесь, и человек видит своё место в очереди.
MAX_PARALLEL = max(1, int(os.environ.get("KBRAG_PARALLEL", "4")))
MAX_SESSIONS = 300  # разговоров в памяти; старые забываются
MAX_TERRITORIES = 400  # сущностей в разговоре (для кнопок картинки связей)
DENIED_EVERY = 600  # как часто чужому напоминать его id, с
SEEN_UPDATES = 1000  # сколько последних update_id помнить, чтобы не взять дважды

COMMANDS = [
    ("search", "поиск фрагментов статей"),
    ("articles", "статьи в базе"),
    ("facts", "факты о чём-то из статей, с цитатами"),
    ("stats", "что в базе и готова ли модель"),
    ("kb", "с какой базой работать"),
    ("fast", "быстрый режим: вкл/выкл"),
    ("conflicts", "сверка статей перед ответом: вкл/выкл"),
    ("new", "новый разговор"),
    ("help", "что умеет бот"),
]

STALE = "Эта кнопка от старого сообщения — бот его уже не помнит (перезапускали?). Спросите заново."


@dataclass
class Session:
    history: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)  # последнего ответа — для /f
    answers: dict[int, dict[str, Any]] = field(default_factory=dict)  # ответы — для кнопок под ними
    lists: dict[int, dict[str, Any]] = field(default_factory=dict)  # выдачи с листанием
    territories: list[dict[str, Any]] = field(default_factory=list)  # список датасетов, как показан
    awaiting: str | None = None  # ждём от человека: "search" — что искать
    busy: bool = False  # модель сейчас пишет ответ
    conflicts: bool | None = None  # сверять статьи (/conflicts); None — как в настройках бота
    fast: bool = False  # быстрый режим (/fast): без разбора вопроса моделью и без сверки
    kbs: list[str] = field(default_factory=list)  # базы, как показаны в /kb (кнопки — номером)
    stop: threading.Event = field(default_factory=threading.Event)
    counter: int = 0

    def next_id(self) -> int:
        self.counter += 1
        return self.counter

    def remember(self, store: dict[int, dict[str, Any]], value: dict[str, Any]) -> int:
        key = self.next_id()
        store[key] = value
        while len(store) > KEEP:
            store.pop(next(iter(store)))
        return key


class BotBase:
    """Общее для частей бота (bot.Bot собран из них): поля, база чата, сверка, защита
    обработчика — и объявления методов, которые одни части зовут у других."""

    def __init__(
        self,
        tg: Telegram,
        dsn: str | None,
        embedder: Any,
        chat_settings: Any,
        users_file: Path = USERS_FILE,
        log: Any = print,
        kb_file: Path = KB_FILE,
    ) -> None:
        self.tg = tg
        self.dsn = dsn
        self.embedder = embedder
        self.chat_settings = chat_settings
        self.users_file = users_file
        self.log = log
        self.sessions: dict[int, Session] = {}
        self.offset: int | None = None
        self.background = False  # в run() — ответы модели в отдельном потоке
        self.workers: dict[int, threading.Thread] = {}
        self.slots = threading.BoundedSemaphore(MAX_PARALLEL)  # ответов одновременно
        self.lock = threading.Lock()
        self.waiting = 0  # сколько ответов ждут очереди
        self.denied: dict[int, float] = {}  # чужой → когда последний раз отвечали
        self.guest_questions: dict[int, list[float]] = {}  # гость → когда спрашивал
        self.seen: list[int] = []  # последние update_id: не брать дважды
        self.kb_file = kb_file
        self.choice = read_choice(kb_file)  # чат → выбранная база дерева (/kb)

    def kb_of(self, chat_id: int) -> str:
        """С какой базой дерева работает этот чат: выбранная /kb, иначе — с какой
        запущен бот. Выбранной базы больше нет в дереве (её убрали) — тоже по умолчанию."""
        chosen = self.choice.get(chat_id)
        kb = self.chat_settings.kb if chosen is None else chosen
        # Базу, к которой у человека нет доступа (acl.py), — нельзя: тогда его ветки.
        return kb if acl.visible(kb) else ""

    def checks_conflicts(self, session: Session) -> bool:
        """Сверять ли статьи для этого чата: как выбрал человек, иначе — как в настройках."""
        if session.conflicts is None:
            return bool(self.chat_settings.check_conflicts)
        return session.conflicts

    def _guard(self, chat_id: int, what: str, fn: Any, *args: Any) -> None:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 — одно сообщение не должно ронять бота
            self.log(f"  ошибка на {what[:40]!r}: {type(exc).__name__}: {exc}")
            try:
                self.tg.send(chat_id, f"Ошибка: {esc(type(exc).__name__)}: {esc(str(exc)[:300])}")
            except Exception:  # noqa: BLE001
                pass

    # -- объявления: делают другие части ------------------------------------ #
    def show_fragment(
        self, chat_id: int, session: Session, src: dict[str, Any], with_article: bool = True
    ) -> None:
        raise NotImplementedError  # в части класса — см. bot.Bot

    def _formulas_in(self, chunk_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        raise NotImplementedError  # в части класса — см. bot.Bot

    def _figures_in(self, chunk_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        raise NotImplementedError  # в части класса — см. bot.Bot

    def answer_figures(self, chat_id: int, record: dict[str, Any]) -> None:
        raise NotImplementedError  # в части класса — см. bot.Bot
