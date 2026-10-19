"""Телеграм-бот — база дерева (/kb): какую выбрал чат, меню баз, выбор."""

from __future__ import annotations

from .. import acl
from ..kb import KBError
from ..kb import get as get_kb
from . import ui
from .access import save_choice
from .api import TelegramError
from .base import BotBase, Session
from .ui import esc


class KbsMixin(BotBase):
    """Бот — база дерева (/kb): какую выбрал чат, меню баз, выбор."""

    def choose_kb(self, chat_id: int, path: str | None) -> None:
        """Запомнить базу чата (None — как у бота) — и в файле: переживёт перезапуск."""
        if path is None:
            self.choice.pop(chat_id, None)
        else:
            self.choice[chat_id] = path
        save_choice(self.kb_file, self.choice)

    def kb_menu(self, chat_id: int, session: Session, mid: int | None = None) -> None:
        """Дерево баз кнопками; текущая отмечена."""
        from ..index import db
        from ..kbtree import nodes

        try:
            with db.connect(self.dsn) as conn:
                tree = nodes(conn)
        except Exception:  # noqa: BLE001 — база лежит: дерево по папкам настроек
            tree = nodes()
        session.kbs = ["", *(n["path"] for n in tree["nodes"])]
        text, markup = ui.kb_menu(tree, self.kb_of(chat_id), get_kb(self.kb_of(chat_id)).title)
        if mid:
            try:
                self.tg.edit(chat_id, mid, text, markup)
                return
            except TelegramError:
                pass
        self.tg.send(chat_id, text, markup)

    def set_kb(self, chat_id: int, session: Session, path: str, mid: int | None = None) -> str:
        """Выбрать базу; разговор начинается заново — прошлые ответы были по другой."""
        from ..kb import check_path

        try:
            path = check_path(path)
        except KBError as exc:
            self.tg.send(chat_id, esc(str(exc)))
            return "Нет такой базы"
        if not acl.visible(path):
            self.tg.send(chat_id, f"Нет доступа к базе «{esc(path)}».")
            return "Нет доступа"
        self.choose_kb(chat_id, path)
        session.history.clear()
        session.sources = []
        title = get_kb(path).title if path else "все базы"
        text = f"База: <b>{esc(title)}</b>" + (f" ({esc(path)})" if path else " — всё дерево")
        text += ". Поиск, статьи, факты и ответы — теперь по ней; разговор начат заново."
        if mid:
            try:
                self.tg.edit(chat_id, mid, text)
                return title
            except TelegramError:
                pass
        self.tg.send(chat_id, text, ui.menu_keyboard())
        return title
