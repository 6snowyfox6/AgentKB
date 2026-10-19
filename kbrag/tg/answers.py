"""Телеграм-бот — чат-бот: вопрос, очередь к модели, ответ потоком, фрагменты ответа."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from typing import Any

from . import ui
from .api import TelegramError
from .base import EDIT_EVERY, HISTORY_TURNS, KEEP, BotBase, Session
from .ui import LIMIT, esc, split


class AnswersMixin(BotBase):
    """Бот — чат-бот: вопрос, очередь к модели, ответ потоком, фрагменты ответа."""

    def question(
        self, chat_id: int, session: Session, text: str, history: list[dict[str, Any]] | None = None
    ) -> None:
        if session.busy:
            self.tg.send(
                chat_id, "Ещё пишу прошлый ответ — дождитесь или нажмите «Остановить» " "под ним."
            )
            return
        session.busy = True
        session.stop.clear()
        args = (chat_id, session, text, history)
        if self.background:
            worker = threading.Thread(
                target=self._answer_safe, args=args, name=f"tg-answer-{chat_id}", daemon=True
            )
            self.workers[chat_id] = worker
            worker.start()
        else:
            self._answer_safe(*args)

    def _answer_safe(self, chat_id: Any, session: Any, text: Any, history: Any) -> None:
        try:
            self._wait_turn(chat_id)
            try:
                self._guard(chat_id, text, self.answer, chat_id, session, text, history)
            finally:
                self.slots.release()
        finally:
            session.busy = False

    def _wait_turn(self, chat_id: int) -> None:
        """Все места у модели заняты (MAX_PARALLEL) — вопрос ждёт, человеку видно место в очереди."""
        if self.slots.acquire(blocking=False):
            return
        with self.lock:
            self.waiting += 1
            place = self.waiting
        try:
            note = None
            try:
                note = self.tg.send(
                    chat_id,
                    f"<i>Модель сейчас отвечает другим — вы в очереди, "
                    f"перед вами {place}. Ответ начнётся сам.</i>",
                )
            except TelegramError:
                pass
            self.slots.acquire()
            if note:
                try:
                    self.tg.edit(
                        chat_id, note["message_id"], "<i>Очередь подошла — отвечаю ниже.</i>"
                    )
                except TelegramError:
                    pass
        finally:
            with self.lock:
                self.waiting -= 1

    def answer(
        self, chat_id: int, session: Session, text: str, history: list[dict[str, Any]] | None = None
    ) -> None:
        from ..chat import answer as chat
        from ..index import db

        history = list(session.history if history is None else history)
        aid = session.next_id()
        self.tg.typing(chat_id)
        msg = self.tg.send(chat_id, "<i>разбираю вопрос…</i>", ui.stop_keyboard(aid))
        mid = msg["message_id"]
        settings = replace(
            self.chat_settings,
            check_conflicts=self.checks_conflicts(session),
            kb=self.kb_of(chat_id),
            fast=session.fast,
        )
        sources, answer, general, done, error, info = [], "", None, {}, None, {}
        stopped = False
        last_edit = time.monotonic()
        with db.connect(self.dsn) as conn:
            db.autocommit(conn)  # ответ пишется минутами — без открытой транзакции
            events = chat.answer(conn, self.embedder, text, history, settings)
            try:
                for ev in events:
                    if session.stop.is_set():
                        stopped = True
                        break
                    kind = ev["type"]
                    if kind == "status" and not answer:
                        self.tg.edit(
                            chat_id, mid, f"<i>{esc(ev['text'])}</i>", ui.stop_keyboard(aid)
                        )
                    elif kind == "sources":
                        sources, info = ev["sources"], ev
                    elif kind == "general":
                        general, info = ev["text"], ev
                    elif kind == "token":
                        answer += ev["text"]
                        if time.monotonic() - last_edit >= EDIT_EVERY:
                            # Пока пишется — простым текстом: обрезанный кусок с тегами
                            # Телеграм не примет.
                            try:
                                self.tg.edit(
                                    chat_id, mid, esc(answer[-LIMIT:]) + " …", ui.stop_keyboard(aid)
                                )
                            except TelegramError:
                                pass
                            last_edit = time.monotonic()
                    elif kind == "revised":
                        answer = ev["text"]  # без фраз без ссылки на фрагмент
                    elif kind == "done":
                        done = ev
                    elif kind == "error":
                        error = ev["error"]
            finally:
                events.close()  # остановили — модель перестаёт писать
        if error:
            self.tg.edit(chat_id, mid, f"Ошибка: {esc(error)}")
            return
        if stopped:
            done = {**done, "stopped": True}
            if not answer.strip():
                self.tg.edit(chat_id, mid, "<i>Остановлено.</i>")
                return

        session.sources = sources
        session.history = (
            history
            + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": answer, "mode": done.get("mode")},
            ]
        )[-HISTORY_TURNS * 2 :]
        record = {
            "question": text,
            "sources": sources,
            "history": history,
            "info": info,
            "used": done.get("used"),
            "mode": done.get("mode"),
            "model": done.get("model"),
            "seconds": done.get("seconds"),
            "timings": done.get("timings"),
        }
        session.answers[aid] = record
        while len(session.answers) > KEEP:
            session.answers.pop(next(iter(session.answers)))

        markup = ui.answer_keyboard(aid, sources)
        pieces = split(ui.answer_text(answer, general, sources, done))
        try:
            self._deliver(chat_id, mid, pieces, markup)
        except TelegramError as exc:
            if "parse" not in str(exc).lower():
                raise
            # Разметка не разобралась — тот же ответ без неё, чтобы он всё равно дошёл.
            self._deliver(chat_id, mid, [ui.plain(p) for p in pieces], markup)
        self.answer_figures(chat_id, record)

    def _deliver(self, chat_id: int, mid: int, pieces: list[str], markup: Any) -> None:
        """Первый кусок — правкой сообщения «работаю», остальные — новыми; кнопки — у последнего."""
        last = len(pieces) - 1
        self.tg.edit(chat_id, mid, pieces[0], markup if last == 0 else None)
        for i, piece in enumerate(pieces[1:], start=1):
            self.tg.send(chat_id, piece, markup if i == last else None)

    def fragment(self, chat_id: int, session: Session, arg: str) -> None:
        if not session.sources:
            self.tg.send(
                chat_id, "Сначала задайте вопрос — фрагменты берутся из последнего ответа."
            )
            return
        number = int(arg) if arg.isdigit() else 0
        src = next((s for s in session.sources if s["n"] == number), None)
        if src is None:
            self.tg.send(
                chat_id,
                f"Фрагменты последнего ответа: 1–{len(session.sources)}. "
                "Или нажмите номер под ответом.",
            )
            return
        self.show_fragment(chat_id, session, src)

    def show_fragment(
        self, chat_id: int, session: Session, src: dict[str, Any], with_article: bool = True
    ) -> None:
        article = None
        doc_id = src.get("doc_id") or ""
        if with_article and doc_id and src.get("kind") != "датасет":
            ref = session.remember(session.lists, {"kind": "ref", "doc_id": doc_id, "items": []})
            article = f"art:{ref}"
        chunk_id = src.get("chunk_id")
        found = self._formulas_in([chunk_id]).get(chunk_id, []) if chunk_id else []
        pictures = self._figures_in([chunk_id]).get(chunk_id, []) if chunk_id else []
        markup = ui.fragment_keyboard(src, article, found, pictures)
        pieces = split(ui.fragment_text(src))
        for i, piece in enumerate(pieces):
            self.tg.send(chat_id, piece, markup if i == len(pieces) - 1 else None)
