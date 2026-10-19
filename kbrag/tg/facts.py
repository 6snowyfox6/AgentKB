"""Телеграм-бот — факты из графа: датасеты, связи под ответом, картинка связей, таблица CSV."""

from __future__ import annotations

from typing import Any

from ..kb import get as get_kb
from . import ui
from .api import TelegramError
from .base import MAX_TERRITORIES, BotBase, Session
from .ui import esc, split


class FactsMixin(BotBase):
    """Бот — факты из графа: датасеты, связи под ответом, картинка связей, таблица CSV."""

    def datasets(self, chat_id: int, session: Session, mid: int | None = None) -> None:
        from ..graph import api
        from ..graph.synonyms import load
        from ..index import db

        with db.connect(self.dsn) as conn:
            kb = self.kb_of(chat_id)
            syn = load(kb=get_kb(kb))
            data = api.datasets_response(conn, syn, limit=ui.DATASET_BUTTONS, kb=kb)
        rows = data["datasets"]
        session.territories = rows
        text, markup = ui.datasets_text(rows, data.get("total")), ui.datasets_keyboard(rows)
        if mid:
            try:
                self.tg.edit(chat_id, mid, text, markup)
                return
            except TelegramError:
                pass  # под картинкой текст не правится — новым
        self.tg.send(chat_id, text, markup)

    def _dataset(self, chat_id: int, name: str) -> dict[str, Any]:
        from ..graph import api
        from ..graph.synonyms import load
        from ..index import db

        with db.connect(self.dsn) as conn:
            kb = self.kb_of(chat_id)
            return api.dataset_response(conn, load(kb=get_kb(kb)), {"name": name}, kb)

    def dataset_card(
        self, chat_id: int, session: Session, idx: int, mid: int | None = None
    ) -> None:
        data = self._dataset(chat_id, session.territories[idx]["name"])
        pieces = split(ui.dataset_text(data))
        markup = ui.dataset_keyboard(idx)
        if mid and len(pieces) == 1:
            self.tg.edit(chat_id, mid, pieces[0], markup)
            return
        for i, piece in enumerate(pieces):
            self.tg.send(chat_id, piece, markup if i == len(pieces) - 1 else None)

    def dataset_by_name(self, chat_id: int, session: Session, name: str) -> None:
        from ..graph.api import RequestError

        try:
            data = self._dataset(chat_id, name)
        except RequestError as exc:
            near = ", ".join(exc.known[:8])
            self.tg.send(
                chat_id,
                f"«{esc(name)}»: {esc(exc.message)}."
                + (f" Больше всего фактов у: {esc(near)}" if near else ""),
            )
            return
        self.graph_card(chat_id, session, self._territory(session, data), data=data)

    def _links(self, chat_id: int, chunk_ids: list[int]) -> dict[str, list[dict[str, Any]]]:
        """Фрагмент → связи графа, выписанные из него (graph.api.facts_response)."""
        from ..graph import api
        from ..graph.synonyms import load
        from ..index import db

        with db.connect(self.dsn) as conn:
            chunks: dict[str, list[dict[str, Any]]] = api.facts_response(
                conn, load(kb=get_kb(self.kb_of(chat_id))), chunk_ids, self.kb_of(chat_id)
            )["chunks"]
            return chunks

    def answer_links(self, chat_id: int, session: Session, record: dict[str, Any]) -> None:
        """«Связи» под ответом: что граф выписал из процитированных фрагментов; кнопки
        с именами — картинка связей сущности, оттуда — дальше по соседям."""
        sources = [s for s in record.get("sources") or [] if s.get("chunk_id")]
        used = set(record.get("used") or [])
        pool = [s for s in sources if s["n"] in used] or sources
        by_chunk = self._links(chat_id, [s["chunk_id"] for s in pool]) if pool else {}
        rows = [(s["n"], f) for s in pool for f in by_chunk.get(str(s["chunk_id"]), [])]
        if not rows:
            self.tg.send(
                chat_id,
                "Граф не выписал связей из этих фрагментов (или ещё не построен: "
                "python kbrag.py graph).",
            )
            return
        names = list(dict.fromkeys(x for _, f in rows for x in (f["src"], f["dst"])))
        shown = names[: ui.NEIGHBOR_BUTTONS]
        # Список сущностей полон — заново до кнопок, а не посреди них: иначе первые
        # кнопки сообщения вели бы на чужие номера.
        if len(session.territories) + len(shown) > MAX_TERRITORIES:
            session.territories = []
        buttons = [(name, self._entity_index(session, name, reset=False)) for name in shown]
        self.tg.send(chat_id, ui.links_text(rows), ui.links_keyboard(buttons))

    def _entity_index(
        self,
        session: Session,
        name: str,
        documents: int = 0,
        facts: int = 0,
        reset: bool = True,
    ) -> int:
        """Номер сущности в списке разговора — для кнопок (в callback_data только номер).
        Нет в списке — дописывается; список полон (reset) — начинается заново, и старые
        кнопки ответят «от старого сообщения»."""
        idx = next((i for i, r in enumerate(session.territories) if r["name"] == name), None)
        if idx is None:
            if reset and len(session.territories) >= MAX_TERRITORIES:
                session.territories = []
            session.territories.append({"name": name, "documents": documents, "facts": facts})
            idx = len(session.territories) - 1
        return idx

    def _territory(self, session: Session, data: dict[str, Any]) -> int:
        """Номер сущности из датасета — со счётом статей и фактов."""
        return self._entity_index(session, data["name"], data["documents"], len(data["facts"]))

    def graph_card(
        self,
        chat_id: int,
        session: Session,
        idx: int,
        mid: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Картинка связей сущности, под ней — соседи кнопками, цитаты, таблица, вопрос."""
        from . import picture

        data = data or self._dataset(chat_id, session.territories[idx]["name"])
        try:
            png, neighbors = picture.render(data)
        except Exception as exc:  # noqa: BLE001 — не нарисовалось: факты списком, как раньше
            self.log(f"  картинка связей не нарисовалась: {type(exc).__name__}: {exc}")
            self.dataset_card(chat_id, session, idx)
            return
        idx = self._territory(session, data)  # сначала сама сущность: список мог обнулиться
        buttons = [
            (name, self._entity_index(session, name, reset=False))
            for name in neighbors[: ui.NEIGHBOR_BUTTONS]
        ]
        caption, markup = ui.graph_caption(data, len(neighbors)), ui.graph_keyboard(idx, buttons)
        if mid:
            try:
                self.tg.edit_photo(chat_id, mid, png, caption, markup)
                return
            except TelegramError:
                pass  # не картинка или устарело — новым сообщением
        self.tg.photo(chat_id, png, caption, markup)

    def dataset_csv(self, chat_id: int, name: str) -> None:
        from ..graph import api

        data = self._dataset(chat_id, name)
        self.tg.document(
            chat_id,
            f"факты-{data['name'][:60]}.csv",
            api.dataset_csv(data).encode("utf-8"),
            caption=f"Факты о «{data['name']}» с цитатами — открывается в Excel",
        )
