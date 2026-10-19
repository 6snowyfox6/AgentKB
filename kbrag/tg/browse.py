"""Телеграм-бот — поиск и статьи: выдачи с листанием, карточка статьи, её фрагменты."""

from __future__ import annotations

from typing import Any

from ..kb import get as get_kb
from . import ui
from .base import SEARCH_HITS, BotBase, Session
from .ui import esc


class BrowseMixin(BotBase):
    """Бот — поиск и статьи: выдачи с листанием, карточка статьи, её фрагменты."""

    def _list(self, chat_id: int, session: Session, lst: dict[str, Any], extra: Any = None) -> None:
        """Список с листанием. extra(номер списка) → кнопки под номерами."""
        lst["page"] = 0
        pid = session.remember(session.lists, lst)
        if extra:
            lst["extra"] = extra(pid)
        self.tg.send(chat_id, ui.list_text(lst, 0), ui.list_keyboard(pid, lst, 0, lst.get("extra")))

    def search(self, chat_id: int, session: Session, query: str) -> None:
        from ..chat.search import source_dict
        from ..index import db
        from ..index.search import hybrid_search

        query = query.strip()
        if not query:
            self.tg.send(chat_id, "Что искать? Например: /search плотность линеаментов")
            return
        from ..graph.synonyms import cached, synonym_variants

        self.tg.typing(chat_id)
        kb = self.kb_of(chat_id)
        also = synonym_variants(query, cached(kb=get_kb(kb)))  # «Донбасс» — и «Donetsk basin»
        with db.connect(self.dsn) as conn:
            hits = hybrid_search(
                conn,
                self.embedder,
                query,
                limit=SEARCH_HITS,
                max_per_doc=2,
                min_similarity=0.45,
                alternatives=also,
                kb=kb,
            )
        if not hits:
            self.tg.send(
                chat_id,
                f"По запросу «{esc(query)}» близкого в базе не нашлось.\n"
                "Попробуйте другими словами или по-английски.",
            )
            return
        items = [source_dict(i, h) for i, h in enumerate(hits, start=1)]
        self._list(
            chat_id,
            session,
            {
                "kind": "search",
                "items": items,
                "query": query,
                "title": f"<b>Поиск:</b> {esc(query)} — найдено {len(items)}",
            },
            extra=lambda pid: [ui.button("Спросить бота об этом", f"askq:{pid}")],
        )

    def articles(self, chat_id: int, session: Session) -> None:
        from ..index import db

        with db.connect(self.dsn) as conn:
            docs = db.documents(conn, limit=1000, kb=self.kb_of(chat_id))
        if not docs:
            self.tg.send(chat_id, "В базе пока нет статей.")
            return
        self._list(
            chat_id,
            session,
            {
                "kind": "articles",
                "items": docs,
                "title": f"<b>Статей в базе: {len(docs)}</b> " "(новые сверху)",
            },
        )

    def open_item(
        self, chat_id: int, session: Session, pid: int, lst: dict[str, Any], idx: int
    ) -> None:
        if not 0 <= idx < len(lst["items"]):
            return
        item = lst["items"][idx]
        if lst["kind"] == "search":
            self.show_fragment(chat_id, session, item)
        elif lst["kind"] == "articles":
            self.article_card(chat_id, session, item["doc_id"], item)
        else:
            src = {**item, "title": lst.get("title_plain") or "", "url": lst.get("url") or ""}
            self.show_fragment(chat_id, session, src, with_article=False)

    def article_card(
        self, chat_id: int, session: Session, doc_id: str, doc: dict[str, Any] | None = None
    ) -> None:
        from ..index import db

        if doc is None:
            with db.connect(self.dsn) as conn:
                doc = next(
                    (d for d in db.documents(conn, limit=100000) if d["doc_id"] == doc_id), None
                )
            if doc is None:
                self.tg.send(chat_id, "Этой статьи в базе уже нет.")
                return
        ref = session.remember(
            session.lists,
            {
                "kind": "ref",
                "doc_id": doc_id,
                "items": [],
                "title": doc.get("title") or doc_id,
                "url": doc.get("url") or "",
            },
        )
        row = [ui.button("Фрагменты статьи", f"chunks:{ref}")]
        if ui.is_link(doc.get("url")):
            row.insert(0, ui.url_button("У издателя", doc["url"]))
        self.tg.send(chat_id, ui.article_text(doc), ui.inline([row]))

    def article_chunks(
        self, chat_id: int, session: Session, doc_id: str, title: str, url: str = ""
    ) -> None:
        from ..index import db

        with db.connect(self.dsn) as conn:
            chunks = db.document_chunks(conn, doc_id)
        if not chunks:
            self.tg.send(chat_id, "У статьи нет фрагментов в базе.")
            return
        items = [{**c, "doc_id": doc_id} for c in chunks]
        self._list(
            chat_id,
            session,
            {
                "kind": "chunks",
                "items": items,
                "doc_id": doc_id,
                "title": f"<b>{esc(title[:120])}</b> — фрагментов {len(items)}",
                "title_plain": title,
                "url": url,
            },
        )
