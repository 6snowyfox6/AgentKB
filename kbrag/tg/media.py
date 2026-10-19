"""Телеграм-бот — формулы и рисунки: кнопки под фрагментом, картинки, рисунки под ответом."""

from __future__ import annotations

from typing import Any

from . import ui
from .base import STALE, BotBase


class MediaMixin(BotBase):
    """Бот — формулы и рисунки: кнопки под фрагментом, картинки, рисунки под ответом."""

    def _formulas_in(self, chunk_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        """Формулы, стоящие во фрагментах (index/formulas.py); база не ответила — нет."""
        from ..index import db, formulas

        try:
            with db.connect(self.dsn) as conn:
                return formulas.in_chunks(conn, chunk_ids)
        except Exception:  # noqa: BLE001 — без кнопок формул фрагмент всё равно покажем
            return {}

    def _figures_in(self, chunk_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
        """Рисунки фрагментов (index/figures.py); база не ответила — нет."""
        from ..index import db, figures

        try:
            with db.connect(self.dsn) as conn:
                return figures.in_chunks(conn, chunk_ids)
        except Exception:  # noqa: BLE001 — без рисунков ответ всё равно дойдёт
            return {}

    def send_figure(self, chat_id: int, figure_id: int, n: int | None = None) -> str | None:
        """Рисунок фото с подписью; файла нет — подпись текстом."""
        from ..index import db, figures
        from ..library import objects

        with db.connect(self.dsn) as conn:
            item = figures.by_id(conn, figure_id)
        if item is None:
            return STALE
        path = objects.figure_file(str(item["doc_id"]), int(item["n"]))
        caption = ui.figure_caption(item, n)
        if path is None:
            self.tg.send(chat_id, caption + "\n<i>Файла рисунка нет.</i>")
        else:
            self.tg.photo(chat_id, path.read_bytes(), caption)
        return None

    def answer_figures(self, chat_id: int, record: dict[str, Any]) -> None:
        """Рисунки фрагментов, на которые сослался ответ, — фото после ответа: сначала
        тот, что стоит во фрагменте (описание, подпись), потом те, на что фрагмент
        ссылается; не больше FIGURES_IN_ANSWER."""
        from ..index import db, figures

        try:
            with db.connect(self.dsn) as conn:
                chosen = figures.for_answer(
                    conn, record.get("sources") or [], record.get("used"), ui.FIGURES_IN_ANSWER
                )
        except Exception:  # noqa: BLE001 — без рисунков ответ всё равно дошёл
            return
        for item in chosen:
            try:
                self.send_figure(chat_id, int(item["id"]), item["source"])
            except Exception as exc:  # noqa: BLE001 — рисунок не дошёл: ответ уже есть
                self.log(f"рисунок {item['id']}: {type(exc).__name__}: {exc}")

    def send_formula(self, chat_id: int, formula_id: int) -> str | None:
        """Формула картинкой (вырезка со страницы PDF или нарисованный LaTeX) с подписью;
        картинки нет — LaTeX текстом."""
        from ..index import db, formulas
        from ..library import objects

        with db.connect(self.dsn) as conn:
            item = formulas.by_id(conn, formula_id)
        if item is None:
            return STALE
        png = objects.formula_png(item)
        caption = ui.formula_caption(item)
        if png is None:
            self.tg.send(chat_id, caption)
        else:
            self.tg.photo(chat_id, png, caption)
        return None
