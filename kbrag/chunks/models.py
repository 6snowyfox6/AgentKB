"""Фрагмент базы: то, что уходит в модель эмбеддинга и что цитируется."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    """Чанк, готовый к эмбеддингу и к цитированию.

    text       — оригинал, его показываем пользователю и цитируем;
    embed_text — то, что уходит в модель эмбеддинга (с заголовками разделов сверху).
    """

    doc_id: str
    index: int
    text: str
    embed_text: str
    headings: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    n_tokens: int = 0
    has_table: bool = False
    source_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
