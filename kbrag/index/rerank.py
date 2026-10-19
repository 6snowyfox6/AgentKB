"""Реранкер: насколько фрагмент отвечает на вопрос — отдельной моделью, а не языковой.

bge-reranker-v2-m3 (BAAI) читает пару «вопрос — фрагмент» целиком и даёт число от 0
до 1. Мультиязычный (та же основа XLM-R, что у векторов BGE-M3): русский вопрос
оценивает и английский фрагмент. В отличие от векторов, он видит вопрос и фрагмент
вместе, поэтому «похожие слова, но о другом» оценивает низко — то, ради чего кандидатов
раньше читала языковая модель, пачками и по нескольку секунд.

Чат-бот берёт его для первого круга отбора (chat/retrieval.py); модель — только чтобы
сказать, чего не хватает, если реранкер нашёл не всё. Модель грузится один раз на
процесс, при первом вопросе. Где считать — как у векторов (embed._auto_device): на
видеокарте, если на ней есть место рядом с vLLM, иначе на процессоре. Другое —
KBRAG_RERANK_DEVICE (cuda, cpu); другая модель — KBRAG_RERANKER; выключить —
KBRAG_RERANKER=off (тогда отбирает модель, как раньше).
"""

from __future__ import annotations

import threading
from typing import Any

from ..machine import value

MODEL = "BAAI/bge-reranker-v2-m3"
MAX_LENGTH = 512  # токенов на пару: вопрос и начало фрагмента (кандидат — до 1000 знаков)
BATCH = 16


class Reranker:
    def __init__(self, model_id: str = MODEL, device: str = "auto") -> None:
        from sentence_transformers import CrossEncoder

        from .embed import _auto_device

        self.device = _auto_device() if device == "auto" else device
        self.model_id = model_id
        kwargs: dict[str, Any] = {}
        if self.device.startswith("cuda"):
            import torch

            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        self.model = CrossEncoder(model_id, device=self.device, max_length=MAX_LENGTH, **kwargs)
        self.lock = threading.Lock()  # одна модель на все потоки сервера

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Оценка каждого текста для вопроса, 0–1 (сигмоида: у модели один выход)."""
        if not texts:
            return []
        with self.lock:
            scores = self.model.predict(
                [(query, t) for t in texts], batch_size=BATCH, show_progress_bar=False
            )
        return [float(s) for s in scores]


_SHARED: dict[str, Any] = {}
_LOCK = threading.Lock()


def shared() -> Reranker | None:
    """Реранкер процесса; выключен, нет sentence-transformers или модель не загрузилась —
    None (и попытка не повторяется: отбор идёт моделью, как раньше)."""
    model_id = value("KBRAG_RERANKER", MODEL)
    if model_id.lower() in ("off", "no", "0", "нет"):
        return None
    with _LOCK:
        if model_id not in _SHARED:
            try:
                _SHARED[model_id] = Reranker(model_id, value("KBRAG_RERANK_DEVICE", "auto"))
            except Exception as exc:  # noqa: BLE001 — без реранкера бот всё равно отвечает
                print(f"Реранкер не загрузился ({type(exc).__name__}: {exc}) — отбирает модель.")
                _SHARED[model_id] = None
        found: Reranker | None = _SHARED[model_id]
        return found
