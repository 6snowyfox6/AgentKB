"""Эмбеддинги BGE-M3 — sentence-transformers прямо в процессе приложения.

Почему именно эта модель: мультиязычная (русский и английский в одном пространстве —
запрос на русском находит английскую статью), 1024 измерения, 8192 токена контекста,
и её же токенайзер уже используется при чанкинге. Токенайзер и модель обязаны
совпадать, иначе счёт токенов при нарезке врёт.

На видеокарте — в половинной точности (fp16): тысяча фрагментов за 5 с, вопрос
пользователя — 25 мс. Поиск и загрузка не зависят от того, запущен ли сервер
языковой модели. На процессоре (--device cpu) вопрос — 60 мс, а загрузка статей
в 80 раз медленнее (2,4 фрагмента в секунду): годится, если видеокарту целиком
занимает языковая модель.

Векторы нормируются, поэтому косинус и скалярное произведение совпадают, и в
базе работает оператор <=>.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

VECTOR_DIM = 1024


# Сколько свободной видеопамяти нужно, чтобы считать векторы на видеокарте. Сама
# модель с контекстом CUDA занимает до 2,6 ГБ, а vLLM в работе прирастает ещё на
# полгигабайта. Рядом с vLLM на 16 ГБ свободно около 3 ГБ — этого мало: память
# кончалась, Windows гоняла её через общую, и модель писала 20–35 токенов в секунду
# вместо 76 (замер 12.10.2026). Поэтому при работающей модели векторы — на процессоре.
GPU_FREE_NEEDED = 4e9


def _gpu_free() -> float | None:
    """Свободная видеопамять по nvidia-smi — с учётом всех процессов. torch под
    Windows видит только своё: память, занятую vLLM внутри Docker (WSL2), он
    считает свободной, и модель векторов, влезая туда, гоняла бы память через
    общую — в разы медленнее и для неё, и для vLLM."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        used, total = (float(x) for x in out.splitlines()[0].split(","))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    return (total - used) * 1024 * 1024


def _auto_device() -> str:
    """Видеокарта, если на ней есть место, иначе процессор. Языковая модель (vLLM)
    занимает видеокарту почти целиком; без места векторы считаются на процессоре:
    вопрос — 60 мс вместо 25, загрузка статей — 2,4 фрагмента в секунду вместо 190."""
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    free = _gpu_free()
    if free is None:
        free = float(torch.cuda.mem_get_info()[0])
    return "cuda" if free >= GPU_FREE_NEEDED else "cpu"


def offline_first[T](load: Callable[[], T]) -> T:
    """Загрузить модель из кэша Hugging Face, не спрашивая сеть; нет в кэше — скачать.

    local_files_only помогает не везде: sentence-transformers всё равно спрашивает Hub
    про отдельные файлы модели. При медленной связи с ним каждый такой запрос — до
    минуты (20.10.2026: вопрос в чат-боте искался 245 с, из них 242 — рукопожатия с
    huggingface.co). Поэтому на время загрузки Hub выключается целиком."""
    from huggingface_hub import constants

    before = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = True
    try:
        return load()
    except (OSError, ValueError):  # ещё не скачана
        pass
    finally:
        constants.HF_HUB_OFFLINE = before
    return load()


class Embedder:
    """BGE-M3 через sentence-transformers, на видеокарте или процессоре."""

    name = "local"

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 1024,
        half: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.half = half
        self._model = None
        # Модель одна на веб-сервер и Телеграм-бота, а запросы идут из разных
        # потоков. Токенайзер HuggingFace при одновременных вызовах падает с
        # «Already borrowed» — поэтому вызовы по очереди.
        self._lock = threading.Lock()

    # Модель весит около 2 ГБ и грузится секунды: берём её только когда реально нужна.
    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                self._model = self._build()
        return self._model

    def _build(self) -> Any:
        from sentence_transformers import SentenceTransformer

        device = _auto_device() if self.device == "auto" else self.device
        model = offline_first(lambda: SentenceTransformer(self.model_id, device=device))
        model.max_seq_length = self.max_length
        # fp16 на видеокарте — втрое быстрее и вдвое меньше памяти. Векторы те же:
        # косинус с fp32 не ниже 0,9997, первые 10 в выдаче совпадают. Поэтому
        # переиндексировать базу, посчитанную в fp32, не нужно. На процессоре
        # половинная точность медленнее — там остаётся fp32.
        if self.half and model.device.type == "cuda":
            model.half()
        return model

    @property
    def dim(self) -> int:
        return VECTOR_DIM

    def encode(self, texts: list[str], progress: bool = False) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        with self._lock:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,  # косинус = скалярное произведение
                show_progress_bar=progress,
                convert_to_numpy=True,
            )
        return [v.tolist() for v in vectors]

    def encode_one(self, text: str) -> list[float]:
        # У BGE-M3 нет отдельной инструкции для запроса: запрос и документ
        # кодируются одинаково. Добавлять префиксы от других моделей — вредно.
        return self.encode([text])[0]


def build_embedder(
    model_id: str | None = None,
    device: str = "auto",
    batch_size: int = 8,
) -> Embedder:
    """Считалка векторов: BGE-M3 на видеокарте (или на процессоре: device="cpu")."""
    return Embedder(model_id or "BAAI/bge-m3", device=device, batch_size=batch_size)
