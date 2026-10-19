"""Настройки чат-бота: модель и её сервер, сколько фрагментов, запросов и частей вопроса."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..kb import default_path
from ..llm import DEFAULT_MODEL, LLM, LLM_HOST, NUM_CTX
from ..machine import integer

MODEL = DEFAULT_MODEL

# Сколько фрагментов давать модели для ответа: восемь, из одной статьи до
# трёх. Вместе с соседними кусками (см. search.NEIGHBOR_CHARS) это около 7 тысяч
# токенов — окно модели 12 тысяч, с запасом на ответ и историю разговора. С окном
# больше (KBRAG_NUM_CTX) можно и фрагментов больше: KBRAG_TOP_K, KBRAG_PER_DOC в
# .local/settings.env (machine.py). Не влезут в окно — лишние отбросит fit_budget.
TOP_K = integer("KBRAG_TOP_K", 8)
PER_DOC = integer("KBRAG_PER_DOC", 3)
QUERIES = 3  # сколько поисковых запросов на часть вопроса
PARTS = 3  # на сколько частей самое большее делится вопрос


@dataclass
class Settings:
    model: str = MODEL
    host: str = LLM_HOST
    top_k: int = TOP_K
    per_doc: int = PER_DOC
    queries: int = QUERIES
    temperature: float = 0.1
    num_ctx: int = NUM_CTX
    timeout: int = 300
    parts: int = PARTS  # на сколько частей самое большее делить вопрос
    # Вопрос не по теме и в базе пусто: False — короткий отказ, True — ответ из общих знаний.
    general_off_topic: bool = False
    # Сверять, не расходятся ли статьи (find_conflicts): около 4 с на ответ.
    check_conflicts: bool = True
    # Узел дерева баз (kb.py): поиск и граф — по его поддереву, промпты — из его настроек.
    kb: str = field(default_factory=default_path)
    # Отбор кандидатов реранкером (index/rerank.py), модель — только «чего не хватает».
    # Выключить на машине — KBRAG_RERANKER=off в .local/settings.env.
    rerank: bool = True
    # Быстрый режим: без разбора вопроса моделью, без модели в отборе и без сверки —
    # поиск, реранкер и сразу ответ. Модель отвечает одним вызовом.
    fast: bool = False

    def client(self) -> LLM:
        """Клиент сервера модели с этими настройками."""
        return LLM(
            model=self.model,
            host=self.host,
            timeout=self.timeout,
            temperature=self.temperature,
        )
