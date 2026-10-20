"""Очередь ответов чат-бота — одна на процесс: страница, Телеграм-бот (web --tg), MCP.

Ответ — это несколько запросов к модели подряд и вперемешку (разбор вопроса, отбор
пачками, сверка, сам ответ) и соединение с базой на всё время, пока он пишется. Сверх
MAX_PARALLEL ответов сразу вопрос ждёт здесь: человек видит своё место в очереди, а
соединение с базой он берёт, только когда очередь подошла, — ждущие не выбирают пул.

Раньше очередь была только у Телеграм-бота, а страница пускала к модели всех сразу:
два человека с двух устройств — и модели на 16 ГБ не хватало памяти под задания,
запросы вытесняли друг друга и падали по таймауту у обоих. Сколько заданий модель
вмещает по токенам, следит ещё и llm.py (Budget) — для любых вызовов, не только ответов.

    with turns.turn(on_wait=lambda place: ...):
        ... ответ ...
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

from .machine import integer

# Ответов сразу. На 16 ГБ под задания модели — около 13 тыс. токенов: это 2 обычных
# задания чат-бота (5–7 тыс.). На машине мощнее — KBRAG_PARALLEL в .local/settings.env.
MAX_PARALLEL = integer("KBRAG_PARALLEL", 2)
POLL = 2.0  # с: как часто ждущему напоминать его место (и проверять, не ушёл ли он)


class Turns:
    def __init__(self, size: int = MAX_PARALLEL) -> None:
        self.size = max(1, size)
        self._slots = threading.BoundedSemaphore(self.size)
        self._lock = threading.Lock()
        self.waiting = 0  # сколько ответов ждут очереди

    @contextmanager
    def turn(
        self, on_wait: Callable[[int], None] | None = None, poll: float | None = None
    ) -> Iterator[None]:
        """Место у модели на время блока. Мест нет — ждать; on_wait(место) зовётся сразу и
        потом раз в poll секунд (по умолчанию POLL): страница так узнаёт, что человек ушёл
        (запись падает) — исключение из on_wait снимает его с очереди."""
        poll = POLL if poll is None else poll
        if not self._slots.acquire(blocking=False):
            with self._lock:
                self.waiting += 1
                place = self.waiting
            try:
                if on_wait:
                    on_wait(place)
                while not self._slots.acquire(timeout=poll):
                    if on_wait:
                        on_wait(place)
            finally:
                with self._lock:
                    self.waiting -= 1
        try:
            yield
        finally:
            self._slots.release()


SHARED = Turns()


def turn(
    on_wait: Callable[[int], None] | None = None, poll: float | None = None
) -> AbstractContextManager[None]:
    """Место в общей очереди процесса (SHARED)."""
    return SHARED.turn(on_wait, poll)
