"""Диагностика сбоев разбора: что упало, где и почему — в журнал, а не в никуда.

Сбой без диагностики хуже самого сбоя: «exitcode=3221225477» или «ValueError: …» без
места не говорит, что чинить. Здесь — описание исключения с местом в коде, расшифровка
кода выхода упавшего процесса и стек, который упавший или зависший процесс docling
успевает записать сам (faulthandler, см. worker.py).
"""

from __future__ import annotations

import signal
import sys
import traceback
from pathlib import Path

# Коды завершения Windows (NTSTATUS), с которыми падает нативный код: docling-parse,
# torch, onnxruntime. Python-исключение так процесс не завершает — это всегда сбой ниже.
_WINDOWS_CODES = {
    0xC0000005: "нарушение доступа к памяти — сбой в нативном коде",
    0xC00000FD: "переполнение стека",
    0xC0000409: "аварийное завершение нативного кода (fail-fast, переполнение буфера)",
    0xC0000374: "повреждение памяти (кучи)",
    0xC0000017: "не хватило памяти",
    0xC000013A: "прерван (Ctrl+C)",
}


def describe(exc: BaseException, frames: int = 3) -> str:
    """«ValueError: текст [файл.py:12 функция ← …]» — тип, сообщение и где упало."""
    head = f"{type(exc).__name__}: {exc}".rstrip(": ")
    stack = traceback.extract_tb(exc.__traceback__)[-frames:]
    where = " ← ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in reversed(stack))
    return f"{head} [{where}]" if where else head


def full_traceback(exc: BaseException) -> str:
    """Полный стек — в журнал рядом с коротким описанием."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()


def exit_reason(code: int | None) -> str:
    """Код выхода процесса — словами."""
    if code is None:
        return "код выхода неизвестен"
    if code < 0:  # POSIX: убит сигналом
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"сигнал {-code}"
        hint = " — скорее всего, системе не хватило памяти" if name == "SIGKILL" else ""
        return f"убит сигналом {name}{hint}"
    if sys.platform == "win32" and (code & 0xFFFFFFFF) in _WINDOWS_CODES:
        return f"{_WINDOWS_CODES[code & 0xFFFFFFFF]} (код 0x{code & 0xFFFFFFFF:08X})"
    if code == 1:
        return "ошибка Python в процессе (код 1; текст — в его выводе в консоли)"
    return f"код выхода {code}"


def tail(path: Path | None, offset: int = 0, lines: int = 20) -> str:
    """Конец файла начиная с offset — например, стек, записанный упавшим процессом."""
    if path is None or not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(offset)
            text = fh.read()
    except OSError:
        return ""
    return "\n".join(text.strip().splitlines()[-lines:])
