"""Docling в отдельном процессе с жёстким таймаутом на документ.

Это не паранойя: у docling есть известные случаи зависания в нативном парсере, где ни
document_timeout, ни Ctrl+C не помогают — процесс просто перестаёт отвечать. Для
импорта папки это означало бы остановку всей очереди, поэтому родитель ждёт результат
и убивает зависший процесс. Модели живут в дочернем процессе и грузятся один раз.

Ни один сбой не проходит молча — у каждого в ошибке документа есть причина и место:

- упал в нативном коде (память, сегфолт) — код выхода словами и стек, который процесс
  успел записать сам (faulthandler) в свой файл в папке журналов;
- завис — стек «где висел»: процесс пишет его незадолго до таймаута;
- исключение в разборе — тип, сообщение, место и полный стек;
- не запустился (docling не импортируется, нет защиты __main__ у скрипта) — отдельно
  от «упал на документе»: родитель ждёт от процесса сигнала «готов»;
- результат не передаётся между процессами — он сериализуется заранее, в самом
  процессе: иначе ошибка теряется в фоновом потоке очереди, а родитель ждёт весь таймаут.
"""

from __future__ import annotations

import faulthandler
import multiprocessing as mp
import os
import pickle
import queue
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import Settings
from ..diag import describe, exit_reason, full_traceback, tail
from ..models import DOCLING, FAILED, TIMEOUT, ParsedDoc, ParseInput, ParserSetupError

START_TIMEOUT_SEC = 300  # импорт docling на холодном диске — десятки секунд
HANG_DUMP_SHARE = 0.9  # стек зависания — на 90 % таймаута, до того как процесс убьют
STACK_LINES = 40  # стек пишется по всем потокам — место сбоя бывает не в самом конце

# ("ready", "") | ("setup", ошибка) | ("fatal", ошибка настройки) | ("result", байты ParsedDoc)
Message = tuple[str, Any]


def _no_crash_dialog() -> None:
    """Windows: упавший процесс иначе ждёт закрытия окна «Программа не работает» —
    ночной прогон висел бы весь таймаут и писал «завис» вместо «упал». Без окна процесс
    сразу завершается с кодом ошибки, а стек успевает записать faulthandler."""
    if sys.platform == "win32":
        import ctypes

        sem_failcriticalerrors, sem_nogpfaulterrorbox = 0x0001, 0x0002
        ctypes.windll.kernel32.SetErrorMode(sem_failcriticalerrors | sem_nogpfaulterrorbox)


def _worker_loop(
    tasks: mp.Queue[ParseInput | None],
    results: mp.Queue[Message],
    settings: Settings,
    crash_log: str,
    make_parser: Callable[[Settings], Any] | None,
) -> None:  # pragma: no cover — живёт в дочернем процессе
    """Разбирает документы, пока вместо документа не придёт None."""
    _no_crash_dialog()
    crash = open(crash_log, "a", encoding="utf-8")  # noqa: SIM115 — открыт до конца процесса
    faulthandler.enable(file=crash, all_threads=True)
    try:
        if make_parser is None:
            from .docling import DoclingParser

            make_parser = DoclingParser
        parser = make_parser(settings)
    except ParserSetupError as exc:  # разбирать с такой настройкой нельзя — даже запасным
        results.put(("fatal", str(exc)))
        return
    except Exception as exc:  # noqa: BLE001 — docling может быть не установлен
        results.put(("setup", f"{describe(exc)}\n{full_traceback(exc)}"))
        return
    results.put(("ready", ""))

    while (inp := tasks.get()) is not None:
        faulthandler.dump_traceback_later(settings.doc_timeout_sec * HANG_DUMP_SHARE, file=crash)
        try:
            parsed = parser.parse(inp)
        except Exception as exc:  # noqa: BLE001
            parsed = ParsedDoc.failed(inp, DOCLING, describe(exc), full_traceback(exc))
        finally:
            faulthandler.cancel_dump_traceback_later()
        try:
            payload = pickle.dumps(parsed)
        except Exception as exc:  # noqa: BLE001
            payload = pickle.dumps(
                ParsedDoc.failed(
                    inp,
                    DOCLING,
                    f"результат не передаётся из процесса: {describe(exc)}",
                    duration=parsed.duration_sec,
                )
            )
        results.put(("result", payload))


class DoclingWorker:
    """Один дочерний процесс + жёсткий таймаут на документ.

    make_parser — чем разбирать в дочернем процессе (по умолчанию DoclingParser); должен
    передаваться между процессами, то есть быть классом или функцией модуля.
    """

    def __init__(self, settings: Settings, make_parser: Callable[[Settings], Any] | None = None):
        self.settings = settings
        self.make_parser = make_parser
        self._ctx = mp.get_context("spawn")
        self._proc: mp.process.BaseProcess | None = None
        self.crash_log: Path | None = None
        self._starts = 0

    def parse(self, source: Path | str | ParseInput) -> ParsedDoc:
        inp = ParseInput.of(source)
        started = time.monotonic()

        def failed(*errors: str, status: str = FAILED) -> ParsedDoc:
            elapsed = round(time.monotonic() - started, 2)
            return ParsedDoc.failed(inp, DOCLING, *errors, status=status, duration=elapsed)

        if self._proc is None or not self._proc.is_alive():
            problem = self._start()
            if problem:
                return failed(problem)
        offset = self._crash_size()
        self.tasks.put(inp)
        sent = time.monotonic()  # таймаут — на разбор, запуск процесса в него не входит

        timeout = self.settings.doc_timeout_sec
        while True:
            try:
                kind, payload = self.results.get(timeout=2)
            except queue.Empty:
                if self._proc is None or not self._proc.is_alive():
                    return failed(self._died("процесс docling упал на этом документе", offset))
                if time.monotonic() - sent > timeout:
                    where = tail(self.crash_log, offset, lines=STACK_LINES)
                    self._stop()
                    return failed(
                        f"docling не ответил за {timeout} с — процесс убит"
                        + (f"\nгде висел ({self.crash_log}):\n{where}" if where else ""),
                        status=TIMEOUT,
                    )
                continue
            if kind != "result":
                self._stop()
                return failed(f"процесс docling прислал «{kind}» вместо результата — перезапущен")
            try:
                parsed: ParsedDoc = pickle.loads(payload)
            except Exception as exc:  # noqa: BLE001
                return failed(f"ответ процесса docling не читается: {describe(exc)}")
            if parsed.doc_id != inp.doc_id:
                self._stop()
                return failed(f"процесс docling ответил о другом документе ({parsed.doc_id})")
            return parsed

    def close(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            try:
                self.tasks.put(None)
                self._proc.join(timeout=10)
            except Exception:  # noqa: BLE001, S110 — всё равно убьём ниже
                pass
        self._stop()
        self._drop_empty_log()

    # --- внутреннее ---------------------------------------------------------
    def _start(self) -> str:
        """Поднять процесс и дождаться «готов». Пусто — готов, иначе — причина."""
        self._stop()
        self._drop_empty_log()
        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        self._starts += 1
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.crash_log = self.settings.log_dir / f"docling-{stamp}-{os.getpid()}-{self._starts}.log"
        self.tasks: mp.Queue[ParseInput | None] = self._ctx.Queue()
        self.results: mp.Queue[Message] = self._ctx.Queue()
        try:
            self._proc = self._ctx.Process(
                target=_worker_loop,
                args=(
                    self.tasks,
                    self.results,
                    self.settings,
                    str(self.crash_log),
                    self.make_parser,
                ),
                daemon=True,
            )
            self._proc.start()
        except Exception as exc:  # noqa: BLE001 — например, make_parser не передаётся в процесс
            self._proc = None
            return f"процесс docling не запустился: {describe(exc)}"

        deadline = time.monotonic() + START_TIMEOUT_SEC
        while True:
            try:
                kind, payload = self.results.get(timeout=1)
            except queue.Empty:
                if not self._proc.is_alive():
                    return self._died("процесс docling не запустился")
                if time.monotonic() > deadline:
                    self._stop()
                    return f"процесс docling не запустился за {START_TIMEOUT_SEC} с"
                continue
            if kind == "ready":
                return ""
            self._stop()
            if kind == "fatal":
                raise ParserSetupError(payload)
            return f"docling не загрузился: {payload}"

    def _died(self, what: str, offset: int = 0) -> str:
        """Процесс умер: код выхода словами и стек, который он успел записать."""
        code = self._proc.exitcode if self._proc is not None else None
        self._stop()
        stack = tail(self.crash_log, offset, lines=STACK_LINES)
        return f"{what}: {exit_reason(code)}" + (
            f"\nстек процесса ({self.crash_log}):\n{stack}" if stack else ""
        )

    def _stop(self) -> None:
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.kill()
            self._proc.join(timeout=10)
        self._proc = None

    def _crash_size(self) -> int:
        try:
            return self.crash_log.stat().st_size if self.crash_log else 0
        except OSError:
            return 0

    def _drop_empty_log(self) -> None:
        """Файл стеков без единой записи не нужен; с записями — остаётся для разбора."""
        if self.crash_log is not None and self._crash_size() == 0:
            try:
                self.crash_log.unlink(missing_ok=True)
            except OSError:
                pass
