"""Конвейер разбора: зонд → docling → проверка → PyMuPDF4LLM → оркестратор → фрагменты.

Зонд (profile) за доли секунды решает, что за документ: скан дальше не идёт. docling
считает в отдельном процессе (worker) с жёстким таймаутом. Если он упал, завис или не
прошёл проверку (validate), документ разбирает PyMuPDF4LLM; из попыток берётся лучшая.
Её доводит оркестратор (orchestrator): каждая найденная проблема PDF — отдельный случай
со своей проверкой и обработкой; что нашлось и что сделано — в журнал и в итог документа.

Журнал пишется построчно в JSONL: шаг, сколько занял, результат, ошибки — ровно то,
что потом понадобится, чтобы понять, где узкое место. Сбой одного документа очередь
не роняет: process_document не бросает исключений, а непредвиденная ошибка уходит в
журнал с причиной, местом и полным стеком (шаг «сбой»), документ — со статусом failed.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import median_high
from typing import Any

from ..chunks.models import Chunk
from . import orchestrator
from .config import Settings
from .diag import describe, full_traceback
from .models import (
    FAILED,
    MANUAL_REVIEW,
    OK,
    PARTIAL,
    PIPELINE,
    PROBE,
    SCAN,
    TIMEOUT,
    ParsedDoc,
    ParseInput,
    ParserSetupError,
    ValidationReport,
)
from .orchestrator import profile
from .parsers.pymupdf import parse_with_pymupdf
from .parsers.worker import DoclingWorker
from .validate import accuracy, load_golden, validate


def say(line: str) -> None:
    """Строка в консоль. Вывод в файл бывает не в UTF-8 (cp1251 при перенаправлении) —
    знак, которого в кодировке нет, заменяется, а не роняет разбор."""
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(encoding, errors="replace").decode(encoding), flush=True)


class StepLogger:
    def __init__(self, log_dir: Path, run_id: str | None = None, prefix: str = "parse"):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.path = log_dir / f"{prefix}-{self.run_id}.jsonl"

    def log(self, doc_id: str, step: str, status: str, duration: float, **extra: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "doc": doc_id,
            "step": step,
            "status": status,
            "duration_sec": round(duration, 2),
            **extra,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        say(
            f"  [{step}] {status} за {duration:.1f}с"
            + (f" — {extra.get('detail')}" if extra.get("detail") else "")
        )
        errors = extra.get("errors") or []
        if status in (FAILED, TIMEOUT) and errors:  # причина — сразу видна, не только в журнале
            say(f"    ↳ {str(errors[0]).splitlines()[0][:300]}")


@dataclass
class DocResult:
    doc_id: str
    source_path: str
    parser: str
    status: str
    accuracy: float = 0.0
    pages: int = 0
    sections: int = 0
    tables: int = 0
    chunks: int = 0
    duration_sec: float = 0.0
    failed_checks: list[str] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)
    variant: str = ""  # вариант документа по зонду (profile): скан, с переносами…
    cases: list[str] = field(default_factory=list)  # найденные оркестратором случаи

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "duration_sec": round(self.duration_sec, 2)}


def process_document(
    source: Path | str | ParseInput,
    settings: Settings,
    worker: DoclingWorker,
    logger: StepLogger,
    make_chunks: bool = True,
    write_outputs: bool = True,
) -> tuple[DocResult, ParsedDoc, list[Chunk]]:
    """Один документ от PDF до фрагментов. Любой сбой — статус документа и запись в
    журнале с причиной и стеком. Исключение одно — ParserSetupError: парсер настроен не
    так, как задано, и разбор останавливается целиком.

    Возвращает сводку, лучший разобранный документ и чанки — импорту нужны сами
    данные, а не только статистика.
    """
    inp = ParseInput.of(source)
    started = time.monotonic()
    try:
        return _process(inp, settings, worker, logger, make_chunks, write_outputs, started)
    except ParserSetupError:
        raise
    except Exception as exc:  # noqa: BLE001 — сбой в нашем же коде не должен ронять очередь
        return _crashed(inp, exc, logger, started)


def _process(
    inp: ParseInput,
    settings: Settings,
    worker: DoclingWorker,
    logger: StepLogger,
    make_chunks: bool,
    write_outputs: bool,
    started: float,
) -> tuple[DocResult, ParsedDoc, list[Chunk]]:
    kind = profile.probe(inp)
    logger.log(
        inp.doc_id,
        "профиль",
        kind.variant,
        kind.seconds,
        detail=f"{kind.variant}: {kind.reason}",
        profile=kind.to_dict(),
    )
    scanned = kind.variant == profile.SCAN
    if scanned and not settings.ocr:
        return _scan(inp, kind, logger, started)
    if scanned:
        # Скан с распознаванием (import --ocr): только docling с OCR — у запасного
        # PyMuPDF текста взять неоткуда.
        inp = replace(inp, ocr=True)
        logger.log(inp.doc_id, "OCR", kind.variant, 0.0, detail="скан — распознаю текст страниц")

    try:
        golden = load_golden(inp.doc_id, settings)
    except (ValueError, OSError) as exc:  # битый эталон — сверка пропускается, не документ
        golden = None
        logger.log(inp.doc_id, "эталон", FAILED, 0.0, detail=f"сверка пропущена: {describe(exc)}")

    # Попытки по очереди, пока одна не пройдёт проверку: docling, затем PyMuPDF4LLM —
    # быстрее и проще, но хуже держит сложную вёрстку и таблицы.
    tried: list[tuple[ParsedDoc, ValidationReport]] = []
    for parse in (worker.parse,) if scanned else (worker.parse, parse_with_pymupdf):
        parsed = parse(inp)
        report = validate(parsed, settings, golden)
        logger.log(
            inp.doc_id,
            parsed.parser,
            parsed.status,
            parsed.duration_sec,
            detail=f"страниц {parsed.page_count}, разделов {len(parsed.sections)}, "
            f"таблиц {parsed.table_count}, символов {len(parsed.text)}",
            validation=report.to_dict(),
            errors=parsed.errors,
        )
        tried.append((parsed, report))
        if _passed(parsed, report):
            break

    # Лучшая попытка — с наибольшей долей пройденных проверок (из равных — первая):
    # если ни одна не прошла, на ручную проверку уходит лучшее, а не пустой файл.
    parsed, report = max(tried, key=lambda pr: accuracy(pr[1]))
    findings = _orchestrate(inp, parsed, kind, tried, logger)
    passed = _passed(parsed, report)
    status = parsed.status if passed else MANUAL_REVIEW
    chunks, chunk_problem = _chunk(parsed, settings, logger) if passed and make_chunks else ([], "")
    findings.append(orchestrator.chunks(chunk_problem))

    if write_outputs:
        _write_outputs(parsed, report, chunks, settings, status)

    result = DocResult(
        doc_id=parsed.doc_id,
        source_path=parsed.source_path,
        parser=parsed.parser,
        status=status,
        accuracy=accuracy(report),
        pages=parsed.page_count,
        sections=len(parsed.sections),
        tables=parsed.table_count,
        chunks=len(chunks),
        duration_sec=time.monotonic() - started,
        failed_checks=[f"{c.name}: {c.detail}" for c in report.failed]
        + ([f"нарезка: {chunk_problem}"] if chunk_problem else []),
        attempts=[f"{p.parser}:{p.status}" for p, _ in tried],
        variant=kind.variant,
        cases=[f.short() for f in findings if f.found or f.error],
    )
    logger.log(
        inp.doc_id,
        "document",
        status,
        result.duration_sec,
        detail=f"парсер {parsed.parser}, точность {result.accuracy:.0%}, чанков {len(chunks)}",
    )
    return result, parsed, chunks


def _orchestrate(
    inp: ParseInput,
    parsed: ParsedDoc,
    kind: profile.Profile,
    tried: list[tuple[ParsedDoc, ValidationReport]],
    logger: StepLogger,
) -> list[orchestrator.Finding]:
    """Случаи документа: что сказал зонд, понадобился ли запасной парсер, доводка
    выбранной попытки. В журнал — все случаи (нашёлся или нет), в консоль — найденные."""
    t0 = time.monotonic()
    findings = orchestrator.before(kind) + [orchestrator.attempts(tried)]
    findings += orchestrator.after(parsed, inp, kind)
    broken = [f for f in findings if f.error]
    logger.log(
        inp.doc_id,
        "оркестратор",
        FAILED if broken else OK,
        time.monotonic() - t0,
        detail=orchestrator.summary(findings),
        cases=[f.to_dict() for f in findings],
        **(
            {
                "errors": [f"{f.case}: {f.error}" for f in broken],
                "traceback": [f.trace for f in broken],
            }
            if broken
            else {}
        ),
    )
    return findings


def _crashed(
    inp: ParseInput, exc: Exception, logger: StepLogger, started: float
) -> tuple[DocResult, ParsedDoc, list[Chunk]]:
    """Непредвиденный сбой: в журнал — причина, место и полный стек; документ — failed."""
    reason = describe(exc)
    duration = time.monotonic() - started
    logger.log(
        inp.doc_id,
        "сбой",
        FAILED,
        duration,
        detail=reason,
        errors=[reason],
        traceback=full_traceback(exc),
    )
    parsed = ParsedDoc.failed(inp, PIPELINE, reason, duration=round(duration, 2))
    result = DocResult(
        doc_id=inp.doc_id,
        source_path=inp.source,
        parser=PIPELINE,
        status=FAILED,
        duration_sec=duration,
        failed_checks=[f"сбой конвейера: {reason}"],
        attempts=[f"{PIPELINE}:{FAILED}"],
    )
    return result, parsed, []


def _scan(
    inp: ParseInput, kind: profile.Profile, logger: StepLogger, started: float
) -> tuple[DocResult, ParsedDoc, list[Chunk]]:
    """Скан не разбирается: текста нет, распознавание картинок не делаем."""
    parsed = ParsedDoc.failed(inp, PROBE, kind.reason, status=SCAN)
    parsed.page_count = kind.pages
    result = DocResult(
        doc_id=inp.doc_id,
        source_path=inp.source,
        parser=PROBE,
        status=SCAN,
        pages=kind.pages,
        duration_sec=time.monotonic() - started,
        failed_checks=[f"скан: {kind.reason}"],
        attempts=[f"{PROBE}:скан"],
        variant=kind.variant,
        cases=[f.short() for f in orchestrator.before(kind) if f.found],
    )
    logger.log(inp.doc_id, "document", SCAN, result.duration_sec, detail="скан — не разбирается")
    return result, parsed, []


def _passed(parsed: ParsedDoc, report: ValidationReport) -> bool:
    return report.ok and parsed.status in (OK, PARTIAL)


def _chunk(parsed: ParsedDoc, settings: Settings, logger: StepLogger) -> tuple[list[Chunk], str]:
    """Чанки разобранного документа и что с ними не так (пусто — всё в порядке).
    Не вышло или вышло ноль — документ не попадёт в поиск, причина — в журнал и отчёт."""
    t0 = time.monotonic()
    try:
        from ..chunks.text import chunk_parsed_doc

        chunks = chunk_parsed_doc(parsed, settings)
    except Exception as exc:  # noqa: BLE001
        problem = describe(exc)
        logger.log(
            parsed.doc_id,
            "chunking",
            FAILED,
            time.monotonic() - t0,
            detail=problem,
            errors=[problem],
            traceback=full_traceback(exc),
        )
        return [], problem
    if not chunks:
        problem = "ни одного фрагмента — документ не попадёт в поиск"
        logger.log(parsed.doc_id, "chunking", FAILED, time.monotonic() - t0, detail=problem)
        return [], problem
    tokens = median_high([c.n_tokens for c in chunks])
    logger.log(
        parsed.doc_id,
        "chunking",
        OK,
        time.monotonic() - t0,
        detail=f"{len(chunks)} чанков, медиана {tokens} токенов",
    )
    return chunks, ""


def _write_outputs(
    parsed: ParsedDoc,
    report: ValidationReport,
    chunks: list[Chunk],
    settings: Settings,
    status: str,
) -> None:
    out = settings.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = parsed.doc_id

    (out / f"{stem}.md").write_text(parsed.text, encoding="utf-8")

    payload = {**parsed.summary(), "final_status": status, "validation": report.to_dict()}
    (out / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # DoclingDocument нужен на следующих шагах (перечанкинг без повторного парсинга).
    if parsed.docling_doc:
        (out / f"{stem}.docling.json").write_text(
            json.dumps(parsed.docling_doc, ensure_ascii=False), encoding="utf-8"
        )

    if chunks:
        (out / f"{stem}.chunks.json").write_text(
            json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def process_all(paths: list[Path], settings: Settings, make_chunks: bool = True) -> list[DocResult]:
    logger = StepLogger(settings.log_dir)
    worker = DoclingWorker(settings)
    results: list[DocResult] = []

    try:
        for i, path in enumerate(paths, start=1):
            say(f"\n[{i}/{len(paths)}] {path.name}")
            doc_result, _parsed, _chunks = process_document(
                path, settings, worker, logger, make_chunks=make_chunks
            )
            results.append(doc_result)
    finally:
        worker.close()

    settings.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = settings.out_dir / f"report-{logger.run_id}.json"
    report_path.write_text(
        json.dumps([r.to_dict() for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    say(f"\nОтчёт: {report_path}\nЛог:    {logger.path}")
    return results
