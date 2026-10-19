"""Импорт своих PDF и переразбор уже загруженных.

import — PDF с диска (файл или папка, с подпапками) → разбор → папка статей:

* повтор отсекается по отпечатку файла (sha256): тот же файл второй раз не
  разбирается. Если такой файл уже был в базе без PDF (статья прежней добычи), к
  записи дописывается путь — и её можно переразобрать (reparse);
* метаданные — из YAML и BibTeX рядом и из самого PDF (meta.py), недостающее можно
  спросить в терминале (--ask);
* путь к PDF остаётся в записи (pdf_path): сам файл не копируется;
* рисунки сохраняются файлами (save_images), а с --figures их ещё и описывает
  модель со зрением — описания становятся фрагментами.

reparse — то же для статей, у которых есть PDF: разобрать заново (поменялись
настройки разбора, вышла новая версия docling). Загрузка в базу после него
переписывает фрагменты на месте (index/db.py, replace_chunks): у фрагмента с тем же
текстом остаются и номер, и факты графа — модель проходит только изменившиеся.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..chunks.figures import (
    FIGURE_HOST,
    FIGURE_MODEL,
    describe_saved,
    description_chunks,
    save_images,
)
from ..kb import KB
from ..parse.config import Settings
from ..parse.models import OK, PARTIAL, SCAN, ParseInput
from ..parse.orchestrator.profile import SCAN as SCAN_VARIANT
from ..parse.parsers.worker import DoclingWorker
from ..parse.pipeline import StepLogger, process_document
from . import meta, store

# Что получилось со статьёй — в отчёт импорта.
NEW = "разобрана"
SAME = "уже есть"  # тот же файл уже разобран
ATTACHED = "PDF привязан"  # статья была без PDF (прежняя добыча) — теперь он есть
SCANNED = "скан"  # текстового слоя нет — не разбирается
FAILED = "не разобрана"  # ни одна попытка не прошла проверку — смотреть человеку
NO_PDF = "нет PDF"  # reparse: файла нет или он другой
READ_ERROR = "не читается"


@dataclass
class Item:
    """Одна статья в отчёте импорта или переразбора."""

    file: str
    status: str
    doc_id: str = ""
    title: str = ""
    chunks: int = 0
    figures: int = 0
    note: str = ""
    origin: list[str] = field(default_factory=list)  # откуда взяты метаданные

    @property
    def indexed(self) -> bool:
        """Есть ли что загружать в базу (и класть в ветку)."""
        return self.status in (NEW, SAME, ATTACHED) and bool(self.doc_id)


def find_pdfs(paths: list[Path]) -> list[Path]:
    """PDF из файлов и папок (с подпапками), без повторов, по порядку."""
    out: list[Path] = []
    for path in paths:
        found = sorted(path.rglob("*")) if path.is_dir() else [path]
        out += [p for p in found if p.is_file() and p.suffix.lower() == ".pdf"]
    return list(dict.fromkeys(p.resolve() for p in out))


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Parser:
    """Разбор статей по одной: процесс docling живёт, пока идёт импорт (модели
    грузятся один раз), рисунки сохраняются и, если надо, описываются моделью."""

    def __init__(
        self,
        settings: Settings,
        articles: Path,
        describe: bool = False,
        kb: KB | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.settings = settings
        self.articles = articles
        self.describe = describe
        self.kb = kb
        self.log = log
        self.worker = DoclingWorker(settings)
        self.logger = StepLogger(settings.log_dir, prefix="import")

    def close(self) -> None:
        self.worker.close()

    def parse(self, record: dict[str, Any], pdf: Path, data: bytes) -> Item:
        """Разобрать PDF статьи и записать всё в папку статей. record — её запись
        (метаданные); дополняется тем, как прошёл разбор."""
        doc_id = record["doc_id"]
        inp = ParseInput(doc_id=doc_id, data=data, display_name=pdf.name, origin=str(pdf))
        result, parsed, chunks = process_document(
            inp, self.settings, self.worker, self.logger, write_outputs=False
        )
        figures: list[dict[str, Any]] = []
        if parsed is not None and parsed.docling_doc:
            figures = save_images(parsed.docling_doc, doc_id, self.articles)
            keep_descriptions(record.get("figures") or [], figures)
            if self.describe and result.status in (OK, PARTIAL) and chunks:
                self._describe(doc_id, figures)
            if result.status in (OK, PARTIAL) and chunks:
                chunks = [*chunks, *description_chunks(doc_id, figures, len(chunks))]
        status = {OK: NEW, PARTIAL: NEW, SCAN: SCANNED}.get(result.status, FAILED)
        ocr = result.variant == SCAN_VARIANT and status == NEW  # скан, распознанный OCR
        note = {SCANNED: "скан: текстового слоя нет — не разбирается (--ocr — распознать)"}.get(
            status, "" if status == NEW else "не прошёл проверку, нужен человек"
        )
        if status == NEW and ocr:
            note = "скан, текст распознан (OCR) — возможны ошибки распознавания"
        if status == NEW and not chunks:
            # Текст есть, а фрагментов нет — нарезка не удалась: в базу класть нечего.
            status, note = FAILED, "фрагментов нет: нарезка не удалась — причина в журнале"
        record.update(
            status={NEW: "acquired", SCANNED: "scan"}.get(status, "parse_failed"),
            parser=result.parser,
            accuracy=result.accuracy,
            pages=result.pages,
            sections=result.sections,
            tables=result.tables,
            chunks=len(chunks),
            duration_sec=round(result.duration_sec, 2),
            failed_checks=result.failed_checks,
            attempts=result.attempts,
            cases=result.cases,
            figures=figures,
            ocr=ocr,
            parsed_at=_now(),
            note=note,
        )
        store.write(self.articles, record, None if status == SCANNED else parsed, chunks)
        return Item(
            str(pdf),
            status,
            doc_id,
            str(record.get("title") or ""),
            len(chunks),
            len(figures),
            record["note"],
        )

    def _describe(self, doc_id: str, figures: list[dict[str, Any]]) -> None:
        """Рисунки статьи — моделью со зрением; описание — в запись рисунка."""
        from ..llm import LLM

        step = time.monotonic()
        client = LLM(model=FIGURE_MODEL, host=FIGURE_HOST, temperature=0, timeout=300)
        done = describe_saved(figures, self.articles, client, kb=self.kb, log=self.log)
        self.logger.log(doc_id, "figures", OK, time.monotonic() - step, detail=f"описано: {done}")


def keep_descriptions(old: list[dict[str, Any]], new: list[dict[str, Any]]) -> int:
    """Описания рисунков прежнего разбора — тем же рисункам нового (та же страница и
    подпись): переразбор не стирает работу модели со зрением. Сколько перенесено."""
    known = {
        (f.get("page"), " ".join(str(f.get("caption") or "").split())): f["description"]
        for f in old
        if isinstance(f, dict) and f.get("description")
    }
    moved = 0
    for f in new:
        text = known.get((f.get("page"), " ".join(str(f.get("caption") or "").split())))
        if text and not f.get("description"):
            f["description"] = text
            moved += 1
    return moved


def import_pdfs(
    paths: list[Path],
    settings: Settings,
    articles: Path,
    *,
    ask: bool = False,
    force: bool = False,
    describe: bool = False,
    kb: KB | None = None,
    log: Callable[[str], None] = print,
    question: Callable[[str], str] = input,
) -> list[Item]:
    """PDF → папка статей. force — разобрать и тот файл, что уже был."""
    pdfs = find_pdfs(paths)
    found = store.records(articles)
    hashes = store.by_hash(found)
    entries: dict[Path, list[dict[str, Any]]] = {}
    items: list[Item] = []
    parser: Parser | None = None
    try:
        for i, pdf in enumerate(pdfs, start=1):
            log(f"\n[{i}/{len(pdfs)}] {pdf.name}")
            try:
                data = pdf.read_bytes()
            except OSError as exc:
                items.append(Item(str(pdf), READ_ERROR, note=str(exc)))
                continue
            sha = sha256_of(data)
            known = hashes.get(sha)
            if known and not force:
                items.append(_attach(articles, found[known], pdf, log))
                continue
            if pdf.parent not in entries:
                entries[pdf.parent] = meta.bib_entries(pdf.parent)
            try:
                fields, origin = meta.collect(pdf, data, entries[pdf.parent])
            except Exception as exc:  # noqa: BLE001 — битый PDF или YAML: эту статью пропустить
                items.append(Item(str(pdf), READ_ERROR, note=f"{type(exc).__name__}: {exc}"))
                continue
            if ask:
                fields = meta.ask_missing(fields, pdf, question)
            doc_id = known or meta.doc_id_for(fields, pdf, sha)
            same_doi = found.get(doc_id)
            if same_doi and not force and same_doi.get("sha256") != sha:
                # Та же статья (тот же DOI), но другой файл: препринт и журнальная версия.
                items.append(
                    Item(
                        str(pdf),
                        SAME,
                        doc_id,
                        str(same_doi.get("title") or ""),
                        note="статья с этим DOI уже есть (другой файл) — заменить: --force",
                    )
                )
                continue
            record = {
                "doc_id": doc_id,
                "source": "import",
                "title": fields.get("title") or pdf.stem,
                "authors": fields.get("authors") or [],
                "year": fields.get("year"),
                "journal": fields.get("journal"),
                "doi": fields.get("doi"),
                "language": fields.get("language"),
                "url": fields.get("url")
                or (f"https://doi.org/{fields['doi']}" if fields.get("doi") else ""),
                "pdf_path": str(pdf),
                "sha256": sha,
                "bytes_len": len(data),
                "fetched_at": _now(),  # ingest пишет это время в базу как «когда получена»
                "meta_from": origin,
            }
            parser = parser or Parser(settings, articles, describe, kb, log)
            item = parser.parse(record, pdf, data)
            item.origin = origin
            items.append(item)
            found[doc_id] = record
            hashes[sha] = doc_id
            log(
                f"  {item.status}: {item.chunks} фрагментов, рисунков {item.figures}"
                + (f" — {item.note}" if item.note else "")
            )
    finally:
        if parser is not None:
            parser.close()
    return items


def _attach(articles: Path, record: dict[str, Any], pdf: Path, log: Callable[[str], None]) -> Item:
    """Тот же файл уже разобран. Если у записи нет пути к PDF или файл переехал —
    путь дописывается: по нему статью можно переразобрать."""
    title = str(record.get("title") or "")
    if record.get("pdf_path") == str(pdf):
        log(f"  уже есть: {record['doc_id']}")
        return Item(str(pdf), SAME, record["doc_id"], title, int(record.get("chunks") or 0))
    status = SAME if record.get("pdf_path") else ATTACHED
    record["pdf_path"] = str(pdf)
    store.save_record(articles, record)
    log(f"  {status}: {record['doc_id']} — путь к PDF записан")
    return Item(str(pdf), status, record["doc_id"], title, int(record.get("chunks") or 0))


def reparse(
    doc_ids: list[str],
    settings: Settings,
    articles: Path,
    *,
    describe: bool = False,
    kb: KB | None = None,
    log: Callable[[str], None] = print,
) -> list[Item]:
    """Разобрать заново статьи, у которых есть PDF (и это тот самый файл)."""
    found = store.records(articles)
    items: list[Item] = []
    parser: Parser | None = None
    try:
        for i, doc_id in enumerate(doc_ids, start=1):
            record = found.get(doc_id)
            title = str((record or {}).get("title") or "")
            path = Path(str((record or {}).get("pdf_path") or ""))
            log(f"\n[{i}/{len(doc_ids)}] {doc_id}")
            if record is None or not record.get("pdf_path") or not path.is_file():
                items.append(
                    Item(
                        str(path) if record and record.get("pdf_path") else "",
                        NO_PDF,
                        doc_id,
                        title,
                        note="PDF нет: импортируйте файл (import) — путь запишется",
                    )
                )
                continue
            data = path.read_bytes()
            if record.get("sha256") and sha256_of(data) != record["sha256"]:
                items.append(
                    Item(
                        str(path),
                        NO_PDF,
                        doc_id,
                        title,
                        note="файл по этому пути другой (отпечаток не тот) — import --force",
                    )
                )
                continue
            parser = parser or Parser(settings, articles, describe, kb, log)
            item = parser.parse(record, path, data)
            items.append(item)
            log(f"  {item.status}: {item.chunks} фрагментов, рисунков {item.figures}")
    finally:
        if parser is not None:
            parser.close()
    return items
