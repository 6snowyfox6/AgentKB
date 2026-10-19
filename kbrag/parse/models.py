"""Структуры данных, которые ходят между шагами пайплайна."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Статусы парсинга документа
OK = "ok"  # распарсили и прошли валидацию
PARTIAL = "partial"  # распарсили, но парсер сам сообщил об ошибках
FAILED = "failed"  # парсер упал
TIMEOUT = "timeout"  # процесс убит по таймауту
MANUAL_REVIEW = "manual_review"  # ни одна попытка не прошла проверку — смотреть человеку
SCAN = "scan"  # текстового слоя нет (скан) — не разбирается (profile.probe)

# Парсеры — они же шаги в журнале и в списке попыток («docling:ok»)
DOCLING = "docling"
PYMUPDF = "pymupdf4llm"  # запасной: когда docling упал, завис или не прошёл проверку
PROBE = "зонд"  # разбор остановлен зондом (скан)
PIPELINE = "конвейер"  # непредвиденный сбой в самом конвейере — причина и стек в журнале


class ParserSetupError(RuntimeError):
    """Парсер не настроился так, как задано, и разбирал бы с тихо испорченным результатом
    (модель формул docling не найдена). Это не сбой документа: разбор останавливается
    целиком, а не переходит на запасной парсер."""


@dataclass
class ParseInput:
    """Что парсим: файл на диске или байты в памяти.

    Второй вариант — у импорта (library): файл читается один раз — и для отпечатка,
    и для разбора.
    """

    doc_id: str
    path: Path | None = None
    data: bytes | None = None
    display_name: str = ""
    origin: str = ""  # ссылка на статью — с ней документ живёт дальше вместо PDF
    ocr: bool = False  # скан: распознать текст с картинок страниц (parsers/docling.py)

    @classmethod
    def of(cls, source: Path | str | ParseInput) -> ParseInput:
        if isinstance(source, ParseInput):
            return source
        path = Path(source)
        return cls(doc_id=path.stem, path=path, display_name=path.name, origin=str(path))

    @property
    def name(self) -> str:
        return self.display_name or f"{self.doc_id}.pdf"

    @property
    def source(self) -> str:
        """Откуда документ — ссылка, путь или имя: с этим он живёт в отчётах вместо PDF."""
        return self.origin or str(self.path or self.name)


@dataclass
class ParsedDoc:
    """Результат работы одного парсера над одним PDF."""

    doc_id: str
    source_path: str
    parser: str  # DOCLING | PYMUPDF | PROBE
    status: str  # OK | PARTIAL | FAILED | TIMEOUT | SCAN
    markdown: str = ""
    page_count: int = 0
    sections: list[str] = field(default_factory=list)
    table_count: int = 0
    pages_text: dict[int, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_sec: float = 0.0
    # Сериализованный DoclingDocument — нужен для чанкинга. У fallback-парсеров его нет.
    docling_doc: dict[str, Any] | None = None

    @classmethod
    def failed(
        cls,
        inp: ParseInput,
        parser: str,
        *errors: str,
        status: str = FAILED,
        duration: float = 0.0,
    ) -> ParsedDoc:
        """Пустой результат попытки: с него разбор начинается, им же кончается при сбое."""
        return cls(
            doc_id=inp.doc_id,
            source_path=inp.source,
            parser=parser,
            status=status,
            errors=list(errors),
            duration_sec=duration,
        )

    @property
    def text(self) -> str:
        return self.markdown or "\n\n".join(
            self.pages_text.get(p, "") for p in sorted(self.pages_text)
        )

    def summary(self) -> dict[str, Any]:
        """Всё, кроме разметки docling, — для отчёта рядом с результатом."""
        d = asdict(self)
        d.pop("docling_doc", None)
        d["pages_text"] = {str(k): v for k, v in self.pages_text.items()}
        return d


@dataclass
class Check:
    name: str
    ok: bool
    critical: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    ok: bool = True
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, ok: bool, critical: bool, detail: str = "") -> None:
        self.checks.append(Check(name=name, ok=ok, critical=critical, detail=detail))
        if critical and not ok:
            self.ok = False

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": [c.to_dict() for c in self.checks]}
