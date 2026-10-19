"""Сбой настройки парсера останавливает разбор (П3), а не прячется в статус документа.

Парсер в проверке — подставной, но дочерний процесс настоящий (parse/parsers/worker.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kbrag.parse import pipeline
from kbrag.parse.config import Settings
from kbrag.parse.models import FAILED, ParserSetupError
from kbrag.parse.parsers.worker import DoclingWorker


def broken(settings: Any) -> Any:
    raise ParserSetupError("модель формул docling не найдена (проба)")


def missing_docling(settings: Any) -> Any:
    raise ImportError("No module named 'docling'")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.log_dir = tmp_path / "logs"
    return s


def test_setup_error_stops(settings: Settings) -> None:
    worker = DoclingWorker(settings, broken)
    try:
        with pytest.raises(ParserSetupError, match="модель формул"):
            worker.parse(Path("x.pdf"))
    finally:
        worker.close()


def test_other_start_error_is_per_document(settings: Settings) -> None:
    """Docling не ставится — это сбой документа: дальше пробует запасной парсер."""
    worker = DoclingWorker(settings, missing_docling)
    try:
        parsed = worker.parse(Path("x.pdf"))
    finally:
        worker.close()
    assert parsed.status == FAILED and "docling не загрузился" in parsed.errors[0]


def test_pipeline_lets_setup_error_through(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logger = pipeline.StepLogger(tmp_path)

    def fail(*args: Any) -> Any:
        raise ParserSetupError("проба")

    monkeypatch.setattr(pipeline, "_process", fail)
    with pytest.raises(ParserSetupError):
        pipeline.process_document(Path("x.pdf"), settings, None, logger)  # type: ignore[arg-type]

    def other(*args: Any) -> Any:
        raise ValueError("обычный сбой")

    monkeypatch.setattr(pipeline, "_process", other)
    result, _, _ = pipeline.process_document(Path("x.pdf"), settings, None, logger)  # type: ignore[arg-type]
    assert result.status == FAILED
