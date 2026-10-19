"""Разбор эталонных PDF (tests/data, пересоздаются make_pdfs.py).

Зонд и запасной парсер (PyMuPDF) идут везде, docling — только где он установлен:
без него (проверки в CI, requirements-dev.txt) эти проверки пропускаются."""

from __future__ import annotations

from pathlib import Path

import pytest

from kbrag.parse.models import OK, SCAN, ParseInput
from kbrag.parse.orchestrator import hyphens, profile

pymupdf = pytest.importorskip("pymupdf")

VARIANTS = {
    "plain_ru.pdf": (profile.PLAIN, "ru", 2),
    "hyphenated_ru.pdf": (profile.HYPHENATED, "ru", 1),
    "formulas_ru.pdf": (profile.PLAIN, "ru", 1),
    "greek_en.pdf": (profile.PLAIN, "en", 1),
    "scan.pdf": (profile.SCAN, "en", 1),
}


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_probe_variant(data: Path, name: str) -> None:
    variant, language, pages = VARIANTS[name]
    found = profile.probe(ParseInput.of(data / name))
    assert (found.variant, found.pages) == (variant, pages), found.reason
    if variant != profile.SCAN:
        assert found.language == language


def test_text_layer_is_exact(data: Path) -> None:
    """Эталон читается так, как напечатан: обычные дефисы и пробелы, греческие буквы."""
    text = pymupdf.open(data / "greek_en.pdf")[0].get_text()
    assert "σ-meson" in text and "ω-meson" in text
    assert "­" not in text and " " not in text


def test_line_hyphens_joined(data: Path) -> None:
    """Вёрстка с переносами: «преде- лах» — одно слово (так текст отдаёт парсер)."""
    text = " ".join(pymupdf.open(data / "hyphenated_ru.pdf")[0].get_text().split())
    fixed, count = hyphens.fix_line_hyphens(text, profile.HYPHENATED)
    assert count == 8
    assert "в пределах всей рудной зоны" in fixed
    assert "северо-западного" in fixed  # настоящий дефис остаётся


def test_fallback_parser(data: Path) -> None:
    pytest.importorskip("pymupdf4llm")
    from kbrag.parse.parsers.pymupdf import parse_with_pymupdf

    parsed = parse_with_pymupdf(data / "plain_ru.pdf")
    assert parsed.status == OK and parsed.page_count == 2
    assert sorted(parsed.pages_text) == [1, 2]
    assert parsed.sections == ["Строение Северного блока", "Методы"]
    assert "Линеаментный анализ" in parsed.pages_text[2]


def test_scan_is_not_parsed(data: Path, tmp_path: Path) -> None:
    """Скан конвейер не разбирает: статус scan, без попыток парсеров."""
    from kbrag.parse import pipeline
    from kbrag.parse.config import Settings

    settings = Settings()
    settings.log_dir = tmp_path
    logger = pipeline.StepLogger(tmp_path)
    result, parsed, chunks = pipeline.process_document(
        data / "scan.pdf", settings, worker=None, logger=logger, write_outputs=False
    )
    assert result.status == SCAN and parsed.status == SCAN and not chunks


@pytest.mark.slow
def test_docling_formulas(data: Path) -> None:
    """Docling (если установлен): формулы со ссылками «(1)», «(2)» разбираются."""
    pytest.importorskip("docling")
    from kbrag.parse.config import Settings
    from kbrag.parse.parsers.docling import DoclingParser

    parsed = DoclingParser(Settings()).parse(data / "formulas_ru.pdf")
    assert parsed.status == OK, parsed.errors[:2]
    assert "Плотность пород" in parsed.text
