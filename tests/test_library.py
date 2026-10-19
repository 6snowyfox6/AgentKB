"""Импорт своих PDF (library): метаданные, повторы, привязка PDF, переразбор, рисунки.

Сам разбор подменён: проверяется то, что вокруг него, — docling и нарезка не нужны."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import pytest

from kbrag.chunks.figures import save_images
from kbrag.chunks.models import Chunk
from kbrag.library import importer, meta, store
from kbrag.parse.config import Settings
from kbrag.parse.models import OK, SCAN, ParsedDoc
from kbrag.parse.pipeline import DocResult

pymupdf = pytest.importorskip("pymupdf")


def png(width: int, height: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


def docling_doc(*sizes: tuple[int, int]) -> dict[str, Any]:
    """Разметка docling с рисунками (картинка внутри) и подписью у первого."""
    pictures = []
    for i, (w, h) in enumerate(sizes):
        uri = "data:image/png;base64," + base64.b64encode(png(w, h)).decode()
        pictures.append(
            {
                "image": {"uri": uri},
                "captions": [{"$ref": "#/texts/0"}] if i == 0 else [],
                "prov": [{"page_no": i + 2}],
            }
        )
    return {
        "texts": [{"text": "Рис. 1. Схема разломов"}],
        "pictures": pictures,
        "pages": {"1": {"image": {"uri": "data:image/png;base64,AAAA"}}},
    }


class FakeWorker:
    def __init__(self, settings: Any) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def fake_parse(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Разбор без docling: скан — по имени файла, у остальных два фрагмента и рисунок."""
    calls: list[str] = []

    def process(inp: Any, settings: Any, worker: Any, logger: Any, **kw: Any) -> Any:
        calls.append(inp.doc_id)
        if "scan" in inp.name:
            return DocResult(inp.doc_id, inp.source, "зонд", SCAN), None, []
        parsed = ParsedDoc(inp.doc_id, inp.source, "docling", OK, markdown="текст")
        parsed.docling_doc = docling_doc((400, 300), (8, 8))
        chunks = [Chunk(inp.doc_id, i, f"фрагмент {i}", f"фрагмент {i}") for i in range(2)]
        return DocResult(inp.doc_id, inp.source, "docling", OK, pages=1), parsed, chunks

    monkeypatch.setattr(importer, "process_document", process)
    monkeypatch.setattr(importer, "DoclingWorker", FakeWorker)
    return calls


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.log_dir = tmp_path / "logs"
    return s


def copy(data: Path, folder: Path, *names: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes((data / name).read_bytes())
    return folder


# --------------------------------------------------------------------------- #
#  Метаданные
# --------------------------------------------------------------------------- #
def test_doi_normalized() -> None:
    assert meta.normalize_doi("https://doi.org/10.3390/Min12.") == "10.3390/min12"
    assert meta.find_doi("см. doi:10.1103/PhysRevC.99.1, с. 3") == "10.1103/physrevc.99.1"
    assert meta.normalize_doi("не doi") is None


def test_bibtex() -> None:
    text = """@comment{x}
    @article{greek_en,
      title = {Meson {Exchange} in \\emph{Nuclear} Matter},
      author = {Smith, John and Doe, Jane},
      year = 2019, journal = "Phys. Rev. C",
      file = {Full Text PDF:files/12/greek_en.pdf:application/pdf}
    }"""
    [entry] = meta.parse_bibtex(text)
    assert entry["title"] == "Meson Exchange in Nuclear Matter"
    assert meta.clean(entry)["authors"] == ["Smith, John", "Doe, Jane"]
    assert meta.match_bib([entry], Path("x/greek_en.pdf"), None) is entry


def test_yaml_beats_bibtex_beats_pdf(data: Path, tmp_path: Path) -> None:
    folder = copy(data, tmp_path, "greek_en.pdf")
    (folder / "greek_en.yaml").write_text("название: Своё название\n", encoding="utf-8")
    entries = meta.parse_bibtex("@article{greek_en, title={Из BibTeX}, year={2019}}")
    pdf = folder / "greek_en.pdf"
    fields, origin = meta.collect(pdf, pdf.read_bytes(), entries)
    assert fields["title"] == "Своё название"  # из YAML
    assert fields["year"] == 2019  # из BibTeX
    assert origin[0].startswith("yaml")


def test_junk_pdf_title_dropped() -> None:
    doc = pymupdf.open()
    doc.new_page()
    doc.set_metadata({"title": "Microsoft Word - статья.docx", "author": "Иванов И. И."})
    fields = meta.from_pdf(doc.tobytes())
    assert "title" not in fields and fields["authors"] == ["Иванов И. И."]


def test_ask_missing() -> None:
    answers = iter(["Название", "2020", "", ""])  # авторы есть — о них не спросит
    got = meta.ask_missing({"authors": ["А"]}, Path("x.pdf"), lambda _: next(answers))
    assert got == {"authors": ["А"], "title": "Название", "year": 2020}


def test_doc_id() -> None:
    assert meta.doc_id_for({"doi": "10.1234/ab/c"}, Path("x.pdf"), "ff" * 32) == "10.1234_ab_c"
    assert meta.doc_id_for({}, Path("Моя статья.pdf"), "ab12cd34ef") == "pdf-ab12cd34"
    assert meta.doc_id_for({}, Path("Paper 1.pdf"), "ab12cd34ef") == "paper_1-ab12cd34"


# --------------------------------------------------------------------------- #
#  Рисунки
# --------------------------------------------------------------------------- #
def test_save_images(tmp_path: Path) -> None:
    doc = docling_doc((400, 300), (8, 8))
    saved = save_images(doc, "d1", tmp_path)
    assert saved == [
        {
            "n": 1,
            "label": "1",
            "file": "d1.figures/fig-001.png",
            "page": 2,
            "caption": "Рис. 1. Схема разломов",
            "width": 400,
            "height": 300,
            "bbox": None,
        }
    ]
    assert (tmp_path / "d1.figures" / "fig-001.png").is_file()
    assert doc["pictures"][0]["image"]["uri"] == "d1.figures/fig-001.png"
    assert "uri" not in doc["pictures"][1]["image"]  # точка 8×8 — не рисунок
    assert "uri" not in doc["pages"]["1"]["image"]  # картинки страниц не храним


def test_figures_read_back_from_files(tmp_path: Path) -> None:
    from kbrag.chunks.figures import load

    doc = docling_doc((400, 300))
    [meta, *_] = save_images(doc, "d1", tmp_path)
    figure = load(meta, tmp_path)
    assert figure is not None
    assert figure.page == 2 and figure.caption == "Рис. 1. Схема разломов"


# --------------------------------------------------------------------------- #
#  Импорт и переразбор
# --------------------------------------------------------------------------- #
def test_import_and_repeat(
    data: Path, tmp_path: Path, settings: Settings, fake_parse: list[str]
) -> None:
    pdfs = copy(data, tmp_path / "pdfs", "plain_ru.pdf", "greek_en.pdf", "scan.pdf")
    articles = tmp_path / "articles"
    first = importer.import_pdfs([pdfs], settings, articles, log=lambda _: None)
    assert [i.status for i in first] == [importer.NEW, importer.NEW, importer.SCANNED]
    assert [i.indexed for i in first] == [True, True, False]
    record = store.records(articles)[first[1].doc_id]
    assert record["pdf_path"] == str((pdfs / "plain_ru.pdf").resolve())
    assert record["figures"][0]["file"].endswith("fig-001.png")
    assert len(json.loads((articles / f"{first[1].doc_id}.chunks.json").read_text("utf-8"))) == 2

    again = importer.import_pdfs([pdfs], settings, articles, log=lambda _: None)
    assert [i.status for i in again] == [importer.SAME] * 3  # по отпечатку — не разбирается
    assert len(fake_parse) == 3

    forced = importer.import_pdfs(
        [pdfs / "plain_ru.pdf"], settings, articles, force=True, log=lambda _: None
    )
    assert forced[0].status == importer.NEW and forced[0].doc_id == first[1].doc_id


def test_attach_pdf_to_old_record(
    data: Path, tmp_path: Path, settings: Settings, fake_parse: list[str]
) -> None:
    """Статья прежней добычи (без PDF, тот же файл) — путь дописывается, не разбирая."""
    pdf = copy(data, tmp_path / "pdfs", "plain_ru.pdf") / "plain_ru.pdf"
    articles = tmp_path / "articles"
    sha = importer.sha256_of(pdf.read_bytes())
    store.save_record(articles, {"doc_id": "10.1_old", "sha256": sha, "url": "https://x"})
    [item] = importer.import_pdfs([pdf], settings, articles, log=lambda _: None)
    assert item.status == importer.ATTACHED and item.doc_id == "10.1_old"
    assert store.records(articles)["10.1_old"]["pdf_path"] == str(pdf.resolve())
    assert not fake_parse


def test_same_doi_other_file(
    data: Path, tmp_path: Path, settings: Settings, fake_parse: list[str]
) -> None:
    folder = copy(data, tmp_path / "pdfs", "plain_ru.pdf", "greek_en.pdf")
    for name in ("plain_ru", "greek_en"):
        (folder / f"{name}.yaml").write_text("doi: 10.1234/same\n", encoding="utf-8")
    items = importer.import_pdfs([folder], settings, tmp_path / "a", log=lambda _: None)
    assert [i.status for i in items] == [importer.NEW, importer.SAME]
    assert "--force" in items[1].note


def test_reparse(data: Path, tmp_path: Path, settings: Settings, fake_parse: list[str]) -> None:
    pdf = copy(data, tmp_path / "pdfs", "plain_ru.pdf") / "plain_ru.pdf"
    articles = tmp_path / "articles"
    [item] = importer.import_pdfs([pdf], settings, articles, log=lambda _: None)
    store.save_record(articles, {"doc_id": "old", "url": "https://x"})  # без PDF
    items = importer.reparse([item.doc_id, "old", "нет"], settings, articles, log=lambda _: None)
    assert [i.status for i in items] == [importer.NEW, importer.NO_PDF, importer.NO_PDF]
    pdf.write_bytes(pdf.read_bytes() + b"%")  # файл по пути уже другой
    [changed] = importer.reparse([item.doc_id], settings, articles, log=lambda _: None)
    assert changed.status == importer.NO_PDF and "отпечаток" in changed.note
