"""Языки (П11, П12, Д21, Д22): словарь поиска по языку фрагмента, поиск по парам знаков
для китайского и японского, чистка текста только для русского и английского,
распознавание сканов."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fakedb import FakeConn

from kbrag.index import search, textlang
from kbrag.parse.orchestrator import hyphens, words
from kbrag.parse.orchestrator.profile import HYPHENATED

GERMAN = (
    "Die Erz- und Kupferlagerstätten der Region werden durch hydrothermale Prozesse "
    "gebildet, die mit dem Magmatismus verbunden sind und sich über viele Jahre entwickeln."
)
RUSSIAN = (
    "Рудные тела контролируются разломами, и метасома- тические изменения прослежены "
    "вдоль всей зоны, которая также может быть связана с интрузией гранитов."
)


# --------------------------------------------------------------------------- #
#  П11: словарь — по языку фрагмента
# --------------------------------------------------------------------------- #
def test_config_of_language() -> None:
    assert textlang.config_of("ru") == "russian"
    assert textlang.config_of("en") == "russian"  # одна конфигурация на русский с английским
    assert textlang.config_of("de") == "german"
    assert textlang.config_of("uk") == "simple"  # своего стеммера в Postgres нет
    assert textlang.config_of("zh") == "simple"
    assert textlang.config_of(None) == "russian"


def test_migration_sql() -> None:
    sql = textlang.migrate_sql()
    assert f"CREATE OR REPLACE FUNCTION {textlang.CONFIG_FUNCTION}(lang text)" in sql
    assert "IMMUTABLE" in sql and "WHEN lang = 'de' THEN 'german'::regconfig" in sql
    assert f"to_tsvector({textlang.CONFIG_FUNCTION}(language), text)" in sql
    assert "CREATE EXTENSION IF NOT EXISTS pg_trgm" in sql
    assert "%" not in sql.replace("%%", "")  # без подстановок psycopg


def test_query_configs_from_languages_of_base() -> None:
    textlang._CACHE.clear()
    conn = FakeConn([("de", 50), ("en", 40), ("uk", 3), ("ru", 2)])
    assert textlang.query_configs(conn, "geology") == ["russian", "german", "simple"]
    assert "dk.path <@ %(kb)s::ltree" in conn.log[0][0]
    conn.rows = []
    assert textlang.query_configs(conn, "geology") == ["russian", "german", "simple"]  # запомнено
    textlang._CACHE.clear()


def test_tsquery_or_over_configs() -> None:
    assert textlang.tsquery_sql(["russian"]) == "to_tsquery('russian', %(q)s)"
    assert textlang.tsquery_sql(["russian", "german"], web=True) == (
        "(websearch_to_tsquery('russian', %(q)s) || websearch_to_tsquery('german', %(q)s))"
    )
    assert textlang.tsquery_sql(["bad'; drop"]) == "to_tsquery('russian', %(q)s)"


def test_text_search_uses_configs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(textlang, "query_configs", lambda conn, kb: ["russian", "german"])
    conn = FakeConn([])
    search.text_search(conn, "Kupferlagerstätte Erz", 5)
    sql = conn.log[-1][0]
    assert "c.tsv @@ (to_tsquery('russian', %(q)s) || to_tsquery('german', %(q)s))" in sql


# --------------------------------------------------------------------------- #
#  Д21: китайский и японский — по парам знаков
# --------------------------------------------------------------------------- #
def test_cjk_grams() -> None:
    assert textlang.cjk_grams("斑岩铜矿") == ["斑岩", "岩铜", "铜矿"]
    assert textlang.cjk_grams("金 porphyry 鉱床") == ["金", "鉱床"]
    assert textlang.cjk_grams("рудный узел") == []
    assert textlang.has_cjk("что такое 斑岩") and not textlang.has_cjk("porphyry copper")


def test_cjk_search_sql() -> None:
    conn = FakeConn([])
    textlang.cjk_search(
        conn, "斑岩铜矿", 7, " AND d.year >= %(year_from)s", {"year_from": 2000}, "c.id"
    )
    sql, params = conn.log[-1]
    assert "c.language IN ('zh', 'ja', 'ko')" in sql and "strpos(c.text, g) > 0" in sql
    assert params["grams"] == ["斑岩", "岩铜", "铜矿"] and params["need"] == 2
    assert params["limit"] == 7 and params["year_from"] == 2000


def test_hybrid_adds_cjk_list(monkeypatch: pytest.MonkeyPatch) -> None:
    row = (5, "d1", 0, "斑岩铜矿床", [], [], "T", 2020, None, "", [], 1.0)
    monkeypatch.setattr(search, "vector_search", lambda *a, **k: [])
    monkeypatch.setattr(search, "text_search", lambda *a, **k: [])
    monkeypatch.setattr(textlang, "cjk_search", lambda *a, **k: [row])

    class Embedder:
        def encode_one(self, text: str) -> list[float]:
            return [0.0]

    [hit] = search.hybrid_search(FakeConn([]), Embedder(), "斑岩铜矿", require_words=True)
    assert hit.chunk_id == 5 and hit.fts_strict  # все пары нашлись — как все слова


# --------------------------------------------------------------------------- #
#  П12: чистка — только для русского и английского
# --------------------------------------------------------------------------- #
def test_own_language() -> None:
    assert words.own_language(RUSSIAN) and words.own_language("The ore bodies are " * 4)
    assert not words.own_language(GERMAN)
    assert words.own_language("коротко")  # мало текста — правила работают, как раньше


def test_hyphens_only_ru_en() -> None:
    fixed, n = hyphens.fix_line_hyphens(RUSSIAN, HYPHENATED)
    assert n == 1 and "метасоматические" in fixed
    assert hyphens.fix_line_hyphens(GERMAN, HYPHENATED) == (GERMAN, 0)


def test_homoglyphs_only_ru_en() -> None:
    mixed = "Верхняя".replace("р", "p") + " " + RUSSIAN  # латинская p в русском слове
    assert words.fix_homoglyphs(mixed).startswith("Верхняя ")
    serbian = (
        "Рудна тела су контролисана раседима и та зона је повезана са интрузијом, као и "
        "многе друге зоне у региону које су проучаване."
    )
    odd = serbian + " Mеtal"  # кириллическая е в латинском слове сербского текста
    assert words.fix_homoglyphs(odd) == odd
    assert words.join_soft_hyphens("Kupfer­ lagerstätte") == "Kupferlagerstätte"


# --------------------------------------------------------------------------- #
#  Д22: распознавание сканов
# --------------------------------------------------------------------------- #
def test_ocr_options(monkeypatch: pytest.MonkeyPatch) -> None:
    from kbrag import machine
    from kbrag.parse.config import ocr_options

    values = {"KBRAG_OCR_ENGINE": "tesseract", "KBRAG_OCR_LANGS": "ru, en,zh"}
    monkeypatch.setattr(
        machine, "value", lambda name, default, path=None: values.get(name, default)
    )
    assert ocr_options() == ("tesseract", ["rus", "eng", "chi_sim"])
    values["KBRAG_OCR_ENGINE"] = "easyocr"
    assert ocr_options() == ("easyocr", ["ru", "en", "ch_sim"])
    values["KBRAG_OCR_ENGINE"] = "непонятный"
    assert ocr_options()[0] == "easyocr"


def test_scan_with_ocr(data: Path, tmp_path: Path) -> None:
    """Скан с ocr: только docling с распознаванием (у PyMuPDF текста нет), и он разобран."""
    from kbrag.parse import pipeline
    from kbrag.parse.config import Settings
    from kbrag.parse.models import OK, ParsedDoc

    text = "Распознанный текст скана о рудных телах и разломах. " * 30
    seen: list[Any] = []

    class Worker:
        def parse(self, inp: Any) -> ParsedDoc:
            seen.append(inp)
            parsed = ParsedDoc(inp.doc_id, inp.source, "docling", OK, markdown=text)
            parsed.sections = [text]
            parsed.pages_text = {1: text}
            parsed.page_count = 1
            return parsed

    settings = Settings()
    settings.log_dir = tmp_path
    settings.ocr = True
    result, parsed, _ = pipeline.process_document(
        data / "scan.pdf",
        settings,
        Worker(),  # type: ignore[arg-type]
        pipeline.StepLogger(tmp_path),
        make_chunks=False,
        write_outputs=False,
    )
    assert [i.ocr for i in seen] == [True]
    assert result.status == OK and result.variant == "скан"
    assert result.attempts == ["docling:ok"]
