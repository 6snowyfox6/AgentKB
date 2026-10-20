"""Настройки этапа парсинга.

Всё, что можно захотеть покрутить, живёт здесь, а не разбросано по коду.
Пути по умолчанию (для разбора папки PDF командой python -m kbrag.parse.cli):

    .local/pdf/          — исходные PDF
    .local/parsed/       — результат: <stem>.json, <stem>.md, <stem>.chunks.json
    .local/tests/golden/ — эталоны (golden standard), имя = имя PDF + .json
    .local/logs/         — JSONL-лог по шагам
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..paths import LOCAL, LOGS


@dataclass
class Settings:
    # --- пути ---
    pdf_dir: Path = LOCAL / "pdf"
    out_dir: Path = LOCAL / "parsed"
    golden_dir: Path = LOCAL / "tests" / "golden"
    log_dir: Path = LOGS

    # --- таблицы ---
    table_mode: str = "accurate"  # accurate | fast (accurate медленнее, но держит сложные шапки)

    # --- формулы ---
    # Распознавать формулы в LaTeX (модель CodeFormulaV2, 0,64 ГБ, скачивается сама при
    # первом разборе). Без неё формула — пустая заглушка, и из базы она пропадает.
    # Проверка на проверочном PDF (.local/tests/formulas): 13 из 13 формул с латинскими
    # обозначениями верно; русские индексы модель читает латиницей («обр» → «o6p»),
    # поэтому к такой формуле приписывается её текст из PDF (formulas.fix_formula).
    formulas: bool = True
    # На сколько расширять рамку формулы, вырезая картинку для модели (у docling — 0,18).
    # Рамка и так касается соседней строки; с расширением модель видит её половину и
    # пишет по буквам в \intertext — бывает до предела длины ответа. Замер 13.10.2026:
    # статья Nature (13 формул) 0,18 → 68 с и 5588 знаков мусора, 0,05 → 13 с и 213
    # (их снимает formulas._INTERTEXT), 0 → 70 с; проверочный PDF — 13 из 15 при всех;
    # учебник (32 формулы) и страница формул — одинаково или лучше.
    formula_expansion: float = 0.05
    # Сколько формул модель распознаёт за один заход (у docling — 5). Видеокарта при 5
    # загружена на 16–22 %. Замер 14.10.2026, статья FMM (31 стр., 67 формул, RTX 5070
    # Ti): без формул 13 с, пачка 5 — 186 с, пачка 32 — 53 с; LaTeX 63 формул дословно
    # тот же, разница — в формулах, которые модель путает при любой пачке. Мало
    # видеопамяти — уменьшить.
    formula_batch_size: int = 32

    # --- рисунки ---
    # Вырезать рисунки: import сохраняет их файлами рядом со статьёй (save_images), а с
    # --figures их ещё описывает модель со зрением. Без этого docling картинок не отдаёт.
    figures: bool = False
    figures_scale: float = 2.0  # 144 точки на дюйм: мелкие подписи на карте читаются

    # --- распознавание сканов (OCR) ---
    # Скан (страницы-картинки без текстового слоя) по умолчанию не разбирается: зонд
    # останавливает его (profile.SCAN). С ocr = True (import --ocr) docling распознаёт
    # текст с картинок страниц — медленно (десятки секунд на страницу на процессоре) и
    # с ошибками распознавания, поэтому включается только для сканов и только по просьбе.
    # Движок и языки — настройки машины (.local/settings.env, ocr_options()).
    ocr: bool = False

    # --- железо ---
    device: str = "auto"  # auto | cuda | cpu | mps
    num_threads: int = 8

    # --- защита от зависаний ---
    # document_timeout внутри docling срабатывает не всегда (парсер может залипнуть
    # в нативном слое), поэтому документ считает отдельный процесс, который мы убиваем.
    doc_timeout_sec: int = 900
    max_pages: int = 400

    # --- чанкинг ---
    # Токенайзер обязан совпадать с моделью эмбеддинга, иначе счёт токенов врёт:
    # XLM-R (BGE-M3) режет русский примерно вдвое экономнее, чем англоязычные BERT-токенайзеры.
    embed_model_id: str = "BAAI/bge-m3"
    max_tokens: int = 512

    # --- пороги валидации ---
    min_chars_per_page: int = 200  # меньше — текста почти нет
    max_garbage_ratio: float = 0.10  # доля «мусорных» символов (битые шрифты, cid)
    max_stray_cjk: float = 0.01  # доля латинских слов с иероглифами внутри (validate.stray_cjk)
    min_sections_ratio: float = 0.8  # доля найденных разделов эталона
    min_entities_ratio: float = 0.8  # доля найденных сущностей эталона
    table_tolerance: int = 1  # допустимое расхождение в числе таблиц
    max_line_repeats: int = 30  # одна и та же строка N раз — признак поломки


# --------------------------------------------------------------------------- #
#  Распознавание сканов: движок и языки (настройки машины)
# --------------------------------------------------------------------------- #
OCR_ENGINES = ("easyocr", "tesseract", "rapidocr")
# Код языка (ISO 639-1) → код движка. EasyOCR: китайский — только вместе с английским.
_EASYOCR = {"zh": "ch_sim", "ja": "ja", "ko": "ko", "uk": "uk", "be": "be", "bg": "bg"}
_TESSERACT = {
    "ru": "rus",
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "uk": "ukr",
    "pl": "pol",
    "cs": "ces",
    "zh": "chi_sim",
    "ja": "jpn",
    "ko": "kor",
    "tr": "tur",
    "nl": "nld",
    "sv": "swe",
}


def ocr_options() -> tuple[str, list[str]]:
    """Движок OCR и его коды языков: KBRAG_OCR_ENGINE (easyocr — по умолчанию,
    tesseract — нужна программа tesseract, rapidocr) и KBRAG_OCR_LANGS («ru,en»)."""
    from .. import machine

    engine = machine.value("KBRAG_OCR_ENGINE", "easyocr").lower()
    if engine not in OCR_ENGINES:
        engine = "easyocr"
    langs = [x.strip().lower() for x in machine.value("KBRAG_OCR_LANGS", "ru,en").split(",")]
    langs = [x for x in langs if x] or ["ru", "en"]
    if engine == "tesseract":
        return engine, [_TESSERACT.get(x, x) for x in langs]
    if engine == "easyocr":
        return engine, [_EASYOCR.get(x, x) for x in langs]
    return engine, langs
