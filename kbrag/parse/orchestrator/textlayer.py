"""Текст из PyMuPDF по рамкам docling — когда docling испортил знаки.

docling-parse (текстовый слой docling) в части старых PDF читает математические
шрифты без таблицы Unicode мусором: статья J. Comput. Phys. 1999 (Acrobat Distiller 3.0,
TeX-шрифты) — «X 0 D .‰; fi; fl/» вместо «X0 = (ρ, α, β)», «4 : 2 ¢ 10 ¡ 4» вместо
«4.2 · 10⁻⁴». PyMuPDF те же знаки читает верно. Структура, таблицы и формулы docling при
этом хороши — поэтому берётся всё от docling, а текст абзацев и ячеек — из PyMuPDF по
тем же рамкам на странице.

Когда: если знаков (не букв) у docling заметно больше, чем у PyMuPDF на тех же
страницах (garbled_share). Калибровка 14.10.2026, 10 PDF по 12 страниц, без дефисов
(их docling ставит по-своему): статья FMM — 3,2 % лишних знаков; LaTeX, Word, ABBYY,
КиберЛенинка, презентация, книга — от 0 до 0,1 %. Порог 1 %.

Абзац заменяется, только если слова совпали: те же слова от 3 букв (мусор docling — это
обрывки вроде «fi», «D»), без лишних (рамки соседних строк у docling бывают внахлёст —
тогда берётся подходящий кусок строк). Не совпало — остаётся текст docling.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Any

from ..models import ParseInput
from ..parsers.structure import FURNITURE, label_of
from .words import tidy_text

SAMPLE_PAGES = 12  # страниц для решения — как у зонда
GARBLED_SHARE = 0.01  # доля лишних знаков docling, от которой текст берётся из PyMuPDF
_IGNORED = set("-‐")  # дефисы docling расставляет по-своему — не порча

# Знаки, которые и PyMuPDF читает не тем: шрифт → {прочитано: на самом деле}. Проверено
# на статье FMM (14.10.2026): MathTime (RMTMI) — заглавные греческие, CMEX10 в перекодировке
# Distiller — скобки, сумма, интеграл, корень. Для других шрифтов — как читает PyMuPDF.
FONT_FIX = {
    "RMTMI": {"8": "Φ", "9": "Ψ", "Ä": "Ω"},
    "CMEX10": {
        "P": "∑", "X": "∑", "Z": "∫", "p": "√", "s": "√",
        "¡": "(", "µ": "(", "Ã": "(", "¢": ")", "¶": ")", "!": ")", "¯": "|",
    },
}  # fmt: skip
_SUPER = dict(
    zip(
        "0123456789+-−=()abcdefghijklmnoprstuvwxyz",
        "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ",
        strict=True,
    )
)
_KEEP_UP = set("∗†′*.,;:")  # сноски, штрих и знаки препинания «наверху» — как есть
_AFTER = [
    ("̸=", "≠"), ("7→", "↦"), ("c⃝", "©"),
    ("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("ﬄ", "ffl"),
]  # fmt: skip
_SUPER_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
GAP = 0.12  # зазор между буквами больше 0.12 кегля — пробел (в PDF пробелов в строке нет)
PAD = 1.5  # pt: рамка docling плотная, центр буквы может лечь на край
NARROW = 0.5  # сначала узкая: широкая цепляет черту соседней формулы («p ≥ 1, |»)
_L = "A-Za-zА-Яа-яЁё"
_LOWER = "a-zа-яё"

Char = tuple[str, bool, float, tuple[float, ...]]  # знак, надстрочный, кегль, рамка


# --------------------------------------------------------------------------- #
#  Когда
# --------------------------------------------------------------------------- #
def garbled_share(doc: Any, inp: ParseInput) -> float:
    """Доля знаков (не букв), которых у docling больше, чем у PyMuPDF на тех же страницах.
    doc — разметка docling (DoclingDocument)."""
    docling_text = "\n".join(
        item.text or ""
        for item in doc.texts
        if label_of(item) not in FURNITURE | {"formula"}
        and any(p.page_no <= SAMPLE_PAGES for p in item.prov or [])
    )
    with _open(inp) as pdf:
        pymupdf_text = "\n".join(
            pdf[i].get_text() for i in range(min(SAMPLE_PAGES, pdf.page_count))
        )
    ours, theirs = _symbols(docling_text), _symbols(pymupdf_text)
    excess = sum(n - theirs[c] for c, n in ours.items() if n > theirs[c])
    return excess / max(1, sum(1 for ch in docling_text if not ch.isspace()))


def _symbols(text: str) -> Counter[str]:
    return Counter(
        ch for ch in text if not ch.isalnum() and not ch.isspace() and ch not in _IGNORED
    )


# --------------------------------------------------------------------------- #
#  Как
# --------------------------------------------------------------------------- #
def restore(doc: Any, inp: ParseInput) -> dict[str, int]:
    """Текст абзацев и ячеек разметки docling (DoclingDocument) — из PyMuPDF по их рамкам.
    Формулы (LaTeX от модели) не трогаются. Замены сначала собираются, потом ставятся
    разом: сбой посередине не оставит документ наполовину заменённым."""
    stats = {"абзацев": 0, "оставлено docling": 0, "ячеек": 0}
    changes: list[tuple[Any, str]] = []  # (абзац или ячейка, новый текст)
    with _open(inp) as pdf:
        pages = {i + 1: _PageText(pdf[i]) for i in range(pdf.page_count)}
        vocab = _Vocabulary("\n".join(pdf[i].get_text() for i in range(pdf.page_count)))

    def rect_of(page_no: int, bbox: Any, pad: float) -> tuple[float, float, float, float]:
        b = bbox.to_top_left_origin(page_height=doc.pages[page_no].size.height)
        return (b.l - pad, b.t - pad, b.r + pad, b.b + pad)

    for item in doc.texts:
        label = label_of(item)
        if label in FURNITURE | {"formula"} or not item.prov or not item.text:
            continue
        tries = [
            [
                line
                for prov in item.prov
                if prov.page_no in pages
                for line in pages[prov.page_no].clip(rect_of(prov.page_no, prov.bbox, pad))
            ]
            for pad in (NARROW, PAD)
        ]
        new = _matching(item.text, tries, vocab)
        if new is None:
            stats["оставлено docling"] += 1
            continue
        if label == "list_item" and not re.match(r"\d+\.", item.text):
            new = re.sub(r"^\d+\.\s*", "", new)  # номер пункта docling хранит отдельно
        changes.append((item, new))
        stats["абзацев"] += 1

    for table in doc.tables:
        if not table.prov or table.prov[0].page_no not in pages:
            continue
        page = pages[table.prov[0].page_no]
        for cell in table.data.table_cells:
            if cell.bbox is None:
                continue
            b = cell.bbox  # у ячеек — уже от верхнего левого угла
            lines = page.clip((b.l - PAD, b.t - PAD, b.r + PAD, b.b + PAD))
            # Рамки ячеек внахлёст по строкам: в ячейку попадает показатель степени из
            # следующей строки («4.2 · 10⁻⁴», «⁻⁴») — берётся кусок с теми же буквами и цифрами.
            for i, j in _runs(len(lines)):
                candidate = " ".join(lines[i:j])
                if _letters(candidate) == _letters(cell.text) and _digits(candidate) == _digits(
                    cell.text
                ):
                    changes.append((cell, candidate))
                    stats["ячеек"] += 1
                    break

    for target, text in changes:
        target.text = text
    return stats


def _open(inp: ParseInput) -> Any:
    import pymupdf

    if inp.data is not None:
        return pymupdf.open(stream=inp.data, filetype="pdf")
    return pymupdf.open(inp.path)


class _PageText:
    """Буквы страницы со шрифтом, кеглем и флагом «надстрочный» — для выборки по рамке."""

    def __init__(self, page: Any):
        self.lines: list[list[Char]] = []
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                chars: list[Char] = []
                for span in line["spans"]:
                    fix = FONT_FIX.get(span["font"], {})
                    up = bool(span["flags"] & 1)
                    for ch in span["chars"]:
                        chars.append((fix.get(ch["c"], ch["c"]), up, span["size"], ch["bbox"]))
                self.lines.append(chars)

    def clip(self, rect: tuple[float, float, float, float]) -> list[str]:
        """Строки текста внутри рамки (по центру буквы)."""
        x0, y0, x1, y1 = rect
        out = []
        for chars in self.lines:
            inside = [
                c
                for c in chars
                if x0 <= (c[3][0] + c[3][2]) / 2 <= x1 and y0 <= (c[3][1] + c[3][3]) / 2 <= y1
            ]
            if inside:
                line = _assemble(inside)
                if line:
                    out.append(line)
        return out


def _assemble(chars: list[Char]) -> str:
    """Буквы строки → текст: надстрочные — знаками ⁻⁴, пробел там, где в PDF зазор."""
    parts: list[str] = []
    run: list[str] = []
    run_up = False
    prev: tuple[float, float] | None = None  # правый край и кегль предыдущей буквы
    for c, up, size, bbox in chars:
        if up != run_up and run:
            parts.append(_superscript("".join(run)) if run_up else "".join(run))
            run = []
        if prev is not None and c != " " and not up:
            last = run[-1] if run else (parts[-1][-1:] if parts else " ")
            index = size < 0.8 * prev[1]  # начало нижнего индекса: «X N» → «XN»
            if bbox[0] - prev[0] > GAP * min(size, prev[1]) and not index and last != " ":
                run.append(" ")
        run.append(c)
        run_up = up
        prev = (bbox[2], size)
    if run:
        parts.append(_superscript("".join(run)) if run_up else "".join(run))
    text = "".join(parts)
    for a, b in _AFTER:
        text = text.replace(a, b)
    return re.sub(r"[ \t]+", " ", text).strip()


def _superscript(text: str) -> str:
    """«−4» надстрочным → «⁻⁴»; чего нет в надстрочных знаках — «^(…)». Пробелы по краям
    остаются: «R³ with», а не «R³with»."""
    core = text.strip()
    if not core:
        return text
    if all(ch in _KEEP_UP for ch in core):
        up = core
    elif all(ch in _SUPER for ch in core):
        up = "".join(_SUPER[ch] for ch in core)
    else:
        up = f"^({core})"
    return (" " if text[0] == " " else "") + up + (" " if text[-1] == " " else "")


class _Vocabulary:
    """Как решить про дефис на конце строки без словаря, по порядку:
    1. слово стоит через дефис посреди строки где-то в документе («three-digit») — дефис;
    2. как решил сам docling в этом абзаце: буквы он читает верно, путает только знаки
       («nonuni-/form» у него «nonuniform»), но склеивает все переносы подряд — поэтому
       второй, после свидетельства самого документа («three-digit» он делает «threedigit»);
    3. слово целиком есть в документе («distributions») — склеить;
    4. ответа нет — решат правила варианта (оркестратор, «дефисы на концах строк»)."""

    def __init__(self, text: str):
        self.whole = {w.lower() for w in re.findall(rf"[{_L}]+", text)}
        self.hyphenated = {
            w.lower() for w in re.findall(rf"(?<![\w-])[{_L}]+-[{_LOWER}]+(?![\w-])", text)
        }

    def join(self, first: str, second: str, docling_text: str) -> str:
        hyphen, whole = f"{first}-{second}", first + second
        local = docling_text.lower()
        if hyphen.lower() in self.hyphenated or _has_word(local, hyphen.lower()):
            return hyphen
        if _has_word(local, whole.lower()):
            return whole
        if whole.lower() in self.whole:
            return whole
        return f"{first}- {second}"


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![{_LOWER}-]){re.escape(word)}(?![{_LOWER}-])", text) is not None


def _join_lines(lines: list[str], vocab: _Vocabulary, docling_text: str) -> str:
    text = ""
    for line in lines:
        m = re.search(rf"([{_L}]+)-$", text)
        n = re.match(rf"([{_LOWER}]+)(.*)$", line)
        if m and n:
            joined = vocab.join(m.group(1), n.group(1), docling_text)
            text = text[: m.start()] + joined + n.group(2)
        else:
            text = f"{text} {line}" if text else line
    return text


def _matching(old: str, tries: list[list[str]], vocab: _Vocabulary) -> str | None:
    """Текст из строк рамки, совпадающий по словам с текстом docling, или None.
    tries — строки узкой и широкой рамки: сначала узкая."""
    for lines in tries:
        for i, j in _runs(len(lines)):
            new = tidy_text(_join_lines(lines[i:j], vocab, old))
            if _same_words(old, new):
                return new
    # У docling слова бывают склеены («Supposethat») — тогда по словам не сравнить;
    # сравниваем буквы целиком: 0,9 пропускал потерю строки, поэтому строго.
    whole = tidy_text(_join_lines(tries[-1], vocab, old))
    ratio = difflib.SequenceMatcher(None, _letters(old), _letters(whole), autojunk=False).ratio()
    return whole if whole and ratio >= 0.97 else None


def _runs(n: int) -> list[tuple[int, int]]:
    """Подряд идущие куски строк — от самого длинного."""
    return sorted(
        ((i, j) for i in range(n) for j in range(i + 1, n + 1)), key=lambda r: r[0] - r[1]
    )


def _letters(text: str) -> str:
    return re.sub(rf"[^{_LOWER}]", "", text.lower())


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text.translate(_SUPER_DIGITS))


def _words(text: str) -> list[str]:
    return re.findall(rf"[{_LOWER}]{{3,}}", text.lower())


def _same_words(old: str, new: str) -> bool:
    """Тот же абзац? Слова от 3 букв: мусор docling в формулах — короткие обрывки
    («fi», «D»), а слова прозы оба читают одинаково. Лишние слова в новом — рамка
    захватила соседнюю строку."""
    a, b = _words(old), _words(new)
    if not a:  # строка из одних знаков: «Bl C 1» → «Bl+1»
        return bool(new) and len(new) <= 2 * len(old) + 5
    rest = Counter(b)
    hit = 0
    for w in a:
        if rest[w] > 0:
            rest[w] -= 1
            hit += 1
    return hit >= 0.9 * len(a) and sum(rest.values()) <= 0.15 * len(b)
