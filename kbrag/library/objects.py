"""Формулы и рисунки статьи — из того, что лежит в папке статей после разбора.

Рисунки — файлы, сохранённые при импорте (save_images), и их перечень в записи статьи
(поле figures). Формулы — из разметки docling (<doc_id>.docling.json): LaTeX там уже
чистый и с номером «\\quad (N)» (parse/orchestrator/formulas.py). Отдельная таблица
формул со ссылками «(N)» из текста — следующий шаг (Д10, Д11); здесь — то, что можно
отдать уже сейчас: страница, LaTeX, текст из PDF и абзацы до и после.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..paths import ARTICLES
from . import store

CONTEXT_CHARS = 500  # текста до и после формулы
_NUMBER = re.compile(r"\\quad\s*\((\d{1,3})\)\s*$")  # номер, который дописала чистка
_ORIG_NUMBER = re.compile(r"(?<![\w.])\((\d{1,3})\)\s*$")  # номер в конце текста из PDF
TEXT_LABELS = {"text", "paragraph", "list_item", "caption", "section_header", "title"}


CROP_ZOOM = 3.0  # 216 точек на дюйм: индексы формулы читаются в Телеграме
CROP_MARGIN = 4.0  # пт вокруг рамки: рамка docling касается знаков


def formula_png(
    item: dict[str, Any], articles: Path = ARTICLES, zoom: float = CROP_ZOOM
) -> bytes | None:
    """Формула картинкой: вырезка со страницы PDF по рамке (точно как в статье), а
    нет PDF или рамки — LaTeX, нарисованный matplotlib (если он есть). Не вышло — None."""
    pdf = (record(str(item.get("doc_id") or ""), articles) or {}).get("pdf_path")
    bbox, page_no = item.get("bbox"), item.get("page")
    if pdf and bbox and page_no and Path(str(pdf)).is_file():
        try:
            return _crop(Path(str(pdf)), int(page_no), bbox, zoom)
        except Exception:  # noqa: BLE001 — не вырезалось: нарисуем по LaTeX
            pass
    return render_latex(str(item.get("latex") or ""))


def _crop(pdf: Path, page_no: int, bbox: dict[str, Any], zoom: float) -> bytes:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        page = doc[page_no - 1]
        left, right = float(bbox["l"]), float(bbox["r"])
        top, bottom = float(bbox["t"]), float(bbox["b"])
        if str(bbox.get("coord_origin", "BOTTOMLEFT")).upper().startswith("BOTTOM"):
            top, bottom = page.rect.height - top, page.rect.height - bottom
        rect = pymupdf.Rect(
            left - CROP_MARGIN,
            min(top, bottom) - CROP_MARGIN,
            right + CROP_MARGIN,
            max(top, bottom) + CROP_MARGIN,
        )
        pix = page.get_pixmap(clip=rect, matrix=pymupdf.Matrix(zoom, zoom))
        data: bytes = pix.tobytes("png")
        return data


def render_latex(latex: str) -> bytes | None:
    """LaTeX → PNG через mathtext matplotlib (подмножество LaTeX); нет его или формула
    ему не по силам — None (тогда показывается сам LaTeX)."""
    import io

    clean = re.sub(r"\\quad\s*\((\d{1,3})\)\s*$", r"\\quad (\1)", latex or "").strip()
    if not clean:
        return None
    try:
        from matplotlib import mathtext

        buf = io.BytesIO()
        mathtext.math_to_image(f"${clean}$", buf, dpi=200, format="png")
        return buf.getvalue()
    except Exception:  # noqa: BLE001 — нет matplotlib или не разобрал формулу
        return None


def record(doc_id: str, articles: Path = ARTICLES) -> dict[str, Any] | None:
    """Запись статьи (<doc_id>.json); нет такой — None."""
    path = articles / f"{doc_id}.json"
    if not path.is_file() or path.name.endswith(store.SUFFIXES):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def figures(doc_id: str, articles: Path = ARTICLES) -> list[dict[str, Any]]:
    """Рисунки статьи: номер, файл, страница, подпись, размер (как сохранены)."""
    found = (record(doc_id, articles) or {}).get("figures") or []
    return [f for f in found if isinstance(f, dict)]


def figure_file(doc_id: str, n: int, articles: Path = ARTICLES) -> Path | None:
    """Файл рисунка n статьи; путь — только внутри папки статей."""
    for f in figures(doc_id, articles):
        if f.get("n") == n and f.get("file"):
            path = (articles / str(f["file"])).resolve()
            if path.is_file() and articles.resolve() in path.parents:
                return path
    return None


def _page(item: dict[str, Any]) -> int | None:
    prov = item.get("prov") or []
    page = prov[0].get("page_no") if prov else None
    return int(page) if isinstance(page, int) else None


def _near(texts: list[dict[str, Any]], start: int, step: int) -> str:
    """Текст соседних абзацев (не формул) — до CONTEXT_CHARS, в сторону step."""
    out: list[str] = []
    i = start + step
    while 0 <= i < len(texts) and sum(len(t) for t in out) < CONTEXT_CHARS:
        item = texts[i]
        if item.get("label") in TEXT_LABELS and item.get("text"):
            out.append(" ".join(str(item["text"]).split()))
        i += step
    joined = " ".join(out if step > 0 else reversed(out))
    return joined[:CONTEXT_CHARS] if step > 0 else joined[-CONTEXT_CHARS:]


def formulas(doc_id: str, articles: Path = ARTICLES) -> list[dict[str, Any]]:
    """Формулы статьи по порядку: номер «(N)», LaTeX, текст из PDF, страница, рамка и
    абзацы до и после. Разметки docling нет (разобрано запасным парсером) — пусто."""
    path = articles / f"{doc_id}.docling.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    texts = [t for t in doc.get("texts") or [] if isinstance(t, dict)]
    out: list[dict[str, Any]] = []
    for i, item in enumerate(texts):
        if item.get("label") != "formula" or not item.get("text"):
            continue
        latex = str(item["text"]).strip()
        orig = " ".join(str(item.get("orig") or "").split())
        number = _NUMBER.search(latex) or _ORIG_NUMBER.search(orig)
        prov = (item.get("prov") or [{}])[0]
        out.append(
            {
                "index": len(out) + 1,
                "number": number.group(1) if number else None,
                "latex": latex,
                "pdf_text": orig,
                "page": _page(item),
                "bbox": prov.get("bbox"),
                "before": _near(texts, i, -1),
                "after": _near(texts, i, +1),
            }
        )
    return out


# --------------------------------------------------------------------------- #
#  Формула словами — для векторов (поиск по смыслу) и для людей
# --------------------------------------------------------------------------- #
# Команды LaTeX → знаки: BGE-M3 понимает «ρ = ρ0 (1 − φ)», а «\rho = \rho_0 (1 - \varphi)»
# для неё — набор служебных слов.
SYMBOLS = {
    **{
        f"\\{name}": char
        for name, char in zip(
            "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa "
            "lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi "
            "psi omega".split(),
            "αβγδεεζηθϑικλμνξπϖρϱσςτυφφχψω",
            strict=True,
        )
    },
    **{
        f"\\{name}": char
        for name, char in zip(
            "Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega".split(),
            "ΓΔΘΛΞΠΣΥΦΨΩ",
            strict=True,
        )
    },
    "\\cdot": "·",
    "\\times": "×",
    "\\pm": "±",
    "\\mp": "∓",
    "\\leq": "≤",
    "\\le": "≤",
    "\\geq": "≥",
    "\\ge": "≥",
    "\\neq": "≠",
    "\\ne": "≠",
    "\\approx": "≈",
    "\\sim": "∼",
    "\\propto": "∝",
    "\\infty": "∞",
    "\\sum": "Σ",
    "\\prod": "Π",
    "\\int": "∫",
    "\\oint": "∮",
    "\\partial": "∂",
    "\\nabla": "∇",
    "\\to": "→",
    "\\rightarrow": "→",
    "\\leftarrow": "←",
    "\\in": "∈",
    "\\degree": "°",
    "\\circ": "°",
    "\\%": "%",
    "\\ln": "ln",
    "\\log": "log",
    "\\exp": "exp",
    "\\sin": "sin",
    "\\cos": "cos",
    "\\tan": "tan",
    "\\max": "max",
    "\\min": "min",
    "\\lim": "lim",
}
_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_WRAP = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|text|textrm|operatorname|boldsymbol)\s*\{([^{}]*)\}"
)
_COMMAND = re.compile(r"\\[A-Za-z]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Абзац после формулы, который объясняет её обозначения: «где ρ — плотность…».
_DEFINES = re.compile(r"^(?:где|здесь|в которой|в котором|причём|where|here|with)\b", re.IGNORECASE)
DESCRIPTION_CHARS = 700


def plain_latex(latex: str) -> str:
    """LaTeX → строка знаками: \\frac{a}{b} → (a)/(b), \\rho → ρ, ^{2} → ^2. Не точная
    отрисовка, а то, что читается моделью векторов и человеком в описании."""
    text = re.sub(r"\\quad\s*\(\d{1,3}\)\s*$", "", latex or "")  # номер — отдельно
    text = re.sub(r"\\left|\\right|\\[,;:!]|\\quad|\\qquad|~", " ", text)
    for _ in range(4):  # изнутри наружу: индексы и степени, потом дроби и корни вокруг них
        text = _WRAP.sub(r"\1", text)
        text = re.sub(r"\^\{([^{}]*)\}", r"^\1", text)
        text = re.sub(r"_\{([^{}]*)\}", r"_\1", text)
        text = _FRAC.sub(r"(\1)/(\2)", text)
        text = _SQRT.sub(r"√(\1)", text)
    for command in sorted(SYMBOLS, key=len, reverse=True):
        text = re.sub(re.escape(command) + r"(?![A-Za-z])", SYMBOLS[command], text)
    text = _COMMAND.sub(" ", text)
    text = text.replace("{", "").replace("}", "").replace("-", "−")
    return " ".join(text.split())


def describe(item: dict[str, Any]) -> str:
    """Формула словами: номер, она сама знаками, фраза перед ней и объяснение
    обозначений после («где …»). Это идёт в вектор вместо голого LaTeX — так формулу
    находит запрос по смыслу («как плотность зависит от пористости»)."""
    number = f" ({item['number']})" if item.get("number") else ""
    parts = [f"Формула{number}: {plain_latex(item.get('latex', ''))}."]
    before = _SENTENCE_END.split(str(item.get("before") or "").strip())
    if before and before[-1]:
        parts.append(before[-1].rstrip(":").strip() + ".")
    after = str(item.get("after") or "").strip()
    if _DEFINES.match(after):
        parts.append(" ".join(_SENTENCE_END.split(after)[:2]))
    return " ".join(" ".join(parts).split())[:DESCRIPTION_CHARS]


# Ссылка на формулу в тексте: «формула (55)», «уравнение 3», «eq. (2)», «Eqs. 4» — или просто
# «(55)», если формула с таким номером в статье есть («(2015)» номером не бывает: до 3 цифр).
_NAMED_REF = re.compile(
    r"(?:формул\w*|уравнени\w*|выражени\w*|соотношени\w*|равенств\w*|"
    r"eqs?\.?|equations?|formulas?|formulae)\s*\(?\s*(\d{1,3})\s*\)?",
    re.IGNORECASE,
)
_BARE_REF = re.compile(r"(?<![\w.(])\((\d{1,3})\)(?![\w(])")


def references(text: str, numbers: set[str]) -> set[str]:
    """Номера формул статьи, на которые ссылается текст. Сами формулы ($$…$$) не в счёт:
    их собственный номер «\\quad (N)» — не ссылка."""
    text = re.sub(r"\$\$[\s\S]*?\$\$", " ", text or "")
    found = {m.group(1) for m in _NAMED_REF.finditer(text)}
    found |= {m.group(1) for m in _BARE_REF.finditer(text)}
    return found & numbers


def latex_pattern(latex: str) -> re.Pattern[str] | None:
    """Где LaTeX формулы стоит в тексте фрагмента: те же знаки, пробелы — любые."""
    tokens = (latex or "").split()
    if not tokens:
        return None
    return re.compile(r"\${0,2}" + r"\s*".join(re.escape(t) for t in tokens) + r"\${0,2}")


def mark_formulas(
    text: str, items: list[dict[str, Any]], replace: Any = None
) -> tuple[str, list[int]]:
    """Формулы во фрагменте — в $$…$$ (по ним рисует страница и их видит модель ответа),
    или replace(формула) вместо них (описание — для вектора). Какие формулы нашлись —
    их index в статье."""
    found: list[int] = []
    for item in items:
        pattern = latex_pattern(item.get("latex", ""))
        if pattern is None:
            continue
        new: str = replace(item) if replace else f"$${item['latex']}$$"
        # Строка замены — функцией: в LaTeX обратные косые, а в строке замены re их читает.
        text, count = pattern.subn(lambda _m: new, text, count=1)  # noqa: B023 — сразу
        if count:
            found.append(int(item["index"]))
    return text, found
