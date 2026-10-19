"""Рисунки статьи → фрагменты базы: подпись и описание моделью со зрением.

Подписи рисунков в базе и так есть — docling кладёт их в текст, и поиск их находит
(проверено 11.10.2026: подписи всех 1006 крупных рисунков базы — во фрагментах).
Нет того, что на самом рисунке: легенды карты, названий на схеме, осей графика,
координат. Это пишет модель со зрением — по рисунку, вырезанному
docling (`generate_picture_images`), — и получается ещё один фрагмент статьи:

    Рисунок (с. 9). Рис. 8. Схема прогноза нефтегазоносности тюменской свиты…
    Что на рисунке (описание модели по изображению): изолинии абсолютных отметок…

Проба 11.10.2026 (qwen3.5:9b) на 8 рисунках (карты, схема алгоритма, графики): по 1–2 с на
рисунок; описания по существу верны, в двух — по одной ошибке (перепутаны условные
знаки, «синий — выше» без основания); со схемы алгоритма прочитаны названия методов,
которых в подписи нет (CatBoost, LightGBM, SHAP). Включается флагом --figures при
импорте (import, reparse).

Сами рисунки сохраняются файлами рядом со статьёй (save_images): папка
<doc_id>.figures в папке статей, а в разметке docling вместо картинки — ссылка на
файл. Описывает модель по этим файлам (describe_saved) — и сразу при импорте, и потом
(python kbrag.py figures), не разбирая PDF заново.

Модель со зрением — сервис vision в docker-compose.yml (у основной Qwen3-14B
зрения нет). Как ей делить видеокарту с основной — KBRAG_VISION_MODE в
.local/settings.env (vision_mode): turns — по очереди, shared — обе сразу, single —
одна модель со зрением вместо двух. Адрес и имя можно задать и прямо: KBRAG_FIGURE_LLM,
KBRAG_FIGURE_MODEL. Как запустить — docs/КОМАНДЫ.md, «Рисунки».
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import machine
from ..kb import KB
from ..kb import get as get_kb
from ..llm import DEFAULT_MODEL, LLM_HOST
from .models import Chunk

# Как модель со зрением делит видеокарту с основной (настройка машины):
#   turns  — по очереди (16 ГБ): на время описания рисунков основная модель гасится,
#            поднимается vision, потом обратно; запускатель делает это сам;
#   shared — обе сразу: видеопамяти хватает на две (KBRAG_GPU_MEMORY и
#            KBRAG_VISION_GPU_MEMORY в сумме не больше 0,95);
#   single — одна модель со зрением вместо двух: сервис llm запускает её
#            (KBRAG_LLM_MODEL), и рисунки описывает она же.
VISION_MODES = ("turns", "shared", "single")
VISION_PORT = 8101  # снаружи; сервис vision в docker-compose.yml


def vision_mode() -> str:
    mode = machine.value("KBRAG_VISION_MODE", "turns").lower()
    return mode if mode in VISION_MODES else "turns"


def _figure_host() -> str:
    if vision_mode() == "single":
        return LLM_HOST
    return f"http://localhost:{VISION_PORT}"


FIGURE_MODEL = os.environ.get("KBRAG_FIGURE_MODEL", "").strip() or (
    DEFAULT_MODEL if vision_mode() == "single" else "qwen3-vl"
)
FIGURE_HOST = os.environ.get("KBRAG_FIGURE_LLM", "").strip() or _figure_host()
MIN_SIDE = 300  # меньше — значок, логотип, формула
MAX_SIDE = 1280  # больше модели не нужно: дольше и не точнее
HEADING = "Рисунки"
MARK = "Что на рисунке (описание модели по изображению):"

# Слова темы — из настроек базы (kb.py); шаблон целиком — «рисунок».
PROMPT = """Это рисунок из статьи по {область}. Подпись под ним: «{caption}».

Опиши рисунок так, чтобы по описанию его можно было найти поиском:
1) что это: {виды_рисунков};
2) что изображено: {что_на_рисунке};
3) легенда, подписи осей и условные знаки;
4) названия и числа, которые на рисунке читаются.
Пиши только то, что действительно видно на рисунке; неразборчивое не додумывай и
не пересказывай подпись. Пиши {язык_ответа}, 3–6 предложений."""


@dataclass
class Figure:
    page: int | None
    caption: str
    png: bytes
    size: tuple[int, int]


def _ref_text(doc: dict[str, Any], ref: dict[str, Any]) -> str:
    """Текст по ссылке docling вида {"$ref": "#/texts/12"}."""
    path = str(ref.get("$ref", "")).lstrip("#/").split("/")
    if len(path) != 2 or not path[1].isdigit():
        return ""
    items = doc.get(path[0]) or []
    index = int(path[1])
    return str(items[index].get("text", "")).strip() if index < len(items) else ""


def _picture_bytes(uri: str, base: Path | None) -> bytes | None:
    """Картинка рисунка: вложенная в разметку (data:image/…) или файл рядом со статьёй
    (после save_images ссылка — путь от папки статей)."""
    if uri.startswith("data:image/") and "," in uri:
        return base64.b64decode(uri.split(",", 1)[1])
    if uri and base is not None and not uri.startswith(("http:", "https:", "data:")):
        path = base / uri
        return path.read_bytes() if path.is_file() else None
    return None


# Номер рисунка в самой статье — из подписи: «Рис. 3.», «Рисунок 2а», «Fig. 4:», «Figure 1».
_LABEL = re.compile(r"^\s*(?:рис(?:унок)?|fig(?:ure)?)\.?\s*(\d{1,3}[а-яa-z]?)\b", re.IGNORECASE)


def label_of(caption: str) -> str | None:
    """Номер рисунка в статье по его подписи («3», «2а»); нет — None."""
    found = _LABEL.match(caption or "")
    return found.group(1).lower() if found else None


def caption_of(doc: dict[str, Any], pic: dict[str, Any]) -> str:
    return " ".join(" ".join(_ref_text(doc, c) for c in pic.get("captions") or []).split())


def page_of(pic: dict[str, Any]) -> int | None:
    prov = pic.get("prov") or []
    page = prov[0].get("page_no") if prov else None
    return int(page) if isinstance(page, int) else None


def load(meta: dict[str, Any], articles: Path) -> Figure | None:
    """Сохранённый рисунок (запись из перечня save_images) — для модели со зрением.
    Мелкие (значки, логотипы, формулы) и без подписи на первой странице (обложка,
    логотип журнала) — None: их не описываем."""
    from PIL import Image

    path = articles / str(meta.get("file") or "")
    if not meta.get("file") or not path.is_file():
        return None
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if min(image.size) < MIN_SIDE:
        return None
    caption = " ".join(str(meta.get("caption") or "").split())
    if not caption and meta.get("page") == 1:
        return None
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return Figure(meta.get("page"), caption, buf.getvalue(), image.size)


def describe_saved(
    figures: list[dict[str, Any]],
    articles: Path,
    client: Any,
    kb: KB | None = None,
    log: Any = None,
    redo: bool = False,
) -> int:
    """Описать моделью со зрением рисунки из перечня (по их файлам): описание — в поле
    description записи рисунка. Уже описанные — только с redo. Сколько описано.
    Модель не ответила на рисунок — он пропускается, остальные идут."""
    done = 0
    for meta in figures:
        if meta.get("description") and not redo:
            continue
        figure = load(meta, articles)
        if figure is None:
            continue
        try:
            text = describe(figure, client, kb)
        except Exception as exc:  # noqa: BLE001 — рисунок без описания не повод терять статью
            if log:
                log(f"рисунок {meta.get('n')} (с. {figure.page}): {type(exc).__name__}: {exc}")
            continue
        if text:
            meta["description"] = text
            done += 1
    return done


def describe(figure: Figure, client: Any, kb: KB | None = None) -> str:
    """Описание рисунка моделью со зрением; client — llm.LLM (stream)."""
    picture = "data:image/png;base64," + base64.b64encode(figure.png).decode()
    prompt = (kb or get_kb()).prompt("рисунок", PROMPT, caption=figure.caption or "нет")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": picture}},
            ],
        }
    ]
    parts = [p.get("text", "") for p in client.stream(messages, temperature=0)]
    return " ".join("".join(parts).split())


def as_chunk(doc_id: str, index: int, meta: dict[str, Any], source: str) -> Chunk:
    """Описанный рисунок — фрагмент «Рисунок»: его находит поиск и цитирует чат-бот.
    Номер рисунка в перечне — в тексте («Рисунок 3»): по нему фрагмент — к своему рисунку."""
    page = meta.get("page")
    where = f" (с. {page})" if page else ""
    caption = " ".join(str(meta.get("caption") or "").split())
    lines = [
        f"Рисунок {meta['n']}{where}. {caption}".rstrip(),
        f"{MARK} {meta.get('description') or ''}",
    ]
    text = "\n".join(lines)
    return Chunk(
        doc_id=doc_id,
        index=index,
        text=text,
        embed_text=f"{HEADING}\n{text}",
        headings=[HEADING],
        pages=[page] if page else [],
        n_tokens=len(text) // 3,
        has_table=False,
        source_path=source,
    )


def description_chunks(
    doc_id: str, figures: list[dict[str, Any]], start: int, source: str = ""
) -> list[Chunk]:
    """Фрагменты «Рисунок» из описанных рисунков; номера — после текстовых (с start)."""
    described = [f for f in figures if f.get("description")]
    return [as_chunk(doc_id, start + i, f, source) for i, f in enumerate(described)]


def is_description(chunk: dict[str, Any]) -> bool:
    """Фрагмент «Рисунок» (описание моделью), а не текст статьи."""
    return (chunk.get("headings") or [None])[0] == HEADING and MARK in str(chunk.get("text"))


FIGURES_DIR = "{doc_id}.figures"  # папка рисунков статьи — рядом с её файлами
MIN_SAVED_SIDE = 16  # меньше — точка или линия разметки, а не рисунок


def save_images(doc: dict[str, Any], doc_id: str, articles: Path) -> list[dict[str, Any]]:
    """Рисунки DoclingDocument — файлами в папку статьи, в разметке вместо картинки —
    путь к файлу от папки статей. Картинки страниц (их docling делает для разбора)
    выбрасываются: с ними файл разметки весит мегабайты.

    Возвращает перечень рисунков — номер, файл, страница, подпись, размер — для
    метаданных статьи. Прежние файлы рисунков статьи стираются: номера идут заново."""
    from PIL import Image

    folder = articles / FIGURES_DIR.format(doc_id=doc_id)
    if folder.is_dir():
        for old in folder.glob("fig-*.*"):
            old.unlink()
    saved: list[dict[str, Any]] = []
    for pic in doc.get("pictures") or []:
        image = pic.get("image")
        if not isinstance(image, dict):
            continue
        uri = str(image.get("uri", ""))
        data = _picture_bytes(uri, None)
        if data is None:
            continue
        try:
            with Image.open(io.BytesIO(data)) as img:
                size = img.size
                kind = (img.format or "PNG").lower()
        except OSError:
            image.pop("uri", None)  # не картинка — не храним
            continue
        if min(size) < MIN_SAVED_SIDE:
            image.pop("uri", None)
            continue
        number = len(saved) + 1
        name = f"fig-{number:03d}.{'jpg' if kind == 'jpeg' else kind}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(data)
        image["uri"] = f"{folder.name}/{name}"
        caption = caption_of(doc, pic)
        prov = (pic.get("prov") or [{}])[0]
        saved.append(
            {
                "n": number,
                "label": label_of(caption),
                "file": image["uri"],
                "page": page_of(pic),
                "caption": caption,
                "width": size[0],
                "height": size[1],
                "bbox": prov.get("bbox"),
            }
        )
    for page in (doc.get("pages") or {}).values():
        if isinstance(page, dict) and isinstance(page.get("image"), dict):
            page["image"].pop("uri", None)
    return saved
