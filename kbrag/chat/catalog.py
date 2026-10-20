"""Вопросы о самом наборе статей: «сколько статей про Марокко», «перечисли статьи
Иноуэ», «какие статьи за 2020–2023 о радаре».

Обычный ответ идёт по 8 фрагментам — посчитать по ним статьи базы нельзя: на «сколько
статей посвящено Марокко» бот отвечал «в базе нет», хотя только в названиях их 40
(стресс-тест 20.10.2026). Здесь вопрос идёт по всей базе (её поддереву), без модели:

* тема — основы слов на разных языках от разбора вопроса (plan.py, «каталог»): статья
  засчитывается, если тема есть в названии или хотя бы в MIN_CHUNKS кусках текста —
  одно упоминание в обзоре литературы статью «о Марокко» не делает;
* год и автор — фильтры по записи статьи.

Ответ собирает код: число, сколько по названию и сколько только по тексту, список
статей. Выдумать тут нечего, поэтому и модель не нужна.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MIN_CHUNKS = 2  # кусков с темой, чтобы статья считалась о ней (без темы в названии)
LIST_MAX = 60  # статей в списке ответа — остальные числом
MIN_STEM = 3  # короче — не основа, а шум («Rif» нашёлся бы в «rift»: такие — целым словом)
WHOLE_WORD = 5  # основы короче — ищутся целым словом (CNN, SAR), длиннее — подстрокой


@dataclass
class CatalogQuery:
    """Что посчитать: тема (основы слов), годы, автор."""

    stems: list[str] = field(default_factory=list)
    topic: str = ""  # тема словами вопроса — для ответа
    year_from: int | None = None
    year_to: int | None = None
    author: str = ""

    def empty(self) -> bool:
        return not (self.stems or self.year_from or self.year_to or self.author)


def _year(value: Any) -> int | None:
    m = re.search(r"\d{4}", str(value or ""))
    return int(m.group()) if m and 1800 <= int(m.group()) <= 2100 else None


# Слова о самих статьях, а не о теме: модель клала в основы «Марокко статьи», «работы
# Иваситы», «Morocco articles» — с ними подстрока не находится нигде.
_NOT_TOPIC = re.compile(
    r"\b(?:стать[яиейюёам]*|работ[аыуе]?|публикаци\w*|исследовани\w*|articles?|papers?|"
    r"stud(?:y|ies)|publications?)\b",
    re.IGNORECASE,
)


def _clean_stem(raw: Any) -> str:
    s = _NOT_TOPIC.sub(" ", str(raw).replace("%", " ").replace("_", " "))
    s = " ".join(s.split()).strip(" ,.;:–—-")
    if re.fullmatch(r"[\d\s–—\-.,]+", s):  # годы и числа — это фильтр, а не тема
        return ""
    return s


def _author_key(author: str) -> str:
    """Фамилия для поиска по авторам — начало, без хвоста транслитерации: «Iwasita» модели
    и «Iwashita» в базе, «Inoue» и «Inouye» сходятся по началу."""
    name = (author.replace(",", " ").split() or [""])[-1]
    return name[: max(4, len(name) - 3)]


def author_pattern(author: str) -> str:
    """Регулярное выражение Postgres на фамилию: начало слова, начало фамилии, v и w —
    одно («Ivasita» и «Iwasita» модели — «Iwashita» в базе)."""
    key = re.escape(_author_key(author))
    return r"\m" + re.sub(r"[vVwW]", "[vw]", key)


def parse(data: Any) -> CatalogQuery | None:
    """«каталог» из разбора вопроса → CatalogQuery; не каталожный вопрос — None."""
    if not isinstance(data, dict):
        return None
    raw = data.get("основы") or data.get("слова") or []
    if isinstance(raw, str):  # «Sentinel-1, радар» — строкой, а не списком
        raw = re.split(r"[,;]", raw)
    author = " ".join(str(data.get("автор") or "").split())[:80]
    if author.lower() in {"null", "none", "нет"}:
        author = ""
    stems: list[str] = []
    for item in raw if isinstance(raw, list) else [raw]:
        s = _clean_stem(item)
        if author and _looks_like_name(s, author):
            continue  # фамилия автора — фильтр по авторам, а не тема текста
        if len(s) >= MIN_STEM and s.lower() not in {x.lower() for x in stems}:
            stems.append(s)
    return CatalogQuery(
        stems=stems[:12],
        topic=_clean_stem(data.get("тема") or "")[:120],
        year_from=_year(data.get("год_от")),
        year_to=_year(data.get("год_до")),
        author=author,
    )


_TRANSLIT = str.maketrans("абвгдеёжзийклмнопрстуфхцыэ", "abvgdeejziiklmnoprstufhcye")


def _looks_like_name(stem: str, author: str) -> bool:
    """Основа — та же фамилия, в том числе по-русски («Ивасита» при авторе Iwashita):
    первые три буквы сходятся при грубой транслитерации (w и v — одно)."""

    def key(text: str) -> str:
        word = (text.replace(",", " ").split() or [""])[-1]
        return word.lower().translate(_TRANSLIT).replace("w", "v")[:3]

    return len(stem.split()) <= 2 and key(stem) == key(author)


def _pattern(stem: str) -> tuple[str, str]:
    """Условие Postgres на одну основу: короткая — целым словом (\\m…\\M), иначе подстрокой."""
    if len(stem) < WHOLE_WORD and re.fullmatch(r"[\w-]+", stem):
        return "~*", rf"\m{re.escape(stem)}\M"
    return "ILIKE", "%" + stem.replace("\\", "\\\\") + "%"


def find(conn: Any, query: CatalogQuery, kb: str = "") -> list[dict[str, Any]]:
    """Статьи поддерева kb по запросу, от самых «тематических»: тема в названии, затем по
    числу кусков с темой, затем новее. У каждой — по названию ли и сколько кусков."""
    from ..index.db import in_kb

    where, params = in_kb(kb)
    conds = []
    if query.year_from:
        conds.append("d.year >= %(year_from)s")
        params["year_from"] = query.year_from
    if query.year_to:
        conds.append("d.year <= %(year_to)s")
        params["year_to"] = query.year_to
    if query.author:
        conds.append("EXISTS (SELECT 1 FROM unnest(d.authors) a WHERE a ~* %(author)s)")
        params["author"] = author_pattern(query.author)
    title_any, text_any = "false", "false"
    if query.stems:
        title, text = [], []
        for i, stem in enumerate(query.stems):
            op, value = _pattern(stem)
            params[f"s{i}"] = value
            title.append(f"d.title {op} %(s{i})s")
            text.append(f"c.text {op} %(s{i})s")
        title_any, text_any = " OR ".join(title), " OR ".join(text)
    extra = "".join(f" AND {c}" for c in conds)
    params["min_chunks"] = MIN_CHUNKS
    topic = (
        "true"
        if not query.stems
        else (
            f"({title_any}) OR (SELECT count(*) FROM chunks c WHERE c.doc_id = d.doc_id "
            f"AND ({text_any})) >= %(min_chunks)s"
        )
    )
    sql = f"""
        SELECT d.doc_id, d.title, d.year, d.authors, d.journal, d.url, d.doi,
               ({title_any}) AS by_title,
               (SELECT count(*) FROM chunks c WHERE c.doc_id = d.doc_id AND ({text_any})) AS hits
        FROM documents d
        WHERE true{where}{extra} AND ({topic})
        ORDER BY by_title DESC, hits DESC, d.year DESC NULLS LAST, d.title
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return [
        {
            "doc_id": r[0],
            "title": r[1] or r[0],
            "year": r[2],
            "authors": list(r[3] or []),
            "journal": r[4],
            "url": r[5] or "",
            "doi": r[6] or "",
            "by_title": bool(r[7]),
            "hits": int(r[8] or 0),
        }
        for r in rows
    ]


def _plural(n: int, one: str, few: str, many: str) -> str:
    m10, m100 = n % 10, n % 100
    word = (
        one if m10 == 1 and m100 != 11 else few if 2 <= m10 <= 4 and not 12 <= m100 <= 14 else many
    )
    return f"{n} {word}"


def describe(query: CatalogQuery, found: list[dict[str, Any]], base_title: str) -> tuple[str, str]:
    """Ответ на каталожный вопрос: (шапка — число и как считалось, список статей)."""
    what = []
    if query.stems:
        what.append(f"о теме «{query.topic or ', '.join(query.stems)}»")
    if query.author:
        # Как фамилия записана в самих статьях (модель пишет «Iwasita», в базе — «Iwashita»).
        rx = re.compile(author_pattern(query.author).replace(r"\m", r"\b"), re.IGNORECASE)
        names = [a for doc in found for a in doc["authors"] if rx.search(a)]
        what.append(f"с автором «{names[0] if names else query.author}»")
    if query.year_from or query.year_to:
        span = (
            f"{query.year_from}–{query.year_to}"
            if query.year_from and query.year_to
            else f"с {query.year_from}" if query.year_from else f"по {query.year_to}"
        )
        what.append(f"за {span} год")
    about = " ".join(what) or "всего"
    n = len(found)
    lines = [f"**В базе «{base_title}» {_plural(n, 'статья', 'статьи', 'статей')} {about}.**"]
    if query.stems:
        titled = sum(1 for a in found if a["by_title"])
        lines.append(
            f"Тема в названии — у {titled}, только в тексте (не меньше {MIN_CHUNKS} кусков) — "
            f"у {n - titled}. Искал: {', '.join(query.stems)}. Счёт по словам: статья, где тема "
            "названа иначе или упомянута лишь раз, не учтена."
        )
    if n > LIST_MAX:
        lines.append(f"Ниже — первые {LIST_MAX}, остальные {n - LIST_MAX} — во вкладке «Статьи».")
    head = "\n\n".join(lines)
    items = []
    for i, a in enumerate(found[:LIST_MAX], start=1):
        authors = ", ".join(a["authors"][:3]) + (" и др." if len(a["authors"]) > 3 else "")
        year = f" ({a['year']})" if a.get("year") else ""
        how = f" — {note(a, query)}" if note(a, query) else ""
        items.append(f"{i}. {a['title']}{year}{'. ' + authors if authors else ''}{how}")
    return head, "\n".join(items)


def note(article: dict[str, Any], query: CatalogQuery) -> str:
    """Почему статья в списке: тема в названии или сколько кусков текста с ней."""
    if not query.stems:
        return ""
    return "в названии" if article.get("by_title") else f"в тексте, кусков: {article.get('hits')}"
