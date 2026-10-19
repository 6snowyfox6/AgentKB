"""Служебный текст статей: что выбрасывается из фрагментов перед загрузкой в базу.

Ручная проверка 65 фрагментов против оригиналов PDF (.local/logs/quality-20261005)
показала: около 10 % фрагментов — не статья, а служебный текст журнала. Такие
фрагменты находит поиск, на них ссылается чат-бот. Чистка трёх видов:

1. Служебные разделы — целиком, по заголовку: благодарности, финансирование,
   конфликт интересов, вклад авторов, «Publisher's note», сведения об авторах.
2. Колонтитулы — кусок, который повторяется во многих фрагментах одной статьи
   и выглядит как шапка журнала (ISSN, DOI, «№ 2/2026, с. 5-25», ©, www) или
   как название самой статьи. Только повтор не годится: в статьях повторяются
   и названия районов, и столбцы таблиц — это содержание.
3. Мелочь внутри текста: заглушки формул и рисунков Docling, адреса e-mail,
   стандартные фразы издательств («All claims expressed…», лицензия CC).

Фрагмент, от которого после чистки почти ничего не осталось, выбрасывается.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

# 1. Служебные разделы. Заголовок сверяется целиком (без номера раздела и
# двоеточия): «Machine Learning contributions» — раздел статьи, а не вклад авторов.
_SERVICE_HEADING = re.compile(
    r"^(?:acknowledge?ments?|funding(?:\s+(?:statement|declaration|sources?))?|"
    r"(?:declarations?\s+of\s+)?(?:competing|conflicts?)\s+(?:of\s+)?interests?|"
    r"competing\s+interest|author(?:s|'s|s')?\s+contributions?(?:\s+statement)?|"
    r"authorship\s+contribution\s+statement|contributions?|"
    r"(?:code\s+and\s+)?data\s+(?:and\s+code\s+)?availability(?:\s+statement)?|"
    r"publisher\s*'?\s*s\s+note|declarations?|statements?\s+and\s+declarations|"
    r"ethic(?:s|al)\b.*|consent\s+(?:for\s+publication|to\s+participate)|copyright\b.*|"
    r"information\s+about\s+the\s+authors?|author\s+profile|supplementary\s+information|"
    r"abbreviations|additional\s+information|благодарност\w*|финансировани\w*|источник\w*\s+финансирования|"
    r"конфликт\s+интересов.*|вклад\s+авторов.*|информация\s+об\s+авторах|"
    r"сведения\s+об\s+авторах)$",
    re.IGNORECASE,
)
_HEADING_NUMBER = re.compile(r"^[\d.\s]+|[\s:.\d]+$")

# 2. Признаки шапки журнала в повторяющемся куске.
_HEADER_MARK = re.compile(
    r"©|\bISSN\b|www\.|https?://|h\s?t\s?t\s?p\s?s?\s?:|\bDOI\b|doi\.org|№\s*\d|"
    r"\bVol(?:ume)?\.?\s*\d|\bIssue\b|\b(?:с|р|p|pp)\.\s*\d+\s*[-–]\s*\d+|"
    r"\bJournal\b|Original\s+Research|Peer\s+Reviewed|Creative\s+Commons",
    re.IGNORECASE,
)
REPEAT_WORDS = 6  # повтор считается кусками по 6 слов
REPEAT_MIN = 3  # не меньше чем в 3 фрагментах статьи…
REPEAT_SHARE = 0.10  # …и не меньше чем в 10 % её фрагментов
TITLE_SHARE = 0.6  # колонтитул-название — не меньше 60 % слов названия
# Колонтитул короткий. Повтор длиннее — это шапка, склеенная с повторённым
# текстом статьи: лучше оставить шапку, чем вырезать текст.
MAX_HEADER_WORDS = 45

# 3. Мелочь внутри текста.
_INLINE = re.compile(
    r"<!--\s*(?:formula-not-decoded|image)\s*-->|"
    r"\[[^\]\s]+@[^\]\s]+\]\(mailto:[^)]+\)|"
    r"(?:e-?mail\s*:?\s*)?[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    re.IGNORECASE,
)
_BOILERPLATE = re.compile(
    r"All claims expressed in this (?:article|manuscript)[^.]*\.(?:\s*Any product[^.]*\.)?|"
    r"This is an open access article[^.]*\.|"
    r"©\s*(?:\d{4}\s*)?(?:The\s+)?Author\(?s?\)?[^.]*\.?|"
    r"(?:Licensed\s+)?under\s+(?:a\s+)?Creative\s+Commons[^.]*\.?|"
    r"Data\s+(?:shall|will|can)\s+be\s+made\s+available[^.]*\.?|"
    # «Publisher's note: Springer Nature remains neutral with regard to jurisdictional claims…»
    r"(?:Disclaimer\s*[./]?\s*)?Publisher\s*'?\s*s\s+Note\s*:?[^.]*\.(?:[^.]*\bsolely\b[^.]*\.)?|"
    r"Reprints\s+and\s+permissions\s+information\s+is\s+available[^.]*\.",
    re.IGNORECASE,
)
MIN_WORDS = 12  # меньше — после чистки от фрагмента ничего не осталось…
MIN_SENTENCE_WORDS = 4  # …если это не законченное предложение («Mining commenced at Polaris.»)
HEADER_ONLY_WORDS = 40  # короткий фрагмент из одних примет журнала — это шапка

# 4. Фрагмент из одних авторов и их мест работы: в абзаце статьи институты
# упоминаются редко, а здесь — через каждые несколько слов.
_AFFILIATION = re.compile(
    r"\b(?:institut\w*|universit\w*|department|faculty|school\s+of|laborator\w*|"
    r"cent(?:re|er)\b|academy|college|survey\s+of|университет\w*|институт\w*|"
    r"кафедр\w*|факультет\w*|лаборатор\w*|академи\w*|обособленное\s+подразделение)",
    re.IGNORECASE,
)
AFFILIATION_MIN = 3  # не меньше трёх упоминаний…
AFFILIATION_SHARE = 0.06  # …на каждые ~16 слов…
AFFILIATION_MAX_WORDS = 150  # …и фрагмент короткий: в длинном это вступление статьи


@dataclass
class Cleaned:
    chunks: list[dict[str, Any]]
    dropped_sections: int = 0  # служебные разделы
    dropped_empty: int = 0  # пусто после чистки или одна шапка журнала
    dropped_authors: int = 0  # одни авторы и места их работы
    trimmed: int = 0  # фрагменты, из которых что-то вырезано


def is_service_heading(heading: str) -> bool:
    return bool(_SERVICE_HEADING.match(_HEADING_NUMBER.sub("", heading.strip()).strip()))


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def is_affiliations(text: str) -> bool:
    words = _letter_words(text)
    found = len(_AFFILIATION.findall(text))
    return (
        found >= AFFILIATION_MIN
        and words < AFFILIATION_MAX_WORDS
        and found / max(words, 1) >= AFFILIATION_SHARE
    )


def _letter_words(text: str) -> int:
    """Слова из букв: «УДК 553.411.071:550.4» — это одно слово, а не шесть."""
    return len(re.findall(r"[^\W\d_]{2,}", text))


def _running_spans(texts: list[str], title: str) -> list[list[tuple[int, int]]]:
    """Куски каждого текста, которые повторяются по статье и похожи на колонтитул."""
    tokens = [list(re.finditer(r"\S+", t)) for t in texts]
    grams: Counter[str] = Counter()
    for toks in tokens:
        words = [m.group() for m in toks]
        grams.update(
            {" ".join(words[i : i + REPEAT_WORDS]) for i in range(len(words) - REPEAT_WORDS + 1)}
        )
    need = max(REPEAT_MIN, REPEAT_SHARE * len(texts))
    title_words = set(_words(title))
    spans: list[list[tuple[int, int]]] = []
    for text, toks in zip(texts, tokens, strict=True):
        words = [m.group() for m in toks]
        covered = [False] * len(words)
        for i in range(len(words) - REPEAT_WORDS + 1):
            if grams[" ".join(words[i : i + REPEAT_WORDS])] >= need:
                for j in range(i, i + REPEAT_WORDS):
                    covered[j] = True
        found: list[tuple[int, int]] = []
        i = 0
        while i < len(words):
            if not covered[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(words) and covered[j + 1]:
                j += 1
            start, end = toks[i].start(), toks[j].end()
            piece = text[start:end]
            piece_words = _words(piece)
            # Название статьи как колонтитул повторяется почти целиком; кусок
            # из него («Kwa Mutonga - Vonza area») — это просто название района.
            # Название в метаданных бывает обрезано, а к колонтитулу приклеены
            # авторы — поэтому считаем, сколько слов названия кусок покрывает.
            from_title = len(set(piece_words) & title_words)
            in_title = (
                from_title >= max(REPEAT_WORDS, TITLE_SHARE * len(title_words))
                and from_title / len(set(piece_words)) >= TITLE_SHARE
            )
            short = _letter_words(piece) <= MAX_HEADER_WORDS
            if short and (_HEADER_MARK.search(piece) or in_title):
                found.append((start, end))
            i = j + 1
        spans.append(found)
    return spans


def _cut(text: str, spans: list[tuple[int, int]]) -> str:
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + " " + text[end:]
    return text


def _tidy(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def clean_document(chunks: list[dict[str, Any]], title: str = "") -> Cleaned:
    """Фрагменты одной статьи без служебного текста. Исходные словари не меняются."""
    out = Cleaned(chunks=[])
    content = []
    for chunk in chunks:
        headings = [str(h) for h in chunk.get("headings") or []]
        if headings and is_service_heading(headings[-1]):
            out.dropped_sections += 1
        else:
            content.append(chunk)

    texts = [str(c.get("text") or "") for c in content]
    spans = _running_spans(texts, title or "")
    for chunk, text, cut in zip(content, texts, spans, strict=True):
        cleaned = _cut(text, cut)
        cleaned = _INLINE.sub(" ", cleaned)
        cleaned = _BOILERPLATE.sub(" ", cleaned)
        cleaned = _tidy(cleaned)
        words = _letter_words(cleaned)
        sentence = cleaned.rstrip().endswith((".", "!", "?"))
        too_short = words < MIN_SENTENCE_WORDS or (words < MIN_WORDS and not sentence)
        header_only = words < HEADER_ONLY_WORDS and len(set(_HEADER_MARK.findall(cleaned))) >= 2
        if too_short or header_only:
            out.dropped_empty += 1
            continue
        if is_affiliations(cleaned):
            out.dropped_authors += 1
            continue
        if cleaned == _tidy(text):
            out.chunks.append(chunk)
            continue
        out.trimmed += 1
        fixed = dict(chunk)
        fixed["text"] = cleaned
        heads = "\n".join(str(h) for h in chunk.get("headings") or [])
        fixed["embed_text"] = f"{heads}\n{cleaned}" if heads else cleaned
        out.chunks.append(fixed)
    return out
