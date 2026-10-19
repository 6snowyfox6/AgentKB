"""Метаданные статьи из PDF и файлов рядом с ним.

Откуда берётся, по старшинству (что нашлось выше, то и главнее):

1. YAML рядом с PDF — `статья.yaml` (или `.yml`) с тем же именем, что у `статья.pdf`:

       title: Строение Северного блока        # или: название
       authors: [Иванов И. И., Петров П. П.]  # или: авторы; можно строкой через «;»
       year: 2021                             # или: год
       journal: Отечественная геология        # или: журнал
       doi: 10.1234/abcd
       url: https://…                         # или: ссылка
       language: ru                           # или: язык

2. BibTeX в той же папке (любой `*.bib`, например экспорт Zotero или JabRef): запись
   находится по полю `file` с именем PDF, по ключу записи, равному имени файла, или
   по DOI из самого PDF.
3. Сам PDF: его метаданные (название, авторы — если это не «Microsoft Word - …») и
   DOI с первых страниц.
4. Чего нет нигде — спрашивается в терминале (import --ask) или остаётся пустым;
   без названия статья называется по имени файла.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

FIELDS = ("title", "authors", "year", "journal", "doi", "url", "language")
# Русские ключи YAML — те же поля.
ALIASES = {
    "название": "title",
    "заглавие": "title",
    "авторы": "authors",
    "author": "authors",
    "год": "year",
    "журнал": "journal",
    "booktitle": "journal",
    "ссылка": "url",
    "язык": "language",
}
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>{}\[\]]+)", re.IGNORECASE)
YEAR = re.compile(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)")
DOI_PAGES = 2  # на скольких первых страницах искать DOI
# Не название, а след программы, в которой статью верстали.
_JUNK_TITLE = re.compile(
    r"^(microsoft\s+word|untitled|документ\d*|document\d*|\s*$)|\.(docx?|pdf|tex|rtf)$",
    re.IGNORECASE,
)


def slug(value: str) -> str:
    """Идентификатор статьи из DOI или имени файла — тот же, что у прежней добычи."""
    return re.sub(r"[^a-z0-9._-]+", "_", value.strip().lower()).strip("_")


def normalize_doi(doi: str | None) -> str | None:
    """«https://doi.org/10.3390/X», «doi:10.3390/X», « 10.3390/X. » → «10.3390/x»."""
    doi = (doi or "").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if doi.lower().startswith(prefix):
            doi = doi[len(prefix) :]
    doi = doi.strip().rstrip(".,;)").lower()
    return doi if DOI.fullmatch(doi) else None


def find_doi(text: str) -> str | None:
    found = DOI.search(text or "")
    return normalize_doi(found.group(1)) if found else None


def _authors(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(v) for v in value]
    else:
        items = re.split(r"\s*;\s*|\s+and\s+|\s+и\s+(?=[А-ЯЁA-Z])", str(value or ""))
    return [" ".join(a.split()) for a in items if a and a.strip()]


def clean(data: dict[str, Any]) -> dict[str, Any]:
    """Поля к одному виду: русские ключи — английские, авторы — списком, год — числом."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        key = ALIASES.get(str(key).strip().lower(), str(key).strip().lower())
        if key not in FIELDS or value in (None, "", []):
            continue
        if key == "authors":
            value = _authors(value)
        elif key == "year":
            found = YEAR.search(str(value))
            value = int(found.group(1)) if found else None
        elif key == "doi":
            value = normalize_doi(str(value))
        else:
            value = " ".join(str(value).split())
        if value not in (None, "", []):
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
#  BibTeX
# --------------------------------------------------------------------------- #
_ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,\s]*)\s*,", re.IGNORECASE)
_FIELD = re.compile(r"\s*,?\s*([A-Za-z][\w-]*)\s*=\s*")


def _value(text: str, i: int) -> tuple[str, int]:
    """Значение поля BibTeX с позиции i: {…} со вложенными скобками, "…" или слово."""
    if text[i] == "{":
        depth, j = 0, i
        while j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                return text[i + 1 : j], j + 1
            j += 1
        return text[i + 1 :], len(text)
    if text[i] == '"':
        j = i + 1
        while j < len(text) and not (text[j] == '"' and text[j - 1] != "\\"):
            j += 1
        return text[i + 1 : j], j + 1
    found = re.match(r"[^,}\s]+", text[i:])
    word = found.group(0) if found else ""
    return word, i + len(word)


def _latex(value: str) -> str:
    value = re.sub(r"\\[a-zA-Z]+\s*", "", value) if "\\" in value else value
    return " ".join(value.replace("{", "").replace("}", "").split())


def parse_bibtex(text: str) -> list[dict[str, Any]]:
    """Записи BibTeX: {"type", "key", поля…}. Комментарии и @string пропускаются."""
    entries = []
    for match in _ENTRY.finditer(text):
        kind = match.group(1).lower()
        if kind in ("comment", "string", "preamble"):
            continue
        entry: dict[str, Any] = {"type": kind, "key": match.group(2)}
        i = match.end()
        while i < len(text):
            field = _FIELD.match(text, i)
            if not field:
                break
            name, i = field.group(1).lower(), field.end()
            if i >= len(text):
                break
            value, i = _value(text, i)
            entry[name] = _latex(value)
            while i < len(text) and text[i] in " \t\r\n":
                i += 1
            if i < len(text) and text[i] == "}":
                break
        entries.append(entry)
    return entries


def bib_entries(folder: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bib in sorted(folder.glob("*.bib")):
        try:
            out += parse_bibtex(bib.read_text(encoding="utf-8-sig", errors="replace"))
        except OSError:
            continue
    return out


def match_bib(entries: list[dict[str, Any]], pdf: Path, doi: str | None) -> dict[str, Any]:
    """Запись BibTeX этого PDF: по полю file, по ключу = имени файла, по DOI."""
    name, stem = pdf.name.lower(), pdf.stem.lower()
    for entry in entries:
        # Zotero: «Full Text PDF:files/12/имя.pdf:application/pdf», JabRef: «:имя.pdf:PDF».
        if name in str(entry.get("file", "")).lower().replace("\\", "/"):
            return entry
    for entry in entries:
        if str(entry.get("key", "")).lower() == stem:
            return entry
    if doi:
        for entry in entries:
            if normalize_doi(entry.get("doi")) == doi:
                return entry
    return {}


# --------------------------------------------------------------------------- #
#  PDF и всё вместе
# --------------------------------------------------------------------------- #
def from_pdf(data: bytes) -> dict[str, Any]:
    """Метаданные самого PDF и DOI с первых страниц."""
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        info = doc.metadata or {}
        text = "".join(doc[i].get_text() for i in range(min(DOI_PAGES, doc.page_count)))
    out: dict[str, Any] = {}
    title = " ".join(str(info.get("title") or "").split())
    if title and not _JUNK_TITLE.search(title) and len(title) >= 5:
        out["title"] = title
    if info.get("author"):
        out["authors"] = info["author"]
    doi = find_doi(text)
    if doi:
        out["doi"] = doi
    return clean(out)


def sidecar(pdf: Path) -> dict[str, Any]:
    import yaml

    for suffix in (".yaml", ".yml"):
        path = pdf.with_suffix(suffix)
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                raise ValueError(f"{path.name}: нужен словарь «поле: значение»")
            return clean(data)
    return {}


def collect(
    pdf: Path, data: bytes, entries: list[dict[str, Any]] | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Метаданные статьи и откуда каждое взято (для отчёта импорта)."""
    own = from_pdf(data)
    bib = clean(match_bib(entries or [], pdf, own.get("doi")))
    side = sidecar(pdf)
    meta: dict[str, Any] = {}
    origin: list[str] = []
    for label, source in (("yaml", side), ("bibtex", bib), ("pdf", own)):
        added = [k for k, v in source.items() if k not in meta]
        meta.update({k: source[k] for k in added})
        if added:
            origin.append(f"{label}: {', '.join(added)}")
    return meta, origin


def ask_missing(meta: dict[str, Any], pdf: Path, ask: Any = input) -> dict[str, Any]:
    """Спросить в терминале то, чего нет: Enter — оставить как есть."""
    prompts = {
        "title": "название",
        "authors": "авторы через «;»",
        "year": "год",
        "journal": "журнал",
        "doi": "DOI",
    }
    lacking = {
        "title": "названия",
        "authors": "авторов",
        "year": "года",
        "journal": "журнала",
        "doi": "DOI",
    }
    missing = [k for k in prompts if not meta.get(k)]
    if not missing:
        return meta
    print(f"\n{pdf.name}: нет {', '.join(lacking[k] for k in missing)}. Enter — пропустить.")
    answers = {k: ask(f"  {prompts[k]}: ").strip() for k in missing}
    return {**meta, **clean({k: v for k, v in answers.items() if v})}


def doc_id_for(meta: dict[str, Any], pdf: Path, sha256: str) -> str:
    """По DOI — как у прежней добычи (одна статья — один идентификатор), без DOI — по
    имени файла и началу отпечатка."""
    if meta.get("doi"):
        return slug(str(meta["doi"]))
    return f"{slug(pdf.stem)[:60] or 'pdf'}-{sha256[:8]}"
