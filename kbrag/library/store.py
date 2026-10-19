"""Папка статей: что лежит на диске после разбора.

На статью — файлы с её идентификатором (doc_id) в имени; их читает ingest:

    <doc_id>.json          запись: метаданные, путь к PDF, отпечаток файла, как разобрана
    <doc_id>.md            текст статьи
    <doc_id>.docling.json  разметка docling — перечанковать, не разбирая PDF заново
    <doc_id>.chunks.json   фрагменты для базы
    <doc_id>.figures/      рисунки файлами (chunks/figures.py, save_images)

Сам PDF не копируется: в записи — путь к нему (pdf_path) и отпечаток (sha256).
По отпечатку один и тот же файл второй раз не разбирается, а по пути — переразбирается
(reparse). Так же лежат и статьи прежней добычи — у них вместо пути ссылка (url).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUFFIXES = (".chunks.json", ".docling.json")


def record_files(articles: Path) -> list[Path]:
    """Файлы-записи статей (<doc_id>.json), без нарезки и разметки."""
    return [p for p in sorted(articles.glob("*.json")) if not p.name.endswith(SUFFIXES)]


def records(articles: Path) -> dict[str, dict[str, Any]]:
    """Все записи папки: doc_id → запись. Нечитаемые пропускаются."""
    out: dict[str, dict[str, Any]] = {}
    for path in record_files(articles):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            out[str(data.get("doc_id") or path.stem)] = data
    return out


def by_hash(found: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Отпечаток файла → статья."""
    return {str(r["sha256"]): doc_id for doc_id, r in found.items() if r.get("sha256")}


def save_record(articles: Path, record: dict[str, Any]) -> None:
    articles.mkdir(parents=True, exist_ok=True)
    (articles / f"{record['doc_id']}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write(articles: Path, record: dict[str, Any], parsed: Any, chunks: list[Any]) -> None:
    """Запись статьи и всё, что из неё получилось. Нет фрагментов — прежние стираются:
    иначе ingest загрузил бы старую нарезку к новой записи."""
    save_record(articles, record)
    stem = record["doc_id"]
    if parsed is not None:
        (articles / f"{stem}.md").write_text(parsed.text, encoding="utf-8")
        if parsed.docling_doc:
            (articles / f"{stem}.docling.json").write_text(
                json.dumps(parsed.docling_doc, ensure_ascii=False), encoding="utf-8"
            )
    chunks_path = articles / f"{stem}.chunks.json"
    if chunks:
        chunks_path.write_text(
            json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif chunks_path.exists():
        chunks_path.unlink()
