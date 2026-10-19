"""Ответ из графа: датасет предмета вопроса и фрагменты о нём.

«Собрать всё» и вопросы о связях: поиск приносит восемь похожих кусков, а фактов
о предмете в базе может быть из тридцати статей — граф знает все, со счётом статей
и цитатами. Вопрос о названном предмете: фрагменты, из которых выписаны факты о
нём, — в кандидаты для модели."""

from __future__ import annotations

import re
from typing import Any

# «Собрать всё» о предмете вопроса: все факты о нём из графа и фрагменты к главным.
DATASET_EVIDENCE = 5
# Вопрос о названном предмете (не «собрать всё»): фрагменты, из которых граф
# выписал факты о нём, идут модели в кандидаты вместе с найденным поиском.
GRAPH_CANDIDATES = 6
DATASET_ITEMS = 40  # сколько фактов (×2) попадает в сводку


def _synonyms(kb: str) -> Any:
    from ..graph import synonyms
    from ..kb import get

    return synonyms.load(kb=get(kb))


def _stems(text: str) -> set[str]:
    """Основы значимых слов: первые 5 букв слов от 4 букв — падеж не мешает."""
    return {w[:5] for w in re.findall(r"\w+", text.lower()) if len(w) >= 4}


def resolve_entity(g: Any, hint: str | None, question: str) -> str | None:
    """О чём вопрос: по подсказке модели или по самому вопросу — среди сущностей графа."""
    from ..text import normalize, phrase_pattern

    if hint:
        name = g.resolve(hint)
        if name:
            return str(name)
    # По тексту вопроса: самое длинное имя (или синоним), что в нём встречается.
    # Если модель назвала предмет, а граф его не знает («Анабарский щит», о котором
    # статей нет), годится только имя с общим словом из её подсказки, иначе вместо
    # предмета подхватится общее слово вопроса («геологическое строение»).
    text = normalize(question)
    want = _stems(hint) if hint else set()
    hits = []
    for name in g.entities:
        spellings = [name, *g.syn.groups.get(name, [])]
        for spelled in spellings:
            if len(spelled) < 4 or (want and not want & _stems(spelled)):
                continue
            try:
                if phrase_pattern(spelled).search(text):
                    hits.append(name)
                    break
            except re.error:
                continue
    return max(hits, key=len) if hits else None


def collect_dataset(
    conn: Any, hint: str | None, question: str, kb: str = ""
) -> dict[str, Any] | None:
    """Датасет для вопроса «собрать всё»: факты о сущности и фрагменты к главным —
    по статьям поддерева kb. None — сущность не узнана, фактов нет (граф не строили)
    или про неё ничего нет."""
    if conn is None:
        return None
    from ..graph import api, store

    try:
        syn = _synonyms(kb)
        g = api.graph(conn, syn, kb)
        name = resolve_entity(g, hint, question)
        if not name:
            return None
        data = api.dataset(conn, syn, name, g, kb)
        if not data["facts"]:
            return None
        chunk_ids = []
        for row in data["facts"]:
            for cid in row["chunk_ids"][:1]:
                if cid not in chunk_ids:
                    chunk_ids.append(cid)
            if len(chunk_ids) >= DATASET_EVIDENCE:
                break
        frags = store.fragments(conn, chunk_ids)
        data["evidence"] = [frags[c] for c in chunk_ids if c in frags]
        data["pass"] = store.summary(conn, kb)
        return data
    except Exception:  # noqa: BLE001 — нет фактов или словаря: обычный поиск
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return None


def graph_candidates(conn: Any, hint: str | None, question: str, kb: str = "") -> list[Any]:
    """Фрагменты, где сказано о предмете вопроса, — по графу, а не по похожести.

    Редкое название («Персияновский разлом») векторный поиск находит плохо:
    для него это просто похожие слова. Граф знает, из каких фрагментов о нём
    выписаны факты. Эти фрагменты — кандидаты: брать ли их, решает модель, как
    и для найденного поиском. Графа нет или предмет не узнан — пусто."""
    if conn is None or not hint:
        return []
    from ..graph import api, store
    from ..index.search import Hit

    try:
        syn = _synonyms(kb)
        g = api.graph(conn, syn, kb)
        name = resolve_entity(g, hint, question)
        if not name:
            return []
        ids: list[int] = []
        for row in api.dataset(conn, syn, name, g, kb)["facts"]:
            for cid in row["chunk_ids"]:
                if cid not in ids:
                    ids.append(cid)
            if len(ids) >= GRAPH_CANDIDATES:
                break
        ids = ids[:GRAPH_CANDIDATES]
        frags = store.fragments(conn, ids)
    except Exception:  # noqa: BLE001 — нет графа или словаря: обойдёмся поиском
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return []
    return [
        Hit(
            chunk_id=cid,
            doc_id=f["doc_id"],
            ord=f["ord"],
            text=f["text"],
            headings=f["headings"],
            pages=f["pages"],
            title=f["title"],
            year=f["year"],
            journal=f["journal"],
            url=f["url"],
            authors=f["authors"],
            found_by="граф",
        )
        for cid in ids
        if (f := frags.get(cid))
    ]


def _count(n: int) -> str:
    last2, last = n % 100, n % 10
    word = (
        "статья"
        if last == 1 and last2 != 11
        else "статьи" if 2 <= last <= 4 and not 12 <= last2 <= 14 else "статей"
    )
    return f"{n} {word}"


def dataset_summary(data: dict[str, Any]) -> str:
    """Факты о сущности одним текстом — «фрагмент» для модели."""
    includes = data.get("includes") or []
    # «Полный список» — только если модель прошла все фрагменты базы. Граф
    # строится ночью порциями, и новые статьи попадают в него не сразу.
    passed = data.get("pass") or {}
    done, total = passed.get("passed") or 0, passed.get("chunks") or 0
    scope = (
        "это полный список по графу"
        if not total or done >= total
        else f"это всё, что есть в графе, но модель прошла пока {done} из {total} фрагментов "
        "базы — в остальных может быть ещё"
    )
    lines = [
        f"Факты о «{data['name']}»"
        + (f" (вместе с тем, что в неё входит: {', '.join(includes)})" if includes else "")
        + " — выписаны моделью из статей базы и проверены по цитатам, а не найдены "
        f"поиском; {scope}. Статей: {data['documents']}."
    ]
    for row in data["facts"][: DATASET_ITEMS * 2]:
        fact = (
            f"{row['about']} — {row['relation']} — {row['other']}"
            if row["direction"] == "→"
            else f"{row['other']} — {row['relation']} — {row['about']}"
        )
        lines.append(f"- {fact} ({_count(row['documents'])})")
    more = len(data["facts"]) - DATASET_ITEMS * 2
    if more > 0:
        lines.append(f"- и ещё {more}")
    return "\n".join(lines)


def dataset_sources(data: dict[str, Any], start: int = 1) -> list[dict[str, Any]]:
    """Сводка фактов — первым фрагментом, за ней фрагменты статей к главным фактам."""
    out = [
        {
            "n": start,
            "doc_id": f"датасет:{data['name']}",
            "ord": -1,
            "title": f"Факты из графа: {data['name']}",
            "year": None,
            "journal": None,
            "authors": [],
            "pages": [],
            "headings": [],
            "url": "",
            "text": dataset_summary(data),
            "similarity": None,
            "found_by": "датасет",
            "part": None,
            "kind": "датасет",
        }
    ]
    for e in data.get("evidence") or []:
        out.append(
            {
                "n": start + len(out),
                "doc_id": e["doc_id"],
                "ord": e["ord"],
                "title": e.get("title") or e["doc_id"],
                "year": e.get("year"),
                "journal": e.get("journal"),
                "authors": e.get("authors") or [],
                "pages": e.get("pages") or [],
                "headings": e.get("headings") or [],
                "url": e.get("url") or "",
                "text": e["text"],
                "similarity": None,
                "found_by": "датасет",
                "part": None,
            }
        )
    return out
