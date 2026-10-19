"""Автосинонимы: одно и то же имя, записанное чуть иначе, сводится в один узел само.

«тектонические нарушения» и «тектоническим нарушениям», «Landsat-8 OLI» и «Landsat8
OLI», «рудопроявление Эльвенейское» и «Эльвенейское рудопроявление» — модель пишет
по-разному, и в графе это были разные узлы. Ключ без падежей (text.name_key) сводит
не всё, словарь синонимов базы (synonyms.yaml) — только то, что в него внесли руками.

Как сводится. Имя каждой сущности превращается в вектор (BGE-M3) и хранится в
таблице entity_names с индексом HNSW. Новое имя сравнивается с уже известными —
не со всеми попарно, а через индекс (ближайшие 5): это и есть «добавить, не
пересчитывая». Похожесть от AUTO (0,95) и правила кода ниже не против — новое имя
становится написанием уже известного (canon). Иначе — своё, отдельное.

Почему порог такой высокий. Ручная проверка 100 пар соседних имён графа (2026-10-11,
.local/logs/graph-speed-20261010): похожесть ≥ 0,95 — одно и то же в 22 случаях из 25
(а с правилами кода — 22 из 23 слитых); 0,85–0,95 — только в 28 из 50, и Qwen,
решая по таким парам, ошибается в каждом третьем (верных 22 из 31). Поэтому
серая зона сама не сливается — её предлагает человеку команда synonyms. Переводы и
сокращения («золото» — «gold», «PCA») векторы тоже не ловят: у синонимов из
словаря похожесть в среднем 0,70, у «WofE» — 0,28. Для них словарь остаётся.

Словарь главнее: имя из synonyms.yaml автосинонимы не трогают.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from typing import Any

from ..text import name_key
from .synonyms import Synonyms

AUTO = 0.95  # похожесть имён, с которой сливается без человека
NEIGHBORS = 5  # сколько ближайших известных имён проверять
BATCH = 256  # сколько имён считать за раз

SCHEMA = """
CREATE TABLE IF NOT EXISTS entity_names (
    key        text PRIMARY KEY,
    name       text NOT NULL,
    embedding  vector(%(dim)s) NOT NULL,
    canon      text,
    similarity real,
    checked_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS entity_names_hnsw ON entity_names
    USING hnsw (embedding vector_cosine_ops);
"""

# Слова, которыми почти одинаковые имена различаются по сути: стороны света,
# «верхний — нижний», «высокий — низкий», «положительный — отрицательный». «Восточное
# рудное тело» и «Западное рудное тело» похожи на 0,95, но это два разных тела.
_OPPOSITE = re.compile(
    # русские — по началу слова (окончания любые), английские — целым словом:
    # иначе «min» находился в «mineralization», а «late» — в «lateral»
    r"^(?:восточн|западн|северн|южн|север|юг|восток|запад|верхн|нижн|высок|низк|ранн|"
    r"поздн|внутренн|внешн|левобереж|правобереж|положительн|отрицательн|максимальн|"
    r"минимальн|повышенн|пониженн|сильн|слаб)|"
    r"^(?:(?:north|south)?(?:east|west)(?:ern|erly|ward|wards)?|north(?:ern|erly|ward)?|"
    r"south(?:ern|erly|ward)?|upper|lower|high|higher|low|early|late|inner|outer|left|right|"
    r"positive|negative|maximum|maximal|minimum|minimal|increas\w*|decreas\w*|strong|weak|"
    r"enrich\w*|deplet\w*)$",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W\d_]+")
# «northeastern» — «northeast», «northern» — «north»: сторона света целиком, без хвоста.
_SIDE_TAIL = re.compile(r"(?:ern|erly|wards?|ly)$")


def _marks(name: str) -> list[tuple[str, str]]:
    """Слова-различители. Сторона света — целиком («northeastern» и «northern» —
    разные). Когда их два и больше — вместе со следующим словом: «high thorium and low
    uranium» и «high uranium and low thorium» — те же слова, но про разное. Одно такое
    слово в разном месте — то же самое («magnetic low zones» и «low magnetic zones»)."""
    words = _WORD.findall(name.lower())
    found = []
    for i, w in enumerate(words):
        if _OPPOSITE.match(w):
            mark = _SIDE_TAIL.sub("", w) if re.match(r"[a-z]", w) else w[:6]
            after = next((x[:5] for x in words[i + 1 :] if len(x) > 3), "")
            found.append((mark, after))
    if len(found) < 2:
        return [(mark, "") for mark, _ in found]
    return sorted(found)


def _proper(name: str) -> set[str]:
    return {w.lower()[:5] for w in _WORD.findall(name) if w[:1].isupper() and not w.isupper()}


def guard(a: str, b: str) -> str | None:
    """Почему эти два похожих имени нельзя слить без человека — или None.

    Разные числа (образец JCBN50 и JCBN45, разлом 318° и 337°), разные стороны и
    степени (восточное — западное), разные собственные имена («Монголо-Охотского» и
    «Кураминского» пояса)."""
    if sorted(re.findall(r"\d+", a)) != sorted(re.findall(r"\d+", b)):
        return "разные числа"
    if _marks(a) != _marks(b):
        return "разные стороны или степени"
    pa, pb = _proper(a), _proper(b)
    if (pa or pb) and pa != pb:
        return "разные собственные имена"
    return None


def init(conn: Any, dim: int = 1024) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA % {"dim": dim})
    conn.commit()


def _names_in_facts(conn: Any) -> dict[str, tuple[str, int]]:
    """Ключ → (самое частое написание, сколько раз) по фактам, что идут в граф."""
    from . import store

    spelled: dict[str, Counter[str]] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT src, dst FROM facts WHERE votes IS NULL OR votes >= %s", (store.MIN_VOTES,)
        )
        for src, dst in cur.fetchall():
            for raw in (src, dst):
                spelled.setdefault(name_key(raw), Counter())[raw] += 1
    return {k: (c.most_common(1)[0][0], sum(c.values())) for k, c in spelled.items() if k}


def update(
    conn: Any, embedder: Any, syn: Synonyms, redo: bool = False, log: Any = print
) -> dict[str, int]:
    """Новые имена графа → в таблицу; похожие на известные — их написаниями.

    Проходит только имена, которых в таблице ещё нет: новая статья — новые имена,
    остальное не пересчитывается. redo — всё заново (порог или правила поменялись):
    тогда первыми идут самые частые имена, и узел подписывается ими."""
    from ..index.db import vector_literal

    init(conn, len(embedder.encode_one("проверка")))
    if redo:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM entity_names")
        conn.commit()
    names = _names_in_facts(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT key FROM entity_names")
        known = {r[0] for r in cur.fetchall()}
    todo = sorted((k for k in names if k not in known), key=lambda k: -names[k][1])
    stats = {"new": len(todo), "merged": 0, "stopped_by_rules": 0}
    if not todo:
        return stats
    for start in range(0, len(todo), BATCH):
        keys = todo[start : start + BATCH]
        vectors = embedder.encode([names[k][0] for k in keys])
        with conn.cursor() as cur:
            for key, vector in zip(keys, vectors, strict=True):
                name = names[key][0]
                literal = vector_literal(vector)
                canon, similarity = None, None
                if syn.name(name) is None:  # имя из словаря сводит словарь
                    cur.execute(
                        """
                        SELECT key, name, canon, 1 - (embedding <=> %s::vector)
                        FROM entity_names ORDER BY embedding <=> %s::vector LIMIT %s
                        """,
                        (literal, literal, NEIGHBORS),
                    )
                    for other_key, other_name, other_canon, sim in cur.fetchall():
                        if sim < AUTO:
                            break
                        if guard(name, other_name):
                            stats["stopped_by_rules"] += 1
                            continue
                        canon, similarity = other_canon or other_key, float(sim)
                        break
                cur.execute(
                    """
                    INSERT INTO entity_names (key, name, embedding, canon, similarity)
                    VALUES (%s, %s, %s::vector, %s, %s) ON CONFLICT (key) DO NOTHING
                    """,
                    (key, name, literal, canon, similarity),
                )
                stats["merged"] += canon is not None
        conn.commit()
        if log:
            log(
                f"  имён проверено {min(start + BATCH, len(todo))} из {len(todo)}, "
                f"сведено к известным {stats['merged']}"
            )
    return stats


def mapping(conn: Any) -> tuple[dict[str, str], tuple[Any, ...]]:
    """Ключ имени → каноническое написание (только сведённые) и отпечаток таблицы."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT n.key, c.name FROM entity_names n JOIN entity_names c ON c.key = n.canon
                WHERE n.canon IS NOT NULL
                """)
            rows = cur.fetchall()
            cur.execute("SELECT count(*), count(canon), max(checked_at) FROM entity_names")
            stamp = tuple(cur.fetchone() or ())
    except Exception:  # noqa: BLE001 — таблицы ещё нет: автосинонимов нет
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001
            pass
        return {}, ()
    return {k: v for k, v in rows}, stamp


def extend(syn: Synonyms, auto: dict[str, str], stamp: tuple[Any, ...]) -> Synonyms:
    """Словарь + автосинонимы. Словарь главнее; в «группы» (что внесено руками)
    автосинонимы не попадают."""
    if not auto:
        return syn
    names = {**auto, **syn.names}
    return replace(syn, names=names, sha=f"{syn.sha}+auto{stamp}")
