"""Граф из фактов: сеть для страницы, датасеты, отчёт — всё с цитатами.

Факты хранятся как написала модель; здесь имена сводятся к одному:

* сначала словарь синонимов базы (synonyms.yaml) — «Donetsk basin» →
  «Донецкий бассейн»;
* чего в словаре нет — по ключу без падежей и числа: «Донецкого бассейна» и
  «Донецкий бассейн» — один узел, подписанный тем написанием, что чаще.

Датасет — у любой сущности: все факты о ней и о том, что в неё входит
(по фактам «входит в»), с цитатами и статьями. Его показывает карточка узла на
странице, отдаёт Телеграм-бот, по нему отвечает чат-бот на «что известно о …».
"""

from __future__ import annotations

import csv
import io
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .. import acl
from ..index.db import in_kb
from ..text import name_key, normalize
from . import autosyn, store
from .synonyms import PART_OF, Synonyms

ENTITY = "сущность"
ARTICLE = "статья"
QUOTES_PER_LINK = 3


class RequestError(ValueError):
    def __init__(self, message: str, detail: str = "", known: list[str] | None = None):
        super().__init__(message)
        self.message, self.detail, self.known = message, detail, known or []

    def payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {"error": self.message}
        if self.detail:
            out["detail"] = self.detail
        if self.known:
            out["known"] = self.known[:50]
        return out


@dataclass
class Link:
    """Связь между двумя сущностями: все факты с этими концами и этой связью."""

    src: str
    relation: str
    dst: str
    facts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def documents(self) -> int:
        return len({f["doc_id"] for f in self.facts})

    def quotes(self, limit: int = QUOTES_PER_LINK) -> list[dict[str, Any]]:
        out, docs = [], set()
        for f in self.facts:  # сначала — из разных статей
            if f["doc_id"] in docs:
                continue
            docs.add(f["doc_id"])
            out.append(_quote(f))
            if len(out) >= limit:
                break
        return out


def _quote(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "quote": f["quote"],
        "doc_id": f["doc_id"],
        "title": f["title"],
        "year": f["year"],
        "url": f["url"] or (f"https://doi.org/{f['doi']}" if f.get("doi") else ""),
        "chunk_id": f["chunk_id"],
    }


class Graph:
    """Факты с именами, сведёнными словарём и ключом без падежей.

    Граф не меняется после сборки: его в это время читают другие запросы. Новые
    факты дают новый граф — extended(): копируются только словари верхнего уровня
    и то, чего новые факты коснулись, остальное общее. Так новая статья
    добавляется к графу за сотые доли секунды, а не пересборкой из всех фактов
    (на 90 тысячах фактов — 19 мс против 12 с)."""

    def __init__(self, rows: list[dict[str, Any]], syn: Synonyms):
        self.syn = syn
        self._spelled: dict[str, Counter[str]] = defaultdict(Counter)
        for r in rows:
            for raw in (r["src"], r["dst"]):
                self._spelled[name_key(raw)][raw] += 1
        self.display = {k: c.most_common(1)[0][0] for k, c in self._spelled.items()}
        self.links: dict[tuple[str, str, str], Link] = {}
        self.entities: dict[str, dict[str, Any]] = {}
        self.touching: dict[str, list[Link]] = defaultdict(list)  # сущность → её связи
        self.children: dict[str, list[str]] = defaultdict(list)  # целое → что входит
        self._keys: dict[str, list[str]] = {}  # сущность → слова ключа (для related)
        self._by_key: dict[str, str] = {}  # ключ без падежей → сущность (для resolve)
        # Слова, которые где-то в графе пишутся со строчной: «high reflectance»,
        # «главная зона». Значит, «High» или «Главная» с заглавной — просто начало
        # имени, а не имя собственное.
        self._common: set[str] = set()
        self.max_id = 0  # последний факт в графе — с него extended() продолжает
        self._own: dict[str, set[Any]] | None = None  # None — всё своё, копировать нечего
        self._by_chunk: dict[int, list[tuple[Link, dict[str, Any]]]] | None = None
        self._take(rows)

    def extended(self, rows: list[dict[str, Any]]) -> Graph:
        """Этот граф и ещё факты — новым графом. Этот остаётся как был."""
        g = Graph.__new__(Graph)
        g.syn = self.syn
        g._spelled = defaultdict(Counter, self._spelled)
        g.display = dict(self.display)
        g.links = dict(self.links)
        g.entities = dict(self.entities)
        g.touching = defaultdict(list, self.touching)
        g.children = defaultdict(list, self.children)
        g._keys, g._by_key, g._common = dict(self._keys), dict(self._by_key), set(self._common)
        g.max_id = self.max_id
        g._by_chunk = None
        g._own = {"links": set(), "entities": set(), "touching": set(), "children": set()}
        fresh: dict[str, Counter[str]] = defaultdict(Counter)
        for r in rows:
            for raw in (r["src"], r["dst"]):
                key = name_key(raw)
                g._spelled[key] = g._spelled[key] + Counter({raw: 1})  # копия, не правка
                fresh[key][raw] += 1
        # Подпись узла не меняется — иначе старые связи и новые разошлись бы по разным
        # узлам. Новый ключ подписывается самым частым написанием среди новых фактов.
        for key, counts in fresh.items():
            g.display.setdefault(key, counts.most_common(1)[0][0])
        g._take(rows)
        return g

    def _mine(self, part: str, key: Any) -> bool:
        """Своё ли это у графа (можно менять) — или общее с прежним (сначала копия)."""
        if self._own is None or key in self._own[part]:
            return True
        self._own[part].add(key)
        return False

    def _take(self, rows: list[dict[str, Any]]) -> None:
        syn = self.syn
        added: list[str] = []
        for r in rows:
            self.max_id = max(self.max_id, int(r.get("id") or 0))
            src_raw, dst_raw = r["src"], r["dst"]
            # «щит включает зону» — это «зона входит в щит»: факт разворачивается.
            relation, flip = syn.relation_dir(r["relation"])
            if flip:
                src_raw, dst_raw = dst_raw, src_raw
            src, dst = self.canonical(src_raw), self.canonical(dst_raw)
            if src == dst:
                continue
            key = (src, relation, dst)
            link = self.links.get(key)
            if link is None:
                link = self.links[key] = Link(src, relation, dst)
                self._mine("links", key)
                for end in (src, dst):
                    if not self._mine("touching", end):
                        self.touching[end] = list(self.touching.get(end, []))
                    self.touching[end].append(link)
                if relation == PART_OF:
                    if not self._mine("children", dst):
                        self.children[dst] = list(self.children.get(dst, []))
                    self.children[dst].append(src)
            elif not self._mine("links", key):
                link = self.links[key] = Link(src, relation, dst, list(link.facts))
                for end in (src, dst):  # связи сущности указывают на прежнюю копию
                    if not self._mine("touching", end):
                        self.touching[end] = list(self.touching.get(end, []))
                    self.touching[end] = [
                        link if (x.src, x.relation, x.dst) == key else x for x in self.touching[end]
                    ]
            link.facts.append(r)
            for name, raw in ((src, src_raw), (dst, dst_raw)):
                e = self.entities.get(name)
                if e is None:
                    e = self.entities[name] = {
                        "name": name,
                        "facts": 0,
                        "docs": set(),
                        "spellings": set(),
                        "in_dictionary": name in syn.groups,
                    }
                    self._mine("entities", name)
                    added.append(name)
                elif not self._mine("entities", name):
                    e = self.entities[name] = {
                        **e,
                        "docs": set(e["docs"]),
                        "spellings": set(e["spellings"]),
                    }
                e["facts"] += 1
                e["docs"].add(r["doc_id"])
                if normalize(raw) != normalize(name):
                    e["spellings"].add(raw)
        for name in added:
            self._keys[name] = name_key(name).split()
            self._by_key.setdefault(" ".join(self._keys[name]), name)
            self._common.update(
                w.lower() for w in re.findall(r"[^\W\d_]+", name) if w[:1].islower()
            )

    def by_chunk(self) -> dict[int, list[tuple[Link, dict[str, Any]]]]:
        """Фрагмент → связи, выписанные из него (с самим фактом). Считается один раз
        на граф, при первом вопросе: 13 тысяч фактов — миллисекунды."""
        if self._by_chunk is None:
            index: dict[int, list[tuple[Link, dict[str, Any]]]] = defaultdict(list)
            for link in self.links.values():
                for f in link.facts:
                    index[int(f["chunk_id"])].append((link, f))
            self._by_chunk = index
        return self._by_chunk

    def canonical(self, raw: str) -> str:
        return self.syn.name(raw) or self.display.get(name_key(raw)) or raw

    def resolve(self, raw: str) -> str | None:
        """Имя, как его написал человек, → сущность графа (или None).

        Такой сущности нет, но имя собственное стоит внутри других («зоне
        Персияновского разлома», «южнее Персияновского разлома…» для
        «Персияновский разлом») — возвращается само имя: датасет соберёт их."""
        if not raw:
            return None
        name = self.syn.name(raw)
        if name in self.entities:
            return name
        entity = self._by_key.get(name_key(raw))
        if entity is not None:
            return entity
        wanted = name or " ".join(raw.split())
        return wanted if self.related(wanted) else None

    def related(self, name: str) -> list[str]:
        """Сущности, в названии которых целиком стоит это имя собственное.

        Модель пишет «зоне Персияновского разлома», «дайки вдоль Персияновского
        разлома» — это всё о разломе, но отдельные узлы. Только для имён
        собственных: у «золото» таких «родственников» сотни, и они о другом."""
        if not self.proper(name):
            return []
        key = name_key(name).split()
        if not key:
            return []
        n = len(key)
        return [
            e
            for e, words in self._keys.items()
            if e != name
            and len(words) > n
            and any(words[i : i + n] == key for i in range(len(words) - n + 1))
        ]

    def proper(self, name: str) -> bool:
        return _proper(name, self._common)

    def parts(self, name: str) -> list[str]:
        """Само имя и всё, что в него входит по фактам «входит в», на любую глубину."""
        out, frontier = [name], [name]
        while frontier:
            for part in self.children.get(frontier.pop(), []):
                if part not in out:
                    out.append(part)
                    frontier.append(part)
        return out


def _proper(name: str, common: set[str] | frozenset[str] = frozenset()) -> bool:
    """Есть ли в имени имя собственное: слово с большой буквы, не аббревиатура и не
    слово, которое в других именах пишется со строчной («High», «Hydrothermal»)."""
    return any(
        w[:1].isupper() and not w.isupper() and w.lower() not in common
        for w in re.findall(r"[^\W\d_]+", name)
    )


# Готовый граф — на каждый узел дерева баз отдельно (ключ — путь, "" — всё дерево):
# у узла свой словарь синонимов и свои статьи. Самые давние узлы забываются.
_CACHE: dict[tuple[str, Any], dict[str, Any]] = {}  # (база, доступ)
_CACHE_LOCK = threading.Lock()
CACHE_NODES = 16


def _stamp(conn: Any, kb: str = "") -> Any:
    """Отпечаток фактов поддерева kb: сколько их, последний номер, подтверждения — и
    раскладка статей по веткам (статью перенесли — граф узла другой). Только у
    настоящей базы."""
    try:
        import psycopg
    except ImportError:
        return None
    if not isinstance(conn, psycopg.Connection):
        return None
    where, params = in_kb(kb, "f.doc_id")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), coalesce(max(id), 0), count(votes), coalesce(sum(votes), 0) "
            f"FROM facts f WHERE true{where}",
            params,
        )
        row = cur.fetchone()
        if not row:
            return None
        placed: tuple[Any, ...] = ()
        if kb:
            cur.execute(
                "SELECT count(*), coalesce(sum(hashtext(doc_id || ' ' || path::text)), 0) "
                "FROM document_kbs WHERE path <@ %(kb)s::ltree",
                params,
            )
            placed = tuple(cur.fetchone() or ())
        return tuple(row) + placed


def _only_added(conn: Any, old: tuple[Any, ...], kb: str = "") -> bool:
    """Прежние факты (номер ≤ прежнего последнего) не тронуты — только дописаны новые.

    Отпечаток прежних фактов сверяется заново: сколько их, сколько проверено и сумма
    подтверждений. Удаление, перепроверка или подтверждение старого факта его меняют —
    тогда граф собирается с нуля."""
    count, last, checked, total = old[:4]
    where, params = in_kb(kb, "f.doc_id")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), count(votes), coalesce(sum(votes), 0) FROM facts f "
            f"WHERE id <= %(last)s{where}",
            {**params, "last": last},
        )
        row = cur.fetchone()
    return bool(row) and tuple(row) == (count, checked, total)


def graph(conn: Any, syn: Synonyms, kb: str = "") -> Graph:
    """Граф из фактов статей поддерева kb (пустой — всей базы). Собирается только когда
    факты, раскладка статей или словарь поменялись (раньше — на каждый запрос страницы
    и каждый вопрос чат-бота). Если к фактам только добавились новые — новая статья
    прошла через graph, — они дописываются к готовому графу (Graph.extended), а не
    собирается всё заново."""
    stamp = _stamp(conn, kb)
    if stamp is None:
        return Graph(store.all_facts(conn, kb=kb), syn)
    # Автосинонимы (autosyn.py) — к словарю; сведено новое — граф собирается заново.
    auto, auto_stamp = autosyn.mapping(conn)
    syn = autosyn.extend(syn, auto, auto_stamp)
    words = syn.sha or repr(
        (sorted(syn.names.items()), sorted(syn.relations.items()), sorted(syn.inverse.items()))
    )
    with _CACHE_LOCK:
        slot = _CACHE.get(acl.cache_key(kb)) or {}
        cached: Graph | None = slot.get("graph") if slot.get("words") == words else None
        old = slot.get("stamp")
    if cached is not None and old == stamp:
        return cached
    last = stamp[1]
    if (
        cached is not None
        and old is not None
        and old[4:] == stamp[4:]  # статьи по веткам не переносили
        and stamp[0] > old[0]
        and _only_added(conn, old, kb)
    ):
        g = cached.extended(store.all_facts(conn, after=old[1], upto=last, kb=kb))
    else:
        g = Graph(store.all_facts(conn, upto=last, kb=kb), syn)
    with _CACHE_LOCK:
        _CACHE.pop(acl.cache_key(kb), None)  # в конец: самые давние — в начале словаря
        _CACHE[acl.cache_key(kb)] = {"words": words, "stamp": stamp, "graph": g}
        while len(_CACHE) > CACHE_NODES:
            _CACHE.pop(next(iter(_CACHE)))
    return g


# --------------------------------------------------------------------------- #
#  Сеть для страницы
# --------------------------------------------------------------------------- #
def network_response(
    conn: Any, syn: Synonyms, with_articles: bool = True, kb: str = ""
) -> dict[str, Any]:
    """Узлы — сущности (и статьи), рёбра — связи с цитатами (и «из статьи»)."""
    g = graph(conn, syn, kb)
    nodes, edges = [], []
    for e in g.entities.values():
        nodes.append(
            {
                "id": f"{ENTITY}:{e['name']}",
                "kind": ENTITY,
                "name": e["name"],
                "facts": e["facts"],
                "documents": len(e["docs"]),
                "spellings": sorted(e["spellings"])[:12],
                "in_dictionary": e["in_dictionary"],
            }
        )
    for link in g.links.values():
        edges.append(
            {
                "source": f"{ENTITY}:{link.src}",
                "target": f"{ENTITY}:{link.dst}",
                "type": "факт",
                "label": link.relation,
                "weight": link.documents,
                "fragments": len(link.facts),
                "quotes": link.quotes(),
            }
        )
    if with_articles:
        docs: dict[str, dict[str, Any]] = {}
        mentioned: dict[tuple[str, str], int] = Counter()
        for link in g.links.values():
            for f in link.facts:
                docs.setdefault(f["doc_id"], f)
                mentioned[(f["doc_id"], link.src)] += 1
                mentioned[(f["doc_id"], link.dst)] += 1
        for doc_id, f in docs.items():
            nodes.append(
                {
                    "id": f"{ARTICLE}:{doc_id}",
                    "kind": ARTICLE,
                    "name": f["title"],
                    "doc_id": doc_id,
                    "year": f["year"],
                    "url": f["url"],
                    "doi_url": f"https://doi.org/{f['doi']}" if f.get("doi") else None,
                }
            )
        for (doc_id, name), n in mentioned.items():
            edges.append(
                {
                    "source": f"{ARTICLE}:{doc_id}",
                    "target": f"{ENTITY}:{name}",
                    "type": "из статьи",
                    "weight": n,
                }
            )
    return {
        "nodes": nodes,
        "edges": edges,
        "pass": store.summary(conn, kb),
        "dictionary": {"names": len(syn.groups), "sha": syn.sha},
    }


LINKS_PER_CHUNK = 12  # связей одного фрагмента под ответом — больше не читают


def facts_response(conn: Any, syn: Synonyms, chunk_ids: list[int], kb: str = "") -> dict[str, Any]:
    """Связи, которые граф выписал из этих фрагментов, — под ответом чат-бота: щёлкнуть
    имя и открыть его на графе. Ответ строит не граф, а поиск по фрагментам; граф тут —
    карта, по которой можно дальше копаться руками (замер 13.10.2026: ответам граф
    почти ничего не добавляет, а связи с цитатами людям нужны)."""
    g = graph(conn, syn, kb)
    index = g.by_chunk()
    out: dict[str, list[dict[str, Any]]] = {}
    for cid in chunk_ids:
        items, seen = [], set()
        for link, f in index.get(cid, []):
            key = (link.src, link.relation, link.dst)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "src": link.src,
                    "relation": link.relation,
                    "dst": link.dst,
                    "src_id": f"{ENTITY}:{link.src}",
                    "dst_id": f"{ENTITY}:{link.dst}",
                    "quote": f["quote"],
                    "documents": link.documents,
                    "src_facts": g.entities[link.src]["facts"],
                    "dst_facts": g.entities[link.dst]["facts"],
                }
            )
        # Сначала — связи, за которыми на графе больше всего: их есть куда раскрывать.
        items.sort(key=lambda x: -(x["src_facts"] + x["dst_facts"] + 5 * x["documents"]))
        out[str(cid)] = items[:LINKS_PER_CHUNK]
    return {"chunks": out}


# --------------------------------------------------------------------------- #
#  Датасет — у любой сущности
# --------------------------------------------------------------------------- #
def dataset(
    conn: Any, syn: Synonyms, name: str, g: Graph | None = None, kb: str = ""
) -> dict[str, Any]:
    """Все факты о сущности и о том, что в неё входит: с кем как связана, где сказано."""
    g = g or graph(conn, syn, kb)
    names: list[str] = []
    for root in [name, *g.related(name)]:
        for part in g.parts(root):
            if part not in names:
                names.append(part)
    inside = set(names)
    rows: list[dict[str, Any]] = []
    docs: set[str] = set()
    seen: set[int] = set()
    for link in (lk for n in names for lk in g.touching.get(n, [])):
        if id(link) in seen:
            continue
        seen.add(id(link))
        if link.src in inside:
            about, direction, other = link.src, "→", link.dst
        elif link.dst in inside:
            about, direction, other = link.dst, "←", link.src
        else:
            continue
        if link.relation == PART_OF and link.src in inside and link.dst in inside:
            continue  # «зона входит в щит» — это состав, а не факт о нём
        docs |= {f["doc_id"] for f in link.facts}
        rows.append(
            {
                "about": about,
                "direction": direction,
                "relation": link.relation,
                "other": other,
                "documents": link.documents,
                "fragments": len(link.facts),
                "quotes": link.quotes(),
                "chunk_ids": [f["chunk_id"] for f in link.facts][:QUOTES_PER_LINK],
            }
        )
    rows.sort(key=lambda r: (-r["documents"], -r["fragments"], r["relation"], r["other"]))
    return {
        "name": name,
        "includes": names[1:],
        "documents": len(docs),
        "facts": rows,
        "in_dictionary": name in syn.groups,
    }


def datasets_response(conn: Any, syn: Synonyms, limit: int = 300, kb: str = "") -> dict[str, Any]:
    """Датасеты не хранятся: собираются из фактов при каждом запросе — после нового
    прохода модели появляются новые сущности и новые факты у старых.
    Числа те же, что в карточке датасета: строк-фактов и статей за ними."""
    g = graph(conn, syn, kb)
    rows = []
    for name in g.entities:
        d = dataset(conn, syn, name, g)
        if d["facts"] or d["includes"]:
            rows.append({"name": name, "facts": len(d["facts"]), "documents": d["documents"]})
    rows.sort(key=lambda r: (-r["documents"], -r["facts"], r["name"]))
    return {"datasets": rows[:limit], "total": len(rows), "pass": store.summary(conn, kb)}


def dataset_response(
    conn: Any, syn: Synonyms, payload: dict[str, Any], kb: str = ""
) -> dict[str, Any]:
    raw = str(payload.get("name") or payload.get("territory") or "").strip()
    g = graph(conn, syn, kb)
    known = [e["name"] for e in sorted(g.entities.values(), key=lambda e: -e["facts"])]
    if not raw:
        raise RequestError("не указано, чей датасет (name)", known=known)
    name = g.resolve(raw)
    if name is None:
        raise RequestError("такого в фактах из статей нет", raw, known)
    return dataset(conn, syn, name, g)


CSV_COLUMNS = [
    "о чём",
    "направление",
    "связь",
    "с чем",
    "статей",
    "фрагментов",
    "цитата",
    "статья",
    "ссылка",
]


def dataset_csv(data: dict[str, Any]) -> str:
    """Датасет таблицей: открывается в Excel (точка с запятой, UTF-8 с BOM)."""
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";", lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for r in data["facts"]:
        q = r["quotes"][0] if r["quotes"] else {}
        title = f"{q.get('title', '')}{', ' + str(q['year']) if q.get('year') else ''}"
        writer.writerow(
            [
                r["about"],
                r["direction"],
                r["relation"],
                r["other"],
                r["documents"],
                r["fragments"],
                q.get("quote", ""),
                title,
                q.get("url", ""),
            ]
        )
    return "﻿" + buf.getvalue()


# --------------------------------------------------------------------------- #
#  Отчёт: что нашла модель и где ошибалась
# --------------------------------------------------------------------------- #
def report(conn: Any, syn: Synonyms, top: int = 40, kb: str = "") -> dict[str, Any]:
    g = graph(conn, syn, kb)
    entities = sorted(g.entities.values(), key=lambda e: (-len(e["docs"]), -e["facts"], e["name"]))
    relations: Counter[str] = Counter()
    for link in g.links.values():
        relations[link.relation] += len(link.facts)
    return {
        "pass": store.summary(conn, kb),
        "entities": [
            (e["name"], len(e["docs"]), e["facts"], e["in_dictionary"], sorted(e["spellings"])[:4])
            for e in entities[:top]
        ],
        "entities_total": len(entities),
        "relations": relations.most_common(top),
        "rejected": store.rejected_reasons(conn, kb).most_common(12),
    }
