"""База: схема, подключение, запись документов и чанков.

Одна таблица на документы, одна на чанки. У чанка три способа быть найденным:

* embedding — вектор BGE-M3, косинусная близость, индекс HNSW;
* tsv — полнотекстовый поиск Postgres, индекс GIN. Он ловит то, что вектор
  пропускает: точные названия площадей, номера ГОСТов, фамилии;
* поля документа — год, источник, язык: по ним отсекают лишнее до поиска.

Вектор передаётся строкой с приведением `%s::vector`. Так не нужен бинарный
адаптер, который на Windows любит не совпасть версией с psycopg.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..machine import integer

VECTOR_DIM = 1024  # BGE-M3

# Порт 5433 — тот, что публикует docker-compose.yml, чтобы не спорить
# с уже установленным на машине PostgreSQL на 5432.
DEFAULT_DSN = "postgresql://kbrag:kbrag@localhost:5433/kbrag"

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id      text PRIMARY KEY,
    title       text NOT NULL DEFAULT '',
    authors     text[] NOT NULL DEFAULT '{}',
    year        int,
    journal     text,
    doi         text,
    url         text NOT NULL DEFAULT '',
    source      text NOT NULL DEFAULT '',
    language    text,
    status      text NOT NULL DEFAULT '',
    sha256      text NOT NULL DEFAULT '',
    pages       int NOT NULL DEFAULT 0,
    parser      text NOT NULL DEFAULT '',
    accuracy    real NOT NULL DEFAULT 0,
    fetched_at  text NOT NULL DEFAULT '',
    added_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    doc_id      text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ord         int NOT NULL,
    text        text NOT NULL,
    headings    text[] NOT NULL DEFAULT '{}',
    pages       int[] NOT NULL DEFAULT '{}',
    n_tokens    int NOT NULL DEFAULT 0,
    has_table   boolean NOT NULL DEFAULT false,
    embedding   vector(%(dim)s),
    -- tsv (полнотекстовый поиск) — словарём языка фрагмента: столбец добавляет
    -- index/textlang.py (migrate_sql) после столбца language.
    UNIQUE (doc_id, ord)
);

CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);

-- Формулы статей (index/formulas.py): из разметки docling при загрузке. idx — по порядку
-- в статье, number — её «(N)»; chunk_id — фрагмент, где она стоит; description — она
-- словами с контекстом, по нему вектор (поиск формулы по смыслу).
CREATE TABLE IF NOT EXISTS formulas (
    id          bigserial PRIMARY KEY,
    doc_id      text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    idx         int  NOT NULL,
    number      text,
    latex       text NOT NULL,
    pdf_text    text NOT NULL DEFAULT '',
    page        int,
    bbox        jsonb,
    before      text NOT NULL DEFAULT '',
    after       text NOT NULL DEFAULT '',
    description text NOT NULL DEFAULT '',
    chunk_id    bigint REFERENCES chunks(id) ON DELETE SET NULL,
    embedding   vector(%(dim)s),
    UNIQUE (doc_id, idx)
);
CREATE INDEX IF NOT EXISTS formulas_number_idx ON formulas (doc_id, number);

-- Ссылки на формулы из текста: «из формулы (55)», «уравнение 3», «(2)».
CREATE TABLE IF NOT EXISTS formula_refs (
    chunk_id    bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    formula_id  bigint NOT NULL REFERENCES formulas(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, formula_id)
);

-- Рисунки статей (index/figures.py): из перечня рисунков записи статьи. n — по порядку
-- в статье, label — номер в её подписи («Рис. 3» → 3); file — картинка от папки статей;
-- chunk_id — фрагмент, где рисунок стоит (описание моделью или подпись); вектор — по
-- подписи и описанию.
CREATE TABLE IF NOT EXISTS figures (
    id          bigserial PRIMARY KEY,
    doc_id      text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    n           int  NOT NULL,
    label       text,
    file        text NOT NULL DEFAULT '',
    page        int,
    bbox        jsonb,
    caption     text NOT NULL DEFAULT '',
    description text NOT NULL DEFAULT '',
    width       int,
    height      int,
    chunk_id    bigint REFERENCES chunks(id) ON DELETE SET NULL,
    embedding   vector(%(dim)s),
    UNIQUE (doc_id, n)
);
CREATE INDEX IF NOT EXISTS figures_label_idx ON figures (doc_id, label);

-- Ссылки на рисунки из текста: «рис. 3», «на рисунке 2а», «Fig. 4», и подпись рисунка.
CREATE TABLE IF NOT EXISTS figure_refs (
    chunk_id    bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    figure_id   bigint NOT NULL REFERENCES figures(id) ON DELETE CASCADE,
    PRIMARY KEY (chunk_id, figure_id)
);

-- Резюме статей и разделов (index/summaries.py): пишет модель при загрузке или
-- командой summaries. kind — article (статья целиком, section пустой) или section;
-- ords — фрагменты раздела; source_hash — отпечаток текста, по которому написано.
CREATE TABLE IF NOT EXISTS summaries (
    id          bigserial PRIMARY KEY,
    doc_id      text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    kind        text NOT NULL CHECK (kind IN ('article', 'section')),
    section     text NOT NULL DEFAULT '',
    ords        int[] NOT NULL DEFAULT '{}',
    text        text NOT NULL,
    source_hash text NOT NULL DEFAULT '',
    model       text,
    embedding   vector(%(dim)s),
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_id, kind, section)
);

-- Язык фрагмента (lang.py, код ISO 639-1): определяется при загрузке; не понять —
-- язык статьи. Таблица могла быть создана раньше — столбец добавляется к ней.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS language text;

-- Дерево баз знаний (kb.py): узел — путь ltree, geology.ore.anabar. Статья лежит в
-- одной или нескольких ветках; фрагменты и факты у неё одни на все ветки. Поиск и
-- граф по узлу берут статьи его поддерева: path <@ 'geology.ore'.
CREATE EXTENSION IF NOT EXISTS ltree;

CREATE TABLE IF NOT EXISTS kbs (
    path        ltree PRIMARY KEY,
    title       text NOT NULL DEFAULT '',
    added_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS kbs_path_idx ON kbs USING gist (path);

CREATE TABLE IF NOT EXISTS document_kbs (
    doc_id      text  NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    path        ltree NOT NULL REFERENCES kbs(path) ON UPDATE CASCADE ON DELETE CASCADE,
    PRIMARY KEY (doc_id, path)
);
CREATE INDEX IF NOT EXISTS document_kbs_path_idx ON document_kbs USING gist (path);
"""

# HNSW строится отдельно: на пустой таблице он бесполезен, а на больших данных
# создаётся долго. Поэтому — отдельным шагом и только если его ещё нет.
HNSW = """
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""


def dsn_from_env(default: str = DEFAULT_DSN) -> str:
    return os.environ.get("KBRAG_DSN") or default


def vector_literal(values: Any) -> str:
    """Вектор в том виде, в каком его понимает pgvector: [0.1,0.2,...]."""
    return "[" + ",".join(f"{float(v):.7g}" for v in values) + "]"


# Пул соединений — один на процесс и адрес базы: страница, Телеграм-бот и прочие входы
# в одном процессе (web --tg) берут соединения из него, а не открывают новое на каждый
# запрос (около 30 мс). Размер — KBRAG_DB_POOL в .local/settings.env; 1 — без пула.
POOL_SIZE = integer("KBRAG_DB_POOL", 10)
POOL_WAIT = 30  # с: сколько ждать свободного соединения, прежде чем сказать об ошибке
_POOLS: dict[str, Any] = {}
_POOLS_LOCK = threading.Lock()


def _reset(conn: Any) -> None:
    """Соединение вернулось в пул — к обычному виду: чат-бот включает autocommit."""
    conn.autocommit = False


def pool(dsn: str) -> Any:
    """Пул для этого адреса; нет psycopg_pool или пул выключен — None."""
    if POOL_SIZE <= 1:
        return None
    try:
        from psycopg_pool import ConnectionPool
    except ImportError:
        return None
    with _POOLS_LOCK:
        found = _POOLS.get(dsn)
        if found is None:
            found = ConnectionPool(
                dsn,
                min_size=1,
                max_size=POOL_SIZE,
                timeout=POOL_WAIT,
                reset=_reset,
                name="kbrag",
                open=True,
            )
            _POOLS[dsn] = found
        return found


@contextmanager
def connect(dsn: str | None = None) -> Iterator[Any]:
    """Соединение с базой: без ошибки — изменения сохраняются, с ошибкой — откатываются.
    Из пула, если он есть (pool), иначе — новое на время блока."""
    dsn = dsn or dsn_from_env()
    shared = pool(dsn)
    if shared is not None:
        with shared.connection() as conn:  # сохраняет или откатывает сам, как и ниже
            yield conn
        return
    import psycopg

    conn = psycopg.connect(dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Запасные соединения к той же базе — для поиска в несколько потоков (search.search_many).
# Новое соединение — около 30 мс, а поиск по одному запросу — 20: открывать на каждый
# поиск невыгодно, поэтому отработавшие соединения ждут следующего вопроса здесь.
_SPARE: dict[str, list[Any]] = {}
_SPARE_LOCK = threading.Lock()
SPARE_MAX = 8  # сколько свободных соединений к одной базе держать


def is_real(conn: Any) -> bool:
    """Настоящее соединение psycopg (а не заглушка в проверках)."""
    return type(conn).__module__.startswith("psycopg") and hasattr(conn, "pgconn")


@contextmanager
def sibling(conn: Any) -> Iterator[Any]:
    """Ещё одно соединение к той же базе, что и conn, — из запаса или новое.

    Пароля в conn.info.dsn нет — он берётся у самого соединения. После работы
    соединение возвращается в запас; сломанное (ошибка внутри) — закрывается."""
    import psycopg

    key = conn.info.dsn
    with _SPARE_LOCK:
        spare = _SPARE.get(key) or []
        other = spare.pop() if spare else None
    if other is None or other.closed:
        password = conn.pgconn.password
        other = psycopg.connect(
            key, password=password.decode() if password else None, autocommit=True
        )
    try:
        yield other
    except Exception:
        other.close()
        raise
    with _SPARE_LOCK:
        spare = _SPARE.setdefault(key, [])
        if len(spare) < SPARE_MAX and not other.closed:
            spare.append(other)
            return
    other.close()


def autocommit(conn: Any) -> None:
    """Без открытой транзакции: для соединения, которое живёт, пока модель пишет
    ответ (минуты). Векторный поиск всё равно идёт в своей транзакции (search.py)."""
    try:
        conn.autocommit = True
    except AttributeError:  # заглушка в проверках
        pass


def init_db(conn: Any, dim: int = VECTOR_DIM, with_hnsw: bool = True) -> None:
    from .textlang import migrate_sql

    with conn.cursor() as cur:
        cur.execute(SCHEMA % {"dim": dim})
        cur.execute(migrate_sql())  # tsv по языку фрагмента; поиск пар знаков (CJK)
        if with_hnsw:
            cur.execute(HNSW)
    conn.commit()


# --------------------------------------------------------------------------- #
#  Дерево баз: в каких ветках лежит статья
# --------------------------------------------------------------------------- #
def in_kb(kb: str, doc: str = "d.doc_id") -> tuple[str, dict[str, Any]]:
    """Условие «статья doc — в поддереве узла kb и видна тому, кто спрашивает» для WHERE
    (с AND впереди) и его параметры. Пустой путь — корень: всё дерево, и статьи, ещё не
    разложенные по веткам, — тоже. Кому что видно — acl.py: у человека с ограниченным
    доступом к условию добавляются его ветки."""
    from .. import acl

    rights, rights_params = acl.sql(doc)
    if not kb:
        return rights, rights_params
    return (
        f" AND EXISTS (SELECT 1 FROM document_kbs dk WHERE dk.doc_id = {doc}"
        " AND dk.path <@ %(kb)s::ltree)" + rights,
        {"kb": kb, **rights_params},
    )


def visible_doc(conn: Any, doc_id: str) -> bool:
    """Видна ли статья тому, кто спрашивает (acl.py): для адресов по doc_id и chunk_id."""
    from .. import acl

    if acl.current() is None:
        return True
    where, params = in_kb("", "d.doc_id")
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT 1 FROM documents d WHERE d.doc_id = %(doc)s{where}", {**params, "doc": doc_id}
        )
        return cur.fetchone() is not None


def doc_of_chunk(conn: Any, chunk_id: int) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id FROM chunks WHERE id = %s", (chunk_id,))
        row = cur.fetchone()
    return str(row[0]) if row else None


def ensure_kb(conn: Any, path: str, title: str = "") -> None:
    """Узел и все его предки — в таблице kbs (связь статьи с веткой требует узла)."""
    from ..kb import ancestors

    with conn.cursor() as cur:
        for node in ancestors(path):
            cur.execute(
                "INSERT INTO kbs (path, title) VALUES (%s::ltree, %s) ON CONFLICT (path) DO "
                "UPDATE SET title = CASE WHEN EXCLUDED.title <> '' THEN EXCLUDED.title "
                "ELSE kbs.title END",
                (node, title if node == path else ""),
            )


def assign(conn: Any, doc_ids: list[str], path: str) -> int:
    """Положить статьи в ветку (в других ветках они остаются). Сколько добавлено."""
    if not path or not doc_ids:
        return 0
    ensure_kb(conn, path)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO document_kbs (doc_id, path) "
            "SELECT d.doc_id, %s::ltree FROM documents d WHERE d.doc_id = ANY(%s) "
            "ON CONFLICT DO NOTHING",
            (path, list(doc_ids)),
        )
        return max(cur.rowcount or 0, 0)


def unassign(conn: Any, doc_ids: list[str], path: str) -> int:
    """Убрать статьи из ветки (сами статьи и другие их ветки остаются). Сколько убрано."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM document_kbs WHERE path = %s::ltree AND doc_id = ANY(%s)",
            (path, list(doc_ids)),
        )
        return max(cur.rowcount or 0, 0)


def kb_nodes(conn: Any) -> list[dict[str, Any]]:
    """Узлы дерева баз со счётом статей: своих (лежат в самом узле) и всего в поддереве.
    Счёт — только видимых тому, кто спрашивает (acl.py): чужие статьи не видны и числом."""
    from .. import acl

    rights, params = acl.sql("dk.doc_id")
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT k.path::text, k.title,
                   (SELECT count(*) FROM document_kbs dk WHERE dk.path = k.path{rights}),
                   (SELECT count(DISTINCT dk.doc_id) FROM document_kbs dk
                     WHERE dk.path <@ k.path{rights})
            FROM kbs k ORDER BY k.path
            """,
            params or None,
        )
        return [
            {"path": r[0], "title": r[1] or "", "own": int(r[2]), "subtree": int(r[3])}
            for r in cur.fetchall()
        ]


def unplaced(conn: Any) -> int:
    """Сколько статей не лежит ни в одной ветке (их видит только всё дерево — и только
    тот, у кого доступ ко всему)."""
    from .. import acl

    if acl.current() is not None:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM documents d WHERE NOT EXISTS "
            "(SELECT 1 FROM document_kbs dk WHERE dk.doc_id = d.doc_id)"
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def move_kb(conn: Any, old: str, new: str) -> int:
    """Перенести узел со всем поддеревом: old.x → new.x. Статьи переезжают вместе с
    ним (ссылки на узел — ON UPDATE CASCADE). Сколько узлов перенесено."""
    if not old or not new:
        raise ValueError("корень не переносится и корнем не становится")
    if new == old or new.startswith(old + "."):
        raise ValueError(f"«{new}» внутри «{old}» — узел нельзя перенести в самого себя")
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM kbs WHERE path <@ %s::ltree", (new,))
        busy = cur.fetchone()
        if busy and busy[0]:
            raise ValueError(f"узел «{new}» уже есть — сначала уберите его или выберите другой")
    parent = new.rsplit(".", 1)[0] if "." in new else ""
    if parent:
        ensure_kb(conn, parent)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE kbs SET path = CASE WHEN path = %(old)s::ltree THEN %(new)s::ltree
                ELSE %(new)s::ltree || subpath(path, nlevel(%(old)s::ltree)) END
            WHERE path <@ %(old)s::ltree
            """,
            {"old": old, "new": new},
        )
        return max(cur.rowcount or 0, 0)


def remove_kb(conn: Any, path: str, drop_orphans: bool = False) -> dict[str, int]:
    """Убрать узел с поддеревом. Статьи остаются в других своих ветках; лежавшие только
    здесь — без ветки (их видит всё дерево), а с drop_orphans — удаляются из базы
    вместе с фрагментами и фактами."""
    if not path:
        raise ValueError("корень не убирается")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT dk.doc_id FROM document_kbs dk WHERE dk.path <@ %(p)s::ltree
              AND NOT EXISTS (SELECT 1 FROM document_kbs o WHERE o.doc_id = dk.doc_id
                              AND NOT o.path <@ %(p)s::ltree)
            """,
            {"p": path},
        )
        orphans = [r[0] for r in cur.fetchall()]
        cur.execute("DELETE FROM kbs WHERE path <@ %s::ltree", (path,))
        nodes = max(cur.rowcount or 0, 0)
    dropped = delete_documents(conn, orphans) if drop_orphans and orphans else 0
    return {"nodes": nodes, "orphans": len(orphans), "deleted": dropped}


def assign_unplaced(conn: Any, path: str) -> int:
    """Все статьи, ещё не лежащие ни в одной ветке, — в ветку path. Сколько."""
    if not path:
        return 0
    ensure_kb(conn, path)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO document_kbs (doc_id, path) SELECT d.doc_id, %s::ltree FROM documents d "
            "WHERE NOT EXISTS (SELECT 1 FROM document_kbs dk WHERE dk.doc_id = d.doc_id)",
            (path,),
        )
        return max(cur.rowcount or 0, 0)


def upsert_document(conn: Any, record: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (doc_id, title, authors, year, journal, doi, url,
                                   source, language, status, sha256, pages, parser,
                                   accuracy, fetched_at)
            VALUES (%(doc_id)s, %(title)s, %(authors)s, %(year)s, %(journal)s, %(doi)s,
                    %(url)s, %(source)s, %(language)s, %(status)s, %(sha256)s, %(pages)s,
                    %(parser)s, %(accuracy)s, %(fetched_at)s)
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title, authors = EXCLUDED.authors, year = EXCLUDED.year,
                journal = EXCLUDED.journal, doi = EXCLUDED.doi, url = EXCLUDED.url,
                source = EXCLUDED.source, language = EXCLUDED.language,
                status = EXCLUDED.status, sha256 = EXCLUDED.sha256, pages = EXCLUDED.pages,
                parser = EXCLUDED.parser, accuracy = EXCLUDED.accuracy,
                fetched_at = EXCLUDED.fetched_at
            """,
            record,
        )


def chunk_print(text: str) -> str:
    """Отпечаток текста фрагмента: пометка формул $$…$$ (index/formulas.py) и пробелы —
    не изменение текста."""
    return " ".join((text or "").replace("$$", "").split())


def match_chunks(
    old: list[tuple[int, int, str]], rows: list[dict[str, Any]]
) -> list[tuple[int | None, bool]]:
    """Новые фрагменты статьи → прежние (id, ord, текст): на каждый новый — id прежнего
    фрагмента, чью строку он займёт, и тот ли у него текст.

    Сначала — по тексту: фрагмент с тем же текстом остаётся тем же фрагментом, куда бы
    он ни сдвинулся (новый абзац в начале статьи сдвигает номера всех остальных). Из
    одинаковых текстов — ближайший по номеру. Не нашёлся по тексту — прежний фрагмент на
    том же месте, если его не забрал никто (текст поправлен: id тот же, факты — заново).
    Иначе — новый фрагмент."""
    by_print: dict[str, list[tuple[int, int]]] = {}
    by_ord: dict[int, int] = {}
    for chunk_id, ord_, text in old:
        by_print.setdefault(chunk_print(text), []).append((ord_, chunk_id))
        by_ord[ord_] = chunk_id
    used: set[int] = set()
    plan: list[tuple[int | None, bool]] = [(None, False)] * len(rows)
    for i, row in enumerate(rows):
        free = [(o, c) for o, c in by_print.get(chunk_print(row["text"]), []) if c not in used]
        if free:
            _, chunk_id = min(free, key=lambda oc: (abs(oc[0] - int(row["ord"])), oc[0]))
            used.add(chunk_id)
            plan[i] = (chunk_id, True)
    for i, row in enumerate(rows):
        same_place = by_ord.get(int(row["ord"]))
        if plan[i][0] is None and same_place is not None and same_place not in used:
            used.add(same_place)
            plan[i] = (same_place, False)
    return plan


def replace_chunks(conn: Any, doc_id: str, rows: list[dict[str, Any]]) -> int:
    """Чанки документа пишутся целиком: переиндексация не должна плодить дубли.

    Прежние строки обновляются на месте, а не удаляются и вставляются заново: так у
    фрагмента остаётся прежний id, и факты графа, которые на него ссылаются, не
    пропадают каскадом. Какой новый фрагмент займёт какую строку — по тексту
    (match_chunks): переразбор (новый docling, OCR, другая нарезка) сдвигает номера, и
    сравнение по номеру стирало факты всей статьи. Факты стираются только у фрагментов,
    чей текст правда поменялся, — их модель пройдёт заново сама; фрагменты, которых
    больше нет, уходят вместе со своими фактами.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id, ord, text FROM chunks WHERE doc_id = %s", (doc_id,))
        old = [(int(r[0]), int(r[1]), str(r[2] or "")) for r in cur.fetchall() or []]
        plan = match_chunks(old, rows)
        if old:
            # Номера освобождаются: (doc_id, ord) уникален, а фрагменты меняются местами.
            cur.execute("UPDATE chunks SET ord = -1 - ord WHERE doc_id = %s", (doc_id,))
        changed = []
        for row, (chunk_id, same) in zip(rows, plan, strict=True):
            values = {"language": None, **row}
            if chunk_id is None:
                cur.execute(
                    """
                    INSERT INTO chunks (doc_id, ord, text, headings, pages, n_tokens,
                                        has_table, embedding, language)
                    VALUES (%(doc_id)s, %(ord)s, %(text)s, %(headings)s, %(pages)s,
                            %(n_tokens)s, %(has_table)s, %(embedding)s::vector, %(language)s)
                    """,
                    values,
                )
                continue
            cur.execute(
                """
                UPDATE chunks SET ord = %(ord)s, text = %(text)s, headings = %(headings)s,
                    pages = %(pages)s, n_tokens = %(n_tokens)s, has_table = %(has_table)s,
                    embedding = %(embedding)s::vector, language = %(language)s
                WHERE id = %(chunk_id)s
                """,
                {**values, "chunk_id": chunk_id},
            )
            if not same:
                changed.append(chunk_id)
        kept = {chunk_id for chunk_id, _ in plan if chunk_id is not None}
        gone = [chunk_id for chunk_id, _, _ in old if chunk_id not in kept]
        if gone:
            cur.execute("DELETE FROM chunks WHERE id = ANY(%s)", (gone,))
        if changed:
            cur.execute("SELECT to_regclass('facts') IS NOT NULL")
            found = cur.fetchone()
            if found and found[0]:
                cur.execute("DELETE FROM facts WHERE chunk_id = ANY(%s)", (changed,))
                cur.execute("DELETE FROM facts_pass WHERE chunk_id = ANY(%s)", (changed,))
    return len(rows)


def fill_languages(conn: Any, batch: int = 2000) -> int:
    """Язык фрагментов, у которых его ещё нет (загружены до lang.py), и язык статей без
    него — по большинству фрагментов. Сколько фрагментов дописано."""
    from ..lang import detect, majority

    done = 0
    with conn.cursor() as cur:
        while True:
            cur.execute(
                "SELECT c.id, c.text, coalesce(d.language, '') FROM chunks c "
                "JOIN documents d ON d.doc_id = c.doc_id WHERE c.language IS NULL LIMIT %s",
                (batch,),
            )
            rows = cur.fetchall()
            if not rows:
                break
            for chunk_id, text, fallback in rows:
                # '' — «не понять», чтобы не перебирать фрагмент снова; поиск возьмёт
                # тогда язык статьи.
                cur.execute(
                    "UPDATE chunks SET language = %s WHERE id = %s",
                    (detect(text) or fallback or "", chunk_id),
                )
            done += len(rows)
        cur.execute(
            "SELECT d.doc_id, array_agg(c.language) FROM documents d JOIN chunks c ON "
            "c.doc_id = d.doc_id WHERE coalesce(d.language, '') = '' GROUP BY d.doc_id"
        )
        for doc_id, languages in cur.fetchall():
            found = majority(list(languages or []))
            if found:
                cur.execute("UPDATE documents SET language = %s WHERE doc_id = %s", (found, doc_id))
    return done


def documents_with_titles(conn: Any) -> list[tuple[str, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, coalesce(title, '') FROM documents ORDER BY doc_id")
        return [(str(r[0]), str(r[1])) for r in cur.fetchall()]


def chunk_rows(conn: Any, doc_id: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, ord, text, headings FROM chunks WHERE doc_id = %s ORDER BY ord",
            (doc_id,),
        )
        return [
            {"id": r[0], "ord": r[1], "text": r[2], "headings": list(r[3] or [])}
            for r in cur.fetchall()
        ]


def facts_of_chunk(conn: Any, chunk_id: int) -> list[tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('facts') IS NOT NULL")
        found = cur.fetchone()
        if not (found and found[0]):
            return []
        cur.execute("SELECT id, quote FROM facts WHERE chunk_id = %s", (chunk_id,))
        return [(int(r[0]), str(r[1])) for r in cur.fetchall()]


def tidy_chunks(
    conn: Any,
    drop_ids: list[int],
    updates: list[tuple[int, str, str]],
    stale_facts: list[int],
) -> None:
    """Убрать фрагменты (их факты уходят каскадом), переписать подчищенные."""
    with conn.cursor() as cur:
        if drop_ids:
            cur.execute("DELETE FROM chunks WHERE id = ANY(%s)", (drop_ids,))
        for chunk_id, text, vector in updates:
            cur.execute(
                "UPDATE chunks SET text = %s, embedding = %s::vector WHERE id = %s",
                (text, vector, chunk_id),
            )
        if stale_facts:
            cur.execute("DELETE FROM facts WHERE id = ANY(%s)", (stale_facts,))


def indexed_docs(conn: Any) -> dict[str, int]:
    """Что уже в базе: doc_id → сколько чанков. По этому решаем, что пропустить."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT d.doc_id, count(c.id)
            FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id
            GROUP BY d.doc_id
            """)
        return {row[0]: int(row[1]) for row in cur.fetchall()}


def delete_documents(conn: Any, doc_ids: list[str]) -> int:
    """Забыть статьи: с ними уходят фрагменты, разметка и связи (каскадом)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE doc_id = ANY(%s)", (list(doc_ids),))
        return max(cur.rowcount or 0, 0)


def document_hashes(conn: Any) -> dict[str, str]:
    """Отпечаток файла → статья. Одна и та же статья приходит из разных
    источников под разными идентификаторами (у одного есть DOI, у другого нет),
    а файл у неё один и тот же."""
    with conn.cursor() as cur:
        cur.execute("SELECT sha256, doc_id FROM documents WHERE sha256 <> ''")
        return {row[0]: row[1] for row in cur.fetchall()}


def stats(conn: Any, kb: str = "") -> dict[str, Any]:
    """Сколько статей и фрагментов в поддереве узла kb (пустой — во всей базе)."""
    where, params = in_kb(kb)
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM documents d WHERE true{where}", params)
        docs = cur.fetchone()[0]
        cur.execute(
            "SELECT count(*), count(c.embedding) FROM chunks c "
            f"JOIN documents d ON d.doc_id = c.doc_id WHERE true{where}",
            params,
        )
        chunks, embedded = cur.fetchone()
        cur.execute(
            f"SELECT min(year), max(year) FROM documents d WHERE year IS NOT NULL{where}", params
        )
        lo, hi = cur.fetchone()
    return {
        "documents": docs,
        "chunks": chunks,
        "embedded": embedded,
        "years": (lo, hi),
    }


def documents(conn: Any, limit: int = 500, kb: str = "") -> list[dict[str, Any]]:
    """Список статей поддерева kb с числом фрагментов — для обзора того, что разобрано."""
    where, params = in_kb(kb)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT d.doc_id, d.title, d.year, d.journal, d.url, d.authors,
                   d.source, d.parser, d.pages, count(c.id) AS chunks
            FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id
            WHERE true{where}
            GROUP BY d.doc_id
            ORDER BY d.year DESC NULLS LAST, d.title
            LIMIT %(limit)s
            """,
            {**params, "limit": limit},
        )
        return [
            {
                "doc_id": row[0],
                "title": row[1],
                "year": row[2],
                "journal": row[3],
                "url": row[4],
                "authors": list(row[5] or []),
                "source": row[6],
                "parser": row[7],
                "pages": row[8],
                "chunks": int(row[9]),
            }
            for row in cur.fetchall()
        ]


def document(conn: Any, doc_id: str) -> dict[str, Any] | None:
    """Статья в базе: метаданные, сколько фрагментов и в каких ветках дерева; нет — None."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT d.doc_id, d.title, d.authors, d.year, d.journal, d.doi, d.url, d.language,
                   d.pages, d.parser, d.sha256,
                   (SELECT count(*) FROM chunks c WHERE c.doc_id = d.doc_id),
                   (SELECT coalesce(array_agg(dk.path::text ORDER BY dk.path), '{}')
                      FROM document_kbs dk WHERE dk.doc_id = d.doc_id)
            FROM documents d WHERE d.doc_id = %s
            """,
            (doc_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    keys = ("doc_id", "title", "authors", "year", "journal", "doi", "url", "language")
    out = dict(zip(keys, row[:8], strict=True))
    out.update(
        authors=list(row[2] or []),
        pages=row[8],
        parser=row[9],
        sha256=row[10],
        chunks=int(row[11]),
        kbs=list(row[12] or []),
    )
    return out


def chunk_count(conn: Any, doc_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE doc_id = %s", (doc_id,))
        row = cur.fetchone()
    return int(row[0]) if row else 0


def document_chunks(
    conn: Any, doc_id: str, offset: int = 0, limit: int | None = None
) -> list[dict[str, Any]]:
    """Разобранная статья по порядку — то, что получилось из PDF; offset и limit — часть
    фрагментов (статья по страницам: /api/article, MCP get_article)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.ord, c.text, c.headings, c.pages, c.n_tokens, c.has_table
            FROM chunks c WHERE c.doc_id = %s ORDER BY c.ord
            OFFSET %s LIMIT %s
            """,
            (doc_id, max(0, offset), limit),
        )
        return [
            {
                "ord": row[0],
                "text": row[1],
                "headings": list(row[2] or []),
                "pages": list(row[3] or []),
                "n_tokens": row[4],
                "has_table": row[5],
            }
            for row in cur.fetchall()
        ]
