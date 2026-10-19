"""Поиск по словам с учётом языка фрагмента (П11) и для китайского и японского (Д21).

**Словарь Postgres — по языку фрагмента.** Раньше все фрагменты шли через русскую
конфигурацию: она снимает окончания у кириллицы, а латиницу отдаёт английскому
стеммеру. Немецкий, французский, испанский текст так стеммился по-английски («Lagerstätten»
не находилось по «Lagerstätte»). Теперь столбец tsv строится словарём языка фрагмента
(chunks.language, lang.py): CONFIGS ниже — какие есть в Postgres. Русский, английский
и неизвестный — по-прежнему русская конфигурация: она одна понимает оба языка
(в русской статье английские термины, в английской — русские названия). Языки без
своего стеммера в Postgres (украинский, польский, чешский…) и иероглифы — simple:
слово как есть, без окончаний.

Запрос строится теми словарями, какие есть у фрагментов базы (query_configs): по
каждому — свой to_tsquery, вместе через «или». Индекс GIN по tsv работает: словари в
запросе — постоянные.

**Китайский, японский, корейский.** Слова там не разделены пробелами, и полнотекстовый
поиск Postgres их не видит: весь отрезок иероглифов — одно «слово». pg_bigm (индекс по
парам знаков) в образе pgvector нет, поэтому — пары знаков запроса (биграммы): фрагмент
подходит, если в нём есть хотя бы половина пар, оценка — доля найденных. Ищется только
среди фрагментов этих языков (частичный индекс по языку), а длинные отрезки
(от трёх знаков) ускоряет pg_trgm — он в Postgres из коробки.
"""

from __future__ import annotations

import re
import time
from typing import Any

# Язык фрагмента (ISO 639-1) → словарь Postgres 17. Чего нет — DEFAULT_CONFIG, если
# письменность кириллица или латиница из «своих» языков, иначе simple.
CONFIGS = {
    "ru": "russian",
    "en": "russian",  # русская конфигурация стеммит латиницу по-английски — см. выше
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "sv": "swedish",
    "tr": "turkish",
    "sr": "serbian",
    "el": "greek",
    "hi": "hindi",
    "ar": "arabic",
}
DEFAULT_CONFIG = "russian"  # язык не понят: как было — русский с английским
CJK = ("zh", "ja", "ko")


SIMPLE = ("uk", "bg", "pl", "cs", *CJK, "he", "th", "fa")  # своего стеммера в Postgres нет
CONFIG_VERSION = 1
CONFIG_FUNCTION = f"kbrag_ts_config_v{CONFIG_VERSION}"


def config_sql() -> str:
    """Функция kbrag_ts_config_vN(язык) для столбца tsv. IMMUTABLE — иначе Postgres не
    даст построить по ней вычисляемый столбец; поменялся CONFIGS — новая версия
    (CONFIG_VERSION), и столбец перестраивается сам (migrate_sql)."""
    simple = ", ".join(f"'{lang}'" for lang in SIMPLE)
    cases = "\n".join(
        f"        WHEN lang = '{lang}' THEN '{cfg}'::regconfig" for lang, cfg in CONFIGS.items()
    )
    return f"""CREATE OR REPLACE FUNCTION {CONFIG_FUNCTION}(lang text) RETURNS regconfig
LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $f$
    SELECT CASE
        WHEN lang IN ({simple}) THEN 'simple'::regconfig
{cases}
        ELSE '{DEFAULT_CONFIG}'::regconfig
    END
$f$;"""


def migrate_sql() -> str:
    """Столбец tsv — по словарю языка фрагмента. Был построен иначе (русским для всех,
    прежней версией функции) — пересоздаётся вместе с индексом: Postgres перепишет
    таблицу один раз, на больших базах это минуты."""
    return f"""
{config_sql()}
DO $m$
DECLARE expr text;
BEGIN
    SELECT pg_get_expr(d.adbin, d.adrelid) INTO expr
    FROM pg_attribute a JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
    WHERE a.attrelid = 'chunks'::regclass AND a.attname = 'tsv';
    IF expr IS NULL OR position('{CONFIG_FUNCTION}' IN expr) = 0 THEN
        ALTER TABLE chunks DROP COLUMN IF EXISTS tsv;
        ALTER TABLE chunks ADD COLUMN tsv tsvector GENERATED ALWAYS AS
            (to_tsvector({CONFIG_FUNCTION}(language), text)) STORED;
        CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
    END IF;
END $m$;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS chunks_cjk_idx ON chunks (doc_id) WHERE language IN ('zh', 'ja', 'ko');
CREATE INDEX IF NOT EXISTS chunks_cjk_trgm_idx ON chunks USING gin (text gin_trgm_ops)
    WHERE language IN ('zh', 'ja', 'ko');
"""


def config_of(language: str | None) -> str:
    """Словарь Postgres для языка — так же, как kbrag_ts_config в базе."""
    lang = (language or "").strip()
    if lang in SIMPLE:
        return "simple"
    return CONFIGS.get(lang, DEFAULT_CONFIG)


# --------------------------------------------------------------------------- #
#  Словари запроса — какие есть у фрагментов базы
# --------------------------------------------------------------------------- #
CACHE_SEC = 300.0  # языки базы меняет только загрузка статей
MAX_CONFIGS = 6  # больше словарей в одном запросе — медленнее, а пользы нет
_CACHE: dict[tuple[str, Any], tuple[float, list[str]]] = {}  # (база, доступ)


def query_configs(conn: Any, kb: str = "") -> list[str]:
    """Словари для запроса: русский (всегда — ru, en и неизвестный) и словари языков
    фрагментов поддерева kb, самые частые сначала. Запомнены на CACHE_SEC."""
    from .db import in_kb

    now = time.monotonic()
    from .. import acl

    key = acl.cache_key(kb)
    cached = _CACHE.get(key)
    if cached and now - cached[0] < CACHE_SEC:
        return cached[1]
    clause, params = in_kb(kb, "c.doc_id")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT c.language, count(*) FROM chunks c "
                f"WHERE c.language IS NOT NULL{clause} GROUP BY c.language ORDER BY 2 DESC",
                params,
            )
            rows = cur.fetchall()
    except Exception:  # noqa: BLE001 — нет столбца (старая база): как раньше
        rows = []
    found = [DEFAULT_CONFIG]
    for language, _count in rows:
        config = config_of(language)
        if config not in found:
            found.append(config)
    found = found[:MAX_CONFIGS]
    _CACHE[key] = (now, found)
    return found


def tsquery_sql(configs: list[str], web: bool = False) -> str:
    """Выражение запроса: по каждому словарю — свой tsquery, вместе через «или»."""
    function = "websearch_to_tsquery" if web else "to_tsquery"
    parts = [f"{function}('{c}', %(q)s)" for c in configs if re.fullmatch(r"[a-z_]+", c)]
    if not parts:
        parts = [f"{function}('{DEFAULT_CONFIG}', %(q)s)"]
    return parts[0] if len(parts) == 1 else "(" + " || ".join(parts) + ")"


# --------------------------------------------------------------------------- #
#  Китайский, японский, корейский: пары знаков
# --------------------------------------------------------------------------- #
_CJK_RUN = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]+")
MAX_GRAMS = 24
MIN_SHARE = 0.5  # хотя бы половина пар запроса — во фрагменте


def cjk_grams(query: str) -> list[str]:
    """Пары соседних знаков из отрезков иероглифов (каны, хангыля) запроса; отрезок из
    одного знака — сам знак."""
    grams: list[str] = []
    for run in _CJK_RUN.findall(query or ""):
        pairs = [run] if len(run) == 1 else [run[i : i + 2] for i in range(len(run) - 1)]
        for gram in pairs:
            if gram not in grams:
                grams.append(gram)
    return grams[:MAX_GRAMS]


def has_cjk(query: str) -> bool:
    return bool(_CJK_RUN.search(query or ""))


def cjk_search(
    conn: Any, query: str, limit: int, clause: str, params: dict[str, Any], fields: str
) -> list[tuple[Any, ...]]:
    """Фрагменты на китайском, японском, корейском с парами знаков запроса. Строка —
    поля fields и доля найденных пар; clause и params — условия на статью (search.py)."""
    grams = cjk_grams(query)
    if not grams:
        return []
    need = max(1, round(MIN_SHARE * len(grams)))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT * FROM (
                SELECT {fields},
                       (SELECT count(*) FROM unnest(%(grams)s::text[]) g
                        WHERE strpos(c.text, g) > 0)::float / %(total)s AS score
                FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.language IN ('zh', 'ja', 'ko') {clause}
            ) found
            WHERE score * %(total)s >= %(need)s
            ORDER BY score DESC
            LIMIT %(limit)s
            """,
            {**params, "grams": grams, "total": len(grams), "need": need, "limit": limit},
        )
        rows: list[tuple[Any, ...]] = cur.fetchall()
    return rows
