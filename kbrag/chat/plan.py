"""Разбор вопроса: части, поисковые запросы, предмет вопроса, «собрать всё».

Разбирает модель (plan_question). Не ответила — вопрос одной частью, а «собрать
всё» и вопросы о связях узнаются по словам. Вопрос вдогонку («а по космоснимкам?»)
модель переписывает так, чтобы он был понятен без разговора."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..kb import get as get_kb
from ..kb import languages
from .settings import Settings

# Слова в {фигурных скобках} без значения в коде — из раздела «тема» настроек базы
# (kb.py); узел может заменить шаблон целиком: «разбор.система» и «разбор».
PLAN_SYSTEM = (
    "Ты разбираешь вопрос {пользователь} и составляешь поисковые запросы к базе научных "
    "статей по {область}. Отвечай только JSON, без пояснений."
)

PLAN_USER = """Вопрос пользователя: «{question}»
{context}
Разбери вопрос для поиска по базе статей.
1. Если вопрос составной — в нём несколько разных вещей ({пример_составного}), —
   раздели его на части, не больше {parts}. Простой вопрос — одна часть, не дроби его.
2. На каждую часть — {n} поисковых запроса: 2–6 слов, терминами предметной области,
   без кавычек и логических операторов; хотя бы один {languages};
   запросы различаются по смыслу, а не перестановкой слов.
3. «предмет» — если вопрос про что-то конкретное ({что_конкретное}), его название в
   именительном падеже, как в вопросе; иначе null.
4. «всё» — true, если просят собрать всё известное о нём: что известно, что описано,
   с чем связано, полный перечень, обзор. Иначе false.
5. «вопрос_целиком» — вопрос так, чтобы он был понятен без разговора. Если это
   продолжение предыдущего вопроса ({пример_вдогонку}), допиши предмет из
   предыдущего: {пример_целиком} — и части, запросы и предмет составляй по нему.
   Если вопрос новый и понятен сам по себе — повтори его как есть, предыдущий не
   приплетай.
6. «по_теме» — true, если вопрос {по_теме}; false — если о другом ({не_по_теме}…).
7. «каталог» — если вопрос о самом наборе статей базы: сколько статей о чём-то, перечисли \
или покажи статьи (о теме, автора, за годы), какие статьи или работы есть («сколько статей \
о радаре», «какие статьи Смита есть в базе», «какие работы есть у Петрова», «статьи за 2020 \
год»). Тогда — объект: «тема» — коротко, о чём \
(без слов «статьи», годов и автора); «основы» — СПИСОК основ слов темы для поиска \
подстрокой в названиях и тексте: по 1–2 слова, усечённые до основы, на русском, \
английском и других языках статей, с вариантами названий (для Франции: France, French, \
Франц; для нейросетей: neural network, deep learning, нейросет, CNN); нет темы — пустой \
список; «год_от», «год_до» — числом; «автор» — фамилия латиницей, как в научных статьях; \
не названы — null. Каталожный вопрос всегда «по_теме»: true. Вопрос о содержании статей \
(что в них сказано, как, почему, какие выводы) — не каталог: null.

Верни JSON: {{"вопрос_целиком": "...", "части": [{{"вопрос": "...", "запросы": ["...", "..."]}}], \
"предмет": null, "всё": false, "по_теме": true, "каталог": null}}"""

# Статьи на языках, до которых русский и английский запрос не дотягивается: у базы
# линеаментов китайская статья об Илийском бассейне по-русски и по-английски не
# находилась вовсе, а по-китайски стояла первой (сходство 0,72 против 0,56 у случайных;
# стресс-тест 20.10.2026). Языки — тема.языки_перевода в kb.yaml; нет их — правила нет.
TRANSLATE_RULE = """
8. «перевод» — суть вопроса коротким поисковым запросом (ключевые термины, 4–15 \
иероглифов или 2–6 слов, не весь вопрос) на каждом из языков по порядку: {languages}. \
В базе есть статьи на них, и запрос по-русски или по-английски их не находит. Названия \
мест и терминов — так, как их пишут на этом языке. Пиши именно на этих языках (не \
по-корейски, если корейского в списке нет).
В JSON добавь: "перевод": ["...", "..."] — по запросу на каждый язык, в том же порядке."""


def translate_languages(kb: Any) -> list[str]:
    raw = kb.topic.get("языки_перевода") or []
    return [str(x) for x in (raw if isinstance(raw, list) else [raw]) if str(x).strip()]


# Вопрос «собрать всё» — на случай, когда модель вопрос не разобрала.
_COLLECT = re.compile(
    r"что\s+(?:известно|знаем|есть|описано|написано)|вс[её]\s+(?:о|об|про|что)\b|"
    r"перечисл|полн\w*\s+(?:список|перечень)|обзор|датасет|"
    r"какие\s+(?:признаки|методы|процессы)\s+(?:на|в|для|по)\b",
    re.IGNORECASE,
)

# Вопрос о связях названного предмета («с чем связан X», «с какими минералами
# ассоциирует X», «что сопровождает X») — тоже через граф: сводка всех его связей
# со счётом статей, а не 8 найденных фрагментов. Проверка 13.10.2026 на 12 таких
# вопросах (.local/logs/graph-ab-20261013): через граф ответ лучше вслепую в 7, поиском — в
# 2, вровень — в 3; связей в ответе втрое больше. Модель при разборе такие вопросы
# «собрать всё» не отмечала (0 из 12), поэтому их узнаёт код.
_RELATIONS = re.compile(
    r"с\s+ч[её]м\s+(?:[\w-]+\s+){0,2}?(?:связ|ассоцииру|сопряж|сочета)|"
    r"с\s+какими\s+(?:[\w-]+\s+){0,4}?(?:связ|ассоцииру|сочета|сопровожда)|"
    r"\bчто\s+(?:сопровожда|сопутству|контролиру|влия\w*\s+на)|"
    r"\bкакие\s+(?:[\w-]+\s+){0,3}?(?:сопровожда|сопутству|ассоцииру|связаны\s+с)|"
    r"\bгде\s+встреча",
    re.IGNORECASE,
)


def about_links(question: str) -> bool:
    """Вопрос о связях предмета — его ведёт граф (см. _RELATIONS)."""
    return bool(_RELATIONS.search(question))


@dataclass
class Part:
    """Часть вопроса и её поисковые запросы."""

    question: str
    queries: list[str]
    found: int = 0  # сколько фрагментов за этой частью в ответе


@dataclass
class Plan:
    """Разбор вопроса: части с запросами, предмет вопроса, «собрать всё»."""

    parts: list[Part]
    subject: str | None = None  # о чём вопрос: «Анабарский щит», «окварцевание»
    collect: bool = False
    main: str = ""  # сам вопрос как запрос (с прошлым, если вдогонку)
    extra: list[str] = field(default_factory=list)  # запросы следующих кругов
    missing: list[str] = field(default_factory=list)  # чего не хватало, по словам модели
    standalone: str = ""  # вопрос, понятный без разговора (вдогонку — с предметом)
    on_topic: bool = True  # по теме ли базы вопрос, по словам модели
    # Вопрос на языках_перевода базы. Ищется отдельно и даёт немного кандидатов сверху
    # (retrieval.TRANSLATED): в общем списке запросов китайский запрос со словом
    # «DRASTIC» притягивал таблицы и вытеснял нужную аннотацию.
    translated: list[str] = field(default_factory=list)
    # Вопрос о самом наборе статей («сколько статей про…») — что посчитать (catalog.py).
    catalog: Any = None

    @property
    def queries(self) -> list[str]:
        """Все запросы первого круга, без повторов: сам вопрос, затем по частям."""
        out: list[str] = []
        for q in [self.main, *(q for p in self.parts for q in p.queries)]:
            if q and q.lower() not in {x.lower() for x in out}:
                out.append(q)
        return out

    def __iter__(self) -> Any:
        """Разбор можно перебирать как список запросов — так с ним работал старый код."""
        return iter(self.queries)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {"question": p.question, "queries": p.queries, "found": p.found} for p in self.parts
        ]


def previous_question(history: list[dict[str, Any]]) -> str:
    past = [m["content"] for m in history if m.get("role") == "user" and m.get("content")]
    return past[-1].strip() if past else ""


# Вопрос вдогонку: начинается с «а», «и», «ещё» или ссылается на сказанное
# («его», «там», «этот»). Одной длины мало: «Как испечь ржаной хлеб?» тоже
# короткий, и раньше к нему приклеивался прошлый вопрос про космоснимки.
_FOLLOW_UP = re.compile(
    r"^(?:а|и|ещё|еще|также|тоже|тогда|то\s+есть)\b|"
    r"\b(?:его|её|ее|их|него|неё|нее|них|там|тут|здесь|этот|эта|это|эти|этого|этой|этих|"
    r"этим|этому|этом|том|той|тех|такой|такие|такого|он|она|оно|они)\b",
    re.IGNORECASE,
)
FOLLOW_UP_WORDS = 8


def search_query(question: str, history: list[dict[str, Any]]) -> str:
    """Вопрос как есть — первый из запросов. Короткий вопрос вдогонку («а по
    ASTER?») без прошлого вопроса ничего осмысленного не найдёт — к нему
    приклеивается предыдущий. Это запасной путь: обычно вопрос целиком
    переписывает модель при разборе (Plan.standalone)."""
    question = question.strip()
    previous = previous_question(history)
    if previous and len(question.split()) <= FOLLOW_UP_WORDS and _FOLLOW_UP.search(question):
        return f"{previous} {question}"
    return question


def is_follow_up(question: str, plan: Plan | None, history: list[dict[str, Any]]) -> bool:
    """Продолжение ли разговора. Модель при разборе переписывает такой вопрос
    (вопрос целиком отличается от заданного); молчит — судим по словам."""
    if not previous_question(history):
        return False
    if plan is not None and plan.standalone:
        return " ".join(plan.standalone.lower().split()) != " ".join(question.lower().split())
    return len(question.split()) <= FOLLOW_UP_WORDS and bool(_FOLLOW_UP.search(question))


def clean_queries(values: Any, limit: int) -> list[str]:
    """Запросы от модели: без кавычек и повторов, от 2 до 120 знаков, не больше limit."""
    out: list[str] = []
    for q in values or []:
        q = " ".join(str(q).replace('"', " ").split())
        if 2 <= len(q) <= 120 and q.lower() not in {x.lower() for x in out}:
            out.append(q)
    return out[:limit]


def parse_plan(data: Any, question: str, settings: Settings) -> Plan:
    """JSON модели → разбор вопроса. Непонятное — один вопрос без запросов."""
    if not isinstance(data, dict):
        return Plan([Part(question, [])])
    parts = []
    for item in data.get("части") or data.get("parts") or []:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("вопрос") or item.get("question") or "").split())
        queries = clean_queries(item.get("запросы") or item.get("queries"), settings.queries)
        if text or queries:
            parts.append(Part(text or question, queries))
    if not parts and data.get("queries"):  # старый вид ответа: только запросы
        parts = [Part(question, clean_queries(data["queries"], settings.queries))]
    parts = parts[: max(1, settings.parts)] or [Part(question, [])]
    subject = data.get("предмет") or data.get("территория") or data.get("subject")
    subject = " ".join(str(subject).split()) if isinstance(subject, str) else None
    if subject and subject.lower() in {"null", "none", "нет"}:
        subject = None
    collect = data.get("всё", data.get("все", data.get("collect")))
    collect = collect is True or str(collect).lower() in {"true", "да", "1"}
    collect = collect or about_links(question)
    standalone = " ".join(str(data.get("вопрос_целиком") or data.get("standalone") or "").split())
    topic = data.get("по_теме", data.get("on_topic", True))
    on_topic = not (topic is False or str(topic).lower() in {"false", "нет", "0"})
    translated = data.get("перевод") or data.get("translated")
    if isinstance(translated, str):
        translated = [translated]
    from .catalog import parse as parse_catalog

    catalog = parse_catalog(data.get("каталог") or data.get("catalog"))
    return Plan(
        parts,
        subject or None,
        collect,
        standalone=standalone[:300],
        on_topic=on_topic,
        translated=clean_queries(translated, 4),
        catalog=catalog,
    )


def plan_question(question: str, history: list[dict[str, Any]], settings: Settings) -> Plan:
    """Разбор вопроса моделью: части, запросы, предмет вопроса, «собрать всё».

    Модель не ответила — вопрос одной частью: ищется сам вопрос, а «собрать
    всё» узнаётся по словам («что известно о…», «перечисли…»)."""
    previous = previous_question(history)
    context = f"Предыдущий вопрос в разговоре: «{previous}»\n" if previous else ""
    kb = get_kb(settings.kb)
    user = kb.prompt(
        "разбор",
        PLAN_USER,
        question=question.strip(),
        context=context,
        n=settings.queries,
        parts=max(1, settings.parts),
        languages=languages(kb, " и хотя бы один "),
    )
    extra = translate_languages(kb)
    if extra:
        user += TRANSLATE_RULE.format(languages=", ".join(extra))
    system = kb.prompt("разбор.система", PLAN_SYSTEM)
    try:
        data = settings.client().chat_json(system, user, temperature=0.2, timeout=(10, 90))
    except Exception:  # noqa: BLE001 — без разбора поиск всё равно пойдёт
        return Plan(
            [Part(question, [])], collect=bool(_COLLECT.search(question)) or about_links(question)
        )
    return parse_plan(data, question, settings)


def make_plan(
    question: str, history: list[dict[str, Any]], settings: Settings, planner: Any
) -> Plan:
    """Разбор от planner'а, приведённый к одному виду.

    planner может вернуть Plan или просто список запросов (так было раньше и
    так удобнее в проверках) — тогда вопрос одной частью."""
    raw = planner(question, history, settings)
    if isinstance(raw, Plan):
        plan = raw
    else:
        collect = bool(_COLLECT.search(question)) or about_links(question)
        plan = Plan([Part(question, clean_queries(raw, 99))], collect=collect)
    if not plan.parts:
        plan.parts = [Part(question, [])]
    if not plan.main:
        if plan.standalone:
            # Модель переписала вопрос так, чтобы он был понятен без разговора.
            plan.main = plan.standalone
        elif plan.subject:
            # Вопрос о названном предмете — самостоятельный, прошлый к нему не клеится.
            plan.main = question.strip()
        else:
            plan.main = search_query(question, history)
    return plan
