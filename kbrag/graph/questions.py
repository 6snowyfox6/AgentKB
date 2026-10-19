"""Вопросы для оценки — из фактов графа, с заранее известным ответом.

Оценка (evaluation.py) честна настолько, насколько хороши вопросы. Писать их
руками долго, а у каждого факта графа уже есть всё нужное: цитата (ответ),
фрагмент и статья, где он сказан. Поэтому вопрос строится из факта:

    факт:    «оруденение» — приурочено к — «зоны дробления»
    вопрос:  «К чему приурочено оруденение?»          (модель переформулирует)
    ключевые: «зоны дробления»                         (ответ — второй конец факта)
    статьи:  doc_id статьи с цитатой                  (что поиск обязан найти)

В вопросе нет ответа («к»): иначе поиск находил бы его просто по словам
вопроса, и оценка была бы завышена. Факты берутся из разных статей по очереди —
чтобы проверялась вся база, а не одна подробная статья.

    python kbrag.py questions              → .local/kbs/<база>/eval-graph.yaml
    python kbrag.py eval --questions .local/kbs/<база>/eval-graph.yaml

Модель (Qwen3) только переформулирует факт в вопрос исследователя; нет модели —
вопрос по шаблону. Проверяет ответы, как и прежде, код.
"""

from __future__ import annotations

import re
from typing import Any

from ..kb import KB
from ..kb import get as get_kb
from ..text import mentions, normalize

OUT_NAME = "eval-graph.yaml"  # в папке данных базы (KB.local)

# Слова темы — из настроек базы (kb.py); шаблоны целиком — «вопросы.система» и «вопросы».
SYSTEM = (
    "Ты составляешь вопросы для проверки поиска по базе статей по {область}. "
    "Отвечай только JSON."
)
USER = """Факт из статьи: «{src}» — {relation} — «{dst}».
Цитата: «{quote}»

Составь вопрос {пользователь}, на который отвечает эта цитата. В вопросе назови «{src}»,
но НЕ называй «{dst}» и его синонимов — это ответ. Пиши {язык_ответа}, 6–15 слов, без
кавычек.

Верни JSON: {{"вопрос": "..."}}"""

MAX_NAME_WORDS = 6
_DIGITS = re.compile(r"\d")


def _usable(name: str) -> bool:
    """Годится ли имя для вопроса: коротко, без чисел (количества — не предмет)."""
    return bool(name) and len(name.split()) <= MAX_NAME_WORDS and not _DIGITS.search(name)


def pick(g: Any, limit: int = 30) -> list[Any]:
    """Факты для вопросов: по очереди из разных статей, сначала подтверждённые
    несколькими статьями и с именами покороче."""
    by_doc: dict[str, list[Any]] = {}
    for link in g.links.values():
        if not (_usable(link.src) and _usable(link.dst)):
            continue
        doc = link.facts[0]["doc_id"]
        by_doc.setdefault(doc, []).append(link)
    for links in by_doc.values():
        links.sort(key=lambda lk: (-lk.documents, len(lk.src) + len(lk.dst)))
    out: list[Any] = []
    round_ = 0
    while len(out) < limit:
        taken = False
        for links in by_doc.values():
            if round_ < len(links):
                out.append(links[round_])
                taken = True
                if len(out) >= limit:
                    break
        if not taken:
            break
        round_ += 1
    return out


def template(link: Any) -> str:
    """Вопрос без модели: имя и связь, ответ — пропуск."""
    return f"«{link.src}» — {link.relation} — что именно, по статьям базы?"


def phrase(llm: Any, link: Any, kb: KB | None = None) -> str:
    """Модель переформулирует факт в вопрос. Не ответила или выдала ответ в вопросе — шаблон."""
    quote = link.facts[0]["quote"]
    kb = kb or get_kb()
    try:
        data = llm.chat_json(
            kb.prompt("вопросы.система", SYSTEM),
            kb.prompt(
                "вопросы",
                USER,
                src=link.src,
                relation=link.relation,
                dst=link.dst,
                quote=quote[:600],
            ),
        )
        text = " ".join(str(data.get("вопрос") or "").split()).strip(' «»"')
    except Exception:  # noqa: BLE001
        return template(link)
    # Ответ в вопросе — в любом падеже («зонам дробления» для «зоны дробления»).
    if not 10 <= len(text) <= 200 or mentions(link.dst, text):
        return template(link)
    return text if text.endswith("?") else text + "?"


def keywords(link: Any) -> list[str]:
    """Ключевое для ответа — второй конец факта; через | — его самое длинное слово,
    чтобы сверка по словам не требовала всего имени дословно."""
    words = sorted(re.findall(r"[A-Za-zА-Яа-яЁё]{5,}", link.dst), key=len, reverse=True)
    group = [link.dst] + (
        [words[0]] if words and normalize(words[0]) != normalize(link.dst) else []
    )
    return ["|".join(w.replace("|", "/") for w in group)]


def build(
    g: Any, llm: Any = None, limit: int = 30, log: Any = print, kb: KB | None = None
) -> list[dict[str, Any]]:
    out = []
    for i, link in enumerate(pick(g, limit), start=1):
        question = phrase(llm, link, kb) if llm is not None else template(link)
        docs = sorted({f["doc_id"] for f in link.facts})
        out.append(
            {
                "id": f"g{i:02d}",
                "вопрос": question,
                "ожидание": "база",
                "понятие": link.src,  # о чём вопрос: по нему отчёт ищет, где мало статей
                "ключевые": keywords(link),
                "статьи": docs,
                "факт": f"{link.src} — {link.relation} — {link.dst}",
            }
        )
        if log and i % 10 == 0:
            log(f"  вопросов: {i}")
    return out


def to_yaml(items: list[dict[str, Any]]) -> str:
    import yaml

    head = [
        "# ВОПРОСЫ ОЦЕНКИ ИЗ ГРАФА — сделаны командой python kbrag.py questions,",
        "# пересоздаются ею же (правки руками затрутся — переносите удачные вопросы",
        "# в eval-questions.yaml базы).",
        "#",
        "# У каждого вопроса известен ответ: «факт» из графа, «ключевые» — его второй",
        "# конец, «статьи» — где он сказан. Поиск должен найти эти статьи, ответ —",
        "# назвать ключевое.",
        "#",
        "# Запуск: python kbrag.py eval --questions <этот файл>",
        "",
    ]
    body = yaml.safe_dump({"вопросы": items}, allow_unicode=True, sort_keys=False, width=100)
    return "\n".join(head) + body
