"""Задание модели: инструкция, фрагменты с номерами, пометки, прошлые реплики.

Задание вместе с ответом должно влезть в окно модели (fit_budget)."""

from __future__ import annotations

import re
from typing import Any

from ..kb import KB
from ..kb import get as get_kb
from .checks import CITE_MARK
from .plan import Plan
from .settings import MODEL, Settings

FRAGMENT_CHARS = 2400
DATASET_CHARS = 6000  # сводка фактов из графа (dataset) — длиннее фрагмента

HISTORY_TURNS = 2  # сколько прошлых вопросов-ответов помнит модель
HISTORY_CHARS = 1500

# Окно модели считается грубо: знаков на токен у Qwen3 на русском — около трёх.
# Сообщение и ответ должны влезть в окно модели (num_ctx — то же число, что
# --max-model-len у vLLM), иначе сервер откажет. Если не влезает, сначала укорачивается история, потом
# отбрасываются последние фрагменты.
CHARS_PER_TOKEN = 3.0
ANSWER_RESERVE = 3000  # токенов под сам ответ

# Слова в {фигурных скобках} — из раздела «тема» настроек базы (kb.py); шаблон целиком
# узел может заменить в разделе «промпты»: «ответ» и «без_базы».
SYSTEM = """Ты — помощник {пользователь}. Отвечаешь на вопросы по базе знаний — статьям \
по {область}. Твой единственный источник — фрагменты статей из базы, которые \
даны в сообщении.

Правила:
1. Отвечай ТОЛЬКО по фрагментам. Не добавляй фактов, чисел, названий, методов и ссылок, \
которых во фрагментах нет, — даже если знаешь их сам.
2. Каждое предложение, в котором есть факт, — с номером фрагмента в квадратных \
скобках: [1]; опирается на несколько — [1][3]. Это касается и вводных фраз, и выводов: \
обобщаешь — ставь номера фрагментов, из которых обобщение следует. Чего нельзя \
подкрепить номером фрагмента — не пиши.
3. Если во фрагментах нет ответа на вопрос, напиши только одну фразу: \
«В базе знаний нет ответа на этот вопрос.» — и больше ничего.
4. Пиши {язык_ответа}, даже если фрагменты на другом языке. Термин можно дать в скобках \
на языке фрагмента.
5. Объём и вид ответа — как просит человек в вопросе. Просит подробно, развёрнуто, \
обзор, всё о чём-то — отвечай развёрнуто, по разделам с короткими заголовками жирным \
(например, **Методы**), с числами, условиями и различиями между источниками. Просит \
кратко, в двух словах, одним списком — так и отвечай. Не сказал — по существу: вывод \
в 1–2 предложения и главное списком, без воды и без повторов.
6. Если фрагменты отвечают лишь частично — ответь на то, что в них есть, и одной \
фразой назови, чего во фрагментах нет. Фраза из правила 3 — только когда нет совсем \
ничего.
7. Сведения во фрагментах относятся к конкретным {объекты}. \
Называй, к чему относится утверждение ({пример_привязки}), и не выдавай частный случай \
из одной статьи за общее правило.
8. Формулы во фрагментах стоят между $$ и $$ (LaTeX), номер формулы в статье — «(N)» в \
конце. Нужна формула — приведи её дословно между $$ и $$, назови номер и фрагмент: \
«по формуле (2) [3]: $$…$$». Обозначения объясняй словами фрагмента; формул, которых \
во фрагментах нет, не пиши и не выводи.
9. Фрагмент «Рисунок» — описание рисунка моделью по изображению, а не текст статьи: \
ссылайся «на рис. 3 [4]» и говори, что это видно на рисунке, а не написано авторами.
10. Фрагмент «Резюме статьи» — пересказ всей статьи, а не её слова: годится для обзора \
(что за работа, о чём, к чему пришла), а числа и подробности бери из фрагментов текста."""


GENERAL_SYSTEM = """Ты — помощник {пользователь}, специалист по {область}. В базе знаний \
статей ответа на вопрос не нашлось, поэтому отвечаешь из общих знаний.

Правила:
1. Не ссылайся на статьи, авторов, DOI и номера фрагментов — у тебя их нет. Не ставь \
номера в квадратных скобках.
2. Не выдумывай точных чисел, названий и дат. Если не уверен — так и скажи.
3. Пиши {язык_ответа}. Объём — как просит человек: подробно — развёрнуто, кратко — коротко; \
не сказал — по существу: вывод и главное списком, в конце — что стоит проверить по \
литературе."""


_LITERATURE_REF = re.compile(r"\[(\d+(?:\s*[,;–—-]\s*\d+)*)\]")


def fragment_block(source: dict[str, Any]) -> str:
    where = []
    if source.get("year"):
        where.append(str(source["year"]))
    if source.get("pages"):
        where.append("с. " + ", ".join(str(p) for p in source["pages"][:3]))
    # Ссылки на литературу внутри статьи («показано в [12]») модель переносила в ответ
    # как номера фрагментов — и ответ ссылался на [12], [16], [20], которых не было.
    text = _LITERATURE_REF.sub(r"(лит. \1)", source["text"])
    limit = DATASET_CHARS if source.get("kind") == "датасет" else FRAGMENT_CHARS
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " …"
    head = f"[{source['n']}] «{source['title']}»" + (f" ({', '.join(where)})" if where else "")
    return f"{head}\n{text}"


GENERAL_MARK = (
    "(Этот ответ был не из базы знаний, а из общих знаний модели — "
    "как источник его не используй.)\n"
)


def past_turns(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Прошлые реплики — чтобы понимать вопросы вдогонку. Последние два обмена,
    каждая укорочена.

    Из прошлых ответов убираются номера [n]: они вели на прошлые фрагменты, а в
    новом сообщении те же номера — у других, и модель переносила их. Ответ «без
    базы» (mode) помечается: иначе сказанное из общих знаний попадало в ответ по базе."""
    past = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
    out = []
    for m in past[-HISTORY_TURNS * 2 :]:
        content = str(m["content"])
        if m["role"] == "assistant":
            content = CITE_MARK.sub("", content)
            if m.get("mode") == "без базы":
                content = GENERAL_MARK + content
        out.append({"role": m["role"], "content": content[:HISTORY_CHARS]})
    return out


def conflicts_note(conflicts: list[dict[str, Any]]) -> str:
    """Пометка в задание. Последнее слово — за моделью ответа: она видит фрагменты
    целиком. С твёрдым «источники расходятся» модель говорила это и там, где сверка
    ошиблась (4 ложные пометки из 5), с этой — ни разу; настоящие расхождения
    показаны в 5 из 6 против 6 из 6 (11.10.2026)."""
    lines = "\n".join(
        f"- [{c['a']}] — «{c['said_a']}»; [{c['b']}] — «{c['said_b']}»" for c in conflicts
    )
    return (
        "Возможно, фрагменты из разных статей отвечают на вопрос по-разному:\n"
        + lines
        + "\nПроверь по самим фрагментам. Если они действительно говорят разное об одном и "
        "том же — приведи обе позиции, каждую со своим номером, и прямо скажи, что "
        "источники расходятся. Если они о разных объектах, районах или сторонах — "
        "расхождением это не называй."
    )


def build_messages(
    question: str,
    sources: list[dict[str, Any]],
    history: list[dict[str, Any]],
    model: str = MODEL,
    plan: Plan | None = None,
    conflicts: list[dict[str, Any]] | None = None,
    kb: KB | None = None,
) -> list[dict[str, Any]]:
    system = (kb or get_kb()).prompt("ответ", SYSTEM)
    messages = [{"role": "system", "content": system}, *past_turns(history)]
    fragments = "\n\n".join(fragment_block(s) for s in sources)
    notes = []
    shown = {s.get("n") for s in sources}
    conflicts = [c for c in conflicts or [] if c["a"] in shown and c["b"] in shown]
    if conflicts:
        notes.append(conflicts_note(conflicts))
    summary = next((s for s in sources if s.get("kind") == "датасет"), None)
    if summary is not None:
        notes.append(
            f"Фрагмент [{summary['n']}] — все факты о предмете вопроса из графа знаний "
            "(выписаны из статей базы). Перечисли из него всё, сгруппировав по смыслу, со "
            f"ссылкой [{summary['n']}]; подробности, примеры и числа бери из остальных "
            "фрагментов, с их номерами."
        )
    if plan is not None and len(plan.parts) > 1:
        lines = "\n".join(f"{i}) {p.question}" for i, p in enumerate(plan.parts, start=1))
        notes.append(
            "Вопрос состоит из частей:\n" + lines + "\nОтветь на каждую часть "
            "отдельным разделом. Если по какой-то части во фрагментах ничего нет — "
            "так и напиши в её разделе."
        )
    articles = {s["doc_id"]: s["title"] for s in sources if s.get("kind") != "датасет"}
    if len(articles) == 1 and len(sources) > 1:
        notes.append(
            f"Все фрагменты — из одной статьи «{next(iter(articles.values()))}». "
            "Скажи об этом в ответе и не выдавай сказанное в ней об одном районе или "
            "объекте за общее правило."
        )
    asked = question.strip()
    if plan is not None and plan.standalone and plan.standalone.lower() != asked.lower():
        asked += f"\n(Это продолжение разговора; вопрос целиком: {plan.standalone})"
    note = ("\n\n" + "\n\n".join(notes)) if notes else ""
    user = (
        f"Фрагменты из базы знаний:\n\n{fragments}\n\n"
        f"Вопрос: {asked}{note}\n\nОтвечай только по этим фрагментам, со ссылками."
    )
    messages.append({"role": "user", "content": user})
    return messages


def _tokens(messages: list[dict[str, Any]]) -> int:
    return int(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN) + 1


def fit_budget(
    question: str,
    sources: list[dict[str, Any]],
    history: list[dict[str, Any]],
    settings: Settings,
    plan: Plan | None = None,
    conflicts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Фрагменты и история, которые влезают в окно модели вместе с ответом.

    Сначала уходят старые обмены разговора, потом — последние фрагменты (самые
    слабые: порядок — как отобрала модель), но не меньше трёх. Нумерация не
    сбивается: отбрасываются только с конца."""
    budget = settings.num_ctx - ANSWER_RESERVE
    history = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")][
        -HISTORY_TURNS * 2 :
    ]
    kb = get_kb(settings.kb)

    def size() -> Any:
        return _tokens(
            build_messages(question, sources, history, settings.model, plan, conflicts, kb)
        )

    while history and size() > budget:
        history = history[2:]
    sources = list(sources)
    while len(sources) > 3 and size() > budget:
        sources.pop()
    return sources, history


def general_messages(
    question: str, history: list[dict[str, Any]], model: str = MODEL, kb: KB | None = None
) -> list[dict[str, Any]]:
    """Сообщения для ответа без базы: без фрагментов, с запретом на ссылки."""
    system = (kb or get_kb()).prompt("без_базы", GENERAL_SYSTEM)
    messages = [{"role": "system", "content": system}, *past_turns(history)]
    user = question.strip()
    messages.append({"role": "user", "content": user})
    return messages
