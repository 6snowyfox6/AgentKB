"""Чат-бот: вопрос → поиск по базе вместе с моделью → ответ Qwen3 по найденному.

Обычный поиск (векторы + слова) на сложном вопросе ошибается в обе стороны:
не находит нужное, если оно сказано другими словами, и приносит похожее по
словам, но не о том. Поэтому поиск ведёт модель:

1. **Разбор.** Qwen3 делит составной вопрос на части (до трёх), на каждую
   составляет запросы — по-русски и по-английски, терминами предметной
   области; отмечает, о чём конкретно вопрос и просят ли «собрать всё».
2. **Широкий набор.** По запросам ищется с запасом — до десяти кандидатов на
   часть, с мягким порогом: лучше показать модели лишнее, чем не показать нужное.
3. **Модель отбирает.** Qwen3 читает кандидатов и называет те, где правда есть
   ответ (похожее по словам, но о другом — отбрасывает), говорит, чего не
   хватает, и составляет запросы, чтобы это найти.
4. **Ищет ещё.** По этим запросам — новый набор, модель отбирает снова. Всего
   до трёх кругов; останавливается, когда всего хватает или новое не находится.
5. **«Собрать всё»** («что известно об Анабарском щите») — из графа: все факты
   о предмете вопроса и о том, что в него входит, выписанные моделью из всех
   статей, с цитатами; поиск только добавляет.
6. **Ответ.** Модель отвечает ТОЛЬКО по отобранным фрагментам, после каждого
   утверждения — номер фрагмента [1]. Длина — по просьбе в самом вопросе:
   «подробно», «кратко», «перечисли» — так и отвечает; настроек нет.
7. **Не нашлось** — статьи не показываются: бот говорит, что в базе такого нет,
   и отвечает из общих знаний модели, с пометкой.

Модель не ответила при отборе (сервер модели выключен или сбой) — отбор по порогу
близости, как у обычного поиска: бот работает, только хуже на сложных вопросах.

Здесь — всё вместе; части — в своих модулях: settings (настройки), plan (разбор
вопроса), search (поиск по порогу), retrieval (поиск с моделью), dataset (граф:
«собрать всё» и вопросы о связях), messages (задание модели), conflicts
(расхождения между статьями), checks (проверка ответа: ссылки, отказ).

Ответ идёт потоком — событиями:

    {"type": "status", "text": "…"}                         что сейчас делается
    {"type": "sources", "queries": […], "sources": […],     на что опирается ответ
     "parts": […], "extra": […], "missing": […], "dataset": "…"|None,
     "rounds": 1–3, "checked": N, "judged": true|false}
    {"type": "general", "queries": […], "text": "…", …}     в базе нет — ответ без неё
                                                            (off_topic: true — вопрос не по
                                                            теме базы: только отказ, без ответа)
    {"type": "token", "text": "…"}                          кусок ответа, много раз
    {"type": "revised", "text": "…", "removed": […]}       ответ целиком без фраз без
                                                            ссылки на фрагмент (drop_uncited)
    {"type": "done", "mode": "база"|"без базы", "used": [1, 3], "unknown": [],
     "coverage": [5, 6], "removed": 2, "rounds": …,
     "articles": [{"doc_id", "title", "authors", "year", "journal", "doi", "url",
                   "fragments": [1, 3], "pages": [4, 5], "pdf": true}, …],
     "timings": {"plan": с, "search": с, "judge": с, "conflicts": с, "answer": с}, …}
                                                            (этапы, которых не было, — без ключа;
                                                            те же числа — строкой в
                                                            .local/logs/chat-<дата>.log)
    {"type": "error", "error": "…"}
"""

from __future__ import annotations

import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..kb import get as get_kb
from ..llm import LLM, LLM_HOST, LLMError
from ..paths import LOGS
from . import references
from .checks import NO_ANSWER, citations, coverage, drop_uncited, is_refusal
from .conflicts import find_conflicts
from .dataset import collect_dataset, dataset_sources, graph_candidates
from .messages import build_messages, fit_budget, general_messages
from .plan import Plan, is_follow_up, make_plan, plan_question, search_query
from .retrieval import AUTO, MAX_ROUNDS, Retrieval, judge_fragments, retrieve
from .settings import MODEL, Settings

NOT_IN_BASE = (
    "По вашему запросу в базе знаний такой информации нет. Ниже — ответ модели "
    "из её общих знаний: он не подтверждён статьями базы, проверяйте его."
)

# Вопрос не по теме базы и в базе ничего нет: бот не отвечает из общих знаний
# (раньше выдавал, например, рецепт хлеба). Вернуть старое — Settings.general_off_topic.
# Текст отказа — «отказ_не_по_теме» в настройках базы (kb.yaml, раздел «тема»).
OFF_TOPIC = "отказ_не_по_теме"


# --------------------------------------------------------------------------- #
#  Модель
# --------------------------------------------------------------------------- #
def llm_status(host: str = LLM_HOST, model: str = MODEL, timeout: int = 5) -> dict[str, Any]:
    """Жив ли сервер модели и та ли на нём модель — для подсказки на странице."""
    return LLM(model=model, host=host).status(timeout=timeout)


def quick_plan(question: str, history: list[dict[str, Any]], settings: Settings) -> list[str]:
    """Быстрый режим: без модели — ищется сам вопрос (вдогонку — вместе с прошлым)."""
    return [search_query(question, history)]


def stream_llm(messages: list[dict[str, Any]], settings: Settings) -> Iterator[dict[str, Any]]:
    """Ответ модели кусками: {"text": …} и в конце {"done": True, "tokens": …}."""
    return settings.client().stream(messages)


# --------------------------------------------------------------------------- #
#  Замеры по этапам
# --------------------------------------------------------------------------- #
# Этапы ответа и их имена для человека — в журнале, на странице и в Телеграме.
STAGES = {
    "plan": "разбор",
    "search": "поиск",
    "judge": "отбор",
    "conflicts": "сверка",
    "answer": "ответ",
}


def describe_timings(timings: dict[str, float]) -> str:
    """«разбор 1,2 с · отбор 6,0 с · ответ 9,8 с» — только этапы, которые были."""
    return " · ".join(
        f"{name} {timings[key]:.1f} с".replace(".", ",")
        for key, name in STAGES.items()
        if key in timings
    )


def log_timings(question: str, timings: dict[str, float], seconds: float, mode: str) -> None:
    """Строка в .local/logs/chat-<дата>.log: что сколько заняло. Не записалось — не беда."""
    now = datetime.now()
    line = (
        f"{now:%H:%M:%S}  {describe_timings(timings)} · всего {seconds:.1f} с".replace(".", ",")
        + f"  [{mode}]  {' '.join(question.split())[:200]}\n"
    )
    try:
        LOGS.mkdir(parents=True, exist_ok=True)
        with open(LOGS / f"chat-{now:%Y%m%d}.log", "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
#  Всё вместе
# --------------------------------------------------------------------------- #
@dataclass
class Prepared:
    """Всё, что нужно для ответа: разбор вопроса, датасет, найденное."""

    plan: Plan
    sources: list[dict[str, Any]]
    retrieval: Retrieval
    dataset: dict[str, Any] | None = None
    graph: int = 0  # сколько кандидатов дал граф (вопрос о названном предмете)
    timings: dict[str, float] = field(default_factory=dict)  # plan, search, judge — секунды

    def info(self) -> dict[str, Any]:
        plan = self.plan
        return {
            "queries": plan.queries + plan.extra,
            "standalone": plan.standalone,
            "graph": self.graph,
            "parts": plan.describe() if len(plan.parts) > 1 else [],
            "extra": plan.extra,
            "missing": plan.missing,
            "dataset": self.dataset["name"] if self.dataset else None,
            **self.retrieval.info(),
        }


def is_overview(question: str) -> bool:
    from ..index.summaries import is_overview as check

    return check(question)


def _overview_sources(conn: Any, embedder: Any, question: str, kb: str) -> list[dict[str, Any]]:
    """Резюме статей для обзорного вопроса; таблицы нет или заглушка — без них."""
    from ..index import db
    from ..index.summaries import overview_sources

    if not db.is_real(conn):
        return []
    try:
        return overview_sources(conn, embedder, question, kb)
    except Exception:  # noqa: BLE001
        return []


def prepare(
    conn: Any,
    embedder: Any,
    question: str,
    history: list[dict[str, Any]],
    settings: Settings,
    planner: Any = plan_question,
    judge: Any = judge_fragments,
    datasets: Any = collect_dataset,
    graph_hits: Any = graph_candidates,
    reranker: Any = AUTO,
    overview: Any = None,
) -> Generator[Any, Any, Any]:
    """Разбор вопроса и поиск с моделью. Генератор событий status; в конце — Prepared.
    overview(conn, embedder, вопрос, kb) — источники обзорного вопроса (резюме статей);
    по умолчанию — index/summaries.overview_sources.

    Быстрый режим (settings.fast): вопрос не разбирается моделью — ищется он сам (вектор
    BGE-M3 и так находит статьи на других языках), отбирает реранкер без модели."""
    if overview is None:
        overview = _overview_sources
    if settings.fast:
        planner, judge = quick_plan, None
    else:
        yield {"type": "status", "text": "разбираю вопрос…"}
    started = time.monotonic()
    plan = make_plan(question, history, settings, planner)
    planned = time.monotonic()

    # «Собрать всё» о предмете вопроса — из датасета; поиск только добавляет.
    data = None
    if plan.collect and datasets is not None:
        yield {"type": "status", "text": "собираю факты из графа…"}
        data = datasets(conn, plan.subject, question, kb=settings.kb)
    base = dataset_sources(data) if data else []
    # Обзорный вопрос — по резюме статей (index/summaries.py): источник на статью.
    if not base and overview is not None and is_overview(question):
        base = overview(conn, embedder, plan.standalone or question, settings.kb) or []
        if base:
            yield {"type": "status", "text": f"беру резюме {len(base)} статей…"}

    # Вопрос о названном предмете — фрагменты о нём из графа в кандидаты.
    extra: list[Any] = []
    if not data and plan.subject and graph_hits is not None and judge is not None:
        extra = graph_hits(conn, plan.subject, question, kb=settings.kb) or []
        if extra:
            yield {"type": "status", "text": f"беру из графа фрагменты о «{plan.subject}»…"}

    yield {
        "type": "status",
        "text": (
            f"ищу в базе по {len(plan.parts)} частям вопроса…"
            if len(plan.parts) > 1
            else "ищу в базе…"
        ),
    }
    limit = max(settings.top_k - (len(base) - 1), 3) if base else None
    # Вопрос не по теме базы — один круг: проверить, что в базе правда пусто, и всё.
    rounds = MAX_ROUNDS if plan.on_topic else 1
    found = yield from retrieve(
        conn, embedder, question, plan, settings, judge, base, limit, rounds, extra, reranker
    )
    # Поиск — всё после разбора, кроме отбора моделью: граф, база, соседние куски.
    timings = {
        "plan": round(planned - started, 2),
        "search": round(time.monotonic() - planned - found.judge_sec, 2),
    }
    if found.judge_sec:
        timings["judge"] = round(found.judge_sec, 2)
    return Prepared(plan, base + found.sources, found, data, len(extra), timings)


def answer(
    conn: Any,
    embedder: Any,
    question: str,
    history: list[dict[str, Any]] | None = None,
    settings: Settings | None = None,
    llm: Any = stream_llm,
    planner: Any = plan_question,
    judge: Any = judge_fragments,
    datasets: Any = collect_dataset,
    prepared: Prepared | None = None,
    conflicts: Any = find_conflicts,
    reranker: Any = AUTO,
) -> Generator[dict[str, Any], None, None]:
    """Ответ на вопрос событиями (см. начало файла).

    llm, planner, judge и datasets подменяются в проверках: planner может
    вернуть и просто список запросов, judge — None (отбор по порогу), datasets
    (conn, предмет, вопрос) — датасет или None. prepared — уже найденное
    (оценка ищет один раз и для проверки поиска, и для ответа). conflicts
    (вопрос, фрагменты, настройки) → расхождения между статьями (find_conflicts);
    с подменённой моделью ответа настоящая сверка не идёт — она спросила бы модель."""
    settings = settings or Settings()
    history = history or []
    question = (question or "").strip()
    if not question:
        yield {"type": "error", "error": "пустой вопрос"}
        return

    started = time.monotonic()
    if prepared is None:
        prepared = yield from prepare(
            conn,
            embedder,
            question,
            history,
            settings,
            planner,
            judge,
            datasets,
            reranker=reranker,
        )
    sources, plan, info = prepared.sources, prepared.plan, prepared.info()
    info["fast"] = settings.fast
    timings = dict(prepared.timings)
    kb = get_kb(settings.kb)

    def spent(stage: str, since: float) -> None:
        timings[stage] = round(timings.get(stage, 0.0) + time.monotonic() - since, 2)

    def closing(mode: str) -> dict[str, Any]:
        """Поля конца ответа: всего секунд и по этапам; та же строка — в журнал."""
        seconds = round(time.monotonic() - started, 1)
        log_timings(question, timings, seconds, mode)
        return {"seconds": seconds, "timings": dict(timings)}

    # Разговор модели нужен только для вопроса вдогонку. В самостоятельном
    # вопросе прошлый ответ — помеха: на живом прогоне в ответ о Персияновском
    # разломе попала фраза из прошлого ответа об Ишимбинском, со ссылкой [1].
    if not is_follow_up(question, plan, history):
        history = []

    text, tokens = "", None
    if sources:
        clashes: list[dict[str, Any]] = []
        real = conflicts is not find_conflicts or llm is stream_llm
        if conflicts is not None and settings.check_conflicts and not settings.fast and real:
            yield {"type": "status", "text": "сверяю, не расходятся ли статьи…"}
            since = time.monotonic()
            clashes = conflicts(question, sources, settings) or []
            spent("conflicts", since)
        info["conflicts"] = clashes
        sources, turns = fit_budget(question, sources, history, settings, plan, clashes)
        yield {"type": "status", "text": "пишу ответ по найденному…"}
        asked = time.monotonic()
        stream = _clean(
            llm(
                build_messages(question, sources, turns, settings.model, plan, clashes, kb),
                settings,
            )
        )
        try:
            # Первые слова держатся, пока не ясно, не отказ ли это: если модель
            # прочла фрагменты и ответа в них нет, статьи не показываются вовсе.
            head, finished = "", False
            for part in stream:
                if "done" in part:
                    tokens, finished = part["done"], True
                    break
                head += part["text"]
                if len(head.strip()) >= len(NO_ANSWER) + 10:
                    break
            if head.strip() and not is_refusal(head):
                yield {"type": "sources", "sources": sources, **info}
                text = head
                if head:
                    yield {"type": "token", "text": head}
                if not finished:
                    for part in stream:
                        if "done" in part:
                            tokens = part["done"]
                            break
                        text += part["text"]
                        yield {"type": "token", "text": part["text"]}
                spent("answer", asked)
                # Фразы без ссылки на фрагмент — не из статей: ответ заменяется
                # очищенным, убранное показывается отдельно.
                cleaned, removed = drop_uncited(text)
                if removed and cleaned:
                    text = cleaned
                    yield {"type": "revised", "text": text, "removed": removed}
                else:
                    removed = []
                used, unknown = citations(text, len(sources))
                cited, claims = coverage(text)
                yield {
                    "type": "done",
                    "mode": "база",
                    "used": used,
                    "unknown": unknown,
                    "coverage": [cited, claims],
                    # Статьи, на которые опирается ответ (chat/references.py).
                    "articles": references.articles(conn, sources, used),
                    "removed": len(removed),
                    "model": settings.model,
                    **closing("база"),
                    "tokens": tokens,
                    **info,
                }
                return
        except LLMError as exc:
            yield {"type": "error", "error": str(exc)}
            return
        finally:
            stream.close()
        spent("answer", asked)  # модель отказалась — время этой попытки тоже в ответе

    if not plan.on_topic and not settings.general_off_topic:
        # Не по теме базы и в базе пусто — короткий отказ, без ответа из общих знаний.
        yield {"type": "general", "text": kb.text(OFF_TOPIC), "off_topic": True, **info}
        yield {
            "type": "done",
            "mode": "без базы",
            "used": [],
            "unknown": [],
            "coverage": [0, 0],
            "model": settings.model,
            **closing("не по теме"),
            "tokens": None,
            "off_topic": True,
            **info,
        }
        return

    # В базе нет: ни статей, ни номеров — ответ модели с пометкой.
    yield {"type": "general", "text": NOT_IN_BASE, **info}
    yield {"type": "status", "text": "отвечаю без базы…"}
    asked = time.monotonic()
    try:
        messages = general_messages(question, history, settings.model, kb)
        for part in _clean(llm(messages, settings)):
            if "done" in part:
                tokens = part["done"]
                break
            text += part["text"]
            yield {"type": "token", "text": part["text"]}
    except LLMError as exc:
        yield {"type": "error", "error": str(exc)}
        return
    spent("answer", asked)
    yield {
        "type": "done",
        "mode": "без базы",
        "used": [],
        "unknown": [],
        "coverage": [0, 0],
        "model": settings.model,
        **closing("без базы"),
        "tokens": tokens,
        **info,
    }


def _clean(parts: Any) -> Generator[dict[str, Any], None, None]:
    """Поток модели без блока размышлений: {"text": …} и в конце {"done": токены}."""
    think = ThinkFilter()
    try:
        for part in parts:
            if part.get("done"):
                tail = think.flush()
                if tail:
                    yield {"text": tail}
                yield {"done": part.get("tokens")}
                return
            piece = think.feed(part.get("text", ""))
            if piece:
                yield {"text": piece}
        tail = think.flush()
        if tail:
            yield {"text": tail}
        yield {"done": None}
    finally:
        close = getattr(parts, "close", None)
        if close:
            close()


class ThinkFilter:
    """Срезает блок <think>…</think>, если модель его всё-таки прислала.

    Размышления выключены (llm.NO_THINK), но сервер, запущенный без этой
    поддержки, отдаёт их прямо в тексте. Приходит это кусками, поэтому фильтр
    держит состояние между кусками.
    """

    def __init__(self) -> None:
        self.buffer = ""
        self.state = "start"  # start → (think →) text

    def feed(self, chunk: str) -> str:
        if self.state == "text":
            return chunk
        self.buffer += chunk
        if self.state == "start":
            head = self.buffer.lstrip()
            if not head:
                return ""
            if "<think>".startswith(head[:7]) and len(head) < 7:
                return ""  # ещё не ясно, начало ли это блока
            if head.startswith("<think>"):
                self.state = "think"
            else:
                self.state = "text"
                out, self.buffer = self.buffer.lstrip(), ""
                return out
        if self.state == "think" and "</think>" in self.buffer:
            rest = self.buffer.split("</think>", 1)[1].lstrip()
            self.state, self.buffer = "text", ""
            return rest
        return ""

    def flush(self) -> str:
        out = self.buffer if self.state != "think" else ""
        self.buffer = ""
        return out
