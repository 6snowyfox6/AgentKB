"""Факты из статей: Qwen3 выписывает, код проверяет каждый.

Факт — «от — связь — к» и предложение фрагмента, где он сказан (цитата).
Видов сущностей нет: «от» и «к» — как названо в тексте (место, разлом, метод,
процесс, признак — всё равно), связь — короткая фраза по-русски. Разные
написания одного и того же сводит словарь синонимов (synonyms.py), а не модель.

Как идёт, по фрагментам статей:

1. Модели уходят правила (graph-rules.txt базы, config/kbs/… — правится Блокнотом) и
   фрагмент с пронумерованными предложениями. Модель возвращает JSON:
   {"факты": [[номер предложения, от, связь, к], …]}. Цитату модель не
   переписывает — её берёт код по номеру. Раньше модель переписывала цитату
   сама: это две трети её ответа, а каждая шестая цитата во фрагменте не
   находилась. С номером ответ вдвое короче, а цитата всегда дословная.
2. **Код проверяет каждый факт**, не прошедшее в граф не попадает:
   * в предложении видны оба конца — в любом падеже; из длинного имени можно
     пропустить одно общее слово, но не имя собственное («Онежский рудный
     район» по предложению про «рудный район» не проходит). Имя может стоять
     предложением раньше («This area…» — название района выше): тогда цитата —
     оба предложения;
   * концы — не пустые, не слишком длинные, не одно и то же, не год и не число,
     не количество («около 80 % сульфидов», «385 ± 2 млн лет»), не ссылка на
     работу («Binotto (2015)», «работа [33]»), не положение («южнее разлома»),
     не общее имя без названия («this study», «predictive model», «high
     values»); предлог в начале («в аномалиях Bi») снимается;
   * связь — от 1 до 6 слов.
   Что отброшено и почему — сохраняется: `graph --new` показывает, где модель
   ошибалась чаще всего.
3. Принятое пишется в базу по фрагменту. Проход возобновляемый: прервали —
   продолжится с того же фрагмента; `--redo` — всё заново (после правки правил).
   К модели идут несколько фрагментов сразу (workers): vLLM считает их
   одновременно, одной пачкой, и за то же время выходит в разы больше. С confirm каждый факт подтверждается
   вопросами (confirm.py) тут же, и в базу он ложится уже проверенным.

Ответ старого вида — объекты с «цитатой» — код тоже понимает и проверяет как
раньше: цитата должна найтись во фрагменте дословно.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..kb import KB
from ..kb import get as get_kb
from ..text import locate_quote, mentions, name_key, split_sentences
from . import store
from .synonyms import clean_relation

RULES = "graph-rules.txt"  # ближайший к базе по пути от корня (KB.file)
FRAGMENT_CHARS = 4000
MAX_NAME = 100
MAX_WORDS = 8
MAX_FACTS = 20
# На сколько предложений назад искать имя: «This area was selected…» — название
# района стоит предложением выше. На 120 фрагментах так нашлось 16 фактов из 341;
# двумя выше — ещё 6, но там уже сходятся имена, между которыми связи нет.
SENTENCES_BACK = 1
# Потолок ответа модели по фрагменту, токенов. Обычный ответ — 100–600; зациклившаяся
# модель без потолка пишет до конца окна и держит место, нужное другим фрагментам.
ANSWER_TOKENS = 3000
_NUMBER = re.compile(r"^[\d\s.,–—\-±~<>=%]+(гг?|г\.|лет|млн|ма|ga|ma)?\.?$", re.IGNORECASE)


def find_rules(path: Path | str | None = None, kb: KB | None = None) -> Path:
    """Файл правил: названный явно, иначе ближайший graph-rules.txt по пути базы."""
    if path:
        return Path(path)
    kb = kb or get_kb()
    found = kb.file(RULES)
    if found is None:
        raise FileNotFoundError(f"{RULES} нет ни у базы «{kb.path}», ни выше по дереву")
    return found


def load_rules(path: Path | str | None = None, kb: KB | None = None) -> str:
    """Правила из файла, без строк-комментариев. Нет файла — ошибка: без правил не строим."""
    path = find_rules(path, kb)
    text = path.read_text(encoding="utf-8-sig")
    rules = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    ).strip()
    if len(rules) < 100:
        raise ValueError(f"{path.name}: правил почти нет — граф без правил не строится")
    return rules


def sentences(text: str) -> list[str]:
    """Предложения фрагмента — те же и в том же порядке, что видит модель."""
    return split_sentences((text or "")[:FRAGMENT_CHARS])


def messages(rules: str, text: str) -> tuple[str, str]:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences(text), 1))
    system = f"{rules}\n\nОтвечай только JSON, без пояснений."
    user = (
        f'Фрагмент статьи, предложения пронумерованы:\n"""\n{numbered}\n"""\n\n'
        'Верни JSON: {"факты": [[п, "от", "связь", "к"], ...]}'
    )
    return system, user


@dataclass(frozen=True)
class Fact:
    src: str
    relation: str
    dst: str
    quote: str

    @property
    def src_key(self) -> str:
        return name_key(self.src)

    @property
    def dst_key(self) -> str:
        return name_key(self.dst)


@dataclass
class Checked:
    facts: list[Fact] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


# Количество, а не предмет: «около 80 % сульфидных минералов», «385 ± 2 млн лет»,
# «2 км», «порядка 17,6 млрд т запасов».
_QUANTITY = re.compile(
    r"^(?:около|порядка|примерно|более|менее|свыше|до|от|не\s+(?:более|менее)|"
    r"~|≈|>|<)?\s*[±]?\s*\d[\d.,]*(?:\s*[-–—±]\s*\d[\d.,]*)*(?=\s|%|°|$)",
    re.IGNORECASE,
)  # «3D-модель» — не количество
# Ссылка на работу, а не предмет: «Binotto (2015)», «Иванов и др., 2019», «Smith et al.».
_CITATION = re.compile(
    r"\(\s*(?:1[89]|20)\d{2}[a-zа-я]?\s*\)|\bet\s+al\b|" r"\bи\s+др\.?,?\s*(?:1[89]|20)\d{2}",
    re.IGNORECASE,
)
# Положение, а не предмет: «южнее Персияновского разлома в западной части района».
_POSITION = re.compile(
    r"^(?:южнее|севернее|западнее|восточнее|вдоль|вблизи|около|рядом\s+с|"
    r"между|выше|ниже|внутри|вне)\s",
    re.IGNORECASE,
)
# Ручная проверка 100 фактов (.local/logs/quality-20261005): каждый восьмой «предмет» —
# не предмет. Вторая проверка моделью их не ловила (5 из 24 ошибок при 10 верных
# фактах, отброшенных зря), правила кода — ловят.
# Местоимение вместо названия: «this event», «these structures», «это время».
_PRONOUN = re.compile(
    r"^(?:this|these|those|that|it|they|such|its|their|этот|эта|это|эти|данн\w+|"
    r"такие|такой|он|она|они)\b",
    re.IGNORECASE,
)
# Человек: «В. Н. Бобров», «Smith J.».
_PERSON = re.compile(
    r"^(?:[A-ZА-ЯЁ]\.\s?){1,2}\s?[A-ZА-ЯЁ][a-zа-яё]+$|^[A-ZА-ЯЁ][a-zа-яё]+,?\s(?:[A-ZА-ЯЁ]\.\s?){1,2}$"
)
# Год или десятилетие: «1950s», «1990-е годы».
_YEAR = re.compile(
    r"^(?:early\s+|late\s+|mid-?\s*)?\d{3,4}(?:-?[хxе]|'?s)?"
    r"(?:\s*(?:гг?\.?|годы|годах|year|years))?$",
    re.IGNORECASE,
)
# Счёт без названия: «three areas», «несколько участков».
_COUNTED = re.compile(
    r"^(?:one|two|three|four|five|six|several|many|some|various|other|другие|один|одна|"
    r"два|две|три|четыре|пять|несколько|многие|некоторые|различные)\s",
    re.IGNORECASE,
)
# Общее слово без названия: «месторождение» — какое?
_GENERIC = {
    "месторождение", "месторождения", "структура", "структуры", "район", "участок",
    "участки", "территория", "область", "зона", "зоны", "порода", "породы", "объект",
    "объекты", "изучаемая территория", "исследуемая территория", "area", "areas",
    "region", "deposit", "deposits", "structure", "structures", "zone", "zones", "rocks",
    "study area",
}  # fmt: skip
# Оценка из таблицы вместо предмета: «Barite = High», «Gold = Very high».
_DEGREE = {
    "high", "low", "very", "moderate", "medium", "strong", "weak", "major", "minor",
    "высокий", "высокая", "высокое", "низкий", "низкая", "низкое", "средний", "средняя",
    "среднее", "очень", "умеренный", "умеренная",
}  # fmt: skip
# Общее имя из одних общих слов: «this study», «predictive model», «high positive values»,
# «first phase», «data layer», «каждая последующая компонента». Главное слово ничего не
# называет, и определения при нём тоже общие. Слепая проверка 2026-10-10: из 52 фактов,
# помеченных «бесполезно», правило убирает 28; из 218 верных задевает 2 («третий этап»).
_GENERIC_HEADS = {
    "study", "studies", "research", "work", "works", "paper", "article", "analysis",
    "analyses", "approach", "approaches", "method", "methods", "methodology", "technique",
    "techniques", "model", "models", "data", "dataset", "datasets", "layer", "layers", "map",
    "maps", "figure", "figures", "image", "images", "value", "values", "result", "results",
    "phase", "phases", "stage", "stages", "observation", "observations", "investigation",
    "investigations", "validation", "fieldwork", "factor", "factors", "parameter",
    "parameters", "information", "tool", "tools", "framework", "technology", "technologies",
    "application", "applications", "interpretation", "interpretations", "minerals",
    "lithologies", "types", "type", "unit", "units", "feature", "features", "sample",
    "samples", "process", "processes", "activity", "mosaic", "evidence", "findings",
    "conditions", "properties", "patterns", "pattern", "domain", "domains", "level", "levels",
    "aspects", "variables", "variable", "class", "classes", "category", "categories",
    "component", "components", "set", "sets", "authors", "author", "researchers", "team",
    "project", "survey", "surveys", "test", "tests", "holes", "core", "cores",
    "исследование", "исследования", "работа", "работы", "статья", "анализ", "подход", "метод",
    "методы", "методика", "модель", "модели", "данные", "слой", "слои", "карта", "карты",
    "рисунок", "изображение", "изображения", "значение", "значения", "результат",
    "результаты", "фаза", "этап", "этапы", "стадия", "наблюдения", "изучение", "факторы",
    "фактор", "параметры", "параметр", "информация", "минералы", "типы", "тип", "компонента",
    "компоненты", "каналы", "канал", "керн", "образцы", "образец", "процесс", "процессы",
    "признаки", "свойства", "особенности", "авторы", "скважины",
}  # fmt: skip
_GENERIC_MODS = {
    "this", "these", "the", "a", "an", "our", "current", "present", "existing", "previous",
    "recent", "first", "second", "third", "fourth", "final", "initial", "preliminary", "new",
    "main", "key", "important", "other", "further", "various", "different", "same", "such",
    "several", "many", "predictive", "detected", "selected", "high", "low", "positive",
    "negative", "optical", "remote", "sensing", "statistical", "spatial", "structural",
    "geological", "geologic", "collected", "obtained", "proposed", "derived", "integrated",
    "combined", "available", "resulting", "generated", "produced", "used", "applied",
    "specific", "relevant", "certain", "most", "more", "less", "each", "every", "all", "some",
    "both", "of", "and", "in", "for", "drill",
    "данный", "данные", "данная", "это", "эти", "этот", "настоящий", "настоящая",
    "предыдущий", "первый", "первая", "второй", "вторая", "третий", "каждая", "каждый",
    "последующая", "последующий", "новый", "новые", "основной", "основные", "главный",
    "главные", "другие", "различные", "некоторые", "полученные", "выделенные", "высокие",
    "низкие", "исходные", "существующие", "собранные", "и", "в", "для", "по",
}  # fmt: skip
_NAME_WORD = re.compile(r"[a-zа-яё]+(?:-[a-zа-яё]+)?")
# Ссылка на работу по номеру: «работа [33]», «исследование [12, 14]».
_REFERENCE = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")
# Предположение в цитате, которого нет в связи: «The ores probably formed…» →
# «ores — образовано при — …». Связь получает пометку «предположительно».
_HEDGE_QUOTE = re.compile(
    r"вероятн|по-видимому|предположительн|возможн|не\s+исключен|считает|считают|"
    r"по\s+мнению|косвенн\w*\s+признак|probabl|likely|possibl|presumabl|"
    r"\b(?:is|are|was|were)\s+(?:thought|believed|inferred|presumed)|suggest|"
    r"may\s+(?:have|be|represent|indicate|reflect|serve|host|contain)|might",
    re.IGNORECASE,
)
_HEDGE_RELATION = re.compile(
    r"вероятн|возможн|предполож|может|могут|можно|интерпрет|считает|may|might|likely|"
    r"possib|probab|interpret|suggest|infer",
    re.IGNORECASE,
)
HEDGE = "предположительно"
# Предлог в начале имени модель тащит из текста («в аномалиях Bi») — он снимается.
_PREPOSITION = re.compile(
    r"^(?:в|во|на|к|ко|по|с|со|у|о|об|из|при|для|in|at|on|of)\s+(?=\w)", re.IGNORECASE
)


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\n", " ").split()).strip(" .,;:«»\"'")


def _clean_name(value: Any) -> str:
    return _PREPOSITION.sub("", _clean(value)).strip()


def _bad_name(name: str) -> str | None:
    if not name:
        return "пустое имя"
    if len(name) > MAX_NAME or len(name.split()) > MAX_WORDS:
        return "слишком длинное имя"
    if _NUMBER.match(name) or not name_key(name):
        return "число или год, а не предмет"
    if _QUANTITY.match(name):
        return "количество, а не предмет"
    if _CITATION.search(name) or _REFERENCE.search(name):
        return "ссылка на автора, а не предмет"
    if _POSITION.match(name):
        return "положение, а не предмет"
    if _PRONOUN.match(name):
        return "местоимение, а не предмет"
    if _PERSON.match(name):
        return "человек, а не предмет"
    if _YEAR.match(name):
        return "число или год, а не предмет"
    if _COUNTED.match(name):
        return "счёт без названия, а не предмет"
    if name.lower() in _GENERIC:
        return "общее слово без названия"
    if set(name.lower().replace("-", " ").split()) <= _DEGREE:
        return "оценка из таблицы, а не предмет"
    words = _NAME_WORD.findall(name.lower())
    if (
        words
        and not any(c.isdigit() for c in name)
        and words[-1] in _GENERIC_HEADS
        and all(w in _GENERIC_HEADS or w in _GENERIC_MODS for w in words)
    ):
        return "общее имя без названия"
    return None


def hedged(relation: str, quote: str) -> str:
    """Связь с пометкой «предположительно», если в цитате это предположение."""
    if _HEDGE_QUOTE.search(quote) and not _HEDGE_RELATION.search(relation):
        return f"{HEDGE} {relation}"
    return relation


def _sentence_number(value: Any) -> int | None:
    """«5», 5, «[5]», «5-6» → 5; не число — None."""
    match = re.match(r"\s*\[?\s*(\d+)", str(value if value is not None else ""))
    return int(match.group(1)) if match else None


def _quote_around(src: str, dst: str, n: int, sents: list[str]) -> str | None:
    """Цитата по номеру: предложение n, а если имени там нет — вместе с одним следующим
    или с предыдущим (SENTENCES_BACK; «This area…» — название района предложением выше)."""
    for start in range(n - 1, max(0, n - 1 - SENTENCES_BACK) - 1, -1):
        for end in (n, min(n + 1, len(sents))):
            quote = " ".join(sents[start:end])
            if mentions(src, quote) and mentions(dst, quote):
                return quote
    return None


def check(answer: Any, text: str) -> Checked:
    """Ответ модели → факты, прошедшие проверку, и причины отказа по остальным."""
    out = Checked()
    if not isinstance(answer, dict):
        out.rejected.append("ответ: не объект JSON")
        return out
    items = answer.get("факты") or answer.get("facts") or []
    if not isinstance(items, list):
        out.rejected.append("ответ: «факты» — не список")
        return out
    sents: list[str] | None = None
    seen = set()
    for item in items[: MAX_FACTS * 2]:
        number: int | None = None
        if isinstance(item, list) and len(item) >= 4:  # [п, от, связь, к]
            raw_n, raw_src, raw_rel, raw_dst, raw_quote = item[0], item[1], item[2], item[3], None
        elif isinstance(item, dict):
            raw_n, raw_quote = item.get("п"), item.get("цитата")
            raw_src, raw_rel, raw_dst = item.get("от"), item.get("связь"), item.get("к")
        else:
            out.rejected.append(f"{str(item)[:60]!r}: факт — не объект")
            continue
        src, dst = _clean_name(raw_src), _clean_name(raw_dst)
        relation = clean_relation(raw_rel)
        label = f"«{src or '?'}» — {relation or '?'} — «{dst or '?'}»"
        reason = _bad_name(src) or _bad_name(dst)
        if reason:
            out.rejected.append(f"{label}: {reason}")
            continue
        if not relation or len(relation.split()) > 6:
            out.rejected.append(f"{label}: связь пустая или длиннее 6 слов")
            continue
        if name_key(src) == name_key(dst):
            out.rejected.append(f"{label}: связь с самим собой")
            continue
        if raw_quote is not None and raw_n is None:  # ответ старого вида — с цитатой
            quote = locate_quote(_clean(raw_quote), text)
            if quote is None:
                out.rejected.append(f"{label}: цитаты нет во фрагменте")
                continue
            if not mentions(src, quote) or not mentions(dst, quote):
                missing = src if not mentions(src, quote) else dst
                out.rejected.append(f"{label}: в цитате нет «{missing}»")
                continue
        else:
            if sents is None:
                sents = sentences(text)
            number = _sentence_number(raw_n)
            if number is None or not 1 <= number <= len(sents):
                out.rejected.append(f"{label}: нет предложения {raw_n!s:.10}")
                continue
            found = _quote_around(src, dst, number, sents)
            if found is None:
                here = sents[number - 1]
                missing = src if not mentions(src, here) else dst
                out.rejected.append(f"{label}: в цитате нет «{missing}»")
                continue
            quote = found
        fact = Fact(src, hedged(relation, quote), dst, quote)
        key = (fact.src_key, relation, fact.dst_key)
        if key in seen:
            continue
        seen.add(key)
        out.facts.append(fact)
        if len(out.facts) >= MAX_FACTS:
            break
    return out


def run(
    conn: Any,
    llm: Any,
    limit: int | None = None,
    rules_path: Any = None,
    log: Any = print,
    stop_after_errors: int = 3,
    workers: int = 1,
    confirm: Any = None,
    kb: KB | None = None,
) -> dict[str, Any]:
    """Модель проходит фрагменты статей базы kb (и её поддерева), которых ещё не
    видела, — по правилам этой базы. Возобновляемо.

    workers — сколько фрагментов у модели одновременно: vLLM считает их разом,
    одной пачкой. В базу пишет только этот поток, по фрагменту.

    confirm(llm, факт) → число подтверждений (confirm.fact_votes): факт проверяется
    вопросами сразу, тем же работником, и ложится в базу уже с голосами. Без confirm
    голоса пустые — их потом ставит confirm.run."""
    kb = kb or get_kb()
    rules = load_rules(rules_path, kb)
    rows = store.todo(conn, limit=limit, kb=kb.path)
    model = getattr(llm, "model", "модель")
    stats = {"todo": len(rows), "done": 0, "facts": 0, "rejected": 0, "errors": 0, "stopped": False}

    def work(row: tuple[Any, ...]) -> tuple[Checked, list[int] | None]:
        text = row[2] or ""
        system, user = messages(rules, text)
        checked = check(llm.chat_json(system, user, max_tokens=ANSWER_TOKENS), text)
        votes = [int(confirm(llm, f)) for f in checked.facts] if confirm is not None else None
        return checked, votes

    queue = iter(rows)
    pending: dict[Future[Any], tuple[Any, ...]] = {}
    pool = ThreadPoolExecutor(max(1, workers))

    def submit() -> None:
        row = next(queue, None)
        if row is not None:
            pending[pool.submit(work, row)] = row

    errors_in_row = seen = 0
    started = time.monotonic()
    try:
        for _ in range(max(1, workers)):
            submit()
        while pending and not stats["stopped"]:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                chunk_id, doc_id, _text = pending.pop(future)
                seen += 1
                try:
                    checked, votes = future.result()
                    errors_in_row = 0
                except Exception as exc:  # noqa: BLE001 — модель молчит: этот фрагмент потом
                    stats["errors"] += 1
                    errors_in_row += 1
                    log(f"  [{seen}/{len(rows)}] модель не ответила: {type(exc).__name__}: {exc}")
                    if errors_in_row >= stop_after_errors:
                        stats["stopped"] = True
                        log(
                            "  Модель молчит несколько раз подряд — останавливаюсь. Проверьте "
                            "сервер модели (python kbrag.py status) и запустите снова: "
                            "продолжится с этого места."
                        )
                        break
                    submit()
                    continue
                store.save(conn, chunk_id, doc_id, checked.facts, checked.rejected, model, votes)
                conn.commit()
                stats["done"] += 1
                stats["facts"] += len(checked.facts)
                stats["rejected"] += len(checked.rejected)
                if seen % 10 == 0 or seen == len(rows):
                    pace = (time.monotonic() - started) / seen
                    log(
                        f"  [{seen}/{len(rows)}] фактов {stats['facts']}, отброшено проверкой "
                        f"{stats['rejected']} · {pace:.1f} с на фрагмент"
                    )
                submit()
    finally:
        # Остановка или Ctrl+C: недоделанные фрагменты не записаны — их пройдёт следующий запуск.
        pool.shutdown(wait=False, cancel_futures=True)
    return stats


def recheck(conn: Any, log: Any = print) -> dict[str, int]:
    """Перепроверить готовый граф правилами кода — без модели.

    Правила проверки меняются (новые виды «не предметов», пометка предположений),
    а прогонять модель заново по всем фрагментам — часы. Здесь каждый факт
    проверяется тем же кодом, что и новый: не прошёл — убирается с причиной,
    предположение — получает пометку в связи."""
    stats = {"facts": 0, "dropped": 0, "hedged": 0}
    for fact_id, chunk_id, src, relation, dst, quote in store.stored_facts(conn):
        stats["facts"] += 1
        reason = _bad_name(src) or _bad_name(dst)
        if reason:
            store.drop_fact(conn, fact_id, chunk_id, f"«{src}» — {relation} — «{dst}»: {reason}")
            stats["dropped"] += 1
            continue
        marked = hedged(relation, quote)
        if marked != relation and store.set_relation(conn, fact_id, marked):
            stats["hedged"] += 1
    conn.commit()
    if log:
        log(
            f"Фактов проверено {stats['facts']}: убрано {stats['dropped']}, "
            f"помечено предположением {stats['hedged']}"
        )
    return stats
