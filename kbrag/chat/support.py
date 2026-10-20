"""Подтверждает ли фрагмент утверждение, которое на него ссылается.

Ссылка на фрагмент проверялась только по номеру: есть ли такой фрагмент. А что он
говорит, никто не сверял — и в ответ попадало чужое: «Тяньшань» приписан статье про
Зимбабве, ограничение решёточной КХД — «расчётам eLSM с петлёй Полякова» (стресс-тест
20.10.2026). Здесь каждое утверждение со ссылкой сверяется с текстом своих фрагментов:

1. числа — кодом, без модели: число утверждения не в его фрагментах, а в другом — ссылка
   переставляется на тот; нет нигде — утверждение идёт к модели с пометкой;
2. смысл — одним вызовом модели на все утверждения: «да» (сказано во фрагменте, хотя бы
   пересказом или переводом), «частично», «нет» с названным «чего нет» (во фрагменте
   этого нет или сказано другое: другое место, объект, число). Отвергнутое
   перепроверяется по одному, только со своими фрагментами; «нет» оба раза —
   утверждение убирается из ответа и показывается отдельно.

Модель молчит или ответила не то — ответ остаётся как был: сверка не должна ломать ответ.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .. import acl
from .checks import CITE_MARK, about_sources, cited_numbers, is_claim, numbers, rewrite
from .messages import fragment_block
from .settings import Settings

MAX_CLAIMS = 24  # утверждений на проверку самое большее — остальные не трогаются

SUPPORT_SYSTEM = (
    "Ты проверяешь ответ по научным статьям: подтверждает ли фрагмент статьи утверждение, "
    "которое на него ссылается. Отвечай только JSON."
)
SUPPORT_USER = """Фрагменты статей:

{fragments}

Утверждения ответа, у каждого — номера фрагментов, на которые оно ссылается:

{claims}

Для каждого утверждения реши, подтверждают ли его ЕГО фрагменты:
- «да» — сказанное есть во фрагменте: дословно, пересказом, обобщением или в переводе с \
другого языка;
- «частично» — главное есть, но есть и добавка, которой во фрагменте нет (её — в «чего_нет»);
- «нет» — главного во фрагменте нет или сказано другое: другое место или объект, другое \
число, другая работа или модель, чем в утверждении.
Не придирайся к словам, пересказу и переводу. «нет» — только когда можешь назвать, чего \
именно во фрагменте нет: тогда напиши это в «чего_нет» (например, «во фрагменте Зимбабве, \
а не Тяньшань»). Не можешь назвать — значит, не «нет».

Верни JSON: {{{keys}}}"""


def _claims(answer: str) -> list[str]:
    found: list[str] = []

    def collect(part: str) -> str:
        # Фраза о самих фрагментах («во фрагментах нет точных значений [1]») — не факт
        # статьи: сверять её не с чем, а убиралась бы как неподтверждённая.
        if is_claim(part) and cited_numbers(part) and not about_sources(part):
            found.append(part)
        return part

    rewrite(answer, collect)
    return found


def _recite(claim: str, by_n: dict[int, dict[str, Any]]) -> str | None:
    """Числа утверждения не в его фрагментах, а все — в одном другом: ссылка на тот."""
    need = numbers(claim)
    if not need:
        return None
    cited = [by_n[n] for n in cited_numbers(claim) if n in by_n]
    have = set().union(*(numbers(s.get("text") or "") for s in cited)) if cited else set()
    if need <= have:
        return None
    other = next(
        (n for n, s in by_n.items() if s not in cited and need <= numbers(s.get("text") or "")),
        None,
    )
    if other is None:
        return None
    return (
        CITE_MARK.sub("", claim).rstrip(" .!?…")
        + f" [{other}]"
        + (claim.rstrip()[-1] if re.search(r"[.!?…]$", claim.rstrip()) else "")
    )


_PROPER = re.compile(r"(?<![\w.])[A-ZА-ЯЁ][\w-]{3,}")


def _stems(text: str, skip_first: bool = False) -> set[str]:
    words = _PROPER.findall(text)
    if skip_first and words and text.lstrip().startswith(words[0]):
        words = words[1:]  # первое слово предложения — с заглавной и не будучи именем
    return {w.lower()[:5] for w in words}


def _foreign(reason: str, claim: str, fragments: str) -> bool:
    """В «чего нет» — имя или число из самого утверждения, и в его фрагментах их правда
    нет: место, объект или величина, приписанные фрагменту. Это не добавка к верному.
    Имя, которое во фрагментах есть, — не чужое: «LEFA» модель называла «нет во
    фрагментах 1, 2…, кроме 3», а ссылка на 3 и стояла."""
    have = fragments.lower()
    if {n for n in numbers(reason) & numbers(claim) if n not in numbers(fragments)}:
        return True
    named = _stems(reason) & _stems(CITE_MARK.sub("", claim), skip_first=True)
    return any(stem not in have for stem in named)


def _refuted(verdict: Any, claim: str = "", fragments: str = "") -> bool:
    """Опровергнуто ли утверждение: «нет» с названным «чего нет» — или «частично», где
    «чего нет» — имя или число из утверждения, которых нет в его фрагментах. Без причины
    — не опровергнуто: голое «нет» стояло и у пересказа, который фрагмент прямо
    подтверждает, а «Тяньшань», приписанный статье про Зимбабве, модель называла
    «частично, нет: Тяньшань»."""
    if not isinstance(verdict, dict):
        return False
    answer = str(verdict.get("ответ") or "").strip().lower()
    reason = str(verdict.get("чего_нет") or "").strip()
    if len(reason) < 4:
        return False
    return answer == "нет" or (answer == "частично" and _foreign(reason, claim, fragments))


def _ask(claims: list[str], by_n: dict[int, dict[str, Any]], settings: Settings) -> list[bool]:
    """Модель о каждом утверждении: опровергнуто ли оно его фрагментами. Молчит — нет."""
    cited = sorted({n for c in claims for n in cited_numbers(c) if n in by_n})
    # Фрагмент — тот же, что видела модель при ответе (fragment_block режет до
    # FRAGMENT_CHARS): по первым 1800 знакам сверка убирала верное, что стояло дальше.
    blocks = "\n\n".join(fragment_block(by_n[n]) for n in cited)
    lines = [
        f"{i}) «{CITE_MARK.sub('', c).strip()}» — фрагменты {', '.join(map(str, cited_numbers(c)))}"
        for i, c in enumerate(claims, start=1)
    ]
    keys = ", ".join(
        f'"{i}": {{"ответ": "да|частично|нет", "чего_нет": ""}}' for i in range(1, len(claims) + 1)
    )
    try:
        data = settings.client().chat_json(
            SUPPORT_SYSTEM,
            SUPPORT_USER.format(fragments=blocks, claims="\n".join(lines), keys=keys),
            temperature=0,
            max_tokens=40 * len(claims) + 60,
        )
    except Exception:  # noqa: BLE001 — не вышло сверить: ничего не опровергнуто
        return [False] * len(claims)
    data = data if isinstance(data, dict) else {}
    return [
        _refuted(
            data.get(str(i)),
            c,
            " ".join(
                f"{by_n[n].get('title') or ''} {by_n[n].get('text') or ''}"
                for n in cited_numbers(c)
                if n in by_n
            ),
        )
        for i, c in enumerate(claims, start=1)
    ]


def check_support(
    question: str, answer: str, sources: list[dict[str, Any]], settings: Settings
) -> tuple[str, list[str], int]:
    """Ответ без неподтверждённых утверждений: (текст, убранные, сколько ссылок переставлено)."""
    by_n = {s["n"]: s for s in sources}
    claims = _claims(answer)[:MAX_CLAIMS]
    if not claims:
        return answer, [], 0

    fixed = {c: new for c in claims if (new := _recite(c, by_n)) is not None}
    checked = [fixed.get(c, c) for c in claims]
    first = _ask(checked, by_n, settings)
    # Отвергнутое — ещё раз, по одному, только со своими фрагментами: на ответах
    # стресс-теста общий проход отвергал и верное (около трети из «нет»), а узкая
    # перепроверка ошибается реже. Убирается то, что отвергли оба раза.
    doubtful = [i for i, no in enumerate(first) if no]
    again: list[bool] = []
    if doubtful:
        with ThreadPoolExecutor(min(4, len(doubtful))) as pool:
            again = [
                r[0] if r else False
                for r in pool.map(acl.carry(lambda i: _ask([checked[i]], by_n, settings)), doubtful)
            ]
    refuted = {claims[i] for i, no in zip(doubtful, again, strict=True) if no}
    verdicts = {c: c in refuted for c in claims}
    removed: list[str] = []

    def change(part: str) -> str | None:
        if part not in verdicts:
            return part
        if verdicts[part]:
            removed.append(fixed.get(part, part))
            return None
        return fixed.get(part, part)

    return rewrite(answer, change), removed, len(fixed)
