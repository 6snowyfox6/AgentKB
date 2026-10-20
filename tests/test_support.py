"""Сверка ссылок ответа с фрагментами (chat/support.py) и спасение фраз без ссылки по
числам (checks.cite_by_numbers) — без модели: её ответ подставляется."""

from __future__ import annotations

import re
from typing import Any

import pytest

from kbrag.chat import support
from kbrag.chat.checks import cite_by_numbers, drop_uncited, numbers
from kbrag.chat.settings import Settings

SOURCES = [
    {
        "n": 1,
        "title": "Bulawayo",
        "doc_id": "a",
        "year": 2013,
        "text": "Groundwater in the crystalline basement of Bulawayo, Zimbabwe, is controlled "
        "by lineaments. Lineament density exceeds 2.5 km per km2 near the wells.",
    },
    {
        "n": 2,
        "title": "Patuha",
        "doc_id": "b",
        "year": 2022,
        "text": "Radon concentration at TTN-07 reached a maximum of 5343.33 Bq/m3, mean 3612.08.",
    },
    {
        "n": 3,
        "title": "Yili",
        "doc_id": "c",
        "year": 1997,
        "text": "1:50 000 high-precision aeromagnetic survey; anomalies of 50-500 nT.",
    },
]


def test_numbers_normalized() -> None:
    assert numbers("максимум 5343,33 Бк, масштаб 1:50 000, 50–500 нТл [3]") == {
        "5343.33",
        "50000",
        "50",
        "500",
    }
    assert numbers("в 3 точках [1]") == set()  # одиночные цифры и номера ссылок — не в счёт


def test_uncited_fact_with_numbers_gets_citation() -> None:
    answer = (
        "Корреляция радона с плотностью линеаментов изучена на поле Патуха [2].\n"
        "Максимальная концентрация радона в точке TTN-07 составила 5343,33 Бк/м³.\n"
        "Линеаменты вообще часто бывают связаны с выходами газов из недр земли."
    )
    text, rescued = cite_by_numbers(answer, SOURCES)
    assert rescued == ["Максимальная концентрация радона в точке TTN-07 составила 5343,33 Бк/м³."]
    assert "5343,33 Бк/м³ [2]." in text
    text, removed = drop_uncited(text)
    assert "5343,33" in text and removed == [
        "Линеаменты вообще часто бывают связаны с выходами газов из недр земли."
    ]


class Client:
    """Модель-проверщик: вердикт — по словам утверждения (judge), а не по его номеру:
    при перепроверке по одному у утверждения номер 1. second — вердикт перепроверки,
    если он другой."""

    def __init__(self, judge: Any, second: Any = None) -> None:
        self.judge, self.second = judge, second
        self.asked: list[str] = []

    def chat_json(self, system: str, user: str, **kw: Any) -> dict[str, Any]:
        self.asked.append(user)
        if self.judge is None:
            raise RuntimeError("модель молчит")
        claims = re.findall(r"(?m)^(\d+)\) «(.+?)»", user)
        judge = (
            self.second if self.second and len(claims) == 1 and len(self.asked) > 1 else self.judge
        )
        return {n: judge(text) for n, text in claims}


def run(monkeypatch: pytest.MonkeyPatch, answer: str, judge: Any, second: Any = None):
    client = Client(judge, second)
    monkeypatch.setattr(Settings, "client", lambda self: client)
    return support.check_support("?", answer, SOURCES, Settings()), client


ANSWER = (
    "**Подземные воды**\n"
    "В районе Булавайо (Зимбабве) подземные воды фундамента контролируются линеаментами [1].\n"
    "В районе Тяньшаня анализ линеаментов дал сведения о движении подземных вод [1]."
)


YES = {"ответ": "да", "чего_нет": ""}
NO = {"ответ": "нет", "чего_нет": "во фрагменте Зимбабве, а не Тяньшань"}


def tianshan(verdict: Any) -> Any:
    """Отвергает утверждение про Тяньшань, остальное подтверждает."""
    return lambda claim: verdict if "Тяньшан" in claim else YES


def test_unsupported_claim_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    (text, removed, recited), client = run(monkeypatch, ANSWER, tianshan(NO))
    assert "Булавайо" in text and "Тяньшан" not in text
    assert removed == [ANSWER.splitlines()[2]] and recited == 0
    assert "Bulawayo" in client.asked[0]  # модель видит текст фрагмента
    assert '"1": {"ответ"' in client.asked[0]  # ключи JSON — с одинарными скобками
    assert len(client.asked) == 2
    assert "Тяньшан" in client.asked[1] and "Булавайо" not in client.asked[1]  # по одному


def test_partial_with_foreign_name_is_refuted(monkeypatch: pytest.MonkeyPatch) -> None:
    """«Частично», а нет во фрагменте — места из утверждения: приписано чужое, убрать.
    «Частично» с добавкой общими словами — оставить."""
    foreign = {"ответ": "частично", "чего_нет": "Тяньшань"}
    (text, removed, _), _ = run(monkeypatch, ANSWER, tianshan(foreign))
    assert "Тяньшан" not in text and len(removed) == 1
    extra = {"ответ": "частично", "чего_нет": "сведения о движении вод"}
    (text, removed, _), _ = run(monkeypatch, ANSWER, tianshan(extra))
    assert text == ANSWER and removed == []
    # Имя, которое в процитированном фрагменте есть, — не чужое.
    answer = "В статье о Patuha изучена связь радона с плотностью линеаментов [2]."
    named = {"ответ": "частично", "чего_нет": "Patuha упомянута не во всех фрагментах"}
    (text, removed, _), _ = run(monkeypatch, answer, lambda c: named)
    assert text == answer and removed == []


def test_second_look_saves_claim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Общий проход отверг, перепроверка по одному — нет: утверждение остаётся."""
    (text, removed, _), _ = run(monkeypatch, ANSWER, tianshan(NO), second=lambda c: YES)
    assert text == ANSWER and removed == []


def test_doubt_keeps_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """«Частично», «нет» без названной причины, строка вместо разбора, молчащая модель —
    ответ не трогается."""
    bare_no = {"ответ": "нет", "чего_нет": ""}
    for judge in (tianshan({"ответ": "частично"}), tianshan(bare_no), tianshan("нет"), None):
        (text, removed, _), _ = run(monkeypatch, ANSWER, judge)
        assert text == ANSWER and removed == []


def test_remarks_about_sources_not_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = "Во фрагментах нет точных значений точности выделения линеаментов в процентах [1]."
    (text, removed, _), client = run(monkeypatch, answer, lambda c: NO)
    assert text == answer and removed == [] and not client.asked


def test_number_moves_citation_to_its_fragment(monkeypatch: pytest.MonkeyPatch) -> None:
    answer = "Аэромагнитная съёмка Илийского бассейна выполнена в масштабе 1:50 000 [1]."
    (text, removed, recited), client = run(monkeypatch, answer, lambda c: YES)
    assert text == "Аэромагнитная съёмка Илийского бассейна выполнена в масштабе 1:50 000 [3]."
    assert recited == 1 and removed == []
    assert "фрагменты 3" in client.asked[0]  # проверяется уже с правильной ссылкой
