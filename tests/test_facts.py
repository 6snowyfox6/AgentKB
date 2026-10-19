"""Проверка фактов графа кодом (graph/facts.py): что модель выписала, то и в цитате."""

from __future__ import annotations

import pytest

from kbrag.graph.facts import _bad_name, check

TEXT = (
    "Билляхская зона входит в состав Анабарского щита. "
    "Золото связано с кварцевыми жилами в зонах дробления. "
    "Метод главных компонент применён к снимкам ASTER. "
    "Обмен σ-мезоном даёт притяжение, а ω-мезон отвечает за отталкивание."
)


def test_fact_in_its_sentence_passes() -> None:
    got = check({"факты": [[2, "Золото", "связано с", "кварцевые жилы"]]}, TEXT)
    assert [(f.src, f.relation, f.dst) for f in got.facts] == [
        ("Золото", "связано с", "кварцевые жилы")
    ]
    assert got.facts[0].quote.startswith("Золото связано с кварцевыми жилами")


def test_next_sentence_allowed() -> None:
    """Факт в двух соседних предложениях — номер первого (правило 4 для модели)."""
    got = check({"факты": [[1, "Золото", "связано с", "кварцевые жилы"]]}, TEXT)
    assert len(got.facts) == 1


def test_wrong_sentence_rejected() -> None:
    """Ни в предложении 4, ни рядом с ним (на одно назад и вперёд) золота нет."""
    got = check({"факты": [[4, "Золото", "связано с", "кварцевые жилы"]]}, TEXT)
    assert not got.facts
    assert "в цитате нет" in got.rejected[0]


def test_sentence_out_of_range_rejected() -> None:
    got = check({"факты": [[9, "Золото", "связано с", "кварцевые жилы"]]}, TEXT)
    assert not got.facts and "нет предложения" in got.rejected[0]


def test_greek_names_stay_different() -> None:
    got = check({"факты": [[4, "σ-мезон", "связано с", "ω-мезон"]]}, TEXT)
    assert len(got.facts) == 1  # не «связь с самим собой»


def test_self_link_rejected() -> None:
    got = check({"факты": [[1, "Билляхская зона", "входит в", "Билляхской зоны"]]}, TEXT)
    assert not got.facts and "самим собой" in got.rejected[0]


def test_duplicates_once() -> None:
    fact = [2, "Золото", "связано с", "кварцевые жилы"]
    assert len(check({"факты": [fact, list(fact)]}, TEXT).facts) == 1


def test_not_json_object() -> None:
    assert check(["факты"], TEXT).rejected == ["ответ: не объект JSON"]


@pytest.mark.parametrize(
    "name",
    ["2020", "385 ± 2 млн лет", "около 80 % сульфидов", "Binotto (2015)", "Иванов и др., 2019", ""],
)
def test_not_a_subject(name: str) -> None:
    assert _bad_name(name)


@pytest.mark.parametrize("name", ["Анабарский щит", "3D-модель", "石英脉", "σ-мезон"])
def test_subject(name: str) -> None:
    assert _bad_name(name) is None
