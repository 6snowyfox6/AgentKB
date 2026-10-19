"""text.py: ключи имён графа, упоминание имени в цитате, фраза в любой форме, цитаты."""

from __future__ import annotations

import pytest

from kbrag.text import locate_quote, mentions, name_key, normalize, phrase_pattern, split_sentences


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Донецкий бассейн", "Донецкого бассейна"),
        ("донецкий  бассейн", "Донецкий бассейн"),
        ("Donetsk basins", "Donetsk basin"),
        ("рудный узел", "рудного узла"),
        ("разлом", "разломов"),
        ("Σ-мезон", "σ-мезон"),
        ("processes", "process"),
    ],
)
def test_same_key(a: str, b: str) -> None:
    assert name_key(a) == name_key(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("σ-мезон", "ω-мезон"),  # греческая буква — часть имени
        ("β-кварц", "α-кварц"),
        ("соль", "сель"),
        ("зона", "золото"),
    ],
)
def test_different_key(a: str, b: str) -> None:
    assert name_key(a) != name_key(b)


@pytest.mark.parametrize("name", ["石英脉", "金矿", "石英脈", "석영맥", "الكوارتز", "β-кварц"])
def test_key_not_empty_in_any_script(name: str) -> None:
    """Имя на любом языке даёт ключ: иначе факт отбрасывался как «число, а не предмет»."""
    assert name_key(name)


def test_key_keeps_numbers_and_apostrophe() -> None:
    assert name_key("O'Neil fault") == "o'neil fault"
    assert name_key("2020") == "2020"


@pytest.mark.parametrize(
    ("name", "text", "found"),
    [
        ("Анабарский щит", "в пределах Анабарского щита", True),
        ("σ-мезон", "обмен σ-мезоном между нуклонами", True),
        ("ω-мезон", "обмен σ-мезоном между нуклонами", False),
        ("магнетит", "титаномагнетит в породе", False),  # не часть слова
        ("щит", "для защиты", False),
        ("石英脉", "金矿化与石英脉密切相关。", True),  # письмо без пробелов
        ("金矿", "金矿化与石英脉密切相关。", True),
        ("석영맥", "금 광화작용은 석영맥은 관련이 있다", True),  # частица после слова
        ("كوارتز", "يرتبط الذهب بعروق الكوارتز", True),  # артикль перед словом
        ("Leimengou porphyry system", "the Leimengou porphyry Mo system", True),
        ("Онежский рудный район", "рудного района", False),  # имя собственное пропущено
    ],
)
def test_mentions(name: str, text: str, found: bool) -> None:
    assert mentions(name, text) is found


def test_phrase_pattern_any_form() -> None:
    pattern = phrase_pattern("рудный узел")
    assert pattern.search(normalize("Выделение рудных узлов по линеаментам"))
    assert not pattern.search(normalize("узел рудный"))


def test_split_sentences_keeps_abbreviations() -> None:
    """Инициалы, «и др.», «см. рис.» и «г.» предложение не рвут."""
    text = "Зона описана Иванов И.И. и др. в 2019 г. Она сложена гнейсами (см. рис. 2). Всё."
    assert split_sentences(text) == [
        "Зона описана Иванов И.И. и др. в 2019 г. Она сложена гнейсами (см. рис. 2).",
        "Всё.",
    ]


def test_locate_quote_returns_article_spelling() -> None:
    text = "Золото  связано с «кварцевыми жилами» — это видно на всех участках."
    quote = locate_quote('золото связано с "кварцевыми жилами" - это видно', text)
    assert quote == "Золото связано с «кварцевыми жилами» — это видно"
    assert locate_quote("серебро связано с кварцевыми жилами повсюду", text) is None
