"""Запрос поиска по словам (index/search.py, build_tsquery): русская форма слова в обе
стороны. Стеммер Postgres беглую гласную не знает: «узел» → основа «узел», «узла» →
«узл», — поэтому запрос несёт оба написания, от какой бы формы ни шёл."""

from __future__ import annotations

import re

import pytest

from kbrag.index.search import build_tsquery


@pytest.mark.parametrize(
    ("query", "must"),
    [
        ("рудного узла", {"узла", "узел"}),  # косвенная → начальная: находит «Рудный узел»
        ("рудный узел", {"узел", "узл"}),  # начальная → косвенная: находит «рудного узла»
        ("песка", {"песок"}),
        ("узлы", {"узел"}),
        ("конца", {"конец"}),
        ("угла", {"угол"}),
        ("разлом", {"разлома"}),  # -ом не окончание: вариант с «-а»
    ],
)
def test_both_directions(query: str, must: set[str]) -> None:
    assert must <= set(re.findall(r"\w+", build_tsquery(query)))


@pytest.mark.parametrize("word", ["метода", "района", "разлома", "окварцевания", "rocks"])
def test_no_vowel_where_none_drops(word: str) -> None:
    """Основа не на «согласная + л, к, ц, н, р» — вариантов нет."""
    assert build_tsquery(word) == word.lower()


def test_all_and_any_words() -> None:
    assert build_tsquery("рудного узла") == "рудного & (узла | узел | узол)"
    assert build_tsquery("рудного узла", any_word=True) == "рудного | (узла | узел | узол)"
