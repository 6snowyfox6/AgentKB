"""Чистка LaTeX от модели формул docling (parse/orchestrator/formulas.py)."""

from __future__ import annotations

from kbrag.parse.orchestrator.formulas import fix_formula, tidy_formula


def test_number_from_pdf_text() -> None:
    """Номер «(N)» — из текста PDF: модель его не видит, а текст ссылается на «(2)»."""
    fix = fix_formula(
        r"\Delta g = \frac{4}{3} \pi G \Delta\rho R^3 / h^2", "Δg = 4/3 π G Δρ R³ / h² (2)"
    )
    assert fix.number == "2"
    assert fix.text.endswith(r"\quad (2)")


def test_model_number_and_tail_removed() -> None:
    """Номер, дописанный моделью («\\quad ( 8 )»), и мусор после него уходят."""
    assert tidy_formula(r"a = b \quad \ \ ( 8 ) \quad p a r { x }", "") == "a = b"


def test_intertext_tail_removed() -> None:
    fix = fix_formula(r"E = m c^2 \\ \intertext { f l e t h r o s }", "E = mc2")
    assert fix.tail
    assert fix.text == "E = m c^2"


def test_alignment_dropped() -> None:
    assert tidy_formula(r"\rho & = \rho_0 (1 - \varphi) \\", "") == r"\rho = \rho_0 (1 - \varphi)"


def test_russian_symbols_get_pdf_text() -> None:
    """Русские индексы модель читает латиницей — к формуле приписывается текст PDF."""
    fix = fix_formula(r"K_{n} = V_{nop} / V_{o6p}", "K п = V пор / V обр (3)")
    assert fix.russian
    assert r"\text{[K п = V пор / V обр]}" in fix.text
    assert fix.text.endswith(r"\quad (3)")


def test_second_pass_does_not_double() -> None:
    """Повторная чистка (перечанкинг) не дописывает номер и текст второй раз."""
    once = tidy_formula(r"x = y", "x = y (4)")
    assert tidy_formula(once, "x = y (4)") == once


def test_not_a_number_after_letter() -> None:
    """«f(2)» — вызов функции, а не номер формулы."""
    assert fix_formula(r"f(2) = 1", "f(2) = 1").number == ""


def test_empty() -> None:
    assert tidy_formula("", "") == ""
