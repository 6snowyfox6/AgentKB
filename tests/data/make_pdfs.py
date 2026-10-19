"""Эталонные PDF для проверок — маленькие, выдуманные, каждый про один случай разбора.

Лежат рядом готовыми; этот скрипт — чтобы их можно было пересоздать или дополнить:

    python tests/data/make_pdfs.py [--font путь/к/шрифту.ttf]

Шрифт нужен с кириллицей и греческим (встроенные шрифты PDF их не знают). По
умолчанию ищется Arial (Windows) или DejaVu Sans (Linux); в PDF кладутся только
использованные знаки, поэтому файлы — единицы килобайт.

    plain_ru.pdf       русский текст без переносов, две страницы, заголовки
    hyphenated_ru.pdf  вёрстка с переносами: строки кончаются «сло-» + «во»
    scan.pdf           картинка без текстового слоя — скан
    formulas_ru.pdf    формулы с номерами «(1)», «(2)» и ссылки на них в тексте
    greek_en.pdf       английский текст с греческими буквами: σ-meson, ω-meson
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
_BFCHAR = re.compile(r"(?m)^(<[0-9A-Fa-f]{4}>\s+)(<00(?:ad|AD|a0|A0)>)$")
_PLAIN = {"<00ad>": "<002D>", "<00a0>": "<0020>"}
FONTS = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
]
SIZE = 11
LEAD = 15  # межстрочный интервал, пт
LEFT, TOP = 72, 80

PLAIN = [
    [
        "# Строение Северного блока",
        "Северный блок сложен гранитогнейсами раннего архея.",
        "Рудная минерализация приурочена к зонам дробления.",
        "Кварцевые жилы секут гнейсы под крутыми углами.",
        "Золото встречается в сульфидных прожилках.",
        "Мощность зон дробления достигает сорока метров.",
    ],
    [
        "# Методы",
        "Линеаментный анализ выполнен по снимкам Landsat-8.",
        "Гравиметрическая съёмка выявила локальный минимум.",
        "Плотность разломов рассчитана в окне два километра.",
    ],
]

# Перенос: строка кончается дефисом, следующая начинается строчной буквой.
HYPHENATED = [
    [
        "# Гидротермальные изменения",
        "Окварцевание и серицитизация развиты в преде-",
        "лах всей рудной зоны и сопровождаются суль-",
        "фидной вкрапленностью. Метасоматиты образу-",
        "ют линейные тела вдоль крупных разло-",
        "мов северо-западного простирания. Их мощ-",
        "ность меняется от первых метров до деся-",
        "тков метров, а границы с вмещающими поро-",
        "дами постепенные. Золото связано с пири-",
        "том и арсенопиритом.",
    ]
]

FORMULAS = [
    [
        "# Модель плотности",
        "Плотность пород выражается через пористость:",
        "ρ = ρ0 (1 − φ)                                          (1)",
        "где ρ0 — плотность минерального скелета, φ — пористость.",
        "Аномалия силы тяжести над телом радиуса R:",
        "Δg = 4/3 π G Δρ R³ / h²                               (2)",
        "Из формулы (2) видно, что аномалия растёт с контрастом",
        "плотности, а формула (1) связывает контраст с пористостью.",
    ]
]

GREEK = [
    [
        "# Meson exchange",
        "The nucleon-nucleon force is mediated by σ-meson exchange at medium range.",
        "The ω-meson produces the short-range repulsion between nucleons.",
        "Both mesons enter the relativistic mean field model.",
        "The coupling constant gσ is fitted to the saturation density.",
    ]
]


def _font() -> Path:
    for path in FONTS:
        if path.is_file():
            return path
    raise SystemExit("Нет шрифта с кириллицей: укажите его ключом --font")


def _fix_tounicode(doc: object) -> None:
    """У шрифтов дефис и мягкий перенос (пробел и неразрывный пробел) — один глиф, и
    PyMuPDF пишет в таблицу ToUnicode не тот знак: текст PDF читался бы «nucleon\\xadnucleon».
    В эталоне печатались только обычные дефисы и пробелы — их и возвращаем."""
    for xref in range(1, doc.xref_length()):  # type: ignore[attr-defined]
        if doc.xref_get_key(xref, "Type")[1] != "/Font":  # type: ignore[attr-defined]
            continue
        kind, ref = doc.xref_get_key(xref, "ToUnicode")  # type: ignore[attr-defined]
        if kind != "xref":
            continue
        target = int(ref.split()[0])
        cmap = doc.xref_stream(target).decode("latin-1")  # type: ignore[attr-defined]
        # Только знак, в который читается глиф (второе поле строки bfchar), — не номер
        # глифа: <00a0> бывает и номером другого глифа.
        fixed = _BFCHAR.sub(lambda m: m[1] + _PLAIN[m[2].lower()], cmap)
        if fixed != cmap:
            doc.update_stream(target, fixed.encode("latin-1"))  # type: ignore[attr-defined]


def _write(name: str, pages: list[list[str]], font: Path, title: str) -> None:
    import pymupdf

    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_font(fontname="F0", fontfile=str(font))
        y = TOP
        for line in lines:
            heading = line.startswith("# ")
            text = line[2:] if heading else line
            page.insert_text((LEFT, y), text, fontname="F0", fontsize=SIZE + 3 if heading else SIZE)
            y += LEAD + (6 if heading else 0)
    doc.set_metadata({"title": title, "creator": "kbrag tests/data/make_pdfs.py"})
    doc.subset_fonts()
    _fix_tounicode(doc)
    doc.save(HERE / name, garbage=4, deflate=True)
    doc.close()


def _scan(name: str) -> None:
    """Скан: страница — только картинка, текстового слоя нет."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 200, 120), False)
    pix.clear_with(200)
    page.insert_image(pymupdf.Rect(72, 72, 472, 312), pixmap=pix)
    doc.set_metadata({"title": "Скан", "creator": "kbrag tests/data/make_pdfs.py"})
    doc.save(HERE / name, garbage=4, deflate=True)
    doc.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font", type=Path, default=None)
    args = parser.parse_args()
    font = args.font or _font()
    _write("plain_ru.pdf", PLAIN, font, "Строение Северного блока")
    _write("hyphenated_ru.pdf", HYPHENATED, font, "Гидротермальные изменения")
    _write("formulas_ru.pdf", FORMULAS, font, "Модель плотности")
    _write("greek_en.pdf", GREEK, font, "Meson exchange")
    _scan("scan.pdf")
    for path in sorted(HERE.glob("*.pdf")):
        print(f"{path.name}: {path.stat().st_size} байт")


if __name__ == "__main__":
    main()
