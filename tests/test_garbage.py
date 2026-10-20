"""Доля мусора (validate.garbage_ratio): статья на любом письме — не мусор, битый шрифт — мусор."""

from __future__ import annotations

import pytest

from kbrag.parse.validate import garbage_ratio, stray_cjk

CLEAN = [
    "Линеаменты выделены по снимкам Landsat 8, угол падения 45° ± 5°.",
    "断层形成于海西期。后者在地表有较大的弯曲变形，断面倾角小。",
    "日本列島は植生が多く,浸食作用が卓越し,地質学的に変動帯である。",
    "Tự động tách chiết các yếu tố dạng tuyến từ ảnh viễn thám.",
    "การสกัดแนวเส้นจากข้อมูลการรับรู้ระยะไกล",
    "გად იშიფრებიან აერო- და კოსმოსურ სურათებზე",
    "Intégração de dados de sensoriamento remoto — géologie structurale.",
]

BROKEN = [
    "(cid:34)(cid:12)(cid:90)(cid:7) (cid:55)(cid:3)(cid:44)",
    "  текст ",
    "Ð»Ð¸Ð½ÐµÐ°Ð¼ÐµÐ½Ñ‚Ñ‹",  # UTF-8, прочитанный как Latin-1
    "■□■□▪▫■□ ◆◇◆◇ �����",
]


@pytest.mark.parametrize("text", CLEAN)
def test_any_script_is_not_garbage(text: str) -> None:
    assert garbage_ratio(text) < 0.05


@pytest.mark.parametrize("text", BROKEN)
def test_broken_fonts_are_garbage(text: str) -> None:
    assert garbage_ratio(text) > 0.3


BROKEN_LATIN = (
    "The result of 血isstudy aim to set up a complete e紅白quake hazard map. The selected "
    "ぽeais located in the Taungtonelone area, northern My町田1低 Myanmar is an earthquake-prone "
    "country. It lies in a major earthquake zone of the world called Mediterranean belt."
)


def test_cjk_inside_latin_words_is_broken_font() -> None:
    assert stray_cjk(BROKEN_LATIN) > 0.05
    assert garbage_ratio(BROKEN_LATIN) < 0.10  # доля мусора такого не видит


@pytest.mark.parametrize(
    "text",
    [
        "本研究ではLandsat TM画像を用いてLineamentを抽出し、断層との対応を検討した。",
        "利用ASTER比值和Sobel滤波等增强显示断裂构造，并利用Canny边缘检测与人工解译相结合。",
        "Lineament density was mapped from SRTM DEM data in the Atlas Mountains of Morocco.",
    ],
)
def test_cjk_texts_pass(text: str) -> None:
    assert stray_cjk(text) == 0
