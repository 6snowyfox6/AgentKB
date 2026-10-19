"""Язык фрагмента (lang.py)."""

from __future__ import annotations

import pytest

from kbrag.lang import detect, majority

SAMPLES = {
    "ru": "Рудная минерализация приурочена к зонам дробления, которые также контролируют "
    "положение кварцевых жил и сульфидной вкрапленности в пределах Северного блока.",
    "uk": "Рудна мінералізація пов'язана із зонами дроблення, які також контролюють "
    "положення кварцових жил та сульфідної вкрапленості.",
    "en": "The ore mineralization is related to the fracture zones, which also control "
    "the position of quartz veins and sulfide dissemination in the northern block.",
    "de": "Die Vererzung ist an die Bruchzonen gebunden, die auch die Lage der Quarzgänge "
    "und der Sulfidimprägnation im nördlichen Block bestimmen.",
    "fr": "La minéralisation est liée aux zones de fracturation qui contrôlent aussi la "
    "position des filons de quartz et de la dissémination des sulfures dans le bloc nord.",
    "es": "La mineralización está relacionada con las zonas de fractura, que también "
    "controlan la posición de las vetas de cuarzo y de los sulfuros en el bloque norte.",
    "zh": "金矿化与石英脉密切相关，主要分布在破碎带中，并伴有黄铁矿和毒砂。",
    "ja": "金の鉱化作用は石英脈と密接に関連しており、主に破砕帯に分布している。",
    "ko": "금 광화작용은 석영맥과 밀접한 관련이 있으며 주로 파쇄대에 분포한다.",
    "ar": "يرتبط تمعدن الذهب ارتباطا وثيقا بعروق الكوارتز ويتوزع أساسا في مناطق التكسر.",
}


@pytest.mark.parametrize("lang", sorted(SAMPLES))
def test_detect(lang: str) -> None:
    assert detect(SAMPLES[lang]) == lang


@pytest.mark.parametrize(
    "text",
    ["", "12,5 13,8 14,1 15,0 16,2", r"\rho = \rho_0 (1 - \varphi) \quad (1)", "Fig. 3"],
)
def test_unknown(text: str) -> None:
    """Мало букв или не речь — пусто: фрагмент возьмёт язык статьи."""
    assert detect(text) == ""


def test_majority() -> None:
    assert majority(["ru", "", "en", "ru"]) == "ru"
    assert majority(["", ""]) == ""
