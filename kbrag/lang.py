"""Язык текста — у каждого фрагмента базы, при загрузке (index/ingest.py).

Без внешних библиотек и моделей: фрагмент статьи — сотни слов, и этого хватает.

* Письменность — по буквам: китайский, японский (кана), корейский, арабский, иврит,
  греческий, деванагари, тайский, кириллица, латиница.
* Внутри кириллицы и латиницы — по самым частым служебным словам языка («и», «в», «не»
  — русский; «the», «of», «and» — английский; «der», «und» — немецкий…) и по буквам,
  которые есть только в одном языке (і, ї, є — украинский; ß, ä — немецкий…).

Ответ — код ISO 639-1 («ru», «en», «zh»…) или пустая строка, если текста мало или он
не похож на речь (таблица чисел, формула): тогда фрагмент берёт язык статьи.
"""

from __future__ import annotations

import re
from collections import Counter

MIN_LETTERS = 20  # меньше букв — язык не определяется
MIN_WORDS = 3  # служебных слов языка, чтобы назвать его

_SCRIPTS = {
    "hira_kata": re.compile(r"[぀-ヿ]"),
    "han": re.compile(r"[㐀-䶿一-鿿]"),
    "hangul": re.compile(r"[ᄀ-ᇿ㄰-㆏가-힯]"),
    "arabic": re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]"),
    "hebrew": re.compile(r"[֐-׿]"),
    "greek": re.compile(r"[Ͱ-Ͽἀ-῿]"),
    "devanagari": re.compile(r"[ऀ-ॿ]"),
    "thai": re.compile(r"[฀-๿]"),
    "cyrillic": re.compile(r"[Ѐ-ӿ]"),
    "latin": re.compile(r"[A-Za-zÀ-ɏ]"),
}
_PERSIAN = re.compile(r"[پچژگکی]")  # буквы персидского, которых нет в арабском

# Самые частые служебные слова: по ним язык узнаётся и в научном тексте, где почти всё
# остальное — термины. Слова, общие для нескольких языков («a», «de», «in»), не взяты.
_WORDS = {
    "ru": "и в не на что с по как это из для от к при так же или но был были его её их "
    "который которые также может более между",
    "uk": "і та що не на з як це від для до при або але був були його також може між",
    "bg": "и в не на че с по като това от за при или но беше са",
    "sr": "и у не на да је се са од за као или али су",
    "en": "the of and to is that for with as are this by from be which was were it "
    "these have has been their its can also than",
    "de": "der die und das ist nicht mit den von zu ein eine dem des auf sich für im "
    "wird werden auch als bei durch",
    "fr": "le la les et des est une du que pour dans qui sur par pas au sont ces avec "
    "être été aux leur",
    "es": "el la los las y del que es por una para con se no al como más sus son entre "
    "también fue han",
    "it": "il la di che è e per un una del della con sono non gli dei delle nel anche "
    "come più tra",
    "pt": "o os as e do da que é para um uma com não dos das no na são pelo pela também "
    "entre foi",
    "nl": "de het een en van is dat op te voor met zijn niet aan ook bij door wordt " "worden deze",
    "pl": "i w nie na się z że do jest to od dla są przez jak oraz także który może",
    "cs": "a v se na je že s z do to od pro jsou jako také který může při",
    "sv": "och i att det som en är av för med till på den har inte om ett de",
    "tr": "ve bir bu da için ile olarak olan daha gibi en çok ya kadar sonra",
}
_STOP = {lang: set(words.split()) for lang, words in _WORDS.items()}
# Буквы одного языка — сами по себе почти доказательство.
_LETTERS = {
    "uk": re.compile(r"[іїєґ]"),
    "de": re.compile(r"[ßäöü]"),
    "fr": re.compile(r"[çœàèùâêîôûë]"),
    "es": re.compile(r"[ñ¿¡]"),
    "pt": re.compile(r"[ãõ]"),
    "pl": re.compile(r"[ąćęłńśźż]"),
    "cs": re.compile(r"[ěřůčšž]"),
    "tr": re.compile(r"[ğış]"),
    "sr": re.compile(r"[ђјљњћџ]"),
}
_TOKEN = re.compile(r"[^\W\d_]+")
CYRILLIC_LANGS = ("ru", "uk", "bg", "sr")


def script_counts(text: str) -> Counter[str]:
    return Counter({name: len(rx.findall(text)) for name, rx in _SCRIPTS.items()})


def detect(text: str) -> str:
    """Язык текста (код ISO 639-1) или "", если не понять."""
    text = text or ""
    counts = script_counts(text)
    letters = sum(counts.values())
    if letters < MIN_LETTERS:
        return ""
    if counts["hangul"] >= 0.3 * letters:
        return "ko"
    if counts["hira_kata"] >= 0.1 * letters:
        return "ja"  # в японском тексте кана есть всегда, иероглифы — тоже
    if counts["han"] >= 0.3 * letters:
        return "zh"
    script, top = counts.most_common(1)[0]
    if top < 0.5 * letters:
        return ""  # смесь письменностей — не речь, а формулы и подписи
    if script == "arabic":
        return "fa" if _PERSIAN.search(text) else "ar"
    simple = {"hebrew": "he", "greek": "el", "devanagari": "hi", "thai": "th"}
    if script in simple:
        return simple[script]
    candidates = (
        CYRILLIC_LANGS
        if script == "cyrillic"
        else tuple(lang for lang in _STOP if lang not in CYRILLIC_LANGS)
    )
    return _by_words(text, candidates)


def _by_words(text: str, candidates: tuple[str, ...]) -> str:
    low = text.lower()
    words = Counter(_TOKEN.findall(low))
    score: dict[str, float] = {}
    for lang in candidates:
        hits = sum(n for w, n in words.items() if w in _STOP[lang])
        letters = _LETTERS.get(lang)
        bonus = min(5, len(letters.findall(low))) if letters else 0
        score[lang] = hits + bonus
    best = max(score, key=lambda lang: score[lang])
    if score[best] < MIN_WORDS:
        # Кириллица без служебных слов (список терминов, подпись) — всё равно скорее
        # русский: других языков на кириллице в базе почти нет.
        return "ru" if candidates == CYRILLIC_LANGS else ""
    return best


def majority(languages: list[str]) -> str:
    """Язык статьи — у большинства её фрагментов, где язык понятен."""
    known = Counter(lang for lang in languages if lang)
    return known.most_common(1)[0][0] if known else ""
