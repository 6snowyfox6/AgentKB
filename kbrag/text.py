"""Работа с русским текстом — одна на весь проект.

Раньше окончания русских слов снимались в двух местах двумя разными списками
(проверка разбора и словарь понятий), беглая гласная («узел → узла») была только
в одном, а отбор статей брал внутренние функции проверки разбора. Теперь всё
здесь:

* normalize — строка для сравнения: без регистра, «ё» как «е», пробелы схлопнуты;
* ru_stem / stem_variants — основа слова и вариант с выпавшей гласной;
* phrase_pattern — шаблон фразы в любой форме: «Анабарский щит» находит
  «Анабарского щита», «рудный узел» — «рудного узла»;
* split_sentences — текст на предложения, сокращения и инициалы не рвут.
"""

from __future__ import annotations

import re
import unicodedata

# Окончания, которые снимаются с русского слова. Длинные первыми: «-иями»
# раньше «-ями», иначе останется лишняя «и».
RU_ENDINGS = tuple(
    sorted(
        {
            "ического",
            "ическому",
            "ическими",
            "ическая",
            "ические",
            "ический",
            "иями",
            "ями",
            "ами",
            "ией",
            "иях",
            "ием",
            "ого",
            "его",
            "ому",
            "ему",
            "ыми",
            "ими",
            "ов",
            "ев",
            "ей",
            "ой",
            "ий",
            "ый",
            "ая",
            "яя",
            "ое",
            "ее",
            "ые",
            "ие",
            "ых",
            "их",
            "ую",
            "юю",
            "ом",
            "ем",
            "ам",
            "ям",
            "ах",
            "ях",
            "ия",
            "ии",
            "ию",
            "ья",
            "ье",
            "ьи",
            "ью",
            "а",
            "я",
            "о",
            "е",
            "ы",
            "и",
            "у",
            "ю",
            "ь",
            "й",
        },
        key=len,
        reverse=True,
    )
)

_CONS = "бвгджзклмнпрстфхцчшщ"
# Письменности, где граница слова не видна: китайский, японский и тайский пишутся без
# пробелов, к корейскому и арабскому слову приклеиваются частицы («석영맥은», «الكوارتز»).
# Соседняя буква такой письменности слово не продолжает — иначе имя не найдётся в цитате.
_GLUED = (
    "\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff"  # арабское
    "\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff"  # тайское, лаосское, бирманское, кхмерское
    "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"  # кана, иероглифы
    "\u1100-\u11ff\u3130-\u318f\uac00-\ud7af"  # хангыль
)
# Буква, которая продолжает слово: \w любой письменности (греческой, латиницы с
# диакритикой…), кроме _GLUED.
_WORD_CHAR = rf"[^\W{_GLUED}]"
# Беглая гласная: узел → узла, песок → песка.
_FLEETING = re.compile(rf"^(.*[{_CONS}])[еоё]([{_CONS}])$")


def normalize(text: str) -> str:
    """Строка для сравнения: без регистра, «ё» как «е», пробелы схлопнуты."""
    text = unicodedata.normalize("NFKC", text or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def ru_stem(word: str, min_len: int = 3) -> str:
    """Основа русского слова: срезано типовое окончание. Латиница не трогается."""
    if not re.search(r"[а-яё]", word):
        return word
    for ending in RU_ENDINGS:
        if len(word) - len(ending) >= min_len and word.endswith(ending):
            return word[: -len(ending)]
    return word


def stem_variants(word: str, min_len: int = 3) -> list[str]:
    """Основа плюс вариант с выпавшей беглой гласной, длинные первыми."""
    variants = {ru_stem(word, min_len)}
    match = _FLEETING.match(word)
    if match:
        variants.add(match.group(1) + match.group(2))
    return sorted(variants, key=len, reverse=True)


def phrase_pattern(phrase: str, min_len: int | None = None) -> re.Pattern[str]:
    """Фраза в любой форме, по нормализованному тексту (см. normalize).

    Одиночный термин требует основы длиннее: его не подпирают соседние слова,
    и короткий корень начинает совпадать с чем попало. Исключение — короткие
    слова («зона», «руда»): при основе от четырёх букв у них не снимается
    окончание, и «зоны», «руды» не находились. Совпадение — только с начала
    слова: «щит» не находится в «защиты», «узел» — в «узелковые».
    """
    words = [w for w in re.split(r"[\s\-]+", normalize(phrase)) if w]
    single = len(words) == 1
    parts = []
    for w in words:
        n = min_len if min_len is not None else (4 if single and len(w) > 5 else 3)
        parts.append("(?:" + "|".join(re.escape(v) for v in stem_variants(w, n)) + r")\w*")
    return re.compile(rf"(?<!{_WORD_CHAR})" + r"[\s\-]+".join(parts))


# --------------------------------------------------------------------------- #
#  Предложения
# --------------------------------------------------------------------------- #
# Сокращения, после которых точка не кончает предложение.
_ABBREV = re.compile(
    r"(?:\b(?:т\.\s?е|т\.\s?к|т\.\s?н|и\s?др|и\s?т\.\s?д|и\s?т\.\s?п|см|рис|табл|гг?|вв?|"
    r"с|стр|им|ок|млн|млрд|тыс|кв|e\.g|i\.e|et\s?al|fig|figs|tab|vs|approx)"
    r"|\b[А-ЯЁA-Z])\.$",
    re.IGNORECASE,
)
_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[«\"(\[]?[А-ЯЁA-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Текст → предложения. Сокращения и инициалы предложение не рвут."""
    text = " ".join((text or "").split())
    if not text:
        return []
    sentences: list[str] = []
    for piece in _SPLIT.split(text):
        if sentences and _ABBREV.search(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return [s.strip() for s in sentences if len(s.strip()) >= 3]


# --------------------------------------------------------------------------- #
#  Цитаты и имена — для фактов графа
# --------------------------------------------------------------------------- #
_QUOTES = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "ё": "е",
        "Ё": "е",
    }
)


def _canon_with_map(text: str) -> tuple[str, list[int]]:
    """Текст для сравнения и карта: символ сравниваемого текста → место в исходном.

    Сравнение не различает регистр, «ё/е», виды кавычек и тире и пробелы —
    модель переписывает их как хочет. Карта нужна, чтобы вернуть цитату
    в том виде, в каком она стоит в статье, а не в пересказе модели.
    """
    out: list[str] = []
    where: list[int] = []
    last_space = True
    for i, ch in enumerate(text or ""):
        if ch == "\xad":  # мягкий перенос
            continue
        if ch.isspace():
            if not last_space:
                out.append(" ")
                where.append(i)
            last_space = True
            continue
        out.append(ch.translate(_QUOTES).lower())
        where.append(i)
        last_space = False
    return "".join(out), where


def locate_quote(quote: str, text: str) -> str | None:
    """Найти цитату во фрагменте; вернуть её в написании статьи или None.

    Модель иногда сокращает цитату многоточием — тогда каждая часть должна
    найтись в тексте по порядку, и возвращается кусок от первой до последней.
    """
    parts = [p.strip(" ,;:") for p in re.split(r"…|\.\.\.", quote or "")]
    parts = [p for p in parts if p]
    if not parts or sum(len(p) for p in parts) < 20:
        return None
    canon_text, where = _canon_with_map(text)
    start_at, first, last = 0, None, None
    for part in parts:
        canon_part = _canon_with_map(part)[0].strip()
        pos = canon_text.find(canon_part, start_at)
        if pos < 0:
            return None
        if first is None:
            first = pos
        last = pos + len(canon_part)
        start_at = last
    assert first is not None and last is not None  # цикл выше прошёл хотя бы раз
    begin = where[first]
    end = where[last - 1] + 1
    return " ".join(text[begin:end].split())


# Слово — \w любой письменности: греческая буква остаётся в ключе («σ-мезон» и
# «ω-мезон» — разные узлы), имя по-китайски или по-арабски не даёт пустой ключ.
_WORDS = re.compile(r"\w+(?:['’]\w+)?")
# Беглая гласная для ключа — только там, где она и бывает (узел, угол, песок,
# конец, ветер) и у основы от четырёх букв. Шире нельзя: «зона» давала «зн»,
# «метод» — «метд», а «соль» и «сель» сходились в одно «сл».
_KEY_FLEETING = re.compile(rf"^(?=.{{4}})(.+[{_CONS}])[еоё]([лкцнр])$")
# Существительное на -ом, -ов, -ем, -ев: у «разлом» это не окончание, но срезается
# как окончание, а у «разлома» — нет. Поэтому хвост срезается у любой основы:
# «разлом», «разлома», «разломов» — одно «разл»; «покров», «покрова» — «покр».
_KEY_TAIL = re.compile(r"(?<=...)(?:ом|ов|ем|ев)$")


def _key_word(word: str) -> str:
    if re.search(r"[а-я]", word):
        stem = _KEY_TAIL.sub("", ru_stem(word))
        match = _KEY_FLEETING.match(stem)
        return match.group(1) + match.group(2) if match else stem
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        if word.endswith(("sses", "xes", "ches", "shes")):
            return word[:-2]  # processes → process
        return word[:-1]  # basins → basin
    return word


def name_key(name: str) -> str:
    """Ключ имени: одно и то же в любом падеже и числе даёт один ключ.

    «Донецкий бассейн», «Донецкого бассейна», «донецкий  бассейн» → один
    ключ; «Donetsk basins» и «Donetsk basin» — тоже. Разные языки ключ не
    сводит — это делает словарь синонимов.
    """
    return " ".join(_key_word(w) for w in _WORDS.findall(normalize(name).replace("-", " ")))


def mentions(name: str, text: str) -> bool:
    """Упомянуто ли имя в тексте: целиком в любой форме — или почти все его
    значимые слова (модель пишет «Leimengou porphyry system», а в тексте
    «Leimengou porphyry Mo system»)."""
    name = (name or "").strip()
    if len(name) < 2:
        return False
    norm = normalize(text)
    # Целиком — но как отдельные слова: «магнетит» не упомянут в «титаномагнетит».
    whole = rf"(?<!{_WORD_CHAR})" + re.escape(normalize(name)) + rf"(?!{_WORD_CHAR})"
    if re.search(whole, norm):
        return True
    try:
        if phrase_pattern(name).search(norm):
            return True
    except re.error:
        pass
    original = re.findall(r"\w+", name)
    words = [(w, normalize(w)) for w in original if len(w) >= 3]
    if len(words) < 3:
        return False  # короткое имя — только целиком
    have = {_key_word(w) for w in _WORDS.findall(norm.replace("-", " "))}
    missing = [w for w, low in words if not any(h.startswith(_key_word(low)) for h in have)]
    # Можно пропустить одно общее слово, но не имя собственное и не число:
    # «Онежский рудный район» по тексту «рудного района» — не упоминание.
    return len(missing) <= 1 and not any(w[0].isupper() or w[0].isdigit() for w in missing)
