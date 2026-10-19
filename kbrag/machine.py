"""Настройки этой машины: окно модели, сколько фрагментов давать ей, видеопамять.

Числа по умолчанию подобраны под видеокарту на 16 ГБ (RTX 5070 Ti, Qwen3-14B в 4 бита).
На машине мощнее их можно поднять — одним файлом `.local/settings.env` (в git не идёт):

    KBRAG_NUM_CTX=32768      окно модели вместе с ответом, токенов (vLLM --max-model-len)
    KBRAG_TOP_K=12           сколько фрагментов чат-бот даёт модели для ответа
    KBRAG_PER_DOC=3          из них — не больше стольких из одной статьи
    KBRAG_GPU_MEMORY=0.90    доля видеопамяти для vLLM (--gpu-memory-utilization)
    KBRAG_MAX_SEQS=8         сколько запросов vLLM считает одной пачкой (--max-num-seqs)
    KBRAG_BATCHED_TOKENS=2048   сколько токенов задания читать за шаг (--max-num-batched-tokens)

Тот же файл читает docker compose (python kbrag.py start передаёт его ключом
--env-file): окно в коде и окно сервера модели не разойдутся. Переменная окружения
главнее файла. После правки — python kbrag.py stop и start: сервер модели берёт
окно при запуске.

Только стандартная библиотека: kbrag.py читает это до перехода в .venv.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .paths import LOCAL

FILE = LOCAL / "settings.env"
_LINE = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$")


def read(path: Path = FILE) -> dict[str, str]:
    """Пары ИМЯ=значение из файла; строки с # и пустые пропускаются, кавычки снимаются."""
    out: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return out
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        found = _LINE.match(line.split(" #", 1)[0])
        if found:
            out[found.group(1)] = found.group(2).strip("'\"")
    return out


def value(name: str, default: str, path: Path = FILE) -> str:
    return os.environ.get(name, "").strip() or read(path).get(name, "").strip() or default


def integer(name: str, default: int, low: int = 1, path: Path = FILE) -> int:
    """Целое из настроек; не число или меньше low — значение по умолчанию."""
    try:
        number = int(value(name, str(default), path))
    except ValueError:
        return default
    return number if number >= low else default
