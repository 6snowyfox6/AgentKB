"""Кому бот отвечает, токен, выбор базы по чатам и лимит гостя открытого бота."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..kb import KBError
from ..paths import LOCAL
from . import ui

TOKEN_FILE = LOCAL / "telegram_token.txt"
USERS_FILE = LOCAL / "telegram_users.txt"
KB_FILE = LOCAL / "telegram_kb.json"  # какую базу выбрал каждый чат (/kb)

OPEN_ACCESS = {
    "*",
    "все",
    "всем",
    "all",
    "everyone",
}  # строка в .local/telegram_users.txt: открыт всем

GUEST_WINDOW = 3600.0  # окно лимита гостя, с


def guest_limit() -> int:
    """Сколько вопросов в час гостю открытого бота («*»): KBRAG_TG_GUEST_LIMIT в
    .local/settings.env, по умолчанию 20; 0 — без ограничения. Владельцам — без него."""
    from .. import machine

    return machine.integer("KBRAG_TG_GUEST_LIMIT", 20, low=0)


def read_choice(path: Path = KB_FILE) -> dict[int, str]:
    """Выбор базы по чатам: {чат: путь}. Нет файла или не читается — пусто (все — по
    умолчанию); путь, которому не место в дереве, отбрасывается."""
    from ..kb import check_path

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[int, str] = {}
    for chat, kb_path in (raw.items() if isinstance(raw, dict) else []):
        try:
            out[int(chat)] = check_path(str(kb_path))
        except (KBError, ValueError):
            continue
    return out


def save_choice(path: Path, choice: dict[int, str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({str(k): v for k, v in choice.items()}), encoding="utf-8")
    except OSError:
        pass  # не записалось — выбор живёт до перезапуска


def read_token() -> str:
    """Токен бота. Пробелов и переносов в токене не бывает — если при копировании
    они попали в файл, убираются."""
    token = os.environ.get("KBRAG_TG_TOKEN", "")
    if not token.strip() and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8-sig")
    return "".join(token.split())


def is_question(text: str, session: Any) -> bool:
    """Сообщение — вопрос модели (а не команда, кнопка меню или запрос /search)."""
    if text.startswith("/") or text in ui.MENU or text == "Настройки":
        return False
    return getattr(session, "awaiting", None) != "search"


def access(path: Path = USERS_FILE) -> tuple[bool, set[int]]:
    """(открыт ли бот всем, id владельцев). id по одному в строке, «*» — всем, # — комментарий."""
    env = os.environ.get("KBRAG_TG_USERS", "")
    lines = env.replace(",", "\n").splitlines()
    if path.exists():
        lines += path.read_text(encoding="utf-8-sig").splitlines()
    owners, everyone = set(), False
    for line in lines:
        line = line.split("#", 1)[0].strip()
        if line.lower() in OPEN_ACCESS:
            everyone = True
        elif line.lstrip("-").isdigit():
            owners.add(int(line))
    return everyone, owners
