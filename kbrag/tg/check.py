"""python kbrag.py tg --check: всё ли в порядке с ботом, без базы и моделей."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .access import TOKEN_FILE, USERS_FILE, access, guest_limit, read_token
from .api import API, Telegram, explain_error


def check(
    log: Any = print, token: str | None = None, api: str = API, users_file: Path = USERS_FILE
) -> int:
    """python kbrag.py tg --check: всё ли в порядке с ботом, без базы и моделей."""
    token = read_token() if token is None else token
    log("Проверка Телеграм-бота\n")
    if not token:
        log(
            f"[СБОЙ] Токена нет: нет файла {TOKEN_FILE.name} в папке проекта (или он пустой).\n"
            "       Сохранить: python -c \"open('.local/telegram_token.txt','w',encoding='utf-8')"
            ".write('ТОКЕН')\""
        )
        return 1
    if not re.fullmatch(r"\d{6,}:[\w-]{30,}", token):
        log(
            f"[СБОЙ] Токен не похож на токен (вида 123456789:AA…): «{token[:12]}…», "
            f"длина {len(token)}.\n       Скопируйте его у @BotFather целиком и сохраните заново."
        )
        return 1
    log(f"[OK  ] Токен есть: {token.split(':')[0]}:…")
    tg = Telegram(token, api=api, timeout=20)
    try:
        me = tg.call("getMe")
    except Exception as exc:  # noqa: BLE001
        log("[СБОЙ] " + explain_error(exc))
        return 1
    log(f"[OK  ] Телеграм принял токен. Ваш бот: @{me.get('username')} — пишите именно ему.")
    try:
        hook = tg.call("getWebhookInfo") or {}
        if hook.get("url"):
            tg.call("deleteWebhook")
            log("[OK  ] У бота был webhook — убран (из-за него сообщения не доходили).")
    except Exception:  # noqa: BLE001
        pass
    everyone, users = access(users_file)
    if everyone:
        limit = guest_limit()
        if limit:
            log(
                "[ !! ] Бот открыт всем («*» в .local/telegram_users.txt): любой, кто найдёт "
                f"его, спрашивает модель.\n       Гостю — {limit} вопросов в час "
                "(KBRAG_TG_GUEST_LIMIT в .local/settings.env); владельцам из списка — без "
                "ограничения."
                + (f" Владельцы: {', '.join(map(str, sorted(users)))}." if users else "")
            )
        else:
            log(
                "[ !! ] Бот открыт всем («*» в .local/telegram_users.txt) БЕЗ ограничения "
                "вопросов (KBRAG_TG_GUEST_LIMIT=0):\n       любой, кто найдёт бота, может "
                "занять модель. Ограничить — KBRAG_TG_GUEST_LIMIT=20."
            )
    elif users:
        log(
            f"[OK  ] Кому бот отвечает (.local/telegram_users.txt): {', '.join(map(str, sorted(users)))}"
        )
    else:
        log(
            "[ !! ] В .local/telegram_users.txt никого нет — бот на всё отвечает «Нет доступа. Ваш id».\n"
            "       Открыть всем: строка * в .local/telegram_users.txt."
        )
    try:
        waiting = tg.call("getUpdates", timeout=0, http_timeout=20) or []
    except Exception as exc:  # noqa: BLE001
        log("[СБОЙ] " + explain_error(exc))
        return 1
    senders = {}
    for u in waiting:
        who = (u.get("message") or u.get("callback_query") or {}).get("from") or {}
        if who.get("id"):
            senders[who["id"]] = who.get("first_name") or who.get("username") or ""
    if senders:
        log(f"[ !! ] Боту написали, но бот сейчас НЕ запущен — сообщений ждёт: {len(waiting)}.")
        for uid, name in senders.items():
            mark = (
                "есть доступ"
                if uid in users
                else "гость — бот открыт всем" if everyone else "НЕТ в .local/telegram_users.txt"
            )
            log(f"       от {name} — id {uid} ({mark})")
        if not everyone and any(uid not in users for uid in senders):
            log(
                "       Добавить id: python -c \"open('.local/telegram_users.txt','a',"
                "encoding='utf-8').write('ID\\n')\""
            )
    else:
        log(
            "[ !! ] Непрочитанных сообщений нет. Если вы писали боту — либо он уже запущен и "
            "всё забрал,\n       либо писали не тому боту (нужен @"
            + str(me.get("username"))
            + "), либо не нажали «Начать» (/start)."
        )
    log(
        "\nЗапуск бота: python kbrag.py web --tg (или только бот: python kbrag.py tg).\n"
        "Окно не закрывать — пока оно открыто, бот отвечает."
    )
    return 0
