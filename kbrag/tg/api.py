"""Bot API Телеграма — обычный HTTP через requests, и ошибки по-человечески."""

from __future__ import annotations

import os
from typing import Any

API = os.environ.get("KBRAG_TG_API", "").strip() or "https://api.telegram.org"
POLL_TIMEOUT = 50  # сколько секунд Телеграм держит запрос новых сообщений


class TelegramError(RuntimeError):
    pass


class Telegram:
    def __init__(self, token: str, api: str = API, timeout: int = 30):
        self.base = f"{api.rstrip('/')}/bot{token}"
        self.timeout = timeout

    def call(
        self, method: str, files: Any = None, http_timeout: int | None = None, **params: Any
    ) -> Any:
        import json

        import requests

        url = f"{self.base}/{method}"
        params = {k: v for k, v in params.items() if v is not None}
        if files:
            data = {
                k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                for k, v in params.items()
            }
            response = requests.post(
                url, data=data, files=files, timeout=http_timeout or self.timeout
            )
        else:
            response = requests.post(url, json=params, timeout=http_timeout or self.timeout)
        try:
            data = response.json()
        except ValueError:
            raise TelegramError(f"{method}: ответ не JSON, HTTP {response.status_code}") from None
        if not data.get("ok"):
            raise TelegramError(f"{method}: {data.get('description') or response.status_code}")
        return data.get("result")  # ответ Bot API: объект, список или True

    def send(self, chat_id: int, text: str, markup: dict[str, Any] | None = None) -> dict[str, Any]:
        sent: dict[str, Any] = self.call(
            "sendMessage",
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            disable_web_page_preview=True,
            reply_markup=markup,
        )
        return sent

    def edit(
        self, chat_id: int, message_id: int, text: str, markup: dict[str, Any] | None = None
    ) -> None:
        try:
            self.call(
                "editMessageText",
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode="HTML",
                disable_web_page_preview=True,
                reply_markup=markup,
            )
        except TelegramError as exc:
            if "not modified" not in str(exc):  # тот же текст — не ошибка
                raise

    def photo(
        self, chat_id: int, png: bytes, caption: str = "", markup: dict[str, Any] | None = None
    ) -> Any:
        return self.call(
            "sendPhoto",
            chat_id=chat_id,
            caption=caption,
            parse_mode="HTML",
            reply_markup=markup,
            files={"photo": ("graph.png", png, "image/png")},
        )

    def edit_photo(
        self,
        chat_id: int,
        message_id: int,
        png: bytes,
        caption: str = "",
        markup: dict[str, Any] | None = None,
    ) -> None:
        """Заменить картинку в том же сообщении — переход к соседу не плодит сообщения."""
        media = {
            "type": "photo",
            "media": "attach://graph",
            "caption": caption,
            "parse_mode": "HTML",
        }
        self.call(
            "editMessageMedia",
            chat_id=chat_id,
            message_id=message_id,
            media=media,
            reply_markup=markup,
            files={"graph": ("graph.png", png, "image/png")},
        )

    def document(self, chat_id: int, name: str, data: bytes, caption: str = "") -> None:
        self.call(
            "sendDocument", chat_id=chat_id, caption=caption, files={"document": (name, data)}
        )

    def answer_callback(self, callback_id: str, text: str = "", alert: bool = False) -> None:
        try:
            self.call(
                "answerCallbackQuery",
                callback_query_id=callback_id,
                text=text[:190] or None,
                show_alert=alert or None,
            )
        except TelegramError:
            pass  # нажатие устарело — не беда

    def typing(self, chat_id: int) -> None:
        try:
            self.call("sendChatAction", chat_id=chat_id, action="typing")
        except TelegramError:
            pass

    def updates(self, offset: int | None, timeout: int = POLL_TIMEOUT) -> list[dict[str, Any]]:
        params = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
        if offset is not None:
            params["offset"] = offset
        return self.call("getUpdates", http_timeout=timeout + 15, **params) or []


def explain_error(exc: Exception) -> str:
    """Ошибка Телеграма → что с ней делать, по-человечески."""
    text = str(exc)
    if "409" in text or "Conflict" in text or "terminated by other getUpdates" in text:
        return (
            "Телеграм: этот бот уже запущен в ДРУГОМ окне (или на другом компьютере) — "
            "сообщения забирает та копия. Закройте лишние окна (Ctrl+C) и оставьте одно."
        )
    if "401" in text or "Unauthorized" in text or "Not Found" in text:
        return (
            "Телеграм не принимает токен. Возьмите токен заново у @BotFather "
            "(/mybots → бот → API Token) и сохраните в .local/telegram_token.txt."
        )
    return f"Телеграм не ответил ({type(exc).__name__}: {text[:200]}) — повтор через 5 с"
