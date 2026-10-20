"""Языковая модель — один клиент на весь проект.

Модель работает в vLLM (Docker, `python kbrag.py start`), и проект говорит с ней
по OpenAI-совместимому протоколу /v1/chat/completions. Тот же протокол понимают
и другие серверы (llama.cpp, SGLang), так что сервер можно сменить, не трогая код.
Здесь — адрес, выключение размышлений Qwen3, разбор JSON из ответа и тексты
ошибок; остальные модули только задают вопросы.

vLLM сам ставит одновременные запросы в одну пачку (continuous batching): несколько
пользователей и несколько фрагментов графа считаются разом, без настроек.

Адрес по умолчанию — 127.0.0.1:8100 (на 8000 — веб-страница проекта); другой задаётся
переменной окружения KBRAG_LLM (например, если модель работает на сервере: http://сервер:8100).
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .machine import integer

# Имя модели на сервере (vLLM --served-model-name): сама модель — Qwen/Qwen3-14B-AWQ,
# та же Qwen3-14B, сжатая до 4 бит.
DEFAULT_MODEL = "qwen3-14b"
# Окно модели задаёт сервер (vLLM --max-model-len). Здесь — то же число, чтобы
# задание вместе с ответом в него помещалось (chat.messages.fit_budget). 12 288 — под
# 16 ГБ; на машине мощнее — KBRAG_NUM_CTX в .local/settings.env (machine.py): его же
# берёт docker-compose.yml, так что окно в коде и у сервера одно.
NUM_CTX = integer("KBRAG_NUM_CTX", 12288, low=2048)
# 127.0.0.1, а не localhost — см. DEFAULT_DSN в index/db.py.
LLM_HOST = os.environ.get("KBRAG_LLM", "").strip() or "http://127.0.0.1:8100"

# Qwen3 по умолчанию сначала «думает» вслух — это в разы дольше, а рассуждать в
# наших заданиях не о чем. vLLM выключает размышления так, через шаблон чата.
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
INSTALL_HINT = "Запустите её: python kbrag.py start (модель грузится 1–2 минуты)."

# --------------------------------------------------------------------------- #
#  Сколько заданий модель вмещает сразу
# --------------------------------------------------------------------------- #
# Память vLLM под задания (KV-кэш) — общая на все одновременные запросы. На 16 ГБ это
# 13 088 токенов: одно полное окно (12 288) или два обычных задания чат-бота. Сверх
# этого vLLM вытесняет запросы и считает их заново: при двух людях сразу (каждый ответ —
# ещё и несколько запросов параллельно) запросы стояли до таймаута и падали у обоих.
# Поэтому запрос сначала берёт место в бюджете процесса — оценку своего задания и ответа
# в токенах — и ждёт, пока место есть. Бюджет — у vLLM (/metrics), иначе
# KBRAG_KV_TOKENS в .local/settings.env, иначе окно с запасом на ответ.
KV_TOKENS = integer("KBRAG_KV_TOKENS", 0, low=0)
CHARS_PER_TOKEN = 3  # у Qwen3: английский — около 4 знаков на токен, русский — 2,5–3
IMAGE_TOKENS = 2000  # картинка до 1280 точек по стороне (chunks/figures.py)
JSON_REPLY = 1024  # ответ JSON без max_tokens — с запасом
STREAM_REPLY = 2048  # ответ потоком (чат-бот)
RETRY_METRICS = 60.0  # с: vLLM не ответил про память — когда спросить снова
_KV_BLOCKS = re.compile(r"vllm:cache_config_info\{[^}]*\}")


def _tokens_of(messages: list[dict[str, Any]]) -> int:
    """Оценка задания в токенах — по длине текста (без токенайзера: хватает для бюджета)."""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content) // CHARS_PER_TOKEN
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    total += IMAGE_TOKENS
                else:
                    total += len(str(part.get("text") or "")) // CHARS_PER_TOKEN
        total += 8  # служебные токены сообщения
    return total


def _kv_from_metrics(host: str) -> int | None:
    """Память vLLM под задания, токенов: num_gpu_blocks × block_size из /metrics."""
    import requests

    try:
        text = requests.get(f"{host.rstrip('/')}/metrics", timeout=3).text
    except Exception:  # noqa: BLE001 — сервер не vLLM или не отвечает
        return None
    found = _KV_BLOCKS.search(text)
    if not found:
        return None
    labels = dict(re.findall(r'(\w+)="([^"]*)"', found.group(0)))
    try:
        return int(labels["num_gpu_blocks"]) * int(labels["block_size"])
    except (KeyError, ValueError):
        return None


class Budget:
    """Места под задания у одного сервера модели: hold(n) ждёт, пока занято + n
    влезает в size. Задание больше всего бюджета пускается, когда больше никого нет."""

    def __init__(self, size: int) -> None:
        self.size = max(1, size)
        self.used = 0
        self._cond = threading.Condition()

    @contextmanager
    def hold(self, tokens: int) -> Iterator[None]:
        tokens = max(1, min(tokens, self.size))
        with self._cond:
            while self.used and self.used + tokens > self.size:
                self._cond.wait()
            self.used += tokens
        try:
            yield
        finally:
            with self._cond:
                self.used -= tokens
                self._cond.notify_all()

    def resize(self, size: int) -> None:
        with self._cond:
            self.size = max(1, size)
            self._cond.notify_all()


_BUDGETS: dict[str, Budget] = {}
_ASKED: dict[str, float] = {}  # сервер → когда vLLM не ответил про память
_BUDGETS_LOCK = threading.Lock()


def budget(host: str) -> Budget:
    """Бюджет сервера модели — один на процесс. Пока vLLM не сказал, сколько у него
    памяти (не запущен, не vLLM), — временный, по окну, и раз в минуту спросить снова."""
    import time

    with _BUDGETS_LOCK:
        found = _BUDGETS.get(host)
        unknown_since = _ASKED.get(host)
        if found is not None and (
            unknown_since is None or time.monotonic() - unknown_since < RETRY_METRICS
        ):
            return found
        size = KV_TOKENS or _kv_from_metrics(host)
        if size:
            _ASKED.pop(host, None)
        else:
            _ASKED[host] = time.monotonic()
            size = NUM_CTX + STREAM_REPLY
        if found is None:
            found = _BUDGETS[host] = Budget(size)
        else:
            found.resize(size)  # тот же объект: кто уже держит место, отдаст его сюда же
        return found


class LLMError(RuntimeError):
    """Модель не ответила или ответила ошибкой. Текст — для человека."""


def strip_think(text: str) -> str:
    """Убрать блок размышлений Qwen3, в том числе незакрытый (на всякий случай:
    с NO_THINK его нет)."""
    cleaned = THINK_RE.sub("", text or "")
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1]
    return cleaned


def extract_json(text: str) -> dict[str, Any]:
    """Первый объект JSON из ответа модели: вокруг него бывает лишний текст."""
    cleaned = strip_think(text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"в ответе нет JSON: {cleaned[:200]!r}")
    data: dict[str, Any] = json.loads(cleaned[start : end + 1])
    return data


@dataclass
class LLM:
    model: str = DEFAULT_MODEL
    host: str = LLM_HOST
    timeout: int = 180
    temperature: float = 0.2

    def _url(self, path: str) -> str:
        return f"{self.host.rstrip('/')}/v1{path}"

    def _payload(self, messages: list[dict[str, Any]], temperature: float | None) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            **NO_THINK,
        }

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        timeout: Any = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Один вопрос — один объект JSON в ответ.

        max_tokens — потолок длины ответа: зациклившаяся модель иначе пишет до
        конца окна и держит занятым место, которое нужно другим запросам."""
        import requests

        payload = self._payload(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature,
        )
        payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        need = _tokens_of(payload["messages"]) + (max_tokens or JSON_REPLY)
        # Таймаут — на сам запрос: очередь за местом в бюджете в него не входит.
        with budget(self.host).hold(need):
            response = requests.post(
                self._url("/chat/completions"), json=payload, timeout=timeout or self.timeout
            )
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        return extract_json(message.get("content") or "")

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Ответ кусками: {"text": …}, в конце {"done": True, "tokens": …}.

        Картинка в сообщении — частью {"type": "image_url", "image_url": {"url":
        "data:image/png;base64,…"}}, как в протоколе OpenAI."""
        payload = self._payload(messages, temperature)
        payload.update(stream=True, stream_options={"include_usage": True})
        # Место в бюджете — до конца ответа: генератор закрыли («Стоп») — оно освобождается.
        with budget(self.host).hold(_tokens_of(messages) + STREAM_REPLY):
            yield from self._stream(payload)

    def _stream(self, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        import requests

        try:
            response = requests.post(
                self._url("/chat/completions"),
                json=payload,
                stream=True,
                timeout=(10, self.timeout),
            )
        except requests.RequestException as exc:
            raise LLMError(
                f"Модель не отвечает на {self.host}: {type(exc).__name__}. {INSTALL_HINT}"
            ) from None
        with response:
            if response.status_code == 404:
                raise LLMError(f"На сервере {self.host} нет модели {self.model}")
            if response.status_code >= 400:
                raise LLMError(
                    f"Сервер модели ответил ошибкой {response.status_code}: "
                    f"{response.text[:300]}"
                )
            tokens = None
            for line in response.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise LLMError(f"Модель: {chunk['error']}")
                if chunk.get("usage"):
                    tokens = chunk["usage"].get("completion_tokens")
                for choice in chunk.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content") or ""
                    if text:
                        yield {"text": text}
            yield {"done": True, "tokens": tokens}

    def status(self, timeout: int = 5) -> dict[str, Any]:
        """Жив ли сервер модели и та ли на нём модель — для подсказки человеку."""
        import requests

        try:
            response = requests.get(self._url("/models"), timeout=timeout)
            response.raise_for_status()
            names = [m.get("id", "") for m in response.json().get("data", [])]
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "model": self.model,
                "error": f"Модель не отвечает ({type(exc).__name__}). {INSTALL_HINT}",
            }
        if self.model not in names:
            return {
                "ok": False,
                "model": self.model,
                "models": names,
                "error": f"На сервере {self.host} другая модель: {', '.join(names) or 'нет'}",
            }
        return {"ok": True, "model": self.model}
