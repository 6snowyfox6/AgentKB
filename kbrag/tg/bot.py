"""Телеграм-бот базы знаний. Работает рядом со страницей в браузере, а не вместо неё.

    python kbrag.py tg            только бот
    python kbrag.py web --tg      страница в браузере и бот сразу (модель векторов одна)

Интерфейс — кнопками (как выглядит — kbrag/tg/ui.py):

* меню внизу экрана: Поиск, Статьи, Факты, Новый разговор, Помощь.
  Любой другой текст — вопрос чат-боту;
* под ответом: фрагменты [1]…[8] (нажал — фрагмент целиком), «Как искал»,
  «Новый разговор»; пока бот пишет — «Остановить»;
* поиск и список статей листаются кнопками; у статьи — ссылка на издателя и
  её фрагменты; у датасета — таблица для Excel и «Спросить бота».

Настроек у ответа нет: подробно или кратко — человек пишет в самом вопросе,
модель так и отвечает. Команды тоже работают: /search запрос, /articles,
/dataset [название], /f 2, /new, /stats, /help.

Без новых библиотек: Bot API Телеграма — обычный HTTP, ходим через requests.
Бот сам спрашивает у Телеграма новые сообщения (long polling), поэтому ни
открытого порта, ни внешнего адреса у компьютера не нужно. Ответ модели
пишется в отдельном потоке — бот тем временем принимает нажатия, поэтому
«Остановить» срабатывает сразу.

Доступ — .local/telegram_users.txt, по одной записи в строке, # — комментарий:

* id человека — кому можно;
* строка «*» — бот открыт всем; гостю — KBRAG_TG_GUEST_LIMIT вопросов в час (access.py).

Без «*» чужим бот отвечает только их id (не чаще раза в 10 минут). Файл
перечитывается на каждом сообщении, перезапускать бота после правки не нужно.

Модель пишет до MAX_PARALLEL ответов сразу (по умолчанию 4, переменная
KBRAG_PARALLEL): кто спросил сверх этого, получает «в очереди» с местом и ждёт. Вторая копия бота на том же компьютере не
запускается: две копии забирают одни и те же сообщения, и на один /start
приходило по десять ответов.

Токен бота (выдаёт @BotFather) — в .local/telegram_token.txt или в KBRAG_TG_TOKEN.
Оба файла в .gitignore: в репозиторий не попадают.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

from ..common import kb_from_args
from ..kb import get as get_kb
from . import ui
from .access import (
    GUEST_WINDOW,
    access,
    guest_limit,
    is_question,
    read_token,
)
from .answers import AnswersMixin
from .api import POLL_TIMEOUT, Telegram, explain_error
from .base import (
    COMMANDS,
    DENIED_EVERY,
    HELP,
    MAX_SESSIONS,
    SEEN_UPDATES,
    STALE,
    Session,
)
from .browse import BrowseMixin
from .check import check
from .facts import FactsMixin
from .kbs import KbsMixin
from .media import MediaMixin


def _instance_lock(token: str) -> Any:
    """Одна копия бота на компьютере: занять локальный порт, выведенный из номера бота.
    Порт занят — бот уже запущен в другом окне. Процесс завершился — порт свободен сам."""
    import socket

    bot_id = int(token.split(":", 1)[0]) if token.split(":", 1)[0].isdigit() else 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows: порт только наш
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(("127.0.0.1", 40000 + bot_id % 20000))
    except OSError:
        sock.close()
        return None
    return sock


class Bot(AnswersMixin, MediaMixin, BrowseMixin, FactsMixin, KbsMixin):
    """Бот целиком: ядро здесь — разговоры, доступ, разбор сообщений и кнопок, цикл;
    остальное — в частях (answers, media, browse, facts, kbs)."""

    def session(self, chat_id: int) -> Session:
        session = self.sessions.pop(chat_id, None) or Session()
        self.sessions[chat_id] = session  # в конец: недавний разговор
        while len(self.sessions) > MAX_SESSIONS:  # старые забываются, кроме тех, где пишут
            old = next((k for k, s in self.sessions.items() if not s.busy), None)
            if old is None or old == chat_id:
                break
            self.sessions.pop(old)
        return session

    def allowed(self, user_id: Any) -> bool:
        everyone, owners = access(self.users_file)
        return everyone or user_id in owners

    def guest(self, user_id: Any) -> bool:
        """Не из списка — пишет в открытый бот (для журнала)."""
        return user_id not in access(self.users_file)[1]

    def over_limit(self, user_id: Any, now: float | None = None) -> float:
        """Гость открытого бота исчерпал вопросы за час: через сколько секунд следующий
        (0 — можно спрашивать; вопрос тогда засчитывается). Владельцам — всегда 0."""
        limit = guest_limit()
        if not limit or not self.guest(user_id):
            return 0.0
        now = time.monotonic() if now is None else now
        recent = [t for t in self.guest_questions.get(user_id, []) if now - t < GUEST_WINDOW]
        if len(recent) >= limit:
            self.guest_questions[user_id] = recent
            return GUEST_WINDOW - (now - recent[0])
        self.guest_questions[user_id] = [*recent, now]
        return 0.0

    def deny(self, chat_id: int, user_id: Any) -> None:
        """Чужому — его id, но не на каждое сообщение: не чаще раза в DENIED_EVERY."""
        now = time.monotonic()
        if now - self.denied.get(user_id, -DENIED_EVERY) < DENIED_EVERY:
            return
        self.denied[user_id] = now
        self.tg.send(
            chat_id,
            f"Нет доступа. Ваш id: <code>{user_id}</code>\n"
            "Владелец базы добавляет его в .local/telegram_users.txt.",
        )

    def handle(self, update: dict[str, Any]) -> None:
        if update.get("callback_query"):
            self.on_callback(update["callback_query"])
        else:
            self.on_message(update.get("message") or {})

    def on_message(self, message: dict[str, Any]) -> None:
        chat_id = (message.get("chat") or {}).get("id")
        user_id = (message.get("from") or {}).get("id")
        text = (message.get("text") or "").strip()
        if chat_id is None or not text:
            return
        if not self.allowed(user_id):
            self.deny(chat_id, user_id)
            self.log(f"  чужой пользователь {user_id}: {text[:40]!r}")
            return
        self.log(f"  ← {user_id}{' (гость)' if self.guest(user_id) else ''}: {text[:60]!r}")
        session = self.session(chat_id)
        if is_question(text, session):
            wait = self.over_limit(user_id)
            if wait:
                self.tg.send(
                    chat_id,
                    f"Бот открыт всем, и у гостя — {guest_limit()} вопросов в час. Следующий "
                    f"— через {max(1, round(wait / 60))} мин. Поиск (/search) и статьи — "
                    "без ограничения.",
                )
                self.log(f"  лимит гостя {user_id}: ждать {wait:.0f} с")
                return
        self._guard(chat_id, text, self.route, chat_id, session, text)

    def route(self, chat_id: int, session: Session, text: str) -> None:
        if text in ui.MENU:  # кнопка меню внизу экрана
            command, arg = "/" + ui.MENU[text], ""
        elif text == "Настройки":  # кнопка из старого меню (до v12)
            command, arg = "/settings", ""
        else:
            command, _, arg = text.partition(" ")
            command = command.split("@", 1)[0].lower()
            arg = arg.strip()
            if not command.startswith("/"):
                command, arg = "", text
        if command:
            session.awaiting = None
        if command in ("/start", "/help"):
            self.tg.send(chat_id, HELP, ui.menu_keyboard())
        elif command == "/new":
            self.new_conversation(chat_id, session)
        elif command in ("/brief", "/detail", "/settings"):
            self.tg.send(
                chat_id,
                "Настроек больше нет: нужно подробно или кратко — так и "
                "напишите в вопросе, например «подробно: как "
                "измеряют вязкость?». Что в базе и готова ли модель — /stats",
                ui.menu_keyboard(),
            )
        elif command == "/f":
            self.fragment(chat_id, session, arg)
        elif command == "/search":
            if arg:
                self.search(chat_id, session, arg)
            else:
                session.awaiting = "search"
                self.tg.send(
                    chat_id,
                    "Что искать? Напишите запрос — например: "
                    "<i>методы измерения вязкости расплавов</i>",
                    ui.inline([[ui.button("Отмена", "cancel")]]),
                )
        elif command == "/articles":
            self.articles(chat_id, session)
        elif command in ("/facts", "/fact", "/dataset", "/datasets"):  # /dataset — старое имя
            if arg:
                self.dataset_by_name(chat_id, session, arg)
            else:
                self.datasets(chat_id, session)
        elif command == "/stats":
            self.stats(chat_id)
        elif command == "/fast":
            session.fast = not session.fast
            self.tg.send(
                chat_id,
                (
                    "Быстрый режим включён: вопрос не разбирается моделью, статьи не сверяются — "
                    "ответ начинается сразу, но на сложный вопрос он хуже. Выключить — /fast"
                    if session.fast
                    else "Быстрый режим выключен: вопрос снова разбирает модель."
                ),
                ui.menu_keyboard(),
            )
        elif command == "/kb":
            if arg:
                self.set_kb(chat_id, session, arg)
            else:
                self.kb_menu(chat_id, session)
        elif command == "/conflicts":
            on = not self.checks_conflicts(session)
            session.conflicts = on
            self.tg.send(
                chat_id,
                (
                    "Сверка статей включена: перед ответом бот проверяет, не расходятся ли "
                    "статьи, — это ещё около 4 с."
                    if on
                    else "Сверка статей выключена: ответ начнётся быстрее, но расхождения "
                    "между статьями бот не ищет. Включить — /conflicts"
                ),
                ui.menu_keyboard(),
            )
        elif command:
            self.tg.send(
                chat_id, "Такой команды нет. Что умеет бот — «Помощь» в меню.", ui.menu_keyboard()
            )
        elif session.awaiting == "search":
            session.awaiting = None
            self.search(chat_id, session, arg)
        else:
            self.question(chat_id, session, arg)

    def new_conversation(self, chat_id: int, session: Session) -> None:
        session.history.clear()
        session.sources = []
        session.awaiting = None
        self.tg.send(chat_id, "Начали новый разговор. Задайте вопрос.", ui.menu_keyboard())

    def on_callback(self, cq: dict[str, Any]) -> None:
        cid = cq.get("id", "")
        user_id = (cq.get("from") or {}).get("id")
        message = cq.get("message") or {}
        chat_id = (message.get("chat") or {}).get("id")
        mid = message.get("message_id")
        data = cq.get("data") or ""
        if not self.allowed(user_id):
            self.tg.answer_callback(cid, "Нет доступа", alert=True)
            return
        if chat_id is None or mid is None:
            self.tg.answer_callback(cid)
            return
        session = self.session(chat_id)
        toast = ""
        try:
            toast = self.press(chat_id, mid, session, data) or ""
        except Exception as exc:  # noqa: BLE001
            self.log(f"  ошибка на кнопке {data!r}: {type(exc).__name__}: {exc}")
            toast = f"Ошибка: {type(exc).__name__}: {str(exc)[:120]}"
        self.tg.answer_callback(cid, toast, alert=toast == STALE or toast.startswith("Ошибка"))

    def press(self, chat_id: int, mid: int, session: Session, data: str) -> str | None:
        """Нажатие → действие. Возвращает короткую подсказку для всплывашки."""
        action, *args = data.split(":")
        nums = [int(a) for a in args if a.lstrip("-").isdigit()]
        if action == "noop":
            return None
        if action == "cancel":
            session.awaiting = None
            self.tg.edit(chat_id, mid, "Отменено.")
            return None
        if action == "new":
            self.new_conversation(chat_id, session)
            return "Новый разговор"
        if action == "fm":
            return self.send_formula(chat_id, nums[0]) if nums else STALE
        if action == "fg":
            return self.send_figure(chat_id, nums[0]) if nums else STALE
        if action == "kb":
            if not nums or not 0 <= nums[0] < len(session.kbs):
                return STALE
            return self.set_kb(chat_id, session, session.kbs[nums[0]], mid)
        if action == "stop":
            if session.busy:
                session.stop.set()
                return "Останавливаю…"
            return "Ответ уже готов"
        if action in ("f", "how", "lk"):
            record = session.answers.get(nums[0]) if nums else None
            if record is None:
                return STALE
            if action == "lk":
                self.answer_links(chat_id, session, record)
                return None
            if action == "f":
                src = next((s for s in record["sources"] if s["n"] == nums[1]), None)
                if src is None:
                    return STALE
                self.show_fragment(chat_id, session, src)
            else:
                self.tg.send(chat_id, ui.how_text(record))
            return None
        if action in ("pg", "it"):
            lst = session.lists.get(nums[0]) if nums else None
            if lst is None or len(nums) < 2:
                return STALE
            if action == "pg":
                page = max(0, min(nums[1], ui.pages(len(lst["items"]), lst["kind"]) - 1))
                lst["page"] = page
                self.tg.edit(
                    chat_id,
                    mid,
                    ui.list_text(lst, page),
                    ui.list_keyboard(nums[0], lst, page, lst.get("extra")),
                )
            else:
                self.open_item(chat_id, session, nums[0], lst, nums[1])
            return None
        if action == "art":  # «Вся статья» у фрагмента
            doc_id = session.lists.get(nums[0], {}).get("doc_id") if nums else None
            if not doc_id:
                return STALE
            self.article_card(chat_id, session, doc_id)
            return None
        if action == "chunks":
            lst = session.lists.get(nums[0]) if nums else None
            if lst is None:
                return STALE
            self.article_chunks(
                chat_id, session, lst["doc_id"], lst.get("title") or "", lst.get("url") or ""
            )
            return None
        if action == "askq":
            lst = session.lists.get(nums[0]) if nums else None
            if lst is None or not lst.get("query"):
                return STALE
            self.question(chat_id, session, lst["query"])
            return None
        if action == "dsl":
            self.datasets(chat_id, session, mid)
            return None
        if action in ("ds", "dn", "dt", "csv", "ask"):
            if not nums or not 0 <= nums[0] < len(session.territories):
                return STALE
            name = session.territories[nums[0]]["name"]
            if action == "ds":  # из списка — картинка новым сообщением
                self.graph_card(chat_id, session, nums[0])
            elif action == "dn":  # сосед на картинке — в том же сообщении
                self.graph_card(chat_id, session, nums[0], mid)
            elif action == "dt":  # факты списком, с числом статей
                self.dataset_card(chat_id, session, nums[0])
            elif action == "csv":
                self.dataset_csv(chat_id, name)
                return "Таблица отправлена"
            else:
                self.question(chat_id, session, ui.dataset_question(name))
            return None
        return STALE

    def _stats(self, chat_id: int) -> dict[str, Any] | None:
        from ..index import db

        try:
            with db.connect(self.dsn) as conn:
                return {
                    **db.stats(conn, self.kb_of(chat_id)),
                    "kb_title": get_kb(self.kb_of(chat_id)).title,
                }
        except Exception:  # noqa: BLE001 — база лежит: так и покажем
            return None

    def stats(self, chat_id: int) -> None:
        from ..chat import answer as chat

        status = chat.llm_status(self.chat_settings.host, self.chat_settings.model, timeout=3)
        self.tg.send(chat_id, ui.status_text(status, self._stats(chat_id)))

    def poll_once(self, timeout: int = POLL_TIMEOUT) -> int:
        updates = self.tg.updates(self.offset, timeout)
        for update in updates:
            uid = update["update_id"]
            self.offset = uid + 1
            if uid in self.seen:  # то же сообщение второй раз — не отвечать дважды
                continue
            self.seen = (self.seen + [uid])[-SEEN_UPDATES:]
            self.handle(update)
        return len(updates)

    def run(self, stop: threading.Event | None = None) -> None:
        token = self.tg.base.rsplit("/bot", 1)[-1]
        self._lock = _instance_lock(token)
        if self._lock is None:
            self.log(
                "Телеграм-бот НЕ запущен: он уже работает в другом окне на этом компьютере. "
                "Две копии забирают одни и те же сообщения и отвечают по несколько раз. "
                "Оставьте одну (закройте лишнее окно Ctrl+C)."
            )
            return
        self.background = True
        try:
            self.tg.call(
                "setMyCommands", commands=[{"command": c, "description": d} for c, d in COMMANDS]
            )
        except Exception as exc:  # noqa: BLE001 — без меню команд бот всё равно работает
            self.log(f"Меню команд не задалось: {exc}")
        try:  # с webhook getUpdates не работает — сообщения не приходят
            self.tg.call("deleteWebhook")
        except Exception:  # noqa: BLE001
            pass
        last_error = ""
        while not (stop and stop.is_set()):
            try:
                self.poll_once()
                last_error = ""
            except Exception as exc:  # noqa: BLE001 — сеть моргнула: подождать и дальше
                text = explain_error(exc)
                if text != last_error:  # одно и то же не повторять каждые 5 с
                    self.log(text)
                    last_error = text
                time.sleep(5)


def start_in_thread(
    dsn: Any, embedder: Any, chat_settings: Any, log: Any = print
) -> threading.Thread | None:
    """Бот в фоне того же процесса — для `web --tg`: модель векторов одна на двоих."""
    token = read_token()
    if not token:
        log(
            "Телеграм-бот не запущен: нет токена (.local/telegram_token.txt). Как завести — ЗАПУСК.md."
        )
        return None
    bot = Bot(Telegram(token), dsn, embedder, chat_settings, log=log)
    try:
        me = bot.tg.call("getMe")
    except Exception as exc:  # noqa: BLE001
        log(
            "Телеграм-бот не запущен. "
            + explain_error(exc)
            + "\nПроверить всё по шагам: python kbrag.py tg --check"
        )
        return None
    everyone, owners = access()
    who = "всем" if everyone else f"тем, кто в .local/telegram_users.txt ({len(owners)} чел.)"
    log(
        f"Телеграм-бот: @{me.get('username')} — отвечает {who}; входящие сообщения видны в этом окне"
    )
    thread = threading.Thread(target=bot.run, name="telegram", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> int:
    import argparse

    from ..chat import answer as chat
    from ..chat.settings import Settings
    from ..common import add_db_args, add_embedder_args, add_llm_args
    from ..index import db
    from ..index.embed import build_embedder

    parser = argparse.ArgumentParser(description="Телеграм-бот базы знаний")
    add_db_args(parser)
    add_embedder_args(parser)
    add_llm_args(parser)
    parser.add_argument("--check", action="store_true", help="проверить токен и доступ и выйти")
    args = parser.parse_args(argv)
    if args.check:
        return check()

    token = read_token()
    if not token:
        print(
            "Нет токена бота. Получить у @BotFather в Телеграме и сохранить:\n"
            "    python -c \"open('.local/telegram_token.txt','w').write('ТОКЕН')\"",
            file=sys.stderr,
        )
        return 1
    tg = Telegram(token)
    try:
        me = tg.call("getMe")
    except Exception as exc:  # noqa: BLE001
        print(explain_error(exc) + "\nПодробнее: python kbrag.py tg --check", file=sys.stderr)
        return 1
    base = kb_from_args(args.kb)
    if base is None:
        return 1
    try:
        with db.connect(args.dsn) as conn:
            info = db.stats(conn, base.path)
    except Exception as exc:  # noqa: BLE001
        print(f"База недоступна: {exc}\nПоднимите её: python kbrag.py start", file=sys.stderr)
        return 1
    print(f"База: {base.title} ({base.path or 'всё дерево'})")
    everyone, users = access()
    print(f"В базе: {info['documents']} статей. Бот: @{me.get('username')}")
    if everyone:
        limit = guest_limit()
        print(
            f"Бот открыт всем: гостю — {limit} вопросов в час."
            if limit
            else "Бот открыт всем, без ограничения вопросов (KBRAG_TG_GUEST_LIMIT=0)."
        )
    elif not users:
        print(
            "В .local/telegram_users.txt никого нет — бот никому не ответит по существу. Напишите\n"
            "ему что-нибудь: он пришлёт ваш id, его и записать в .local/telegram_users.txt."
        )
    print("Загружаю модель эмбеддингов…")
    embedder = build_embedder(device=args.device)
    embedder.encode_one("прогрев")
    settings = Settings(model=args.model, host=args.llm_host, kb=base.path)
    status = chat.llm_status(settings.host, settings.model)
    print(f"Чат-бот: {settings.model} — " + ("готов" if status["ok"] else status["error"]))
    print(
        f"Бот работает: пишите @{me.get('username')} в Телеграме (сначала /start). "
        "Входящие сообщения видны здесь. Остановить — Ctrl+C\n"
    )
    try:
        Bot(tg, args.dsn, embedder, settings).run()
    except KeyboardInterrupt:
        print("Остановлен.")
    return 0
