r"""KBRAG — один вход для всех действий.

Запуск из папки проекта в любом терминале — cmd, PowerShell, Терминал Windows:

    python kbrag.py <команда> [ключи]

База и модель
    start                       поднять базу и модель Qwen3 в vLLM (Docker)
    stop                        остановить базу и модель, данные остаются
    status                      что сейчас работает
    init                        создать таблицы — один раз на новой машине
    stats                       сколько статей и фрагментов в базе

Свои статьи
    import "D:/статьи"          разобрать PDF (файл или папку) и загрузить в базу --kb
    import "D:/статьи" --ask    то же, недостающие метаданные спросить в терминале
    import … --figures          и описать рисунки моделью со зрением (qwen3-vl)
    figures                     описать рисунки базы --kb моделью со зрением (по файлам,
                                без переразбора); "doc_id" — одной статьи, --redo — заново
    reparse                     разобрать заново из PDF статьи базы --kb (граф сохраняется)
    reparse "doc_id"            только эту статью
    import … --ocr              и сканы: распознать текст страниц (медленно; docs/КОМАНДЫ.md, «Сканы»)
    ingest                      загрузить разобранное в базу (import делает это сам)
    tidy                        убрать служебный текст (колонтитулы, благодарности) из базы

Поиск и чат-бот
    web                         в браузере: поиск, статьи, чат-бот и граф связей
    web --tg                    то же + Телеграм-бот (в одном окне, модель векторов общая)
    tg                          только Телеграм-бот
    tg --check                  бот молчит? проверить токен, доступ, кто писал
    serve                       то же без браузера (только адреса /api)
    search "запрос"             найти фрагменты статей в терминале (--json — для скриптов)
    ask "вопрос"                спросить чат-бота в терминале (нужна модель; --json — для скриптов)
                                (хотите подробно или кратко — так и напишите в вопросе)

Дерево баз
    kb list                     все базы: названия, сколько статей, какая по умолчанию
    kb create a.b --title "…"   новая база: папка настроек и узел (настройки — от a)
    kb move a.b c.b             перенести базу со всем, что под ней, и её статьи
    kb assign a.b ID …          положить статьи в базу (ещё: --from x.y, --unplaced)
    kb unassign a.b ID …        убрать статьи из базы (сами статьи остаются)
    kb remove a.b               показать, что уберётся; с --yes — убрать
                                (--docs — и статьи, что лежат только здесь; --config — и папки)

Граф связей
    graph                      Qwen3 выписывает факты из статей (правила — graph-rules.txt базы)
    graph --limit 30            только 30 фрагментов — проба
    graph --redo                всё заново (после правки правил)
    graph --new                 что нашла модель и где ошибалась
    graph --recheck             перепроверить готовые факты новыми правилами кода (без модели)
    graph --confirm             подтвердить факты вопросами по цитате (идёт и после graph)
    graph --names               свести написания одного и того же по смыслу (идёт и после graph)
    synonyms                    предложения в словарь синонимов базы (synonyms.yaml)
    facts                       у чего больше всего фактов из статей
    facts "Анабарский щит"      все факты о нём с цитатами, файл CSV для Excel (--json — в stdout)
                                (старое имя команды — dataset — тоже работает)
    questions                   вопросы оценки из фактов графа → .local/kbs/<база>/eval-graph.yaml

Проверка
    test                        проверить код: tests/ (pytest), без базы, модели и видеокарты
    lint                        проверки кода: ruff, black --check, mypy --strict
    eval                        оценить базу и чат-бота по вопросам eval-questions.yaml базы
    eval --only-search          то же, только поиск — без модели
    eval --only s01,t01         только эти вопросы
    eval --no-graph             то же без графа — сравнить с обычным прогоном, что граф даёт
    eval --questions .local/kbs/<база>/eval-graph.yaml   по вопросам из графа (questions)
    formula ID (55)             формула статьи по номеру: LaTeX, словами, где на неё ссылаются
    formula --find "…"          формулы по смыслу (--kb — в какой базе, --json — в stdout)
    access                      кто к каким базам: access add ИМЯ --kb ВЕТКА [--telegram ID]
                                (токен входа — сразу), access list, remove, token, guests
    summaries                   резюме статей и разделов, которых нет (пишет модель; --redo —
                                заново; "doc_id" --show — показать)
    figure ID 3                 рисунок статьи «Рис. 3»: файл, подпись, описание, кто ссылается
    figure --find "…"           рисунки по смыслу — по подписи и описанию
    mcp                         MCP-сервер для Claude Code поверх serve (docs/КОМАНДЫ.md, «Claude Code»)
    unschedule                  убрать ночную добычу статей, если её включали раньше
    help                        эта справка

Ключи
    --limit 10        сколько результатов показать (search); сколько фрагментов (graph)
    --workers 8       сколько фрагментов у модели одновременно (graph; по умолчанию 8)
    --no-llm          вопросы по шаблону, без модели (questions)
    --force           обработать заново даже то, что уже было (import, ingest)
    --ask             спросить в терминале название, авторов, год, если их нет (import)
    --figures         описать рисунки статей моделью со зрением (import, reparse): сервис
                      vision поднимается сам — docs/КОМАНДЫ.md, «Рисунки»
    --kb physics      с какой базой дерева работать (любая команда): поиск, граф и
                      оценка — по её поддереву, промпты и словари — из её настроек
                      (config/kbs/…); «.» — всё дерево. Без ключа — база_по_умолчанию
                      из config/kbs/kb.yaml

Окружение .venv запускатель находит сам: запускать можно обычным `python`,
нужные библиотеки всё равно возьмутся из .venv проекта.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
# Только для этой машины — статьи, логи, ключи, кэши — в .local (kbrag/paths.py).
# Байт-код Python — туда же, а не папками __pycache__ по всему коду; дочерние
# команды получают это через переменную окружения.
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(ROOT / ".local" / "cache" / "python"))
sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
from kbrag.machine import FILE as SETTINGS_ENV  # noqa: E402
from kbrag.machine import value as _machine_value  # noqa: E402
from kbrag.paths import LOCAL  # noqa: E402

LOCAL.mkdir(exist_ok=True)


# Модель и адрес её сервера (vLLM) — одни на весь проект, из kbrag/llm.py (адрес
# можно поменять переменной KBRAG_LLM). Этот модуль лёгкий: только стандартная
# библиотека, поэтому импортируется и до перехода в .venv.
from kbrag.llm import DEFAULT_MODEL as MODEL  # noqa: E402
from kbrag.llm import LLM_HOST as LLM  # noqa: E402

# Имя ночной задачи прежней добычи статей в планировщике Windows: unschedule её убирает.
TASK_NAME = "KBRAG-acquire"

# Проверки кода — в tests/ (pytest). Прежние проверки разработчика лежат в .local/tests
# (в репозиторий не входят): если они есть на этой машине, идут следом.
TESTS_DIR = ROOT / "tests"
OLD_TESTS_DIR = LOCAL / "tests"
OLD_TESTS = {
    "smoke_test.py": "разбор PDF",
    "index_smoke_test.py": "база и поиск",
    "graph_smoke_test.py": "граф и факты",
    "chat_smoke_test.py": "чат-бот",
    "eval_smoke_test.py": "оценка",
    "tg_smoke_test.py": "Телеграм-бот",
}


# --------------------------------------------------------------------------- #
#  Окружение
# --------------------------------------------------------------------------- #
def _venv_python() -> Path | None:
    """Python из .venv проекта. Рядом с проектом тоже смотрим: туда его однажды уже ставили."""
    tail = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    for base in (ROOT / ".venv", ROOT.parent / ".venv"):
        if (base / tail).exists():
            return base / tail
    return None


def _in_venv(python: Path) -> bool:
    """Запущены ли мы уже из этого окружения. Сравниваем папки окружений, а не
    файлы python: в .venv он бывает ссылкой на системный, и файлы совпадут."""
    try:
        return Path(sys.prefix).resolve() == python.parent.parent.resolve()
    except OSError:
        return False


def _relaunch_in_venv() -> None:
    """Если запущены не из .venv — перезапуститься в нём с теми же аргументами."""
    python = _venv_python()
    if python is None or _in_venv(python):
        return
    try:
        code = subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


def _need_venv() -> None:
    if _venv_python() is None:
        print(
            "Не найдено окружение .venv в папке проекта. Создать один раз:\n"
            "    python -m venv .venv\n"
            "    .venv\\Scripts\\python -m pip install -r requirements.txt\n"
            "Подробно — в ЗАПУСК.md.",
            file=sys.stderr,
        )
        sys.exit(1)


def _run(args: list[str], **kwargs: Any) -> int:
    """Запустить команду из папки проекта. Ctrl+C прерывает её, а не запускатель."""
    try:
        return subprocess.call(args, cwd=ROOT, **kwargs)
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError:
        print(f"Не найдена программа: {args[0]}", file=sys.stderr)
        return 127


def _py(module: str, *args: str) -> int:
    return _run([sys.executable, "-m", module, *args])


# --------------------------------------------------------------------------- #
#  База в Docker
# --------------------------------------------------------------------------- #
def _docker(*args: str, quiet: bool = False) -> subprocess.CompletedProcess[str] | None:
    """docker compose. Модель (сервис llm) — в профиле llm: без --profile llm база
    поднимается одна, и загрузка статей не ждёт модель и не делит с ней видеокарту."""
    # Настройки машины (окно модели, видеопамять) — в .local/settings.env: их же
    # читает код (kbrag/machine.py), поэтому окно сервера и окно в коде совпадают.
    env_file = ["--env-file", str(SETTINGS_ENV)] if SETTINGS_ENV.is_file() else []
    command = ["docker", "compose", *env_file, *args]
    try:
        if quiet:
            return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        # --json: свой вывод docker — в stderr, иначе он смешается с JSON команды в stdout.
        json_mode = sys.stdout is sys.stderr
        return subprocess.run(
            command, cwd=ROOT, text=True, stdout=sys.__stderr__ if json_mode else None
        )
    except FileNotFoundError:
        print(
            "Docker не найден. Установите и запустите Docker Desktop — " "см. ЗАПУСК.md.",
            file=sys.stderr,
        )
        return None


def _ensure_database() -> bool:
    """Поднять базу, если она не запущена. Ложь — если Docker недоступен."""
    state = _docker("ps", "--format", "{{.State}}", quiet=True)
    if state is None:
        return False
    if state.returncode != 0:
        print(
            "Docker не отвечает. Запустите Docker Desktop и дождитесь, "
            "пока значок кита перестанет мигать.",
            file=sys.stderr,
        )
        return False
    if "running" not in (state.stdout or ""):
        print("База не запущена. Поднимаю...")
        up = _docker("up", "-d")
        if up is None or up.returncode != 0:
            return False
        time.sleep(5)
    return True


def _llm_up(timeout: int = 5) -> bool:
    """Отвечает ли сервер модели (vLLM): /v1/models."""
    try:
        with urllib.request.urlopen(f"{LLM}/v1/models", timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


VISION = "http://localhost:8101"  # сервис vision (kbrag/chunks/figures.py, VISION_PORT)


def _vision_mode() -> str:
    """Как модель со зрением делит видеокарту: turns, shared, single (chunks/figures.py)."""
    mode = _machine_value("KBRAG_VISION_MODE", "turns").lower()
    return mode if mode in ("turns", "shared", "single") else "turns"


def _answers(url: str, timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def _with_vision(run: Any, limit: int = 1800) -> int:
    """run() — пока модель со зрением отвечает. turns: основная модель гасится на это
    время (видеокарта одна), vision — поднимается, потом всё обратно. single: рисунки
    описывает основная модель. Модель со зрением задана адресом (KBRAG_FIGURE_LLM) —
    её не трогаем."""
    mode = _vision_mode()
    if os.environ.get("KBRAG_FIGURE_LLM", "").strip() or _machine_value("KBRAG_FIGURE_LLM", ""):
        return int(run())
    if mode == "single":
        return int(run()) if _ensure_llm() else 1
    if _answers(VISION):
        return int(run())
    was_up = mode == "turns" and _llm_up()
    if was_up:
        print("Модель ответов гашу на время описания рисунков: видеокарта одна.")
        _docker("--profile", "llm", "stop", "llm")
    print("Поднимаю модель со зрением (сервис vision)...")
    up = _docker("--profile", "vision", "up", "-d", "vision")
    code = 1
    if up is not None and up.returncode == 0:
        started = time.monotonic()
        while time.monotonic() - started < limit and not _answers(VISION):
            time.sleep(5)
        if _answers(VISION):
            print(f"Модель со зрением готова ({time.monotonic() - started:.0f} с).")
            code = int(run())
        else:
            print(
                "Модель со зрением не поднялась. Журнал: docker logs kbrag-vision", file=sys.stderr
            )
    if mode == "turns":
        _docker("--profile", "vision", "stop", "vision")
        if was_up:
            _ensure_llm(wait=False)
    return code


def _ensure_llm(wait: bool = True, limit: int = 1800) -> bool:
    """Поднять модель в vLLM, если не отвечает. wait — дождаться готовности: модель
    грузится 1–2 минуты, а в первый раз ещё и скачивается (около 10 ГБ)."""
    if _llm_up():
        return True
    if not LLM.startswith(("http://localhost", "http://127.0.0.1")):
        print(f"Модель на {LLM} не отвечает — её сервер запускают там.", file=sys.stderr)
        return False
    print("Модель не запущена. Поднимаю vLLM...")
    up = _docker("--profile", "llm", "up", "-d", "llm")
    if up is None or up.returncode != 0:
        return False
    if not wait:
        print("Модель грузится 1–2 минуты; пока она грузится, работает поиск.")
        return True
    started = time.monotonic()
    while time.monotonic() - started < limit:
        if _llm_up():
            print(f"Модель готова ({time.monotonic() - started:.0f} с).")
            return True
        time.sleep(5)
    print("Модель не поднялась. Журнал: docker logs kbrag-llm", file=sys.stderr)
    return False


# --------------------------------------------------------------------------- #
#  Команды
# --------------------------------------------------------------------------- #
def cmd_start(opts: argparse.Namespace) -> int:
    up = _docker("--profile", "llm", "up", "-d")
    if up is None or up.returncode != 0:
        return 1
    print("\nЖду, пока база и модель будут готовы...")
    time.sleep(5)
    _docker("--profile", "llm", "ps")
    return 0 if _ensure_llm() else 1


def cmd_stop(opts: argparse.Namespace) -> int:
    done = _docker("--profile", "llm", "--profile", "vision", "stop")
    if done is not None and done.returncode == 0:
        print("База и модель остановлены. Данные на месте, видеокарта свободна.")
        return 0
    return 1


def cmd_status(opts: argparse.Namespace) -> int:
    done = _docker("--profile", "llm", "--profile", "vision", "ps")
    print(f"Модель на {LLM}: " + ("отвечает" if _llm_up() else "не отвечает"))
    print(f"Модель со зрением на {VISION}: " + ("отвечает" if _answers(VISION) else "не запущена"))
    return 0 if done is not None and done.returncode == 0 else 1


def cmd_init(opts: argparse.Namespace) -> int:
    return _py("kbrag.index.cli", "init") if _ensure_database() else 1


def cmd_stats(opts: argparse.Namespace) -> int:
    return _py("kbrag.index.cli", "stats") if _ensure_database() else 1


def cmd_import(opts: argparse.Namespace) -> int:
    if not opts.text:
        print('Что импортировать: python kbrag.py import "D:/статьи"', file=sys.stderr)
        return 1
    if not _ensure_database():
        return 1
    args = ["import", opts.text, "--device", "cuda" if _has_cuda() else "auto"]
    args += [flag for flag, on in _import_flags(opts) if on]
    return _then_figures(opts, _run_import(opts, args))


def _run_import(opts: argparse.Namespace, args: list[str]) -> int:
    """shared: рисунки описывает сам импорт — модель со зрением должна отвечать."""
    if opts.figures and _vision_mode() == "shared":
        return _with_vision(lambda: _py("kbrag.library.cli", *args))
    return _py("kbrag.library.cli", *args)


def cmd_reparse(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    args = ["reparse", "--device", "cuda" if _has_cuda() else "auto"]
    if opts.text:
        args += ["--doc", opts.text]
    args += [flag for flag, on in _import_flags(opts) if on and flag != "--ask"]
    return _then_figures(opts, _run_import(opts, args), opts.text)


def _import_flags(opts: argparse.Namespace) -> list[tuple[str, bool]]:
    # Рисунки описываются после разбора (_then_figures): docling и модель со зрением
    # на одной видеокарте разом не помещаются. shared — места на обе, описывает импорт.
    during = opts.figures and _vision_mode() == "shared"
    return [
        ("--ask", opts.ask),
        ("--force", opts.force),
        ("--figures", during),
        ("--ocr", opts.ocr),
    ]


def _then_figures(opts: argparse.Namespace, code: int, doc: str = "") -> int:
    """import/reparse --figures: после разбора описать неописанные рисунки (figures)."""
    if code != 0 or not opts.figures or _vision_mode() == "shared":
        return code
    print("\nРисунки — моделью со зрением:")
    return cmd_figures(argparse.Namespace(text=doc, redo=False))


def cmd_figures(opts: argparse.Namespace) -> int:
    """Описать рисунки статей моделью со зрением; потом — загрузить статьи заново (уже
    без неё: векторам нужна видеокарта)."""
    if not _ensure_database():
        return 1
    args = ["--no-ingest"] + (["--doc", opts.text] if opts.text else [])
    args += ["--redo"] if opts.redo else []
    code = _with_vision(lambda: _py("kbrag.library.describe", *args))
    device = "cuda" if _has_cuda() else "auto"
    loaded = _py("kbrag.library.describe", "--ingest-pending", "--device", device)
    return code or loaded


def _has_cuda() -> bool:
    """Есть ли видеокарта NVIDIA: docling разбирает на ней в разы быстрее."""
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except OSError:
        return False


def cmd_ingest(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    args = ["ingest"]
    if opts.force:
        args.append("--force")
    return _py("kbrag.index.cli", *args)


def cmd_tidy(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    args = ["tidy"]
    return _py("kbrag.index.cli", *args)


def cmd_search(opts: argparse.Namespace) -> int:
    if not opts.text:
        print('Нужен запрос: python kbrag.py search "методы измерения"', file=sys.stderr)
        return 1
    if not _ensure_database():
        return 1
    args = ["search", opts.text, "--limit", str(opts.limit or 10)]
    return _py("kbrag.index.cli", *args, *(["--json"] if opts.json else []))


def cmd_web(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    _ensure_llm(wait=False)
    return _py(
        "kbrag.web.server",
        "--model",
        MODEL,
        *(["--tg"] if opts.tg else []),
    )


def cmd_serve(opts: argparse.Namespace) -> int:
    """Тот же сервер, что web, но браузер не открывается (только адреса /api)."""
    if not _ensure_database():
        return 1
    _ensure_llm(wait=False)
    args = ["--no-browser", "--model", MODEL]
    return _py("kbrag.web.server", *args, *(["--tg"] if opts.tg else []))


def cmd_graph(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    if opts.new:
        return _py("kbrag.graph.cli", "report")
    if opts.recheck:
        return _py("kbrag.graph.cli", "recheck")
    if opts.names:
        return _py("kbrag.graph.cli", "names", *(["--redo"] if opts.redo else []))
    workers = ["--workers", str(opts.workers)] if opts.workers else []
    if not _ensure_llm():
        return 1
    if opts.confirm:
        extra = ["--limit", str(opts.limit)] if opts.limit else []
        return _py("kbrag.graph.cli", "confirm", "--model", MODEL, *extra, *workers)
    args = ["build", "--model", MODEL] + (["--redo"] if opts.redo else []) + workers
    if opts.limit:
        args += ["--limit", str(opts.limit)]
    return _py("kbrag.graph.cli", *args)


def cmd_synonyms(opts: argparse.Namespace) -> int:
    if not _ensure_database() or not _ensure_llm():
        return 1
    return _py(
        "kbrag.graph.cli",
        "synonyms",
        "--model",
        MODEL,
    )


def cmd_verify(opts: argparse.Namespace) -> int:
    print(
        "Проверки понятий больше нет: граф строят факты с цитатами, каждый проверяет код "
        "сразу. Строить граф: python kbrag.py graph"
    )
    return 0


def cmd_tg(opts: argparse.Namespace) -> int:
    if opts.check:  # проверка токена и доступа: база и модели не нужны
        return _py("kbrag.tg.bot", "--check")
    if not _ensure_database():
        return 1
    _ensure_llm(wait=False)
    return _py("kbrag.tg.bot", "--model", MODEL)


def cmd_dataset(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    return _py(
        "kbrag.graph.cli",
        "dataset",
        *(["--name", opts.text] if opts.text else []),
        *(["--json"] if opts.json else []),
    )


def cmd_questions(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    return _py(
        "kbrag.graph.cli",
        "questions",
        "--model",
        MODEL,
        *(["--limit", str(opts.limit)] if opts.limit else []),
        *(["--no-llm"] if opts.no_llm else []),
    )


def cmd_ask(opts: argparse.Namespace) -> int:
    if not opts.text:
        print(
            'Нужен вопрос: python kbrag.py ask "какие методы описаны в статьях?"', file=sys.stderr
        )
        return 1
    if not _ensure_database() or not _ensure_llm():
        return 1
    return _py("kbrag.chat.cli", opts.text, "--model", MODEL, *(["--json"] if opts.json else []))


def cmd_eval(opts: argparse.Namespace) -> int:
    if not _ensure_database():
        return 1
    if not opts.only_search and not _ensure_llm():
        return 1
    args = ["--model", MODEL]
    if opts.only_search:
        args.append("--only-search")
    if opts.only:
        args += ["--only", opts.only]
    if opts.questions:
        args += ["--questions", opts.questions]
    if opts.no_graph:
        args.append("--no-graph")
    return _py("kbrag.evaluation", *args)


def cmd_lint(opts: argparse.Namespace) -> int:
    """ruff, black --check и mypy --strict.
    Пакет kbrag и файл kbrag.py проверяются mypy отдельно: у них одно имя."""
    # Прежние проверки лежат в .local (исключена в pyproject) — их файлы передаются явно.
    old = [str(p) for p in sorted(OLD_TESTS_DIR.rglob("*.py"))]
    steps = [
        ("ruff", ["-m", "ruff", "check", "kbrag", "kbrag.py", "tests", *old]),
        ("black", ["-m", "black", "--check", "kbrag", "kbrag.py", "tests", *old]),
        ("mypy --strict: пакет", ["-m", "mypy", "-p", "kbrag"]),
        ("mypy --strict: kbrag.py", ["-m", "mypy", "kbrag.py"]),
    ]
    failed = []
    for title, args in steps:
        print(f"== {title}", flush=True)
        if subprocess.run([sys.executable, *args], cwd=ROOT).returncode != 0:
            failed.append(title)
    print("\nВсе проверки кода прошли." if not failed else f"\nНе прошло: {', '.join(failed)}")
    return 1 if failed else 0


def cmd_test(opts: argparse.Namespace) -> int:
    """Проверки кода. Идут на выдуманных примерах, база и модель не нужны и не трогаются.

    Сначала tests/ (pytest), потом прежние проверки из .local/tests, если они есть. На
    экран — коротко; подробности — у того, что не прошло (или все: --verbose)."""
    print("Проверка кода на выдуманных примерах — ваша база, статьи и граф не трогаются.\n")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    failed = []
    try:
        code = subprocess.run(
            [sys.executable, "-m", "pytest", str(TESTS_DIR), "-v" if opts.verbose else "-q"],
            cwd=ROOT,
            env=env,
        ).returncode
    except KeyboardInterrupt:
        return 130
    if code != 0:
        failed.append("tests/")
    for file, name in OLD_TESTS.items():
        if not (OLD_TESTS_DIR / file).is_file():
            continue
        test = str(OLD_TESTS_DIR / file)
        if opts.verbose:
            code = _run([sys.executable, test])
            out = ""
        else:
            try:
                proc = subprocess.run(
                    [sys.executable, test],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
            except KeyboardInterrupt:
                return 130
            code, out = proc.returncode, proc.stdout
        total = re.findall(r"Итого: (\d+) пройдено", out)
        mark = "OK  " if code == 0 else "СБОЙ"
        print(f"  [{mark}] {name}" + (f" — проверок {total[-1]}" if total and code == 0 else ""))
        if code != 0:
            failed.append(name)
            if out:
                print("\n".join("        " + line for line in out.rstrip().splitlines()[-40:]))
    if failed:
        print(
            "\nНе прошли: " + ", ".join(failed) + ". Пришлите этот вывод целиком.", file=sys.stderr
        )
        return 1
    print("\nВсе проверки прошли.")
    return 0


# --------------------------------------------------------------------------- #
#  Прежняя ночная добыча
# --------------------------------------------------------------------------- #
def cmd_unschedule(opts: argparse.Namespace) -> int:
    """Убрать ночную задачу прежней добычи статей: самой добычи больше нет, и задача
    падала бы каждую ночь."""
    if os.name != "nt":
        print("Уберите строку с kbrag.py из crontab -e.")
        return 0
    done = _run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"])
    if done == 0:
        print("Ночная добыча убрана из планировщика.")
    return done


def cmd_help(opts: argparse.Namespace) -> int:
    print(__doc__)
    return 0


COMMANDS = {
    "start": cmd_start,
    "stop": cmd_stop,
    "status": cmd_status,
    "init": cmd_init,
    "stats": cmd_stats,
    "import": cmd_import,
    "reparse": cmd_reparse,
    "figures": cmd_figures,
    "ingest": cmd_ingest,
    "tidy": cmd_tidy,
    "search": cmd_search,
    "web": cmd_web,
    "serve": cmd_serve,
    "graph": cmd_graph,
    "verify": cmd_verify,
    "synonyms": cmd_synonyms,
    "ask": cmd_ask,
    "eval": cmd_eval,
    "dataset": cmd_dataset,
    "facts": cmd_dataset,
    "questions": cmd_questions,
    "tg": cmd_tg,
    "test": cmd_test,
    "lint": cmd_lint,
    "unschedule": cmd_unschedule,
    "help": cmd_help,
}
# Эти работают без .venv: им нужны только Docker и планировщик.
NO_VENV = {"start", "stop", "status", "help", "unschedule"}


def parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python kbrag.py", add_help=False)
    p.add_argument("command", nargs="?", default="help")
    p.add_argument("text", nargs="?", default="")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--llm", action="store_true")
    p.add_argument("--only-search", action="store_true")
    p.add_argument("--only", default="")
    p.add_argument("--questions", default="")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--new", action="store_true")
    p.add_argument("--recheck", action="store_true")
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--names", action="store_true")
    p.add_argument("--figures", action="store_true")
    p.add_argument("--ocr", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--tg", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--ask", action="store_true")
    p.add_argument("--kb", default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("-h", "--help", action="store_true")
    return p.parse_args(argv)


@dataclass(frozen=True)
class Passthrough:
    """Команда, у которой свои ключи: модуль разбирает их сам, запускатель только готовит
    окружение (база, модель) и передаёт аргументы как есть."""

    module: str
    database: bool = True  # поднять базу
    llm: bool = False  # поднять модель (если не --show)
    quiet: bool = False  # говорит через stdout (MCP) — запускатель молчит
    default: tuple[str, ...] = ()  # аргументы, если их нет
    no_database: tuple[str, ...] = ()  # подкоманды, которым база не нужна


PASSTHROUGH = {
    "kb": Passthrough("kbrag.kbtree", default=("list",), no_database=("list",)),
    "access": Passthrough("kbrag.acl", database=False, default=("list",)),
    "summaries": Passthrough("kbrag.index.summaries", llm=True),
    "formula": Passthrough("kbrag.index.formulas"),
    "figure": Passthrough("kbrag.index.figures"),
    "mcp": Passthrough("kbrag.mcp_server", database=False, quiet=True),
}


def run_passthrough(name: str, args: list[str]) -> int:
    """Сквозная команда (PASSTHROUGH): база, модель — и модуль со своими ключами."""
    spec = PASSTHROUGH[name]
    _need_venv()
    args = args or list(spec.default)
    if "--json" in args and not spec.quiet:
        sys.stdout = sys.stderr  # сообщения запускателя — мимо JSON
    needs_db = spec.database and not (args[:1] and args[0] in spec.no_database)
    if needs_db and not _ensure_database():
        return 1
    if spec.llm and "--show" not in args and not _ensure_llm():
        return 1
    return _py(spec.module, *args)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] and argv[0] in PASSTHROUGH:
        return run_passthrough(argv[0], argv[1:])
    opts = parse(argv)
    if opts.json:
        # Свои сообщения запускателя («поднимаю базу…») — в stderr: в stdout только JSON
        # дочерней команды (она пишет в унаследованный stdout, не в этот объект).
        sys.stdout = sys.stderr
    if opts.help:
        opts.command = "help"
    command = COMMANDS.get(opts.command.lower())
    if command is None:
        print(f"Нет такой команды: {opts.command}. Список — python kbrag.py help", file=sys.stderr)
        return 1
    if opts.command.lower() not in NO_VENV:
        _need_venv()
    if opts.kb is not None:
        # Узел дерева баз — всем командам сразу: модули читают его из KBRAG_KB (kb.py).
        os.environ["KBRAG_KB"] = opts.kb
    return command(opts)


if __name__ == "__main__":
    _relaunch_in_venv()
    raise SystemExit(main())
