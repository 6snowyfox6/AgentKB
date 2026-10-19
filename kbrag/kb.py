"""Дерево баз знаний: узел, его настройки и файлы — с наследованием от корня.

Узел — путь в дереве, как ltree в Postgres: `geology.ore.anabar`. Пустой путь — корень,
то есть все базы сразу. У каждого узла — своя папка настроек в config/kbs, по папке
на шаг пути, и своя папка данных в .local/kbs:

    config/kbs/kb.yaml                      корень: промпты, тема по умолчанию
    config/kbs/geology/kb.yaml              геология: тема, синонимы, правила графа
    config/kbs/geology/ore/anabar/...       Анабарский щит: вопросы оценки
    .local/kbs/geology/ore/anabar/          предложения синонимов, таблицы фактов…

Что ближе к узлу, то главнее:

* kb.yaml — сливается от корня к узлу по ключам (вложенные словари — тоже по ключам);
* synonyms.yaml — тоже сливается, у одного имени побеждает ближний (graph/synonyms.py);
* graph-rules.txt, eval-questions.yaml — берётся ближайший файл (KB.file).

Промпты модели живут в коде шаблонами без слов предметной области: «вопрос
{пользователь}», «статей по {область}». Слова берутся из раздела «тема» в kb.yaml, а
любой шаблон узел может заменить целиком — раздел «промпты» (KB.prompt).
Название базы («название») не наследуется: у безымянной базы — её путь."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import CONFIG, LOCAL

CONFIG_DIR = CONFIG / "kbs"
LOCAL_DIR = LOCAL / "kbs"
SETTINGS = "kb.yaml"
DEFAULT_KEY = "база_по_умолчанию"  # в корневом kb.yaml: с какой базой работать без --kb
ENV = "KBRAG_KB"  # то же переменной окружения — главнее файла

# Метка пути — как у ltree: латиница, цифры и подчёркивание.
_PATH = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*")
_LABEL = re.compile(r"[A-Za-z0-9_]+")


class KBError(ValueError):
    """Путь или настройки базы не годятся. Текст — для человека."""


def check_path(path: str | None) -> str:
    """Путь узла как в ltree; пустой — корень. Не годится — KBError."""
    path = (path or "").strip().strip(".")
    if path and not _PATH.fullmatch(path):
        raise KBError(
            f"путь базы «{path}» не годится: метки через точку, в метке — латиница, "
            "цифры и _, например geology.ore.anabar"
        )
    return path


def ancestors(path: str) -> list[str]:
    """Узел и все его предки, от верхнего: geology, geology.ore, geology.ore.anabar."""
    labels = check_path(path).split(".") if path else []
    return [".".join(labels[: i + 1]) for i in range(len(labels))]


class _Strict(dict[str, Any]):
    def __missing__(self, key: str) -> Any:
        raise KBError(f"в настройках базы нет значения «{key}» (раздел «тема» в kb.yaml)")


def fill(template: str, values: dict[str, Any]) -> str:
    """Шаблон с {подстановками}; фигурные скобки JSON в шаблоне — двойные."""
    return template.format_map(_Strict(values))


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read(path: Path) -> dict[str, Any]:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as exc:
        raise KBError(f"{path}: не читается — {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise KBError(f"{path}: нужен словарь «ключ: значение»")
    return data


@dataclass(frozen=True)
class KB:
    path: str  # "" — корень
    settings: dict[str, Any] = field(default_factory=dict)  # kb.yaml, слитые от корня
    dirs: tuple[Path, ...] = ()  # папки настроек от корня к узлу (есть они или нет)
    own: dict[str, Any] = field(default_factory=dict)  # kb.yaml самого узла, без предков

    @property
    def labels(self) -> list[str]:
        return self.path.split(".") if self.path else []

    @property
    def title(self) -> str:
        """Название — только своё: у безымянной базы — её путь, а не имя родителя."""
        return str(self.own.get("название") or self.path or "все базы")

    @property
    def topic(self) -> dict[str, Any]:
        return dict(self.settings.get("тема") or {})

    @property
    def local(self) -> Path:
        """Папка данных узла на этой машине (.local/kbs/…); корень — .local/kbs."""
        return LOCAL_DIR.joinpath(*self.labels)

    @property
    def own_dir(self) -> Path:
        """Папка настроек самого узла — туда пишется то, что правят для этой базы."""
        return self.dirs[-1] if self.dirs else CONFIG_DIR

    def prompt(self, name: str, default: str, **values: Any) -> str:
        """Промпт модели: шаблон узла (раздел «промпты») или из кода, со словами темы.

        values — подстановки самого задания (вопрос, фрагменты…), они главнее темы."""
        template = (self.settings.get("промпты") or {}).get(name) or default
        return fill(str(template), {**self.topic, **values})

    def text(self, name: str) -> str:
        """Слово или фраза темы как есть (для строк вне промптов: отказ, подсказки)."""
        return str(_Strict(self.topic)[name])

    def files(self, name: str) -> list[Path]:
        """Файлы с этим именем по пути, от корня к узлу — только те, что есть."""
        return [d / name for d in self.dirs if (d / name).is_file()]

    def file(self, name: str) -> Path | None:
        """Ближайший к узлу файл с этим именем; нигде нет — None."""
        found = self.files(name)
        return found[-1] if found else None


_CACHE: dict[str, tuple[tuple[Any, ...], KB]] = {}
_LOCK = threading.Lock()


def get(path: str | None = None) -> KB:
    """Узел дерева с настройками. Без пути — база по умолчанию (default_path).

    Файлы перечитываются, только если их поменяли: настройки можно править, не
    перезапуская страницу и бота."""
    path = default_path() if path is None else check_path(path)
    dirs = (CONFIG_DIR, *(CONFIG_DIR.joinpath(*p.split(".")) for p in ancestors(path)))
    files = [d / SETTINGS for d in dirs]
    stamp = tuple(f.stat().st_mtime if f.is_file() else None for f in files)
    with _LOCK:
        hit = _CACHE.get(path)
        if hit and hit[0] == stamp:
            return hit[1]
    settings: dict[str, Any] = {}
    own: dict[str, Any] = {}
    for f in files:
        own = _read(f) if f.is_file() else {}
        settings = _merge(settings, own)
    kb = KB(path, settings, dirs, own)
    with _LOCK:
        _CACHE[path] = (stamp, kb)
    return kb


def folder(path: str) -> Path:
    """Папка настроек узла (есть она или нет): config/kbs/<метка>/<метка>/…"""
    path = check_path(path)
    return CONFIG_DIR.joinpath(*path.split(".")) if path else CONFIG_DIR


def configured() -> list[str]:
    """Узлы, у которых есть папка настроек, — по порядку пути. Папки, имя которых не
    годится в метку пути (например, «_старое»), пропускаются вместе с содержимым."""
    out: list[str] = []

    def walk(directory: Path, prefix: str) -> None:
        for child in sorted(p for p in directory.iterdir() if p.is_dir()):
            if not _LABEL.fullmatch(child.name):
                continue
            path = f"{prefix}.{child.name}" if prefix else child.name
            out.append(path)
            walk(child, path)

    if CONFIG_DIR.is_dir():
        walk(CONFIG_DIR, "")
    return out


def default_path() -> str:
    """С какой базой работать, если не сказано: KBRAG_KB, иначе корневой kb.yaml."""
    if os.environ.get(ENV, "").strip():
        return check_path(os.environ[ENV])
    root = CONFIG_DIR / SETTINGS
    value = _read(root).get(DEFAULT_KEY) if root.is_file() else None
    return check_path(str(value or ""))


def languages(kb: KB, joiner: str) -> str:
    """«по-русски и по-английски» из списка тема.языки_запросов."""
    raw = kb.topic.get("языки_запросов") or ["по-русски", "по-английски"]
    items = [str(x) for x in (raw if isinstance(raw, list) else [raw])]
    return joiner.join(items)
