"""Общее для проверок: корень проекта в пути импорта, эталонные PDF, дерево баз во
временной папке.

Проверки идут без базы, модели и видеокарты: Postgres и vLLM подменяются
заглушками, docling — только там, где он есть (иначе проверка пропускается).

    python kbrag.py test          все проверки
    python -m pytest tests -q      то же напрямую
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "data"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def no_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ни одна проверка не грузит настоящие модели: реранкер выключен (кому он нужен —
    подставляет свой), настройки машины — не из .local/settings.env этой машины."""
    from kbrag import machine

    monkeypatch.setenv("KBRAG_RERANKER", "off")
    monkeypatch.setattr(machine, "FILE", Path("нет-такого-файла.env"))


@pytest.fixture
def data() -> Path:
    return DATA


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Дерево настроек баз во временной папке: tree(путь, файл, текст) кладёт файл в
    папку узла и возвращает его. Кэш узлов сбрасывается, KBRAG_KB не мешает."""
    from kbrag import kb

    root = tmp_path / "kbs"
    root.mkdir()
    monkeypatch.setattr(kb, "CONFIG_DIR", root)
    monkeypatch.setattr(kb, "LOCAL_DIR", tmp_path / "local")
    monkeypatch.delenv(kb.ENV, raising=False)
    kb._CACHE.clear()

    def put(path: str, name: str, text: str) -> Path:
        folder = root.joinpath(*path.split(".")) if path else root
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / name
        file.write_text(text, encoding="utf-8")
        kb._CACHE.clear()
        return file

    put.root = root  # type: ignore[attr-defined]
    yield put
    kb._CACHE.clear()


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Запускатель kbrag.py модулем: команды можно звать без процесса, а Docker, модель и
    паузы подменять. Настройки машины — значения по умолчанию."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("kbrag_launcher", ROOT / "kbrag.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)  # dataclass ищет модуль здесь
    spec.loader.exec_module(module)
    monkeypatch.delenv("KBRAG_FIGURE_LLM", raising=False)
    monkeypatch.setattr(module, "_machine_value", lambda name, default: default)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return module
