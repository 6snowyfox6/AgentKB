"""Где что лежит.

В репозитории — код, config, docs и проверки кода (tests). Всё, что живёт только на
этой машине, — в одной папке .local (в git не идёт): разобранные статьи, логи, ключи
и токены, файлы, которые пересоздают команды (questions, gaps, synonyms), кэши.
Только стандартная библиотека: kbrag.py импортирует модуль и до перехода в .venv."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
LOCAL = ROOT / ".local"
# Разобранные статьи: записи, текст, фрагменты, разметка docling, рисунки (library/
# store.py). Имя папки — от прежней добычи: так статьи, добытые ею, остаются на месте.
ARTICLES = LOCAL / "acquired"
LOGS = LOCAL / "logs"
