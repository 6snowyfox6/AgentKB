"""Модели из кэша — без сети (embed.offline_first); нет в кэше — качаются как обычно."""

from __future__ import annotations

import pytest
from huggingface_hub import constants

from kbrag.index.embed import offline_first


def test_cached_model_loads_offline() -> None:
    seen: list[bool] = []
    assert offline_first(lambda: seen.append(constants.HF_HUB_OFFLINE) or "модель") == "модель"
    assert seen == [True]


def test_missing_model_downloads_online() -> None:
    before = constants.HF_HUB_OFFLINE
    seen: list[bool] = []

    def load() -> str:
        seen.append(constants.HF_HUB_OFFLINE)
        if constants.HF_HUB_OFFLINE:
            raise OSError("нет в кэше")
        return "скачана"

    assert offline_first(load) == "скачана"
    assert seen == [True, before]
    assert constants.HF_HUB_OFFLINE == before  # настройка вернулась


def test_other_errors_are_not_hidden() -> None:
    def load() -> str:
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        offline_first(load)
