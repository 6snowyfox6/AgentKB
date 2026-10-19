"""Дерево баз (kb.py): путь, наследование настроек и файлов, промпты, синонимы, правила."""

from __future__ import annotations

from typing import Any

import pytest

from kbrag import kb

ROOT_YAML = """
база_по_умолчанию: geo.ore
название: Все
тема:
  пользователь: исследователя
  область: теме базы
  язык_ответа: по-русски
промпты: {}
"""


def test_check_path() -> None:
    assert kb.check_path("geology.ore.anabar") == "geology.ore.anabar"
    assert kb.check_path(".") == ""
    assert kb.check_path(" geo. ") == "geo"
    for bad in ("гео", "geo..ore", "geo ore", "geo-ore!"):
        with pytest.raises(kb.KBError):
            kb.check_path(bad)


def test_ancestors() -> None:
    assert kb.ancestors("a.b.c") == ["a", "a.b", "a.b.c"]
    assert kb.ancestors("") == []


def test_settings_inherit_by_key(tree: Any) -> None:
    tree("", "kb.yaml", ROOT_YAML)
    tree("geo", "kb.yaml", "название: Геология\nтема:\n  пользователь: геолога\n")
    node = kb.get("geo.ore.x")  # своей папки нет — всё от предков
    assert node.topic["пользователь"] == "геолога"  # ближний главнее
    assert node.topic["область"] == "теме базы"  # чего нет у ветки — от корня
    assert node.title == "geo.ore.x"  # название не наследуется
    assert kb.get("geo").title == "Геология"
    assert node.local.parts[-3:] == ("geo", "ore", "x")


def test_default_path(tree: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    tree("", "kb.yaml", ROOT_YAML)
    assert kb.default_path() == "geo.ore"
    monkeypatch.setenv(kb.ENV, "phys")
    assert kb.default_path() == "phys"
    monkeypatch.setenv(kb.ENV, ".")
    assert kb.default_path() == ""


def test_prompt_words_and_override(tree: Any) -> None:
    tree("", "kb.yaml", ROOT_YAML)
    tree(
        "geo", "kb.yaml", "тема:\n  пользователь: геолога\nпромпты:\n  разбор: 'Своё: {question}'\n"
    )
    node = kb.get("geo")
    assert node.prompt("ответ", "Ты — помощник {пользователь}.") == "Ты — помощник геолога."
    assert node.prompt("разбор", "из кода {question}", question="Q") == "Своё: Q"
    assert node.prompt("x", 'JSON: {{"a": 1}} {n}', n=2) == 'JSON: {"a": 1} 2'
    with pytest.raises(kb.KBError, match="нет_такого"):
        node.prompt("y", "{нет_такого}")


def test_files_nearest_and_all(tree: Any) -> None:
    root_rules = tree("", "graph-rules.txt", "корень")
    geo_rules = tree("geo", "graph-rules.txt", "геология")
    assert kb.get("geo.ore").file("graph-rules.txt") == geo_rules
    assert kb.get("phys").file("graph-rules.txt") == root_rules
    assert kb.get("geo.ore").files("graph-rules.txt") == [root_rules, geo_rules]
    assert kb.get("phys").file("eval-questions.yaml") is None


def test_synonyms_merge_nearest_wins(tree: Any) -> None:
    from kbrag.graph import synonyms

    tree("", "synonyms.yaml", "имена:\n  золото: [gold, Au]\n  кварц: [quartz]\n")
    tree("geo", "synonyms.yaml", "имена:\n  самородное золото: [gold]\n  пирит: [pyrite]\n")
    syn = synonyms.load(kb=kb.get("geo.ore"))
    assert syn.name("pyrite") == "пирит"  # из ветки
    assert syn.name("quartz") == "кварц"  # из корня
    assert syn.name("gold") == "самородное золото"  # у одного написания — ближний
    assert syn.name("Au") == "золото"
    assert synonyms.load(kb=kb.get("phys")).name("gold") == "золото"


def test_synonyms_error_names_file(tree: Any) -> None:
    from kbrag.graph import synonyms

    tree("geo", "synonyms.yaml", "имя:\n  золото: [gold]\n")  # раздела «имя» не бывает
    with pytest.raises(synonyms.SynonymsError, match="synonyms.yaml"):
        synonyms.load(kb=kb.get("geo"))


def test_rules_nearest(tree: Any) -> None:
    from kbrag.graph import facts

    tree("", "graph-rules.txt", "# комментарий\n" + "общие правила " * 10)
    tree("geo", "graph-rules.txt", "геологические правила " * 10)
    assert facts.load_rules(kb=kb.get("geo.ore")).startswith("геологические")
    assert facts.load_rules(kb=kb.get("phys")).startswith("общие")


def test_repo_config_fills_every_prompt() -> None:
    """Все промпты кода собираются с настройками репозитория — у любого узла."""
    from kbrag.chat import messages, plan, retrieval
    from kbrag.chunks import figures
    from kbrag.graph import confirm, questions, synonyms

    kb._CACHE.clear()
    for path in ("", "geology", "geology.ore.anabar", "physics.hadron"):
        node = kb.get(path)
        templates = [
            ("ответ", messages.SYSTEM, {}),
            ("без_базы", messages.GENERAL_SYSTEM, {}),
            ("разбор.система", plan.PLAN_SYSTEM, {}),
            (
                "разбор",
                plan.PLAN_USER,
                {"question": "Q", "context": "", "n": 3, "parts": 3, "languages": "x"},
            ),
            ("отбор.система", retrieval.JUDGE_SYSTEM, {}),
            (
                "отбор",
                retrieval.JUDGE_USER,
                {
                    "question": "Q",
                    "parts": "",
                    "taken": "",
                    "candidates": "",
                    "n": 3,
                    "done": "",
                    "languages": "x",
                },
            ),
            ("рисунок", figures.PROMPT, {"caption": "c"}),
            ("подтверждение.система", confirm.SYSTEM, {}),
            ("вопросы.система", questions.SYSTEM, {}),
            ("вопросы", questions.USER, {"src": "a", "relation": "b", "dst": "c", "quote": "q"}),
            ("синонимы.система", synonyms.JUDGE_SYSTEM, {}),
            ("синонимы", synonyms.JUDGE_USER, {"pairs": "1. a — b"}),
        ]
        for name, template, values in templates:
            text = node.prompt(name, template, **values)
            assert "{" not in text.replace('{"', "").replace("{{", ""), (path, name)
        assert node.text("отказ_не_по_теме")
    assert "геолога" in kb.get("geology.ore.anabar").prompt("ответ", messages.SYSTEM)
    assert "геолог" not in kb.get("physics").prompt("ответ", messages.SYSTEM)
