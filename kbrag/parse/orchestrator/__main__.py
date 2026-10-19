"""Список случаев оркестратора таблицей: python -m kbrag.parse.orchestrator"""

import sys

from .cases import table

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
print(table())
