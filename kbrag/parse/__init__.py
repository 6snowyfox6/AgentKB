"""Разбор PDF: от файла до текста с разметкой, номерами страниц и формулами.

Порядок работы (pipeline.process_document):

    orchestrator/profile.py   зонд: что за документ (скан, вариант переносов)
    parsers/                  docling в отдельном процессе; не справился — PyMuPDF4LLM
    validate.py               проверка результата, сверка с эталоном
    orchestrator/             найденные проблемы PDF — случаи с проверкой и обработкой
    kbrag.chunks             нарезка на фрагменты (отдельный пакет)

Где что:

    pipeline.py      конвейер одного документа и папки, журнал JSONL
    cli.py           разбор папки PDF: python -m kbrag.parse.cli
    config.py        настройки разбора и нарезки
    models.py        ParseInput, ParsedDoc, ValidationReport, статусы
    validate.py      проверка результата
    diag.py          диагностика сбоев: причина, место, код выхода, стек процесса
    parsers/         docling.py, worker.py, pymupdf.py, structure.py
    orchestrator/    cases.py (список случаев), run.py (проверка по этапам), обработки
"""
