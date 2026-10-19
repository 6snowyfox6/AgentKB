"""CLI этапа парсинга.

python -m kbrag.parse.cli --input .local/pdf --out .local/parsed
python -m kbrag.parse.cli --input .local/pdf --device cuda --table-mode accurate
python -m kbrag.parse.cli --input .local/pdf/one.pdf --no-chunks   # быстрая проверка парсинга
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings
from .models import MANUAL_REVIEW, ParserSetupError
from .pipeline import process_all


def _collect_pdfs(target: Path) -> list[Path]:
    if target.is_file():
        return [target]
    return sorted(p for p in target.rglob("*.pdf") if p.is_file())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Парсинг PDF для базы знаний KBRAG")
    p.add_argument("--input", type=Path, default=None, help="папка с PDF или один файл")
    p.add_argument("--out", type=Path, default=None, help="куда складывать результат")
    p.add_argument("--golden", type=Path, default=None, help="папка с эталонами")
    p.add_argument("--logs", type=Path, default=None, help="папка для логов")
    p.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--table-mode", choices=["accurate", "fast"], default=None)
    p.add_argument("--max-tokens", type=int, default=None, help="размер чанка в токенах")
    p.add_argument("--timeout", type=int, default=None, help="жёсткий таймаут на документ, сек")
    p.add_argument("--limit", type=int, default=None, help="обработать только первые N файлов")
    p.add_argument("--no-chunks", action="store_true", help="только парсинг и валидация")
    return p


def settings_from_args(args: argparse.Namespace) -> Settings:
    s = Settings()
    if args.input and args.input.is_dir():
        s.pdf_dir = args.input
    if args.out:
        s.out_dir = args.out
    if args.golden:
        s.golden_dir = args.golden
    if args.logs:
        s.log_dir = args.logs
    if args.device:
        s.device = args.device
    if args.threads:
        s.num_threads = args.threads
    if args.table_mode:
        s.table_mode = args.table_mode
    if args.max_tokens:
        s.max_tokens = args.max_tokens
    if args.timeout:
        s.doc_timeout_sec = args.timeout
    return s


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_from_args(args)

    target = args.input or settings.pdf_dir
    pdfs = _collect_pdfs(target)
    if args.limit:
        pdfs = pdfs[: args.limit]

    if not pdfs:
        print(f"Не нашёл PDF в {target}", file=sys.stderr)
        return 1

    print(f"Документов: {len(pdfs)} | устройство: {settings.device}")
    try:
        results = process_all(pdfs, settings, make_chunks=not args.no_chunks)
    except ParserSetupError as exc:
        print(f"\nРазбор остановлен: {exc}", file=sys.stderr)
        return 2

    ok = [r for r in results if r.status != MANUAL_REVIEW]
    review = [r for r in results if r.status == MANUAL_REVIEW]
    total_chunks = sum(r.chunks for r in results)
    mean_acc = sum(r.accuracy for r in results) / len(results)

    print("\n" + "=" * 72)
    print(f"{'файл':<34}{'парсер':<14}{'статус':<16}{'точн.':>6}{'чанки':>7}")
    print("-" * 72)
    for r in results:
        print(f"{r.doc_id[:33]:<34}{r.parser:<14}{r.status:<16}{r.accuracy:>5.0%}{r.chunks:>7}")
    print("-" * 72)
    print(f"Готово: {len(ok)}/{len(results)}, на ручную проверку: {len(review)}")
    print(f"Средняя точность по проверкам: {mean_acc:.0%} | чанков всего: {total_chunks}")

    if review:
        print("\nНа ручную проверку:")
        for r in review:
            print(f"  • {r.doc_id}: " + "; ".join(r.failed_checks[:3] or ["парсер не справился"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
