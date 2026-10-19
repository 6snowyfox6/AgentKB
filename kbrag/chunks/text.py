"""Чанкинг через HybridChunker.

Три вещи, которые в туториалах обычно сделаны неправильно и здесь исправлены:

1. Токенайзер берётся от модели эмбеддинга (BGE-M3), а не от all-MiniLM.
   На русском счёт токенов у этих токенайзеров расходится в разы.
2. Эмбеддится contextualize(chunk) — текст с заголовками разделов сверху,
   а цитируется chunk.text. В базу кладём оба.
3. Со страницами: номер страницы достаётся из провенанса
   chunk.meta.doc_items[].prov[].page_no — без него нечем выполнить
   требование «всегда указывай источник и страницу».

Перекрытия фрагментов нет: HybridChunker режет по границам абзацев и разделов, а
контекст вокруг найденного фрагмента добирается при поиске — соседними фрагментами.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from ..parse.config import Settings
from ..parse.models import ParsedDoc
from .models import Chunk


@lru_cache(maxsize=4)
def _tokenizer(model_id: str) -> Any:
    """Токенайзер модели — один на процесс.

    Загрузка занимает 1,5–2,5 с, а нарезка самой статьи — доли секунды: раньше
    токенайзер грузился заново на каждую статью, и это было 80% времени нарезки.
    Сначала берём скачанный с диска: иначе каждая загрузка спрашивает Hugging Face,
    не вышла ли новая версия, — лишняя секунда и зависимость от сети.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except (OSError, ValueError):  # ещё не скачан — качаем
        return AutoTokenizer.from_pretrained(model_id)


@lru_cache(maxsize=4)
def _chunker(model_id: str, max_tokens: int) -> Any:
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

    hf_tokenizer = _tokenizer(model_id)
    try:  # docling-core >= 2.2x
        from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

        tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=max_tokens)
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    except ImportError:  # pragma: no cover — старый API
        chunker = HybridChunker(  # type: ignore[call-arg]
            tokenizer=hf_tokenizer, max_tokens=max_tokens, merge_peers=True
        )

    return chunker, hf_tokenizer


def _pages_of(chunk: Any) -> list[int]:
    pages: set[int] = set()
    for item in getattr(chunk.meta, "doc_items", None) or []:
        for prov in getattr(item, "prov", None) or []:
            page_no = getattr(prov, "page_no", None)
            if page_no is not None:
                pages.add(int(page_no))
    return sorted(pages)


def _has_table(chunk: Any) -> bool:
    for item in getattr(chunk.meta, "doc_items", None) or []:
        label = getattr(item, "label", "")
        if "table" in str(getattr(label, "value", label)).lower():
            return True
    return False


def chunk_parsed_doc(parsed: ParsedDoc, settings: Settings) -> list[Chunk]:
    """Чанкинг документа. Для docling — по структуре, для fallback — по тексту."""
    if parsed.docling_doc:
        return _chunk_docling(parsed, settings)
    return _chunk_plain_text(parsed, settings)


def _chunk_docling(parsed: ParsedDoc, settings: Settings) -> list[Chunk]:
    from docling_core.types.doc.document import DoclingDocument

    chunker, hf_tokenizer = _chunker(settings.embed_model_id, settings.max_tokens)
    doc = DoclingDocument.model_validate(parsed.docling_doc)

    chunks: list[Chunk] = []
    for i, raw in enumerate(chunker.chunk(dl_doc=doc)):
        embed_text = chunker.contextualize(chunk=raw)
        chunks.append(
            Chunk(
                doc_id=parsed.doc_id,
                index=i,
                text=raw.text,
                embed_text=embed_text,
                headings=list(getattr(raw.meta, "headings", None) or []),
                pages=_pages_of(raw),
                n_tokens=len(hf_tokenizer.encode(embed_text, add_special_tokens=False)),
                has_table=_has_table(raw),
                source_path=parsed.source_path,
            )
        )
    return chunks


def _chunk_plain_text(parsed: ParsedDoc, settings: Settings) -> list[Chunk]:
    """Fallback-чанкинг: структуры нет, режем по абзацам в пределах бюджета токенов.

    Страницу сохраняем — она известна из pages_text, и цитирование не ломается.
    """
    hf_tokenizer = _tokenizer(settings.embed_model_id)
    budget = settings.max_tokens

    chunks: list[Chunk] = []
    index = 0
    for page_no in sorted(parsed.pages_text):
        buffer: list[str] = []
        buffer_tokens = 0
        for para in _pieces(_paragraphs(parsed.pages_text[page_no]), hf_tokenizer, budget):
            n = len(hf_tokenizer.encode(para, add_special_tokens=False))
            if buffer and buffer_tokens + n > budget:
                chunks.append(_plain_chunk(parsed, index, buffer, buffer_tokens, page_no))
                index += 1
                buffer, buffer_tokens = [], 0
            buffer.append(para)
            buffer_tokens += n
        if buffer:
            chunks.append(_plain_chunk(parsed, index, buffer, buffer_tokens, page_no))
            index += 1
    return chunks


def _paragraphs(text: str) -> Iterable[str]:
    for block in text.split("\n\n"):
        block = block.strip()
        if block:
            yield block


def _pieces(paragraphs: Iterable[str], hf_tokenizer: Any, budget: int) -> Iterable[str]:
    """Абзацы, а слишком длинный — по предложениям.

    PDF без разметки абзацев отдаёт страницу одним куском. Раньше такой кусок
    шёл в чанк целиком, и модель эмбеддинга молча обрезала всё дальше своего
    предела: конец страницы в поиске не участвовал.
    """
    from ..text import split_sentences

    for para in paragraphs:
        if len(hf_tokenizer.encode(para, add_special_tokens=False)) <= budget:
            yield para
            continue
        buffer: list[Any] = []
        used = 0
        for sentence in split_sentences(para) or [para]:
            n = len(hf_tokenizer.encode(sentence, add_special_tokens=False))
            if buffer and used + n > budget:
                yield " ".join(buffer)
                buffer, used = [], 0
            if n > budget:  # одно «предложение» длиннее бюджета — режем по словам
                words = sentence.split()
                step = max(1, len(words) * budget // n)
                for start in range(0, len(words), step):
                    yield " ".join(words[start : start + step])
                continue
            buffer.append(sentence)
            used += n
        if buffer:
            yield " ".join(buffer)


def _plain_chunk(
    parsed: ParsedDoc, index: int, buffer: list[str], n_tokens: int, page_no: int
) -> Chunk:
    body = "\n\n".join(buffer)
    return Chunk(
        doc_id=parsed.doc_id,
        index=index,
        text=body,
        embed_text=body,
        headings=[],
        pages=[page_no],
        n_tokens=n_tokens,
        has_table=False,
        source_path=parsed.source_path,
    )
