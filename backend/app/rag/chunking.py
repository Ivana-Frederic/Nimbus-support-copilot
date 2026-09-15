"""Markdown-aware chunker. Splits on headings first so a chunk never
straddles two unrelated sections, then falls back to a sliding character
window with overlap for sections that are still too long. Pure stdlib, so
it's easy to test without any ML dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    text: str
    heading: str
    index: int


def _split_by_heading(markdown: str) -> list[tuple[str, str]]:
    """Returns list of (heading, section_text) preserving order.

    Text before the first heading is kept under heading "".
    """
    matches = list(_HEADING_RE.finditer(markdown))
    if not matches:
        return [("", markdown.strip())] if markdown.strip() else []

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = markdown[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    for i, m in enumerate(matches):
        heading = m.group().lstrip("#").strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        sections.append((heading, body))
    return sections


def _sliding_window(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    windows = []
    start = 0
    step = max(1, max_chars - overlap_chars)
    while start < len(text):
        window = text[start : start + max_chars]
        windows.append(window)
        if start + max_chars >= len(text):
            break
        start += step
    return windows


def chunk_markdown(
    markdown: str,
    *,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> list[Chunk]:
    """Chunk a markdown document into retrieval-sized pieces.

    Each chunk is prefixed with its section heading so embeddings capture
    context even when a chunk is retrieved in isolation.
    """
    chunks: list[Chunk] = []
    idx = 0
    for heading, body in _split_by_heading(markdown):
        if not body:
            continue
        for window in _sliding_window(body, max_chars, overlap_chars):
            prefixed = f"{heading}\n{window}".strip() if heading else window
            chunks.append(Chunk(text=prefixed, heading=heading, index=idx))
            idx += 1
    return chunks
