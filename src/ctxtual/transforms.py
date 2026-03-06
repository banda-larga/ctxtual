"""
Built-in transforms for ctx producers.

Transforms are callables passed to ``@ctx.producer(transform=...)`` that
pre-process the raw return value before it's stored in the workspace.

Available transforms
--------------------

- :func:`chunk_text`     — split a string into overlapping chunks (``list[dict]``)
- :func:`split_sections` — split text by a separator into sections (``list[dict]``)

Usage::

    from ctxtual.transforms import chunk_text

    @ctx.producer(workspace_type="doc", toolsets=[pager, search, pipe],
                    transform=chunk_text(chunk_size=1000, overlap=200))
    def read_pdf(path: str) -> str:
        return extract_text(path)

    # "long string..." → [{"chunk_index": 0, "text": "...", "char_offset": 0}, ...]
"""

import re
from collections.abc import Callable
from typing import Any


def chunk_text(
    chunk_size: int = 1000,
    overlap: int = 200,
) -> Callable[[Any], Any]:
    """Return a transform that splits a string into overlapping chunks.

    Each chunk becomes a dict with:

    - ``chunk_index`` — zero-based position in the sequence
    - ``text``        — the chunk content
    - ``char_offset`` — character offset in the original string

    Non-string values pass through unchanged, so this transform is safe
    to use on producers that *sometimes* return strings.

    Args:
        chunk_size: Maximum characters per chunk.
        overlap:    Characters of overlap between consecutive chunks.
                    Helps preserve context at boundaries.

    Example::

        transform=chunk_text(chunk_size=500, overlap=100)
        # "ABCDE..." (2000 chars) → [
        #   {"chunk_index": 0, "text": "ABC...", "char_offset": 0},
        #   {"chunk_index": 1, "text": "...CDE...", "char_offset": 400},
        #   ...
        # ]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    step = chunk_size - overlap

    def transform(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if not value:
            return []

        chunks: list[dict[str, Any]] = []
        idx = 0
        offset = 0
        while offset < len(value):
            end = min(offset + chunk_size, len(value))
            chunks.append(
                {
                    "chunk_index": idx,
                    "text": value[offset:end],
                    "char_offset": offset,
                }
            )
            offset += step
            idx += 1
        return chunks

    return transform


def split_sections(
    separator: str = "\n\n",
    strip: bool = True,
    min_length: int = 1,
) -> Callable[[Any], Any]:
    """Return a transform that splits text by a separator into sections.

    Each section becomes a dict with:

    - ``section_index`` — zero-based position
    - ``text``          — the section content
    - ``char_offset``   — character offset in the original string

    Useful for paragraph-level splitting (``"\\n\\n"``),
    Markdown header splitting (``"\\n# "``), or any custom delimiter.

    Non-string values pass through unchanged.

    Args:
        separator:  String to split on.
        strip:      Whether to strip whitespace from each section.
        min_length: Minimum character length to keep a section (filters empties).

    Example::

        transform=split_sections(separator="\\n\\n")
        # "Para 1\\n\\nPara 2\\n\\nPara 3"
        # → [{"section_index": 0, "text": "Para 1", "char_offset": 0}, ...]
    """

    def transform(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if not value:
            return []

        sections: list[dict[str, Any]] = []
        idx = 0
        offset = 0

        for part in value.split(separator):
            text = part.strip() if strip else part
            if len(text) >= min_length:
                # Find the actual offset (skip leading whitespace if stripped)
                actual_offset = value.find(part, offset)
                if actual_offset == -1:
                    actual_offset = offset
                sections.append(
                    {
                        "section_index": idx,
                        "text": text,
                        "char_offset": actual_offset,
                    }
                )
                idx += 1
            offset += len(part) + len(separator)

        return sections

    return transform


def split_markdown_sections() -> Callable[[Any], Any]:
    """Return a transform that splits Markdown text by headers.

    Each section becomes a dict with:

    - ``section_index`` — zero-based position
    - ``heading``       — the header text (e.g., ``"Introduction"``)
    - ``level``         — header level (1–6)
    - ``text``          — the body text under this header
    - ``char_offset``   — character offset of the header in the original

    Content before the first header is captured as level 0 with heading
    ``"(preamble)"``.

    Non-string values pass through unchanged.
    """
    _header_re = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)

    def transform(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if not value:
            return []

        sections: list[dict[str, Any]] = []
        matches = list(_header_re.finditer(value))

        if not matches:
            # No headers — return the whole text as one section
            text = value.strip()
            if text:
                return [
                    {
                        "section_index": 0,
                        "heading": "(document)",
                        "level": 0,
                        "text": text,
                        "char_offset": 0,
                    }
                ]
            return []

        # Content before first header
        preamble = value[: matches[0].start()].strip()
        if preamble:
            sections.append(
                {
                    "section_index": 0,
                    "heading": "(preamble)",
                    "level": 0,
                    "text": preamble,
                    "char_offset": 0,
                }
            )

        for i, m in enumerate(matches):
            level = len(m.group(1))
            heading = m.group(2).strip()
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(value)
            text = value[body_start:body_end].strip()

            sections.append(
                {
                    "section_index": len(sections),
                    "heading": heading,
                    "level": level,
                    "text": text,
                    "char_offset": m.start(),
                }
            )

        return sections

    return transform
