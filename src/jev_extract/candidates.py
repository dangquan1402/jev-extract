"""Sentence / chunk candidate proposal for extractive Choice."""

from __future__ import annotations

import math
import re
import warnings
from typing import Sequence

from jev_extract.schemas import Span

# Split after sentence-ending punctuation followed by whitespace, or on newlines.
# Lookbehind keeps the terminator with the preceding sentence.
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+|(?<=\n)\s*|\n+",
)

# Soft cap for criteria text length sent to Jev (not the 255 cardinality limit).
DEFAULT_CRITERIA_TEXT_MAX = 500

JEV_CHOICE_MAX_CRITERIA = 255

# Soft length (chars) above which a sentence is further split into clauses.
DEFAULT_LONG_SENTENCE_CHARS = 180

# Clause separators: comma / semicolon / colon / em-dash / en-dash, with trailing space.
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:—–])\s+")


def _trim_piece(paragraph: str, start: int, end: int) -> tuple[int, int, str] | None:
    """Return trimmed (abs_start, abs_end, text) or None if empty."""
    text = paragraph[start:end]
    if not text.strip():
        return None
    stripped = text.strip()
    lead = len(text) - len(text.lstrip())
    abs_start = start + lead
    abs_end = abs_start + len(stripped)
    return abs_start, abs_end, stripped


def _split_interval(paragraph: str, start: int, end: int, pattern: re.Pattern[str]) -> list[tuple[int, int, str]]:
    """Split ``paragraph[start:end]`` on *pattern*; return trimmed pieces with absolute offsets."""
    segment = paragraph[start:end]
    pieces: list[tuple[int, int, str]] = []
    last = 0
    for match in pattern.finditer(segment):
        piece = _trim_piece(paragraph, start + last, start + match.start())
        if piece is not None:
            pieces.append(piece)
        last = match.end()
    piece = _trim_piece(paragraph, start + last, end)
    if piece is not None:
        pieces.append(piece)
    return pieces


def _split_long_into_clauses(
    paragraph: str,
    start: int,
    end: int,
    text: str,
    *,
    long_chars: int,
) -> list[tuple[int, int, str]]:
    """If *text* is longer than *long_chars*, split on clause punctuation; else keep as-is."""
    if len(text) <= long_chars:
        return [(start, end, text)]
    clauses = _split_interval(paragraph, start, end, _CLAUSE_SPLIT)
    # If clause split did nothing useful, keep the sentence.
    if len(clauses) <= 1:
        return [(start, end, text)]
    return clauses


def propose_candidates(
    paragraph: str,
    *,
    clause_fallback: bool = False,
    long_sentence_chars: int = DEFAULT_LONG_SENTENCE_CHARS,
) -> list[Span]:
    """Split *paragraph* into non-empty sentence-like spans with char offsets.

    Splits on ``.!?`` followed by whitespace, and on newlines. Empty / whitespace-only
    pieces are dropped. Keys are ``s0``, ``s1``, …

    When ``clause_fallback`` is True, sentences longer than ``long_sentence_chars``
    are further split on commas / semicolons / colons / dashes (clause boundaries).
    """
    if not paragraph:
        return []

    # Find split points by iterating matches and carving intervals.
    last = 0
    raw_pieces: list[tuple[int, int, str]] = []

    for match in _SENTENCE_SPLIT.finditer(paragraph):
        start, end = last, match.start()
        piece = _trim_piece(paragraph, start, end)
        if piece is not None:
            raw_pieces.append(piece)
        last = match.end()

    # Tail after the last split.
    if last < len(paragraph):
        piece = _trim_piece(paragraph, last, len(paragraph))
        if piece is not None:
            raw_pieces.append(piece)

    # No split matched — whole paragraph is one span (if non-empty).
    if not raw_pieces and paragraph.strip():
        piece = _trim_piece(paragraph, 0, len(paragraph))
        if piece is not None:
            raw_pieces.append(piece)

    pieces: list[tuple[int, int, str]] = []
    for start, end, txt in raw_pieces:
        if clause_fallback:
            pieces.extend(
                _split_long_into_clauses(
                    paragraph, start, end, txt, long_chars=long_sentence_chars
                )
            )
        else:
            pieces.append((start, end, txt))

    spans: list[Span] = []
    for i, (start, end, txt) in enumerate(pieces):
        spans.append(Span(text=txt, start=start, end=end, key=f"s{i}"))

    return spans


def propose_chunks(
    paragraph: str,
    *,
    long_sentence_chars: int = DEFAULT_LONG_SENTENCE_CHARS,
) -> list[Span]:
    """Default chunker: sentence split with clause fallback for long sentences.

    Prefer this for cascade / chunk-only extract paths.
    """
    return propose_candidates(
        paragraph,
        clause_fallback=True,
        long_sentence_chars=long_sentence_chars,
    )


def limit_candidates(
    spans: Sequence[Span],
    max_candidates: int = JEV_CHOICE_MAX_CRITERIA,
) -> list[Span]:
    """Ensure at most *max_candidates* spans (Jev Choice cardinality).

    If ``len(spans) > max_candidates``, adjacent spans are merged into roughly
    equal-sized chunks until the count is ``<= max_candidates``. Merged spans
    keep contiguous character offsets and concatenated text; keys are reassigned
    as ``s0`` … ``s{n-1}``.

    Emits a :class:`UserWarning` when merging occurs.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")

    items = list(spans)
    if len(items) <= max_candidates:
        return items

    warnings.warn(
        f"Candidate count {len(items)} exceeds Jev Choice limit "
        f"({max_candidates}); merging adjacent spans into ~equal chunks.",
        UserWarning,
        stacklevel=2,
    )

    n = len(items)
    # Target group size: ceil(n / max) so we get at most max_candidates groups.
    group_size = math.ceil(n / max_candidates)
    merged: list[Span] = []
    i = 0
    key_i = 0
    while i < n:
        chunk = items[i : i + group_size]
        text = " ".join(s.text for s in chunk)
        start = chunk[0].start
        end = chunk[-1].end
        merged.append(Span(text=text, start=start, end=end, key=f"s{key_i}"))
        key_i += 1
        i += group_size

    # If still over (shouldn't happen with ceil), take first max_candidates.
    if len(merged) > max_candidates:
        warnings.warn(
            f"After merge still {len(merged)} > {max_candidates}; truncating to first "
            f"{max_candidates}.",
            UserWarning,
            stacklevel=2,
        )
        merged = merged[:max_candidates]
        merged = [
            Span(text=s.text, start=s.start, end=s.end, key=f"s{i}")
            for i, s in enumerate(merged)
        ]

    return merged


def truncate_criteria_text(
    text: str,
    max_len: int = DEFAULT_CRITERIA_TEXT_MAX,
) -> str:
    """Truncate span text for use as a Choice criteria description."""
    if max_len < 1:
        raise ValueError("max_len must be >= 1")
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def spans_to_criteria(
    spans: Sequence[Span],
    *,
    max_text_len: int = DEFAULT_CRITERIA_TEXT_MAX,
) -> dict[str, str]:
    """Map span keys to (possibly truncated) text for ``Choice(criteria=...)``."""
    return {s.key: truncate_criteria_text(s.text, max_text_len) for s in spans}
