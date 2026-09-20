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


def propose_candidates(paragraph: str) -> list[Span]:
    """Split *paragraph* into non-empty sentence-like spans with char offsets.

    Splits on ``.!?`` followed by whitespace, and on newlines. Empty / whitespace-only
    pieces are dropped. Keys are ``s0``, ``s1``, …
    """
    if not paragraph:
        return []

    spans: list[Span] = []
    # Find split points by iterating matches and carving intervals.
    last = 0
    pieces: list[tuple[int, int, str]] = []

    for match in _SENTENCE_SPLIT.finditer(paragraph):
        start, end = last, match.start()
        # Include trailing punctuation already in [last, match.start()); whitespace is the separator.
        text = paragraph[start:end]
        if text.strip():
            # Trim leading/trailing whitespace but keep offsets for the trimmed core.
            stripped = text.strip()
            lead = len(text) - len(text.lstrip())
            abs_start = start + lead
            abs_end = abs_start + len(stripped)
            pieces.append((abs_start, abs_end, stripped))
        last = match.end()

    # Tail after the last split.
    if last < len(paragraph):
        text = paragraph[last:]
        if text.strip():
            stripped = text.strip()
            lead = len(text) - len(text.lstrip())
            abs_start = last + lead
            abs_end = abs_start + len(stripped)
            pieces.append((abs_start, abs_end, stripped))

    # No split matched — whole paragraph is one span (if non-empty).
    if not pieces and paragraph.strip():
        stripped = paragraph.strip()
        lead = len(paragraph) - len(paragraph.lstrip())
        abs_start = lead
        abs_end = abs_start + len(stripped)
        pieces.append((abs_start, abs_end, stripped))

    for i, (start, end, text) in enumerate(pieces):
        spans.append(Span(text=text, start=start, end=end, key=f"s{i}"))

    return spans


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
