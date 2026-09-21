"""Tokenizer v1 and token-span helpers for extractive Choice.

Tokenizer v1
------------
Split text into tokens by **whitespace separators** and by treating each
**punctuation / non-alphanumeric** character as its own token:

* Maximal runs of Unicode alphanumeric characters (``\\w`` with ``UNICODE``,
  which includes digits and letters; underscore counts as alphanumeric) form
  word/number tokens.
* Any other non-whitespace character is a singleton punctuation token.
* Whitespace is never emitted; it only separates tokens.

Examples::

    tokenize("She published notes in 1843.")
    # ["She", "published", "notes", "in", "1843", "."]

    tokenize("Babbage's Engine")
    # ["Babbage", "'", "s", "Engine"]

``token_char_spans(text)`` returns a parallel list of ``(start, end)`` character
offsets (half-open ``[start, end)``) so each token maps back into the source
string. Derived answer strings use ``text[char_spans[i][0]:char_spans[j][1]]``.
"""

from __future__ import annotations

import re
from typing import Sequence

# Alphanumeric run OR a single non-whitespace non-alphanumeric char.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

TOKENIZER_VERSION = "v1"


def tokenize(text: str) -> list[str]:
    """Tokenize *text* with tokenizer v1 (whitespace + punctuation split)."""
    if not text:
        return []
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


def token_char_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` char offsets for each token in *text* (half-open)."""
    if not text:
        return []
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def tokenize_with_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Return ``(tokens, char_spans)`` from a single pass over *text*."""
    if not text:
        return [], []
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in _TOKEN_RE.finditer(text):
        tokens.append(m.group(0))
        spans.append((m.start(), m.end()))
    return tokens, spans


def make_token_ref(tokens: Sequence[str], pos: int) -> dict[str, object]:
    """Build a ``{pos, word}`` dict; raises if *pos* is out of range."""
    if pos < 0 or pos >= len(tokens):
        raise IndexError(f"token pos {pos} out of range for {len(tokens)} tokens")
    return {"pos": int(pos), "word": tokens[pos]}


def validate_token_ref(tokens: Sequence[str], ref: dict[str, object] | object) -> None:
    """Validate that ``ref.pos`` indexes *tokens* and ``ref.word == tokens[pos]``.

    Accepts a mapping with ``pos``/``word`` keys or an object with those attributes.
    Raises :class:`ValueError` on mismatch.
    """
    if isinstance(ref, dict):
        pos = int(ref["pos"])  # type: ignore[index]
        word = str(ref["word"])  # type: ignore[index]
    else:
        pos = int(getattr(ref, "pos"))
        word = str(getattr(ref, "word"))
    if pos < 0 or pos >= len(tokens):
        raise ValueError(f"TokenRef pos {pos} out of range for {len(tokens)} tokens")
    if tokens[pos] != word:
        raise ValueError(
            f"TokenRef word checksum failed at pos={pos}: "
            f"expected {tokens[pos]!r}, got {word!r}"
        )


def answer_from_token_span(
    text: str,
    tokens: Sequence[str],
    char_spans: Sequence[tuple[int, int]],
    start_pos: int,
    end_pos: int,
) -> tuple[str, int, int]:
    """Derive ``(answer, char_start, char_end)`` for inclusive token indices.

    Character offsets are secondary: the answer is the original slice from the
    first token's start through the last token's end (preserving intervening
    whitespace/punctuation exactly as in *text*).
    """
    if start_pos < 0 or end_pos < start_pos or end_pos >= len(tokens):
        raise ValueError(
            f"Invalid token span [{start_pos}, {end_pos}] for {len(tokens)} tokens"
        )
    if len(char_spans) != len(tokens):
        raise ValueError("tokens and char_spans length mismatch")
    char_start = char_spans[start_pos][0]
    char_end = char_spans[end_pos][1]
    return text[char_start:char_end], char_start, char_end


def char_span_to_token_span(
    char_spans: Sequence[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    """Map a gold character span to covering inclusive token indices.

    Prefers the smallest token range that **covers** ``[char_start, char_end)``.
    If the gold sits mid-token, that token is included. Raises ``ValueError``
    when no overlapping tokens exist.
    """
    if char_end < char_start:
        raise ValueError(f"Invalid char span [{char_start}, {char_end})")
    if not char_spans:
        raise ValueError("No tokens to map char span onto")

    start_pos: int | None = None
    end_pos: int | None = None
    for i, (ts, te) in enumerate(char_spans):
        # Overlap with [char_start, char_end)
        if te > char_start and ts < char_end:
            if start_pos is None:
                start_pos = i
            end_pos = i
    if start_pos is None or end_pos is None:
        raise ValueError(
            f"No token covers char span [{char_start}, {char_end})"
        )
    return start_pos, end_pos


def token_windows(
    n_tokens: int,
    *,
    max_size: int = 255,
    overlap: int = 64,
) -> list[tuple[int, int]]:
    """Sliding half-open windows ``[lo, hi)`` of at most *max_size* tokens.

    When ``n_tokens <= max_size``, returns a single ``(0, n_tokens)`` window.
    Otherwise uses stride ``max_size - overlap`` (minimum stride 1). Used when
    Choice cardinality would exceed Jev's 255 criteria limit.
    """
    if max_size < 1:
        raise ValueError("max_size must be >= 1")
    if n_tokens <= 0:
        return []
    if n_tokens <= max_size:
        return [(0, n_tokens)]
    overlap = max(0, min(overlap, max_size - 1))
    stride = max(1, max_size - overlap)
    windows: list[tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + max_size, n_tokens)
        windows.append((start, end))
        if end >= n_tokens:
            break
        start += stride
    return windows
