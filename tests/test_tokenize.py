"""Unit tests for tokenizer v1 and token-span helpers."""

from __future__ import annotations

import pytest

from jev_extract.tokenize import (
    answer_from_token_span,
    char_span_to_token_span,
    make_token_ref,
    token_char_spans,
    token_windows,
    tokenize,
    tokenize_with_spans,
    validate_token_ref,
)


def test_tokenize_basic():
    text = "She published notes in 1843."
    toks = tokenize(text)
    assert toks == ["She", "published", "notes", "in", "1843", "."]
    spans = token_char_spans(text)
    assert len(spans) == len(toks)
    for tok, (s, e) in zip(toks, spans):
        assert text[s:e] == tok


def test_tokenize_punctuation_and_apostrophe():
    text = "Babbage's Engine"
    assert tokenize(text) == ["Babbage", "'", "s", "Engine"]


def test_tokenize_empty():
    assert tokenize("") == []
    assert token_char_spans("") == []
    assert tokenize_with_spans("   ") == ([], [])


def test_make_and_validate_ref():
    toks = tokenize("Ada Lovelace published notes in 1843.")
    ref = make_token_ref(toks, 5)  # 1843
    assert ref == {"pos": 5, "word": "1843"}
    validate_token_ref(toks, ref)
    with pytest.raises(ValueError, match="checksum"):
        validate_token_ref(toks, {"pos": 5, "word": "1903"})
    with pytest.raises(ValueError, match="out of range"):
        validate_token_ref(toks, {"pos": 99, "word": "x"})


def test_answer_from_token_span_preserves_slice():
    text = "She published notes in 1843."
    toks, spans = tokenize_with_spans(text)
    # "notes in 1843"
    i = toks.index("notes")
    j = toks.index("1843")
    ans, cs, ce = answer_from_token_span(text, toks, spans, i, j)
    assert ans == text[cs:ce]
    assert "notes" in ans and "1843" in ans


def test_char_span_to_token_span_covers_gold():
    text = "She published notes in 1843."
    toks, spans = tokenize_with_spans(text)
    gold = "1843"
    gs = text.index(gold)
    ge = gs + len(gold)
    sp, ep = char_span_to_token_span(spans, gs, ge)
    assert toks[sp] == "1843"
    assert sp == ep
    ans, _, _ = answer_from_token_span(text, toks, spans, sp, ep)
    assert ans == gold


def test_char_span_mid_token_expands():
    text = "abcdef"
    # Force mid-token: whole "abcdef" is one token
    toks, spans = tokenize_with_spans(text)
    assert toks == ["abcdef"]
    sp, ep = char_span_to_token_span(spans, 2, 4)  # "cd"
    assert (sp, ep) == (0, 0)


def test_token_windows_single_and_sliding():
    assert token_windows(10, max_size=255) == [(0, 10)]
    wins = token_windows(300, max_size=255, overlap=64)
    assert wins[0] == (0, 255)
    assert wins[-1][1] == 300
    assert all(hi - lo <= 255 for lo, hi in wins)
    # Coverage
    covered = set()
    for lo, hi in wins:
        covered.update(range(lo, hi))
    assert covered == set(range(300))
