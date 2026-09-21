"""Unit tests for cheap span-as-Choice proposers (no API)."""

from __future__ import annotations

import pytest

from jev_extract.questions import (
    SMART_CRITERIA_TEXT_MAX,
    SMART_DENSE_MAX_LEN,
    SMART_PREFER_K,
    SMART_TARGET_CRITERIA,
    build_span_as_choice,
    choose_span_max_len,
    count_span_options,
    enumerate_span_windows,
    propose_span_windows,
    sentence_token_spans,
)
from jev_extract.tokenize import tokenize, tokenize_with_spans


PARAGRAPH = (
    "Marie Curie discovered radium and polonium. "
    "She won the Nobel Prize in Physics in 1903. "
    "Later she won a second Nobel Prize in Chemistry in 1911."
)
QUESTION = "In what year did Marie Curie win the Nobel Prize in Physics?"


def test_enumerate_span_windows_counts():
    n = 10
    wins = enumerate_span_windows(n, 3)
    assert len(wins) == count_span_options(n, 3)
    assert (0, 0) in wins
    assert (7, 9) in wins
    assert (8, 10) not in wins  # end exclusive of out-of-range


def test_enumerate_respects_lo_hi():
    wins = enumerate_span_windows(20, 5, lo=5, hi=12)
    assert all(5 <= s <= e < 12 for s, e in wins)
    assert len(wins) == count_span_options(7, 5)


def test_propose_all_under_cap():
    toks = tokenize(PARAGRAPH)
    k = choose_span_max_len(len(toks), prefer_k=13)
    wins = propose_span_windows(toks, propose="all", max_len=k, prefer_k=13)
    assert len(wins) <= 255
    assert len(wins) == count_span_options(len(toks), k)


def test_propose_smart_caps_and_includes_length1():
    toks, spans = tokenize_with_spans(PARAGRAPH)
    wins = propose_span_windows(
        toks,
        propose="smart",
        question=QUESTION,
        paragraph=PARAGRAPH,
        char_spans=spans,
    )
    assert len(wins) <= 255
    assert len(wins) <= max(SMART_TARGET_CRITERIA, count_span_options(len(toks), SMART_DENSE_MAX_LEN) + 5)
    # Every length-1 window present
    singles = {(s, e) for s, e in wins if s == e}
    assert len(singles) == len(toks)
    # Gold year present
    pos = toks.index("1903")
    assert (pos, pos) in wins
    # Dense lengths fully present for len<=dense
    for length in range(1, min(SMART_DENSE_MAX_LEN, len(toks)) + 1):
        expected = len(toks) - length + 1
        got = sum(1 for s, e in wins if e - s + 1 == length)
        assert got == expected, f"length {length}: got {got} expected {expected}"


def test_propose_smart_includes_sentence_spans():
    toks, spans = tokenize_with_spans(PARAGRAPH)
    sent = sentence_token_spans(PARAGRAPH, spans)
    assert len(sent) >= 2
    wins = set(
        propose_span_windows(
            toks,
            propose="smart",
            question=QUESTION,
            paragraph=PARAGRAPH,
            char_spans=spans,
        )
    )
    for se in sent:
        assert se in wins


def test_propose_in_chunk_denser_inside_range():
    toks, spans = tokenize_with_spans(PARAGRAPH)
    # Second sentence roughly contains 1903
    pos = toks.index("1903")
    lo = max(0, pos - 5)
    hi = min(len(toks), pos + 6)
    wins = propose_span_windows(
        toks,
        propose="in_chunk",
        question=QUESTION,
        paragraph=PARAGRAPH,
        char_spans=spans,
        lo=lo,
        hi=hi,
    )
    assert len(wins) <= 255
    assert all(lo <= s <= e < hi for s, e in wins)
    assert (pos, pos) in wins
    # Should be denser than a tiny shortlist: nearly all windows in range
    assert len(wins) >= count_span_options(hi - lo, min(SMART_DENSE_MAX_LEN, hi - lo)) - 1


def test_propose_rejects_unknown_mode():
    with pytest.raises(ValueError, match="propose"):
        propose_span_windows(["a", "b"], propose="nope")


def test_build_span_as_choice_smart_truncates_criteria():
    toks, spans = tokenize_with_spans(PARAGRAPH)
    choice = build_span_as_choice(
        toks,
        QUESTION,
        propose="smart",
        paragraph=PARAGRAPH,
        char_spans=spans,
        context_radius=1,
    )
    assert len(choice.criteria) <= 255
    assert all(len(v) <= SMART_CRITERIA_TEXT_MAX for v in choice.criteria.values())


def test_build_span_as_choice_all_keeps_long_criteria():
    toks = tokenize(PARAGRAPH)
    k = choose_span_max_len(len(toks), prefer_k=13)
    choice = build_span_as_choice(toks, QUESTION, propose="all", max_len=k)
    # all mode uses 500 soft cap — descriptions typically well under that
    assert len(choice.criteria) == count_span_options(len(toks), k)
    assert any(len(v) > SMART_CRITERIA_TEXT_MAX for v in choice.criteria.values())


def test_propose_dedupes():
    toks, spans = tokenize_with_spans(PARAGRAPH)
    wins = propose_span_windows(
        toks,
        propose="smart",
        question=QUESTION,
        paragraph=PARAGRAPH,
        char_spans=spans,
    )
    assert len(wins) == len(set(wins))


def test_propose_empty_tokens():
    assert propose_span_windows([], propose="smart") == []
    assert propose_span_windows([], propose="all") == []
