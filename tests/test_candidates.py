"""Tests for candidate proposal and limiting (no API)."""

from __future__ import annotations

import warnings

import pytest

from jev_extract.candidates import (
    JEV_CHOICE_MAX_CRITERIA,
    limit_candidates,
    propose_candidates,
    propose_chunks,
    spans_to_criteria,
    truncate_criteria_text,
)
from jev_extract.schemas import Span


def test_propose_simple_sentences():
    text = "Hello world. How are you? I am fine!"
    spans = propose_candidates(text)
    assert len(spans) == 3
    assert spans[0].key == "s0"
    assert spans[0].text == "Hello world."
    assert text[spans[0].start : spans[0].end] == spans[0].text
    assert spans[1].text == "How are you?"
    assert spans[2].text == "I am fine!"
    for s in spans:
        assert text[s.start : s.end] == s.text


def test_propose_newlines():
    text = "First line.\nSecond line here.\n\nThird."
    spans = propose_candidates(text)
    assert len(spans) >= 2
    assert all(s.text.strip() for s in spans)
    for s in spans:
        assert text[s.start : s.end] == s.text


def test_propose_empty_and_whitespace():
    assert propose_candidates("") == []
    assert propose_candidates("   \n\t  ") == []


def test_propose_no_terminator_single_span():
    text = "Just a fragment without end punctuation"
    spans = propose_candidates(text)
    assert len(spans) == 1
    assert spans[0].text == text
    assert spans[0].key == "s0"


def test_dedupe_empty_pieces():
    text = "One.\n\n\nTwo."
    spans = propose_candidates(text)
    assert all(s.text for s in spans)
    assert len(spans) == 2


def test_limit_under_max_unchanged():
    spans = [
        Span(text=f"Sentence {i}.", start=i * 10, end=i * 10 + 9, key=f"s{i}")
        for i in range(10)
    ]
    out = limit_candidates(spans, max_candidates=255)
    assert out == spans


def test_limit_merges_when_over_max():
    n = 300
    spans = [
        Span(text=f"S{i}.", start=i * 4, end=i * 4 + 3, key=f"s{i}") for i in range(n)
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = limit_candidates(spans, max_candidates=255)
    assert len(out) <= 255
    assert len(out) < n
    assert any(issubclass(w.category, UserWarning) for w in caught)
    # Contiguous coverage of original range
    assert out[0].start == spans[0].start
    assert out[-1].end == spans[-1].end
    assert [s.key for s in out] == [f"s{i}" for i in range(len(out))]


def test_limit_exact_boundary():
    spans = [
        Span(text=f"S{i}.", start=i, end=i + 1, key=f"s{i}")
        for i in range(JEV_CHOICE_MAX_CRITERIA)
    ]
    out = limit_candidates(spans)
    assert len(out) == JEV_CHOICE_MAX_CRITERIA


def test_truncate_criteria_text():
    assert truncate_criteria_text("short") == "short"
    long = "x" * 600
    truncated = truncate_criteria_text(long, max_len=50)
    assert len(truncated) == 50
    assert truncated.endswith("...")


def test_spans_to_criteria_keys():
    spans = propose_candidates("Alpha. Beta.")
    criteria = spans_to_criteria(spans)
    assert set(criteria) == {s.key for s in spans}


def test_limit_max_candidates_invalid():
    with pytest.raises(ValueError):
        limit_candidates([], max_candidates=0)


def test_propose_chunks_clause_fallback_long():
    # Long sentence with commas should yield multiple clauses when fallback on.
    long = (
        "The committee reviewed the proposal, considered the budget implications, "
        "and finally approved the multi-year roadmap for the coastal research program "
        "after extensive deliberation among stakeholders across twelve member nations."
    )
    # Ensure it exceeds default long threshold
    assert len(long) > 180
    plain = propose_candidates(long, clause_fallback=False)
    assert len(plain) == 1
    chunks = propose_chunks(long)
    assert len(chunks) > 1
    for s in chunks:
        assert long[s.start : s.end] == s.text


def test_propose_chunks_short_unchanged():
    text = "Short one. Another short."
    a = propose_candidates(text)
    b = propose_chunks(text)
    assert [s.text for s in a] == [s.text for s in b]
