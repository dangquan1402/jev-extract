"""Unit tests for context-disambiguated token Choice criteria (no live API)."""

from __future__ import annotations

from collections import Counter

from jev_extract.questions import (
    build_end_choice,
    build_start_choice,
    format_token_context,
    tokens_to_criteria,
)
from jev_extract.tokenize import tokenize


def test_duplicate_words_get_distinct_context_criteria():
    """Two identical words at different positions → different criteria strings."""
    paragraph = (
        "The bank by the river is closed. "
        "She walked to the bank to deposit cash."
    )
    tokens = tokenize(paragraph)
    # Find both occurrences of "bank"
    bank_positions = [i for i, t in enumerate(tokens) if t == "bank"]
    assert len(bank_positions) >= 2
    p0, p1 = bank_positions[0], bank_positions[1]

    criteria = tokens_to_criteria(tokens, context_radius=4)
    c0, c1 = criteria[str(p0)], criteria[str(p1)]

    assert c0 != c1
    assert "«bank»" in c0 and "«bank»" in c1
    # Different neighbors
    assert "river" in c0 or "closed" in c0
    assert "deposit" in c1 or "cash" in c1 or "walked" in c1
    # Marked once each
    assert c0.count("«bank»") == 1
    assert c1.count("«bank»") == 1


def test_format_token_context_ellipsis_at_edges():
    tokens = ["A", "B", "C", "D", "E", "F", "G"]
    mid = format_token_context(tokens, 3, radius=2)
    assert mid == "… B C «D» E F …"
    # Start edge: no leading ellipsis
    start = format_token_context(tokens, 0, radius=2)
    assert start == "«A» B C …"
    # End edge: no trailing ellipsis
    end = format_token_context(tokens, 6, radius=2)
    assert end == "… E F «G»"


def test_build_end_choice_mentions_start():
    tokens = tokenize("She published notes in 1843. Later notes arrived.")
    start = tokens.index("1843")
    choice = build_end_choice(tokens, "When?", start_pos=start, start_word="1843")
    assert f"start was {start}:1843" in choice.instructions
    # End criteria keys >= start
    assert all(int(k) >= start for k in choice.criteria)
    assert "«1843»" in choice.criteria[str(start)]


def test_build_start_choice_keys_are_pos_only():
    tokens = tokenize("Ada published notes in 1843.")
    choice = build_start_choice(tokens, "When?")
    assert all(k.isdigit() for k in choice.criteria)
    assert any("«1843»" in v for v in choice.criteria.values())


def test_counter_duplicate_detection_matches_auto_trigger():
    tokens = tokenize("notes here and notes there")
    assert Counter(tokens)["notes"] > 1
