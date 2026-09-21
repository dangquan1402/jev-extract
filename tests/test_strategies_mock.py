"""Unit tests for alternate extraction strategies (mocked system_one, no live API)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from jev_extract import Extractor, tokenize
from jev_extract.questions import (
    choose_span_max_len,
    count_span_options,
    format_span_context,
)
from jev_extract.strategies import length_penalty_score, topk_from_probs


class FakeChoiceAnswer:
    def __init__(
        self,
        choice: str,
        confidence: float = 0.9,
        probabilities: dict[str, float] | None = None,
    ):
        self.type = "choice"
        self.choice = choice
        self.confidence = confidence
        self.probabilities = probabilities or {choice: confidence}


class FakeNoulAnswer:
    def __init__(self, noul: float = 0.8):
        self.type = "noul"
        self.noul = noul


class FakeResponse:
    def __init__(self, answers: dict[str, Any], model: str = "jev-latest"):
        self.answers = answers
        self.model = model
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)

    @property
    def choices(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.answers.items()
            if getattr(v, "type", None) == "choice"
        }

    @property
    def nouls(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in self.answers.items()
            if getattr(v, "type", None) == "noul"
        }


class FakeClient:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self._i = 0

    def system_one(
        self, state: Any, questions: dict[str, Any], *, model: str | None = None, **kwargs: Any
    ):
        self.calls.append(
            {"state": state, "questions": questions, "model": model, "kwargs": kwargs}
        )
        if self._i >= len(self.responses):
            raise AssertionError(
                f"Unexpected system_one call #{self._i}; only {len(self.responses)} scripted"
            )
        resp = self.responses[self._i]
        self._i += 1
        return resp


PARAGRAPH = (
    "Ada Lovelace worked on Babbage's Analytical Engine. "
    "She published notes in 1843. "
    "Those notes include an early algorithm."
)


def _year_pos() -> int:
    return tokenize(PARAGRAPH).index("1843")


def test_choose_span_max_len_respects_255():
    n = 35
    k = choose_span_max_len(n, prefer_k=13, max_criteria=255)
    assert count_span_options(n, k) <= 255
    assert k >= 1
    assert choose_span_max_len(28, prefer_k=13) >= 8


def test_format_span_context_marks_range():
    toks = ["a", "b", "c", "d"]
    s = format_span_context(toks, 1, 2, radius=1)
    assert "1:b..2:c" in s
    assert "«b" in s and "c»" in s


def test_length_penalty_prefers_shorter_by_default():
    short = length_penalty_score(0.5, 1, alpha=0.5)
    long = length_penalty_score(0.5, 4, alpha=0.5)
    assert short > long
    long_pref = length_penalty_score(0.5, 4, alpha=0.5, prefer_longer=True)
    short_pref = length_penalty_score(0.5, 1, alpha=0.5, prefer_longer=True)
    assert long_pref > short_pref


def test_topk_from_probs():
    ranked = topk_from_probs(
        {"5": 0.1, "7": 0.6, "8": 0.3}, k=2, lo=5, hi=10, fallback=7
    )
    assert [p for p, _ in ranked] == [7, 8]


def test_span_choice_mode():
    pos = _year_pos()
    key = f"{pos}:{pos}"
    client = FakeClient(
        [FakeResponse({"span": FakeChoiceAnswer(key, confidence=0.92)})]
    )
    ex = Extractor(client=client)
    result = ex.extract_span_choice(
        paragraph=PARAGRAPH, question="When published?", include_raw=True
    )
    assert result.answer == "1843"
    assert result.start.pos == pos
    assert result.end.pos == pos
    assert len(client.calls) == 1
    assert "span" in client.calls[0]["questions"]
    crit = client.calls[0]["questions"]["span"].criteria
    assert key in crit
    assert len(crit) <= 255
    assert result.raw["max_len"] >= 1


def test_span_choice_via_extract_mode():
    pos = _year_pos()
    client = FakeClient(
        [FakeResponse({"span": FakeChoiceAnswer(f"{pos}:{pos}")})]
    )
    ex = Extractor(client=client, extract_mode="span_choice")
    result = ex.extract(paragraph=PARAGRAPH, question="When?")
    assert result.answer == "1843"


def test_anchor_expand_mode():
    toks = tokenize(PARAGRAPH)
    head = toks.index("1843")
    client = FakeClient(
        [
            FakeResponse({"head": FakeChoiceAnswer(str(head), confidence=0.9)}),
            FakeResponse(
                {
                    "left": FakeChoiceAnswer("0", confidence=0.8),
                    "right": FakeChoiceAnswer("0", confidence=0.85),
                }
            ),
        ]
    )
    ex = Extractor(client=client)
    result = ex.extract_anchor_expand(
        paragraph=PARAGRAPH, question="When?", include_raw=True
    )
    assert result.answer == "1843"
    assert result.start.pos == head
    assert result.end.pos == head
    assert result.raw["left_ext"] == 0
    assert result.raw["right_ext"] == 0
    assert len(client.calls) == 2


def test_anchor_expand_with_extents():
    toks = tokenize(PARAGRAPH)
    # head on Engine; expand left 1 to include Analytical
    head = toks.index("Engine")
    client = FakeClient(
        [
            FakeResponse({"head": FakeChoiceAnswer(str(head))}),
            FakeResponse(
                {
                    "left": FakeChoiceAnswer("1"),
                    "right": FakeChoiceAnswer("0"),
                }
            ),
        ]
    )
    ex = Extractor(client=client)
    result = ex.extract_anchor_expand(paragraph=PARAGRAPH, question="What engine?")
    assert result.start.word == "Analytical"
    assert result.end.word == "Engine"
    assert "Analytical" in (result.answer or "")


def test_shape_gate_mode():
    pos = _year_pos()
    client = FakeClient(
        [
            FakeResponse({"shape": FakeChoiceAnswer("number", confidence=0.88)}),
            FakeResponse({"start": FakeChoiceAnswer(str(pos), confidence=0.9)}),
            FakeResponse({"end": FakeChoiceAnswer(str(pos), confidence=0.91)}),
        ]
    )
    ex = Extractor(client=client)
    result = ex.extract_shape_gate(
        paragraph=PARAGRAPH, question="When?", include_raw=True
    )
    assert result.answer == "1843"
    assert result.raw["shape_label"] == "number"
    assert len(client.calls) == 3
    end_instr = client.calls[2]["questions"]["end"].instructions
    assert "NUMBER" in end_instr


def test_topk_noul_picks_passing_candidate():
    pos = _year_pos()
    # Model prefers longer end (pos+2) but Noul says short span is better.
    longer = pos + 2
    probs = {str(pos): 0.35, str(pos + 1): 0.25, str(longer): 0.40}
    client = FakeClient(
        [
            FakeResponse({"start": FakeChoiceAnswer(str(pos), confidence=0.9)}),
            FakeResponse(
                {
                    "end": FakeChoiceAnswer(
                        str(longer), confidence=0.4, probabilities=probs
                    )
                }
            ),
            FakeResponse(
                {
                    "v0": FakeNoulAnswer(0.2),  # longer
                    "v1": FakeNoulAnswer(0.9),  # maybe mid or short depending on order
                    "v2": FakeNoulAnswer(0.85),
                }
            ),
        ]
    )
    ex = Extractor(client=client)
    result = ex.extract_topk_noul(
        paragraph=PARAGRAPH, question="When?", include_raw=True, top_k=3
    )
    assert result.start.pos == pos
    assert result.end.pos <= longer
    assert "noul" in result.raw
    assert len(result.raw["candidates"]) == 3
    # Batched nouls in one call
    assert any(k.startswith("v") for k in client.calls[2]["questions"])


def test_ambiguous_joint_sequential_when_unique():
    pos = _year_pos()  # 1843 appears once
    client = FakeClient(
        [
            FakeResponse(
                {
                    "start": FakeChoiceAnswer(
                        str(pos),
                        confidence=0.95,
                        probabilities={str(pos): 0.95, str(pos - 1): 0.05},
                    )
                }
            ),
            FakeResponse({"end": FakeChoiceAnswer(str(pos), confidence=0.9)}),
        ]
    )
    ex = Extractor(client=client)
    result = ex.extract_ambiguous_joint(
        paragraph=PARAGRAPH, question="When?", include_raw=True
    )
    assert result.answer == "1843"
    assert result.raw["path"] == "sequential"


def test_ambiguous_joint_uses_joint_on_duplicate():
    # "notes" appears twice
    toks = tokenize(PARAGRAPH)
    notes_positions = [i for i, t in enumerate(toks) if t == "notes"]
    assert len(notes_positions) >= 2
    s0 = notes_positions[0]
    # Joint returns s0:s0
    client = FakeClient(
        [
            FakeResponse(
                {
                    "start": FakeChoiceAnswer(
                        str(s0),
                        confidence=0.4,
                        probabilities={
                            str(notes_positions[0]): 0.45,
                            str(notes_positions[1]): 0.40,
                            str(s0 + 1): 0.15,
                        },
                    )
                }
            ),
            FakeResponse(
                {"span": FakeChoiceAnswer(f"{s0}:{s0}", confidence=0.7)}
            ),
        ]
    )
    ex = Extractor(client=client)
    result = ex.extract_ambiguous_joint(
        paragraph=PARAGRAPH, question="What was published?", include_raw=True
    )
    assert result.raw["path"] == "joint"
    assert result.start.word == "notes"
    assert "span" in client.calls[1]["questions"]


def test_length_rerank_prefers_shorter():
    pos = _year_pos()
    longer = pos + 2
    probs = {str(pos): 0.34, str(longer): 0.36, str(pos + 1): 0.30}
    client = FakeClient(
        [
            FakeResponse({"shape": FakeChoiceAnswer("number")}),
            FakeResponse({"start": FakeChoiceAnswer(str(pos))}),
            FakeResponse(
                {
                    "end": FakeChoiceAnswer(
                        str(longer), confidence=0.36, probabilities=probs
                    )
                }
            ),
        ]
    )
    ex = Extractor(client=client)
    result = ex.extract_length_rerank(
        paragraph=PARAGRAPH,
        question="When?",
        include_raw=True,
        alpha=1.0,
        shape_aware=True,
    )
    # Shorter should win despite slightly lower raw prob
    assert result.end.pos == pos
    assert result.raw["prefer_longer"] is False
    assert result.raw["shape_label"] == "number"


def test_length_rerank_full_sentence_prefers_longer():
    pos = _year_pos()
    # Use a start earlier so longer ends exist
    start = tokenize(PARAGRAPH).index("She")
    end_short = start + 1
    end_long = tokenize(PARAGRAPH).index(".")  # end of second sentence — find first . after She
    # second sentence ends at period after 1843
    toks = tokenize(PARAGRAPH)
    end_long = toks.index("1843")  # still shorter; pick last period of sent 2
    # Find period after 1843
    end_long = toks.index("1843") + 1  # "."
    probs = {str(end_short): 0.40, str(end_long): 0.39}
    client = FakeClient(
        [
            FakeResponse({"shape": FakeChoiceAnswer("full_sentence")}),
            FakeResponse({"start": FakeChoiceAnswer(str(start))}),
            FakeResponse(
                {
                    "end": FakeChoiceAnswer(
                        str(end_short), confidence=0.4, probabilities=probs
                    )
                }
            ),
        ]
    )
    ex = Extractor(client=client)
    result = ex.extract_length_rerank(
        paragraph=PARAGRAPH,
        question="Which sentence mentions publication?",
        include_raw=True,
        alpha=1.0,
    )
    assert result.raw["prefer_longer"] is True
    assert result.end.pos == end_long


def test_extract_mode_rejects_unknown():
    with pytest.raises(ValueError, match="extract_mode"):
        Extractor(client=FakeClient([]), extract_mode="nope")
