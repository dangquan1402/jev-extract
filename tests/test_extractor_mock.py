"""Extractor tests with a mocked system_one (no live API)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from jev_extract import (
    Extractor,
    ExtractionError,
    MissingAPIKeyError,
    NoCandidatesError,
    tokenize,
)


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


class FakeScoreAnswer:
    def __init__(self, score: float = 1.5, confidence: float = 0.7):
        self.type = "score"
        self.score = score
        self.confidence = confidence
        self.probabilities = {0: 0.2, 1: 0.5, 2: 0.3}
        self.legend = {0: "low", 1: "mid", 2: "high"}


class FakeResponse:
    def __init__(self, answers: dict[str, Any], model: str = "jev-latest"):
        self.answers = answers
        self.model = model
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1)

    @property
    def choices(self) -> dict[str, Any]:
        return {k: v for k, v in self.answers.items() if getattr(v, "type", None) == "choice"}

    @property
    def nouls(self) -> dict[str, Any]:
        return {k: v for k, v in self.answers.items() if getattr(v, "type", None) == "noul"}

    @property
    def scores(self) -> dict[str, Any]:
        return {k: v for k, v in self.answers.items() if getattr(v, "type", None) == "score"}


class FakeClient:
    """Returns scripted responses in call order (start then end for extract)."""

    def __init__(
        self,
        responses: list[FakeResponse] | FakeResponse | None = None,
        exc: Exception | None = None,
    ):
        if responses is None:
            self.responses: list[FakeResponse] = []
        elif isinstance(responses, list):
            self.responses = list(responses)
        else:
            self.responses = [responses]
        self.exc = exc
        self.calls: list[dict[str, Any]] = []
        self._i = 0

    def system_one(self, state: Any, questions: dict[str, Any], *, model: str | None = None, **kwargs: Any):
        self.calls.append({"state": state, "questions": questions, "model": model, "kwargs": kwargs})
        if self.exc is not None:
            raise self.exc
        if self._i >= len(self.responses):
            raise AssertionError(f"Unexpected system_one call #{self._i}; only {len(self.responses)} scripted")
        resp = self.responses[self._i]
        self._i += 1
        return resp


PARAGRAPH = (
    "Ada Lovelace worked on Babbage's Analytical Engine. "
    "She published notes in 1843. "
    "Those notes include an early algorithm."
)


def _year_pos() -> int:
    toks = tokenize(PARAGRAPH)
    return toks.index("1843")


def test_extract_sequential_start_end():
    pos = _year_pos()
    client = FakeClient(
        [
            FakeResponse({"start": FakeChoiceAnswer(str(pos), confidence=0.95)}),
            FakeResponse({"end": FakeChoiceAnswer(str(pos), confidence=0.9)}),
        ]
    )
    ex = Extractor(client=client)
    result = ex.extract(paragraph=PARAGRAPH, question="When were the notes published?")

    assert result.start.pos == pos
    assert result.start.word == "1843"
    assert result.end.pos == pos
    assert result.end.word == "1843"
    assert result.answer == "1843"
    assert result.tokens[result.start.pos] == "1843"
    assert PARAGRAPH[result.char_start : result.char_end] == "1843"
    assert result.confidence == pytest.approx(0.95 * 0.9)
    assert result.model == "jev-latest"
    assert len(client.calls) == 2
    assert "start" in client.calls[0]["questions"]
    assert "end" in client.calls[1]["questions"]
    # End criteria only includes pos >= start
    end_crit = client.calls[1]["questions"]["end"].criteria
    assert str(pos) in end_crit
    assert all(int(k) >= pos for k in end_crit)


def test_extract_multi_token_span():
    toks = tokenize(PARAGRAPH)
    # "Analytical Engine" — find Engine and the token before if Analytical
    start = toks.index("Analytical")
    end = toks.index("Engine")
    client = FakeClient(
        [
            FakeResponse({"start": FakeChoiceAnswer(str(start))}),
            FakeResponse({"end": FakeChoiceAnswer(str(end))}),
        ]
    )
    ex = Extractor(client=client)
    result = ex.extract(paragraph=PARAGRAPH, question="What engine?")
    assert result.start.pos == start
    assert result.end.pos == end
    assert "Analytical" in (result.answer or "")
    assert "Engine" in (result.answer or "")


def test_extract_unknown_pos_raises():
    client = FakeClient(
        [FakeResponse({"start": FakeChoiceAnswer("999")})]
    )
    ex = Extractor(client=client)
    with pytest.raises(ExtractionError, match="outside allowed range"):
        ex.extract(paragraph=PARAGRAPH, question="What?")


def test_extract_end_before_window_raises():
    pos = _year_pos()
    # Start OK, end returns a pos < start (not in criteria normally, but fake it)
    client = FakeClient(
        [
            FakeResponse({"start": FakeChoiceAnswer(str(pos))}),
            FakeResponse({"end": FakeChoiceAnswer("0")}),
        ]
    )
    ex = Extractor(client=client)
    with pytest.raises(ExtractionError, match="outside allowed range"):
        ex.extract(paragraph=PARAGRAPH, question="When?")


def test_extract_no_tokens():
    client = FakeClient([])
    ex = Extractor(client=client)
    with pytest.raises(NoCandidatesError):
        ex.extract(paragraph="   ", question="Anything?")


def test_extract_sdk_error_wrapped():
    client = FakeClient(exc=RuntimeError("boom"))
    ex = Extractor(client=client)
    with pytest.raises(ExtractionError, match="system_one"):
        ex.extract(paragraph=PARAGRAPH, question="When?")


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        Extractor()


def test_extract_fields_mixed_modes():
    pos = _year_pos()
    # span field needs start+end; then one call for non-span fields
    client = FakeClient(
        [
            FakeResponse({"start": FakeChoiceAnswer(str(pos))}),
            FakeResponse({"end": FakeChoiceAnswer(str(pos))}),
            FakeResponse(
                {
                    "is_history": FakeNoulAnswer(0.91),
                    "topic": FakeChoiceAnswer(
                        "math", confidence=0.8, probabilities={"math": 0.8, "other": 0.2}
                    ),
                    "detail": FakeScoreAnswer(1.7, confidence=0.6),
                }
            ),
        ]
    )
    ex = Extractor(client=client)
    results = ex.extract_fields(
        paragraph=PARAGRAPH,
        fields={
            "when": {"mode": "span", "question": "When published?"},
            "is_history": {"mode": "noul", "question": "Is this historical?"},
            "topic": {
                "mode": "choice",
                "question": "Topic?",
                "criteria": {"math": "Math/computing", "other": "Other"},
            },
            "detail": {
                "mode": "score",
                "question": "Detail level?",
                "criteria": ["low", "medium", "high"],
            },
        },
    )

    assert results["when"].mode == "span"
    assert results["when"].value == "1843"
    assert results["when"].start is not None
    assert getattr(results["when"].start, "pos") == pos
    assert results["is_history"].mode == "noul"
    assert results["is_history"].value == pytest.approx(0.91)
    assert results["topic"].value == "math"
    assert results["detail"].value == pytest.approx(1.7)
    assert results["detail"].probabilities is not None


def test_extract_fields_choice_requires_criteria():
    client = FakeClient([])
    ex = Extractor(client=client)
    with pytest.raises(ExtractionError, match="criteria dict"):
        ex.extract_fields(
            paragraph=PARAGRAPH,
            fields={"t": {"mode": "choice", "question": "Topic?"}},
        )


def test_extract_fields_string_shorthand():
    pos = tokenize(PARAGRAPH).index("Ada")
    client = FakeClient(
        [
            FakeResponse({"start": FakeChoiceAnswer(str(pos))}),
            FakeResponse({"end": FakeChoiceAnswer(str(pos))}),
        ]
    )
    ex = Extractor(client=client)
    results = ex.extract_fields(paragraph=PARAGRAPH, fields={"q": "Who?"})
    assert results["q"].value == "Ada"


def test_include_raw():
    pos = _year_pos()
    r0 = FakeResponse({"start": FakeChoiceAnswer(str(pos))})
    r1 = FakeResponse({"end": FakeChoiceAnswer(str(pos))})
    client = FakeClient([r0, r1])
    ex = Extractor(client=client)
    result = ex.extract(paragraph=PARAGRAPH, question="When?", include_raw=True)
    assert isinstance(result.raw, dict)
    assert result.raw["start"] is r0
    assert result.raw["end"] is r1


def test_start_criteria_format():
    pos = _year_pos()
    client = FakeClient(
        [
            FakeResponse({"start": FakeChoiceAnswer(str(pos))}),
            FakeResponse({"end": FakeChoiceAnswer(str(pos))}),
        ]
    )
    ex = Extractor(client=client)
    ex.extract(paragraph=PARAGRAPH, question="When?")
    criteria = client.calls[0]["questions"]["start"].criteria
    desc = criteria[str(pos)]
    assert "«1843»" in desc
    assert "notes" in desc or "in" in desc
    # End instructions mention chosen start
    end_instr = client.calls[1]["questions"]["end"].instructions
    assert f"start was {pos}:1843" in end_instr


def test_extract_chunk_whole_sentence():
    # Pick s1 ("She published notes in 1843.")
    spans_text = PARAGRAPH
    client = FakeClient(
        [FakeResponse({"chunk": FakeChoiceAnswer("s1", confidence=0.88)})]
    )
    ex = Extractor(client=client)
    result = ex.extract_chunk(paragraph=PARAGRAPH, question="When published?")
    assert "1843" in (result.answer or "")
    assert result.start.pos < result.end.pos or result.start.pos == result.end.pos
    # Whole sentence should include trailing period token
    assert result.answer.endswith(".") or "1843" in result.tokens[result.start.pos : result.end.pos + 1]
    assert len(client.calls) == 1
    assert "chunk" in client.calls[0]["questions"]


def test_extract_cascade_refines_inside_chunk():
    pos = _year_pos()
    client = FakeClient(
        [
            FakeResponse({"chunk": FakeChoiceAnswer("s1", confidence=0.9)}),
            FakeResponse({"start": FakeChoiceAnswer(str(pos), confidence=0.95)}),
            FakeResponse({"end": FakeChoiceAnswer(str(pos), confidence=0.9)}),
        ]
    )
    ex = Extractor(client=client)
    result = ex.extract_cascade(paragraph=PARAGRAPH, question="When were the notes published?")
    assert result.answer == "1843"
    assert result.start.pos == pos
    assert result.end.pos == pos
    assert len(client.calls) == 3
    # Start/end criteria restricted to chunk token range
    start_crit = client.calls[1]["questions"]["start"].criteria
    assert str(pos) in start_crit
    # Positions outside the middle sentence should not appear
    toks = tokenize(PARAGRAPH)
    # first sentence tokens include Ada — should be absent from start criteria
    ada = toks.index("Ada")
    assert str(ada) not in start_crit


def test_extract_cascade_single_chunk_skips_choice():
    # One sentence → no chunk Choice call
    para = "She published notes in 1843."
    toks = tokenize(para)
    pos = toks.index("1843")
    client = FakeClient(
        [
            FakeResponse({"start": FakeChoiceAnswer(str(pos))}),
            FakeResponse({"end": FakeChoiceAnswer(str(pos))}),
        ]
    )
    ex = Extractor(client=client)
    result = ex.extract_cascade(paragraph=para, question="When?")
    assert result.answer == "1843"
    assert len(client.calls) == 2  # start + end only
    assert "start" in client.calls[0]["questions"]
