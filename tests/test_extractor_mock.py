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
    propose_candidates,
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
    def __init__(self, response: FakeResponse | None = None, exc: Exception | None = None):
        self.response = response
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def system_one(self, state: Any, questions: dict[str, Any], *, model: str | None = None, **kwargs: Any):
        self.calls.append({"state": state, "questions": questions, "model": model, "kwargs": kwargs})
        if self.exc is not None:
            raise self.exc
        assert self.response is not None
        return self.response


PARAGRAPH = (
    "Ada Lovelace worked on Babbage's Analytical Engine. "
    "She published notes in 1843. "
    "Those notes include an early algorithm."
)


def test_extract_maps_choice_to_span():
    spans = propose_candidates(PARAGRAPH)
    assert len(spans) >= 2
    target = spans[1]  # "She published notes in 1843."

    client = FakeClient(
        FakeResponse({"extract": FakeChoiceAnswer(target.key, confidence=0.95)})
    )
    ex = Extractor(client=client)
    result = ex.extract(paragraph=PARAGRAPH, question="When were the notes published?")

    assert result.answer == target.text
    assert result.start == target.start
    assert result.end == target.end
    assert result.confidence == pytest.approx(0.95)
    assert result.model == "jev-latest"
    assert len(client.calls) == 1
    q = client.calls[0]["questions"]["extract"]
    assert q.type == "choice"
    assert target.key in q.criteria
    assert client.calls[0]["state"]["paragraph"] == PARAGRAPH


def test_extract_unknown_key_raises():
    client = FakeClient(FakeResponse({"extract": FakeChoiceAnswer("nope")}))
    ex = Extractor(client=client)
    with pytest.raises(ExtractionError, match="unknown span key"):
        ex.extract(paragraph=PARAGRAPH, question="What?")


def test_extract_no_candidates():
    client = FakeClient(FakeResponse({"extract": FakeChoiceAnswer("s0")}))
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
    spans = propose_candidates(PARAGRAPH)
    target = spans[0]

    client = FakeClient(
        FakeResponse(
            {
                "who": FakeChoiceAnswer(target.key),
                "is_history": FakeNoulAnswer(0.91),
                "topic": FakeChoiceAnswer("math", confidence=0.8, probabilities={"math": 0.8, "other": 0.2}),
                "detail": FakeScoreAnswer(1.7, confidence=0.6),
            }
        )
    )
    ex = Extractor(client=client)
    results = ex.extract_fields(
        paragraph=PARAGRAPH,
        fields={
            "who": {"mode": "span", "question": "Who is discussed?"},
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

    assert results["who"].mode == "span"
    assert results["who"].value == target.text
    assert results["who"].start == target.start
    assert results["is_history"].mode == "noul"
    assert results["is_history"].value == pytest.approx(0.91)
    assert results["topic"].value == "math"
    assert results["detail"].value == pytest.approx(1.7)
    assert results["detail"].probabilities is not None


def test_extract_fields_choice_requires_criteria():
    client = FakeClient(FakeResponse({}))
    ex = Extractor(client=client)
    with pytest.raises(ExtractionError, match="criteria dict"):
        ex.extract_fields(
            paragraph=PARAGRAPH,
            fields={"t": {"mode": "choice", "question": "Topic?"}},
        )


def test_extract_fields_string_shorthand():
    spans = propose_candidates(PARAGRAPH)
    target = spans[0]
    client = FakeClient(FakeResponse({"q": FakeChoiceAnswer(target.key)}))
    ex = Extractor(client=client)
    results = ex.extract_fields(paragraph=PARAGRAPH, fields={"q": "Who?"})
    assert results["q"].value == target.text


def test_include_raw():
    spans = propose_candidates(PARAGRAPH)
    target = spans[0]
    resp = FakeResponse({"extract": FakeChoiceAnswer(target.key)})
    client = FakeClient(resp)
    ex = Extractor(client=client)
    result = ex.extract(paragraph=PARAGRAPH, question="Who?", include_raw=True)
    assert result.raw is resp
