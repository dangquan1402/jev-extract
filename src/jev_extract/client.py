"""High-level Extractor wrapping TypeSafe System One / Jev."""

from __future__ import annotations

import os
from typing import Any, Mapping, Protocol

from typesafe_sdk import Choice, TypeSafeClient
from typesafe_sdk import TypeSafeError as SDKTypeSafeError

from jev_extract.candidates import JEV_CHOICE_MAX_CRITERIA, limit_candidates, propose_candidates
from jev_extract.errors import ExtractionError, MissingAPIKeyError, NoCandidatesError
from jev_extract.questions import (
    build_field_question,
    build_span_choice,
    build_state,
    normalize_field_spec,
)
from jev_extract.schemas import ExtractResult, FieldResult, FieldSpec, Span


class _SystemOneClient(Protocol):
    """Minimal protocol so tests can inject a fake client."""

    def system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any: ...


class Extractor:
    """Extractive paragraph QA via TypeSafe Jev closed-set Choice over spans.

    Parameters
    ----------
    client:
        Optional TypeSafe-compatible client. When omitted, a
        :class:`typesafe_sdk.TypeSafeClient` is created from ``TYPESAFE_API_KEY``.
    model:
        Default model override (SDK default is ``jev-latest``).
    api_key:
        Optional explicit API key when constructing the default client.
    max_candidates:
        Soft/hard cap for Choice criteria cardinality (Jev limit is 255).
    """

    def __init__(
        self,
        *,
        client: _SystemOneClient | None = None,
        model: str = "jev-latest",
        api_key: str | None = None,
        max_candidates: int = JEV_CHOICE_MAX_CRITERIA,
    ) -> None:
        self.model = model
        self.max_candidates = max_candidates
        self._owns_client = client is None

        if client is not None:
            self._client = client
        else:
            key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")
            if key is None or not str(key).strip():
                raise MissingAPIKeyError(
                    "No TypeSafe API key found. Set TYPESAFE_API_KEY or pass "
                    "api_key=... / client=... to Extractor()."
                )
            try:
                self._client = TypeSafeClient(api_key=key, model=model)
            except SDKTypeSafeError as exc:
                raise MissingAPIKeyError(str(exc)) from exc

    @property
    def client(self) -> _SystemOneClient:
        return self._client

    def close(self) -> None:
        """Close the underlying SDK client when this Extractor owns it."""
        if self._owns_client:
            closer = getattr(self._client, "close", None)
            if callable(closer):
                closer()

    def __enter__(self) -> Extractor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _prepare_spans(self, paragraph: str) -> list[Span]:
        spans = propose_candidates(paragraph)
        if not spans:
            raise NoCandidatesError("Paragraph produced no candidate spans.")
        return limit_candidates(spans, max_candidates=self.max_candidates)

    def _call_system_one(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
    ) -> Any:
        try:
            return self._client.system_one(
                state,
                questions,
                model=model if model is not None else self.model,
            )
        except Exception as exc:  # noqa: BLE001 — wrap all SDK/network failures
            raise ExtractionError(f"system_one call failed: {exc}") from exc

    def extract(
        self,
        *,
        paragraph: str,
        question: str,
        model: str | None = None,
        include_raw: bool = False,
    ) -> ExtractResult:
        """Select the best supporting span for *question* from *paragraph*.

        Builds a Jev :class:`~typesafe_sdk.Choice` whose criteria are the candidate
        spans, then maps the winning key back to character offsets.
        """
        spans = self._prepare_spans(paragraph)
        by_key = {s.key: s for s in spans}
        choice_q = build_span_choice(spans, question)
        state = build_state(paragraph, question)

        response = self._call_system_one(
            state,
            {"extract": choice_q},
            model=model,
        )

        answer_obj = _get_choice_answer(response, "extract")
        key = str(answer_obj.choice)
        if key not in by_key:
            raise ExtractionError(
                f"Model returned unknown span key {key!r}; expected one of {sorted(by_key)}."
            )
        span = by_key[key]
        confidence = getattr(answer_obj, "confidence", None)
        probabilities = getattr(answer_obj, "probabilities", None)
        if probabilities is not None and not isinstance(probabilities, dict):
            probabilities = dict(probabilities)

        return ExtractResult(
            answer=span.text,
            start=span.start,
            end=span.end,
            confidence=float(confidence) if confidence is not None else None,
            probabilities=probabilities,
            model=getattr(response, "model", None),
            raw=response if include_raw else None,
        )

    def extract_fields(
        self,
        *,
        paragraph: str,
        fields: Mapping[str, FieldSpec | Mapping[str, Any] | str],
        model: str | None = None,
        include_raw: bool = False,
    ) -> dict[str, FieldResult]:
        """Extract multiple fields; modes: ``span``, ``noul``, ``choice``, ``score``.

        - ``span``: closed-set Choice over sentence candidates (same as :meth:`extract`).
        - ``noul``: yes/no probability; optional criteria ``{"true": ..., "false": ...}``.
        - ``choice``: requires ``criteria`` dict of label → description.
        - ``score``: requires ``criteria`` list of ordered level strings.
        """
        if not fields:
            raise ValueError("fields must be a non-empty mapping")

        specs = {name: normalize_field_spec(spec) for name, spec in fields.items()}
        needs_spans = any(s.mode == "span" for s in specs.values())
        spans: list[Span] = self._prepare_spans(paragraph) if needs_spans else []
        by_key = {s.key: s for s in spans}

        questions: dict[str, Any] = {}
        for name, spec in specs.items():
            try:
                questions[name] = build_field_question(name, spec, spans)
            except ValueError as exc:
                raise ExtractionError(str(exc)) from exc

        state = build_state(paragraph)
        response = self._call_system_one(state, questions, model=model)
        resp_model = getattr(response, "model", None)

        results: dict[str, FieldResult] = {}
        for name, spec in specs.items():
            results[name] = _field_result_from_response(
                response,
                name=name,
                spec=spec,
                by_key=by_key,
                model=resp_model,
                include_raw=include_raw,
            )
        return results


def _get_choice_answer(response: Any, name: str) -> Any:
    choices = getattr(response, "choices", None)
    if isinstance(choices, Mapping) and name in choices:
        return choices[name]
    answers = getattr(response, "answers", None)
    if isinstance(answers, Mapping) and name in answers:
        ans = answers[name]
        if getattr(ans, "type", None) == "choice" or hasattr(ans, "choice"):
            return ans
    raise ExtractionError(f"Response missing choice answer for question {name!r}.")


def _get_answer(response: Any, name: str) -> Any:
    answers = getattr(response, "answers", None)
    if isinstance(answers, Mapping) and name in answers:
        return answers[name]
    # Try typed groupings
    for group_name in ("choices", "nouls", "scores"):
        group = getattr(response, group_name, None)
        if isinstance(group, Mapping) and name in group:
            return group[name]
    raise ExtractionError(f"Response missing answer for question {name!r}.")


def _field_result_from_response(
    response: Any,
    *,
    name: str,
    spec: FieldSpec,
    by_key: Mapping[str, Span],
    model: str | None,
    include_raw: bool,
) -> FieldResult:
    ans = _get_answer(response, name)
    raw = ans if include_raw else None

    if spec.mode == "span":
        key = str(ans.choice)
        if key not in by_key:
            raise ExtractionError(
                f"Field '{name}': unknown span key {key!r}; expected one of {sorted(by_key)}."
            )
        span = by_key[key]
        probs = getattr(ans, "probabilities", None)
        return FieldResult(
            mode="span",
            value=span.text,
            confidence=_as_float(getattr(ans, "confidence", None)),
            probabilities=dict(probs) if probs is not None else None,
            model=model,
            start=span.start,
            end=span.end,
            raw=raw,
        )

    if spec.mode == "noul":
        return FieldResult(
            mode="noul",
            value=_as_float(getattr(ans, "noul", None)),
            confidence=None,
            probabilities=None,
            model=model,
            raw=raw,
        )

    if spec.mode == "choice":
        probs = getattr(ans, "probabilities", None)
        return FieldResult(
            mode="choice",
            value=str(ans.choice),
            confidence=_as_float(getattr(ans, "confidence", None)),
            probabilities=dict(probs) if probs is not None else None,
            model=model,
            raw=raw,
        )

    if spec.mode == "score":
        probs = getattr(ans, "probabilities", None)
        # Score probabilities are keyed by int levels
        if probs is not None:
            probs = {int(k): float(v) for k, v in dict(probs).items()}
        return FieldResult(
            mode="score",
            value=_as_float(getattr(ans, "score", None)),
            confidence=_as_float(getattr(ans, "confidence", None)),
            probabilities=probs,
            model=model,
            raw=raw,
        )

    raise ExtractionError(f"Unhandled mode {spec.mode!r} for field '{name}'.")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


# Re-export Choice for type checkers / advanced users inspecting built questions.
__all__ = ["Extractor", "Choice"]
