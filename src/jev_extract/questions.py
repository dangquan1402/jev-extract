"""Helpers that build TypeSafe System One questions for extraction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from typesafe_sdk import Choice, Noul, Score

from jev_extract.candidates import spans_to_criteria
from jev_extract.schemas import FieldSpec, Span

DEFAULT_SPAN_INSTRUCTIONS = (
    "Select the single candidate span that best answers the question. "
    "Prefer the most specific supporting sentence; do not invent text."
)


def build_span_choice(
    spans: Sequence[Span],
    question: str,
    *,
    instructions: str | None = None,
) -> Choice:
    """Build a closed-set :class:`~typesafe_sdk.Choice` over candidate spans."""
    criteria = spans_to_criteria(spans)
    instr = instructions or f"{DEFAULT_SPAN_INSTRUCTIONS}\n\nQuestion: {question}"
    return Choice(instructions=instr, criteria=criteria)


def build_state(paragraph: str, question: str | None = None) -> dict[str, str]:
    """Structured state passed to ``system_one``."""
    state: dict[str, str] = {"paragraph": paragraph}
    if question:
        state["question"] = question
    return state


def build_field_question(name: str, spec: FieldSpec, spans: Sequence[Span]) -> Any:
    """Construct a Choice / Noul / Score question for one named field."""
    q = spec.question or f"Extract field '{name}' from the paragraph."

    if spec.mode == "span":
        return build_span_choice(spans, q)

    if spec.mode == "noul":
        criteria = None
        if isinstance(spec.criteria, Mapping):
            criteria = dict(spec.criteria)  # type: ignore[arg-type]
        return Noul(instructions=q, criteria=criteria)

    if spec.mode == "choice":
        if not isinstance(spec.criteria, Mapping) or not spec.criteria:
            raise ValueError(
                f"Field '{name}' mode='choice' requires a non-empty criteria dict "
                "(label → description)."
            )
        return Choice(instructions=q, criteria=dict(spec.criteria))

    if spec.mode == "score":
        if not isinstance(spec.criteria, Sequence) or isinstance(spec.criteria, (str, bytes)):
            raise ValueError(
                f"Field '{name}' mode='score' requires a criteria list of level strings."
            )
        if len(spec.criteria) == 0:
            raise ValueError(f"Field '{name}' mode='score' criteria list must be non-empty.")
        return Score(instructions=q, criteria=list(spec.criteria))

    raise ValueError(f"Unknown field mode for '{name}': {spec.mode!r}")


def normalize_field_spec(value: FieldSpec | Mapping[str, Any] | str) -> FieldSpec:
    """Accept FieldSpec, dict, or bare question string (defaults to span mode)."""
    if isinstance(value, FieldSpec):
        return value
    if isinstance(value, str):
        return FieldSpec(mode="span", question=value)
    return FieldSpec.model_validate(dict(value))
