"""Pydantic models and field-spec types for jev-extract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Span(BaseModel):
    """A candidate text span with character offsets in the source paragraph."""

    model_config = ConfigDict(frozen=True)

    text: str
    start: int
    end: int
    key: str


class ExtractResult(BaseModel):
    """Result of extractive span selection via Jev Choice."""

    model_config = ConfigDict(frozen=True)

    answer: str
    start: int
    end: int
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    model: str | None = None
    raw: Any | None = Field(default=None, exclude=False)


FieldMode = Literal["span", "noul", "choice", "score"]


class FieldSpec(BaseModel):
    """Specification for one field in :meth:`Extractor.extract_fields`."""

    model_config = ConfigDict(extra="forbid")

    mode: FieldMode = "span"
    question: str | None = None
    """Natural-language question / instructions for this field."""
    criteria: dict[str, Any] | list[str] | None = None
    """Required for ``choice`` (label → description) and ``score`` (ordered level strings)."""


class FieldResult(BaseModel):
    """Result for one field from :meth:`Extractor.extract_fields`."""

    model_config = ConfigDict(frozen=True)

    mode: FieldMode
    value: Any
    confidence: float | None = None
    probabilities: dict[str, float] | dict[int, float] | None = None
    model: str | None = None
    start: int | None = None
    end: int | None = None
    raw: Any | None = None
