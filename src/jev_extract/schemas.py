"""Pydantic models and field-spec types for jev-extract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TokenRef(BaseModel):
    """Reference to a token by 0-based position with a word checksum.

    ``pos`` is the source of truth. ``word`` must equal ``tokens[pos]``.
    Inclusive end spans use ``tokens[start.pos : end.pos + 1]``.
    """

    model_config = ConfigDict(frozen=True)

    pos: int
    word: str


class Span(BaseModel):
    """A sentence / chunk candidate span with character offsets (coarse mode).

    Used by :mod:`jev_extract.candidates` for optional sentence-level Choice.
    Default extract path is token-native (:class:`ExtractResult` + :class:`TokenRef`).
    """

    model_config = ConfigDict(frozen=True)

    text: str
    start: int
    end: int
    key: str


class ExtractResult(BaseModel):
    """Result of token-native extractive span selection via Jev Choice.

    Primary fields are token refs + the full token list. ``char_start``,
    ``char_end``, and ``answer`` are derived from the original paragraph via
    the tokenizer's char-span map.
    """

    model_config = ConfigDict(frozen=True)

    start: TokenRef
    end: TokenRef
    tokens: list[str]
    answer: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    confidence: float | None = None
    """Joint / end-step confidence when available; prefer start_/end_ fields."""
    start_confidence: float | None = None
    end_confidence: float | None = None
    probabilities: dict[str, float] | None = None
    """End-step (or joint) Choice probabilities keyed by position string."""
    start_probabilities: dict[str, float] | None = None
    model: str | None = None
    raw: Any | None = Field(default=None, exclude=False)

    @model_validator(mode="after")
    def _checksum_refs(self) -> ExtractResult:
        n = len(self.tokens)
        for label, ref in (("start", self.start), ("end", self.end)):
            if ref.pos < 0 or ref.pos >= n:
                raise ValueError(f"{label}.pos {ref.pos} out of range for {n} tokens")
            if self.tokens[ref.pos] != ref.word:
                raise ValueError(
                    f"{label}.word checksum failed: tokens[{ref.pos}]="
                    f"{self.tokens[ref.pos]!r} != {ref.word!r}"
                )
        if self.end.pos < self.start.pos:
            raise ValueError(
                f"end.pos {self.end.pos} < start.pos {self.start.pos}"
            )
        return self


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
    start: TokenRef | int | None = None
    end: TokenRef | int | None = None
    tokens: list[str] | None = None
    char_start: int | None = None
    char_end: int | None = None
    raw: Any | None = None
