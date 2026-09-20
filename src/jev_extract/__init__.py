"""jev-extract — paragraph information extraction using TypeSafe AI Jev."""

from __future__ import annotations

from jev_extract.candidates import (
    JEV_CHOICE_MAX_CRITERIA,
    limit_candidates,
    propose_candidates,
)
from jev_extract.client import Extractor
from jev_extract.errors import (
    ExtractionError,
    JevExtractError,
    MissingAPIKeyError,
    NoCandidatesError,
)
from jev_extract.schemas import ExtractResult, FieldResult, FieldSpec, Span

__version__ = "0.1.0"

__all__ = [
    "Extractor",
    "ExtractResult",
    "FieldResult",
    "FieldSpec",
    "Span",
    "propose_candidates",
    "limit_candidates",
    "JEV_CHOICE_MAX_CRITERIA",
    "JevExtractError",
    "MissingAPIKeyError",
    "NoCandidatesError",
    "ExtractionError",
    "__version__",
]
