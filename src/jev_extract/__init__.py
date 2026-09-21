"""jev-extract — paragraph information extraction using TypeSafe AI Jev."""

from __future__ import annotations

from jev_extract.candidates import (
    JEV_CHOICE_MAX_CRITERIA,
    limit_candidates,
    propose_candidates,
    propose_chunks,
)
from jev_extract.client import Extractor
from jev_extract.strategies import EXTRACT_MODES
from jev_extract.errors import (
    ExtractionError,
    JevExtractError,
    MissingAPIKeyError,
    NoCandidatesError,
)
from jev_extract.schemas import ExtractResult, FieldResult, FieldSpec, Span, TokenRef
from jev_extract.tokenize import (
    TOKENIZER_VERSION,
    tokenize,
    token_char_spans,
    tokenize_with_spans,
)

__version__ = "0.2.0"

__all__ = [
    "Extractor",
    "EXTRACT_MODES",
    "ExtractResult",
    "FieldResult",
    "FieldSpec",
    "Span",
    "TokenRef",
    "tokenize",
    "token_char_spans",
    "tokenize_with_spans",
    "TOKENIZER_VERSION",
    "propose_candidates",
    "propose_chunks",
    "limit_candidates",
    "JEV_CHOICE_MAX_CRITERIA",
    "JevExtractError",
    "MissingAPIKeyError",
    "NoCandidatesError",
    "ExtractionError",
    "__version__",
]
