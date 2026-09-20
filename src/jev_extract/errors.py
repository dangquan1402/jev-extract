"""Public exception types for jev-extract."""

from __future__ import annotations


class JevExtractError(Exception):
    """Base error for jev-extract."""


class MissingAPIKeyError(JevExtractError):
    """Raised when no TypeSafe API key is available and no client was supplied."""


class NoCandidatesError(JevExtractError):
    """Raised when a paragraph yields no usable candidate spans."""


class ExtractionError(JevExtractError):
    """Raised when extraction fails (API/SDK error or unexpected response shape)."""
