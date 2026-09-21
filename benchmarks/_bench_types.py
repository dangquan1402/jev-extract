"""Benchmark backends (TypeSafe + stubs)."""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
_SECRETS_ENV = os.environ.get("JEV_EXTRACT_SECRETS_JSON", "").strip()
SECRETS_PATH = Path(_SECRETS_ENV) if _SECRETS_ENV else None
PINNED_MODEL = "jev-latest"

_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

def load_typesafe_api_key() -> str | None:
    """Return TYPESAFE_API_KEY from env, or optional JSON secrets file.

    If ``JEV_EXTRACT_SECRETS_JSON`` points to a JSON file with
    ``{"card": {"TYPESAFE_API_KEY": "..."}}``, that key is used when the env
    var is unset. Never prints key values.
    """
    env = os.environ.get("TYPESAFE_API_KEY")
    if env and str(env).strip():
        return str(env).strip()
    if SECRETS_PATH is not None and SECRETS_PATH.is_file():
        try:
            data = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
            card = data.get("card") or {}
            key = card.get("TYPESAFE_API_KEY")
            if key and str(key).strip():
                os.environ["TYPESAFE_API_KEY"] = str(key).strip()
                return str(key).strip()
        except (OSError, json.JSONDecodeError, TypeError):
            return None
    return None


def _usage_from_response(response: Any) -> tuple[int | None, int | None]:
    """Sum usage from a single response or a dict of responses (start/end)."""
    if response is None:
        return None, None
    if isinstance(response, dict):
        total_in = 0
        total_out = 0
        seen = False
        for v in response.values():
            inp, out = _usage_from_response(v)
            if inp is not None:
                total_in += int(inp)
                seen = True
            if out is not None:
                total_out += int(out)
                seen = True
        if not seen:
            return None, None
        return total_in, total_out
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    return getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None)


# ---------------------------------------------------------------------------
# Backend protocol + registry
# ---------------------------------------------------------------------------


@dataclass
class ClassifyOut:
    pred: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    error: str | None = None


@dataclass
class ExtractOut:
    pred: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    start: int | None = None
    end: int | None = None
    error: str | None = None


class ClassifyBackend(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...
    def classify(self, text: str, labels: dict[str, str]) -> ClassifyOut: ...
    def close(self) -> None: ...


class ExtractBackend(Protocol):
    name: str

    def available(self) -> tuple[bool, str]: ...
    def extract(self, paragraph: str, question: str) -> ExtractOut: ...
    def close(self) -> None: ...


class StubBackend:
    """Placeholder for Gemini / Haiku — not configured yet."""

    def __init__(self, name: str) -> None:
        self.name = name

    def available(self) -> tuple[bool, str]:
        return False, "not configured yet"

    def classify(self, text: str, labels: dict[str, str]) -> ClassifyOut:
        raise RuntimeError(f"{self.name}: not configured yet")

    def extract(self, paragraph: str, question: str) -> ExtractOut:
        raise RuntimeError(f"{self.name}: not configured yet")

    def close(self) -> None:
        return None
