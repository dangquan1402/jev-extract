"""TypeSafe benchmark backends."""
from __future__ import annotations

import time
from typing import Any

from _bench_types import (
    PINNED_MODEL,
    ClassifyBackend,
    ClassifyOut,
    ExtractBackend,
    ExtractOut,
    StubBackend,
    load_typesafe_api_key,
    _usage_from_response,
)

class TypeSafeClassifyBackend:
    name = "typesafe"

    def __init__(self, model: str = PINNED_MODEL) -> None:
        self.model = model
        self._client: Any = None

    def available(self) -> tuple[bool, str]:
        key = load_typesafe_api_key()
        if not key:
            return False, "missing TYPESAFE_API_KEY (env or card secrets)"
        try:
            from typesafe_sdk import TypeSafeClient

            self._client = TypeSafeClient(api_key=key, model=self.model)
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, f"TypeSafeClient unavailable: {exc}"

    def classify(self, text: str, labels: dict[str, str]) -> ClassifyOut:
        from typesafe_sdk import Choice

        assert self._client is not None
        t0 = time.perf_counter()
        try:
            response = self._client.system_one(
                {"text": text},
                {
                    "label": Choice(
                        criteria=dict(labels),
                        instructions=(
                            "Classify the support message into exactly one label. "
                            "Pick the best matching criterion."
                        ),
                    )
                },
                model=self.model,
            )
            latency = (time.perf_counter() - t0) * 1000.0
            ans = response.choices.get("label") or response.answers.get("label")
            pred = str(ans.choice)
            inp, out = _usage_from_response(response)
            return ClassifyOut(
                pred=pred,
                latency_ms=latency,
                input_tokens=inp,
                output_tokens=out,
                model=getattr(response, "model", None) or self.model,
            )
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - t0) * 1000.0
            return ClassifyOut(
                pred="",
                latency_ms=latency,
                error=f"{type(exc).__name__}: {exc}",
                model=self.model,
            )

    def close(self) -> None:
        closer = getattr(self._client, "close", None)
        if callable(closer):
            closer()


class TypeSafeExtractBackend:
    name = "typesafe"

    def __init__(self, model: str = PINNED_MODEL) -> None:
        self.model = model
        self._extractor: Any = None

    def available(self) -> tuple[bool, str]:
        key = load_typesafe_api_key()
        if not key:
            return False, "missing TYPESAFE_API_KEY (env or card secrets)"
        try:
            from jev_extract import Extractor

            self._extractor = Extractor(api_key=key, model=self.model)
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, f"jev_extract.Extractor unavailable: {exc}"

    def extract(self, paragraph: str, question: str) -> ExtractOut:
        assert self._extractor is not None
        t0 = time.perf_counter()
        try:
            result = self._extractor.extract(
                paragraph=paragraph,
                question=question,
                include_raw=True,
            )
            latency = (time.perf_counter() - t0) * 1000.0
            inp, out = _usage_from_response(result.raw)
            start_pos = result.start.pos if result.start is not None else None
            end_pos = result.end.pos if result.end is not None else None
            return ExtractOut(
                pred=result.answer or "",
                latency_ms=latency,
                input_tokens=inp,
                output_tokens=out,
                model=result.model or self.model,
                start=start_pos,
                end=end_pos,
            )
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - t0) * 1000.0
            return ExtractOut(
                pred="",
                latency_ms=latency,
                error=f"{type(exc).__name__}: {exc}",
                model=self.model,
            )

    def close(self) -> None:
        closer = getattr(self._extractor, "close", None)
        if callable(closer):
            closer()


def get_classify_backend(name: str) -> ClassifyBackend:
    registry = {
        "typesafe": lambda: TypeSafeClassifyBackend(),
        "gemini": lambda: StubBackend("gemini"),
        "haiku": lambda: StubBackend("haiku"),
    }
    if name not in registry:
        raise SystemExit(f"Unknown backend {name!r}; choose from {sorted(registry)}")
    return registry[name]()


def get_extract_backend(name: str) -> ExtractBackend:
    registry = {
        "typesafe": lambda: TypeSafeExtractBackend(),
        "gemini": lambda: StubBackend("gemini"),
        "haiku": lambda: StubBackend("haiku"),
    }
    if name not in registry:
        raise SystemExit(f"Unknown backend {name!r}; choose from {sorted(registry)}")
    return registry[name]()



# Re-exports for callers
from _bench_types import (  # noqa: E402
    ClassifyBackend,
    ClassifyOut,
    ExtractBackend,
    ExtractOut,
    StubBackend,
    load_typesafe_api_key,
)
