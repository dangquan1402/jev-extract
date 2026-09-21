"""High-level Extractor wrapping TypeSafe System One / Jev."""

from __future__ import annotations

import os
from typing import Any, Mapping, Protocol

from typesafe_sdk import Choice, TypeSafeClient
from typesafe_sdk import TypeSafeError as SDKTypeSafeError

from jev_extract.candidates import (
    JEV_CHOICE_MAX_CRITERIA,
    limit_candidates,
    propose_candidates,
    propose_chunks,
)
from jev_extract.errors import ExtractionError, MissingAPIKeyError, NoCandidatesError
from jev_extract.questions import (
    DEFAULT_CONTEXT_RADIUS,
    build_end_choice,
    build_field_question,
    build_joint_span_choice,
    build_span_choice,
    build_start_choice,
    build_state,
    normalize_field_spec,
)
from jev_extract.schemas import ExtractResult, FieldResult, FieldSpec, Span, TokenRef
from jev_extract.tokenize import (
    answer_from_token_span,
    char_span_to_token_span,
    token_windows,
    tokenize_with_spans,
    validate_token_ref,
)


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
    """Extractive paragraph QA via TypeSafe Jev closed-set Choice over **tokens**.

    Default path (token-native):
      1. Tokenize the paragraph (tokenizer v1).
      2. If ``len(tokens) > 255``, use sliding windows of size ≤255 (overlap 64).
      3. **Start Choice** — criteria keys are positions (``"4"``); descriptions are
         local context windows with the candidate marked ``«word»``.
      4. **End Choice** — sequential second call; only ``pos >= start.pos``;
         instructions mention ``start was pos:word`` so duplicates do not latch wrong.
      5. Validate word checksums; derive ``answer`` / char offsets from the token map.

    ``extract_mode``: ``"sequential"`` (default path), ``"joint"`` (single Choice over
    ``(start,end)`` pairs when cardinality allows), or ``"auto"`` (sequential; if the
    start word is duplicated and start confidence is missing/low, re-ask end with a
    richer context radius).

    Sentence-candidate Choice remains available via :meth:`extract_sentence` for
    coarse / legacy comparisons.

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
        Cap for Choice criteria cardinality (Jev limit is 255).
    context_radius:
        Tokens of left/right context in Choice descriptions (default 4).
    extract_mode:
        ``"sequential"`` | ``"joint"`` | ``"auto"`` (default ``"auto"``) plus
        strategy modes: ``span_choice`` (smart propose by default),
        ``span_choice_smart``, ``span_choice_in_chunk``, ``anchor_expand``,
        ``shape_gate``, ``topk_noul``, ``ambiguous_joint``, ``length_rerank``.
    """

    def __init__(
        self,
        *,
        client: _SystemOneClient | None = None,
        model: str = "jev-latest",
        api_key: str | None = None,
        max_candidates: int = JEV_CHOICE_MAX_CRITERIA,
        context_radius: int = DEFAULT_CONTEXT_RADIUS,
        extract_mode: str = "auto",
    ) -> None:
        from jev_extract.strategies import EXTRACT_MODES

        # Public extract() routes strategy modes; chunk_only/cascade use dedicated methods.
        _allowed = tuple(m for m in EXTRACT_MODES if m not in ("chunk_only", "cascade"))
        if extract_mode not in _allowed:
            raise ValueError(
                f"extract_mode must be one of {_allowed}; got {extract_mode!r}"
            )
        self.model = model
        self.max_candidates = max_candidates
        self.context_radius = context_radius
        self.extract_mode = extract_mode
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

    def _prepare_chunks(self, paragraph: str) -> list[Span]:
        spans = propose_chunks(paragraph)
        if not spans:
            raise NoCandidatesError("Paragraph produced no candidate chunks.")
        return limit_candidates(spans, max_candidates=self.max_candidates)

    def _choose_chunk(
        self,
        *,
        paragraph: str,
        question: str,
        spans: list[Span],
        state: Any,
        model: str | None,
    ) -> tuple[Span, float | None, dict[str, float] | None, Any]:
        """Run sentence/chunk Choice; return (span, confidence, probs, response)."""
        by_key = {s.key: s for s in spans}
        if len(spans) == 1:
            # No Choice needed — single chunk covers the paragraph.
            return spans[0], None, None, None
        choice_q = build_span_choice(spans, question)
        response = self._call_system_one(state, {"chunk": choice_q}, model=model)
        answer_obj = _get_choice_answer(response, "chunk")
        key = str(answer_obj.choice)
        if key not in by_key:
            raise ExtractionError(
                f"Model returned unknown chunk key {key!r}; expected one of {sorted(by_key)}."
            )
        confidence = getattr(answer_obj, "confidence", None)
        probabilities = getattr(answer_obj, "probabilities", None)
        if probabilities is not None and not isinstance(probabilities, dict):
            probabilities = dict(probabilities)
        return (
            by_key[key],
            float(confidence) if confidence is not None else None,
            probabilities,
            response,
        )

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
        extract_mode: str | None = None,
        context_radius: int | None = None,
    ) -> ExtractResult:
        """Select a token span answering *question* from *paragraph*.

        Runs sequential **start** then **end** Jev Choice calls over tokenizer-v1
        tokens (or a joint ``(start,end)`` Choice when ``extract_mode="joint"``).
        Character offsets and the answer string are derived fields.

        Criterion descriptions use local context with ``«word»`` markers so
        duplicate surface forms stay distinguishable.
        """
        from collections import Counter

        mode = extract_mode if extract_mode is not None else self.extract_mode
        radius = self.context_radius if context_radius is None else context_radius

        # Strategy modes (implemented in jev_extract.strategies).
        if mode in ("span_choice", "span_choice_smart", "span_choice_in_chunk"):
            propose = {
                "span_choice": "smart",
                "span_choice_smart": "smart",
                "span_choice_in_chunk": "in_chunk",
            }[mode]
            return self.extract_span_choice(
                paragraph=paragraph,
                question=question,
                model=model,
                include_raw=include_raw,
                context_radius=radius,
                propose=propose,
            )
        if mode == "anchor_expand":
            return self.extract_anchor_expand(
                paragraph=paragraph,
                question=question,
                model=model,
                include_raw=include_raw,
                context_radius=radius,
            )
        if mode == "shape_gate":
            return self.extract_shape_gate(
                paragraph=paragraph,
                question=question,
                model=model,
                include_raw=include_raw,
                context_radius=radius,
            )
        if mode == "topk_noul":
            return self.extract_topk_noul(
                paragraph=paragraph,
                question=question,
                model=model,
                include_raw=include_raw,
                context_radius=radius,
            )
        if mode == "ambiguous_joint":
            return self.extract_ambiguous_joint(
                paragraph=paragraph,
                question=question,
                model=model,
                include_raw=include_raw,
                context_radius=radius,
            )
        if mode == "length_rerank":
            return self.extract_length_rerank(
                paragraph=paragraph,
                question=question,
                model=model,
                include_raw=include_raw,
                context_radius=radius,
            )
        if mode not in ("sequential", "joint", "auto"):
            raise ValueError(
                f"extract_mode must be a known mode; got {mode!r}"
            )

        tokens, char_spans = tokenize_with_spans(paragraph)
        if not tokens:
            raise NoCandidatesError("Paragraph produced no tokens.")

        windows = token_windows(
            len(tokens),
            max_size=self.max_candidates,
            overlap=min(64, max(0, self.max_candidates - 1)),
        )

        state = build_state(paragraph, question)

        if mode == "joint":
            return self._extract_joint(
                tokens=tokens,
                char_spans=char_spans,
                paragraph=paragraph,
                question=question,
                state=state,
                model=model,
                include_raw=include_raw,
                radius=radius,
            )

        # --- Start Choice (possibly multi-window) ---
        start_pos, start_conf, start_probs, start_resp = self._choose_start(
            tokens=tokens,
            question=question,
            state=state,
            windows=windows,
            model=model,
            context_radius=radius,
        )
        start_ref = TokenRef(pos=start_pos, word=tokens[start_pos])
        validate_token_ref(tokens, start_ref)

        # --- End Choice: only pos >= start; richer context on auto+duplicates ---
        end_hi = min(start_pos + self.max_candidates, len(tokens))
        end_radius = radius
        if mode == "auto":
            counts = Counter(tokens)
            start_word = tokens[start_pos]
            low_conf = start_conf is None or float(start_conf) < 0.55
            if counts[start_word] > 1 and low_conf:
                end_radius = max(radius, radius + 4)

        end_choice = build_end_choice(
            tokens,
            question,
            start_pos=start_pos,
            hi=end_hi,
            start_word=tokens[start_pos],
            context_radius=end_radius,
        )
        end_resp = self._call_system_one(
            state,
            {"end": end_choice},
            model=model,
        )
        end_ans = _get_choice_answer(end_resp, "end")
        end_pos = _parse_pos_key(end_ans.choice, lo=start_pos, hi=end_hi)
        end_ref = TokenRef(pos=end_pos, word=tokens[end_pos])
        validate_token_ref(tokens, end_ref)

        answer, char_start, char_end = answer_from_token_span(
            paragraph, tokens, char_spans, start_pos, end_pos
        )

        end_conf = getattr(end_ans, "confidence", None)
        end_probs = getattr(end_ans, "probabilities", None)
        if end_probs is not None and not isinstance(end_probs, dict):
            end_probs = dict(end_probs)

        joint_conf: float | None = None
        if start_conf is not None and end_conf is not None:
            joint_conf = float(start_conf) * float(end_conf)
        elif end_conf is not None:
            joint_conf = float(end_conf)
        elif start_conf is not None:
            joint_conf = float(start_conf)

        resp_model = (
            getattr(end_resp, "model", None)
            or getattr(start_resp, "model", None)
        )
        raw: Any = None
        if include_raw:
            raw = {"start": start_resp, "end": end_resp}

        return ExtractResult(
            start=start_ref,
            end=end_ref,
            tokens=list(tokens),
            answer=answer,
            char_start=char_start,
            char_end=char_end,
            confidence=joint_conf,
            start_confidence=float(start_conf) if start_conf is not None else None,
            end_confidence=float(end_conf) if end_conf is not None else None,
            probabilities=end_probs,
            start_probabilities=start_probs,
            model=resp_model,
            raw=raw,
        )

    def _choose_start(
        self,
        *,
        tokens: list[str],
        question: str,
        state: Mapping[str, str],
        windows: list[tuple[int, int]],
        model: str | None,
        context_radius: int | None = None,
    ) -> tuple[int, float | None, dict[str, float] | None, Any]:
        """Run start Choice over one or more windows; return best (pos, conf, probs, resp)."""
        radius = self.context_radius if context_radius is None else context_radius
        best_pos: int | None = None
        best_conf = float("-inf")
        best_probs: dict[str, float] | None = None
        best_resp: Any = None
        last_resp: Any = None

        for lo, hi in windows:
            choice_q = build_start_choice(
                tokens, question, lo=lo, hi=hi, context_radius=radius
            )
            resp = self._call_system_one(state, {"start": choice_q}, model=model)
            last_resp = resp
            ans = _get_choice_answer(resp, "start")
            pos = _parse_pos_key(ans.choice, lo=lo, hi=hi)
            conf = getattr(ans, "confidence", None)
            conf_f = float(conf) if conf is not None else 0.0
            probs = getattr(ans, "probabilities", None)
            if probs is not None and not isinstance(probs, dict):
                probs = dict(probs)
            if best_pos is None or conf_f > best_conf:
                best_pos = pos
                best_conf = conf_f
                best_probs = probs
                best_resp = resp

        assert best_pos is not None
        conf_out: float | None = None if best_conf == float("-inf") else best_conf
        # If the API omitted confidence on a single window, treat as None not 0.
        if len(windows) == 1:
            single_ans = _get_choice_answer(best_resp or last_resp, "start")
            raw_conf = getattr(single_ans, "confidence", None)
            conf_out = float(raw_conf) if raw_conf is not None else None
        return best_pos, conf_out, best_probs, best_resp or last_resp

    def _extract_joint(
        self,
        *,
        tokens: list[str],
        char_spans: list[tuple[int, int]],
        paragraph: str,
        question: str,
        state: Mapping[str, str],
        model: str | None,
        include_raw: bool,
        radius: int,
    ) -> ExtractResult:
        """Single Choice over (start,end) pairs from top start alternatives.

        Uses a cheap start Choice first to collect up to 3 start candidates
        (by probability mass), then a joint Choice over those starts × ends
        when cardinality ≤ max_candidates; otherwise falls back to sequential.
        """
        windows = token_windows(
            len(tokens),
            max_size=self.max_candidates,
            overlap=min(64, max(0, self.max_candidates - 1)),
        )
        start_pos, start_conf, start_probs, start_resp = self._choose_start(
            tokens=tokens,
            question=question,
            state=state,
            windows=windows,
            model=model,
            context_radius=radius,
        )

        # Top-k starts from probabilities, else just the chosen start.
        starts: list[int] = [start_pos]
        if start_probs:
            ranked = sorted(
                ((int(k.split(":", 1)[0]) if ":" in str(k) else int(k), float(v))
                 for k, v in start_probs.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )
            for pos, _ in ranked:
                if 0 <= pos < len(tokens) and pos not in starts:
                    starts.append(pos)
                if len(starts) >= 3:
                    break

        end_hi = len(tokens)
        # Cardinality check: sum of (end_hi - s) over starts
        n_pairs = sum(max(0, end_hi - s) for s in starts)
        if n_pairs > self.max_candidates or n_pairs == 0:
            # Fall back to sequential end Choice
            end_hi_seq = min(start_pos + self.max_candidates, len(tokens))
            end_choice = build_end_choice(
                tokens,
                question,
                start_pos=start_pos,
                hi=end_hi_seq,
                start_word=tokens[start_pos],
                context_radius=radius,
            )
            end_resp = self._call_system_one(state, {"end": end_choice}, model=model)
            end_ans = _get_choice_answer(end_resp, "end")
            end_pos = _parse_pos_key(end_ans.choice, lo=start_pos, hi=end_hi_seq)
            end_conf = getattr(end_ans, "confidence", None)
            end_probs = getattr(end_ans, "probabilities", None)
            if end_probs is not None and not isinstance(end_probs, dict):
                end_probs = dict(end_probs)
            raw = {"start": start_resp, "end": end_resp} if include_raw else None
            joint_conf = None
            if start_conf is not None and end_conf is not None:
                joint_conf = float(start_conf) * float(end_conf)
            elif end_conf is not None:
                joint_conf = float(end_conf)
            elif start_conf is not None:
                joint_conf = float(start_conf)
        else:
            joint_q = build_joint_span_choice(
                tokens,
                question,
                start_candidates=starts,
                end_hi=end_hi,
                context_radius=radius,
                max_pairs=self.max_candidates,
            )
            end_resp = self._call_system_one(state, {"span": joint_q}, model=model)
            span_ans = _get_choice_answer(end_resp, "span")
            key = str(span_ans.choice).strip()
            if ":" not in key:
                raise ExtractionError(
                    f"Joint mode expected 'start:end' key, got {key!r}."
                )
            s_str, e_str = key.split(":", 1)
            try:
                start_pos = int(s_str)
                end_pos = int(e_str)
            except ValueError as exc:
                raise ExtractionError(
                    f"Joint mode returned non-integer span {key!r}."
                ) from exc
            if not (0 <= start_pos <= end_pos < len(tokens)):
                raise ExtractionError(
                    f"Joint mode returned invalid span [{start_pos}, {end_pos}]."
                )
            end_conf = getattr(span_ans, "confidence", None)
            end_probs = getattr(span_ans, "probabilities", None)
            if end_probs is not None and not isinstance(end_probs, dict):
                end_probs = dict(end_probs)
            joint_conf = float(end_conf) if end_conf is not None else (
                float(start_conf) if start_conf is not None else None
            )
            raw = {"start": start_resp, "span": end_resp} if include_raw else None

        start_ref = TokenRef(pos=start_pos, word=tokens[start_pos])
        end_ref = TokenRef(pos=end_pos, word=tokens[end_pos])
        validate_token_ref(tokens, start_ref)
        validate_token_ref(tokens, end_ref)
        answer, char_start, char_end = answer_from_token_span(
            paragraph, tokens, char_spans, start_pos, end_pos
        )
        resp_model = (
            getattr(end_resp, "model", None)
            or getattr(start_resp, "model", None)
        )
        return ExtractResult(
            start=start_ref,
            end=end_ref,
            tokens=list(tokens),
            answer=answer,
            char_start=char_start,
            char_end=char_end,
            confidence=joint_conf,
            start_confidence=float(start_conf) if start_conf is not None else None,
            end_confidence=float(end_conf) if end_conf is not None else None,
            probabilities=end_probs if isinstance(end_probs, dict) else None,
            start_probabilities=start_probs,
            model=resp_model,
            raw=raw,
        )

    def extract_sentence(
        self,
        *,
        paragraph: str,
        question: str,
        model: str | None = None,
        include_raw: bool = False,
    ) -> dict[str, Any]:
        """Optional coarse mode: Choice over sentence candidates (legacy path).

        Returns a plain dict with ``answer``, ``start``/``end`` char offsets,
        and optional confidence — not a token-native :class:`ExtractResult`.
        """
        spans = self._prepare_spans(paragraph)
        by_key = {s.key: s for s in spans}
        choice_q = build_span_choice(spans, question)
        state = build_state(paragraph, question)
        response = self._call_system_one(state, {"extract": choice_q}, model=model)
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
        return {
            "answer": span.text,
            "start": span.start,
            "end": span.end,
            "confidence": float(confidence) if confidence is not None else None,
            "probabilities": probabilities,
            "model": getattr(response, "model", None),
            "raw": response if include_raw else None,
            "mode": "sentence",
        }

    def extract_chunk(
        self,
        *,
        paragraph: str,
        question: str,
        model: str | None = None,
        include_raw: bool = False,
    ) -> ExtractResult:
        """Chunk-only extract: Choice over sentence/clause chunks; span = whole chunk.

        Returns a token-native :class:`ExtractResult` whose start/end cover every
        token overlapping the winning chunk (useful for mode comparisons).
        """
        spans = self._prepare_chunks(paragraph)
        tokens, char_spans = tokenize_with_spans(paragraph)
        if not tokens:
            raise NoCandidatesError("Paragraph produced no tokens.")
        state = build_state(paragraph, question)
        chunk, chunk_conf, chunk_probs, chunk_resp = self._choose_chunk(
            paragraph=paragraph,
            question=question,
            spans=spans,
            state=state,
            model=model,
        )
        start_pos, end_pos = char_span_to_token_span(
            char_spans, chunk.start, chunk.end
        )
        start_ref = TokenRef(pos=start_pos, word=tokens[start_pos])
        end_ref = TokenRef(pos=end_pos, word=tokens[end_pos])
        validate_token_ref(tokens, start_ref)
        validate_token_ref(tokens, end_ref)
        answer, char_start, char_end = answer_from_token_span(
            paragraph, tokens, char_spans, start_pos, end_pos
        )
        raw: Any = None
        if include_raw:
            raw = {"chunk": chunk_resp, "chunk_key": chunk.key, "chunk_text": chunk.text}
        return ExtractResult(
            start=start_ref,
            end=end_ref,
            tokens=list(tokens),
            answer=answer,
            char_start=char_start,
            char_end=char_end,
            confidence=chunk_conf,
            start_confidence=chunk_conf,
            end_confidence=chunk_conf,
            probabilities=chunk_probs,
            start_probabilities=None,
            model=getattr(chunk_resp, "model", None) if chunk_resp is not None else self.model,
            raw=raw,
        )

    def extract_cascade(
        self,
        *,
        paragraph: str,
        question: str,
        model: str | None = None,
        include_raw: bool = False,
        context_radius: int | None = None,
    ) -> ExtractResult:
        """Cascade: sentence/clause Choice → token start/end inside the winner only.

        1. Propose chunks (sentence split; clause fallback for long sentences).
        2. Choice picks the best chunk.
        3. Sequential start/end Choice restricted to tokens overlapping that chunk.
        """
        radius = self.context_radius if context_radius is None else context_radius
        spans = self._prepare_chunks(paragraph)
        tokens, char_spans = tokenize_with_spans(paragraph)
        if not tokens:
            raise NoCandidatesError("Paragraph produced no tokens.")
        state = build_state(paragraph, question)
        chunk, chunk_conf, chunk_probs, chunk_resp = self._choose_chunk(
            paragraph=paragraph,
            question=question,
            spans=spans,
            state=state,
            model=model,
        )
        lo, hi_inclusive = char_span_to_token_span(char_spans, chunk.start, chunk.end)
        hi = hi_inclusive + 1  # half-open for Choice windows

        # Start Choice inside winning chunk only.
        start_choice = build_start_choice(
            tokens, question, lo=lo, hi=hi, context_radius=radius
        )
        start_resp = self._call_system_one(state, {"start": start_choice}, model=model)
        start_ans = _get_choice_answer(start_resp, "start")
        start_pos = _parse_pos_key(start_ans.choice, lo=lo, hi=hi)
        start_ref = TokenRef(pos=start_pos, word=tokens[start_pos])
        validate_token_ref(tokens, start_ref)
        start_conf = getattr(start_ans, "confidence", None)
        start_probs = getattr(start_ans, "probabilities", None)
        if start_probs is not None and not isinstance(start_probs, dict):
            start_probs = dict(start_probs)

        # End Choice: pos in [start, chunk_end].
        end_choice = build_end_choice(
            tokens,
            question,
            start_pos=start_pos,
            hi=hi,
            start_word=tokens[start_pos],
            context_radius=radius,
        )
        end_resp = self._call_system_one(state, {"end": end_choice}, model=model)
        end_ans = _get_choice_answer(end_resp, "end")
        end_pos = _parse_pos_key(end_ans.choice, lo=start_pos, hi=hi)
        end_ref = TokenRef(pos=end_pos, word=tokens[end_pos])
        validate_token_ref(tokens, end_ref)

        answer, char_start, char_end = answer_from_token_span(
            paragraph, tokens, char_spans, start_pos, end_pos
        )
        end_conf = getattr(end_ans, "confidence", None)
        end_probs = getattr(end_ans, "probabilities", None)
        if end_probs is not None and not isinstance(end_probs, dict):
            end_probs = dict(end_probs)

        joint_conf: float | None = None
        parts = [c for c in (chunk_conf, start_conf, end_conf) if c is not None]
        if parts:
            joint = 1.0
            for c in parts:
                joint *= float(c)
            joint_conf = joint

        raw: Any = None
        if include_raw:
            raw = {
                "chunk": chunk_resp,
                "chunk_key": chunk.key,
                "chunk_text": chunk.text,
                "start": start_resp,
                "end": end_resp,
            }
        resp_model = (
            getattr(end_resp, "model", None)
            or getattr(start_resp, "model", None)
            or (getattr(chunk_resp, "model", None) if chunk_resp is not None else None)
        )
        return ExtractResult(
            start=start_ref,
            end=end_ref,
            tokens=list(tokens),
            answer=answer,
            char_start=char_start,
            char_end=char_end,
            confidence=joint_conf,
            start_confidence=float(start_conf) if start_conf is not None else None,
            end_confidence=float(end_conf) if end_conf is not None else None,
            probabilities=end_probs,
            start_probabilities=start_probs,
            model=resp_model,
            raw=raw,
        )


    def extract_span_choice(
        self,
        *,
        paragraph: str,
        question: str,
        model: str | None = None,
        include_raw: bool = False,
        context_radius: int | None = None,
        prefer_k: int = 13,
        propose: str = "smart",
    ) -> ExtractResult:
        """Span-as-Choice over token windows.

        ``propose="smart"`` (default) shortlists windows under 255 with truncated
        criteria. ``propose="all"`` is the classic expensive full 1..K path.
        ``propose="in_chunk"`` picks a sentence chunk first, then spans inside it.
        """
        from jev_extract.strategies import run_span_choice

        return run_span_choice(
            self,
            paragraph=paragraph,
            question=question,
            model=model,
            include_raw=include_raw,
            context_radius=context_radius,
            prefer_k=prefer_k,
            propose=propose,
        )

    def extract_anchor_expand(
        self,
        *,
        paragraph: str,
        question: str,
        model: str | None = None,
        include_raw: bool = False,
        context_radius: int | None = None,
        max_expand: int = 12,
    ) -> ExtractResult:
        """Anchor-then-expand: head Choice, then left/right extent Choices."""
        from jev_extract.strategies import run_anchor_expand

        return run_anchor_expand(
            self,
            paragraph=paragraph,
            question=question,
            model=model,
            include_raw=include_raw,
            context_radius=context_radius,
            max_expand=max_expand,
        )

    def extract_shape_gate(
        self,
        *,
        paragraph: str,
        question: str,
        model: str | None = None,
        include_raw: bool = False,
        context_radius: int | None = None,
    ) -> ExtractResult:
        """Answer-shape gate then sequential start/end with shape-specific end prior."""
        from jev_extract.strategies import run_shape_gate

        return run_shape_gate(
            self,
            paragraph=paragraph,
            question=question,
            model=model,
            include_raw=include_raw,
            context_radius=context_radius,
        )

    def extract_topk_noul(
        self,
        *,
        paragraph: str,
        question: str,
        model: str | None = None,
        include_raw: bool = False,
        context_radius: int | None = None,
        top_k: int = 3,
    ) -> ExtractResult:
        """Top-k end candidates from sequential + batched Noul verify."""
        from jev_extract.strategies import run_topk_noul

        return run_topk_noul(
            self,
            paragraph=paragraph,
            question=question,
            model=model,
            include_raw=include_raw,
            context_radius=context_radius,
            top_k=top_k,
        )

    def extract_ambiguous_joint(
        self,
        *,
        paragraph: str,
        question: str,
        model: str | None = None,
        include_raw: bool = False,
        context_radius: int | None = None,
        margin_threshold: float = 0.15,
    ) -> ExtractResult:
        """Sequential end, or joint (start,end) when start is ambiguous."""
        from jev_extract.strategies import run_ambiguous_joint

        return run_ambiguous_joint(
            self,
            paragraph=paragraph,
            question=question,
            model=model,
            include_raw=include_raw,
            context_radius=context_radius,
            margin_threshold=margin_threshold,
        )

    def extract_length_rerank(
        self,
        *,
        paragraph: str,
        question: str,
        model: str | None = None,
        include_raw: bool = False,
        context_radius: int | None = None,
        top_k: int = 3,
        alpha: float = 0.5,
        shape_aware: bool = True,
    ) -> ExtractResult:
        """Re-rank top-k ends by ``prob * length_penalty`` (shape-aware optional)."""
        from jev_extract.strategies import run_length_rerank

        return run_length_rerank(
            self,
            paragraph=paragraph,
            question=question,
            model=model,
            include_raw=include_raw,
            context_radius=context_radius,
            top_k=top_k,
            alpha=alpha,
            shape_aware=shape_aware,
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

        - ``span``: token-native sequential start/end Choice (same as :meth:`extract`).
        - ``noul``: yes/no probability; optional criteria ``{"true": ..., "false": ...}``.
        - ``choice``: requires ``criteria`` dict of label → description.
        - ``score``: requires ``criteria`` list of ordered level strings.
        """
        if not fields:
            raise ValueError("fields must be a non-empty mapping")

        specs = {name: normalize_field_spec(spec) for name, spec in fields.items()}
        results: dict[str, FieldResult] = {}

        # Span fields: each uses token-native extract (2 Choice calls per field).
        for name, spec in specs.items():
            if spec.mode != "span":
                continue
            q = spec.question or f"Extract field '{name}' from the paragraph."
            er = self.extract(
                paragraph=paragraph,
                question=q,
                model=model,
                include_raw=include_raw,
            )
            results[name] = FieldResult(
                mode="span",
                value=er.answer,
                confidence=er.confidence,
                probabilities=er.probabilities,
                model=er.model,
                start=er.start,
                end=er.end,
                tokens=er.tokens,
                char_start=er.char_start,
                char_end=er.char_end,
                raw=er.raw if include_raw else None,
            )

        non_span = {n: s for n, s in specs.items() if s.mode != "span"}
        if not non_span:
            return results

        questions: dict[str, Any] = {}
        for name, spec in non_span.items():
            try:
                questions[name] = build_field_question(name, spec, [])
            except ValueError as exc:
                raise ExtractionError(str(exc)) from exc

        state = build_state(paragraph)
        response = self._call_system_one(state, questions, model=model)
        resp_model = getattr(response, "model", None)

        for name, spec in non_span.items():
            results[name] = _field_result_from_response(
                response,
                name=name,
                spec=spec,
                model=resp_model,
                include_raw=include_raw,
            )
        return results


def _parse_pos_key(choice: Any, *, lo: int, hi: int) -> int:
    """Parse a Choice answer into an int position within ``[lo, hi)``."""
    key = str(choice).strip()
    # Allow "4:1843" style answers as well as bare "4".
    if ":" in key:
        key = key.split(":", 1)[0].strip()
    try:
        pos = int(key)
    except ValueError as exc:
        raise ExtractionError(
            f"Model returned non-integer token position {choice!r}."
        ) from exc
    if pos < lo or pos >= hi:
        raise ExtractionError(
            f"Model returned token pos {pos} outside allowed range [{lo}, {hi})."
        )
    return pos


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
    model: str | None,
    include_raw: bool,
) -> FieldResult:
    ans = _get_answer(response, name)
    raw = ans if include_raw else None

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
