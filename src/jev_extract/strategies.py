"""Alternate extractive strategies for TypeSafe Jev Choice / Noul / Score.

Each ``run_*`` function takes an :class:`~jev_extract.client.Extractor` and
returns an :class:`~jev_extract.schemas.ExtractResult`. Public wrappers live on
``Extractor`` (``extract_mode=...`` or dedicated methods).
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from jev_extract.errors import ExtractionError, NoCandidatesError
from jev_extract.questions import (
    DEFAULT_CONTEXT_RADIUS,
    SHAPE_END_INSTRUCTIONS,
    build_end_choice,
    build_extent_choice,
    build_joint_span_choice,
    build_noul_verify,
    build_shape_gate_choice,
    build_span_as_choice,
    build_start_choice,
    build_state,
    choose_span_max_len,
)
from jev_extract.schemas import ExtractResult, TokenRef
from jev_extract.tokenize import (
    answer_from_token_span,
    token_windows,
    tokenize_with_spans,
    validate_token_ref,
)

# Re-export helpers used by tests / benches
__all__ = [
    "EXTRACT_MODES",
    "run_span_choice",
    "run_anchor_expand",
    "run_shape_gate",
    "run_topk_noul",
    "run_ambiguous_joint",
    "run_length_rerank",
    "length_penalty_score",
    "parse_span_key",
    "topk_from_probs",
]

EXTRACT_MODES = (
    "sequential",
    "joint",
    "auto",
    "span_choice",
    "span_choice_smart",
    "span_choice_in_chunk",
    "anchor_expand",
    "shape_gate",
    "topk_noul",
    "ambiguous_joint",
    "length_rerank",
    "chunk_only",
    "cascade",
)

DEFAULT_AMBIGUOUS_MARGIN = 0.15
DEFAULT_LENGTH_PENALTY_ALPHA = 0.5
DEFAULT_TOPK = 3
DEFAULT_MAX_EXPAND = 12


def parse_span_key(key: Any, *, n_tokens: int) -> tuple[int, int]:
    """Parse ``'start:end'`` into inclusive positions."""
    text = str(key).strip()
    if ":" not in text:
        raise ExtractionError(f"Expected 'start:end' span key, got {key!r}.")
    s_str, e_str = text.split(":", 1)
    try:
        start = int(s_str)
        end = int(e_str)
    except ValueError as exc:
        raise ExtractionError(f"Non-integer span key {key!r}.") from exc
    if not (0 <= start <= end < n_tokens):
        raise ExtractionError(
            f"Span [{start}, {end}] out of range for {n_tokens} tokens."
        )
    return start, end


def topk_from_probs(
    probs: Mapping[str, float] | None,
    *,
    k: int,
    lo: int,
    hi: int,
    fallback: int,
) -> list[tuple[int, float]]:
    """Return up to *k* (pos, prob) from Choice probabilities within [lo, hi)."""
    ranked: list[tuple[int, float]] = []
    if probs:
        for key, val in probs.items():
            raw = str(key).strip()
            if ":" in raw:
                raw = raw.split(":", 1)[0].strip()
            try:
                pos = int(raw)
            except ValueError:
                continue
            if lo <= pos < hi:
                ranked.append((pos, float(val)))
        ranked.sort(key=lambda kv: kv[1], reverse=True)
    out: list[tuple[int, float]] = []
    seen: set[int] = set()
    for pos, p in ranked:
        if pos in seen:
            continue
        seen.add(pos)
        out.append((pos, p))
        if len(out) >= k:
            return out
    if fallback not in seen and lo <= fallback < hi:
        out.append((fallback, ranked[0][1] if ranked else 1.0))
    if not out and lo <= fallback < hi:
        out.append((fallback, 1.0))
    return out[:k]


def length_penalty_score(
    prob: float,
    length: int,
    *,
    alpha: float = DEFAULT_LENGTH_PENALTY_ALPHA,
    prefer_longer: bool = False,
) -> float:
    """Score = prob × length^±alpha (shorter preferred by default)."""
    length = max(1, int(length))
    if prefer_longer:
        return float(prob) * (float(length) ** float(alpha))
    return float(prob) / (float(length) ** float(alpha))


def _parse_pos_key(choice: Any, *, lo: int, hi: int) -> int:
    from jev_extract.client import _parse_pos_key as _pp

    return _pp(choice, lo=lo, hi=hi)


def _get_choice_answer(response: Any, name: str) -> Any:
    from jev_extract.client import _get_choice_answer as _gc

    return _gc(response, name)


def _get_answer(response: Any, name: str) -> Any:
    from jev_extract.client import _get_answer as _ga

    return _ga(response, name)


def _result(
    *,
    paragraph: str,
    tokens: list[str],
    char_spans: list[tuple[int, int]],
    start_pos: int,
    end_pos: int,
    start_conf: float | None = None,
    end_conf: float | None = None,
    confidence: float | None = None,
    probabilities: dict[str, float] | None = None,
    start_probabilities: dict[str, float] | None = None,
    model: str | None = None,
    raw: Any = None,
) -> ExtractResult:
    start_ref = TokenRef(pos=start_pos, word=tokens[start_pos])
    end_ref = TokenRef(pos=end_pos, word=tokens[end_pos])
    validate_token_ref(tokens, start_ref)
    validate_token_ref(tokens, end_ref)
    answer, char_start, char_end = answer_from_token_span(
        paragraph, tokens, char_spans, start_pos, end_pos
    )
    if confidence is None:
        if start_conf is not None and end_conf is not None:
            confidence = float(start_conf) * float(end_conf)
        elif end_conf is not None:
            confidence = float(end_conf)
        elif start_conf is not None:
            confidence = float(start_conf)
    return ExtractResult(
        start=start_ref,
        end=end_ref,
        tokens=list(tokens),
        answer=answer,
        char_start=char_start,
        char_end=char_end,
        confidence=confidence,
        start_confidence=float(start_conf) if start_conf is not None else None,
        end_confidence=float(end_conf) if end_conf is not None else None,
        probabilities=probabilities,
        start_probabilities=start_probabilities,
        model=model,
        raw=raw,
    )


def _tokenize_or_raise(paragraph: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens, char_spans = tokenize_with_spans(paragraph)
    if not tokens:
        raise NoCandidatesError("Paragraph produced no tokens.")
    return tokens, char_spans


# ---------------------------------------------------------------------------
# 1. Span-as-Choice
# ---------------------------------------------------------------------------


def run_span_choice(
    extractor: Any,
    *,
    paragraph: str,
    question: str,
    model: str | None = None,
    include_raw: bool = False,
    context_radius: int | None = None,
    prefer_k: int = 13,
    propose: str = "smart",
) -> ExtractResult:
    """Single Choice over contiguous token windows.

    ``propose``:
      * ``"all"`` — classic full 1..K enumeration (expensive input tokens).
      * ``"smart"`` — default: length-hist prior (K≤8), sentence candidates,
        keyword shortlist ≤96, truncated criteria (cheap).
      * ``"in_chunk"`` — chunk Choice first, then dense span Choice inside winner.
    """
    from jev_extract.questions import (
        SMART_PREFER_K,
        propose_span_windows,
    )
    from jev_extract.tokenize import char_span_to_token_span

    if propose not in ("all", "smart", "in_chunk"):
        raise ValueError(
            f"propose must be 'all'|'smart'|'in_chunk'; got {propose!r}"
        )

    tokens, char_spans = _tokenize_or_raise(paragraph)
    radius = (
        extractor.context_radius if context_radius is None else context_radius
    )
    # Slightly smaller context for span descriptions (span itself is marked).
    # Smart/in_chunk use radius 1 to cut input tokens further.
    if propose in ("smart", "in_chunk"):
        desc_radius = 1
    else:
        desc_radius = max(1, min(2, radius))
    state = build_state(paragraph, question)

    chunk_resp: Any = None
    chunk_key: str | None = None
    lo, hi = 0, len(tokens)

    if propose == "in_chunk":
        spans = extractor._prepare_chunks(paragraph)
        chunk, _chunk_conf, _chunk_probs, chunk_resp = extractor._choose_chunk(
            paragraph=paragraph,
            question=question,
            spans=spans,
            state=state,
            model=model,
        )
        chunk_key = chunk.key
        lo, hi_inclusive = char_span_to_token_span(
            char_spans, chunk.start, chunk.end
        )
        hi = hi_inclusive + 1

    if propose == "all":
        k = choose_span_max_len(
            hi - lo, prefer_k=prefer_k, max_criteria=extractor.max_candidates
        )
        windows = propose_span_windows(
            tokens,
            propose="all",
            max_len=k,
            prefer_k=prefer_k,
            max_criteria=extractor.max_candidates,
            lo=lo,
            hi=hi,
        )
        build_propose = "all"
    else:
        k = min(SMART_PREFER_K, hi - lo)
        windows = propose_span_windows(
            tokens,
            propose=propose,  # smart or in_chunk
            question=question,
            paragraph=paragraph,
            char_spans=char_spans,
            max_len=k,
            prefer_k=prefer_k,
            max_criteria=extractor.max_candidates,
            lo=lo,
            hi=hi,
        )
        build_propose = propose

    choice_q = build_span_as_choice(
        tokens,
        question,
        windows=windows,
        propose=build_propose,
        max_len=k,
        prefer_k=prefer_k,
        max_criteria=extractor.max_candidates,
        context_radius=desc_radius,
        paragraph=paragraph,
        char_spans=char_spans,
        lo=lo,
        hi=hi,
    )
    resp = extractor._call_system_one(state, {"span": choice_q}, model=model)
    ans = _get_choice_answer(resp, "span")
    start_pos, end_pos = parse_span_key(ans.choice, n_tokens=len(tokens))
    conf = getattr(ans, "confidence", None)
    probs = getattr(ans, "probabilities", None)
    if probs is not None and not isinstance(probs, dict):
        probs = dict(probs)
    raw = None
    if include_raw:
        raw = {
            "span": resp,
            "max_len": k,
            "propose": propose,
            "n_windows": len(windows),
            "lo": lo,
            "hi": hi,
        }
        if chunk_resp is not None or chunk_key is not None:
            raw["chunk"] = chunk_resp
            raw["chunk_key"] = chunk_key
    return _result(
        paragraph=paragraph,
        tokens=tokens,
        char_spans=char_spans,
        start_pos=start_pos,
        end_pos=end_pos,
        start_conf=float(conf) if conf is not None else None,
        end_conf=float(conf) if conf is not None else None,
        confidence=float(conf) if conf is not None else None,
        probabilities=probs,
        model=getattr(resp, "model", None),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# 2. Anchor then expand
# ---------------------------------------------------------------------------


def run_anchor_expand(
    extractor: Any,
    *,
    paragraph: str,
    question: str,
    model: str | None = None,
    include_raw: bool = False,
    context_radius: int | None = None,
    max_expand: int = DEFAULT_MAX_EXPAND,
) -> ExtractResult:
    """Choice picks answer-head; then left/right extent Choices; reconstruct span."""
    tokens, char_spans = _tokenize_or_raise(paragraph)
    radius = (
        extractor.context_radius if context_radius is None else context_radius
    )
    n = len(tokens)
    windows = token_windows(
        n,
        max_size=extractor.max_candidates,
        overlap=min(64, max(0, extractor.max_candidates - 1)),
    )
    state = build_state(paragraph, question)

    # Head = start Choice with anchor instructions.
    from jev_extract.questions import DEFAULT_ANCHOR_INSTRUCTIONS

    best_pos: int | None = None
    best_conf = float("-inf")
    best_probs: dict[str, float] | None = None
    head_resp: Any = None
    for lo, hi in windows:
        choice_q = build_start_choice(
            tokens,
            question,
            lo=lo,
            hi=hi,
            context_radius=radius,
            instructions=(
                f"{DEFAULT_ANCHOR_INSTRUCTIONS}\n\nQuestion: {question}"
            ),
        )
        resp = extractor._call_system_one(state, {"head": choice_q}, model=model)
        ans = _get_choice_answer(resp, "head")
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
            head_resp = resp
    assert best_pos is not None
    head = best_pos
    head_conf: float | None = None if best_conf == float("-inf") else best_conf
    if len(windows) == 1:
        raw_c = getattr(_get_choice_answer(head_resp, "head"), "confidence", None)
        head_conf = float(raw_c) if raw_c is not None else None

    max_left = min(max_expand, head)
    max_right = min(max_expand, n - 1 - head)
    left_q = build_extent_choice(
        side="left",
        max_extent=max_left,
        head_pos=head,
        head_word=tokens[head],
        question=question,
    )
    right_q = build_extent_choice(
        side="right",
        max_extent=max_right,
        head_pos=head,
        head_word=tokens[head],
        question=question,
    )
    # Batched left+right in one system_one when both needed.
    extent_questions: dict[str, Any] = {}
    if max_left > 0:
        extent_questions["left"] = left_q
    if max_right > 0:
        extent_questions["right"] = right_q

    left_ext = 0
    right_ext = 0
    left_conf: float | None = None
    right_conf: float | None = None
    extent_resp: Any = None
    if extent_questions:
        extent_resp = extractor._call_system_one(
            state, extent_questions, model=model
        )
        if "left" in extent_questions:
            lans = _get_choice_answer(extent_resp, "left")
            left_ext = _parse_pos_key(lans.choice, lo=0, hi=max_left + 1)
            left_conf = getattr(lans, "confidence", None)
        if "right" in extent_questions:
            rans = _get_choice_answer(extent_resp, "right")
            right_ext = _parse_pos_key(rans.choice, lo=0, hi=max_right + 1)
            right_conf = getattr(rans, "confidence", None)

    start_pos = head - left_ext
    end_pos = head + right_ext
    end_conf = None
    if left_conf is not None and right_conf is not None:
        end_conf = float(left_conf) * float(right_conf)
    elif right_conf is not None:
        end_conf = float(right_conf)
    elif left_conf is not None:
        end_conf = float(left_conf)

    raw = None
    if include_raw:
        raw = {
            "head": head_resp,
            "extent": extent_resp,
            "head_pos": head,
            "left_ext": left_ext,
            "right_ext": right_ext,
        }
    return _result(
        paragraph=paragraph,
        tokens=tokens,
        char_spans=char_spans,
        start_pos=start_pos,
        end_pos=end_pos,
        start_conf=head_conf,
        end_conf=float(end_conf) if end_conf is not None else None,
        start_probabilities=best_probs,
        model=getattr(head_resp, "model", None)
        or (getattr(extent_resp, "model", None) if extent_resp else None),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# 3. Answer-shape gate
# ---------------------------------------------------------------------------


def run_shape_gate(
    extractor: Any,
    *,
    paragraph: str,
    question: str,
    model: str | None = None,
    include_raw: bool = False,
    context_radius: int | None = None,
) -> ExtractResult:
    """Choice among answer shapes; route to shape-specific end instructions."""
    tokens, char_spans = _tokenize_or_raise(paragraph)
    radius = (
        extractor.context_radius if context_radius is None else context_radius
    )
    state = build_state(paragraph, question)
    shape_q = build_shape_gate_choice(question)
    shape_resp = extractor._call_system_one(
        state, {"shape": shape_q}, model=model
    )
    shape_ans = _get_choice_answer(shape_resp, "shape")
    shape = str(shape_ans.choice).strip()
    if shape not in SHAPE_END_INSTRUCTIONS:
        # Unknown label → fall back to short_entity prior
        shape = "short_entity"
    shape_conf = getattr(shape_ans, "confidence", None)

    windows = token_windows(
        len(tokens),
        max_size=extractor.max_candidates,
        overlap=min(64, max(0, extractor.max_candidates - 1)),
    )
    start_pos, start_conf, start_probs, start_resp = extractor._choose_start(
        tokens=tokens,
        question=question,
        state=state,
        windows=windows,
        model=model,
        context_radius=radius,
    )
    end_hi = min(start_pos + extractor.max_candidates, len(tokens))
    end_instr = SHAPE_END_INSTRUCTIONS.get(shape)
    end_choice = build_end_choice(
        tokens,
        question,
        start_pos=start_pos,
        hi=end_hi,
        start_word=tokens[start_pos],
        context_radius=radius,
        instructions=end_instr,
    )
    end_resp = extractor._call_system_one(
        state, {"end": end_choice}, model=model
    )
    end_ans = _get_choice_answer(end_resp, "end")
    end_pos = _parse_pos_key(end_ans.choice, lo=start_pos, hi=end_hi)
    end_conf = getattr(end_ans, "confidence", None)
    end_probs = getattr(end_ans, "probabilities", None)
    if end_probs is not None and not isinstance(end_probs, dict):
        end_probs = dict(end_probs)

    raw = None
    if include_raw:
        raw = {
            "shape": shape_resp,
            "shape_label": shape,
            "start": start_resp,
            "end": end_resp,
        }
    result = _result(
        paragraph=paragraph,
        tokens=tokens,
        char_spans=char_spans,
        start_pos=start_pos,
        end_pos=end_pos,
        start_conf=start_conf,
        end_conf=float(end_conf) if end_conf is not None else None,
        probabilities=end_probs,
        start_probabilities=start_probs,
        model=getattr(end_resp, "model", None)
        or getattr(start_resp, "model", None),
        raw=raw,
    )
    # Attach shape on raw for benches even when include_raw=False via a light touch:
    # callers with include_raw get shape_label; confidence can blend shape conf.
    if shape_conf is not None and result.confidence is not None:
        # Keep joint start*end; shape conf is informational in raw only.
        pass
    return result


# ---------------------------------------------------------------------------
# 4. Top-k + Noul verify
# ---------------------------------------------------------------------------


def run_topk_noul(
    extractor: Any,
    *,
    paragraph: str,
    question: str,
    model: str | None = None,
    include_raw: bool = False,
    context_radius: int | None = None,
    top_k: int = DEFAULT_TOPK,
) -> ExtractResult:
    """Sequential start+end; verify top-k end candidates with batched Noul."""
    top_k = max(1, min(int(top_k), 3))
    tokens, char_spans = _tokenize_or_raise(paragraph)
    radius = (
        extractor.context_radius if context_radius is None else context_radius
    )
    state = build_state(paragraph, question)
    windows = token_windows(
        len(tokens),
        max_size=extractor.max_candidates,
        overlap=min(64, max(0, extractor.max_candidates - 1)),
    )
    start_pos, start_conf, start_probs, start_resp = extractor._choose_start(
        tokens=tokens,
        question=question,
        state=state,
        windows=windows,
        model=model,
        context_radius=radius,
    )
    end_hi = min(start_pos + extractor.max_candidates, len(tokens))
    end_choice = build_end_choice(
        tokens,
        question,
        start_pos=start_pos,
        hi=end_hi,
        start_word=tokens[start_pos],
        context_radius=radius,
    )
    end_resp = extractor._call_system_one(
        state, {"end": end_choice}, model=model
    )
    end_ans = _get_choice_answer(end_resp, "end")
    end_pos_default = _parse_pos_key(end_ans.choice, lo=start_pos, hi=end_hi)
    end_conf = getattr(end_ans, "confidence", None)
    end_probs = getattr(end_ans, "probabilities", None)
    if end_probs is not None and not isinstance(end_probs, dict):
        end_probs = dict(end_probs)

    candidates = topk_from_probs(
        end_probs,
        k=top_k,
        lo=start_pos,
        hi=end_hi,
        fallback=end_pos_default,
    )
    # Ensure the model's top choice is included.
    if not any(p == end_pos_default for p, _ in candidates):
        candidates = [(end_pos_default, float(end_conf or 1.0))] + candidates
        candidates = candidates[:top_k]

    # Build span texts and batched Nouls.
    span_infos: list[tuple[int, float, str]] = []
    for epos, eprob in candidates:
        text, _, _ = answer_from_token_span(
            paragraph, tokens, char_spans, start_pos, epos
        )
        span_infos.append((epos, eprob, text))

    noul_questions = {
        f"v{i}": build_noul_verify(span_text=text, question=question)
        for i, (_, _, text) in enumerate(span_infos)
    }
    noul_resp = extractor._call_system_one(state, noul_questions, model=model)

    best_i = 0
    best_score = float("-inf")
    noul_scores: list[float] = []
    for i, (epos, eprob, _) in enumerate(span_infos):
        ans = _get_answer(noul_resp, f"v{i}")
        noul_val = float(getattr(ans, "noul", 0.0) or 0.0)
        noul_scores.append(noul_val)
        # Prefer high noul; break ties with end probability (and shorter length).
        length = epos - start_pos + 1
        score = noul_val * 10.0 + eprob - 0.01 * length
        if score > best_score:
            best_score = score
            best_i = i

    # If nothing passes a soft threshold, keep the original end choice.
    PASS = 0.5
    if noul_scores and max(noul_scores) < PASS:
        chosen_end = end_pos_default
        chosen_noul = noul_scores[
            next(
                (
                    i
                    for i, (e, _) in enumerate(candidates)
                    if e == end_pos_default
                ),
                0,
            )
        ]
    else:
        chosen_end = span_infos[best_i][0]
        chosen_noul = noul_scores[best_i]

    raw = None
    if include_raw:
        raw = {
            "start": start_resp,
            "end": end_resp,
            "noul": noul_resp,
            "candidates": [
                {
                    "end": e,
                    "prob": p,
                    "text": t,
                    "noul": noul_scores[i],
                }
                for i, (e, p, t) in enumerate(span_infos)
            ],
            "chosen_noul": chosen_noul,
        }
    return _result(
        paragraph=paragraph,
        tokens=tokens,
        char_spans=char_spans,
        start_pos=start_pos,
        end_pos=chosen_end,
        start_conf=start_conf,
        end_conf=float(end_conf) if end_conf is not None else None,
        confidence=(
            (float(start_conf) if start_conf is not None else 1.0)
            * float(chosen_noul)
        ),
        probabilities=end_probs,
        start_probabilities=start_probs,
        model=getattr(noul_resp, "model", None)
        or getattr(end_resp, "model", None),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# 5. Ambiguous → joint
# ---------------------------------------------------------------------------


def run_ambiguous_joint(
    extractor: Any,
    *,
    paragraph: str,
    question: str,
    model: str | None = None,
    include_raw: bool = False,
    context_radius: int | None = None,
    margin_threshold: float = DEFAULT_AMBIGUOUS_MARGIN,
) -> ExtractResult:
    """Start Choice; if duplicate word or low top-2 margin → joint span Choice."""
    tokens, char_spans = _tokenize_or_raise(paragraph)
    radius = (
        extractor.context_radius if context_radius is None else context_radius
    )
    state = build_state(paragraph, question)
    windows = token_windows(
        len(tokens),
        max_size=extractor.max_candidates,
        overlap=min(64, max(0, extractor.max_candidates - 1)),
    )
    start_pos, start_conf, start_probs, start_resp = extractor._choose_start(
        tokens=tokens,
        question=question,
        state=state,
        windows=windows,
        model=model,
        context_radius=radius,
    )

    counts = Counter(tokens)
    start_word = tokens[start_pos]
    ambiguous = counts[start_word] > 1
    margin: float | None = None
    if start_probs and len(start_probs) >= 2:
        ranked_p = sorted((float(v) for v in start_probs.values()), reverse=True)
        margin = ranked_p[0] - ranked_p[1]
        if margin < margin_threshold:
            ambiguous = True
    # Also treat missing/low confidence as ambiguous when duplicates exist.
    if counts[start_word] > 1 and (
        start_conf is None or float(start_conf) < 0.55
    ):
        ambiguous = True

    if not ambiguous:
        # Sequential end (same as token_sequential).
        end_hi = min(start_pos + extractor.max_candidates, len(tokens))
        end_choice = build_end_choice(
            tokens,
            question,
            start_pos=start_pos,
            hi=end_hi,
            start_word=start_word,
            context_radius=radius,
        )
        end_resp = extractor._call_system_one(
            state, {"end": end_choice}, model=model
        )
        end_ans = _get_choice_answer(end_resp, "end")
        end_pos = _parse_pos_key(end_ans.choice, lo=start_pos, hi=end_hi)
        end_conf = getattr(end_ans, "confidence", None)
        end_probs = getattr(end_ans, "probabilities", None)
        if end_probs is not None and not isinstance(end_probs, dict):
            end_probs = dict(end_probs)
        raw = None
        if include_raw:
            raw = {
                "start": start_resp,
                "end": end_resp,
                "path": "sequential",
                "margin": margin,
                "duplicate": counts[start_word] > 1,
            }
        return _result(
            paragraph=paragraph,
            tokens=tokens,
            char_spans=char_spans,
            start_pos=start_pos,
            end_pos=end_pos,
            start_conf=start_conf,
            end_conf=float(end_conf) if end_conf is not None else None,
            probabilities=end_probs,
            start_probabilities=start_probs,
            model=getattr(end_resp, "model", None)
            or getattr(start_resp, "model", None),
            raw=raw,
        )

    # Joint path: top starts × ends.
    starts: list[int] = [start_pos]
    if start_probs:
        ranked = sorted(
            (
                (
                    int(k.split(":", 1)[0]) if ":" in str(k) else int(k),
                    float(v),
                )
                for k, v in start_probs.items()
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
        for pos, _ in ranked:
            if 0 <= pos < len(tokens) and pos not in starts:
                starts.append(pos)
            if len(starts) >= 3:
                break

    end_hi = len(tokens)
    n_pairs = sum(max(0, end_hi - s) for s in starts)
    if n_pairs > extractor.max_candidates or n_pairs == 0:
        # Fall back to sequential end.
        end_hi_seq = min(start_pos + extractor.max_candidates, len(tokens))
        end_choice = build_end_choice(
            tokens,
            question,
            start_pos=start_pos,
            hi=end_hi_seq,
            start_word=start_word,
            context_radius=radius,
        )
        end_resp = extractor._call_system_one(
            state, {"end": end_choice}, model=model
        )
        end_ans = _get_choice_answer(end_resp, "end")
        end_pos = _parse_pos_key(end_ans.choice, lo=start_pos, hi=end_hi_seq)
        end_conf = getattr(end_ans, "confidence", None)
        end_probs = getattr(end_ans, "probabilities", None)
        if end_probs is not None and not isinstance(end_probs, dict):
            end_probs = dict(end_probs)
        raw = None
        if include_raw:
            raw = {
                "start": start_resp,
                "end": end_resp,
                "path": "sequential_fallback",
                "margin": margin,
            }
        return _result(
            paragraph=paragraph,
            tokens=tokens,
            char_spans=char_spans,
            start_pos=start_pos,
            end_pos=end_pos,
            start_conf=start_conf,
            end_conf=float(end_conf) if end_conf is not None else None,
            probabilities=end_probs,
            start_probabilities=start_probs,
            model=getattr(end_resp, "model", None),
            raw=raw,
        )

    joint_q = build_joint_span_choice(
        tokens,
        question,
        start_candidates=starts,
        end_hi=end_hi,
        context_radius=radius,
        max_pairs=extractor.max_candidates,
    )
    span_resp = extractor._call_system_one(
        state, {"span": joint_q}, model=model
    )
    span_ans = _get_choice_answer(span_resp, "span")
    start_pos, end_pos = parse_span_key(span_ans.choice, n_tokens=len(tokens))
    end_conf = getattr(span_ans, "confidence", None)
    end_probs = getattr(span_ans, "probabilities", None)
    if end_probs is not None and not isinstance(end_probs, dict):
        end_probs = dict(end_probs)
    raw = None
    if include_raw:
        raw = {
            "start": start_resp,
            "span": span_resp,
            "path": "joint",
            "margin": margin,
            "duplicate": counts[start_word] > 1,
            "starts": starts,
        }
    return _result(
        paragraph=paragraph,
        tokens=tokens,
        char_spans=char_spans,
        start_pos=start_pos,
        end_pos=end_pos,
        start_conf=start_conf,
        end_conf=float(end_conf) if end_conf is not None else None,
        probabilities=end_probs,
        start_probabilities=start_probs,
        model=getattr(span_resp, "model", None)
        or getattr(start_resp, "model", None),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# 6. Length-aware re-rank (+ optional shape-aware)
# ---------------------------------------------------------------------------


def run_length_rerank(
    extractor: Any,
    *,
    paragraph: str,
    question: str,
    model: str | None = None,
    include_raw: bool = False,
    context_radius: int | None = None,
    top_k: int = DEFAULT_TOPK,
    alpha: float = DEFAULT_LENGTH_PENALTY_ALPHA,
    shape_aware: bool = True,
) -> ExtractResult:
    """Sequential start+end; re-rank top-k ends by prob × length_penalty."""
    top_k = max(1, min(int(top_k), 8))
    tokens, char_spans = _tokenize_or_raise(paragraph)
    radius = (
        extractor.context_radius if context_radius is None else context_radius
    )
    state = build_state(paragraph, question)

    prefer_longer = False
    shape_label: str | None = None
    shape_resp: Any = None
    if shape_aware:
        shape_q = build_shape_gate_choice(question)
        shape_resp = extractor._call_system_one(
            state, {"shape": shape_q}, model=model
        )
        shape_ans = _get_choice_answer(shape_resp, "shape")
        shape_label = str(shape_ans.choice).strip()
        prefer_longer = shape_label == "full_sentence"

    windows = token_windows(
        len(tokens),
        max_size=extractor.max_candidates,
        overlap=min(64, max(0, extractor.max_candidates - 1)),
    )
    start_pos, start_conf, start_probs, start_resp = extractor._choose_start(
        tokens=tokens,
        question=question,
        state=state,
        windows=windows,
        model=model,
        context_radius=radius,
    )
    end_hi = min(start_pos + extractor.max_candidates, len(tokens))
    end_instr = None
    if shape_label and shape_label in SHAPE_END_INSTRUCTIONS:
        end_instr = SHAPE_END_INSTRUCTIONS[shape_label]
    end_choice = build_end_choice(
        tokens,
        question,
        start_pos=start_pos,
        hi=end_hi,
        start_word=tokens[start_pos],
        context_radius=radius,
        instructions=end_instr,
    )
    end_resp = extractor._call_system_one(
        state, {"end": end_choice}, model=model
    )
    end_ans = _get_choice_answer(end_resp, "end")
    end_pos_default = _parse_pos_key(end_ans.choice, lo=start_pos, hi=end_hi)
    end_conf = getattr(end_ans, "confidence", None)
    end_probs = getattr(end_ans, "probabilities", None)
    if end_probs is not None and not isinstance(end_probs, dict):
        end_probs = dict(end_probs)

    candidates = topk_from_probs(
        end_probs,
        k=top_k,
        lo=start_pos,
        hi=end_hi,
        fallback=end_pos_default,
    )
    if not any(p == end_pos_default for p, _ in candidates):
        candidates = [(end_pos_default, float(end_conf or 1.0))] + list(
            candidates
        )
        candidates = candidates[:top_k]

    best_end = end_pos_default
    best_score = float("-inf")
    scored: list[dict[str, Any]] = []
    for epos, eprob in candidates:
        length = epos - start_pos + 1
        score = length_penalty_score(
            eprob, length, alpha=alpha, prefer_longer=prefer_longer
        )
        scored.append(
            {
                "end": epos,
                "prob": eprob,
                "length": length,
                "score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_end = epos

    raw = None
    if include_raw:
        raw = {
            "shape": shape_resp,
            "shape_label": shape_label,
            "prefer_longer": prefer_longer,
            "start": start_resp,
            "end": end_resp,
            "rerank": scored,
            "alpha": alpha,
            "chosen_end": best_end,
        }
    return _result(
        paragraph=paragraph,
        tokens=tokens,
        char_spans=char_spans,
        start_pos=start_pos,
        end_pos=best_end,
        start_conf=start_conf,
        end_conf=float(end_conf) if end_conf is not None else None,
        probabilities=end_probs,
        start_probabilities=start_probs,
        model=getattr(end_resp, "model", None)
        or getattr(start_resp, "model", None),
        raw=raw,
    )
