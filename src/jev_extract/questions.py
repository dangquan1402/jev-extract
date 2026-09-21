"""Helpers that build TypeSafe System One questions for extraction.

Token Choice criteria use **position keys** (``str(pos)``) as the source of
truth. Criterion *descriptions* are local context windows that mark the
candidate with guillemets (``«word»``) so duplicate surface forms remain
distinguishable, e.g. ``"… published notes in «1843» . Later …"``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from typesafe_sdk import Choice, Noul, Score

from jev_extract.candidates import spans_to_criteria
from jev_extract.schemas import FieldSpec, Span

DEFAULT_CONTEXT_RADIUS = 4
"""Default number of tokens kept on each side of the marked candidate."""

DEFAULT_SPAN_INSTRUCTIONS = (
    "Select the single candidate span that best answers the question. "
    "Prefer the most specific supporting sentence; do not invent text."
)

DEFAULT_START_INSTRUCTIONS = (
    "Select the START token of the answer span. "
    "Each criterion key is the 0-based token position; the criterion text is a "
    "local context window with the candidate marked «word». "
    "Use neighboring tokens to disambiguate duplicates. "
    "Prefer the earliest token of the minimal answer (include a leading "
    "determiner like 'The'/'the' when it is part of the answer). "
    "Return exactly one position key. Do not invent tokens."
)

DEFAULT_END_INSTRUCTIONS = (
    "Select the END token of the answer span (inclusive). "
    "The start token was already chosen (see 'start was pos:word' below). "
    "Each criterion key is a 0-based token position >= start; the criterion "
    "text is a local context window with the candidate marked «word». "
    "Prefer the shortest end that fully answers the question — do not extend "
    "into following modifiers or clauses unless they are required. "
    "Do not latch onto another occurrence of a duplicate word closer to the "
    "wrong start — stay consistent with the chosen start. "
    "Return exactly one position key. Do not invent tokens."
)

DEFAULT_JOINT_INSTRUCTIONS = (
    "Select the single (start, end) token-span that best answers the question. "
    "Each criterion key is 'start_pos:end_pos'; the criterion text shows local "
    "context for both endpoints. Prefer the most specific supporting span; "
    "do not invent tokens."
)


def format_token_context(
    tokens: Sequence[str],
    pos: int,
    *,
    radius: int = DEFAULT_CONTEXT_RADIUS,
) -> str:
    """Local window with marked token, e.g. ``… published notes in «1843» . Later …``.

    Adds leading/trailing ellipsis when the window is truncated at paragraph
    edges. ``radius`` controls how many tokens are kept on each side (default 4).
    """
    n = len(tokens)
    if pos < 0 or pos >= n:
        raise ValueError(f"pos {pos} out of range for {n} tokens")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    lo = max(0, pos - radius)
    hi = min(n, pos + radius + 1)
    parts: list[str] = []
    if lo > 0:
        parts.append("…")
    for i in range(lo, hi):
        w = tokens[i]
        parts.append(f"«{w}»" if i == pos else w)
    if hi < n:
        parts.append("…")
    return " ".join(parts)


def tokens_to_criteria(
    tokens: Sequence[str],
    *,
    lo: int = 0,
    hi: int | None = None,
    context_radius: int = DEFAULT_CONTEXT_RADIUS,
) -> dict[str, str]:
    """Map token positions to Choice criteria ``{pos: local_context_with_«word»}``.

    Keys are ``str(pos)`` only (Choice identity + source of truth). Descriptions
    embed a marked local window so identical words at different positions get
    different criterion strings via their neighbors.
    """
    if hi is None:
        hi = len(tokens)
    if lo < 0 or hi > len(tokens) or lo > hi:
        raise ValueError(f"Invalid token range [{lo}, {hi}) for {len(tokens)} tokens")
    return {
        str(i): format_token_context(tokens, i, radius=context_radius)
        for i in range(lo, hi)
    }


def build_span_choice(
    spans: Sequence[Span],
    question: str,
    *,
    instructions: str | None = None,
) -> Choice:
    """Build a closed-set :class:`~typesafe_sdk.Choice` over sentence candidates."""
    criteria = spans_to_criteria(spans)
    instr = instructions or f"{DEFAULT_SPAN_INSTRUCTIONS}\n\nQuestion: {question}"
    return Choice(instructions=instr, criteria=criteria)


def build_start_choice(
    tokens: Sequence[str],
    question: str,
    *,
    lo: int = 0,
    hi: int | None = None,
    context_radius: int = DEFAULT_CONTEXT_RADIUS,
    instructions: str | None = None,
) -> Choice:
    """Choice over token positions for the span **start** (context-disambiguated)."""
    criteria = tokens_to_criteria(
        tokens, lo=lo, hi=hi, context_radius=context_radius
    )
    instr = instructions or (
        f"{DEFAULT_START_INSTRUCTIONS}\n\nQuestion: {question}"
    )
    return Choice(instructions=instr, criteria=criteria)


def build_end_choice(
    tokens: Sequence[str],
    question: str,
    *,
    start_pos: int,
    hi: int | None = None,
    start_word: str | None = None,
    context_radius: int = DEFAULT_CONTEXT_RADIUS,
    instructions: str | None = None,
) -> Choice:
    """Choice over token positions for the span **end** (``pos >= start_pos``).

    End instructions always mention the chosen start as ``start was pos:word``
    so the model does not latch onto another duplicate closer to a wrong start.
    """
    criteria = tokens_to_criteria(
        tokens, lo=start_pos, hi=hi, context_radius=context_radius
    )
    word = start_word if start_word is not None else tokens[start_pos]
    base = instructions or DEFAULT_END_INSTRUCTIONS
    instr = (
        f"{base}\n\nQuestion: {question}\n"
        f"start was {start_pos}:{word}"
    )
    return Choice(instructions=instr, criteria=criteria)


def build_joint_span_choice(
    tokens: Sequence[str],
    question: str,
    *,
    start_candidates: Sequence[int],
    end_hi: int | None = None,
    context_radius: int = DEFAULT_CONTEXT_RADIUS,
    max_pairs: int = 255,
    instructions: str | None = None,
) -> Choice:
    """Closed-set Choice over ``start_pos:end_pos`` pairs (optional joint mode).

    Only builds pairs where ``end >= start``. Raises if cardinality would exceed
    ``max_pairs`` (Jev Choice limit is 255).
    """
    if end_hi is None:
        end_hi = len(tokens)
    criteria: dict[str, str] = {}
    for s in start_candidates:
        if s < 0 or s >= len(tokens):
            continue
        for e in range(s, end_hi):
            key = f"{s}:{e}"
            left = format_token_context(tokens, s, radius=context_radius)
            right = format_token_context(tokens, e, radius=context_radius)
            criteria[key] = f"start {left}  →  end {right}"
            if len(criteria) > max_pairs:
                raise ValueError(
                    f"Joint (start,end) criteria exceed max_pairs={max_pairs} "
                    f"({len(start_candidates)} starts × ends up to {end_hi})"
                )
    if not criteria:
        raise ValueError("No joint (start,end) pairs to choose from")
    instr = instructions or (
        f"{DEFAULT_JOINT_INSTRUCTIONS}\n\nQuestion: {question}"
    )
    return Choice(instructions=instr, criteria=criteria)


def build_state(paragraph: str, question: str | None = None) -> dict[str, str]:
    """Structured state passed to ``system_one``."""
    state: dict[str, str] = {"paragraph": paragraph}
    if question:
        state["question"] = question
    return state


def build_field_question(name: str, spec: FieldSpec, spans: Sequence[Span]) -> Any:
    """Construct a Choice / Noul / Score question for one named field.

    Note: token-native ``span`` fields are handled inside :class:`Extractor`
    (sequential start/end Choice), not via this helper.
    """
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


# ---------------------------------------------------------------------------
# Extra builders for alternate extraction strategies
# ---------------------------------------------------------------------------

ANSWER_SHAPE_CRITERIA: dict[str, str] = {
    "number": "A numeric value, year, count, measurement, or percentage (often 1–3 tokens).",
    "date": "A calendar date or time expression (e.g. November 1, 1843, the 1890s).",
    "short_entity": "A short named entity: person, place, org, product, or technical term (1–4 tokens).",
    "phrase": "A short multi-word phrase or noun phrase that is not a full sentence (2–8 tokens).",
    "full_sentence": "A complete sentence or clause that answers the question; prefer covering the whole sentence.",
}

DEFAULT_SHAPE_INSTRUCTIONS = (
    "Classify the expected answer SHAPE for this extractive question given the paragraph. "
    "Pick the single best label. Prefer 'number'/'date'/'short_entity' when the question "
    "asks for a specific fact; pick 'full_sentence' only when the gold answer should be "
    "an entire sentence or long clause."
)

DEFAULT_SPAN_AS_CHOICE_INSTRUCTIONS = (
    "Select the single contiguous token span that best answers the question. "
    "Each criterion key is 'start_pos:end_pos'; the description shows local context "
    "with the candidate span marked «…». Prefer the shortest span that fully answers; "
    "do not invent tokens."
)

DEFAULT_ANCHOR_INSTRUCTIONS = (
    "Select the ANSWER-HEAD token — the most informative token inside the answer "
    "(e.g. the year in a date, the main noun in an entity). "
    "Each criterion key is the 0-based token position; descriptions mark «word». "
    "Return exactly one position key."
)

DEFAULT_NOUL_VERIFY_INSTRUCTIONS = (
    "Does the candidate span FULLY and CORRECTLY answer the question using only "
    "text from the paragraph? Answer yes only if the span is sufficient and not "
    "missing required tokens; answer no if it is incomplete, too broad, or wrong."
)


def count_span_options(n_tokens: int, max_len: int) -> int:
    """Number of contiguous spans of length 1..max_len over n_tokens."""
    if n_tokens <= 0 or max_len <= 0:
        return 0
    k = min(max_len, n_tokens)
    return k * (2 * n_tokens - k + 1) // 2


def choose_span_max_len(
    n_tokens: int,
    *,
    prefer_k: int = 13,
    max_criteria: int = 255,
) -> int:
    """Largest K ≤ prefer_k with contiguous-span cardinality ≤ max_criteria."""
    prefer_k = max(1, min(prefer_k, n_tokens if n_tokens > 0 else 1))
    for k in range(prefer_k, 0, -1):
        if count_span_options(n_tokens, k) <= max_criteria:
            return k
    return 1


def format_span_context(
    tokens: Sequence[str],
    start: int,
    end: int,
    *,
    radius: int = 2,
) -> str:
    """Local window with the whole [start, end] span marked inside «…»."""
    n = len(tokens)
    if start < 0 or end < start or end >= n:
        raise ValueError(f"Invalid span [{start}, {end}] for {n} tokens")
    lo = max(0, start - radius)
    hi = min(n, end + radius + 1)
    parts: list[str] = []
    if lo > 0:
        parts.append("…")
    for i in range(lo, hi):
        if i == start:
            parts.append("«" + tokens[i])
            if i == end:
                parts[-1] = parts[-1] + "»"
        elif i == end:
            parts.append(tokens[i] + "»")
        elif start < i < end:
            parts.append(tokens[i])
        else:
            parts.append(tokens[i])
    if hi < n:
        parts.append("…")
    label = f"{start}:{tokens[start]}..{end}:{tokens[end]}"
    return f"{label} | {' '.join(parts)}"


# ---------------------------------------------------------------------------
# Cheap span-as-Choice proposal (smart / in-chunk)
# ---------------------------------------------------------------------------

SPAN_PROPOSE_MODES = ("all", "smart", "in_chunk")

# Gold length histogram prior on extraction_50: ~88% ≤5 tokens, ~92% ≤8.
SMART_PREFER_K = 6  # aggressive; longer answers covered via sentence candidates
SMART_DENSE_MAX_LEN = 6  # gold hist: ~90%+ ≤6 tokens on extraction_50
SMART_TARGET_CRITERIA = 180  # soft cap; typically just dense+sentences
SMART_CRITERIA_TEXT_MAX = 64

_QUERY_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
        "is", "are", "was", "were", "be", "been", "being", "by", "with",
        "from", "as", "that", "this", "these", "those", "it", "its",
        "what", "which", "who", "whom", "whose", "when", "where", "why",
        "how", "does", "did", "do", "into", "about", "over", "under",
    }
)


def _query_terms(question: str) -> list[str]:
    """Content terms from the question (stdlib tokenize; light stopword drop)."""
    from jev_extract.tokenize import tokenize as _tok

    out: list[str] = []
    seen: set[str] = set()
    for t in _tok(question or ""):
        low = t.lower()
        if low in _QUERY_STOPWORDS or len(low) <= 1:
            continue
        if low not in seen:
            seen.add(low)
            out.append(low)
    return out


def _window_overlap_score(
    tokens: Sequence[str],
    start: int,
    end: int,
    q_terms: Sequence[str],
) -> float:
    """Light keyword overlap for candidate prune (not used for final extraction)."""
    if not q_terms:
        # Prefer shorter spans when unscored (histogram prior).
        return 1.0 / float(end - start + 1)
    window_lower = {tokens[i].lower() for i in range(start, end + 1)}
    hits = sum(1 for t in q_terms if t in window_lower)
    length = end - start + 1
    # Prefer more hits, then shorter spans (gold hist).
    return float(hits) * 10.0 + (1.0 / float(length))


def sentence_token_spans(
    paragraph: str,
    char_spans: Sequence[tuple[int, int]],
    *,
    lo: int = 0,
    hi: int | None = None,
) -> list[tuple[int, int]]:
    """Map sentence/clause chunks to inclusive token spans within [lo, hi)."""
    from jev_extract.candidates import propose_chunks
    from jev_extract.tokenize import char_span_to_token_span

    if hi is None:
        hi = len(char_spans)
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for sp in propose_chunks(paragraph):
        try:
            s, e = char_span_to_token_span(char_spans, sp.start, sp.end)
        except ValueError:
            continue
        # Clip to active range.
        s2 = max(s, lo)
        e2 = min(e, hi - 1)
        if s2 > e2:
            continue
        key = (s2, e2)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def enumerate_span_windows(
    n_tokens: int,
    max_len: int,
    *,
    lo: int = 0,
    hi: int | None = None,
) -> list[tuple[int, int]]:
    """All contiguous inclusive spans of length 1..max_len inside [lo, hi)."""
    if hi is None:
        hi = n_tokens
    if lo < 0 or hi > n_tokens or lo > hi:
        raise ValueError(f"Invalid token range [{lo}, {hi}) for {n_tokens} tokens")
    k = min(max_len, max(0, hi - lo))
    windows: list[tuple[int, int]] = []
    for length in range(1, k + 1):
        for s in range(lo, hi - length + 1):
            windows.append((s, s + length - 1))
    return windows


def propose_span_windows(
    tokens: Sequence[str],
    *,
    propose: str = "smart",
    question: str | None = None,
    paragraph: str | None = None,
    char_spans: Sequence[tuple[int, int]] | None = None,
    max_len: int | None = None,
    prefer_k: int = 13,
    max_criteria: int = 255,
    lo: int = 0,
    hi: int | None = None,
    dense_max_len: int = SMART_DENSE_MAX_LEN,
    smart_prefer_k: int = SMART_PREFER_K,
    target_criteria: int = SMART_TARGET_CRITERIA,
    include_sentences: bool = True,
) -> list[tuple[int, int]]:
    """Propose (start, end) inclusive token windows for span-as-Choice.

    ``propose``:
      * ``"all"`` — every window of length 1..K (K adaptive ≤ prefer_k), classic path.
      * ``"smart"`` — aggressive K (≤8), always keep length-1 + sentence spans,
        shortlist remaining by keyword overlap under ``target_criteria``.
      * ``"in_chunk"`` — same as smart but intended for a restricted [lo, hi);
        denser fill (higher target) because the range is already small.

    Always returns ≤ ``max_criteria`` unique windows (Jev Choice limit).
    """
    if propose not in ("all", "smart", "in_chunk"):
        raise ValueError(
            f"propose must be one of {SPAN_PROPOSE_MODES}; got {propose!r}"
        )
    n = len(tokens)
    if hi is None:
        hi = n
    if n == 0 or lo >= hi:
        return []

    range_n = hi - lo
    if propose == "all":
        k = (
            max_len
            if max_len is not None
            else choose_span_max_len(
                range_n, prefer_k=prefer_k, max_criteria=max_criteria
            )
        )
        windows = enumerate_span_windows(n, k, lo=lo, hi=hi)
        if len(windows) > max_criteria:
            # Should not happen when K is chosen via choose_span_max_len; truncate.
            windows = windows[:max_criteria]
        return windows

    # smart / in_chunk
    k_cap = smart_prefer_k if max_len is None else max_len
    k = min(k_cap, range_n)
    # Still respect hard cardinality for dense enumeration baseline.
    k = min(k, choose_span_max_len(range_n, prefer_k=k, max_criteria=max_criteria))

    target = target_criteria
    if propose == "in_chunk":
        # Chunk ranges are small — allow denser windows (still ≤ max_criteria).
        target = min(max_criteria, max(target_criteria, count_span_options(range_n, k)))

    must: list[tuple[int, int]] = []
    if include_sentences and paragraph is not None:
        spans_for_sent = char_spans
        if spans_for_sent is None:
            from jev_extract.tokenize import token_char_spans as _tcs

            spans_for_sent = _tcs(paragraph)
        must = sentence_token_spans(
            paragraph, spans_for_sent, lo=lo, hi=hi
        )

    pool = enumerate_span_windows(n, k, lo=lo, hi=hi)
    q_terms = _query_terms(question or "")

    selected: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def _add(se: tuple[int, int]) -> bool:
        if se in seen:
            return False
        if len(selected) >= max_criteria:
            return False
        seen.add(se)
        selected.append(se)
        return True

    for se in must:
        _add(se)

    # Always keep every length-1 window in range (years / short entities).
    for se in pool:
        if se[1] == se[0]:
            _add(se)

    # Dense lengths 2..dense_max_len: always keep all (histogram prior).
    # Only drop if we would exceed max_criteria (should not for typical paras).
    dense_k = min(dense_max_len, k)
    dense_rest = [se for se in pool if 2 <= (se[1] - se[0] + 1) <= dense_k]
    for se in dense_rest:
        _add(se)

    # Longer windows (dense_k+1 .. k): overlap shortlist up to target.
    longer = [se for se in pool if (se[1] - se[0] + 1) > dense_k]
    ranked_long = sorted(
        longer,
        key=lambda se: _window_overlap_score(tokens, se[0], se[1], q_terms),
        reverse=True,
    )
    long_budget = min(target, max_criteria)
    for se in ranked_long:
        if len(selected) >= long_budget:
            break
        _add(se)

    return selected


def build_span_as_choice(
    tokens: Sequence[str],
    question: str,
    *,
    max_len: int | None = None,
    prefer_k: int = 13,
    max_criteria: int = 255,
    context_radius: int = 2,
    instructions: str | None = None,
    windows: Sequence[tuple[int, int]] | None = None,
    propose: str = "smart",
    paragraph: str | None = None,
    char_spans: Sequence[tuple[int, int]] | None = None,
    criteria_text_max: int | None = None,
    lo: int = 0,
    hi: int | None = None,
) -> Choice:
    """One Choice over contiguous token windows.

    By default uses ``propose="smart"`` (aggressive length cap + overlap shortlist
    + sentence candidates + truncated criteria). Pass ``propose="all"`` for the
    classic full 1..K enumeration. Optional ``windows`` skips proposal entirely.
    """
    from jev_extract.candidates import truncate_criteria_text

    if windows is None:
        # "in_chunk" here means denser smart fill inside [lo, hi); chunk
        # selection itself is performed by run_span_choice.
        prop = propose if propose in ("all", "smart", "in_chunk") else "smart"
        windows = propose_span_windows(
            tokens,
            propose=prop,
            question=question,
            paragraph=paragraph,
            char_spans=char_spans,
            max_len=max_len,
            prefer_k=prefer_k,
            max_criteria=max_criteria,
            lo=lo,
            hi=hi,
        )

    text_max = criteria_text_max
    if text_max is None:
        text_max = (
            SMART_CRITERIA_TEXT_MAX
            if propose in ("smart", "in_chunk")
            else 500
        )

    criteria: dict[str, str] = {}
    for s, e in windows:
        if not (0 <= s <= e < len(tokens)):
            continue
        key = f"{s}:{e}"
        if key in criteria:
            continue
        desc = format_span_context(tokens, s, e, radius=context_radius)
        criteria[key] = truncate_criteria_text(desc, text_max)
        if len(criteria) > max_criteria:
            raise ValueError(
                f"Span-as-choice criteria exceed max_criteria={max_criteria}"
            )
    if not criteria:
        raise ValueError("No span candidates to choose from")
    instr = instructions or (
        f"{DEFAULT_SPAN_AS_CHOICE_INSTRUCTIONS}\n\nQuestion: {question}"
    )
    return Choice(instructions=instr, criteria=criteria)


def build_shape_gate_choice(
    question: str,
    *,
    instructions: str | None = None,
) -> Choice:
    """Choice over answer-shape labels for routing end priors."""
    instr = instructions or (
        f"{DEFAULT_SHAPE_INSTRUCTIONS}\n\nQuestion: {question}"
    )
    return Choice(instructions=instr, criteria=dict(ANSWER_SHAPE_CRITERIA))


def build_extent_choice(
    *,
    side: str,
    max_extent: int,
    head_pos: int,
    head_word: str,
    question: str,
) -> Choice:
    """Choice for how many tokens to expand left or right from an anchor head."""
    if max_extent < 0:
        raise ValueError("max_extent must be >= 0")
    criteria = {
        str(i): (
            f"Expand {side} by {i} token(s)"
            if i != 1
            else f"Expand {side} by 1 token"
        )
        for i in range(0, max_extent + 1)
    }
    if max_extent == 0:
        criteria["0"] = f"No {side} expansion (head is the {side} edge)"
    instr = (
        f"The answer-head token is {head_pos}:{head_word}. "
        f"How many tokens should the answer span expand to the {side}? "
        f"0 means the head is already the {side}most token of the answer. "
        f"Prefer the smallest expansion that fully answers the question.\n\n"
        f"Question: {question}"
    )
    return Choice(instructions=instr, criteria=criteria)


def build_noul_verify(
    *,
    span_text: str,
    question: str,
    instructions: str | None = None,
) -> Noul:
    """Noul: does this candidate span fully answer the question?"""
    base = instructions or DEFAULT_NOUL_VERIFY_INSTRUCTIONS
    instr = (
        f"{base}\n\nQuestion: {question}\n"
        f"Candidate span: «{span_text}»"
    )
    return Noul(
        instructions=instr,
        criteria={
            "true": "Yes — the span fully and correctly answers the question.",
            "false": "No — incomplete, too broad, or incorrect.",
        },
    )


SHAPE_END_INSTRUCTIONS: dict[str, str] = {
    "number": (
        DEFAULT_END_INSTRUCTIONS
        + " The expected answer is a NUMBER — stop as soon as the numeric "
        "expression is complete; do not include following units' extra words "
        "unless required by the question."
    ),
    "date": (
        DEFAULT_END_INSTRUCTIONS
        + " The expected answer is a DATE — include the full date expression "
        "but not trailing clause text."
    ),
    "short_entity": (
        DEFAULT_END_INSTRUCTIONS
        + " The expected answer is a SHORT ENTITY — prefer the minimal name; "
        "do not extend into following modifiers."
    ),
    "phrase": (
        DEFAULT_END_INSTRUCTIONS
        + " The expected answer is a short PHRASE — cover the full phrase but "
        "not an entire sentence."
    ),
    "full_sentence": (
        "Select the END token of the answer span (inclusive). "
        "The start token was already chosen (see 'start was pos:word' below). "
        "The expected answer is a FULL SENTENCE or long clause — prefer ending "
        "at the sentence terminator (period/question mark) or the natural "
        "clause boundary; do NOT truncate to just the subject noun phrase. "
        "Return exactly one position key. Do not invent tokens."
    ),
}
