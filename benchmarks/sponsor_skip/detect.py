"""Jev Choice detectors for sponsor / selfpromo segments in transcripts."""
from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from typesafe_sdk import Choice

from interval_metrics import merge_intervals

LABEL_CRITERIA: dict[str, str] = {
    "sponsor": (
        "Paid third-party advertisement or sponsorship read: brand plugs, "
        "discount codes, affiliate links, 'this video is sponsored by…'."
    ),
    "selfpromo": (
        "Creator promoting their own products, merch, secondary channel, "
        "patreon/membership, or unrelated own projects."
    ),
    "content": (
        "Regular video content — review, storytelling, demo, education. "
        "Not a promo break."
    ),
    "interaction": (
        "Engagement CTA only: like, subscribe, comment, bell, follow on socials."
    ),
}

SPONSOR_LIKE = frozenset({"sponsor", "selfpromo"})

CHUNK_INSTRUCTIONS = (
    "Classify the transcript CHUNK in state['chunk'] (with optional neighbors "
    "in state['before'] / state['after']). Pick exactly one label. "
    "Mid-roll sponsorship reads are usually 30–90s of clear product pitching "
    "with a discount code or URL — label those sponsor. Do not label ordinary "
    "brand names mentioned during a review as sponsor unless it is an ad read."
)

PARA_START_INSTRUCTIONS = (
    "Select the sentence index that BEST MARKS THE START of a paid sponsorship "
    "or self-promo ad-read in this transcript. Keys are sentence indices; "
    "descriptions are short excerpts. If there is no sponsorship, pick 'none'."
)

PARA_END_INSTRUCTIONS = (
    "The sponsorship START sentence was already chosen (see state['start_index']). "
    "Select the END sentence index (inclusive, >= start) where the ad-read finishes "
    "and regular content resumes. Prefer the shortest span that covers the full ad."
)


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "input_tokens", 0) or 0), int(
        getattr(usage, "output_tokens", 0) or 0
    )


def _get_choice(response: Any, name: str) -> Any:
    choices = getattr(response, "choices", None)
    if isinstance(choices, Mapping) and name in choices:
        return choices[name]
    answers = getattr(response, "answers", None)
    if isinstance(answers, Mapping) and name in answers:
        return answers[name]
    raise KeyError(name)


class SponsorDetector:
    """Closed-set Choice detectors over transcript chunks / sentences."""

    def __init__(self, client: Any, *, model: str = "jev-latest") -> None:
        self.client = client
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.latencies_ms: list[float] = []
        self.call_log: list[dict[str, Any]] = []

    def _call(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        tag: str,
    ) -> Any:
        t0 = time.perf_counter()
        resp = self.client.system_one(dict(state), dict(questions), model=self.model)
        ms = (time.perf_counter() - t0) * 1000.0
        inp, out = _usage(resp)
        self.total_input_tokens += inp
        self.total_output_tokens += out
        self.latencies_ms.append(ms)
        self.call_log.append(
            {"tag": tag, "latency_ms": ms, "input_tokens": inp, "output_tokens": out}
        )
        return resp

    def classify_chunk(
        self,
        chunk: Mapping[str, Any],
        *,
        before: str = "",
        after: str = "",
    ) -> dict[str, Any]:
        state = {
            "chunk": chunk["text"][:1200],
            "before": (before or "")[-400:],
            "after": (after or "")[:400],
            "t_start": round(float(chunk["start"]), 2),
            "t_end": round(float(chunk["end"]), 2),
        }
        q = Choice(instructions=CHUNK_INSTRUCTIONS, criteria=dict(LABEL_CRITERIA))
        resp = self._call(state, {"label": q}, tag="chunk_label")
        ans = _get_choice(resp, "label")
        label = str(ans.choice)
        conf = getattr(ans, "confidence", None)
        probs = getattr(ans, "probabilities", None)
        if probs is not None and not isinstance(probs, dict):
            probs = dict(probs)
        return {
            "index": chunk["index"],
            "start": float(chunk["start"]),
            "end": float(chunk["end"]),
            "label": label,
            "confidence": float(conf) if conf is not None else None,
            "probabilities": probs,
            "text_preview": chunk["text"][:160],
        }

    def detect_chunk_choice(
        self,
        chunks: Sequence[Mapping[str, Any]],
        *,
        context_neighbors: int = 1,
        sponsor_labels: frozenset[str] = SPONSOR_LIKE,
        merge_gap: float = 8.0,
    ) -> dict[str, Any]:
        """Classify each ~15s chunk; merge adjacent sponsor-like labels into segments."""
        labeled: list[dict[str, Any]] = []
        n = len(chunks)
        for i, ch in enumerate(chunks):
            before = " ".join(
                chunks[j]["text"] for j in range(max(0, i - context_neighbors), i)
            )
            after = " ".join(
                chunks[j]["text"]
                for j in range(i + 1, min(n, i + 1 + context_neighbors))
            )
            labeled.append(self.classify_chunk(ch, before=before, after=after))

        # Merge adjacent sponsor-like chunks
        raw_intervals: list[tuple[float, float]] = []
        run_start: float | None = None
        run_end: float | None = None
        for row in labeled:
            if row["label"] in sponsor_labels:
                if run_start is None:
                    run_start = row["start"]
                run_end = row["end"]
            else:
                if run_start is not None and run_end is not None:
                    raw_intervals.append((run_start, run_end))
                run_start = run_end = None
        if run_start is not None and run_end is not None:
            raw_intervals.append((run_start, run_end))

        segments = merge_intervals(raw_intervals, gap=merge_gap)
        return {
            "mode": "chunk_choice",
            "chunk_labels": labeled,
            "segments": [{"start": a, "end": b, "category": "sponsor"} for a, b in segments],
            "intervals": segments,
        }

    def detect_chunk_choice_batched(
        self,
        chunks: Sequence[Mapping[str, Any]],
        *,
        batch_size: int = 8,
        context_neighbors: int = 1,
        sponsor_labels: frozenset[str] = SPONSOR_LIKE,
        merge_gap: float = 8.0,
    ) -> dict[str, Any]:
        """Same as chunk_choice but batches up to *batch_size* Choice questions/call.

        State holds chunk texts keyed by id; criteria stay short (labels only).
        """
        labeled: list[dict[str, Any]] = []
        n = len(chunks)
        for batch_start in range(0, n, batch_size):
            batch = list(chunks[batch_start : batch_start + batch_size])
            state: dict[str, Any] = {"chunks": {}}
            questions: dict[str, Any] = {}
            for ch in batch:
                i = int(ch["index"])
                before = " ".join(
                    chunks[j]["text"]
                    for j in range(max(0, i - context_neighbors), i)
                )
                after = " ".join(
                    chunks[j]["text"]
                    for j in range(i + 1, min(n, i + 1 + context_neighbors))
                )
                cid = f"c{i}"
                state["chunks"][cid] = {
                    "text": ch["text"][:1000],
                    "before": before[-300:],
                    "after": after[:300],
                    "t_start": round(float(ch["start"]), 2),
                    "t_end": round(float(ch["end"]), 2),
                }
                questions[cid] = Choice(
                    instructions=(
                        f"Classify state['chunks']['{cid}'] only. "
                        "Pick exactly one label. Mid-roll sponsorship reads are "
                        "clear product pitches with discount codes/URLs."
                    ),
                    criteria=dict(LABEL_CRITERIA),
                )
            resp = self._call(state, questions, tag=f"chunk_batch_{batch_start}")
            for ch in batch:
                cid = f"c{int(ch['index'])}"
                ans = _get_choice(resp, cid)
                conf = getattr(ans, "confidence", None)
                probs = getattr(ans, "probabilities", None)
                if probs is not None and not isinstance(probs, dict):
                    probs = dict(probs)
                labeled.append(
                    {
                        "index": ch["index"],
                        "start": float(ch["start"]),
                        "end": float(ch["end"]),
                        "label": str(ans.choice),
                        "confidence": float(conf) if conf is not None else None,
                        "probabilities": probs,
                        "text_preview": ch["text"][:160],
                    }
                )

        labeled.sort(key=lambda r: r["index"])
        raw_intervals: list[tuple[float, float]] = []
        run_start = run_end = None
        for row in labeled:
            if row["label"] in sponsor_labels:
                if run_start is None:
                    run_start = row["start"]
                run_end = row["end"]
            else:
                if run_start is not None and run_end is not None:
                    raw_intervals.append((run_start, run_end))
                run_start = run_end = None
        if run_start is not None and run_end is not None:
            raw_intervals.append((run_start, run_end))
        segments = merge_intervals(raw_intervals, gap=merge_gap)
        return {
            "mode": "chunk_choice_batched",
            "chunk_labels": labeled,
            "segments": [{"start": a, "end": b, "category": "sponsor"} for a, b in segments],
            "intervals": segments,
        }

    def detect_paragraph_span(
        self,
        sentences: Sequence[Mapping[str, Any]],
        *,
        max_regions: int = 3,
    ) -> dict[str, Any]:
        """Paragraph-level: repeated start/end Choice over sentence indices (≤255).

        Adds a synthetic 'none' criterion so the model can abstain. After each
        region is found, those sentence indices are removed from candidates.
        """
        remaining = {int(s["index"]): s for s in sentences}
        found: list[tuple[float, float]] = []
        traces: list[dict[str, Any]] = []

        for _ in range(max_regions):
            if len(remaining) < 2:
                break
            # Start Choice
            criteria: dict[str, str] = {
                "none": "No sponsorship / self-promo ad-read remains in the candidates."
            }
            for idx, s in sorted(remaining.items()):
                preview = s["text"][:140].replace("\n", " ")
                criteria[str(idx)] = f"[{s['start']:.0f}s–{s['end']:.0f}s] {preview}"

            # Keep criteria under Jev 255 limit (none + sentences)
            if len(criteria) > 255:
                # Drop excess highest indices
                keys = sorted((k for k in criteria if k != "none"), key=int)
                for k in keys[254:]:
                    criteria.pop(k, None)

            state = {
                "transcript_excerpt": "\n".join(
                    f"[{i}] ({remaining[i]['start']:.0f}s) {remaining[i]['text'][:100]}"
                    for i in sorted(remaining)[:80]
                )[:6000],
                "n_candidates": len(criteria) - 1,
            }
            resp = self._call(
                state,
                {
                    "start": Choice(
                        instructions=PARA_START_INSTRUCTIONS, criteria=criteria
                    )
                },
                tag="para_start",
            )
            start_ans = _get_choice(resp, "start")
            start_key = str(start_ans.choice).strip()
            if start_key.lower() == "none":
                traces.append({"start": start_key, "end": None, "abstain": True})
                break
            try:
                start_idx = int(start_key.split(":", 1)[0])
            except ValueError:
                traces.append({"start": start_key, "end": None, "abstain": True, "error": "bad_start"})
                break
            if start_idx not in remaining:
                traces.append({"start": start_idx, "end": None, "abstain": True, "error": "missing_idx"})
                break

            # End Choice: only indices >= start
            end_criteria: dict[str, str] = {}
            for idx, s in sorted(remaining.items()):
                if idx < start_idx:
                    continue
                preview = s["text"][:140].replace("\n", " ")
                end_criteria[str(idx)] = f"[{s['start']:.0f}s–{s['end']:.0f}s] {preview}"
            if not end_criteria:
                break
            if len(end_criteria) > 255:
                keys = sorted(end_criteria, key=int)
                end_criteria = {k: end_criteria[k] for k in keys[:255]}

            end_state = {
                **state,
                "start_index": start_idx,
                "start_text": remaining[start_idx]["text"][:200],
            }
            resp2 = self._call(
                end_state,
                {
                    "end": Choice(
                        instructions=PARA_END_INSTRUCTIONS, criteria=end_criteria
                    )
                },
                tag="para_end",
            )
            end_ans = _get_choice(resp2, "end")
            end_key = str(end_ans.choice)
            try:
                end_idx = int(end_key)
            except ValueError:
                traces.append(
                    {"start": start_idx, "end": end_key, "error": "bad_end"}
                )
                break
            if end_idx < start_idx or end_idx not in remaining:
                # clamp
                end_idx = max(
                    (i for i in remaining if i >= start_idx), default=start_idx
                )

            a = float(remaining[start_idx]["start"])
            b = float(remaining[end_idx]["end"])
            found.append((a, b))
            traces.append(
                {
                    "start": start_idx,
                    "end": end_idx,
                    "t_start": a,
                    "t_end": b,
                    "start_conf": getattr(start_ans, "confidence", None),
                    "end_conf": getattr(end_ans, "confidence", None),
                }
            )
            # Remove covered sentences (+ small pad)
            for idx in list(remaining):
                s = remaining[idx]
                if s["end"] >= a - 1 and s["start"] <= b + 1:
                    remaining.pop(idx, None)

        segments = merge_intervals(found, gap=5.0)
        return {
            "mode": "paragraph_span",
            "traces": traces,
            "segments": [{"start": a, "end": b, "category": "sponsor"} for a, b in segments],
            "intervals": segments,
        }

    def refine_boundaries(
        self,
        cues: Sequence[Mapping[str, Any]],
        rough: Sequence[tuple[float, float]],
        *,
        pad_sec: float = 20.0,
    ) -> list[tuple[float, float]]:
        """Optional span refine: start/end Choice over cue indices inside padded region."""
        if not rough or not cues:
            return list(rough)
        refined: list[tuple[float, float]] = []
        for a, b in rough:
            lo, hi = a - pad_sec, b + pad_sec
            window = [
                (i, c)
                for i, c in enumerate(cues)
                if float(c["end"]) >= lo and float(c["start"]) <= hi
            ]
            if len(window) < 2:
                refined.append((a, b))
                continue
            # Cap at 255
            if len(window) > 255:
                # downsample evenly
                step = len(window) / 255
                window = [window[int(i * step)] for i in range(255)]

            criteria = {
                str(i): f"[{c['start']:.1f}s] {c['text'][:100]}" for i, c in window
            }
            idx_to_cue = {str(i): c for i, c in window}
            state = {
                "region": f"Rough sponsor region {a:.1f}–{b:.1f}s; tighten boundaries."
            }
            resp = self._call(
                state,
                {
                    "start": Choice(
                        instructions=(
                            "Select the cue index where the sponsorship ad-read STARTS."
                        ),
                        criteria=criteria,
                    )
                },
                tag="refine_start",
            )
            start_key = str(_get_choice(resp, "start").choice)
            if start_key not in idx_to_cue:
                refined.append((a, b))
                continue
            start_i = int(start_key)
            end_criteria = {
                k: v
                for k, v in criteria.items()
                if int(k) >= start_i
            }
            resp2 = self._call(
                {**state, "start_key": start_key},
                {
                    "end": Choice(
                        instructions=(
                            "Select the cue index where the sponsorship ad-read ENDS "
                            f"(start was cue {start_key})."
                        ),
                        criteria=end_criteria,
                    )
                },
                tag="refine_end",
            )
            end_key = str(_get_choice(resp2, "end").choice)
            if end_key not in idx_to_cue:
                refined.append((a, b))
                continue
            na = float(idx_to_cue[start_key]["start"])
            nb = float(idx_to_cue[end_key]["end"])
            if nb > na:
                refined.append((na, nb))
            else:
                refined.append((a, b))
        return merge_intervals(refined, gap=3.0)


STRICT_LABEL_CRITERIA: dict[str, str] = {
    "sponsor": (
        "ONLY a paid third-party advertisement / sponsorship read: the creator "
        "explicitly plugs an external brand that paid for the segment, usually "
        "with 'sponsored by', 'our sponsor', 'segue to our sponsor', a discount "
        "code, or a CTA link. Product review praise, brand names in a review, "
        "merch teases, and 'segue' jokes WITHOUT an actual paid read are NOT sponsor."
    ),
    "selfpromo": (
        "Creator promoting their OWN products, merch, secondary channel, "
        "Patreon/membership, floatplane, or unrelated own projects — not a "
        "paid third-party ad."
    ),
    "content": (
        "Regular video content: review, storytelling, demo, education, banter. "
        "Includes product praise during a review and promotional-sounding "
        "language that is NOT a paid ad-read."
    ),
    "interaction": (
        "Engagement CTA only: like, subscribe, comment, bell, follow on socials."
    ),
}

STRICT_BATCH_INSTRUCTIONS = (
    "Classify state['chunks']['{cid}'] only. Pick exactly one label. "
    "STRICT: sponsor = paid third-party ad-read only. Product praise / review "
    "mentions / own merch ≠ sponsor. Prefer content when unsure."
)


def _merge_labeled_sponsor_runs(
    labeled: Sequence[Mapping[str, Any]],
    *,
    sponsor_labels: frozenset[str],
    merge_gap: float,
) -> list[tuple[float, float]]:
    raw_intervals: list[tuple[float, float]] = []
    run_start: float | None = None
    run_end: float | None = None
    for row in labeled:
        if row["label"] in sponsor_labels:
            if run_start is None:
                run_start = float(row["start"])
            run_end = float(row["end"])
        else:
            if run_start is not None and run_end is not None:
                raw_intervals.append((run_start, run_end))
            run_start = run_end = None
    if run_start is not None and run_end is not None:
        raw_intervals.append((run_start, run_end))
    return merge_intervals(raw_intervals, gap=merge_gap)


def _classify_chunks_batched_strict(
    detector: SponsorDetector,
    chunks: Sequence[Mapping[str, Any]],
    all_chunks: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    context_neighbors: int,
    tag_prefix: str,
    strong_cue_times: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Batch-classify *chunks* with strict sponsor criteria (neighbors from all_chunks)."""
    labeled: list[dict[str, Any]] = []
    n = len(all_chunks)
    by_idx = {int(c["index"]): (i, c) for i, c in enumerate(all_chunks)}
    cue_times = list(strong_cue_times or [])

    def near_cue(start: float, end: float) -> bool:
        for t in cue_times:
            if start - 5 <= t <= end + 5:
                return True
        return False

    def neighbor_text(idx: int, direction: str) -> str:
        pos = by_idx.get(idx)
        if pos is None:
            return ""
        list_i = pos[0]
        if direction == "before":
            rng = range(max(0, list_i - context_neighbors), list_i)
        else:
            rng = range(list_i + 1, min(n, list_i + 1 + context_neighbors))
        return " ".join(all_chunks[j]["text"] for j in rng)

    for batch_start in range(0, len(chunks), batch_size):
        batch = list(chunks[batch_start : batch_start + batch_size])
        if not batch:
            break
        state: dict[str, Any] = {"chunks": {}}
        questions: dict[str, Any] = {}
        for ch in batch:
            i = int(ch["index"])
            cid = f"c{i}"
            has_cue = near_cue(float(ch["start"]), float(ch["end"]))
            state["chunks"][cid] = {
                "text": ch["text"][:1000],
                "before": neighbor_text(i, "before")[-300:],
                "after": neighbor_text(i, "after")[:300],
                "t_start": round(float(ch["start"]), 2),
                "t_end": round(float(ch["end"]), 2),
                "near_sponsor_cue": has_cue,
            }
            extra = (
                " A regex sponsor cue (e.g. 'our sponsor' / 'segue to our sponsor') "
                "falls in or next to this chunk — lean sponsor if the pitch/CTA follows."
                if has_cue
                else ""
            )
            questions[cid] = Choice(
                instructions=STRICT_BATCH_INSTRUCTIONS.format(cid=cid) + extra,
                criteria=dict(STRICT_LABEL_CRITERIA),
            )
        resp = detector._call(state, questions, tag=f"{tag_prefix}_{batch_start}")
        for ch in batch:
            cid = f"c{int(ch['index'])}"
            ans = _get_choice(resp, cid)
            conf = getattr(ans, "confidence", None)
            probs = getattr(ans, "probabilities", None)
            if probs is not None and not isinstance(probs, dict):
                probs = dict(probs)
            labeled.append(
                {
                    "index": ch["index"],
                    "start": float(ch["start"]),
                    "end": float(ch["end"]),
                    "label": str(ans.choice),
                    "confidence": float(conf) if conf is not None else None,
                    "probabilities": probs,
                    "text_preview": ch["text"][:160],
                }
            )
    labeled.sort(key=lambda r: r["index"])
    return labeled


def detect_regex_propose_jev_confirm(
    self: SponsorDetector,
    chunks: Sequence[Mapping[str, Any]],
    cues: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 8,
    context_neighbors: int = 1,
    pad_before: float = 12.0,
    pad_after: float = 80.0,
    merge_gap: float = 8.0,
    video_duration: float | None = None,
    include_weak: bool = False,
) -> dict[str, Any]:
    """Regex/cue propose windows → Jev Choice confirm on overlapping chunks only.

    Strong cues only by default (weak URL-ish cues inflate FPs). Once any chunk
    inside a proposed window is confirmed ``sponsor``, expand to cue→end-of-ad
    (covers narrative/sketch mid-rolls).
    """
    from regex_propose import (
        chunks_overlapping_windows,
        expand_confirmed_windows,
        find_cue_hits,
        pattern_hit_summary,
        propose_windows_from_hits,
        refine_boundaries_heuristic,
    )

    hits = find_cue_hits(cues, include_weak=include_weak)
    windows = propose_windows_from_hits(
        hits,
        pad_before=pad_before,
        pad_after=pad_after,
        video_duration=video_duration,
    )
    cand_chunks = chunks_overlapping_windows(chunks, windows)
    strong_times = [h.start for h in hits if h.strength == "strong"]
    labeled = _classify_chunks_batched_strict(
        self,
        cand_chunks,
        chunks,
        batch_size=batch_size,
        context_neighbors=context_neighbors,
        tag_prefix="regex_jev_batch",
        strong_cue_times=strong_times,
    )
    # Primary: expand each proposed window that got ≥1 sponsor confirmation
    intervals = expand_confirmed_windows(
        windows, labeled, cues, hits, sponsor_labels=frozenset({"sponsor"})
    )
    # Fallback: merge labeled runs if expand produced nothing but labels exist
    if not intervals:
        intervals = _merge_labeled_sponsor_runs(
            labeled, sponsor_labels=frozenset({"sponsor"}), merge_gap=merge_gap
        )
        intervals = refine_boundaries_heuristic(cues, intervals, hits)
    return {
        "mode": "regex_propose_jev_confirm",
        "cue_hits": [
            {
                "start": h.start,
                "end": h.end,
                "pattern_name": h.pattern_name,
                "strength": h.strength,
                "text": h.text[:160],
            }
            for h in hits
        ],
        "pattern_counts": pattern_hit_summary(hits),
        "proposed_windows": windows,
        "n_candidate_chunks": len(cand_chunks),
        "chunk_labels": labeled,
        "segments": [{"start": a, "end": b, "category": "sponsor"} for a, b in intervals],
        "intervals": intervals,
    }


def detect_jev_then_regex_gate(
    self: SponsorDetector,
    chunks: Sequence[Mapping[str, Any]],
    cues: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 8,
    context_neighbors: int = 1,
    merge_gap: float = 8.0,
    high_conf: float = 0.9,
    cue_pad: float = 8.0,
) -> dict[str, Any]:
    """Jev-classify all chunks (strict), drop sponsor segs lacking a strong regex cue."""
    from regex_propose import (
        cue_in_interval,
        find_cue_hits,
        pattern_hit_summary,
        refine_boundaries_heuristic,
    )

    hits = find_cue_hits(cues, include_weak=False)
    labeled = _classify_chunks_batched_strict(
        self,
        list(chunks),
        chunks,
        batch_size=batch_size,
        context_neighbors=context_neighbors,
        tag_prefix="jev_gate_batch",
        strong_cue_times=[h.start for h in hits],
    )
    raw = _merge_labeled_sponsor_runs(
        labeled, sponsor_labels=frozenset({"sponsor"}), merge_gap=merge_gap
    )

    gated: list[tuple[float, float]] = []
    dropped: list[dict[str, Any]] = []
    for a, b in raw:
        has_cue = cue_in_interval(hits, a, b, pad=cue_pad, strong_only=True)
        confs = [
            float(r["confidence"])
            for r in labeled
            if r["label"] == "sponsor"
            and r["confidence"] is not None
            and float(r["end"]) >= a
            and float(r["start"]) <= b
        ]
        max_c = max(confs) if confs else 0.0
        if has_cue or max_c >= high_conf:
            gated.append((a, b))
        else:
            dropped.append(
                {"start": a, "end": b, "max_conf": max_c, "reason": "no_cue_low_conf"}
            )

    # Expand gated runs via the same cue-window logic as regex_propose
    from regex_propose import expand_confirmed_windows, propose_windows_from_hits

    windows = propose_windows_from_hits(hits, pad_before=12.0, pad_after=80.0)
    # Only keep windows that overlap a gated interval (cue-backed or high-conf)
    kept_windows = []
    for w in windows:
        ws, we = float(w["start"]), float(w["end"])
        if any(not (we < a or ws > b) for a, b in gated):
            kept_windows.append(w)
    # Also synthesize windows from high-conf gated segs with no cue window
    for a, b in gated:
        if not any(not (float(w["end"]) < a or float(w["start"]) > b) for w in windows):
            kept_windows.append(
                {"start": a, "end": b, "strength": "strong", "pattern_names": ["high_conf"], "hits": []}
            )
    expanded = expand_confirmed_windows(
        kept_windows or [
            {"start": a, "end": b, "strength": "strong", "pattern_names": [], "hits": []}
            for a, b in gated
        ],
        labeled,
        cues,
        hits,
        sponsor_labels=frozenset({"sponsor"}),
    )
    gated = expanded if expanded else refine_boundaries_heuristic(cues, gated, hits)
    return {
        "mode": "jev_then_regex_gate",
        "cue_hits": [
            {
                "start": h.start,
                "end": h.end,
                "pattern_name": h.pattern_name,
                "strength": h.strength,
                "text": h.text[:160],
            }
            for h in hits
        ],
        "pattern_counts": pattern_hit_summary(hits),
        "dropped_segments": dropped,
        "chunk_labels": labeled,
        "segments": [{"start": a, "end": b, "category": "sponsor"} for a, b in gated],
        "intervals": gated,
    }



def detect_regex_jev_refined(
    self: SponsorDetector,
    chunks: Sequence[Mapping[str, Any]],
    cues: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 8,
    context_neighbors: int = 1,
    pad_before: float = 12.0,
    pad_after: float = 80.0,
    merge_gap: float = 8.0,
    video_duration: float | None = None,
    include_weak: bool = False,
    confidence_threshold: float = 0.75,
    offer_pad_sec: float = 8.0,
    lead_in_lookback: float = 30.0,
) -> dict[str, Any]:
    """Regex→Jev confirm + boundary snap + confidence gate (recommended EN default).

    Same proposal/confirm as ``regex_propose_jev_confirm``, then:

    1. **Boundary snap** — start prefers lead-in / first strong CTA; end snaps to
       return-to-content cues or caps shortly after offer CTA (conservative:
       never cut the ad short; avoid weak early starts).
    2. **Confidence gate** — attach Jev Choice confidence per segment; below
       *confidence_threshold* → ``auto_skip=False`` / ``needs_confirm=True``
       (still returned for UI ask).
    """
    from regex_propose import (
        attach_confidence_gate,
        chunks_overlapping_windows,
        confidence_gate_summary,
        expand_confirmed_windows,
        find_cue_hits,
        pattern_hit_summary,
        propose_windows_from_hits,
        refine_boundaries_heuristic,
        snap_boundaries_refined,
    )

    hits = find_cue_hits(cues, include_weak=include_weak)
    windows = propose_windows_from_hits(
        hits,
        pad_before=pad_before,
        pad_after=pad_after,
        video_duration=video_duration,
    )
    cand_chunks = chunks_overlapping_windows(chunks, windows)
    strong_times = [h.start for h in hits if h.strength == "strong"]
    labeled = _classify_chunks_batched_strict(
        self,
        cand_chunks,
        chunks,
        batch_size=batch_size,
        context_neighbors=context_neighbors,
        tag_prefix="regex_jev_refined_batch",
        strong_cue_times=strong_times,
    )
    intervals = expand_confirmed_windows(
        windows, labeled, cues, hits, sponsor_labels=frozenset({"sponsor"})
    )
    if not intervals:
        intervals = _merge_labeled_sponsor_runs(
            labeled, sponsor_labels=frozenset({"sponsor"}), merge_gap=merge_gap
        )
        intervals = refine_boundaries_heuristic(cues, intervals, hits)

    snapped = snap_boundaries_refined(
        cues,
        intervals,
        hits,
        lead_in_lookback=lead_in_lookback,
        offer_pad_sec=offer_pad_sec,
    )
    segments = attach_confidence_gate(
        snapped, labeled, confidence_threshold=confidence_threshold
    )
    gate = confidence_gate_summary(segments)
    return {
        "mode": "regex_jev_refined",
        "confidence_threshold": confidence_threshold,
        "cue_hits": [
            {
                "start": h.start,
                "end": h.end,
                "pattern_name": h.pattern_name,
                "strength": h.strength,
                "text": h.text[:160],
            }
            for h in hits
        ],
        "pattern_counts": pattern_hit_summary(hits),
        "proposed_windows": windows,
        "n_candidate_chunks": len(cand_chunks),
        "chunk_labels": labeled,
        "intervals_pre_snap": intervals,
        "segments": segments,
        "intervals": [(s["start"], s["end"]) for s in segments],
        "confidence_gate": gate,
    }


# Bind hybrid detectors onto SponsorDetector
SponsorDetector.detect_regex_propose_jev_confirm = detect_regex_propose_jev_confirm  # type: ignore[method-assign]
SponsorDetector.detect_regex_jev_refined = detect_regex_jev_refined  # type: ignore[method-assign]
SponsorDetector.detect_jev_then_regex_gate = detect_jev_then_regex_gate  # type: ignore[method-assign]


def detect_tony_line_scan(
    self: SponsorDetector,
    cues: Sequence[Mapping[str, Any]],
    *,
    title: str = "unknown",
    max_segments: int = 6,
    scan_workers: int = 4,
    trace_lead_in: bool = True,
) -> dict[str, Any]:
    """Tony-style labelled-line scan / refine (see tony_line_scan.py)."""
    from tony_line_scan import detect_tony_line_scan as _tony

    return _tony(
        self,
        cues,
        title=title,
        max_segments=max_segments,
        scan_workers=scan_workers,
        trace_lead_in=trace_lead_in,
    )


SponsorDetector.detect_tony_line_scan = detect_tony_line_scan  # type: ignore[method-assign]
