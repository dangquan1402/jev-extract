"""Regex / light-heuristic sponsor cue proposal and boundary refine."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from interval_metrics import merge_intervals

# Strong paid-read / CTA language (case-insensitive).
STRONG_PATTERNS: list[tuple[str, str]] = [
    ("sponsored_by", r"\bsponsored by\b"),
    ("thanks_to_sponsor", r"\bthanks to (our )?sponsor\b"),
    ("video_sponsored", r"\bthis (video|episode) is sponsored\b"),
    ("brought_to_you", r"\bbrought to you by\b"),
    ("paid_partnership", r"\bpaid partnership\b"),
    ("segue_sponsor", r"\bsegue to our sponsor\b"),
    ("segue_to_our", r"\bsegue to our\b"),  # ASR truncation / garble
    ("todays_sponsor", r"\btoday'?s sponsor\b"),
    ("our_sponsor", r"\bour sponsor\b"),
    ("use_code", r"\buse (code|promo)\b"),
    ("promo_code", r"\bpromo code\b"),
    ("discount_code", r"\bdiscount code\b"),
    ("pct_off", r"\b\d+\s*%\s*off\b"),
    ("link_in_desc", r"\blink in (the )?description\b"),
    ("link_down_below", r"\blink down below\b"),
    ("with_the_link", r"\bwith the link\b"),
    ("check_out_at", r"\bcheck out .{2,40} at\b"),
]

# Weaker cues — still propose, but Jev must confirm more carefully.
WEAK_PATTERNS: list[tuple[str, str]] = [
    ("brand_urlish", r"\b\w+\.(com|net|io|dev|app)\b"),
    ("visit_site", r"\b(visit|go to) \w+\.(com|net|io|dev)\b"),
    ("first_month", r"\bfirst (month|year)\b"),
    ("free_shipping", r"\bfree shipping\b"),
]

# End-of-ad / return-to-content cues.
END_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.I)
    for p in [
        r"\bback to the (video|show|review|topic)\b",
        r"\banyway\b",
        r"\bwith that said\b",
        r"\bif you (guys )?enjoyed\b",
        r"\bnow[,.]?\s+(to return|back to|let'?s get back)\b",
        r"\blet'?s get back\b",
        r"\bmoving on\b",
        r"\bso (anyway|back to)\b",
    ]
]

_COMPILED_STRONG = [(n, re.compile(p, re.I)) for n, p in STRONG_PATTERNS]
_COMPILED_WEAK = [(n, re.compile(p, re.I)) for n, p in WEAK_PATTERNS]


@dataclass
class CueHit:
    start: float
    end: float
    text: str
    pattern_name: str
    strength: str  # "strong" | "weak"
    cue_index: int


def find_cue_hits(
    cues: Sequence[Mapping[str, Any]],
    *,
    include_weak: bool = True,
) -> list[CueHit]:
    """Scan transcript cues for sponsor-language regex hits."""
    hits: list[CueHit] = []
    for i, c in enumerate(cues):
        text = str(c.get("text") or "")
        if not text.strip():
            continue
        matched: CueHit | None = None
        for name, rx in _COMPILED_STRONG:
            if rx.search(text):
                matched = CueHit(
                    start=float(c["start"]),
                    end=float(c["end"]),
                    text=text,
                    pattern_name=name,
                    strength="strong",
                    cue_index=i,
                )
                break
        if matched is None and include_weak:
            for name, rx in _COMPILED_WEAK:
                if rx.search(text):
                    matched = CueHit(
                        start=float(c["start"]),
                        end=float(c["end"]),
                        text=text,
                        pattern_name=name,
                        strength="weak",
                        cue_index=i,
                    )
                    break
        if matched is not None:
            hits.append(matched)
    return hits


def propose_windows_from_hits(
    hits: Sequence[CueHit],
    *,
    pad_before: float = 12.0,
    pad_after: float = 75.0,
    merge_gap: float = 20.0,
    video_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Expand cue hits into candidate time windows and merge nearby ones."""
    if not hits:
        return []
    raw: list[tuple[float, float]] = []
    meta: list[list[CueHit]] = []
    for h in hits:
        a = max(0.0, h.start - pad_before)
        b = h.start + pad_after
        if video_duration is not None:
            b = min(b, float(video_duration))
        raw.append((a, b))
        meta.append([h])

    # Merge overlapping / near windows; track contributing hits
    order = sorted(range(len(raw)), key=lambda i: raw[i][0])
    merged_iv: list[tuple[float, float]] = []
    merged_hits: list[list[CueHit]] = []
    for i in order:
        a, b = raw[i]
        if not merged_iv:
            merged_iv.append((a, b))
            merged_hits.append(list(meta[i]))
            continue
        la, lb = merged_iv[-1]
        if a <= lb + merge_gap:
            merged_iv[-1] = (la, max(lb, b))
            merged_hits[-1].extend(meta[i])
        else:
            merged_iv.append((a, b))
            merged_hits.append(list(meta[i]))

    windows: list[dict[str, Any]] = []
    for (a, b), hs in zip(merged_iv, merged_hits):
        names = sorted({h.pattern_name for h in hs})
        strength = "strong" if any(h.strength == "strong" for h in hs) else "weak"
        windows.append(
            {
                "start": a,
                "end": b,
                "strength": strength,
                "pattern_names": names,
                "hits": [
                    {
                        "start": h.start,
                        "end": h.end,
                        "pattern_name": h.pattern_name,
                        "strength": h.strength,
                        "text": h.text[:160],
                        "cue_index": h.cue_index,
                    }
                    for h in hs
                ],
            }
        )
    return windows


def chunks_overlapping_windows(
    chunks: Sequence[Mapping[str, Any]],
    windows: Sequence[Mapping[str, Any]],
    *,
    min_overlap: float = 1.0,
) -> list[dict[str, Any]]:
    """Return chunks that overlap any proposed window by ≥ min_overlap seconds."""
    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for ch in chunks:
        cs, ce = float(ch["start"]), float(ch["end"])
        for w in windows:
            lo = max(cs, float(w["start"]))
            hi = min(ce, float(w["end"]))
            if hi - lo >= min_overlap:
                idx = int(ch["index"])
                if idx not in seen:
                    seen.add(idx)
                    out.append(dict(ch))
                break
    out.sort(key=lambda c: int(c["index"]))
    return out


def cue_in_interval(
    hits: Sequence[CueHit] | Sequence[Mapping[str, Any]],
    start: float,
    end: float,
    *,
    pad: float = 5.0,
    strong_only: bool = False,
) -> bool:
    lo, hi = start - pad, end + pad
    for h in hits:
        if isinstance(h, CueHit):
            hs, he, strength = h.start, h.end, h.strength
        else:
            hs, he = float(h["start"]), float(h["end"])
            strength = str(h.get("strength") or "strong")
        if strong_only and strength != "strong":
            continue
        if he >= lo and hs <= hi:
            return True
    return False


def refine_boundaries_heuristic(
    cues: Sequence[Mapping[str, Any]],
    intervals: Sequence[tuple[float, float]],
    hits: Sequence[CueHit] | None = None,
    *,
    max_span: float = 95.0,
    search_end_pad: float = 5.0,
    min_keep_end: float | None = None,
) -> list[tuple[float, float]]:
    """Snap start to cue hit; expand/snap end to return-to-content / topic shift / cap.

    Never shrinks *end* below the input interval end (confirmed content), unless an
    explicit end-of-ad cue appears earlier — and even then keeps at least 15s span.
    """
    if not intervals:
        return []
    hit_list = list(hits) if hits is not None else find_cue_hits(cues, include_weak=False)
    refined: list[tuple[float, float]] = []

    for a, b in intervals:
        cue_starts = [
            h.start
            for h in hit_list
            if h.strength == "strong" and (a - 12.0) <= h.start <= (b + 2.0)
        ]
        if cue_starts:
            na = min(cue_starts)
        else:
            na = a
            for c in cues:
                if float(c["end"]) >= a and float(c["start"]) <= a + 15:
                    na = float(c["start"])
                    break

        search_hi = min(na + max_span, max(b, na) + 40.0)
        end_cue_t: float | None = None
        for c in cues:
            cs = float(c["start"])
            if cs < na + 15.0:
                continue
            if cs > search_hi:
                break
            text = str(c.get("text") or "")
            if any(rx.search(text) for rx in END_PATTERNS):
                end_cue_t = cs
                break

        # Keep confirmed body (b); optionally extend to end-cue or max_span
        nb = max(b, na + 15.0)
        if end_cue_t is not None and end_cue_t > na + 15.0:
            # Prefer end-cue if it is after most of the confirmed body
            if end_cue_t >= b - 5.0:
                nb = end_cue_t
            else:
                # end-cue early (false "anyway") — keep confirmed end, still cap
                nb = min(max(b, end_cue_t), na + max_span)
        else:
            nb = min(max(b, na + 25.0), na + max_span)
            # Snap to last cue end inside span
            last_end = None
            for c in cues:
                if float(c["start"]) < na:
                    continue
                if float(c["start"]) > nb:
                    break
                last_end = float(c["end"])
            if last_end is not None:
                nb = max(nb, last_end) if last_end <= na + max_span else last_end
                nb = min(nb, na + max_span)

        if nb > na + 3.0:
            refined.append((na, nb))
        else:
            refined.append((a, b))

    return merge_intervals(refined, gap=8.0)


def expand_confirmed_windows(
    windows: Sequence[Mapping[str, Any]],
    labeled: Sequence[Mapping[str, Any]],
    cues: Sequence[Mapping[str, Any]],
    hits: Sequence[CueHit],
    *,
    sponsor_labels: frozenset[str] = frozenset({"sponsor"}),
    max_span: float = 95.0,
) -> list[tuple[float, float]]:
    """If any chunk inside a proposed window is confirmed sponsor, take cue→end span.

    Handles narrative/sketch mid-rolls where only the CTA chunks look like ads to Jev.
    """
    accepted: list[tuple[float, float]] = []
    for w in windows:
        ws, we = float(w["start"]), float(w["end"])
        confirmed = [
            row
            for row in labeled
            if row["label"] in sponsor_labels
            and float(row["end"]) >= ws
            and float(row["start"]) <= we
        ]
        # Also treat strong-cue-bearing chunks as confirmation anchors
        strong_in_window = [
            h for h in hits if h.strength == "strong" and ws - 2 <= h.start <= we + 2
        ]
        if not confirmed and not strong_in_window:
            continue
        if not confirmed:
            # Cue alone is not enough — require Jev sponsor somewhere in window
            continue

        # Anchor start at earliest strong cue (or window start)
        if strong_in_window:
            na = min(h.start for h in strong_in_window)
        else:
            na = min(float(r["start"]) for r in confirmed)

        body_end = max(float(r["end"]) for r in confirmed)
        # Find return-to-content after body
        search_hi = na + max_span
        end_cue_t = None
        for c in cues:
            cs = float(c["start"])
            if cs < max(na + 15.0, body_end - 10.0):
                continue
            if cs > search_hi:
                break
            text = str(c.get("text") or "")
            if any(rx.search(text) for rx in END_PATTERNS):
                end_cue_t = cs
                break
        if end_cue_t is not None and end_cue_t > body_end - 5:
            nb = end_cue_t
        else:
            nb = min(max(body_end, na + 25.0), na + max_span)
            # Include trailing CTA cues (% off / link) still inside search_hi
            for c in cues:
                cs, ce = float(c["start"]), float(c["end"])
                if cs < body_end - 2 or cs > search_hi:
                    continue
                t = str(c.get("text") or "")
                if re.search(
                    r"(\d+\s*%\s*off|link (down below|in)|with the link|promo code|discount)",
                    t,
                    re.I,
                ):
                    nb = max(nb, ce)
            nb = min(nb, na + max_span)
        if nb > na + 3:
            accepted.append((na, nb))
    return merge_intervals(accepted, gap=10.0)


def pattern_hit_summary(hits: Sequence[CueHit]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for h in hits:
        counts[h.pattern_name] = counts.get(h.pattern_name, 0) + 1
    return counts

# ---------------------------------------------------------------------------
# Refined boundary snap + confidence gate (English regex_jev_refined)
# ---------------------------------------------------------------------------

# Lead-in / paid-read openers (prefer these for segment *start*).
LEAD_IN_PATTERN_NAMES = frozenset(
    {
        "sponsored_by",
        "thanks_to_sponsor",
        "video_sponsored",
        "brought_to_you",
        "paid_partnership",
        "segue_sponsor",
        "segue_to_our",
        "todays_sponsor",
        "our_sponsor",
    }
)

# Offer / CTA closers — after these, cap length if no explicit end cue.
OFFER_CTA_PATTERN_NAMES = frozenset(
    {
        "use_code",
        "promo_code",
        "discount_code",
        "pct_off",
        "link_in_desc",
        "link_down_below",
        "with_the_link",
        "check_out_at",
    }
)

# Extra end-of-ad cues beyond END_PATTERNS (compiled below with the base set).
_REFINED_END_EXTRA = [
    r"\bnow[,.]?\s+(back\s+)?to\b",
    r"\bbefore we go (any )?(further|on)?\b",
    r"\b(alright|all right)[,.]?\s+(so|now|back)\b",
    r"\bthat rhymed\b",
    r"\bthanks (again )?to .{0,40} for sponsor",
]

_REFINED_END_PATTERNS: list[re.Pattern[str]] = END_PATTERNS + [
    re.compile(p, re.I) for p in _REFINED_END_EXTRA
]

# Soft lead-in language immediately before a strong cue (include if adjacent).
_SOFT_LEAD_IN = re.compile(
    r"\b("
    r"speaking of|which brings me|that brings us|brings me to|"
    r"before we (continue|move on|get back)|"
    r"real quick|quick (word|shout|thanks)|"
    r"just like .{0,40} (on this|with this)|"
    r"what I do know is|and are we going to|"
    r"now[,.]?\s+if you('ll| will) excuse|"
    r"pop in this segue"
    r")\b",
    re.I,
)


def _cue_window_text(
    cues: Sequence[Mapping[str, Any]], index: int, *, neighbors: int = 1
) -> str:
    """Join cue text with the next *neighbors* cues (ASR often splits phrases)."""
    parts: list[str] = []
    for j in range(index, min(len(cues), index + 1 + neighbors)):
        parts.append(str(cues[j].get("text") or ""))
    return " ".join(parts)


def _iter_end_cue_times(
    cues: Sequence[Mapping[str, Any]],
    *,
    after: float,
    until: float,
) -> list[float]:
    """Return start times of end-of-ad cues (single or adjacent-cue match).

    If the phrase only completes on the *next* cue (ASR split), use that cue's
    start so we don't end at an earlier offer/link cue.
    """
    times: list[float] = []
    for i, c in enumerate(cues):
        cs = float(c["start"])
        if cs > until:
            break
        alone = str(c.get("text") or "")
        window = _cue_window_text(cues, i, neighbors=1)
        matched_alone = any(rx.search(alone) for rx in _REFINED_END_PATTERNS)
        matched_window = any(rx.search(window) for rx in _REFINED_END_PATTERNS)
        if not matched_window:
            continue
        if matched_alone:
            t = cs
        elif i + 1 < len(cues):
            t = float(cues[i + 1]["start"])
        else:
            t = cs
        if after <= t <= until:
            times.append(t)
    # de-dupe preserving order
    out: list[float] = []
    seen: set[float] = set()
    for t in times:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _soft_lead_in_start(
    cues: Sequence[Mapping[str, Any]],
    anchor_start: float,
    *,
    lookback: float = 25.0,
) -> float | None:
    """If a soft lead-in cue sits just before *anchor_start*, return its start."""
    best: float | None = None
    for c in cues:
        cs, ce = float(c["start"]), float(c["end"])
        if ce < anchor_start - lookback:
            continue
        if cs >= anchor_start:
            break
        # Must be close to the anchor (same breath / adjacent roll-up)
        if anchor_start - ce > 8.0 and anchor_start - cs > 15.0:
            continue
        text = str(c.get("text") or "")
        if _SOFT_LEAD_IN.search(text):
            best = cs if best is None else min(best, cs)
    return best


def snap_boundaries_refined(
    cues: Sequence[Mapping[str, Any]],
    intervals: Sequence[tuple[float, float]],
    hits: Sequence[CueHit] | None = None,
    *,
    lead_in_lookback: float = 30.0,
    offer_pad_sec: float = 8.0,
    max_span: float = 100.0,
    min_span: float = 8.0,
) -> list[tuple[float, float]]:
    """Conservative start/end snap after a confirmed sponsor window.

    Start: prefer clear lead-in / first strong CTA within lookback; never pull
    start earlier on weak guesses (Tony KEEP_CONTENT — start slightly late if
    unsure). End: snap to return-to-content cues, else cap shortly after the
    last offer CTA (+offer_pad_sec); prefer ending slightly late over cutting
    the ad short.
    """
    if not intervals:
        return []
    hit_list = list(hits) if hits is not None else find_cue_hits(cues, include_weak=False)
    refined: list[tuple[float, float]] = []

    for a, b in intervals:
        # --- Start ---
        lead_hits = [
            h
            for h in hit_list
            if h.strength == "strong"
            and h.pattern_name in LEAD_IN_PATTERN_NAMES
            and (a - lead_in_lookback) <= h.start <= (b + 2.0)
        ]
        cta_hits = [
            h
            for h in hit_list
            if h.strength == "strong"
            and (a - 5.0) <= h.start <= (b + 2.0)
        ]
        if lead_hits:
            # Earliest lead-in near the window (don't start mid-pitch)
            na = min(h.start for h in lead_hits)
            soft = _soft_lead_in_start(cues, na, lookback=lead_in_lookback)
            if soft is not None and soft >= a - lead_in_lookback:
                # Only pull earlier when soft lead-in is clear and close
                na = soft
        elif cta_hits:
            # First strong CTA — start slightly late into the read if no lead-in
            na = min(h.start for h in cta_hits)
        else:
            na = a

        # Never start earlier than lookback before original confirmed start
        na = max(na, a - lead_in_lookback)
        # Never start later than original confirmed body by > ~20s (keep coverage)
        if na > a + 20.0:
            na = a

        # --- End ---
        search_hi = min(na + max_span, max(b, na) + 45.0)
        end_times = _iter_end_cue_times(cues, after=na + min_span, until=search_hi)

        # Offer CTA anchors use cue *start* (YouTube roll-up cue ends inflate).
        _OFFER_RX = re.compile(
            r"("
            r"\buse (code|promo)\b|\bpromo code\b|\bdiscount code\b|"
            r"\b\d+\s*%\s*off\b|"
            r"\blink (down below|in (the )?description|below)\b|"
            r"\blinks? below\b|\bwith the link\b|"
            r"\bbook (a |that )?demo\b|\bgo to \w+\.(com|net|io)\b"
            r")",
            re.I,
        )
        offer_starts: list[float] = []
        for h in hit_list:
            if h.pattern_name not in OFFER_CTA_PATTERN_NAMES:
                continue
            if h.start < na + 5.0 or h.start > search_hi:
                continue
            offer_starts.append(h.start)
        for i, c in enumerate(cues):
            cs = float(c["start"])
            if cs < na + 5.0 or cs > search_hi:
                continue
            window = _cue_window_text(cues, i, neighbors=1)
            if _OFFER_RX.search(window):
                offer_starts.append(cs)

        # Core span floor — do not trust an inflated *b* when cues say the ad ended
        core_lo = na + min_span
        last_offer = max(offer_starts) if offer_starts else None

        # Prefer return-to-content cue after the offer (or mid-window). Allows
        # pulling *back* from an overshot confirmed end (common FP).
        chosen_end: float | None = None
        end_floor = core_lo
        if last_offer is not None:
            end_floor = max(core_lo, last_offer - 2.0)
        for et in end_times:
            if et >= end_floor:
                chosen_end = et
                break

        if chosen_end is not None:
            nb = chosen_end
        elif last_offer is not None:
            # Cap shortly after offer CTA *start* (not roll-up end)
            nb = last_offer + offer_pad_sec
            # Prefer slightly late vs cutting the offer: allow up to +4s past pad
            # if original confirmed end is nearby; otherwise pull back hard.
            if b <= nb + 4.0:
                nb = max(nb, min(b, last_offer + offer_pad_sec + 4.0))
            # Never keep a huge overshoot past the offer
            if b > last_offer + offer_pad_sec + 4.0:
                nb = last_offer + offer_pad_sec
        else:
            # No cue: keep confirmed end (slightly late), capped
            nb = min(max(b, na + 20.0), na + max_span)

        nb = min(max(nb, core_lo), na + max_span)
        if nb > na + 3.0:
            refined.append((na, nb))
        else:
            refined.append((a, b))

    return merge_intervals(refined, gap=8.0)


def segment_confidence_from_labels(
    labeled: Sequence[Mapping[str, Any]],
    start: float,
    end: float,
    *,
    sponsor_labels: frozenset[str] = frozenset({"sponsor"}),
) -> float | None:
    """Per-segment confidence from overlapping Jev Choice rows.

    Uses max of ``probabilities[label]`` when present, else ``confidence``.
    """
    scores: list[float] = []
    for row in labeled:
        if row.get("label") not in sponsor_labels:
            continue
        rs, re_ = float(row["start"]), float(row["end"])
        if re_ < start or rs > end:
            continue
        probs = row.get("probabilities")
        label = str(row["label"])
        score: float | None = None
        if isinstance(probs, Mapping) and label in probs:
            try:
                score = float(probs[label])
            except (TypeError, ValueError):
                score = None
        if score is None and row.get("confidence") is not None:
            try:
                score = float(row["confidence"])
            except (TypeError, ValueError):
                score = None
        if score is not None:
            scores.append(score)
    if not scores:
        return None
    return max(scores)


def attach_confidence_gate(
    intervals: Sequence[tuple[float, float]],
    labeled: Sequence[Mapping[str, Any]],
    *,
    confidence_threshold: float = 0.75,
    category: str = "sponsor",
) -> list[dict[str, Any]]:
    """Build segment dicts with confidence + auto_skip / needs_confirm flags.

    Segments below *confidence_threshold* are still returned but marked
    ``auto_skip=False`` / ``needs_confirm=True`` (ask the user).
    """
    segs: list[dict[str, Any]] = []
    for a, b in intervals:
        conf = segment_confidence_from_labels(labeled, a, b)
        # Missing confidence → treat as needs confirm (conservative)
        auto = conf is not None and conf >= confidence_threshold
        segs.append(
            {
                "start": a,
                "end": b,
                "category": category,
                "confidence": conf,
                "auto_skip": auto,
                "needs_confirm": not auto,
            }
        )
    return segs


def confidence_gate_summary(segments: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(segments)
    n_auto = sum(1 for s in segments if s.get("auto_skip"))
    confs = [
        float(s["confidence"])
        for s in segments
        if s.get("confidence") is not None
    ]
    return {
        "n_segments": n,
        "n_auto_skip": n_auto,
        "n_needs_confirm": n - n_auto,
        "auto_skip_rate": (n_auto / n) if n else None,
        "mean_confidence": (sum(confs) / len(confs)) if confs else None,
        "min_confidence": min(confs) if confs else None,
    }
