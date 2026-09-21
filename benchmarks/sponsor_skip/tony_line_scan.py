"""Tony-style line-ID sponsor scan (Python port of youtube-sponsor-detection/src/jev.js).

Stages:
  1. Scan overlapping ~80-line windows (Noul: begin here? Choice: naming line).
  2. Refine: confirm + naming line + end line; optional lead-in traceback.
  3. Map line IDs → seconds (phrase cut omitted for cost; line boundaries stand).

Sponsor definition matches Tony: lead-in + pitch + offer; merch/selfpromo/subscribe ≠ sponsor.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping, Sequence

from typesafe_sdk import Choice, Noul

from tony_transcript import (
    Line,
    WINDOW_LINES,
    build_lines,
    estimate_tokens,
    render_lines,
    window_lines,
)

FOUND = 0.7
MAYBE = 0.35
REFINE_BEFORE = 45
REFINE_AFTER = 40
NO_START = "none"
RUNS_PAST_EXCERPT = "continues_past_excerpt"
MAX_SEGMENTS = 6
BLIND_MASK_LINES = 12
SCAN_WORKERS = 4

SPONSOR_DEF = (
    "A sponsor segment is the part of a video that exists to promote a third party "
    "that paid for placement: a product, service, app or company. It is usually "
    "read by the creator."
)
SPONSOR_SHAPE = (
    "A sponsor segment normally has three parts, and it begins with the first one: "
    "1. a lead-in, where the creator leaves the subject of the video and starts a "
    "story, anecdote, problem, question, joke or 'quick break' whose only purpose "
    "is to arrive at the sponsor; "
    "2. the pitch, where the sponsor or its product is named and described; "
    "3. the offer, with a link, discount code, free trial, QR code or "
    "'link in the description'. "
    "The lead-in can last several minutes and can sound like normal content until "
    "the sponsor is named. It still belongs to the sponsor segment from its first line."
)
NOT_SPONSOR = (
    "Not a sponsor: creator promoting their own merchandise, membership, Patreon, "
    "newsletter, courses or other videos; asking viewers to like/comment/subscribe; "
    "thanking viewers/patrons/crew; content that stays on the subject of the video."
)

SPONSOR_CTX = {
    "definition": SPONSOR_DEF,
    "shape": SPONSOR_SHAPE,
    "not_a_sponsor_segment": NOT_SPONSOR,
}


def _best_label(probabilities: Mapping[str, float] | None, allowed: set[str]) -> tuple[str | None, float]:
    if not probabilities:
        return None, 0.0
    best_id: str | None = None
    best = -1.0
    for label, p in probabilities.items():
        if label not in allowed:
            continue
        pf = float(p)
        if pf > best:
            best = pf
            best_id = label
    return best_id, (0.0 if best < 0 else best)


def _probs(ans: Any) -> dict[str, float]:
    probs = getattr(ans, "probabilities", None)
    if probs is None:
        return {}
    if not isinstance(probs, dict):
        probs = dict(probs)
    return {str(k): float(v) for k, v in probs.items()}


def _noul(ans: Any) -> float:
    v = getattr(ans, "noul", None)
    return float(v) if v is not None else 0.0


def _get_ans(response: Any, name: str) -> Any:
    answers = getattr(response, "answers", None)
    if isinstance(answers, Mapping) and name in answers:
        return answers[name]
    for group_name in ("choices", "nouls", "scores"):
        group = getattr(response, group_name, None)
        if isinstance(group, Mapping) and name in group:
            return group[name]
    raise KeyError(name)


def _line_criteria(lines: Sequence[Line], *, extra: dict[str, str] | None = None) -> dict[str, str | None]:
    opts: dict[str, str | None] = {ln.id: (ln.text[:120] or None) for ln in lines}
    if extra:
        opts.update(extra)
    return opts


def scan_questions(lines: Sequence[Line]) -> dict[str, Any]:
    opts = _line_criteria(
        lines,
        extra={NO_START: "No line in this excerpt names a sponsor, its product or its offer."},
    )
    return {
        "sponsor_starts_here": Noul(
            instructions={
                "question": (
                    "Does a sponsor segment begin somewhere in this excerpt of "
                    "the video transcript?"
                ),
                **SPONSOR_CTX,
            },
            criteria={
                "true": (
                    "Somewhere in this excerpt the creator leaves the subject of "
                    "the video and starts a sponsor segment: a lead-in, a pitch or "
                    "an offer for a paying third party."
                ),
                "false": (
                    "No sponsor segment begins in this excerpt. Either it is all "
                    "regular content, or a sponsor segment that started before "
                    "this excerpt is still running through it."
                ),
            },
        ),
        "anchor_line": Choice(
            instructions={
                "question": (
                    "Which labelled line is the first line that names the sponsor, "
                    "its product, or its offer? For example 'thanks to X for "
                    "sponsoring', 'X is an app that', 'today's video is brought to "
                    "you by X', or a discount code or link for X. Choose the first "
                    "such line, not the lead-in before it."
                ),
                **SPONSOR_CTX,
            },
            criteria=opts,
        ),
    }


def anchor_questions(lines: Sequence[Line]) -> dict[str, Any]:
    anchor_opts = _line_criteria(
        lines,
        extra={NO_START: "No line in this excerpt names a sponsor, its product or its offer."},
    )
    end_opts = _line_criteria(
        lines,
        extra={
            RUNS_PAST_EXCERPT: "The sponsor segment is still running at the end of this excerpt.",
            NO_START: "This excerpt contains no sponsor segment.",
        },
    )
    return {
        "has_sponsor": Noul(
            instructions={
                "question": "Does this excerpt of the video transcript contain a sponsor segment?",
                **SPONSOR_CTX,
            },
            criteria={
                "true": (
                    "A sponsor segment for a paying third party is in this excerpt: "
                    "its lead-in, its pitch, its offer, or all three."
                ),
                "false": "This excerpt is the video's own content with no sponsor segment in it.",
            },
        ),
        "anchor_line": Choice(
            instructions={
                "question": (
                    "Which labelled line is the first line that names the sponsor, "
                    "its product, or its offer? Choose the first such line, not "
                    "the lead-in before it."
                ),
                **SPONSOR_CTX,
            },
            criteria=anchor_opts,
        ),
        "end_line": Choice(
            instructions={
                "question": (
                    "Which labelled line is the LAST line of the sponsor segment: "
                    "the final line of the pitch or the offer, after which the "
                    "creator returns to the video's own content, signs off, or "
                    "the video ends?"
                ),
                **SPONSOR_CTX,
            },
            criteria=end_opts,
        ),
    }


def start_questions(lines: Sequence[Line]) -> dict[str, Any]:
    opts = _line_criteria(lines)
    return {
        "start_line": Choice(
            instructions={
                "question": (
                    "The sponsor is named on the line given in `sponsor_named_at`. "
                    "Reading backwards from that line, which labelled line is the "
                    "FIRST line of the sponsor segment: the moment the creator "
                    "leaves the subject of the video (`video_title`) and begins "
                    "the lead-in that ends at the sponsor?"
                ),
                "rules": [
                    "The lead-in belongs to the sponsor segment from its first line, "
                    "even when it sounds like a personal story and the sponsor is "
                    "only named later.",
                    "A lead-in exists to arrive at the sponsor.",
                    "Lines still about the subject of the video are not part of the "
                    "sponsor segment.",
                    "Wrapping up, thanking hosts/viewers, and like/subscribe CTAs "
                    "belong to the video, not the sponsor segment.",
                    "If there is no lead-in, choose the line given in sponsor_named_at.",
                ],
                **SPONSOR_CTX,
            },
            criteria=opts,
        )
    }


def looks_like_sponsor(presence: float, p_none: float) -> float:
    return max(float(presence), 1.0 - float(p_none))


def detect_tony_line_scan(
    detector: Any,
    cues: Sequence[Mapping[str, Any]],
    *,
    title: str = "unknown",
    max_segments: int = MAX_SEGMENTS,
    scan_workers: int = SCAN_WORKERS,
    trace_lead_in: bool = True,
) -> dict[str, Any]:
    """Full Tony-style pipeline; uses detector._call for token accounting."""
    lines = build_lines(cues)
    if not lines:
        return {
            "mode": "tony_line_scan",
            "status": "no-transcript",
            "segments": [],
            "intervals": [],
            "n_lines": 0,
            "windows_scanned": 0,
            "traces": [],
        }

    line_by_id = {ln.id: ln for ln in lines}
    windows = window_lines(lines, size=WINDOW_LINES)
    traces: list[dict[str, Any]] = []
    call_lock = threading.Lock()

    def ask(state: dict[str, Any], questions: dict[str, Any], *, tag: str) -> Any:
        with call_lock:
            return detector._call(state, questions, tag=tag)

    def scan_one(window: list[Line], index: int, total: int) -> dict[str, Any]:
        state = {
            "video_title": title,
            "video_transcript_excerpt": render_lines(window),
            "excerpt_position": f"part {index + 1} of {total} of the video",
        }
        result = ask(state, scan_questions(window), tag=f"tony_scan_{index}")
        presence = _noul(_get_ans(result, "sponsor_starts_here"))
        probs = _probs(_get_ans(result, "anchor_line"))
        allowed = {ln.id for ln in window}
        pick_id, pick_p = _best_label(probs, allowed)
        return {
            "index": index,
            "lines": window,
            "from": window[0].start,
            "to": window[-1].end,
            "presence": presence,
            "p_none": float(probs.get(NO_START, 0.0)),
            "start_line_id": pick_id,
            "start_line_probability": pick_p,
            "estimated_state_tokens": estimate_tokens(state["video_transcript_excerpt"]),
        }

    # Stage 1 — parallel scans
    scans: list[dict[str, Any] | None] = [None] * len(windows)
    if scan_workers <= 1 or len(windows) <= 1:
        for i, w in enumerate(windows):
            scans[i] = scan_one(w, i, len(windows))
    else:
        with ThreadPoolExecutor(max_workers=min(scan_workers, len(windows))) as pool:
            futs = {
                pool.submit(scan_one, w, i, len(windows)): i
                for i, w in enumerate(windows)
            }
            for fut in as_completed(futs):
                scans[futs[fut]] = fut.result()
    scans_list = [s for s in scans if s is not None]

    segments: list[dict[str, Any]] = []
    taken: set[str] = set()

    def refine(winner: dict[str, Any]) -> dict[str, Any] | None:
        centre = next(
            (i for i, ln in enumerate(lines) if ln.id == winner["start_line_id"]),
            -1,
        )
        if centre < 0:
            return None
        from_i = max(0, centre - REFINE_BEFORE)
        to_i = min(len(lines), centre + REFINE_AFTER)
        slice_lines = [ln for ln in lines[from_i:to_i] if ln.id not in taken]
        if len(slice_lines) < 2:
            return None

        position = (
            f"an excerpt from the video, starting around "
            f"{round(slice_lines[0].start)} seconds in"
        )
        anchored = ask(
            {
                "video_title": title,
                "video_transcript_excerpt": render_lines(slice_lines),
                "excerpt_position": position,
            },
            anchor_questions(slice_lines),
            tag="tony_anchor",
        )
        presence = _noul(_get_ans(anchored, "has_sponsor"))
        if presence < MAYBE:
            return None

        allowed = {ln.id for ln in slice_lines}
        by_id = {ln.id: ln for ln in slice_lines}
        anchor_pick, anchor_p = _best_label(_probs(_get_ans(anchored, "anchor_line")), allowed)
        end_probs = _probs(_get_ans(anchored, "end_line"))
        end_pick, end_p = _best_label(end_probs, allowed)
        end_runs_on = float(end_probs.get(RUNS_PAST_EXCERPT, 0.0))

        anchor_line = (
            by_id.get(anchor_pick or "")
            or by_id.get(winner["start_line_id"] or "")
            or slice_lines[0]
        )
        anchor_index = slice_lines.index(anchor_line)

        start_line = anchor_line
        start_p = anchor_p
        if trace_lead_in:
            before = slice_lines[: anchor_index + 1]
            if len(before) > 1:
                traced = ask(
                    {
                        "video_title": title,
                        "video_transcript_excerpt": render_lines(before),
                        "sponsor_named_at": anchor_line.id,
                        "sponsor_named_at_text": anchor_line.text,
                        "excerpt_position": position,
                    },
                    start_questions(before),
                    tag="tony_leadin",
                )
                start_pick, start_p2 = _best_label(
                    _probs(_get_ans(traced, "start_line")),
                    {ln.id for ln in before},
                )
                if start_pick and start_pick in by_id:
                    start_line = by_id[start_pick]
                    start_p = start_p2

        end_line = by_id.get(end_pick or "") if end_pick else None
        end_ok = (
            end_line is not None
            and end_runs_on < end_p
            and end_line.end > start_line.start
        )

        start_index = slice_lines.index(start_line)
        if end_ok and end_line is not None:
            end_index = slice_lines.index(end_line)
        else:
            end_index = min(len(slice_lines) - 1, anchor_index + BLIND_MASK_LINES)
        if end_index < start_index:
            end_index = start_index

        covered = slice_lines[start_index : end_index + 1]
        line_ids = [ln.id for ln in covered]
        conf = min(
            looks_like_sponsor(winner["presence"], winner["p_none"]),
            presence,
        )
        return {
            "confidence": conf,
            "scan_presence": winner["presence"],
            "refine_presence": presence,
            "start": {
                "line_id": start_line.id,
                "seconds": start_line.start,
                "text": start_line.text[:200],
                "probability": start_p,
            },
            "anchor": {
                "line_id": anchor_line.id,
                "seconds": anchor_line.start,
                "text": anchor_line.text[:200],
                "probability": anchor_p,
            },
            "end": {
                "line_id": covered[-1].id,
                "seconds": covered[-1].end,
                "text": covered[-1].text[:200],
                "probability": end_p if end_ok else None,
                "runs_past_excerpt": end_runs_on,
                "end_ok": end_ok,
            },
            "line_ids": line_ids,
            "t_start": start_line.start,
            "t_end": covered[-1].end,
        }

    while len(segments) < max_segments:
        candidates = [
            s
            for s in scans_list
            if looks_like_sponsor(s["presence"], s["p_none"]) >= MAYBE
            and s.get("start_line_id")
            and s["start_line_id"] not in taken
        ]
        if not candidates:
            break
        winner = max(
            candidates,
            key=lambda s: looks_like_sponsor(s["presence"], s["p_none"]),
        )
        segment = refine(winner)
        if segment:
            segments.append(segment)
            for lid in segment["line_ids"]:
                taken.add(lid)
            traces.append(
                {
                    "window": winner["index"],
                    "anchor": segment["anchor"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "confidence": segment["confidence"],
                }
            )

        remaining = [ln for ln in winner["lines"] if ln.id not in taken]
        if not segment or len(remaining) < 3:
            winner["presence"] = 0.0
            winner["p_none"] = 1.0
            winner["start_line_id"] = None
            continue
        # Rescan remaining lines in this window
        rescan = scan_one(remaining, winner["index"], len(windows))
        winner.update(rescan)
        winner["lines"] = remaining

    segments.sort(key=lambda s: s["t_start"])
    intervals = [(float(s["t_start"]), float(s["t_end"])) for s in segments]
    status = (
        "not-found"
        if not segments
        else ("found" if any(s["confidence"] >= FOUND for s in segments) else "uncertain")
    )

    return {
        "mode": "tony_line_scan",
        "status": status,
        "n_lines": len(lines),
        "windows_scanned": len(windows),
        "traces": traces,
        "tony_segments": segments,
        "segments": [
            {"start": a, "end": b, "category": "sponsor"} for a, b in intervals
        ],
        "intervals": intervals,
        "line_index_preview": [
            {"id": ln.id, "start": ln.start, "end": ln.end, "text": ln.text[:80]}
            for ln in lines[:5]
        ],
    }
