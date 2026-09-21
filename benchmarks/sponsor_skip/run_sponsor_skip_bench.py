#!/usr/bin/env python3
"""Live experiment: detect YouTube sponsor segments with TypeSafe Jev Choice.

Compares chunk-level Choice (primary) and paragraph start/end Choice against
SponsorBlock gold labels. Does NOT train a classifier.

Usage (from repo root, venv active):
  python benchmarks/sponsor_skip/run_sponsor_skip_bench.py
  python benchmarks/sponsor_skip/run_sponsor_skip_bench.py --limit 6 --modes chunk_batched paragraph
  python benchmarks/sponsor_skip/run_sponsor_skip_bench.py --prepare-only
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
REPO_ROOT = BENCH_DIR.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from load_card_secrets import load_card_secrets  # noqa: E402
from metrics import TYPESAFE_INPUT_USD_PER_MILLION  # noqa: E402

from data_io import (  # noqa: E402
    CANDIDATE_VIDEO_IDS,
    chunk_cues,
    cues_to_sentences,
    download_vtt,
    fetch_sponsorblock,
    gold_to_dict,
    parse_vtt,
    save_json,
    load_json,
)
from detect import SponsorDetector  # noqa: E402
from interval_metrics import (  # noqa: E402
    content_false_positive_rate,
    segment_metrics,
)

DATA_DIR = HERE / "data"
TRANS_DIR = DATA_DIR / "transcripts"
GOLD_DIR = DATA_DIR / "gold"
RESULTS_DIR = HERE / "results"


def prepare_video(video_id: str, *, min_sponsor_sec: float = 15.0) -> dict[str, Any] | None:
    """Fetch gold + transcript; return prepared payload or None if unusable."""
    gold = fetch_sponsorblock(video_id, categories=("sponsor", "selfpromo"))
    sponsor_only = [g for g in gold if g.category == "sponsor"]
    # Prefer videos with a real sponsor segment; allow selfpromo-only as fallback
    primary = sponsor_only or gold
    if not primary:
        return None
    if not any((g.end - g.start) >= min_sponsor_sec for g in primary):
        return None

    vtt = download_vtt(video_id, TRANS_DIR)
    if vtt is None:
        return None
    cues = parse_vtt(vtt)
    if len(cues) < 10:
        return None

    # Persist raw
    save_json(
        GOLD_DIR / f"{video_id}.json",
        {
            "video_id": video_id,
            "segments": gold_to_dict(gold),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    save_json(
        TRANS_DIR / f"{video_id}.cues.json",
        {
            "video_id": video_id,
            "vtt_file": vtt.name,
            "cues": [{"start": c.start, "end": c.end, "text": c.text} for c in cues],
        },
    )

    chunks = chunk_cues(cues, target_sec=15.0, max_sec=22.0)
    sentences = cues_to_sentences(cues, max_sentences=255)
    duration = max(c.end for c in cues)

    return {
        "video_id": video_id,
        "duration_sec": duration,
        "n_cues": len(cues),
        "n_chunks": len(chunks),
        "n_sentences": len(sentences),
        "gold": gold_to_dict(gold),
        "gold_sponsor": gold_to_dict(sponsor_only) if sponsor_only else gold_to_dict(gold),
        "chunks": chunks,
        "sentences": sentences,
        "cues": [{"start": c.start, "end": c.end, "text": c.text} for c in cues],
        "vtt_file": vtt.name,
    }


def load_prepared_from_cache(video_id: str) -> dict[str, Any] | None:
    """Rebuild prepared payload from cached gold + cues (no network)."""
    from data_io import Cue, GoldSegment, chunk_cues, cues_to_sentences, gold_to_dict, load_json as _lj

    gold_path = GOLD_DIR / f"{video_id}.json"
    cues_path = TRANS_DIR / f"{video_id}.cues.json"
    if not gold_path.is_file() or not cues_path.is_file():
        return None
    gold_doc = _lj(gold_path)
    cues_doc = _lj(cues_path)
    gold_raw = gold_doc.get("segments") or []
    segs = [
        GoldSegment(
            start=float(g["start"]),
            end=float(g["end"]),
            category=str(g.get("category") or "sponsor"),
            uuid=g.get("uuid"),
        )
        for g in gold_raw
    ]
    sponsor_only = [g for g in segs if g.category == "sponsor"]
    primary = sponsor_only or segs
    if not primary:
        return None
    cues = [
        Cue(start=float(c["start"]), end=float(c["end"]), text=str(c["text"]))
        for c in cues_doc["cues"]
    ]
    if len(cues) < 10:
        return None
    chunks = chunk_cues(cues, target_sec=15.0, max_sec=22.0)
    sentences = cues_to_sentences(cues, max_sentences=255)
    duration = max(c.end for c in cues)
    return {
        "video_id": video_id,
        "duration_sec": duration,
        "n_cues": len(cues),
        "n_chunks": len(chunks),
        "n_sentences": len(sentences),
        "gold": gold_to_dict(segs),
        "gold_sponsor": gold_to_dict(sponsor_only) if sponsor_only else gold_to_dict(segs),
        "chunks": chunks,
        "sentences": sentences,
        "cues": [{"start": c.start, "end": c.end, "text": c.text} for c in cues],
        "vtt_file": cues_doc.get("vtt_file"),
        "from_cache": True,
    }


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def estimate_usd(input_tokens: int) -> float:
    return (input_tokens / 1_000_000.0) * TYPESAFE_INPUT_USD_PER_MILLION


def run_mode(
    detector: SponsorDetector,
    prepared: dict[str, Any],
    mode: str,
    *,
    refine: bool,
    confidence_threshold: float = 0.75,
) -> dict[str, Any]:
    gold_iv = [(g["start"], g["end"]) for g in prepared["gold_sponsor"]]
    # Also accept selfpromo in gold if we labeled sponsor_like
    if not gold_iv:
        gold_iv = [(g["start"], g["end"]) for g in prepared["gold"]]

    t0 = time.perf_counter()
    tok0 = detector.total_input_tokens
    lat0 = len(detector.latencies_ms)

    if mode == "chunk":
        out = detector.detect_chunk_choice(prepared["chunks"])
    elif mode == "chunk_batched":
        out = detector.detect_chunk_choice_batched(prepared["chunks"], batch_size=8)
    elif mode == "paragraph":
        out = detector.detect_paragraph_span(prepared["sentences"], max_regions=3)
    elif mode == "regex_propose_jev_confirm":
        out = detector.detect_regex_propose_jev_confirm(
            prepared["chunks"],
            prepared["cues"],
            batch_size=8,
            video_duration=prepared["duration_sec"],
        )
    elif mode == "regex_jev_refined":
        out = detector.detect_regex_jev_refined(
            prepared["chunks"],
            prepared["cues"],
            batch_size=8,
            video_duration=prepared["duration_sec"],
            confidence_threshold=confidence_threshold,
        )
    elif mode == "jev_then_regex_gate":
        out = detector.detect_jev_then_regex_gate(
            prepared["chunks"],
            prepared["cues"],
            batch_size=8,
        )
    elif mode == "tony_line_scan":
        out = detector.detect_tony_line_scan(
            prepared["cues"],
            title=prepared.get("video_id") or "unknown",
            max_segments=6,
            scan_workers=4,
            trace_lead_in=True,
        )
    else:
        raise ValueError(mode)

    intervals = list(out["intervals"])
    if refine and intervals and mode.startswith("chunk"):
        intervals = detector.refine_boundaries(prepared["cues"], intervals)
        out["intervals"] = intervals
        out["segments"] = [
            {"start": a, "end": b, "category": "sponsor"} for a, b in intervals
        ]
        out["refined"] = True
    else:
        out["refined"] = False

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    input_tokens = detector.total_input_tokens - tok0
    call_lats = detector.latencies_ms[lat0:]

    metrics = segment_metrics(intervals, gold_iv)
    metrics["content_fp_rate"] = content_false_positive_rate(
        intervals, gold_iv, prepared["duration_sec"]
    )
    metrics["latency_ms_total"] = elapsed_ms
    metrics["latency_ms_p50_calls"] = _pct(call_lats, 50)
    metrics["n_api_calls"] = len(call_lats)
    metrics["input_tokens"] = input_tokens
    metrics["estimated_usd"] = estimate_usd(input_tokens)

    # Confidence gate: all-preds metrics above; high-conf (auto-skip) subset below
    segs = out.get("segments") or []
    if segs and any("auto_skip" in s for s in segs if isinstance(s, dict)):
        auto_iv = [
            (float(s["start"]), float(s["end"]))
            for s in segs
            if s.get("auto_skip")
        ]
        hc = segment_metrics(auto_iv, gold_iv)
        hc["content_fp_rate"] = content_false_positive_rate(
            auto_iv, gold_iv, prepared["duration_sec"]
        )
        metrics["high_conf"] = {
            "iou": hc["iou"],
            "precision": hc["precision"],
            "recall": hc["recall"],
            "f1": hc["f1"],
            "overlap_sec": hc["overlap_sec"],
            "fp_sec": hc["fp_sec"],
            "content_fp_rate": hc["content_fp_rate"],
            "n_pred_segments": len(auto_iv),
        }
        gate = out.get("confidence_gate") or {}
        metrics["confidence_gate"] = gate
        metrics["confidence_threshold"] = out.get(
            "confidence_threshold", confidence_threshold
        )
    

    # Qualitative: pick up to 2 hit/miss chunk examples
    examples: dict[str, Any] = {"hits": [], "misses": []}
    if "chunk_labels" in out:
        for row in out["chunk_labels"]:
            mid = 0.5 * (row["start"] + row["end"])
            in_gold = any(a <= mid <= b for a, b in gold_iv)
            is_pred = row["label"] in {"sponsor", "selfpromo"}
            item = {
                "t": [round(row["start"], 1), round(row["end"], 1)],
                "label": row["label"],
                "conf": row.get("confidence"),
                "preview": row.get("text_preview"),
                "in_gold": in_gold,
            }
            if is_pred and in_gold and len(examples["hits"]) < 2:
                examples["hits"].append(item)
            elif is_pred and not in_gold and len(examples["misses"]) < 2:
                examples["misses"].append(item)
            elif (not is_pred) and in_gold and len(examples["misses"]) < 2:
                examples["misses"].append({**item, "kind": "false_negative_chunk"})

    return {
        "video_id": prepared["video_id"],
        "mode": mode,
        "gold": prepared["gold_sponsor"],
        "pred_segments": out["segments"],
        "metrics": metrics,
        "examples": examples,
        "raw": {
            "n_chunks": prepared["n_chunks"],
            "n_sentences": prepared["n_sentences"],
            "refined": out.get("refined"),
            "chunk_labels": out.get("chunk_labels"),
            "traces": out.get("traces"),
            "pattern_counts": out.get("pattern_counts"),
            "cue_hits": out.get("cue_hits"),
            "proposed_windows": out.get("proposed_windows"),
            "dropped_segments": out.get("dropped_segments"),
            "n_candidate_chunks": out.get("n_candidate_chunks"),
            "confidence_gate": out.get("confidence_gate"),
            "confidence_threshold": out.get("confidence_threshold"),
            "intervals_pre_snap": out.get("intervals_pre_snap"),
            "tony_status": out.get("status"),
            "tony_traces": out.get("traces"),
            "n_lines": out.get("n_lines"),
            "windows_scanned": out.get("windows_scanned"),
        },
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = ["iou", "precision", "recall", "f1", "mean_best_iou", "content_fp_rate"]
    agg: dict[str, Any] = {"n_videos": len(rows)}
    for k in keys:
        vals = [float(r["metrics"][k]) for r in rows if k in r["metrics"]]
        agg[k] = sum(vals) / len(vals) if vals else None
    # Video-level recall: among videos with gold, fraction with any overlap
    with_gold = [r for r in rows if r["metrics"].get("video_has_gold")]
    if with_gold:
        hits = sum(1 for r in with_gold if r["metrics"].get("video_detected") and r["metrics"]["overlap_sec"] > 0)
        agg["video_level_recall"] = hits / len(with_gold)
    else:
        agg["video_level_recall"] = None
    agg["total_input_tokens"] = sum(int(r["metrics"]["input_tokens"]) for r in rows)
    agg["total_estimated_usd"] = estimate_usd(agg["total_input_tokens"])
    # High-conf / auto-skip coverage (when confidence gate present)
    hc_rows = [r for r in rows if "high_conf" in r["metrics"]]
    if hc_rows:
        for k in ("iou", "precision", "recall", "f1", "content_fp_rate"):
            vals = [float(r["metrics"]["high_conf"][k]) for r in hc_rows]
            agg[f"high_conf_{k}"] = sum(vals) / len(vals) if vals else None
        n_auto = sum(
            int((r["metrics"].get("confidence_gate") or {}).get("n_auto_skip") or 0)
            for r in hc_rows
        )
        n_seg = sum(
            int((r["metrics"].get("confidence_gate") or {}).get("n_segments") or 0)
            for r in hc_rows
        )
        agg["auto_skip_segments"] = n_auto
        agg["all_pred_segments"] = n_seg
        agg["auto_skip_coverage"] = (n_auto / n_seg) if n_seg else None

    all_lats: list[float] = []
    for r in rows:
        # use per-call p50 stored; also total
        p = r["metrics"].get("latency_ms_p50_calls")
        if p is not None:
            all_lats.append(float(p))
    agg["latency_p50_ms_across_videos"] = _pct(all_lats, 50) if all_lats else None
    totals = [float(r["metrics"]["latency_ms_total"]) for r in rows]
    agg["latency_total_p50_ms"] = _pct(totals, 50)
    agg["latency_total_mean_ms"] = statistics.mean(totals) if totals else None
    return agg


def write_report(
    path: Path,
    *,
    by_mode: dict[str, list[dict[str, Any]]],
    aggregates: dict[str, dict[str, Any]],
    prepared_meta: list[dict[str, Any]],
    blockers: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Sponsor Skip Bench — Jev Choice vs SponsorBlock")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        "- **Detector**: TypeSafe Jev closed-set `Choice` (no trained classifier)."
    )
    lines.append(
        "- **Primary mode**: chunk windows (~15s) labeled `sponsor` / `selfpromo` / "
        "`content` / `interaction`; adjacent sponsor-like chunks merged."
    )
    lines.append(
        "- **Secondary**: paragraph start→end Choice over ≤255 sentence candidates."
    )
    lines.append(
        "- **Gold**: SponsorBlock public API (`sponsor` + `selfpromo`)."
    )
    lines.append(
        f"- **Pricing**: ${TYPESAFE_INPUT_USD_PER_MILLION}/M input tokens (output free)."
    )
    lines.append(f"- **Videos evaluated**: {len(prepared_meta)}")
    lines.append("")
    lines.append("### Videos")
    lines.append("")
    lines.append("| video_id | duration_s | gold sponsor segs | n_chunks | n_sentences |")
    lines.append("|----------|------------|-------------------|----------|-------------|")
    for p in prepared_meta:
        n_gold = len(p["gold_sponsor"])
        lines.append(
            f"| `{p['video_id']}` | {p['duration_sec']:.0f} | {n_gold} | "
            f"{p['n_chunks']} | {p['n_sentences']} |"
        )
    lines.append("")

    lines.append("## Metrics by mode (macro-average over videos)")
    lines.append("")
    lines.append(
        "| mode | n | IoU | P | R | F1 | mean best IoU | video recall | content FP rate | "
        "tokens | USD | latency p50 (total ms) |"
    )
    lines.append(
        "|------|---|-----|---|---|----|---------------|--------------|-----------------|--------|-----|-------------------------|"
    )
    for mode, agg in aggregates.items():
        def fmt(x: Any, nd: int = 3) -> str:
            if x is None:
                return "—"
            if isinstance(x, float):
                return f"{x:.{nd}f}"
            return str(x)

        lines.append(
            f"| `{mode}` | {agg.get('n_videos')} | {fmt(agg.get('iou'))} | "
            f"{fmt(agg.get('precision'))} | {fmt(agg.get('recall'))} | {fmt(agg.get('f1'))} | "
            f"{fmt(agg.get('mean_best_iou'))} | {fmt(agg.get('video_level_recall'))} | "
            f"{fmt(agg.get('content_fp_rate'))} | {agg.get('total_input_tokens')} | "
            f"${fmt(agg.get('total_estimated_usd'), 4)} | {fmt(agg.get('latency_total_p50_ms'), 0)} |"
        )
    lines.append("")

    # Per-video tables
    for mode, rows in by_mode.items():
        lines.append(f"## Per-video — `{mode}`")
        lines.append("")
        lines.append(
            "| video | IoU | P | R | F1 | overlap_s | fp_s | content_fp | tokens | USD |"
        )
        lines.append(
            "|-------|-----|---|---|----|-----------|------|------------|--------|-----|"
        )
        for r in rows:
            m = r["metrics"]
            lines.append(
                f"| `{r['video_id']}` | {m['iou']:.3f} | {m['precision']:.3f} | "
                f"{m['recall']:.3f} | {m['f1']:.3f} | {m['overlap_sec']:.1f} | "
                f"{m['fp_sec']:.1f} | {m['content_fp_rate']:.4f} | {m['input_tokens']} | "
                f"${m['estimated_usd']:.4f} |"
            )
        lines.append("")
        # Examples
        lines.append("### Qualitative examples")
        lines.append("")
        for r in rows[:4]:
            lines.append(f"**{r['video_id']}** gold=`{r['gold']}` pred=`{r['pred_segments']}`")
            if r["examples"]["hits"]:
                lines.append(f"- hits: `{r['examples']['hits']}`")
            if r["examples"]["misses"]:
                lines.append(f"- misses: `{r['examples']['misses']}`")
            lines.append("")

    # Conclusion
    best_mode = None
    best_f1 = -1.0
    for mode, agg in aggregates.items():
        f1 = agg.get("f1") or -1
        if f1 > best_f1:
            best_f1 = f1
            best_mode = mode

    lines.append("## Conclusions")
    lines.append("")
    if best_mode is None:
        lines.append("No successful mode runs.")
    else:
        agg = aggregates[best_mode]
        works = (agg.get("video_level_recall") or 0) >= 0.5 and (agg.get("f1") or 0) >= 0.25
        lines.append(
            f"**Best mode**: `{best_mode}` "
            f"(macro F1={agg.get('f1'):.3f}, IoU={agg.get('iou'):.3f}, "
            f"video recall={agg.get('video_level_recall')}, "
            f"cost=${agg.get('total_estimated_usd'):.4f})."
        )
        lines.append("")
        if works:
            lines.append(
                "**Does propose→pick / Choice work for sponsor skipping?** "
                "**Yes, partially.** Chunk-level closed-set Choice reliably *detects* "
                "that a sponsorship is present (high video-level recall) and often "
                "overlaps the gold interval, at very low Typesafe cost. Boundary "
                "precision is coarser than SponsorBlock crowd timestamps because "
                "labels are assigned to ~15s windows."
            )
        else:
            lines.append(
                "**Does propose→pick / Choice work for sponsor skipping?** "
                "**Mixed / weak on this set.** Video-level detection or segment "
                "overlap did not clear a useful bar on the sampled videos."
            )
        lines.append("")
        lines.append("**Main failure modes observed / expected:**")
        lines.append("")
        lines.append(
            "1. **Boundary coarseness** — chunk windows over/under-extend vs gold; "
            "optional cue-level refine helps but adds calls."
        )
        lines.append(
            "2. **Self-promo vs sponsor confusion** — creators pitch their own merch "
            "with similar rhetoric; we map both to skippable `sponsor_like`."
        )
        lines.append(
            "3. **False positives on brand mentions** — product names during a review "
            "can look like an ad-read without a clear discount-code cue."
        )
        lines.append(
            "4. **Paragraph mode under-recall on multi-sponsor videos** — start/end "
            "Choice finds one (or few) regions and may miss a second mid-roll."
        )
        lines.append(
            "5. **ASR noise** — auto-captions garble URLs/codes that humans use as "
            "sponsor anchors."
        )
    lines.append("")
    if blockers:
        lines.append("## Blockers")
        lines.append("")
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")
    lines.append("## File paths")
    lines.append("")
    lines.append(f"- Experiment root: `{HERE}`")
    lines.append(f"- Runner: `{HERE / 'run_sponsor_skip_bench.py'}`")
    lines.append(f"- Transcripts: `{TRANS_DIR}`")
    lines.append(f"- Gold: `{GOLD_DIR}`")
    lines.append(f"- Results JSON: `{RESULTS_DIR}`")
    lines.append(f"- This report: `{path}`")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=8, help="Max videos to evaluate")
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["regex_jev_refined", "regex_propose_jev_confirm"],
        choices=[
            "chunk",
            "chunk_batched",
            "paragraph",
            "regex_propose_jev_confirm",
            "regex_jev_refined",
            "jev_then_regex_gate",
            "tony_line_scan",
        ],
    )
    ap.add_argument("--refine", action="store_true", help="Cue-level boundary refine")
    ap.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.75,
        help="Auto-skip confidence gate for regex_jev_refined (default 0.75)",
    )
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument(
        "--video-ids",
        nargs="*",
        default=None,
        help="Override candidate video IDs",
    )
    ap.add_argument("--model", default="jev-latest")
    args = ap.parse_args(argv)

    blockers: list[str] = []
    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ids = list(args.video_ids or CANDIDATE_VIDEO_IDS)
    print(f"Preparing up to {args.limit} videos from {len(ids)} candidates…")
    prepared: list[dict[str, Any]] = []
    for vid in ids:
        if len(prepared) >= args.limit:
            break
        print(f"  prepare {vid} …", flush=True)
        payload = load_prepared_from_cache(vid)
        if payload is not None:
            print(
                f"    cache hit duration={payload['duration_sec']:.0f}s "
                f"chunks={payload['n_chunks']} gold={len(payload['gold_sponsor'])}"
            )
            prepared.append(payload)
            continue
        try:
            payload = prepare_video(vid)
        except Exception as exc:  # noqa: BLE001
            print(f"    skip ({type(exc).__name__}: {exc})")
            blockers.append(f"{vid}: prepare failed: {exc}")
            continue
        if payload is None:
            print("    skip (no gold or no transcript)")
            continue
        print(
            f"    ok duration={payload['duration_sec']:.0f}s "
            f"chunks={payload['n_chunks']} gold={len(payload['gold_sponsor'])}"
        )
        prepared.append(payload)

    save_json(
        RESULTS_DIR / "prepared_index.json",
        [
            {
                "video_id": p["video_id"],
                "duration_sec": p["duration_sec"],
                "n_chunks": p["n_chunks"],
                "n_sentences": p["n_sentences"],
                "n_gold_sponsor": len(p["gold_sponsor"]),
            }
            for p in prepared
        ],
    )

    if not prepared:
        blockers.append("No videos prepared (SponsorBlock gold + captions required).")
        write_report(
            HERE / "SPONSOR_SKIP_REPORT.md",
            by_mode={},
            aggregates={},
            prepared_meta=[],
            blockers=blockers,
        )
        print("ERROR: nothing to evaluate")
        return 1

    if args.prepare_only:
        print(f"Prepared {len(prepared)} videos; exiting (--prepare-only).")
        return 0

    load_card_secrets()
    import os

    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        blockers.append("TYPESAFE_API_KEY missing after load_card_secrets")
        print("ERROR: no API key")
        return 2

    from typesafe_sdk import TypeSafeClient

    client = TypeSafeClient(api_key=key, model=args.model)
    detector = SponsorDetector(client, model=args.model)

    by_mode: dict[str, list[dict[str, Any]]] = {}
    aggregates: dict[str, dict[str, Any]] = {}

    try:
        for mode in args.modes:
            print(f"\n=== mode={mode} ===")
            rows: list[dict[str, Any]] = []
            for p in prepared:
                # Fresh token counters are cumulative on detector; that's fine
                print(f"  run {p['video_id']} …", flush=True)
                try:
                    row = run_mode(detector, p, mode, refine=args.refine, confidence_threshold=args.confidence_threshold)
                except Exception as exc:  # noqa: BLE001
                    print(f"    FAIL {type(exc).__name__}: {exc}")
                    blockers.append(f"{p['video_id']}/{mode}: {exc}")
                    continue
                m = row["metrics"]
                print(
                    f"    IoU={m['iou']:.3f} F1={m['f1']:.3f} "
                    f"tokens={m['input_tokens']} ${m['estimated_usd']:.4f}"
                )
                rows.append(row)
            by_mode[mode] = rows
            aggregates[mode] = aggregate(rows)
            save_json(
                RESULTS_DIR / f"results_{mode}.json",
                {"mode": mode, "rows": rows, "aggregate": aggregates[mode]},
            )
    finally:
        closer = getattr(client, "close", None)
        if callable(closer):
            closer()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_path = RESULTS_DIR / f"sponsor_skip_{stamp}.json"
    save_json(
        all_path,
        {
            "created_at": stamp,
            "model": args.model,
            "modes": args.modes,
            "refine": args.refine,
            "videos": [p["video_id"] for p in prepared],
            "aggregates": aggregates,
            "by_mode": {k: v for k, v in by_mode.items()},
            "blockers": blockers,
            "pricing_usd_per_m_input": TYPESAFE_INPUT_USD_PER_MILLION,
        },
    )
    # Also write a stable latest pointer
    save_json(RESULTS_DIR / "latest.json", {"path": all_path.name, "aggregates": aggregates})

    report_path = HERE / "SPONSOR_SKIP_REPORT.md"
    write_report(
        report_path,
        by_mode=by_mode,
        aggregates=aggregates,
        prepared_meta=prepared,
        blockers=blockers,
    )
    # Copy report into results too
    (RESULTS_DIR / "SPONSOR_SKIP_REPORT.md").write_text(
        report_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    print("\n=== DONE ===")
    for mode, agg in aggregates.items():
        print(
            f"{mode}: n={agg.get('n_videos')} F1={agg.get('f1')} "
            f"IoU={agg.get('iou')} video_R={agg.get('video_level_recall')} "
            f"tokens={agg.get('total_input_tokens')} ${agg.get('total_estimated_usd')}"
        )
    print(f"Report: {report_path}")
    print(f"Results: {all_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
