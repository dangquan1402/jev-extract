#!/usr/bin/env python3
"""Compare extraction modes vs token_sequential baseline.

Modes: token_sequential, chunk_only, cascade, span_choice, anchor_expand,
shape_gate, topk_noul, ambiguous_joint, length_rerank.

Usage:
  export JEV_EXTRACT_SECRETS_JSON=/home/box/agent-data/box-secrets.json
  python benchmarks/run_mode_compare.py --limit 20 --stratified
  python benchmarks/run_mode_compare.py --limit 50 --all-modes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
DEFAULT_OUT_DIR = BENCH_DIR / "results"
EXTRACTION_DATA = BENCH_DIR / "data" / "extraction_50.jsonl"
PINNED_MODEL = "jev-latest"

if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metrics import (  # noqa: E402
    TYPESAFE_INPUT_USD_PER_MILLION,
    contains_gold,
    estimated_usd,
    exact_match,
    latency_stats,
    span_f1,
    sum_optional,
    token_span_exact,
    token_span_f1,
    token_span_iou,
)
from _bench_io import load_jsonl, validate_extraction  # noqa: E402
from _bench_types import _usage_from_response, load_typesafe_api_key  # noqa: E402


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def stratified_sample(examples: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {"1": [], "2-3": [], "4+": []}
    for ex in examples:
        length = int(ex["end"]["pos"]) - int(ex["start"]["pos"]) + 1
        if length == 1:
            buckets["1"].append(ex)
        elif length <= 3:
            buckets["2-3"].append(ex)
        else:
            buckets["4+"].append(ex)
    sizes = {k: len(v) for k, v in buckets.items() if v}
    total = sum(sizes.values()) or 1
    targets = {k: max(1, round(n * sizes[k] / total)) for k in sizes}
    while sum(targets.values()) > n:
        k = max(targets, key=lambda x: targets[x])
        if targets[k] > 1:
            targets[k] -= 1
        else:
            break
    while sum(targets.values()) < n:
        k = max(sizes, key=lambda x: sizes[x] - targets.get(x, 0))
        if targets.get(k, 0) < sizes[k]:
            targets[k] = targets.get(k, 0) + 1
        else:
            break
    out: list[dict[str, Any]] = []
    for k, t in targets.items():
        out.extend(buckets[k][:t])
    chosen = {ex["id"] for ex in out}
    for ex in examples:
        if len(out) >= n:
            break
        if ex["id"] not in chosen:
            out.append(ex)
            chosen.add(ex["id"])
    return out[:n]


def run_mode(
    mode: str,
    examples: list[dict[str, Any]],
    *,
    warmup: int,
    api_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from jev_extract import Extractor

    extractor = Extractor(
        api_key=api_key, model=PINNED_MODEL, extract_mode="sequential"
    )
    records: list[dict[str, Any]] = []

    def _call(paragraph: str, question: str):
        if mode == "token_sequential":
            return extractor.extract(
                paragraph=paragraph,
                question=question,
                include_raw=True,
                extract_mode="sequential",
            )
        if mode == "chunk_only":
            return extractor.extract_chunk(
                paragraph=paragraph, question=question, include_raw=True
            )
        if mode == "cascade":
            return extractor.extract_cascade(
                paragraph=paragraph, question=question, include_raw=True
            )
        if mode == "span_choice":
            # Classic expensive path (full 1..K) for A/B baselines.
            return extractor.extract_span_choice(
                paragraph=paragraph,
                question=question,
                include_raw=True,
                propose="all",
            )
        if mode in ("span_choice_smart", "span_choice_cheap"):
            return extractor.extract_span_choice(
                paragraph=paragraph,
                question=question,
                include_raw=True,
                propose="smart",
            )
        if mode == "span_choice_in_chunk":
            return extractor.extract_span_choice(
                paragraph=paragraph,
                question=question,
                include_raw=True,
                propose="in_chunk",
            )
        if mode == "anchor_expand":
            return extractor.extract_anchor_expand(
                paragraph=paragraph, question=question, include_raw=True
            )
        if mode == "shape_gate":
            return extractor.extract_shape_gate(
                paragraph=paragraph, question=question, include_raw=True
            )
        if mode == "topk_noul":
            return extractor.extract_topk_noul(
                paragraph=paragraph, question=question, include_raw=True, top_k=3
            )
        if mode == "ambiguous_joint":
            return extractor.extract_ambiguous_joint(
                paragraph=paragraph, question=question, include_raw=True
            )
        if mode == "length_rerank":
            return extractor.extract_length_rerank(
                paragraph=paragraph,
                question=question,
                include_raw=True,
                top_k=3,
                shape_aware=True,
            )
        raise ValueError(f"unknown mode {mode}")

    def _one(ex: dict[str, Any], *, is_warmup: bool) -> dict[str, Any]:
        g_s, g_e = int(ex["start"]["pos"]), int(ex["end"]["pos"])
        t0 = time.perf_counter()
        err = None
        pred = ""
        p_s = p_e = None
        model = PINNED_MODEL
        inp = out = None
        chunk_key = None
        try:
            result = _call(ex["paragraph"], ex["question"])
            latency = (time.perf_counter() - t0) * 1000.0
            pred = result.answer or ""
            p_s = result.start.pos if result.start is not None else None
            p_e = result.end.pos if result.end is not None else None
            model = result.model or PINNED_MODEL
            inp, out = _usage_from_response(result.raw)
            if isinstance(result.raw, dict):
                chunk_key = result.raw.get("chunk_key")
        except Exception as exc:  # noqa: BLE001
            latency = (time.perf_counter() - t0) * 1000.0
            err = f"{type(exc).__name__}: {exc}"

        if err:
            tem = False
            tf1 = tiou = 0.0
            em = cg = False
            f1v = 0.0
        else:
            tem = token_span_exact(g_s, g_e, p_s, p_e)
            tf1 = token_span_f1(g_s, g_e, p_s, p_e)
            tiou = token_span_iou(g_s, g_e, p_s, p_e)
            em = exact_match(ex["gold"], pred)
            f1v = span_f1(ex["gold"], pred)
            cg = contains_gold(ex["gold"], pred)

        return {
            "id": ex["id"],
            "mode": mode,
            "gold": ex["gold"],
            "gold_start_pos": g_s,
            "gold_end_pos": g_e,
            "pred": pred,
            "pred_start": p_s,
            "pred_end": p_e,
            "token_em": tem,
            "token_f1": tf1,
            "token_iou": tiou,
            "exact_match": em,
            "span_f1": f1v,
            "contains_gold": cg,
            "latency_ms": latency,
            "input_tokens": inp,
            "output_tokens": out,
            "estimated_usd": estimated_usd(inp, out),
            "model": model,
            "chunk_key": chunk_key,
            "error": err,
            "warmup": is_warmup,
            "paragraph": ex["paragraph"],
            "question": ex["question"],
        }

    try:
        for i in range(min(warmup, len(examples))):
            records.append(_one(examples[i], is_warmup=True))
        for ex in examples:
            rec = _one(ex, is_warmup=False)
            records.append(rec)
            print(
                f"  [{mode}] {ex['id']} em={rec['token_em']} "
                f"cg={rec['contains_gold']} err={rec['error']}"
            )
    finally:
        extractor.close()

    scored = [r for r in records if not r["warmup"]]
    n = len(scored)
    errors = sum(1 for r in scored if r["error"])
    ok_rows = [r for r in scored if r["error"] is None]
    lats = [r["latency_ms"] for r in ok_rows] or [r["latency_ms"] for r in scored]
    stats = latency_stats(lats)
    inp = sum_optional(r["input_tokens"] for r in scored)
    out_tok = sum_optional(r["output_tokens"] for r in scored)
    summary = {
        "mode": mode,
        "n": n,
        "errors": errors,
        "token_em": (sum(1 for r in scored if r["token_em"]) / n) if n else 0.0,
        "token_f1": (sum(float(r["token_f1"]) for r in scored) / n) if n else 0.0,
        "token_iou": (sum(float(r["token_iou"]) for r in scored) / n) if n else 0.0,
        "exact_match": (sum(1 for r in scored if r["exact_match"]) / n) if n else 0.0,
        "span_f1": (sum(float(r["span_f1"]) for r in scored) / n) if n else 0.0,
        "contains_gold": (sum(1 for r in scored if r["contains_gold"]) / n) if n else 0.0,
        "sum_input_tokens": inp,
        "sum_output_tokens": out_tok,
        "total_usd": estimated_usd(inp, out_tok),
        "p50_ms": stats["p50_ms"],
        "p95_ms": stats["p95_ms"],
        "mean_ms": stats["mean_ms"],
        "resolved_models": sorted({r["model"] for r in scored if r.get("model")}),
        "pricing_note": f"TypeSafe Jev: ${TYPESAFE_INPUT_USD_PER_MILLION}/M input, output free",
    }
    return records, summary


def classify_miss(ex: dict[str, Any], rec: dict[str, Any]) -> str:
    if rec.get("error"):
        return "api_error"
    if not (rec.get("pred") or "").strip():
        return "empty_pred"
    g_s, g_e = rec["gold_start_pos"], rec["gold_end_pos"]
    p_s, p_e = rec.get("pred_start"), rec.get("pred_end")
    if p_s is None or p_e is None:
        return "missing_span"
    if p_e < g_s or p_s > g_e:
        return "wrong_chunk_or_span"
    if p_s == g_s and p_e > g_e:
        return "end_over_extend"
    if p_s == g_s and p_e < g_e:
        return "end_under_extend"
    if p_s < g_s and p_e == g_e:
        return "start_early"
    if p_s > g_s and p_e == g_e:
        return "start_late"
    if p_s < g_s and p_e > g_e:
        return "span_superset"
    if p_s > g_s and p_e < g_e:
        return "span_subset"
    if p_s != g_s and p_e != g_e:
        para = ex.get("paragraph") or rec.get("paragraph") or ""
        gold_words = str(rec["gold"]).split()
        if gold_words and para.count(gold_words[0]) > 1:
            return "duplicate_word"
        return "both_endpoints_wrong"
    if rec.get("contains_gold") and not rec.get("token_em"):
        return "superset_or_subset"
    return "other"


def analyze_misses(
    examples: list[dict[str, Any]],
    mode_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    by_id = {ex["id"]: ex for ex in examples}
    out: dict[str, Any] = {}
    for mode, records in mode_records.items():
        scored = [r for r in records if not r.get("warmup")]
        misses = [r for r in scored if not r.get("token_em")]
        labels: list[str] = []
        detail = []
        for r in misses:
            ex = by_id[r["id"]]
            label = classify_miss(ex, r)
            labels.append(label)
            detail.append(
                {
                    "id": r["id"],
                    "label": label,
                    "gold": r["gold"],
                    "pred": r["pred"],
                    "gold_span": [r["gold_start_pos"], r["gold_end_pos"]],
                    "pred_span": [r.get("pred_start"), r.get("pred_end")],
                    "contains_gold": r.get("contains_gold"),
                    "error": r.get("error"),
                }
            )
        out[mode] = {
            "n_miss": len(misses),
            "counts": dict(Counter(labels)),
            "details": detail,
        }
    return out


def format_metrics_table(summaries: list[dict[str, Any]]) -> str:
    headers = [
        "mode",
        "n",
        "token_em",
        "token_f1",
        "contains_gold",
        "exact_match",
        "p50_ms",
        "total_usd",
        "errors",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:" for _ in headers[1:]]) + "|",
    ]
    for s in summaries:
        lines.append(
            "| {mode} | {n} | {token_em:.1%} | {token_f1:.3f} | {contains_gold:.1%} | "
            "{exact_match:.1%} | {p50_ms:.0f} | ${total_usd:.5f} | {errors} |".format(**s)
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    stamp: str,
    n: int,
    stratified: bool,
    summaries: list[dict[str, Any]],
    miss_analysis: dict[str, Any],
    result_files: list[Path],
) -> None:
    lines = [
        "# Extraction mode comparison — miss analysis & recommendations",
        "",
        f"Generated: {stamp} (UTC)",
        f"Samples: **{n}** ({'stratified gold-length subset' if stratified else 'prefix / full load'})",
        f"Pricing: ${TYPESAFE_INPUT_USD_PER_MILLION}/M input tokens (output free).",
        "",
        "## Metrics",
        "",
        format_metrics_table(summaries),
        "",
        "## Failure taxonomy (non-token_em)",
        "",
    ]
    for mode, block in miss_analysis.items():
        lines.append(f"### {mode} — {block['n_miss']} misses")
        lines.append("")
        if not block["counts"]:
            lines.append("_No misses._")
            lines.append("")
            continue
        lines.append("| failure_mode | count |")
        lines.append("|---|---:|")
        for k, v in sorted(block["counts"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{k}` | {v} |")
        lines.append("")
        lines.append("<details><summary>Miss details</summary>")
        lines.append("")
        for d in block["details"]:
            lines.append(
                f"- **{d['id']}** `{d['label']}`: gold={d['gold']!r} pred={d['pred']!r} "
                f"spans gold={d['gold_span']} pred={d['pred_span']}"
            )
        lines.append("")
        lines.append("</details>")
        lines.append("")

    all_counts: Counter[str] = Counter()
    for block in miss_analysis.values():
        all_counts.update(block["counts"])
    focus_counts: Counter[str] = Counter()
    for m in ("token_sequential", "cascade"):
        if m in miss_analysis:
            focus_counts.update(miss_analysis[m]["counts"])

    lines.extend(
        [
            "## Ranked improvement methods (tied to observed misses)",
            "",
            "Ordered by expected impact given miss counts on **token_sequential + cascade** "
            "(chunk_only EM is structurally low when gold is a short subspan). "
            "**Not implemented** unless noted.",
            "",
        ]
    )
    recommendations = [
        (
            "Bias shorter end spans / length prior on end Choice",
            "end_over_extend",
            "Add instruction + soft prior favoring minimal ends (or re-rank top-k ends by length×prob). "
            "Expected: cut end_over_extend; small risk of end_under_extend on multi-token golds.",
        ),
        (
            "Joint start–end when start word is duplicated",
            "duplicate_word",
            "When the chosen start surface form appears >1×, use joint pair Choice (or top-3 starts × ends). "
            "Expected: fewer wrong latch points on repeated words.",
        ),
        (
            "Second-pass re-rank of top-k ends",
            "end_over_extend",
            "Keep top-3 end probabilities; pick shortest that still answers the question / "
            "has high joint conf. Expected: improve EM without schema changes.",
        ),
        (
            "Leading-determiner / start-early fix",
            "start_early",
            "Post-check: if pred starts with a determiner/preposition not required by the question, "
            "shift start +1. Low-cost heuristic for contract/on-November style misses.",
        ),
        (
            "Cascade chunker tuning (clause vs sentence; ID-pointer)",
            "wrong_chunk_or_span",
            "If cascade wrong_chunk dominates: try tagged state + criteria={id: None}, or tighter "
            "clause splits. On this 3-sentence set, chunk selection is usually easy.",
        ),
        (
            "Exists / Noul abstention gate",
            "empty_pred",
            "Pre-ask Noul 'does paragraph contain answer?'; skip span Choice on low noul. "
            "Helps empty/short pathologies more than EM on this dense extractive set.",
        ),
        (
            "Strengthen end-under / start-late instructions",
            "end_under_extend",
            "When gold is a full sentence but pred truncates (sentence-gold items), bias toward "
            "including trailing predicate tokens — opposite prior of short-answer items. "
            "A question-type gate (entity vs sentence) would help.",
        ),
    ]
    scored_recs = []
    for title, key, body in recommendations:
        scored_recs.append((focus_counts.get(key, 0) + all_counts.get(key, 0), title, key, body))
    scored_recs.sort(key=lambda x: -x[0])
    for i, (cnt, title, key, body) in enumerate(scored_recs, 1):
        lines.append(f"{i}. **{title}** _(drives `{key}`, observed≈{cnt})_")
        lines.append(f"   - {body}")
        lines.append("")

    lines.extend(
        [
            "## Result files",
            "",
            *[f"- `{p}`" for p in result_files],
            "",
            "## Notes",
            "",
            "- `token_sequential`: start/end Choice (`extract_mode=sequential`) — baseline.",
            "- `chunk_only`: sentence/clause Choice; span = whole chunk.",
            "- `cascade`: chunk Choice → start/end inside winner.",
            "- `span_choice`: classic full 1..K windows (expensive input tokens).",
            "- `span_choice_smart` / `span_choice_cheap`: dense ≤6 + sentences + truncated criteria.",
            "- `span_choice_in_chunk`: chunk Choice → dense span Choice inside winner.",
            "- `anchor_expand`: head Choice → left/right extent Choices.",
            "- `shape_gate`: shape Choice → shape-specific end instructions.",
            "- `topk_noul`: top-k ends + batched Noul verify.",
            "- `ambiguous_joint`: joint (start,end) when start ambiguous; else sequential.",
            "- `length_rerank`: argmax(prob × length_penalty); shape-aware longer for sentences.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare token / chunk / cascade extraction modes")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--stratified", action="store_true", help="Stratify by gold token length")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--modes",
        nargs="+",
        default=["token_sequential", "chunk_only", "cascade"],
        choices=['token_sequential', 'chunk_only', 'cascade', 'span_choice', 'span_choice_smart', 'span_choice_cheap', 'span_choice_in_chunk', 'anchor_expand', 'shape_gate', 'topk_noul', 'ambiguous_joint', 'length_rerank'],
    )
    p.add_argument(
        "--all-modes",
        action="store_true",
        help="Run every registered mode (overrides --modes)",
    )
    p.add_argument(
        "--write-all-report",
        action="store_true",
        help="Also write benchmarks/results/ALL_MODES_REPORT.md",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    examples = load_jsonl(EXTRACTION_DATA, None)
    verrs = validate_extraction(examples)
    if verrs:
        print("validation failed:", verrs[:10], file=sys.stderr)
        return 2

    if args.stratified:
        examples = stratified_sample(examples, args.limit)
    else:
        examples = examples[: args.limit]

    if args.all_modes:
        args.modes = ['token_sequential', 'chunk_only', 'cascade', 'span_choice', 'anchor_expand', 'shape_gate', 'topk_noul', 'ambiguous_joint', 'length_rerank']

    print(f"Running modes={args.modes} n={len(examples)} stratified={args.stratified}")
    print("ids:", [e["id"] for e in examples])

    if args.dry_run:
        print("dry-run OK")
        return 0

    if not os.environ.get("TYPESAFE_API_KEY") and not os.environ.get("JEV_EXTRACT_SECRETS_JSON"):
        secrets = Path("/home/box/agent-data/box-secrets.json")
        if secrets.is_file():
            os.environ["JEV_EXTRACT_SECRETS_JSON"] = str(secrets)

    key = load_typesafe_api_key()
    if not key:
        print("ERROR: TYPESAFE_API_KEY not available", file=sys.stderr)
        return 3

    stamp = _ts()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    mode_records: dict[str, list[dict[str, Any]]] = {}
    result_files: list[Path] = []

    for mode in args.modes:
        print(f"\n=== {mode} ===")
        records, summary = run_mode(mode, examples, warmup=args.warmup, api_key=key)
        summaries.append(summary)
        mode_records[mode] = records
        out_path = args.out_dir / f"extraction_mode_{mode}_{stamp}.json"
        payload = {
            "meta": {
                "mode": mode,
                "n": len(examples),
                "stratified": args.stratified,
                "limit": args.limit,
                "warmup": args.warmup,
                "model_requested": PINNED_MODEL,
                "timestamp_utc": stamp,
                "ids": [e["id"] for e in examples],
            },
            "summary": summary,
            "results": records,
        }
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        result_files.append(out_path)
        print(f"wrote {out_path}")
        print(format_metrics_table([summary]))

    miss_analysis = analyze_misses(examples, mode_records)
    report_path = args.out_dir / f"MODE_COMPARE_REPORT_{stamp}.md"
    write_report(
        report_path,
        stamp=stamp,
        n=len(examples),
        stratified=args.stratified,
        summaries=summaries,
        miss_analysis=miss_analysis,
        result_files=result_files,
    )
    latest = args.out_dir / "MODE_COMPARE_REPORT.md"
    latest.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    combined = args.out_dir / f"extraction_mode_compare_{stamp}.json"
    combined.write_text(
        json.dumps(
            {
                "meta": {
                    "n": len(examples),
                    "stratified": args.stratified,
                    "timestamp_utc": stamp,
                    "ids": [e["id"] for e in examples],
                },
                "summaries": summaries,
                "miss_analysis": miss_analysis,
                "result_files": [str(p) for p in result_files],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.write_all_report or args.all_modes or len(args.modes) >= 6:
        all_report = args.out_dir / "ALL_MODES_REPORT.md"
        # Rank by token_em then token_f1 then lower usd
        ranked = sorted(
            summaries,
            key=lambda s: (-float(s["token_em"]), -float(s["token_f1"]), float(s.get("total_usd") or 0)),
        )
        lines = [
            "# ALL MODES REPORT — extraction strategy A/B",
            "",
            f"Generated: {stamp} (UTC)",
            f"Samples: **{len(examples)}** ({'stratified gold-length subset' if args.stratified else 'prefix / full load'})",
            f"Ids: {', '.join(e['id'] for e in examples)}",
            "",
            "## Metrics (ranked by token_em)",
            "",
            format_metrics_table(ranked),
            "",
            "## Recommendation",
            "",
        ]
        if ranked:
            best = ranked[0]
            base = next((s for s in summaries if s["mode"] == "token_sequential"), None)
            lines.append(
                f"- **Best EM:** `{best['mode']}` at **{best['token_em']:.1%}** "
                f"token_em / {best['token_f1']:.3f} token_f1 "
                f"(p50 {best['p50_ms']:.0f} ms, ${best['total_usd']:.5f})."
            )
            if base and best["mode"] != "token_sequential":
                delta = best["token_em"] - base["token_em"]
                lines.append(
                    f"- vs `token_sequential` ({base['token_em']:.1%} EM): "
                    f"{'+' if delta >= 0 else ''}{delta:.1%} EM; "
                    f"baseline p50 {base['p50_ms']:.0f} ms / ${base['total_usd']:.5f}."
                )
            elif base:
                lines.append("- Baseline `token_sequential` is best or tied on EM for this run.")
            # Cost/latency leaders among high-EM
            high = [s for s in ranked if s["token_em"] >= (ranked[0]["token_em"] - 0.05)]
            cheap = min(high, key=lambda s: (s.get("total_usd") or 0, s.get("p50_ms") or 0))
            fast = min(high, key=lambda s: (s.get("p50_ms") or 0, s.get("total_usd") or 0))
            lines.append(
                f"- **Among near-best EM (±5pp):** cheapest `{cheap['mode']}` "
                f"(${cheap['total_usd']:.5f}); fastest `{fast['mode']}` "
                f"(p50 {fast['p50_ms']:.0f} ms)."
            )
        lines.extend(["", "## Failure taxonomy", ""])
        for mode, block in miss_analysis.items():
            lines.append(f"### {mode} — {block['n_miss']} misses")
            lines.append("")
            if not block["counts"]:
                lines.append("_No misses._")
                lines.append("")
                continue
            lines.append("| failure_mode | count |")
            lines.append("|---|---:|")
            for k, v in sorted(block["counts"].items(), key=lambda kv: -kv[1]):
                lines.append(f"| `{k}` | {v} |")
            lines.append("")
            # Highlight a few misses
            for d in block["details"][:8]:
                lines.append(
                    f"- **{d['id']}** `{d['label']}`: gold={d['gold']!r} pred={d['pred']!r}"
                )
            if len(block["details"]) > 8:
                lines.append(f"- _… {len(block['details']) - 8} more_")
            lines.append("")
        lines.extend(
            [
                "## Result files",
                "",
                *[f"- `{p}`" for p in result_files],
                f"- `{combined}`",
                f"- `{report_path}`",
                "",
            ]
        )
        all_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"ALL_MODES_REPORT: {all_report}")

    print(f"\nReport: {report_path}")
    print(f"Latest: {latest}")
    print(f"Combined: {combined}")
    print("\n" + format_metrics_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
