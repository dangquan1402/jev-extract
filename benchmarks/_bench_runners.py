"""Classification/extraction runners + result writer."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from metrics import (
    accuracy,
    classification_correct,
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
from _bench_backends import ClassifyBackend, ExtractBackend
from _bench_io import TaskSummary

def run_classification(
    backend: ClassifyBackend,
    examples: list[dict[str, Any]],
    warmup: int,
) -> tuple[list[dict[str, Any]], TaskSummary]:
    records: list[dict[str, Any]] = []

    def _one(ex: dict[str, Any], *, is_warmup: bool) -> dict[str, Any]:
        out = backend.classify(ex["text"], ex["labels"])
        ok = False if out.error else classification_correct(ex["label"], out.pred)
        usd = estimated_usd(out.input_tokens, out.output_tokens)
        return {
            "id": ex["id"],
            "task": "classification",
            "backend": backend.name,
            "gold": ex["label"],
            "pred": out.pred,
            "correct": ok,
            "latency_ms": out.latency_ms,
            "input_tokens": out.input_tokens,
            "output_tokens": out.output_tokens,
            "estimated_usd": usd,
            "model": out.model,
            "error": out.error,
            "warmup": is_warmup,
        }

    for i in range(min(warmup, len(examples))):
        records.append(_one(examples[i], is_warmup=True))

    for ex in examples:
        records.append(_one(ex, is_warmup=False))

    scored = [r for r in records if not r["warmup"]]
    n = len(scored)
    errors = sum(1 for r in scored if r["error"])
    ok_rows = [r for r in scored if r["error"] is None]
    lats = [r["latency_ms"] for r in ok_rows] or [r["latency_ms"] for r in scored]
    stats = latency_stats(lats)
    inp = sum_optional(r["input_tokens"] for r in scored)
    out = sum_optional(r["output_tokens"] for r in scored)
    models = sorted({r["model"] for r in scored if r.get("model")})
    summary = TaskSummary(
        task="classification",
        backend=backend.name,
        n=n,
        errors=errors,
        sum_input_tokens=inp,
        sum_output_tokens=out,
        total_usd=estimated_usd(inp, out),
        accuracy=accuracy([bool(r["correct"]) for r in scored]) if n else 0.0,
        resolved_models=models,
        **stats,
    )
    return records, summary


def run_extraction(
    backend: ExtractBackend,
    examples: list[dict[str, Any]],
    warmup: int,
) -> tuple[list[dict[str, Any]], TaskSummary]:
    records: list[dict[str, Any]] = []

    def _gold_token_span(ex: dict[str, Any]) -> tuple[int, int]:
        return int(ex["start"]["pos"]), int(ex["end"]["pos"])

    def _one(ex: dict[str, Any], *, is_warmup: bool) -> dict[str, Any]:
        out = backend.extract(ex["paragraph"], ex["question"])
        g_s, g_e = _gold_token_span(ex)
        if out.error:
            tem = False
            tf1 = tiou = 0.0
            em = cg = False
            f1v = 0.0
        else:
            tem = token_span_exact(g_s, g_e, out.start, out.end)
            tf1 = token_span_f1(g_s, g_e, out.start, out.end)
            tiou = token_span_iou(g_s, g_e, out.start, out.end)
            em = exact_match(ex["gold"], out.pred)
            f1v = span_f1(ex["gold"], out.pred)
            cg = contains_gold(ex["gold"], out.pred)
        usd = estimated_usd(out.input_tokens, out.output_tokens)
        return {
            "id": ex["id"],
            "task": "extraction",
            "backend": backend.name,
            "gold": ex["gold"],
            "gold_start_pos": g_s,
            "gold_end_pos": g_e,
            "pred": out.pred,
            "token_em": tem,
            "token_f1": tf1,
            "token_iou": tiou,
            "exact_match": em,
            "span_f1": f1v,
            "contains_gold": cg,
            "latency_ms": out.latency_ms,
            "input_tokens": out.input_tokens,
            "output_tokens": out.output_tokens,
            "estimated_usd": usd,
            "model": out.model,
            "pred_start": out.start,
            "pred_end": out.end,
            "error": out.error,
            "warmup": is_warmup,
        }

    for i in range(min(warmup, len(examples))):
        records.append(_one(examples[i], is_warmup=True))

    for ex in examples:
        records.append(_one(ex, is_warmup=False))

    scored = [r for r in records if not r["warmup"]]
    n = len(scored)
    errors = sum(1 for r in scored if r["error"])
    ok_rows = [r for r in scored if r["error"] is None]
    lats = [r["latency_ms"] for r in ok_rows] or [r["latency_ms"] for r in scored]
    stats = latency_stats(lats)
    inp = sum_optional(r["input_tokens"] for r in scored)
    out_tok = sum_optional(r["output_tokens"] for r in scored)
    models = sorted({r["model"] for r in scored if r.get("model")})
    summary = TaskSummary(
        task="extraction",
        backend=backend.name,
        n=n,
        errors=errors,
        sum_input_tokens=inp,
        sum_output_tokens=out_tok,
        total_usd=estimated_usd(inp, out_tok),
        token_em=(sum(1 for r in scored if r["token_em"]) / n) if n else 0.0,
        token_f1=(sum(float(r["token_f1"]) for r in scored) / n) if n else 0.0,
        token_iou=(sum(float(r["token_iou"]) for r in scored) / n) if n else 0.0,
        exact_match=(sum(1 for r in scored if r["exact_match"]) / n) if n else 0.0,
        span_f1=(sum(float(r["span_f1"]) for r in scored) / n) if n else 0.0,
        contains_gold=(sum(1 for r in scored if r["contains_gold"]) / n) if n else 0.0,
        resolved_models=models,
        **stats,
    )
    return records, summary


def write_results(
    path: Path,
    *,
    task: str,
    backend: str,
    meta: dict[str, Any],
    summary: TaskSummary,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "summary": asdict(summary),
        "results": records,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


