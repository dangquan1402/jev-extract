#!/usr/bin/env python3
"""Comparison benchmark CLI: classification + extraction (TypeSafe Jev first).

Usage:
  python benchmarks/run_bench.py classification --backend typesafe --limit 50
  python benchmarks/run_bench.py extraction --backend typesafe --limit 50
  python benchmarks/run_bench.py all --backend typesafe
  python benchmarks/run_bench.py classification --backend typesafe --dry-run
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
DEFAULT_OUT_DIR = BENCH_DIR / "results"
CLASSIFICATION_DATA = BENCH_DIR / "data" / "classification_50.jsonl"
EXTRACTION_DATA = BENCH_DIR / "data" / "extraction_50.jsonl"
PINNED_MODEL = "jev-latest"

if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from metrics import TYPESAFE_INPUT_USD_PER_MILLION  # noqa: E402
from _bench_backends import get_classify_backend, get_extract_backend  # noqa: E402
from _bench_io import (  # noqa: E402
    TaskSummary,
    format_classification_table,
    format_extraction_table,
    load_jsonl,
    validate_classification,
    validate_extraction,
)
from _bench_runners import run_classification, run_extraction, write_results  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Comparison benchmark: classification & extraction (TypeSafe Jev first)",
    )
    p.add_argument("task", choices=["classification", "extraction", "all"])
    p.add_argument("--backend", default="typesafe", choices=["typesafe", "gemini", "haiku"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args(argv)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_task(task, backend_name, *, limit, warmup, dry_run, out_dir):
    if task == "classification":
        data_path = CLASSIFICATION_DATA
        examples = load_jsonl(data_path, limit)
        verrs = validate_classification(examples)
        backend = get_classify_backend(backend_name)
    else:
        data_path = EXTRACTION_DATA
        examples = load_jsonl(data_path, limit)
        verrs = validate_extraction(examples)
        backend = get_extract_backend(backend_name)

    if verrs:
        print(f"Dataset validation failed ({data_path}):", file=sys.stderr)
        for e in verrs[:20]:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(2)

    print(f"[{task}] loaded {len(examples)} examples from {data_path}")
    ok, reason = backend.available()
    if not ok:
        print(f"[{task}] SKIP {backend_name}: {reason}")
        summary = TaskSummary(
            task=task,
            backend=backend_name,
            n=0,
            errors=0,
            sum_input_tokens=0,
            sum_output_tokens=0,
            total_usd=0.0,
            p50_ms=0.0,
            p95_ms=0.0,
            mean_ms=0.0,
            accuracy=0.0 if task == "classification" else None,
            token_em=0.0 if task == "extraction" else None,
            token_f1=0.0 if task == "extraction" else None,
            token_iou=0.0 if task == "extraction" else None,
            exact_match=0.0 if task == "extraction" else None,
            span_f1=0.0 if task == "extraction" else None,
            contains_gold=0.0 if task == "extraction" else None,
            skipped=True,
            skip_reason=reason,
        )
        table = (
            format_classification_table(summary)
            if task == "classification"
            else format_extraction_table(summary)
        )
        return summary, None, table

    if dry_run:
        print(f"[{task}] OK {backend_name}: available (dry-run, no API calls)")
        print(f"[{task}] data validation: PASS ({len(examples)} rows)")
        summary = TaskSummary(
            task=task,
            backend=backend_name,
            n=len(examples),
            errors=0,
            sum_input_tokens=0,
            sum_output_tokens=0,
            total_usd=0.0,
            p50_ms=0.0,
            p95_ms=0.0,
            mean_ms=0.0,
            accuracy=None,
            token_em=None,
            token_f1=None,
            token_iou=None,
            exact_match=None,
            span_f1=None,
            contains_gold=None,
            skipped=True,
            skip_reason="dry-run (no API calls)",
        )
        table = (
            format_classification_table(summary)
            if task == "classification"
            else format_extraction_table(summary)
        )
        backend.close()
        return summary, None, table

    print(f"[{task}] RUN {backend_name} n={len(examples)} warmup={warmup} …")
    try:
        if task == "classification":
            records, summary = run_classification(backend, examples, warmup)
            table = format_classification_table(summary)
        else:
            records, summary = run_extraction(backend, examples, warmup)
            table = format_extraction_table(summary)
    finally:
        backend.close()

    stamp = _ts()
    out_path = out_dir / f"{task}_{backend_name}_{stamp}.json"
    meta = {
        "task": task,
        "backend": backend_name,
        "data": str(data_path),
        "limit": limit,
        "warmup": warmup,
        "n_examples": len(examples),
        "model_requested": PINNED_MODEL,
        "pricing": {
            "input_usd_per_million": TYPESAFE_INPUT_USD_PER_MILLION,
            "output_usd_per_million": 0.0,
            "note": "TypeSafe Jev public pricing",
        },
        "timestamp_utc": stamp,
    }
    write_results(
        out_path, task=task, backend=backend_name, meta=meta, summary=summary, records=records
    )
    print(f"[{task}] wrote {out_path}")
    if summary.resolved_models:
        print(f"[{task}] resolved model(s): {', '.join(summary.resolved_models)}")
    return summary, out_path, table


def main(argv=None) -> int:
    args = parse_args(argv)
    tasks = ["classification", "extraction"] if args.task == "all" else [args.task]
    tables = []
    paths = []
    summaries = []
    for task in tasks:
        summary, path, table = run_task(
            task,
            args.backend,
            limit=args.limit,
            warmup=args.warmup,
            dry_run=args.dry_run,
            out_dir=args.out_dir,
        )
        summaries.append(summary)
        tables.append((task, table))
        if path is not None:
            paths.append(path)
        print()
        print(f"### {task}")
        print(table)
        print()

    summary_md = args.out_dir / "SUMMARY.md"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    nl = chr(10)
    lines = [
        "# Benchmark summary",
        "",
        f"Generated: {_ts()} (UTC)",
        f"Backend: `{args.backend}`",
        f"Dry-run: {args.dry_run}",
        "",
        f"Pricing (TypeSafe Jev): **${TYPESAFE_INPUT_USD_PER_MILLION} / million input tokens**, output free.",
        "",
    ]
    for task, table in tables:
        lines.extend([f"## {task.title()}", "", table, ""])
        matching = [s for s in summaries if s.task == task]
        if matching and matching[0].resolved_models:
            lines.extend(
                [f"Resolved model(s): `{', '.join(matching[0].resolved_models)}`", ""]
            )
    if paths:
        lines.extend(["## Result files", ""])
        lines.extend([f"- `{p}`" for p in paths])
        lines.append("")
    lines.extend(
        [
            "## Fairness note",
            "",
            "Jev extraction is **token-native**: sequential start/end Choice over tokenizer-v1 tokens "
            "(criteria keys = positions, labels = `pos:word`). Primary metric is **token-span exact "
            "match** (`token_em`); secondary are token-index F1/IoU. String `exact_match` / "
            "`contains_gold` remain as optional diagnostics. Sentence-candidate Choice is still "
            "available via `Extractor.extract_sentence` for coarse comparisons.",
            "",
        ]
    )
    summary_md.write_text(nl.join(lines) + nl, encoding="utf-8")
    print(f"Wrote {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
