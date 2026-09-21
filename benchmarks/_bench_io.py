"""Dataset IO + validation + summary tables."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from metrics import TYPESAFE_INPUT_USD_PER_MILLION

def _ensure_jsonl(path: Path) -> None:
    """If path missing, try assemble_*.py gzip+b64 packs in the same directory."""
    if path.is_file():
        return
    stem = path.stem  # extraction_50 / classification_50
    asm = path.parent / f"assemble_{stem}.py"
    if asm.is_file():
        import runpy
        runpy.run_path(str(asm), run_name="__main__")


def load_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Load JSONL from path, assemble packs, or path.stem.part*.jsonl shards."""
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    _ensure_jsonl(path)
    if path.is_file():
        sources = [path]
    else:
        parts = sorted(path.parent.glob(f"{path.stem}.part*.jsonl"))
        if not parts:
            raise FileNotFoundError(path)
        sources = parts
    for src in sources:
        with src.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    return rows
    return rows


def validate_classification(rows: list[dict[str, Any]]) -> list[str]:
    errs: list[str] = []
    for i, r in enumerate(rows):
        for key in ("id", "text", "label", "labels"):
            if key not in r:
                errs.append(f"row {i}: missing {key}")
                continue
        labels = r.get("labels")
        if not isinstance(labels, dict) or not labels:
            errs.append(f"row {i} ({r.get('id')}): labels must be non-empty dict")
        elif r.get("label") not in labels:
            errs.append(
                f"row {i} ({r.get('id')}): gold label {r.get('label')!r} "
                f"not in criteria keys {sorted(labels)}"
            )
    return errs


def validate_extraction(rows: list[dict[str, Any]]) -> list[str]:
    """Validate token-native extraction rows (tokenizer v1 gold spans)."""
    errs: list[str] = []
    for i, r in enumerate(rows):
        for key in ("id", "paragraph", "question", "gold", "tokens", "start", "end"):
            if key not in r:
                errs.append(f"row {i}: missing {key}")
        if errs and errs[-1].startswith(f"row {i}:"):
            continue
        para = r["paragraph"]
        tokens = r["tokens"]
        start_ref, end_ref = r["start"], r["end"]
        if not isinstance(start_ref, dict) or not isinstance(end_ref, dict):
            errs.append(f"row {i} ({r['id']}): start/end must be {{pos, word}} dicts")
            continue
        try:
            sp, ep = int(start_ref["pos"]), int(end_ref["pos"])
            sw, ew = str(start_ref["word"]), str(end_ref["word"])
        except (KeyError, TypeError, ValueError) as exc:
            errs.append(f"row {i} ({r['id']}): bad start/end refs: {exc}")
            continue
        if not (0 <= sp <= ep < len(tokens)):
            errs.append(f"row {i} ({r['id']}): invalid token span [{sp}, {ep}] n={len(tokens)}")
            continue
        if tokens[sp] != sw or tokens[ep] != ew:
            errs.append(
                f"row {i} ({r['id']}): word checksum failed "
                f"(start {sw!r} vs {tokens[sp]!r}, end {ew!r} vs {tokens[ep]!r})"
            )
            continue
        # Prefer char_start/char_end; fall back to gold_start/gold_end
        cs = r.get("char_start", r.get("gold_start"))
        ce = r.get("char_end", r.get("gold_end"))
        if cs is not None and ce is not None:
            cs, ce = int(cs), int(ce)
            if not (0 <= cs < ce <= len(para)):
                errs.append(f"row {i} ({r['id']}): invalid char offsets [{cs}, {ce})")
            elif para[cs:ce] != r["gold"]:
                errs.append(
                    f"row {i} ({r['id']}): paragraph[char_start:char_end] != gold "
                    f"({para[cs:ce]!r} != {r['gold']!r})"
                )
    return errs


# ---------------------------------------------------------------------------
# Summaries + tables
# ---------------------------------------------------------------------------


@dataclass
class TaskSummary:
    task: str
    backend: str
    n: int
    errors: int
    sum_input_tokens: int
    sum_output_tokens: int
    total_usd: float
    p50_ms: float
    p95_ms: float
    mean_ms: float
    # classification
    accuracy: float | None = None
    # extraction (primary: token span; secondary: token F1/IoU; optional string)
    token_em: float | None = None
    token_f1: float | None = None
    token_iou: float | None = None
    exact_match: float | None = None
    span_f1: float | None = None
    contains_gold: float | None = None
    resolved_models: list[str] = field(default_factory=list)
    pricing_note: str = (
        f"TypeSafe Jev: ${TYPESAFE_INPUT_USD_PER_MILLION}/M input tokens, output free"
    )
    skipped: bool = False
    skip_reason: str = ""


def format_classification_table(s: TaskSummary) -> str:
    header = (
        "| backend | n | accuracy | p50_ms | p95_ms | mean_ms | "
        "sum_input_tokens | sum_output_tokens | total_usd | errors |\n"
        "|---------|---|----------:|-------:|-------:|--------:|"
        "-----------------:|------------------:|----------:|-------:|"
    )
    if s.skipped:
        return header + f"\n| {s.backend} | — | — | — | — | — | — | — | — | skipped: {s.skip_reason} |"
    assert s.accuracy is not None
    return (
        header
        + f"\n| {s.backend} | {s.n} | {s.accuracy:.3f} | {s.p50_ms:.1f} | "
        f"{s.p95_ms:.1f} | {s.mean_ms:.1f} | {s.sum_input_tokens} | "
        f"{s.sum_output_tokens} | {s.total_usd:.6f} | {s.errors} |"
    )


def format_extraction_table(s: TaskSummary) -> str:
    header = (
        "| backend | n | token_em | token_f1 | token_iou | exact_match | contains_gold | "
        "p50_ms | p95_ms | mean_ms | sum_input_tokens | sum_output_tokens | total_usd | errors |\n"
        "|---------|---|---------:|---------:|----------:|------------:|--------------:|"
        "-------:|-------:|--------:|-----------------:|------------------:|----------:|-------:|"
    )
    if s.skipped:
        return (
            header
            + f"\n| {s.backend} | — | — | — | — | — | — | — | — | — | — | — | — | "
            f"skipped: {s.skip_reason} |"
        )
    assert s.token_em is not None and s.token_f1 is not None and s.token_iou is not None
    assert s.exact_match is not None and s.contains_gold is not None
    return (
        header
        + f"\n| {s.backend} | {s.n} | {s.token_em:.3f} | {s.token_f1:.3f} | {s.token_iou:.3f} | "
        f"{s.exact_match:.3f} | {s.contains_gold:.3f} | {s.p50_ms:.1f} | {s.p95_ms:.1f} | "
        f"{s.mean_ms:.1f} | {s.sum_input_tokens} | {s.sum_output_tokens} | "
        f"{s.total_usd:.6f} | {s.errors} |"
    )


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


