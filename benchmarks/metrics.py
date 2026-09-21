"""Benchmark metrics: normalization, accuracy, token-span overlap, latency, cost."""

from __future__ import annotations

import re
import statistics
from typing import Iterable, Sequence

# TypeSafe Jev public pricing (document + compute): input billed, output free.
TYPESAFE_INPUT_USD_PER_MILLION = 0.042
TYPESAFE_OUTPUT_USD_PER_MILLION = 0.0

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return _WS.sub(" ", (text or "").strip().lower())


def tokenize(text: str) -> list[str]:
    """Lightweight alphanumeric tokenizer for *string* F1 (legacy / optional)."""
    return re.findall(r"[a-z0-9]+", normalize(text))


def exact_match(gold: str, pred: str) -> bool:
    return normalize(gold) == normalize(pred)


def contains_gold(gold: str, pred: str) -> bool:
    g, p = normalize(gold), normalize(pred)
    if not g or not p:
        return g == p
    return g in p or p in g


def span_f1(gold: str, pred: str) -> float:
    """Token-level multiset overlap F1 between gold and predicted *strings*."""
    g_toks = tokenize(gold)
    p_toks = tokenize(pred)
    if not g_toks and not p_toks:
        return 1.0
    if not g_toks or not p_toks:
        return 0.0
    g_counts: dict[str, int] = {}
    p_counts: dict[str, int] = {}
    for t in g_toks:
        g_counts[t] = g_counts.get(t, 0) + 1
    for t in p_toks:
        p_counts[t] = p_counts.get(t, 0) + 1
    overlap = sum(min(g_counts[t], p_counts.get(t, 0)) for t in g_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_toks)
    recall = overlap / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def token_span_exact(
    gold_start: int,
    gold_end: int,
    pred_start: int | None,
    pred_end: int | None,
) -> bool:
    """Primary extraction metric: inclusive token-index exact match."""
    if pred_start is None or pred_end is None:
        return False
    return int(gold_start) == int(pred_start) and int(gold_end) == int(pred_end)


def token_span_iou(
    gold_start: int,
    gold_end: int,
    pred_start: int | None,
    pred_end: int | None,
) -> float:
    """IoU over inclusive token index sets."""
    if pred_start is None or pred_end is None:
        return 0.0
    if gold_end < gold_start or pred_end < pred_start:
        return 0.0
    g = set(range(int(gold_start), int(gold_end) + 1))
    p = set(range(int(pred_start), int(pred_end) + 1))
    if not g and not p:
        return 1.0
    inter = len(g & p)
    union = len(g | p)
    return inter / union if union else 0.0


def token_span_f1(
    gold_start: int,
    gold_end: int,
    pred_start: int | None,
    pred_end: int | None,
) -> float:
    """Precision/recall F1 over inclusive token index sets."""
    if pred_start is None or pred_end is None:
        return 0.0
    if gold_end < gold_start or pred_end < pred_start:
        return 0.0
    g = set(range(int(gold_start), int(gold_end) + 1))
    p = set(range(int(pred_start), int(pred_end) + 1))
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    inter = len(g & p)
    if inter == 0:
        return 0.0
    precision = inter / len(p)
    recall = inter / len(g)
    return 2 * precision * recall / (precision + recall)


def classification_correct(gold: str, pred: str) -> bool:
    return normalize(gold) == normalize(pred)


def accuracy(correct_flags: Sequence[bool]) -> float:
    if not correct_flags:
        return 0.0
    return sum(1 for c in correct_flags if c) / len(correct_flags)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile; ``p`` in [0, 100]."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    return {
        "p50_ms": percentile(latencies_ms, 50),
        "p95_ms": percentile(latencies_ms, 95),
        "mean_ms": float(statistics.mean(latencies_ms)),
    }


def estimated_usd(
    input_tokens: int | None,
    output_tokens: int | None = 0,
    *,
    input_usd_per_million: float = TYPESAFE_INPUT_USD_PER_MILLION,
    output_usd_per_million: float = TYPESAFE_OUTPUT_USD_PER_MILLION,
) -> float:
    """Estimate USD cost from token counts (TypeSafe Jev: output free by default)."""
    inp = int(input_tokens or 0)
    out = int(output_tokens or 0)
    return (inp * input_usd_per_million + out * output_usd_per_million) / 1_000_000.0


def sum_optional(values: Iterable[int | None]) -> int:
    total = 0
    for v in values:
        if v is not None:
            total += int(v)
    return total
