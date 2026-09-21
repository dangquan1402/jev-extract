"""Interval IoU / overlap metrics for sponsor segments (seconds)."""
from __future__ import annotations

from typing import Sequence


Interval = tuple[float, float]


def _clip(iv: Interval) -> Interval | None:
    a, b = float(iv[0]), float(iv[1])
    if b <= a:
        return None
    return (a, b)


def merge_intervals(intervals: Sequence[Interval], *, gap: float = 1.0) -> list[Interval]:
    """Merge overlapping / nearly-adjacent intervals (gap seconds)."""
    cleaned = [x for x in (_clip(i) for i in intervals) if x is not None]
    if not cleaned:
        return []
    cleaned.sort()
    out = [cleaned[0]]
    for a, b in cleaned[1:]:
        la, lb = out[-1]
        if a <= lb + gap:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out


def interval_overlap(a: Interval, b: Interval) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    return max(0.0, hi - lo)


def interval_iou(a: Interval, b: Interval) -> float:
    inter = interval_overlap(a, b)
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def total_overlap(pred: Sequence[Interval], gold: Sequence[Interval]) -> float:
    p = merge_intervals(pred)
    g = merge_intervals(gold)
    return sum(interval_overlap(a, b) for a in p for b in g)


def total_duration(intervals: Sequence[Interval]) -> float:
    return sum(b - a for a, b in merge_intervals(intervals))


def segment_metrics(
    pred: Sequence[Interval],
    gold: Sequence[Interval],
) -> dict[str, float | bool | int]:
    """Compute overlap / IoU style metrics between predicted and gold intervals."""
    p = merge_intervals(pred)
    g = merge_intervals(gold)
    ov = total_overlap(p, g)
    pred_dur = total_duration(p)
    gold_dur = total_duration(g)
    union = pred_dur + gold_dur - ov
    iou = (ov / union) if union > 0 else (1.0 if not p and not g else 0.0)
    precision = (ov / pred_dur) if pred_dur > 0 else (1.0 if gold_dur == 0 else 0.0)
    recall = (ov / gold_dur) if gold_dur > 0 else (1.0 if pred_dur == 0 else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    # Best-match mean IoU (pred→gold)
    best_ious: list[float] = []
    for pi in p:
        best_ious.append(max((interval_iou(pi, gi) for gi in g), default=0.0))
    mean_best_iou = sum(best_ious) / len(best_ious) if best_ious else (1.0 if not g else 0.0)

    video_recall = bool(g) and bool(p) and ov > 0
    video_has_gold = bool(g)
    video_detected = bool(p)

    # False-positive seconds on content-only stretches = pred not overlapping gold
    fp_sec = max(0.0, pred_dur - ov)
    content_sec_proxy = None  # filled by caller with video duration

    return {
        "n_pred": len(p),
        "n_gold": len(g),
        "overlap_sec": ov,
        "pred_dur_sec": pred_dur,
        "gold_dur_sec": gold_dur,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_best_iou": mean_best_iou,
        "video_recall_hit": video_recall if video_has_gold else (not video_detected),
        "video_has_gold": video_has_gold,
        "video_detected": video_detected,
        "fp_sec": fp_sec,
    }


def content_false_positive_rate(
    pred: Sequence[Interval],
    gold: Sequence[Interval],
    video_duration: float,
) -> float:
    """Fraction of non-gold (content) time incorrectly labeled sponsor."""
    g = merge_intervals(gold)
    gold_dur = total_duration(g)
    content_dur = max(0.0, float(video_duration) - gold_dur)
    if content_dur <= 0:
        return 0.0
    ov = total_overlap(pred, g)
    pred_dur = total_duration(pred)
    fp = max(0.0, pred_dur - ov)
    return fp / content_dur
