#!/usr/bin/env python3
"""Regenerate figures from the latest typesafe result JSON files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"


def latest(pattern: str) -> Path:
    files = sorted(RESULTS.glob(pattern))
    if not files:
        raise SystemExit(f"No files matching {pattern}")
    return files[-1]


def non_warmup(results: list[dict]) -> list[dict]:
    return [r for r in results if not r.get("warmup")]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cls = json.loads(latest("classification_typesafe_*.json").read_text())
    ext = json.loads(latest("extraction_typesafe_*.json").read_text())
    cls_r = non_warmup(cls["results"])
    ext_r = non_warmup(ext["results"])

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "font.size": 11,
        }
    )

    cls_lat = [r["latency_ms"] for r in cls_r]
    ext_lat = [r["latency_ms"] for r in ext_r]
    costs = [
        sum(r.get("estimated_usd") or 0 for r in cls_r),
        sum(r.get("estimated_usd") or 0 for r in ext_r),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "TypeSafe Jev (jev-1.13.0) — Comparison Bench Overview",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0, 0]
    metrics = {
        "Classification\naccuracy": sum(1 for r in cls_r if r.get("correct")) / len(cls_r),
        "Extraction\ntoken EM": sum(1 for r in ext_r if r.get("token_em")) / len(ext_r),
        "Extraction\ntoken F1": sum(r.get("token_f1") or 0 for r in ext_r) / len(ext_r),
        "Extraction\ntoken IoU": sum(r.get("token_iou") or 0 for r in ext_r) / len(ext_r),
    }
    colors = ["#2563eb", "#dc2626", "#ea580c", "#16a34a"]
    bars = ax.bar(list(metrics.keys()), list(metrics.values()), color=colors, width=0.65)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Quality metrics (n=50 each)")
    for b, v in zip(bars, metrics.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.1%}", ha="center", va="bottom", fontsize=10)

    ax = axes[0, 1]
    bp = ax.boxplot([cls_lat, ext_lat], tick_labels=["Classification", "Extraction"], patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#93c5fd", "#fdba74"]):
        patch.set_facecolor(c)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency distribution")

    ax = axes[1, 0]
    x = np.arange(2)
    width = 0.35
    inp = [sum(r["input_tokens"] or 0 for r in cls_r), sum(r["input_tokens"] or 0 for r in ext_r)]
    outt = [sum(r["output_tokens"] or 0 for r in cls_r), sum(r["output_tokens"] or 0 for r in ext_r)]
    b1 = ax.bar(x - width / 2, inp, width, label="Input tokens", color="#2563eb")
    b2 = ax.bar(x + width / 2, outt, width, label="Output tokens", color="#64748b")
    ax.set_xticks(x)
    ax.set_xticklabels(["Classification", "Extraction"])
    ax.set_ylabel("Total tokens (n=50)")
    ax.set_title("Token usage")
    ax.legend(frameon=False)
    for bars_ in (b1, b2):
        for b in bars_:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{int(b.get_height()):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax = axes[1, 1]
    bars = ax.bar(["Classification", "Extraction"], costs, color=["#2563eb", "#ea580c"], width=0.5)
    ax.set_ylabel("Estimated USD")
    ax.set_title("Cost @ $0.042 / M input (output free)")
    for b, v in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, v, f"${v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, max(costs) * 1.35)
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Latency detail — TypeSafe Jev", fontsize=13, fontweight="bold")
    ax = axes[0]
    for label, data, color in [
        ("Classification", cls_lat, "#2563eb"),
        ("Extraction", ext_lat, "#ea580c"),
    ]:
        s = np.sort(data)
        y = np.arange(1, len(s) + 1) / len(s)
        ax.plot(s, y, color=color, label=label, linewidth=2)
        ax.axvline(float(np.median(data)), color=color, linestyle=":", alpha=0.7)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF (dotted = p50)")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.scatter(range(len(cls_lat)), cls_lat, s=18, alpha=0.7, c="#2563eb", label="Classification")
    ax.scatter(range(len(ext_lat)), ext_lat, s=18, alpha=0.7, c="#ea580c", label="Extraction")
    ax.set_xlabel("Example index")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-example latency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "latency.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Extraction quality — token-native start/end Choice", fontsize=13, fontweight="bold")
    ax = axes[0]
    f1s = [r.get("token_f1") or 0 for r in ext_r]
    ax.hist(f1s, bins=12, color="#ea580c", edgecolor="white", range=(0, 1))
    ax.axvline(float(np.mean(f1s)), color="#7c2d12", linestyle="--", label=f"mean={np.mean(f1s):.2f}")
    ax.set_xlabel("Token F1")
    ax.set_ylabel("Count")
    ax.set_title("Token-span F1 distribution")
    ax.legend(frameon=False)
    ax = axes[1]
    em = sum(1 for r in ext_r if r.get("token_em"))
    partial = sum(1 for r in ext_r if not r.get("token_em") and (r.get("token_f1") or 0) > 0)
    miss = sum(1 for r in ext_r if (r.get("token_f1") or 0) == 0)
    vals = [em, partial, miss]
    bars = ax.bar(
        ["Token EM", "Partial overlap", "Miss"],
        vals,
        color=["#16a34a", "#2563eb", "#dc2626"],
        width=0.6,
    )
    ax.set_ylabel("Examples (n=50)")
    ax.set_title("Outcome breakdown")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "extraction_quality.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax2 = ax.twinx()
    p50 = [float(np.median(cls_lat)), float(np.median(ext_lat))]
    b1 = ax.bar(np.arange(2) - 0.15, p50, 0.3, color="#2563eb", label="p50 latency (ms)")
    b2 = ax2.bar(np.arange(2) + 0.15, costs, 0.3, color="#16a34a", label="total USD")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Classification", "Extraction"])
    ax.set_ylabel("p50 latency (ms)", color="#2563eb")
    ax2.set_ylabel("Total USD (n=50)", color="#16a34a")
    ax.set_title("Speed vs cost — TypeSafe Jev")
    ax.legend([b1, b2], ["p50 latency (ms)", "total USD"], loc="upper center", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "speed_vs_cost.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()
