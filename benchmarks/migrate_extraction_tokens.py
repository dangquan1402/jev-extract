#!/usr/bin/env python3
"""Migrate extraction_50.jsonl to token-native gold spans (tokenizer v1).

For each item: tokenize paragraph, map gold char span to covering tokens,
derive gold answer from the original character slice between first/last token,
and rewrite the JSONL. Prints a change report to stdout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parent
sys.path.insert(0, str(REPO / "src"))

from jev_extract.tokenize import (  # noqa: E402
    TOKENIZER_VERSION,
    answer_from_token_span,
    char_span_to_token_span,
    tokenize_with_spans,
)

SRC = BENCH / "data" / "extraction_50.jsonl"
OUT = SRC  # in-place
BACKUP = BENCH / "data" / "extraction_50_char_legacy.jsonl"


def migrate_row(row: dict) -> tuple[dict, dict]:
    paragraph = row["paragraph"]
    old_gold = row["gold"]
    old_start = int(row["gold_start"])
    old_end = int(row["gold_end"])

    tokens, char_spans = tokenize_with_spans(paragraph)
    if paragraph[old_start:old_end] != old_gold:
        raise ValueError(
            f"{row['id']}: paragraph[gold_start:gold_end] != gold "
            f"({paragraph[old_start:old_end]!r} != {old_gold!r})"
        )

    t_start, t_end = char_span_to_token_span(char_spans, old_start, old_end)
    new_gold, c_start, c_end = answer_from_token_span(
        paragraph, tokens, char_spans, t_start, t_end
    )

    note = None
    if new_gold != old_gold:
        note = {"old_gold": old_gold, "new_gold": new_gold, "reason": "token-aligned slice"}

    out = {
        "id": row["id"],
        "paragraph": paragraph,
        "question": row["question"],
        "tokens": tokens,
        "start": {"pos": t_start, "word": tokens[t_start]},
        "end": {"pos": t_end, "word": tokens[t_end]},
        "gold": new_gold,
        "char_start": c_start,
        "char_end": c_end,
        # Keep legacy keys for optional string metrics / debugging
        "gold_start": c_start,
        "gold_end": c_end,
        "tokenizer": TOKENIZER_VERSION,
    }
    meta = {
        "id": row["id"],
        "token_span": [t_start, t_end],
        "n_tokens": len(tokens),
        "gold_changed": new_gold != old_gold,
        "note": note,
    }
    return out, meta


def verify_row(row: dict) -> None:
    tokens = row["tokens"]
    paragraph = row["paragraph"]
    _, char_spans = tokenize_with_spans(paragraph)
    assert tokens == [paragraph[s:e] for s, e in char_spans], "token/char map mismatch"
    sp, ep = int(row["start"]["pos"]), int(row["end"]["pos"])
    assert row["start"]["word"] == tokens[sp]
    assert row["end"]["word"] == tokens[ep]
    gold_slice = paragraph[char_spans[sp][0] : char_spans[ep][1]]
    assert gold_slice == row["gold"], (
        f"{row['id']}: reconstructed {gold_slice!r} != gold {row['gold']!r}"
    )
    assert row["char_start"] == char_spans[sp][0]
    assert row["char_end"] == char_spans[ep][1]


def main() -> int:
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Backup legacy char-offset format once
    if not BACKUP.exists():
        BACKUP.write_text(SRC.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Backed up legacy dataset → {BACKUP}")

    migrated = []
    changed = []
    for row in rows:
        # Support re-run on already-migrated rows
        if "tokens" in row and "start" in row and isinstance(row["start"], dict):
            verify_row(row)
            migrated.append(row)
            continue
        out, meta = migrate_row(row)
        verify_row(out)
        migrated.append(out)
        if meta["gold_changed"]:
            changed.append(meta)

    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in migrated) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(migrated)} rows → {OUT}")
    print(f"Gold text changes: {len(changed)}")
    for m in changed[:20]:
        print(f"  - {m['id']}: {m['note']}")
    # Stats
    lens = [len(r["tokens"]) for r in migrated]
    print(f"Token counts: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)}")
    over = sum(1 for n in lens if n > 255)
    print(f"Rows with >255 tokens: {over}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
