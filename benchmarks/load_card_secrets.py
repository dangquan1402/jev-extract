#!/usr/bin/env python3
"""Load TYPESAFE_API_KEY (and related) from box-secrets.json into the environment.

Usage:
  eval "$(python benchmarks/load_card_secrets.py)"
  # or
  python benchmarks/load_card_secrets.py --export   # prints export lines
  from benchmarks.load_card_secrets import load_card_secrets; load_card_secrets()
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CANDIDATE_PATHS = [
    Path(os.environ["JEV_EXTRACT_SECRETS_JSON"])
    if os.environ.get("JEV_EXTRACT_SECRETS_JSON")
    else None,
    Path("/home/box/agent-data/box-secrets.json"),
    Path("/home/box/sand-data/box-secrets.json"),
    Path(__file__).resolve().parent.parent / "box-secrets.json",
]


def find_secrets_path() -> Path | None:
    for p in CANDIDATE_PATHS:
        if p is not None and p.is_file():
            return p
    return None


def load_card_secrets(
    path: Path | str | None = None,
    *,
    set_env: bool = True,
) -> dict[str, str]:
    """Return card secrets dict; optionally set os.environ for known keys."""
    p = Path(path) if path else find_secrets_path()
    if p is None or not p.is_file():
        raise FileNotFoundError(
            "box-secrets.json not found. Set JEV_EXTRACT_SECRETS_JSON or place "
            "secrets at /home/box/agent-data/box-secrets.json"
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    card = data.get("card") or {}
    out: dict[str, str] = {}
    for key, val in card.items():
        if val is None:
            continue
        s = str(val).strip()
        if not s:
            continue
        out[str(key)] = s
        if set_env and key in (
            "TYPESAFE_API_KEY",
            "SUPABASE_SERVICE_ROLE_KEY",
            "REFERO_API_TOKEN",
        ):
            os.environ.setdefault(key, s)
    # Also set JEV_EXTRACT_SECRETS_JSON for _bench_types.load_typesafe_api_key
    if set_env:
        os.environ.setdefault("JEV_EXTRACT_SECRETS_JSON", str(p))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", type=Path, default=None)
    ap.add_argument(
        "--export",
        action="store_true",
        help="Print shell export lines for TYPESAFE_API_KEY (no other secrets).",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Exit 0 if TYPESAFE_API_KEY is loadable; print only 'ok' / 'missing'.",
    )
    args = ap.parse_args(argv)
    try:
        card = load_card_secrets(args.path, set_env=True)
    except FileNotFoundError as exc:
        if args.check:
            print("missing")
            return 1
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    key = card.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
    if args.check:
        print("ok" if key else "missing")
        return 0 if key else 1
    if args.export:
        if not key:
            print("ERROR: TYPESAFE_API_KEY absent in card", file=sys.stderr)
            return 2
        # Safe for eval: key is opaque token; still quote.
        print(f'export TYPESAFE_API_KEY={json.dumps(key)}')
        print(f'export JEV_EXTRACT_SECRETS_JSON={json.dumps(os.environ.get("JEV_EXTRACT_SECRETS_JSON", ""))}')
        return 0
    print(
        f"Loaded {len(card)} card keys from secrets "
        f"(TYPESAFE_API_KEY={'yes' if key else 'no'})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
