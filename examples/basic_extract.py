"""Minimal extractive QA example (requires TYPESAFE_API_KEY)."""

from __future__ import annotations

import os
import sys

from jev_extract import Extractor, MissingAPIKeyError


def main() -> int:
    if not os.environ.get("TYPESAFE_API_KEY"):
        print("Set TYPESAFE_API_KEY before running this example.", file=sys.stderr)
        return 1

    paragraph = (
        "Marie Curie discovered radium and polonium. "
        "She won the Nobel Prize in Physics in 1903. "
        "Later she won a second Nobel Prize in Chemistry in 1911."
    )

    try:
        with Extractor() as ex:
            result = ex.extract(
                paragraph=paragraph,
                question="In what year did Marie Curie win the Nobel Prize in Physics?",
            )
    except MissingAPIKeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"answer:      {result.answer!r}")
    print(f"offsets:     [{result.start}, {result.end})")
    print(f"confidence:  {result.confidence}")
    print(f"model:       {result.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
