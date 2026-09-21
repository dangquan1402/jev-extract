"""Tony-style labelled transcript lines (port of youtube-sponsor-detection/src/transcript.js).

Jev picks a line ID; code maps ID → seconds. Cue-level times only (no word timing
in our cached VTT), so phrases use equal character splits within a line when needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

LINE_MIN_CHARS = 70
LINE_MAX_SECONDS = 8.0
WINDOW_LINES = 80
WINDOW_OVERLAP = 6
PHRASE_WORDS = 3
MAX_WORD_SECONDS = 1.5


@dataclass
class Line:
    id: str
    text: str
    start: float
    end: float
    words: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Phrase:
    id: str
    line_id: str
    text: str
    start: float
    end: float


def _round(n: float) -> float:
    return float(f"{n:.2f}")


def _cue_words(cue: Mapping[str, Any], text: str) -> list[dict[str, Any]]:
    """Share cue span across tokens by character count (no word-level VTT)."""
    start = float(cue["start"])
    end = max(start, float(cue["end"]))
    tokens = text.split()
    if not tokens:
        return []
    chars = sum(len(t) + 1 for t in tokens)
    words: list[dict[str, Any]] = []
    at = 0
    for i, tok in enumerate(tokens):
        w_start = start + ((end - start) * at) / chars
        at += len(tok) + 1
        next_start = (
            start + ((end - start) * at) / chars if i + 1 < len(tokens) else end
        )
        w_end = min(next_start, w_start + MAX_WORD_SECONDS)
        words.append(
            {"text": tok, "start": _round(w_start), "end": _round(max(w_start, w_end))}
        )
    return words


def build_lines(cues: Sequence[Mapping[str, Any]]) -> list[Line]:
    """Merge cues into ~sentence lines labelled L001, L002, …"""
    lines: list[Line] = []
    buf: dict[str, Any] | None = None

    def full() -> bool:
        if not buf:
            return False
        joined = " ".join(buf["parts"])
        return len(joined) >= LINE_MIN_CHARS or (buf["end"] - buf["start"]) >= LINE_MAX_SECONDS

    def finish() -> None:
        nonlocal buf
        assert buf is not None
        idx = len(lines)
        lines.append(
            Line(
                id=f"L{idx + 1:03d}",
                text=" ".join(buf["parts"]).replace("  ", " ").strip(),
                start=_round(float(buf["start"])),
                end=_round(float(buf["end"])),
                words=list(buf["words"]),
            )
        )
        buf = None

    for cue in cues:
        text = " ".join(str(cue.get("text") or "").split()).strip()
        if not text:
            continue
        if full():
            finish()
        if buf is None:
            buf = {
                "start": float(cue["start"]),
                "end": float(cue["end"]),
                "parts": [],
                "words": [],
            }
        buf["parts"].append(text)
        buf["words"].extend(_cue_words(cue, text))
        buf["end"] = float(cue["end"])
    if buf:
        finish()
    return lines


def render_lines(lines: Sequence[Line]) -> str:
    return "\n".join(f"{l.id}| {l.text}" for l in lines)


def window_lines(
    lines: Sequence[Line],
    *,
    size: int = WINDOW_LINES,
    overlap: int = WINDOW_OVERLAP,
) -> list[list[Line]]:
    if len(lines) <= size:
        return [list(lines)]
    step = max(1, size - overlap)
    windows: list[list[Line]] = []
    for start in range(0, len(lines), step):
        windows.append(list(lines[start : start + size]))
        if start + size >= len(lines):
            break
    return windows


def build_phrases(lines: Sequence[Line]) -> list[Phrase]:
    phrases: list[Phrase] = []
    for line in lines:
        words = line.words or [
            {"text": line.text, "start": line.start, "end": line.end}
        ]
        i = 0
        while i < len(words):
            last = i + PHRASE_WORDS >= len(words) - 1
            chunk = words[i:] if last else words[i : i + PHRASE_WORDS]
            phrases.append(
                Phrase(
                    id=f"P{len(phrases) + 1:02d}",
                    line_id=line.id,
                    text=" ".join(w["text"] for w in chunk),
                    start=float(chunk[0]["start"]),
                    end=float(chunk[-1]["end"]),
                )
            )
            if last:
                break
            i += PHRASE_WORDS
    return phrases


def estimate_tokens(text: str) -> int:
    return (len(text) + 3) // 4
