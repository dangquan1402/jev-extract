"""Fetch SponsorBlock gold labels and YouTube timed transcripts (via yt-dlp VTT)."""
from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SPONSORBLOCK_BASE = "https://sponsor.ajay.app/api/skipSegments"
DEFAULT_CATEGORIES = ("sponsor", "selfpromo")

# Curated candidates: recent English tech/creator videos known to carry mid-roll
# sponsors (discovered via channel uploads × SponsorBlock probe, Sep 2026).
CANDIDATE_VIDEO_IDS: list[str] = [
    "qgLaCZyKv_8",
    "zg-rMHEqg-4",
    "3xngArcFpek",
    "3PGKMwgjla0",
    "i_nj_vkD03g",
    "pvSdeU13hKc",
    "VfQdPRyqQW4",
    "sL6OWsT47zc",
    "2KztkwBYD68",
    "nZwdIPg7N3A",
    "Dznsr0KN1XU",
    "2q5x8pdLRpo",
]


@dataclass
class Cue:
    start: float
    end: float
    text: str


@dataclass
class GoldSegment:
    start: float
    end: float
    category: str
    uuid: str | None = None


_TS = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})"
)
_TAG = re.compile(r"<[^>]+>")
_CUE_SETTINGS = re.compile(r"\s+(?:align|position|size|line|vertical):[^\s]+")


def _ts_to_sec(h: str | None, m: str, s: str, ms: str) -> float:
    hours = int(h or 0)
    return hours * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(path: Path) -> list[Cue]:
    """Parse a WebVTT file into cleaned cues (YouTube auto-caption aware).

    YouTube auto VTTs use roll-up: each cue repeats the previous phrase plus new
    ``<c>``-timed words, then a ~10ms "commit" cue. We keep only payloads from
    timed ``<c>`` words (new content), skip ultra-short commit echoes, and drop
    consecutive duplicates.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", raw)
    cues: list[Cue] = []
    prev_text = ""
    for block in blocks:
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        ts_idx = None
        for i, ln in enumerate(lines):
            if "-->" in ln:
                ts_idx = i
                break
        if ts_idx is None:
            continue
        m = _TS.search(lines[ts_idx])
        if not m:
            continue
        start = _ts_to_sec(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _ts_to_sec(m.group(5), m.group(6), m.group(7), m.group(8))
        if end - start < 0.05:
            # Commit echo of previous phrase — skip
            continue
        text_lines = lines[ts_idx + 1 :]
        timed = [ln for ln in text_lines if "<c>" in ln or re.search(r"<\d", ln)]
        if timed:
            # New words only: strip tags from timed lines (drop untimed carryover)
            parts = [_TAG.sub("", ln) for ln in timed]
        else:
            # Manual captions: use all lines
            parts = [_TAG.sub("", ln) for ln in text_lines]
        text = " ".join(parts)
        text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
        text = text.replace(">>", " ").strip()
        text = re.sub(r"\s+", " ", text)
        if not text or text == prev_text:
            continue
        if text.lower() in {"[music]", "[applause]", "[laughter]"}:
            continue
        cues.append(Cue(start=start, end=end, text=text))
        prev_text = text
    return cues


def fetch_sponsorblock(
    video_id: str,
    *,
    categories: tuple[str, ...] = DEFAULT_CATEGORIES,
    timeout: float = 20.0,
) -> list[GoldSegment]:
    """Return gold skip segments for *video_id* (empty if none / 404)."""
    segs: list[GoldSegment] = []
    for cat in categories:
        qs = urllib.parse.urlencode({"videoID": video_id, "category": cat})
        url = f"{SPONSORBLOCK_BASE}?{qs}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            raise
        for item in data:
            a, b = item["segment"]
            segs.append(
                GoldSegment(
                    start=float(a),
                    end=float(b),
                    category=str(item.get("category") or cat),
                    uuid=item.get("UUID"),
                )
            )
    segs.sort(key=lambda s: (s.start, s.end))
    return segs


def download_vtt(video_id: str, dest_dir: Path, *, timeout: int = 90) -> Path | None:
    """Download English (auto) subtitles with yt-dlp; return path or None."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(dest_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--write-auto-sub",
        "--write-sub",
        "--sub-lang",
        "en",
        "--skip-download",
        "--sub-format",
        "vtt",
        "-o",
        outtmpl,
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    # Prefer manual en over auto
    candidates = sorted(dest_dir.glob(f"{video_id}*.vtt"))
    if not candidates:
        return None
    # Prefer *.en.vtt without .en-orig if both exist
    for p in candidates:
        if p.name.endswith(".en.vtt"):
            return p
    return candidates[0]


def chunk_cues(
    cues: list[Cue],
    *,
    target_sec: float = 15.0,
    max_sec: float = 22.0,
) -> list[dict[str, Any]]:
    """Merge consecutive cues into ~target_sec windows with start/end timestamps."""
    if not cues:
        return []
    chunks: list[dict[str, Any]] = []
    buf: list[Cue] = []
    buf_start = cues[0].start

    def flush() -> None:
        nonlocal buf, buf_start
        if not buf:
            return
        text = " ".join(c.text for c in buf).strip()
        chunks.append(
            {
                "index": len(chunks),
                "start": buf_start,
                "end": buf[-1].end,
                "text": text,
                "n_cues": len(buf),
            }
        )
        buf = []

    for cue in cues:
        if not buf:
            buf = [cue]
            buf_start = cue.start
            continue
        projected_end = cue.end
        span = projected_end - buf_start
        if span > max_sec and buf:
            flush()
            buf = [cue]
            buf_start = cue.start
        elif span >= target_sec and (projected_end - buf_start) >= target_sec * 0.7:
            buf.append(cue)
            flush()
        else:
            buf.append(cue)
    flush()
    return chunks


def cues_to_sentences(cues: list[Cue], *, max_sentences: int = 255) -> list[dict[str, Any]]:
    """Build sentence-like units from cues (split on [.!?] when possible)."""
    sentences: list[dict[str, Any]] = []
    buf_text: list[str] = []
    buf_start: float | None = None
    buf_end: float = 0.0

    def flush() -> None:
        nonlocal buf_text, buf_start, buf_end
        if not buf_text or buf_start is None:
            return
        text = " ".join(buf_text).strip()
        if text:
            sentences.append(
                {
                    "index": len(sentences),
                    "start": buf_start,
                    "end": buf_end,
                    "text": text,
                }
            )
        buf_text = []
        buf_start = None

    for cue in cues:
        if buf_start is None:
            buf_start = cue.start
        buf_text.append(cue.text)
        buf_end = cue.end
        joined = " ".join(buf_text)
        if re.search(r"[.!?…][\"')\]]?\s*$", joined) and len(joined) > 40:
            flush()
        elif len(joined) > 180:
            flush()
    flush()
    if len(sentences) > max_sentences:
        # Merge evenly into max_sentences buckets
        step = len(sentences) / max_sentences
        merged: list[dict[str, Any]] = []
        i = 0.0
        while int(i) < len(sentences) and len(merged) < max_sentences:
            a = int(i)
            b = max(a + 1, int(i + step))
            group = sentences[a:b]
            merged.append(
                {
                    "index": len(merged),
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "text": " ".join(g["text"] for g in group),
                }
            )
            i += step
        sentences = merged
    return sentences


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def gold_to_dict(segs: list[GoldSegment]) -> list[dict[str, Any]]:
    return [asdict(s) for s in segs]
