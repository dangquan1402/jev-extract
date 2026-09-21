# Sponsor Skip (English)

Detect paid sponsor mid-rolls in **English** YouTube transcripts with TypeSafe Jev
`Choice` / `Noul`. Gold = SponsorBlock. All code lives under this folder —
**core `jev_extract` IE defaults are not modified.**

## Language note

**English-only for v1.** Regex cue patterns and Jev prompts are tuned for English
paid-read language (`segue to our sponsor`, `use code`, `link in description`, …).
Non-English captions (e.g. Vietnamese auto-captions on Tony’s eval set) are out of
scope for this default and will under-fire.

## Recommended default (EN v1)

**`regex_jev_refined`** = `regex_propose_jev_confirm` + boundary snap + confidence gate.

| Mode | Role |
|------|------|
| `regex_propose_jev_confirm` | Baseline: strong regex proposes windows → Jev confirms chunks → expand |
| `regex_jev_refined` | **Recommended** — same confirm, then start/end snap + per-segment confidence |
| `tony_line_scan` | Regex-free labelled-line scan (higher cost) |
| `chunk_batched` | Classify every ~15s chunk (weak baseline) |

### Confidence gate

After confirm, each segment gets Jev Choice confidence (`probabilities[label]` or
`confidence`). Configurable threshold (default **0.75**):

- `confidence >= threshold` → `auto_skip=True` (safe to skip)
- below → `auto_skip=False` / `needs_confirm=True` (still returned; ask the user)

Bench reports **all preds** IoU/P/R and **high-conf only** (auto-skip set) when available.

### Boundary snap

- **Start:** prefer lead-in / first strong CTA near the proposed start; include a clear
  soft lead-in within ~15–30s before; if unsure, start slightly late into the read.
- **End:** snap to return-to-content cues (`back to the video/show`, `anyway`,
  `with that said`, `now (back )?to`, `before we go further`, …) or cap ~5–15s after
  the last offer CTA (`use code`, `link in description`, …). Prefer ending slightly
  late over cutting the ad short.

## Run LTT-6 A/B

```bash
# from repo root, venv active; TYPESAFE_API_KEY via load_card_secrets / box-secrets
python benchmarks/sponsor_skip/run_sponsor_skip_bench.py \
  --limit 6 \
  --modes regex_propose_jev_confirm regex_jev_refined \
  --confidence-threshold 0.75 \
  --video-ids qgLaCZyKv_8 zg-rMHEqg-4 3xngArcFpek 3PGKMwgjla0 i_nj_vkD03g pvSdeU13hKc
```

Cached transcripts/gold under `data/` are reused (no network required for those).

## Paths

- Runner: `run_sponsor_skip_bench.py`
- Detectors: `detect.py`
- Regex / snap / gate: `regex_propose.py`
- Report: `SPONSOR_SKIP_REPORT.md`
- Demo: `demo/`
