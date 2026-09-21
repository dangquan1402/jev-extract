# Sponsor Skip Bench — Jev Choice vs SponsorBlock

Generated: 2026-09-21 07:32 UTC

## Setup

- **Detector**: TypeSafe Jev closed-set `Choice` / `Noul` (no trained classifier).
- **Gold**: SponsorBlock public API (`sponsor` primary).
- **Pricing**: $0.042/M input tokens (output free).
- **Videos evaluated**: 6 (cached transcripts/gold reused).
- **Core `jev_extract` IE defaults**: unchanged — all work under `benchmarks/sponsor_skip/`.
- **Language**: **English-only v1** (see `README.md`).

### Videos (LTT-6)

| video_id | duration_s | gold sponsor segs | n_chunks | n_sentences |
|----------|------------|-------------------|----------|-------------|
| `qgLaCZyKv_8` | 1052 | 2 | 63 | 99 |
| `zg-rMHEqg-4` | 1198 | 2 | 74 | 171 |
| `3xngArcFpek` | 585 | 2 | 36 | 55 |
| `3PGKMwgjla0` | 1427 | 2 | 89 | 152 |
| `i_nj_vkD03g` | 1535 | 2 | 93 | 222 |
| `pvSdeU13hKc` | 1143 | 2 | 71 | 139 |

## English v1 quality upgrades — confidence gate + boundary snap

**Recommended default:** `regex_jev_refined` (= `regex_propose_jev_confirm` + start/end snap + confidence gate).
Baseline `regex_propose_jev_confirm` kept for A/B.

### What changed

1. **Confidence gate** (default threshold **0.75**): after Jev confirm, each segment gets
   Choice confidence (`probabilities[label]` else `confidence`). Below threshold →
   `auto_skip=False` / `needs_confirm=True` (still returned). Metrics report **all preds**
   and **high-conf only** (auto-skip set).
2. **Boundary snap**: start prefers lead-in / first strong CTA (soft lead-in within ~30s);
   end snaps to return-to-content cues (incl. ASR-split adjacent cues) or caps ~8s after
   offer CTA *start* (not roll-up cue end). Conservative: never cut the ad short; avoid
   weak early starts (Tony KEEP_CONTENT spirit).

### Macro comparison (this run)

| mode | n | IoU | P | R | F1 | video recall | content FP | tokens | USD | auto-skip segs | auto-skip coverage |
|------|---|-----|---|---|----|--------------|------------|--------|-----|----------------|--------------------|
| `regex_propose_jev_confirm` | 6 | 0.787 | 0.879 | 0.888 | 0.880 | 1.000 | 0.014 | 48751 | $0.0020 | — | — |
| `regex_jev_refined (all preds)` | 6 | 0.879 | 0.916 | 0.957 | 0.934 | 1.000 | 0.009 | 48751 | $0.0020 | 9/13 | 0.692 |
| `regex_jev_refined` high-conf only | 6 | 0.827 | 0.965 | 0.859 | 0.903 | — | 0.002 | 48751 | $0.0020 | 9/13 | 0.692 |

### vs prior LTT-6 baseline (IoU ~0.838)

| reference | IoU | P | R | F1 | USD |
|-----------|-----|---|---|----|-----|
| Prior `regex_propose_jev_confirm` (2026-09-21 05:24Z) | 0.838 | 0.902 | 0.920 | 0.910 | $0.0020 |
| This run `regex_propose_jev_confirm` | 0.787 | 0.879 | 0.888 | 0.880 | $0.0020 |
| This run **`regex_jev_refined`** | **0.879** | 0.916 | 0.957 | **0.934** | $0.0020 |

**Result:** refined IoU **0.879** ≥ prior **0.838** (+0.041); same token/cost as baseline confirm (snap/gate are free local post-process).

> Note: this-run baseline IoU (0.787) is below the prior 0.838 due to Jev Choice
> non-determinism (extra FP / missed intro on some videos). Refined still lifts
> this-run baseline by **+0.092 IoU** via boundary snap alone.

### Per-video — `regex_propose_jev_confirm`

| video | IoU | P | R | F1 | overlap_s | fp_s | content_fp | tokens | USD |
|-------|-----|---|---|----|-----------|------|------------|--------|-----|
| `qgLaCZyKv_8` | 0.747 | 0.792 | 0.930 | 0.855 | 116.3 | 30.6 | 0.0330 | 11388 | $0.0005 |
| `zg-rMHEqg-4` | 0.835 | 0.917 | 0.904 | 0.910 | 89.0 | 8.1 | 0.0073 | 8011 | $0.0003 |
| `3xngArcFpek` | 0.736 | 0.805 | 0.896 | 0.848 | 61.9 | 15.0 | 0.0290 | 6808 | $0.0003 |
| `3PGKMwgjla0` | 0.853 | 0.892 | 0.952 | 0.921 | 76.4 | 9.3 | 0.0069 | 7927 | $0.0003 |
| `i_nj_vkD03g` | 0.763 | 0.868 | 0.863 | 0.866 | 76.8 | 11.7 | 0.0081 | 6968 | $0.0003 |
| `pvSdeU13hKc` | 0.784 | 0.998 | 0.785 | 0.879 | 80.6 | 0.2 | 0.0002 | 7649 | $0.0003 |

### Per-video — `regex_jev_refined`

| video | IoU | P | R | F1 | auto_skip | needs_confirm | mean_conf | hc IoU | tokens | USD |
|-------|-----|---|---|----|-----------|---------------|-----------|--------|--------|-----|
| `qgLaCZyKv_8` | 0.772 | 0.797 | 0.961 | 0.871 | 1 | 2 | 0.67 | 0.778 | 11388 | $0.0005 |
| `zg-rMHEqg-4` | 0.865 | 0.953 | 0.904 | 0.928 | 1 | 1 | 0.71 | 0.692 | 8011 | $0.0003 |
| `3xngArcFpek` | 0.965 | 0.992 | 0.973 | 0.982 | 2 | 0 | 1.00 | 0.965 | 6808 | $0.0003 |
| `3PGKMwgjla0` | 0.853 | 0.892 | 0.952 | 0.921 | 2 | 0 | 1.00 | 0.853 | 7927 | $0.0003 |
| `i_nj_vkD03g` | 0.888 | 0.912 | 0.971 | 0.941 | 2 | 0 | 0.98 | 0.888 | 6968 | $0.0003 |
| `pvSdeU13hKc` | 0.928 | 0.949 | 0.978 | 0.963 | 1 | 1 | 0.74 | 0.784 | 7649 | $0.0003 |

### Confidence gate summary (`regex_jev_refined`, threshold=0.75)

- Segments predicted: **13**
- Auto-skip (conf ≥ 0.75): **9** (0.692)
- Needs confirm: **4**
- High-conf-only macro IoU / F1: **0.827** / **0.903**

Low-confidence segments are still returned for UI “ask” — they are not dropped.

## Conclusions

- **`regex_jev_refined` is the English v1 recommended default** — IoU **0.879** / F1 **0.934**
  at the same ~$0.002 cost as regex→Jev confirm.
- Boundary snap recovers short-intro overshoots (e.g. `3xngArcFpek` IoU 0.736→0.965)
  and mid-roll ends via adjacent-cue end phrases.
- Confidence gate documents auto-skip coverage without deleting low-conf preds.
- Keep `regex_propose_jev_confirm` for A/B; `tony_line_scan` remains the regex-free alternative.

## Blockers

- youtube-transcript-api blocked from this host; cached yt-dlp VTT transcripts reused.
- Jev Choice labels are mildly non-deterministic across runs (baseline IoU fluctuates ~0.79–0.84).
- English-only regex cues; non-English captions out of scope for v1.
- No changes to core `jev_extract` IE defaults.

## File paths

- Experiment root: `/workspace/jev-extract/benchmarks/sponsor_skip`
- README (English-only note): `/workspace/jev-extract/benchmarks/sponsor_skip/README.md`
- Runner: `/workspace/jev-extract/benchmarks/sponsor_skip/run_sponsor_skip_bench.py`
- Snap + gate: `/workspace/jev-extract/benchmarks/sponsor_skip/regex_propose.py`
- Detector: `/workspace/jev-extract/benchmarks/sponsor_skip/detect.py` (`regex_jev_refined`)
- Results JSON: `/workspace/jev-extract/benchmarks/sponsor_skip/results/sponsor_skip_20260921T073219Z.json`
- Mode results: `results_regex_propose_jev_confirm.json`, `results_regex_jev_refined.json`
- This report: `/workspace/jev-extract/benchmarks/sponsor_skip/SPONSOR_SKIP_REPORT.md`

## Prior notes (Tony eval / Tony line-scan A/B)

Earlier LTT-6 A/B (regex vs tony_line_scan vs chunk) and Tony Dinh eval (n=19) remain
summarized in git history / prior report revision (2026-09-21 05:31 UTC). Headline:
Tony-set regex IoU collapsed (~0.28) under domain/language shift; LTT-6 English
regex→Jev remains strong. This upgrade targets LTT-6 English quality only.
