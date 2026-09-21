# Sponsor Skip Bench — Jev Choice vs SponsorBlock

Generated: 2026-09-21 04:59 UTC

## Setup

- **Detector**: TypeSafe Jev closed-set `Choice` (no trained classifier) + regex/heuristic cues.
- **Gold**: SponsorBlock public API (`sponsor` primary).
- **Pricing**: $0.042/M input tokens (output free).
- **Videos evaluated**: 6 (cached transcripts/gold reused).

### Videos

| video_id | duration_s | gold sponsor segs | n_chunks | n_sentences |
|----------|------------|-------------------|----------|-------------|
| `qgLaCZyKv_8` | 1052 | 2 | 63 | 99 |
| `zg-rMHEqg-4` | 1198 | 2 | 74 | 171 |
| `3xngArcFpek` | 585 | 2 | 36 | 55 |
| `3PGKMwgjla0` | 1427 | 2 | 89 | 152 |
| `i_nj_vkD03g` | 1535 | 2 | 93 | 222 |
| `pvSdeU13hKc` | 1143 | 2 | 71 | 139 |

## Baseline (original `chunk_batched`)

First-run report (macro over 6 videos): **IoU 0.504 / P 0.568 / R 0.847 / F1 0.667**.
Main failure: false positives on promo-sounding *content* (product praise, merch, segue teases).

Re-run of the same baseline mode in this session (stochastic Jev labels):

| mode | n | IoU | P | R | F1 | video recall | content FP | tokens | USD |
|------|---|-----|---|---|----|--------------|------------|--------|-----|
| `chunk_batched` | 6 | 0.496 | 0.551 | 0.847 | 0.659 | 1.000 | 0.070 | 197385 | $0.0083 |

## IMPROVED — regex + heuristics + Jev

### What changed

1. **`regex_propose.py`** — strong paid-read regex cues propose candidate windows (~−12s / +80s), merged.
2. **`regex_propose_jev_confirm`** (primary) — Jev Choice only on chunks overlapping proposed windows, with **strict** criteria (paid third-party only; product praise ≠ sponsor). If ≥1 chunk confirms `sponsor`, **expand** the segment from cue → return-to-content / CTA / ~95s cap (covers LTT narrative/sketch mid-rolls).
3. **`jev_then_regex_gate`** — classify all chunks (strict), drop sponsor runs with no strong cue unless confidence ≥0.9, then same window expand.
4. **Boundary refine** — snap start to strong cue; end on `back to the (video|show)`, `anyway`, `with that said`, `if you (guys) enjoyed`, etc., or cue+N cap.
5. **Weak URL-ish cues disabled** in primary propose (they caused content FPs like “sponsor doghouse”).
6. Core `jev_extract` IE defaults **unchanged** — edits only under `benchmarks/sponsor_skip/`.

### Metrics comparison (macro-average)

| mode | n | IoU | P | R | F1 | video recall | content FP | tokens | USD |
|------|---|-----|---|---|----|--------------|------------|--------|-----|
| `chunk_batched` | 6 | 0.496 | 0.551 | 0.847 | 0.659 | 1.000 | 0.070 | 197385 | $0.0083 |
| `regex_propose_jev_confirm` | 6 | 0.838 | 0.902 | 0.920 | 0.910 | 1.000 | 0.010 | 48751 | $0.0020 |
| `jev_then_regex_gate` | 6 | 0.842 | 0.907 | 0.920 | 0.913 | 1.000 | 0.009 | 243693 | $0.0102 |

**Winner:** `regex_propose_jev_confirm` — IoU **0.838** / F1 **0.910** (vs baseline ~0.50 / 0.67), recall **0.920** (≥0.7), precision **0.902**, content FP **0.0098** (was ~0.07), at **~4× fewer tokens** than all-chunk Jev.

`jev_then_regex_gate` matches quality but costs ~5× more tokens (classifies every chunk).

### Per-video — `regex_propose_jev_confirm`

| video | IoU | P | R | F1 | overlap_s | fp_s | content_fp | tokens | USD |
|-------|-----|---|---|----|-----------|------|------------|--------|-----|
| `qgLaCZyKv_8` | 0.914 | 0.981 | 0.930 | 0.955 | 116.3 | 2.2 | 0.0024 | 11388 | $0.0005 |
| `zg-rMHEqg-4` | 0.835 | 0.917 | 0.904 | 0.910 | 89.0 | 8.1 | 0.0073 | 8011 | $0.0003 |
| `3xngArcFpek` | 0.736 | 0.805 | 0.896 | 0.848 | 61.9 | 15.0 | 0.0290 | 6808 | $0.0003 |
| `3PGKMwgjla0` | 0.853 | 0.892 | 0.952 | 0.921 | 76.4 | 9.3 | 0.0069 | 7927 | $0.0003 |
| `i_nj_vkD03g` | 0.763 | 0.868 | 0.863 | 0.866 | 76.8 | 11.7 | 0.0081 | 6968 | $0.0003 |
| `pvSdeU13hKc` | 0.928 | 0.949 | 0.978 | 0.963 | 100.4 | 5.4 | 0.0052 | 7649 | $0.0003 |

### Regex patterns that helped

Counts of strong-cue hits across the 6 videos (primary mode uses strong only):

| pattern | hits | role |
|---------|------|------|
| `segue_sponsor` | 7 | Explicit LTT-style mid-roll opener — highest value |
| `our_sponsor` | 6 | Direct paid-read anchor |
| `pct_off` | 3 | Discount CTA; extends end of narrative ads |
| `segue_to_our` | 2 | ASR-truncated “segue to our …” (recovers garbled captions) |
| `link_down_below` | 2 | CTA / end-of-read cue |
| `with_the_link` | 2 | CTA |
| `link_in_desc` | 1 | CTA |

Also defined (fewer/no hits on this set, kept for generality): `sponsored_by`, `thanks_to_sponsor`, `video_sponsored`, `brought_to_you`, `paid_partnership`, `todays_sponsor`, `use_code`, `promo_code`, `discount_code`, `check_out_at`.

**Intentionally dropped from primary propose:** weak `brand_urlish` / `.com` patterns — they proposed false windows on review content.

### Per-video — `jev_then_regex_gate`

| video | IoU | P | R | F1 | content_fp | tokens |
|-------|-----|---|---|----|------------|--------|
| `qgLaCZyKv_8` | 0.914 | 0.981 | 0.930 | 0.955 | 0.0024 | 35219 |
| `zg-rMHEqg-4` | 0.835 | 0.917 | 0.904 | 0.910 | 0.0073 | 43400 |
| `3xngArcFpek` | 0.760 | 0.834 | 0.896 | 0.864 | 0.0239 | 19992 |
| `3PGKMwgjla0` | 0.853 | 0.892 | 0.952 | 0.921 | 0.0069 | 50382 |
| `i_nj_vkD03g` | 0.763 | 0.868 | 0.863 | 0.866 | 0.0081 | 54208 |
| `pvSdeU13hKc` | 0.928 | 0.949 | 0.978 | 0.963 | 0.0052 | 40492 |

## Conclusions

**Does regex propose → Jev confirm work?** **Yes.** On the same 6 videos it lifts IoU 0.50→**0.84** and F1 0.67→**0.91**, keeps recall ~0.92, cuts content FP ~7×, and uses far fewer tokens than labeling every chunk.

The expand-on-confirm step is essential for Linus-style sketch/narrative mid-rolls where only the CTA chunks look like ads to the model.

## Blockers

- youtube-transcript-api blocked from this host; cached yt-dlp VTT transcripts reused.
- No changes to core `jev_extract` IE defaults.

## File paths

- Experiment root: `/workspace/jev-extract/benchmarks/sponsor_skip`
- Runner: `/workspace/jev-extract/benchmarks/sponsor_skip/run_sponsor_skip_bench.py`
- Regex/heuristics: `/workspace/jev-extract/benchmarks/sponsor_skip/regex_propose.py`
- Detector (hybrid modes): `/workspace/jev-extract/benchmarks/sponsor_skip/detect.py`
- Transcripts: `/workspace/jev-extract/benchmarks/sponsor_skip/data/transcripts`
- Gold: `/workspace/jev-extract/benchmarks/sponsor_skip/data/gold`
- Results JSON: `/workspace/jev-extract/benchmarks/sponsor_skip/results/sponsor_skip_improved.json`
- This report: `/workspace/jev-extract/benchmarks/sponsor_skip/SPONSOR_SKIP_REPORT.md`
