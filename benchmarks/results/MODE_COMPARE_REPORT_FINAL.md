# Extraction mode comparison — miss analysis & recommendations

Generated: 20260921T023700Z (UTC)  
Samples: **50** (full `extraction_50.jsonl`)  
Also ran stratified **20** first (token EM 95% / cascade 90% / chunk 5%).  
Pricing: $0.042 / M input tokens (output free). Model: `jev-1.13.0`.

## Metrics (n=50)

| mode | n | token_em | token_f1 | contains_gold | exact_match | p50_ms | total_usd | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| token_sequential | 50 | 78.0% | 0.924 | 100.0% | 78.0% | 347 | $0.00421 | 0 |
| chunk_only | 50 | 10.0% | 0.393 | 100.0% | 10.0% | 164 | $0.00093 | 0 |
| cascade | 50 | 78.0% | 0.916 | 100.0% | 78.0% | 498 | $0.00364 | 0 |

### Stratified 20 (sanity)

| mode | n | token_em | token_f1 | contains_gold | p50_ms | total_usd |
|---|---:|---:|---:|---:|---:|---:|
| token_sequential | 20 | 95.0% | 0.967 | 100.0% | 335 | $0.00164 |
| chunk_only | 20 | 5.0% | 0.379 | 100.0% | 190 | $0.00037 |
| cascade | 20 | 90.0% | 0.950 | 100.0% | 539 | $0.00143 |

## Failure taxonomy (non-token_em, n=50)

### token_sequential — 11 misses

| failure_mode | count | interpretation |
|---|---:|---|
| `end_over_extend` | 4 | Correct start; end includes trailing modifiers (`wavelengths`, `process node`, `physical volumes`, `spectators`) |
| `end_under_extend` | 4 | Mostly sentence-gold items truncated to subject/NP (`The IPCC`, `Guido van Rossum`, `The Pacific Ocean`) + one trailing-period miss |
| `start_early` | 2 | Extra leading function word (`on November 1`, `around 1600`) |
| `span_subset` | 1 | Sentence-gold collapsed to entity (`Jezero Crater`) |

### cascade — 11 misses

| failure_mode | count | interpretation |
|---|---:|---|
| `end_over_extend` | 4 | Same short-entity over-extend pattern as token |
| `end_under_extend` | 3 | Same sentence-gold truncation pattern |
| `span_subset` | 2 | Sentence-gold → entity (`Jezero Crater`, `IPCC`) |
| `start_early` | 1 | `Service begins on November 1` (worse than token's `on November 1`) |
| `start_late` | 1 | Dropped determiner (`1890s` vs `the 1890s`) |

**Overlap:** 10/11 misses shared between token and cascade. Token-only: `ext_shakespeare`. Cascade-only: `ext_tesla_coil`.  
Cascade does **not** reduce end-over-extend on this set (chunk window still allows the same trailing nouns).

### chunk_only — 45 misses

| failure_mode | count | interpretation |
|---|---:|---|
| `span_superset` | 40 | **Correct sentence**, but span = whole sentence while gold is a short subspan |
| `end_over_extend` | 5 | Near-miss when gold nearly fills the sentence |

**Wrong chunk: 0 / 50.** Sentence Choice is perfect on this 3-sentence dataset. Low token_em is structural (not a selection failure). `contains_gold = 100%` confirms that.

## Ranked improvement methods (token + cascade misses)

Counts below sum token_sequential + cascade miss labels (≈22 labeled misses, 10 shared examples).

1. **Bias shorter end spans / length prior on end Choice** — drives `end_over_extend` (8)  
   - Strengthen "prefer shortest end" instruction; optional soft prior / re-rank top-k ends by `prob × length_penalty`.  
   - **Expected:** recover ~4–6 EM points on entity golds (`400,000`, `52,000`, `5-nanometer`, `blue and red`). Small risk of more `end_under_extend`.

2. **Question-type gate (entity vs sentence-span)** — drives `end_under_extend` + `span_subset` (≈9)  
   - Detect sentence-gold questions ("which sentence…", full-clause answers) vs short-entity questions; flip end prior (prefer longer / include trailing predicate).  
   - **Expected:** largest upside on `ext_sent_*` items where both modes truncate to NP.

3. **Second-pass re-rank of top-k ends** — drives `end_over_extend` (8)  
   - Keep top-3 end probabilities from Choice; pick shortest that still covers question keywords / has high joint conf.  
   - **Expected:** similar to (1) without changing Choice schema; easy to A/B.

4. **Leading function-word strip (start-early fix)** — drives `start_early` (3)  
   - If pred starts with `on`/`around`/`in`/`the` and gold does not, shift start +1 (or drop leading stopwords in a post-check).  
   - **Expected:** fix `November 1` / `1600` style misses cheaply.

5. **Include optional leading determiner on start** — drives `start_late` (1)  
   - Mirror of (4): if gold often includes `the`/`a`, prefer start that includes it when adjacent. Low count here.

6. **Cascade is not a free win on this set** — wrong_chunk = 0  
   - Chunk selection is already perfect; refine errors match token. Cascade costs more latency (p50 498 vs 347) for equal EM. Prefer cascade when paragraphs get long (token windows / duplicate-word density), not on 3-sentence paras.

7. **Clause vs sentence chunking A/B** — not needed yet  
   - Clause fallback rarely fires (paras are short). Revisit on longer documents.

8. **ID-pointer chunk Choice (`criteria={id: None}` + tagged state)** — not needed yet  
   - With 3 short sentences, criteria-text Choice already gets 100% chunk accuracy. Try ID-pointer when chunk texts are long/truncated.

9. **Exists / Noul abstention** — drives `empty_pred` (0)  
   - No empty predictions on this dense extractive set. Defer.

10. **Joint start–end on duplicated start words** — drives `duplicate_word` (0 on this run)  
    - Context-labeled criteria already helped; keep as fallback when `Counter(tokens)[start_word] > 1` and start confidence is low (already partially in `extract_mode=auto`).

## Result files

- `benchmarks/results/extraction_mode_token_sequential_20260921T023535Z.json`
- `benchmarks/results/extraction_mode_chunk_only_20260921T023535Z.json`
- `benchmarks/results/extraction_mode_cascade_20260921T023535Z.json`
- `benchmarks/results/extraction_mode_compare_20260921T023535Z.json`
- Stratified-20: `benchmarks/results/extraction_mode_*_20260921T023508Z.json`

## Notes

- `token_sequential`: start/end Choice over all paragraph tokens (`extract_mode=sequential`).
- `chunk_only`: sentence/clause Choice; predicted span = entire winning chunk.
- `cascade`: chunk Choice → start/end only inside winner (`Extractor.extract_cascade`).
- Default chunker: sentence split with clause fallback for long sentences (`propose_chunks`).
