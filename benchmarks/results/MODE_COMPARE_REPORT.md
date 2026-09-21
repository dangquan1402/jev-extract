# Extraction mode comparison — miss analysis & recommendations

Generated: 20260921T030819Z (UTC)
Samples: **50** (prefix / full load)
Pricing: $0.042/M input tokens (output free).

## Metrics

| mode | n | token_em | token_f1 | contains_gold | exact_match | p50_ms | total_usd | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| token_sequential | 50 | 74.0% | 0.911 | 100.0% | 74.0% | 353 | $0.00421 | 0 |
| span_choice | 50 | 78.0% | 0.907 | 100.0% | 78.0% | 260 | $0.02135 | 0 |
| span_choice_smart | 50 | 76.0% | 0.897 | 100.0% | 76.0% | 236 | $0.01249 | 0 |
| span_choice_in_chunk | 50 | 78.0% | 0.907 | 100.0% | 78.0% | 353 | $0.00493 | 0 |

## Failure taxonomy (non-token_em)

### token_sequential — 13 misses

| failure_mode | count |
|---|---:|
| `end_under_extend` | 5 |
| `end_over_extend` | 4 |
| `start_early` | 2 |
| `span_subset` | 1 |
| `start_late` | 1 |

<details><summary>Miss details</summary>

- **ext_acme_countries** `end_under_extend`: gold='twelve countries' pred='twelve' spans gold=[26, 27] pred=[26, 26]
- **ext_sent_mars** `span_subset`: gold="NASA's Perseverance rover landed in Jezero Crater." pred='Jezero Crater' spans gold=[0, 9] pred=[7, 8]
- **ext_sent_vaccine** `end_under_extend`: gold='Cold-chain logistics remained a deployment challenge.' pred='Cold-chain logistics remained a deployment challenge' spans gold=[16, 24] pred=[16, 23]
- **ext_sent_python** `end_under_extend`: gold='Guido van Rossum created it in the late 1980s.' pred='Guido van Rossum' spans gold=[8, 17] pred=[8, 10]
- **ext_tesla_coil** `start_late`: gold='the 1890s' pred='1890s' spans gold=[15, 16] pred=[16, 16]
- **ext_photosynthesis** `end_over_extend`: gold='blue and red' pred='blue and red wavelengths' spans gold=[11, 13] pred=[11, 14]
- **ext_process_node** `end_over_extend`: gold='5-nanometer' pred='5-nanometer process node' spans gold=[4, 6] pred=[4, 8]
- **ext_library** `end_over_extend`: gold='400,000' pred='400,000 physical volumes' spans gold=[15, 17] pred=[15, 19]
- **ext_stadium** `end_over_extend`: gold='52,000' pred='52,000 spectators' spans gold=[3, 5] pred=[3, 6]
- **ext_contract** `start_early`: gold='November 1' pred='on November 1' spans gold=[24, 25] pred=[23, 25]
- **ext_sent_climate** `end_under_extend`: gold='The IPCC synthesizes climate science for policymakers.' pred='The IPCC' spans gold=[11, 18] pred=[11, 12]
- **ext_shakespeare** `start_early`: gold='1600' pred='around 1600' spans gold=[15, 15] pred=[14, 15]
- **ext_sent_ocean** `end_under_extend`: gold="The Pacific Ocean is the largest of Earth's oceanic divisions." pred='The Pacific Ocean' spans gold=[0, 12] pred=[0, 2]

</details>

### span_choice — 11 misses

| failure_mode | count |
|---|---:|
| `end_under_extend` | 4 |
| `start_late` | 4 |
| `span_subset` | 3 |

<details><summary>Miss details</summary>

- **ext_acme_countries** `end_under_extend`: gold='twelve countries' pred='twelve' spans gold=[26, 27] pred=[26, 26]
- **ext_northwind_rev** `start_late`: gold='$48 million' pred='48 million' spans gold=[10, 12] pred=[11, 12]
- **ext_invoice_amount** `start_late`: gold='$12,450' pred='12,450' spans gold=[14, 17] pred=[15, 17]
- **ext_sent_mars** `span_subset`: gold="NASA's Perseverance rover landed in Jezero Crater." pred='Jezero Crater' spans gold=[0, 9] pred=[7, 8]
- **ext_sent_vaccine** `end_under_extend`: gold='Cold-chain logistics remained a deployment challenge.' pred='Cold-chain logistics' spans gold=[16, 24] pred=[16, 19]
- **ext_sent_python** `end_under_extend`: gold='Guido van Rossum created it in the late 1980s.' pred='Guido van Rossum' spans gold=[8, 17] pred=[8, 10]
- **ext_paris_agree** `start_late`: gold='The Paris Agreement' pred='Paris Agreement' spans gold=[17, 19] pred=[18, 19]
- **ext_tesla_coil** `start_late`: gold='the 1890s' pred='1890s' spans gold=[15, 16] pred=[16, 16]
- **ext_contract_states** `end_under_extend`: gold='five states' pred='five' spans gold=[18, 19] pred=[18, 18]
- **ext_sent_climate** `span_subset`: gold='The IPCC synthesizes climate science for policymakers.' pred='IPCC' spans gold=[11, 18] pred=[12, 12]
- **ext_sent_ocean** `span_subset`: gold="The Pacific Ocean is the largest of Earth's oceanic divisions." pred='Pacific Ocean' spans gold=[0, 12] pred=[1, 2]

</details>

### span_choice_smart — 12 misses

| failure_mode | count |
|---|---:|
| `end_under_extend` | 5 |
| `start_late` | 4 |
| `span_subset` | 3 |

<details><summary>Miss details</summary>

- **ext_acme_countries** `end_under_extend`: gold='twelve countries' pred='twelve' spans gold=[26, 27] pred=[26, 26]
- **ext_northwind_rev** `start_late`: gold='$48 million' pred='48 million' spans gold=[10, 12] pred=[11, 12]
- **ext_invoice_amount** `start_late`: gold='$12,450' pred='12,450' spans gold=[14, 17] pred=[15, 17]
- **ext_sent_mars** `span_subset`: gold="NASA's Perseverance rover landed in Jezero Crater." pred='Jezero Crater' spans gold=[0, 9] pred=[7, 8]
- **ext_sent_vaccine** `end_under_extend`: gold='Cold-chain logistics remained a deployment challenge.' pred='Cold-chain logistics' spans gold=[16, 24] pred=[16, 19]
- **ext_sent_python** `end_under_extend`: gold='Guido van Rossum created it in the late 1980s.' pred='Guido van Rossum' spans gold=[8, 17] pred=[8, 10]
- **ext_paris_agree** `start_late`: gold='The Paris Agreement' pred='Paris Agreement' spans gold=[17, 19] pred=[18, 19]
- **ext_tesla_coil** `start_late`: gold='the 1890s' pred='1890s' spans gold=[15, 16] pred=[16, 16]
- **ext_stadium** `end_under_extend`: gold='52,000' pred='52' spans gold=[3, 5] pred=[3, 3]
- **ext_contract_states** `end_under_extend`: gold='five states' pred='five' spans gold=[18, 19] pred=[18, 18]
- **ext_sent_climate** `span_subset`: gold='The IPCC synthesizes climate science for policymakers.' pred='IPCC' spans gold=[11, 18] pred=[12, 12]
- **ext_sent_ocean** `span_subset`: gold="The Pacific Ocean is the largest of Earth's oceanic divisions." pred='Pacific Ocean' spans gold=[0, 12] pred=[1, 2]

</details>

### span_choice_in_chunk — 11 misses

| failure_mode | count |
|---|---:|
| `end_under_extend` | 4 |
| `start_late` | 4 |
| `span_subset` | 3 |

<details><summary>Miss details</summary>

- **ext_acme_countries** `end_under_extend`: gold='twelve countries' pred='twelve' spans gold=[26, 27] pred=[26, 26]
- **ext_northwind_rev** `start_late`: gold='$48 million' pred='48 million' spans gold=[10, 12] pred=[11, 12]
- **ext_invoice_amount** `start_late`: gold='$12,450' pred='12,450' spans gold=[14, 17] pred=[15, 17]
- **ext_sent_mars** `span_subset`: gold="NASA's Perseverance rover landed in Jezero Crater." pred='Jezero Crater' spans gold=[0, 9] pred=[7, 8]
- **ext_sent_vaccine** `end_under_extend`: gold='Cold-chain logistics remained a deployment challenge.' pred='Cold-chain logistics' spans gold=[16, 24] pred=[16, 19]
- **ext_sent_python** `end_under_extend`: gold='Guido van Rossum created it in the late 1980s.' pred='Guido van Rossum' spans gold=[8, 17] pred=[8, 10]
- **ext_paris_agree** `start_late`: gold='The Paris Agreement' pred='Paris Agreement' spans gold=[17, 19] pred=[18, 19]
- **ext_tesla_coil** `start_late`: gold='the 1890s' pred='1890s' spans gold=[15, 16] pred=[16, 16]
- **ext_contract_states** `end_under_extend`: gold='five states' pred='five' spans gold=[18, 19] pred=[18, 18]
- **ext_sent_climate** `span_subset`: gold='The IPCC synthesizes climate science for policymakers.' pred='IPCC' spans gold=[11, 18] pred=[12, 12]
- **ext_sent_ocean** `span_subset`: gold="The Pacific Ocean is the largest of Earth's oceanic divisions." pred='Pacific Ocean' spans gold=[0, 12] pred=[1, 2]

</details>

## Ranked improvement methods (tied to observed misses)

Ordered by expected impact given miss counts on **token_sequential + cascade** (chunk_only EM is structurally low when gold is a short subspan). **Not implemented** unless noted.

1. **Strengthen end-under / start-late instructions** _(drives `end_under_extend`, observed≈23)_
   - When gold is a full sentence but pred truncates (sentence-gold items), bias toward including trailing predicate tokens — opposite prior of short-answer items. A question-type gate (entity vs sentence) would help.

2. **Bias shorter end spans / length prior on end Choice** _(drives `end_over_extend`, observed≈8)_
   - Add instruction + soft prior favoring minimal ends (or re-rank top-k ends by length×prob). Expected: cut end_over_extend; small risk of end_under_extend on multi-token golds.

3. **Second-pass re-rank of top-k ends** _(drives `end_over_extend`, observed≈8)_
   - Keep top-3 end probabilities; pick shortest that still answers the question / has high joint conf. Expected: improve EM without schema changes.

4. **Leading-determiner / start-early fix** _(drives `start_early`, observed≈4)_
   - Post-check: if pred starts with a determiner/preposition not required by the question, shift start +1. Low-cost heuristic for contract/on-November style misses.

5. **Joint start–end when start word is duplicated** _(drives `duplicate_word`, observed≈0)_
   - When the chosen start surface form appears >1×, use joint pair Choice (or top-3 starts × ends). Expected: fewer wrong latch points on repeated words.

6. **Cascade chunker tuning (clause vs sentence; ID-pointer)** _(drives `wrong_chunk_or_span`, observed≈0)_
   - If cascade wrong_chunk dominates: try tagged state + criteria={id: None}, or tighter clause splits. On this 3-sentence set, chunk selection is usually easy.

7. **Exists / Noul abstention gate** _(drives `empty_pred`, observed≈0)_
   - Pre-ask Noul 'does paragraph contain answer?'; skip span Choice on low noul. Helps empty/short pathologies more than EM on this dense extractive set.

## Result files

- `/workspace/jev-extract/benchmarks/results/extraction_mode_token_sequential_20260921T030819Z.json`
- `/workspace/jev-extract/benchmarks/results/extraction_mode_span_choice_20260921T030819Z.json`
- `/workspace/jev-extract/benchmarks/results/extraction_mode_span_choice_smart_20260921T030819Z.json`
- `/workspace/jev-extract/benchmarks/results/extraction_mode_span_choice_in_chunk_20260921T030819Z.json`

## Notes

- `token_sequential`: start/end Choice (`extract_mode=sequential`) — baseline.
- `chunk_only`: sentence/clause Choice; span = whole chunk.
- `cascade`: chunk Choice → start/end inside winner.
- `span_choice`: classic full 1..K windows (expensive input tokens).
- `span_choice_smart` / `span_choice_cheap`: dense ≤6 + sentences + truncated criteria.
- `span_choice_in_chunk`: chunk Choice → dense span Choice inside winner.
- `anchor_expand`: head Choice → left/right extent Choices.
- `shape_gate`: shape Choice → shape-specific end instructions.
- `topk_noul`: top-k ends + batched Noul verify.
- `ambiguous_joint`: joint (start,end) when start ambiguous; else sequential.
- `length_rerank`: argmax(prob × length_penalty); shape-aware longer for sentences.

