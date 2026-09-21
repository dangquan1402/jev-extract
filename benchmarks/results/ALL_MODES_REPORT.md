# ALL MODES REPORT — extraction strategy A/B

Generated: 2026-09-21 (Asia/Saigon)

## Protocol

1. **Stratified n=20** (gold length buckets 1 / 2–3 / 4+) across **all 9 modes**.
2. **Full n=50** on top contenders + remaining implemented modes (`token_sequential`, `cascade`, `topk_noul`, `ambiguous_joint`, `length_rerank`, `shape_gate`, `span_choice`, `anchor_expand`). `chunk_only` full-50 taken from prior baseline run (structural low EM).

Dataset: `benchmarks/data/extraction_50.jsonl`. Model: `jev-latest`. Pricing: $0.042/M input (output free).

## Full n=50 metrics (ranked by token_em)

| mode | n | token_em | token_f1 | contains_gold | exact_match | p50_ms | total_usd | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| span_choice | 50 | 78.0% | 0.907 | 100.0% | 78.0% | 284 | $0.02135 | 0 |
| token_sequential | 50 | 76.0% | 0.917 | 100.0% | 76.0% | 389 | $0.00421 | 0 |
| cascade | 50 | 76.0% | 0.909 | 100.0% | 76.0% | 512 | $0.00364 | 0 |
| ambiguous_joint | 50 | 76.0% | 0.904 | 100.0% | 76.0% | 361 | $0.00455 | 0 |
| topk_noul | 50 | 72.0% | 0.902 | 100.0% | 72.0% | 547 | $0.00558 | 0 |
| shape_gate | 50 | 70.0% | 0.903 | 100.0% | 70.0% | 540 | $0.00547 | 0 |
| length_rerank | 50 | 68.0% | 0.896 | 100.0% | 68.0% | 565 | $0.00547 | 0 |
| anchor_expand | 50 | 60.0% | 0.841 | 96.0% | 60.0% | 359 | $0.00430 | 0 |
| chunk_only | 50 | 10.0% | 0.393 | 100.0% | 10.0% | 164 | $0.00093 | 0 |

## Stratified n=20 metrics (ranked)

| mode | n | token_em | token_f1 | contains_gold | exact_match | p50_ms | total_usd | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cascade | 20 | 95.0% | 0.967 | 100.0% | 95.0% | 503 | $0.00143 | 0 |
| token_sequential | 20 | 95.0% | 0.967 | 100.0% | 95.0% | 365 | $0.00164 | 0 |
| ambiguous_joint | 20 | 90.0% | 0.942 | 100.0% | 90.0% | 347 | $0.00166 | 0 |
| topk_noul | 20 | 90.0% | 0.939 | 100.0% | 90.0% | 534 | $0.00218 | 0 |
| shape_gate | 20 | 75.0% | 0.926 | 100.0% | 75.0% | 515 | $0.00214 | 0 |
| length_rerank | 20 | 75.0% | 0.926 | 100.0% | 75.0% | 537 | $0.00214 | 0 |
| span_choice | 20 | 75.0% | 0.923 | 100.0% | 75.0% | 243 | $0.00856 | 0 |
| anchor_expand | 20 | 65.0% | 0.897 | 95.0% | 65.0% | 395 | $0.00171 | 0 |
| chunk_only | 20 | 5.0% | 0.379 | 100.0% | 5.0% | 179 | $0.00037 | 0 |

## Recommendation

### Winner on EM

- **Best token_em (n=50):** `span_choice` at **78.0%** (token_f1 0.907, p50 284 ms, $0.02135).

- Baseline `token_sequential`: 76.0% EM, p50 389 ms, $0.00421.
- Delta vs baseline: **+2.0%** EM.

### Cost / latency tradeoffs

- **`span_choice`** edges EM (+2pp vs sequential) and is **fastest** (p50 ~284 ms) but **~5× cost** (large Choice criteria over all windows). Use when latency matters more than $.
- **`token_sequential` / `ambiguous_joint` / `cascade`**: tied ~76% EM. Prefer `token_sequential` as default (simple, solid F1 0.917). `ambiguous_joint` is nearly free insurance when start words repeat; `cascade` saves a bit of $ but adds latency.
- **`topk_noul`**: did **not** beat sequential on this set (72% EM); Noul sometimes prefers an incomplete shorter span. Higher $ and latency for a loss.
- **`shape_gate` / `length_rerank`**: short-answer bias helped some over-extends but **hurt** multi-token / sentence golds → net EM drop (68–70%). Needs softer α or better shape labels before production.
- **`anchor_expand`**: weakest (60% EM); head+extent is brittle on phrases.
- **`chunk_only`**: ~10% EM by construction (span = whole sentence); keep for `contains_gold` / coarse routing only.

### Practical default

Ship **`token_sequential`** (or `ambiguous_joint` as drop-in). A/B **`span_choice`** only if the ~5× token cost is acceptable for +2pp EM and lower latency. Do not enable `length_rerank`/`shape_gate`/`topk_noul` without retuning.

## Failure taxonomy (n=50, non-token_em)

### span_choice — 11 misses

| failure_mode | count |
|---|---:|
| `end_under_extend` | 4 |
| `start_late` | 4 |
| `span_subset` | 3 |

Miss highlights:
- **ext_acme_countries** `end_under_extend`: gold='twelve countries' pred='twelve'
- **ext_northwind_rev** `start_late`: gold='$48 million' pred='48 million'
- **ext_invoice_amount** `start_late`: gold='$12,450' pred='12,450'
- **ext_sent_mars** `span_subset`: gold="NASA's Perseverance rover landed in Jezero Crater." pred='Jezero Crater'
- **ext_sent_vaccine** `end_under_extend`: gold='Cold-chain logistics remained a deployment challenge.' pred='Cold-chain logistics'
- **ext_sent_python** `end_under_extend`: gold='Guido van Rossum created it in the late 1980s.' pred='Guido van Rossum'
- _… 5 more_

### token_sequential — 12 misses

| failure_mode | count |
|---|---:|
| `end_under_extend` | 4 |
| `end_over_extend` | 4 |
| `start_early` | 2 |
| `span_subset` | 1 |
| `start_late` | 1 |

Miss highlights:
- **ext_sent_mars** `span_subset`: gold="NASA's Perseverance rover landed in Jezero Crater." pred='Jezero Crater'
- **ext_sent_vaccine** `end_under_extend`: gold='Cold-chain logistics remained a deployment challenge.' pred='Cold-chain logistics remained a deployment challenge'
- **ext_sent_python** `end_under_extend`: gold='Guido van Rossum created it in the late 1980s.' pred='Guido van Rossum'
- **ext_tesla_coil** `start_late`: gold='the 1890s' pred='1890s'
- **ext_photosynthesis** `end_over_extend`: gold='blue and red' pred='blue and red wavelengths'
- **ext_process_node** `end_over_extend`: gold='5-nanometer' pred='5-nanometer process node'
- _… 6 more_

### cascade — 12 misses

| failure_mode | count |
|---|---:|
| `end_under_extend` | 4 |
| `end_over_extend` | 4 |
| `span_subset` | 2 |
| `start_late` | 1 |
| `start_early` | 1 |

Miss highlights:
- **ext_acme_countries** `end_under_extend`: gold='twelve countries' pred='twelve'
- **ext_sent_mars** `span_subset`: gold="NASA's Perseverance rover landed in Jezero Crater." pred='Jezero Crater'
- **ext_sent_vaccine** `end_under_extend`: gold='Cold-chain logistics remained a deployment challenge.' pred='Cold-chain logistics remained a deployment challenge'
- **ext_sent_python** `end_under_extend`: gold='Guido van Rossum created it in the late 1980s.' pred='Guido van Rossum'
- **ext_tesla_coil** `start_late`: gold='the 1890s' pred='1890s'
- **ext_photosynthesis** `end_over_extend`: gold='blue and red' pred='blue and red wavelengths'
- _… 6 more_

### ambiguous_joint — 12 misses

| failure_mode | count |
|---|---:|
| `end_under_extend` | 4 |
| `end_over_extend` | 4 |
| `span_subset` | 3 |
| `start_late` | 1 |

Miss highlights:
- **ext_sent_mars** `span_subset`: gold="NASA's Perseverance rover landed in Jezero Crater." pred='Jezero Crater'
- **ext_sent_vaccine** `end_under_extend`: gold='Cold-chain logistics remained a deployment challenge.' pred='Cold-chain logistics remained a deployment challenge'
- **ext_sent_python** `end_under_extend`: gold='Guido van Rossum created it in the late 1980s.' pred='Guido van Rossum'
- **ext_paris_agree** `span_subset`: gold='The Paris Agreement' pred='Paris'
- **ext_tesla_coil** `start_late`: gold='the 1890s' pred='1890s'
- **ext_photosynthesis** `end_over_extend`: gold='blue and red' pred='blue and red wavelengths'
- _… 6 more_

### topk_noul — 14 misses

| failure_mode | count |
|---|---:|
| `end_over_extend` | 6 |
| `end_under_extend` | 4 |
| `start_early` | 2 |
| `span_subset` | 1 |
| `start_late` | 1 |

Miss highlights:
- **ext_invoice_amount** `end_over_extend`: gold='$12,450' pred='$12,450 excluding tax'
- **ext_payment_terms** `end_over_extend`: gold='net 30' pred='net 30 from the invoice date.'
- **ext_sent_mars** `span_subset`: gold="NASA's Perseverance rover landed in Jezero Crater." pred='Jezero Crater'
- **ext_sent_vaccine** `end_under_extend`: gold='Cold-chain logistics remained a deployment challenge.' pred='Cold-chain logistics remained a deployment challenge'
- **ext_sent_python** `end_under_extend`: gold='Guido van Rossum created it in the late 1980s.' pred='Guido van Rossum'
- **ext_tesla_coil** `start_late`: gold='the 1890s' pred='1890s'
- _… 8 more_

### shape_gate — 15 misses

| failure_mode | count |
|---|---:|
| `end_under_extend` | 11 |
| `start_early` | 2 |
| `span_subset` | 1 |
| `end_over_extend` | 1 |

Miss highlights:
- **ext_everest** `end_under_extend`: gold='8,849 meters' pred='8,849'
- **ext_acme_countries** `end_under_extend`: gold='twelve countries' pred='twelve'
- **ext_northwind_rev** `end_under_extend`: gold='$48 million' pred='$48'
- **ext_ship_weight** `end_under_extend`: gold='4.2 kilograms' pred='4.2'
- **ext_sent_mars** `span_subset`: gold="NASA's Perseverance rover landed in Jezero Crater." pred='Jezero Crater'
- **ext_sent_vaccine** `end_under_extend`: gold='Cold-chain logistics remained a deployment challenge.' pred='Cold-chain logistics remained a deployment challenge'
- _… 9 more_

### length_rerank — 16 misses

| failure_mode | count |
|---|---:|
| `end_under_extend` | 11 |
| `start_early` | 2 |
| `span_subset` | 1 |
| `start_late` | 1 |
| `end_over_extend` | 1 |

Miss highlights:
- **ext_everest** `end_under_extend`: gold='8,849 meters' pred='8,849'
- **ext_acme_countries** `end_under_extend`: gold='twelve countries' pred='twelve'
- **ext_northwind_rev** `end_under_extend`: gold='$48 million' pred='$48'
- **ext_ship_weight** `end_under_extend`: gold='4.2 kilograms' pred='4.2'
- **ext_sent_mars** `span_subset`: gold="NASA's Perseverance rover landed in Jezero Crater." pred='Jezero Crater'
- **ext_sent_vaccine** `end_under_extend`: gold='Cold-chain logistics remained a deployment challenge.' pred='Cold-chain logistics remained a deployment challenge'
- _… 10 more_

### anchor_expand — 20 misses

| failure_mode | count |
|---|---:|
| `end_under_extend` | 7 |
| `start_late` | 5 |
| `span_subset` | 4 |
| `both_endpoints_wrong` | 2 |
| `start_early` | 2 |

Miss highlights:
- **ext_everest** `start_late`: gold='8,849 meters' pred=',849 meters'
- **ext_acme_countries** `end_under_extend`: gold='twelve countries' pred='twelve'
- **ext_northwind_rev** `both_endpoints_wrong`: gold='$48 million' pred='reached $48'
- **ext_ship_weight** `start_late`: gold='4.2 kilograms' pred='.2 kilograms'
- **ext_invoice_amount** `start_late`: gold='$12,450' pred=',450'
- **ext_payment_terms** `start_early`: gold='net 30' pred='Payment terms are net 30'
- _… 14 more_

## Files changed

- `src/jev_extract/strategies.py` — all 6 new strategies
- `src/jev_extract/questions.py` — span/shape/extent/noul builders
- `src/jev_extract/client.py` — `extract_mode` routing + dedicated methods
- `src/jev_extract/__init__.py` — export `EXTRACT_MODES`
- `tests/test_strategies_mock.py` — mock unit tests per mode
- `benchmarks/run_mode_compare.py` — `--all-modes`, `--write-all-report`
- `README.md` — modes table

## Result artifacts

- Stratified-20: `extraction_mode_compare_20260921T025634Z.json`
- Full-50: `extraction_mode_compare_20260921T025758Z.json`
- Per-mode JSON under `benchmarks/results/extraction_mode_*_20260921T025758Z.json`

