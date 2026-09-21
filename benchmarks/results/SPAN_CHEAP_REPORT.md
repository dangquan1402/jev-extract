# Cheaper span_choice — propose shortlist / in-chunk

Generated: 2026-09-21 03:08 UTC+7 (bench stamp `20260921T030819Z`)  
Samples: **50** (full `extraction_50.jsonl`)  
Model: `jev-1.13.0` · Pricing: $0.042 / M input tokens (output free)

## Goal

Cut `span_choice` input tokens / USD while keeping EM within ~2pp of the classic
78% EM path (every contiguous window as a Choice criterion ≈ $0.021/50).

## What changed

| File | Change |
|---|---|
| `src/jev_extract/questions.py` | `propose_span_windows`, `enumerate_span_windows`, `sentence_token_spans`; `build_span_as_choice(propose=..., windows=...)` |
| `src/jev_extract/strategies.py` | `run_span_choice(propose="all\|smart\|in_chunk")`; modes `span_choice_smart`, `span_choice_in_chunk` |
| `src/jev_extract/client.py` | `extract_span_choice(propose=...)`; `extract_mode` routes smart / in_chunk |
| `benchmarks/run_mode_compare.py` | Bench modes for smart / in_chunk; `span_choice` forces classic `propose=all` for A/B |
| `benchmarks/load_card_secrets.py` | Load `TYPESAFE_API_KEY` from `box-secrets.json` |
| `tests/test_span_propose.py` | Proposer counting, caps, sentence inclusion, truncation |

### Propose modes

- **`all`** — classic full length-1..K windows (K adaptive ≤13, ≤255 criteria). Expensive.
- **`smart`** (default for `extract_span_choice` / `extract_mode=span_choice`) —
  - Cap `K=6` (gold hist: most answers ≤5–8 tokens)
  - Always keep **all** windows of length 1..6 + **full sentence** token spans
  - Keyword-overlap shortlist only for longer windows; dedupe; truncate criteria to 64 chars; context radius 1
- **`in_chunk`** — sentence/clause Choice first → dense span Choice **only inside** the winning chunk (chunk-first preferred over regex)

Offline gold-coverage of smart/in_chunk proposers on extraction_50: **50/50** (oracle chunk for in_chunk).

## Live A/B (n=50)

| mode | EM | F1 | contains_gold | p50_ms | total_usd | mean input_tokens | vs classic span USD |
|---|---:|---:|---:|---:|---:|---:|---|
| token_sequential | 74.0% | 0.911 | 100% | 353 | $0.00421 | 2,006 | — |
| span_choice (`all`) | **78.0%** | 0.907 | 100% | 260 | $0.02135 | **10,166** | baseline |
| span_choice_smart | **76.0%** | 0.897 | 100% | **236** | $0.01249 | **5,946** | **−41%** |
| span_choice_in_chunk | **78.0%** | 0.907 | 100% | 353 | **$0.00493** | **2,346** | **−77%** |

### Mean windows proposed (offline)

| propose | mean #windows |
|---|---:|
| all | 243.8 |
| smart | 158.8 |
| in_chunk (oracle chunk) | 42.3 |

## Success check

| Criterion | Result |
|---|---|
| Input tokens / USD clearly down vs classic span_choice | **Yes** — smart −41% USD / −41% mean input; in_chunk −77% USD / −77% mean input |
| EM within ~2pp of 78% or better | **Yes** — smart 76% (−2pp); **in_chunk 78% (tied)** |

**Recommendation:** ship **`propose="in_chunk"`** (`extract_mode=span_choice_in_chunk` or `extract_span_choice(propose="in_chunk")`) as the cheap high-EM span path. Keep `propose="all"` only for latency-sensitive A/Bs where ~5× cost is acceptable. Default `extract_span_choice` / `span_choice` mode now uses **`smart`**.

## Tests

```
pytest tests/  →  67 passed
```

New proposer tests: `tests/test_span_propose.py` (11 cases: enumerate counts, caps ≤255, length-1 + dense completeness, sentence inclusion, in-chunk range, truncation, dedupe, empty).

## Result artifacts

- `benchmarks/results/extraction_mode_token_sequential_20260921T030819Z.json`
- `benchmarks/results/extraction_mode_span_choice_20260921T030819Z.json`
- `benchmarks/results/extraction_mode_span_choice_smart_20260921T030819Z.json`
- `benchmarks/results/extraction_mode_span_choice_in_chunk_20260921T030819Z.json`
- `benchmarks/results/extraction_mode_compare_20260921T030819Z.json`
- `benchmarks/results/MODE_COMPARE_REPORT_20260921T030819Z.md`

## Blockers

None. API key loaded via `benchmarks/load_card_secrets.py` → `/home/box/agent-data/box-secrets.json`.
