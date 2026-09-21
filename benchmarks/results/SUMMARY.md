# Benchmark summary

Generated: 20260920T125658Z (UTC)
Backend: `typesafe`
Dry-run: True

Pricing (TypeSafe Jev): **$0.042 / million input tokens**, output free.

## Classification

| backend | n | accuracy | p50_ms | p95_ms | mean_ms | sum_input_tokens | sum_output_tokens | total_usd | errors |
|---------|---|----------:|-------:|-------:|--------:|-----------------:|------------------:|----------:|-------:|
| typesafe | — | — | — | — | — | — | — | — | skipped: dry-run (no API calls) |

## Extraction

| backend | n | token_em | token_f1 | token_iou | exact_match | contains_gold | p50_ms | p95_ms | mean_ms | sum_input_tokens | sum_output_tokens | total_usd | errors |
|---------|---|---------:|---------:|----------:|------------:|--------------:|-------:|-------:|--------:|-----------------:|------------------:|----------:|-------:|
| typesafe | — | — | — | — | — | — | — | — | — | — | — | — | skipped: dry-run (no API calls) |

## Fairness note

Jev extraction is **token-native**: sequential start/end Choice over tokenizer-v1 tokens (criteria keys = positions, labels = `pos:word`). Primary metric is **token-span exact match** (`token_em`); secondary are token-index F1/IoU. String `exact_match` / `contains_gold` remain as optional diagnostics. Sentence-candidate Choice is still available via `Extractor.extract_sentence` for coarse comparisons.

