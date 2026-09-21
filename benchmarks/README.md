# Benchmarks — Jev vs future LLMs

Comparison harness for **TypeSafe Jev** (implemented) vs **Gemini / Haiku** (stubs until keys + adapters land).

## Tasks

| Task | Dataset | Backend path | Quality metrics |
|------|---------|--------------|-----------------|
| `classification` | `data/classification_50.jsonl` | Jev `Choice` over shared label criteria | accuracy |
| `extraction` | `data/extraction_50.jsonl` | `jev_extract.Extractor` (token start/end Choice) | token_em, token_f1, token_iou (+ optional string EM) |

Both record: `latency_ms`, `input_tokens`, `output_tokens`, `estimated_usd`.

## Pricing

TypeSafe Jev (public): **$0.042 / million input tokens**, **output free**.  
Constant: `metrics.TYPESAFE_INPUT_USD_PER_MILLION`. Usage is read from `SystemOneResponse.usage` when present.

## Run (TypeSafe only today)

From repo root (venv with `pip install -e ".[bench]"`):

```bash
export TYPESAFE_API_KEY=tsk_...   # or set JEV_EXTRACT_SECRETS_JSON

python benchmarks/run_bench.py classification --backend typesafe --limit 50
python benchmarks/run_bench.py extraction --backend typesafe --limit 50
python benchmarks/run_bench.py all --backend typesafe

# Validate JSONL + spans; no API calls
python benchmarks/run_bench.py all --backend typesafe --dry-run
```

Selecting `--backend gemini` or `--backend haiku` prints **not configured yet** and skips.

### CLI behavior

- Warmup **1** call per task (default) — recorded but **excluded** from aggregates.
- Per-example records + summary JSON → `benchmarks/results/`.
- Markdown tables printed to stdout; `results/SUMMARY.md` refreshed.
- Model pin: **`jev-latest`** (resolved model string recorded from the response).

## Fairness / known bias

**Jev extraction = token-native start/end Choice** with local-context criterion descriptions (`«word»` + neighbors) so duplicate words disambiguate. Primary metric is inclusive token-index exact match (`token_em`). String EM / contains_gold are optional diagnostics.

**Future free-span LLMs** can copy arbitrary substrings and are advantaged on short golds. Do not treat raw `exact_match` as a fair head-to-head without that caveat.

Classification is a cleaner closed-set comparison (same Choice criteria for all backends once LLM adapters map labels the same way).

## Dataset schemas

### Classification

```json
{
  "id": "cls_001_billing",
  "text": "…",
  "label": "billing",
  "labels": {
    "billing": "…",
    "technical": "…",
    "account": "…",
    "shipping": "…",
    "other": "…"
  }
}
```

Shared schema for all 50 items. Gold `label` ∈ criteria keys.

### Extraction

```json
{
  "id": "ext_curie_physics",
  "paragraph": "…",
  "question": "…",
  "tokens": ["Marie", "Curie", "...", "1903", "."],
  "start": {"pos": 15, "word": "1903"},
  "end": {"pos": 15, "word": "1903"},
  "gold": "1903",
  "char_start": 82,
  "char_end": 86
}
```

Offsets verified at generation time (`generate_datasets.py`) and again on `--dry-run`.

## Metrics module

`metrics.py`: `normalize`, `exact_match`, `span_f1`, `contains_gold`, classification accuracy, latency percentiles, `estimated_usd`.

## Regenerating data

```bash
python benchmarks/generate_datasets.py
```

Synthetic English templates + variation (not 50 near-duplicates); balanced classification labels (10 each).
