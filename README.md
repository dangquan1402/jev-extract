# jev-extract — Jev comparison benchmark + extractive IE library

**Comparison framework** for [TypeSafe AI](https://typesafe.ai) **Jev** (System One) against future LLM backends (Gemini / Haiku stubs — not required today).

This repo ships:

1. **`jev_extract`** — paragraph **information extraction** library: convert extractive QA into sequential Jev **Choice** over **token** start/end positions.
2. **`benchmarks/`** — side-by-side **classification** and **extraction** benches measuring **accuracy, latency, tokens, and estimated USD**.

> TypeSafe-only today. Gemini/Haiku backends are registered stubs (`not configured yet`) so the harness is ready without those API keys.

Blog / product context: [Introducing System One models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

## What we measure

| Metric | Classification | Extraction |
|--------|----------------|------------|
| Quality | `accuracy` | `token_em`, `token_f1`, `token_iou` (primary); optional `exact_match`, `contains_gold` |
| Speed | `latency_ms` (p50 / p95 / mean) | same |
| Cost | `input_tokens`, `output_tokens`, `estimated_usd` | same |

**TypeSafe Jev public pricing** (documented + used in cost estimates): **$0.042 / million input tokens**, **output free**.

## Protocols

### Classification (50 samples)

Support-ticket / short-paragraph multi-class labeling (`billing`, `technical`, `account`, `shipping`, `other`) via Jev **Choice** with a shared criteria schema.

```bash
python benchmarks/run_bench.py classification --backend typesafe --limit 50
```

### Information extraction (50 samples)

Extractive QA → `jev_extract.Extractor` (**token-native** sequential start/end Choice). Gold spans are inclusive token positions `{pos, word}` with derived char offsets.

```bash
python benchmarks/run_bench.py extraction --backend typesafe --limit 50
```

### Both

```bash
python benchmarks/run_bench.py all --backend typesafe
```

Dry-run (validate data only, no API spend):

```bash
python benchmarks/run_bench.py all --backend typesafe --dry-run
```

## Fairness

**Jev extraction is token-native.** The extractor tokenizes the paragraph (whitespace + punctuation split), then runs two Choice calls: **start** token, then **end** token (`pos >= start`). Criteria keys are positions (`"4"`); descriptions are **local context windows** with the candidate marked `«word»` (default radius 4) so duplicate words stay distinguishable. End instructions mention `start was pos:word`. Primary bench metric is token-span exact match.

Sentence-candidate Choice remains available via `Extractor.extract_sentence` for coarse mode. See [benchmarks/README.md](benchmarks/README.md).

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[bench]"   # or pip install -e ".[dev]" for pytest/ruff
```

Requires Python **≥ 3.10**.

## Environment

```bash
export TYPESAFE_API_KEY=tsk_...
```

Optional: `TYPESAFE_DEFAULT_MODEL` (bench pins **`jev-latest`**).  
Optional: `JEV_EXTRACT_SECRETS_JSON` path to a JSON file with `{"card":{"TYPESAFE_API_KEY":"..."}}` when the env var is unset (never prints the value).

## Library quickstart

```python
from jev_extract import Extractor

ex = Extractor()  # reads TYPESAFE_API_KEY; model="jev-latest"

paragraph = (
    "Ada Lovelace worked on Babbage's Analytical Engine. "
    "She published notes in 1843. Those notes include an early algorithm."
)
result = ex.extract(paragraph=paragraph, question="When were the notes published?")
print(result.answer, result.start, result.end, result.char_start, result.char_end, result.confidence)
# result.start / result.end are TokenRef(pos=..., word=...)
```

### Extraction modes

`Extractor.extract(..., extract_mode=...)` and dedicated methods:

| Mode | Method | Idea |
|------|--------|------|
| `sequential` | `extract` | Start Choice → end Choice (baseline) |
| `joint` / `auto` | `extract` | Pair Choice / duplicate-aware end |
| `span_choice` | `extract_span_choice` | One Choice over contiguous windows (len 1..K≤255) |
| `anchor_expand` | `extract_anchor_expand` | Head token → left/right extent Choices |
| `shape_gate` | `extract_shape_gate` | Shape Choice → shape-specific end prior |
| `topk_noul` | `extract_topk_noul` | Top-k ends + batched Noul verify |
| `ambiguous_joint` | `extract_ambiguous_joint` | Joint when start word duplicated / low margin |
| `length_rerank` | `extract_length_rerank` | `argmax(prob × length_penalty)`; shape-aware |
| — | `extract_chunk` / `extract_cascade` | Chunk-only / cascade baselines |

Compare live:

```bash
python benchmarks/run_mode_compare.py --limit 20 --stratified \
  --modes token_sequential span_choice length_rerank shape_gate topk_noul ambiguous_joint anchor_expand
```

See the package docstring / [benchmarks/README.md](benchmarks/README.md) for field modes (`span`, `noul`, `choice`, `score`) and harness details.

## Results

Live runs write per-example JSON under `benchmarks/results/` and refresh `benchmarks/results/SUMMARY.md` with markdown tables.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests benchmarks
```

Regenerate synthetic datasets:

```bash
python benchmarks/generate_datasets.py
```

## License

MIT © 2026 Quan Dang / dangquan1402
