# jev-extract

**Paragraph information extraction** using [TypeSafe AI](https://typesafe.ai) **Jev** (System One).

Convert extractive QA into a **closed-set Jev Choice** over candidate sentence spans: the model picks which span answers the question — it does **not** freely generate answer text.

> Honest scope: this library is **extractive-only**. Answers are always substrings (or merged chunks) of the input paragraph. It is not a generative IE / OpenIE toolkit, and it is **not** a drop-in replacement for [Instructor](https://python.useinstructor.com/) (which structures LLM JSON outputs). Jev Choice gives you calibrated probabilities over a finite candidate set.

Blog / product context: [Introducing System One models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

## Install

```bash
pip install jev-extract
# or from source
pip install -e ".[dev]"
```

Requires Python **≥ 3.10**.

## Environment

```bash
export TYPESAFE_API_KEY=tsk_...
```

Optional: `TYPESAFE_DEFAULT_MODEL` (SDK default is `jev-latest`).

## Quickstart

```python
from jev_extract import Extractor

ex = Extractor()  # reads TYPESAFE_API_KEY; model="jev-latest" by default

paragraph = (
    "Ada Lovelace worked on Babbage's Analytical Engine. "
    "She published notes in 1843. Those notes include an early algorithm."
)
result = ex.extract(
    paragraph=paragraph,
    question="When were the notes published?",
)
print(result.answer)       # e.g. "She published notes in 1843."
print(result.start, result.end)
print(result.confidence)   # float in [0, 1] when returned by the API
```

### Multiple fields

```python
results = ex.extract_fields(
    paragraph=paragraph,
    fields={
        "when": {"mode": "span", "question": "When were the notes published?"},
        "about_computing": {
            "mode": "noul",
            "question": "Is this paragraph about early computing?",
        },
        "topic": {
            "mode": "choice",
            "question": "Primary topic?",
            "criteria": {
                "math": "Mathematics or algorithms",
                "biology": "Biology or medicine",
                "other": "Something else",
            },
        },
        "specificity": {
            "mode": "score",
            "question": "How specific are the historical details?",
            "criteria": ["vague", "moderate", "highly specific"],
        },
    },
)
```

Field modes:

| Mode | Needs | Value |
|------|--------|--------|
| `span` | question | Selected candidate text + `start`/`end` |
| `noul` | question; optional `criteria` `{true, false}` | Probability of yes ∈ [0, 1] |
| `choice` | question + `criteria` dict | Winning label |
| `score` | question + `criteria` list of level strings | Expected score |

### Custom / mock client

```python
ex = Extractor(client=my_fake_client, model="jev-latest")
```

Useful for unit tests — no live API required.

## How it works

1. **Propose candidates** — split the paragraph into sentence-like spans (`propose_candidates`) with character offsets (`s0`, `s1`, …).
2. **Respect Choice cardinality** — Jev Choice supports at most **255** criteria. If there are more spans, `limit_candidates` merges adjacent spans into roughly equal chunks until ≤ 255 (and emits a warning).
3. **Ask System One** — call `TypeSafeClient.system_one` with `Choice(criteria={key: text}, …)` and structured state `{paragraph, question}`.
4. **Map back** — the winning choice key is looked up to recover `answer`, `start`, and `end`.

## Cardinality (255)

Jev’s Choice primitive caps the number of alternatives. This package:

- Documents the limit as `JEV_CHOICE_MAX_CRITERIA = 255`
- Merges surplus adjacent sentences rather than silently dropping middle content

Very long individual span texts are truncated when used as criteria descriptions (offsets still refer to the full span in the paragraph).

## vs Instructor

| | **jev-extract** | **Instructor** |
|--|-----------------|----------------|
| Backend | TypeSafe System One / Jev | OpenAI-compatible chat models |
| Task | Closed-set span selection + Noul/Choice/Score | Schema-constrained generation |
| Answer text | Must appear in the paragraph | Model may paraphrase / invent |
| Calibration | Choice / score probabilities | Depends on the underlying LLM |

Use **jev-extract** when you want extractive grounding and System One primitives. Use Instructor when you need free-form structured generation.

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

## License

MIT © 2026 Quan Dang / dangquan1402
