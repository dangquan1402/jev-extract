# Sponsor Skip Demo

Minimal static page: pick a cached video (or paste a YouTube URL), see gold vs predicted sponsor intervals, and **seek / skip to segment end** via the YouTube IFrame Player API. Fallback: `&t=Ns` links.

## Open

```bash
cd benchmarks/sponsor_skip/demo
python -m http.server 8765
# → http://127.0.0.1:8765/
```

Must be served over HTTP (not `file://`) so `data/preds.json` and the YT API can load.

## Data

- `data/preds.json` — built from `../results/results_regex_propose_jev_confirm.json` (gold + pred seconds + metrics).
- Refresh after a bench run:

```bash
# from repo root
python - <<'PY'
import json
from pathlib import Path
HERE = Path("benchmarks/sponsor_skip")
rows = json.loads((HERE/"results/results_regex_propose_jev_confirm.json").read_text())["rows"]
# or results_tony_line_scan.json
videos = [{
  "video_id": r["video_id"],
  "gold": r["gold"],
  "pred": r["pred_segments"],
  "metrics": {k: r["metrics"].get(k) for k in ("iou","precision","recall","f1","estimated_usd")},
  "source": "results_regex_propose_jev_confirm.json",
} for r in rows]
(HERE/"demo/data/preds.json").write_text(json.dumps({"videos": videos}, indent=2))
print("updated", len(videos))
PY
```

## Live detect (BYOK)

Not wired into the browser (keeps the demo static). Run:

```bash
python benchmarks/sponsor_skip/run_sponsor_skip_bench.py \
  --limit 6 --modes tony_line_scan regex_propose_jev_confirm \
  --video-ids qgLaCZyKv_8
```

Then refresh `preds.json` from the mode results you want to demo.
