# dealscan — "is it a deal?" checked as you browse

A visual proof of concept for a Jev-style **System One decision model** applied to
online shopping. Browse a resale storefront and a decision model evaluates each listing
the moment it scrolls into view. You get a verdict chip backed by probabilities, not prose.

Two local engines speak the same System One protocol, and either one can drive the shop:

| engine | what it is | per listing on this M4 Pro |
|---|---|---|
| **Laya** (`laya-server/`) | 421M ModernBERT decision encoder on MLX, fine-tuned here on deal data | ~100 ms |
| **Kev** ([jaredpalmer/kev](https://github.com/jaredpalmer/kev)) | Qwen3.5-0.8B decision model on the Apple GPU | ~0.7 s |

## How it works

```
┌─────────────┐   POST /api/score {ids}   ┌──────────────────┐   POST /v1/systemone   ┌──────────────────┐
│ storefront  │ ────────────────────────▶ │  backend/server  │ ─────────────────────▶ │ Laya :8010       │
│ (frontend/) │ ◀──────────────────────── │  backend/deal.py │ ◀───────────────────── │  or Kev :8009    │
└─────────────┘   verdict + probabilities └──────────────────┘   choice/score/noul    └──────────────────┘
```

Each listing is sent as `state`: title, asking price, MSRP, **90-day price history**
(typical / low / high), condition, seller feedback and description. Three typed
questions ride along:

| Question | Type | What it answers |
|---|---|---|
| `verdict` | `choice` | great_deal / fair_price / bad_deal, with probabilities |
| `value` | `score` | 0–4 value-for-money rating |
| `red_flags` | `noul` | probability the listing is a scam |

The shop shows a verdict only when its top probability clears **60%** ("not sure"
otherwise), and a red-flag probability of **50% or more overrides the price verdict**
("Likely scam"). A $39 pair of AirPods is not a great deal if it probably isn't real.

**Why the price history is in the state.** Earlier versions made the model recall 2026
street prices from memory. No model under 1B parameters can do that reliably, and the demo's
own reference prices went stale anyway. Deal trackers (Keepa, CamelCamelCamel) show a price
history, so the model gets one too. Its job becomes comparing the listing against it and
reading the seller.

Laya answers a whole screen of listings in **one batched request**. Kev answers one
listing at a time, so cards light up in sequence. With no engine running, the backend
falls back to a transparent price-ratio heuristic (**demo mode**) and the strip at the
top says so. If a live engine errors, cards say "Scorer error". They never quietly show
heuristic numbers under a live-model label.

## Quick start

```bash
./run.sh                       # → http://127.0.0.1:8000 (demo mode until an engine is up)
```

**Laya, the fast engine.** In another terminal:

```bash
cd laya-server
uv run serve.py --model ../models/laya-deals      # the fine-tuned checkpoint
# uv run serve.py                                 # zero-shot laya-typed-decisions, for comparison
```

`models/laya-deals` isn't committed (≈850 MB). Build it with
`uv run --project laya-server python scripts/finetune_laya.py`.

**Kev, the generative baseline:**

```bash
git submodule update --init kev    # once: kev/ is a submodule of jaredpalmer/kev
cd kev
KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009
```

Use **0.8B, not 4B**. On MPS, `kev-4b` deadlocks after the first request. On CPU it takes
about 26 s per listing, past the backend's 20 s timeout.

The backend uses Laya if it answers on :8010, else Kev on :8009, else demo mode, and
re-checks every 10 s.

### The lab: how it decides

[http://127.0.0.1:8000/lab.html](http://127.0.0.1:8000/lab.html) is the same storefront,
but every evaluation is a live run walked through six stages:

1. the listing
2. the exact state string
3. the three typed questions
4. the model pass, drawn for whichever engine answered
5. the structured output, verbatim
6. the verdict with a timing waterfall

"See how Laya decided →" in a product drawer deep-links to that listing's run (`/lab.html#p2`).

## Data, evaluation and fine-tuning

```
data/products.json        25 storefront listings with hidden labels (8 great · 8 fair · 6 bad · 3 scams)
scripts/make_eval.py      synthetic listings from 40 product templates, split by template
scripts/bench.py          accuracy, calibration, scam recall, latency per engine → docs/benchmark.md
scripts/finetune_laya.py  fine-tunes laya-typed-decisions natively in MLX → models/laya-deals
```

The label rule lives at the top of `scripts/make_eval.py`. Both the synthetic splits and
the storefront catalog are labelled by that rule, so the demo heuristic *is* the rule and
scores a perfect 1.000 by construction. That is the ceiling, not a baseline, and no model
number here is measured against human judgement. A human-labelled set is the piece this
project still lacks. [docs/benchmark.md](docs/benchmark.md) says so next to the numbers.

For background on how a fine-tune like `finetune_laya.py` works, see
[fine-tuning/](fine-tuning/README.md), a hands-on tutorial that runs on an Apple Silicon
Mac. It fine-tunes a ModernBERT encoder classifier (the kind of model Laya is) and a LoRA
decision model on Qwen3-0.6B (the approach Kev takes), then calibrates and serves both.
The notebooks are committed with their outputs, so you can read them without running
anything.

## Project layout

```
kev-deal-scorer/
├── backend/
│   ├── server.py      # stdlib HTTP server: static frontend + /api/*
│   └── deal.py        # engine detection, System One requests, demo heuristic
├── laya-server/
│   └── serve.py       # laya-mlx behind /v1/systemone, with a batched `states` extension
├── frontend/
│   ├── index.html     # the shop: grid, product drawer, cart
│   ├── lab.html       # the six-stage run inspector
│   ├── catalog.js     # shared grid, scoring queue, verdict rules, sort + filters
│   ├── app.js / lab.js
│   ├── styles.css
│   └── img/           # product photos (Unsplash License; credits in data/photo-credits.json)
├── scripts/           # make_eval, bench, finetune_laya, metrics
├── data/              # catalog, eval splits, photo credits
├── docs/benchmark.md
├── kev/               # submodule: jaredpalmer/kev, the Kev engine
├── fine-tuning/       # tutorial notebooks: fine-tuning small decision models on a Mac
└── run.sh
```

## API

- `GET /api/health` → `{"engine": "laya", "run": "…/laya-deals", "batch": true, "products": 25}`
- `GET /api/products` → listings (labels stripped)
- `POST /api/score` `{"id": "p4"}` → one result; `{"ids": [...]}` → `{"results": [...]}`.
  Each result carries `verdict`, `probabilities`, `confidence` (top probability),
  `value_score`, `red_flag_prob`, `mode` (`live` / `demo` / `error`), `engine`, and a full
  request/response `trace`.
