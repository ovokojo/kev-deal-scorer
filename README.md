# Dealscan — "is it a deal?" scorer powered by Kev

A visual proof of concept for a Jev-style **System One decision model** applied to
online shopping: swipe through mock listings and the model instantly tells you
whether each one is a great deal, a fair price, or a bad deal — with probabilities,
not prose.

Built around [Kev](https://github.com/jaredpalmer/kev) (Jared Palmer's open-source
Jev clone, Apache 2.0), which speaks TypeSafe's System One API: one request carries
a state (the listing) plus typed questions, and the model answers all of them in a
single forward pass.

## How it works

```
┌─────────────┐      POST /api/score       ┌──────────────────┐     POST /v1/systemone    ┌─────────────┐
│  swipe UI   │ ─────────────────────────▶ │  backend/server  │ ────────────────────────▶ │ Kev :8009   │
│ (frontend/) │ ◀───────────────────────── │  backend/deal.py │ ◀──────────────────────── │ (kev.serve) │
└─────────────┘      verdict + probs       └──────────────────┘    Choice/Score/Noul     └─────────────┘
```

Each listing is sent as `state`, with three questions asked in one pass:

| Question | Type | What it answers |
|---|---|---|
| `verdict` | `choice` | great_deal / fair_price / bad_deal (+ probabilities, confidence) |
| `value` | `score` | 0–4 value-for-money rating |
| `red_flags` | `noul` | probability the listing looks scammy |

**Demo mode:** if no Kev server is reachable, the backend falls back to a
transparent price-vs-market heuristic so the UI works with zero setup. The pill in
the header always shows which mode you're in.

## Quick start

```bash
./run.sh
# → http://127.0.0.1:8000
```

No dependencies — the backend is stdlib-only Python 3.

### Using the real model (recommended for the demo)

On your Mac, in another terminal:

```bash
git clone https://github.com/jaredpalmer/kev.git
cd kev
uv sync --extra serve
KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

Then re-run `./run.sh` — the header pill flips to "kev live" and every card is
scored by the model. On Apple Silicon where latency matters, the Qwen3 build is
faster: `--run jaredpalmer/kev-4b@qwen3`.

Note: `market_price` in `data/products.json` is used only by demo mode and is
**not** sent to the model — Kev judges from its own knowledge, like a real shopper.

## Project layout

```
kev-deal-scorer/
├── backend/
│   ├── server.py   # stdlib HTTP server: static frontend + /api/*
│   └── deal.py     # System One request builder, Kev client, demo fallback
├── frontend/
│   ├── index.html  # swipeable card UI (mobile-first)
│   ├── styles.css
│   └── app.js
├── data/
│   └── products.json  # 12 mock listings: great deals, fair prices, traps
└── run.sh
```

## API

- `GET /api/health` → `{"kev": true, "products": 12}`
- `GET /api/products` → listing array
- `POST /api/score` `{"id": "p4"}` → verdict, probabilities, value score, red-flag
  probability, latency, and `mode` (`"kev"` or `"demo"`)

## Ideas for tomorrow's build session

- Swap the mock catalog for a real feed (eBay / Facebook Marketplace API).
- Add a `choice` question that picks *which* listing to buy from a shortlist.
- Threshold the confidence: only show the badge when confidence ≥ 0.6, else "not sure".
- Try the hosted Jev API via the same SDK for a side-by-side calibration comparison.
