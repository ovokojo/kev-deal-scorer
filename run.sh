#!/bin/bash
# Start the deal scorer. Uses the real Kev model if its server is up,
# otherwise runs in demo mode (heuristic scoring, zero setup).
cd "$(dirname "$0")"

if curl -sf http://127.0.0.1:8009/v1/models >/dev/null 2>&1; then
  echo "Kev server detected on :8009 — using the real model."
else
  echo "Kev server not detected on :8009 — starting in DEMO MODE."
  echo ""
  echo "To use the real model, open another terminal and run:"
  echo "  git clone https://github.com/jaredpalmer/kev.git"
  echo "  cd kev && uv sync --extra serve"
  echo "  KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009"
  echo "  (On a Mac where latency matters: --run jaredpalmer/kev-4b@qwen3)"
  echo ""
fi

echo "Opening http://127.0.0.1:8000 ..."
exec python3 backend/server.py
