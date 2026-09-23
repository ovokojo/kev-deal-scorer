#!/bin/bash
# Start the dealscan storefront. It scores listings with the first engine that answers:
# Laya on :8010, then Kev on :8009, else a transparent heuristic (demo mode).
cd "$(dirname "$0")"

if curl -sf http://127.0.0.1:8010/v1/models >/dev/null 2>&1; then
  echo "Laya detected on :8010, using it."
elif curl -sf http://127.0.0.1:8009/v1/models >/dev/null 2>&1; then
  echo "Kev detected on :8009, using it."
else
  echo "No model server detected, starting in DEMO MODE (heuristic scoring)."
  echo ""
  echo "Fast engine (recommended): Laya on MLX, in another terminal:"
  echo "  cd laya-server && uv run serve.py --model ../models/laya-deals"
  echo "  (models/laya-deals comes from scripts/finetune_laya.py; without it, drop --model"
  echo "   to serve the zero-shot checkpoint, which is much less accurate, see docs/benchmark.md)"
  echo ""
  echo "Generative baseline: Kev 0.8B on the Apple GPU:"
  echo "  cd ~/Desktop/test/kev && KEV_DTYPE=bf16 uv run --extra serve \\"
  echo "    python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009"
  echo "  (not kev-4b: on MPS it deadlocks after the first request, and on CPU it"
  echo "   takes ~26 s per listing, past the backend's 20 s timeout)"
  echo ""
fi

echo "Opening http://127.0.0.1:8000 ..."
exec python3 backend/server.py
