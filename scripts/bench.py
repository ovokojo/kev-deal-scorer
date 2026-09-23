"""Benchmark deal-scoring engines on the synthetic test split and the storefront catalog.

    # one engine at a time; predictions land in runs/bench/<name>-<split>.json
    uv run --project laya-server python scripts/bench.py run laya-zs http://127.0.0.1:8010
    uv run --project laya-server python scripts/bench.py run kev-0.8b http://127.0.0.1:8009
    uv run --project laya-server python scripts/bench.py run demo demo

    # compare everything saved so far and write docs/benchmark.md
    uv run --project laya-server python scripts/bench.py report

Every number is measured on this machine. Latency is measured two ways: one listing per
request (what a scroll-triggered card pays) and, for engines that batch, all listings in
one request (what the storefront pays when it scores a whole screen at once).
"""
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import deal  # noqa: E402
from metrics import auroc, expected_calibration_error, paired_bootstrap  # noqa: E402

OUT = ROOT / "runs" / "bench"
VERDICTS = ["great_deal", "fair_price", "bad_deal"]
LATENCY_SAMPLE = 40


def load_split(name):
    if name == "catalog":
        return json.loads((ROOT / "data" / "products.json").read_text())
    return [json.loads(l) for l in (ROOT / "data" / "eval" / f"{name}.jsonl").read_text().splitlines()]


def post(url, payload, timeout=60):
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run_engine(url, rows):
    """Returns (predictions, single-request latencies ms, batch latency ms per listing or None)."""
    questions = deal.build_questions()
    preds, single_ms, batch_ms = [], [], None
    if url == "demo":
        for p in rows:
            t0 = time.perf_counter()
            r = deal.score_demo(p)
            single_ms.append((time.perf_counter() - t0) * 1000)
            preds.append({"probs": r["probabilities"], "red": r["red_flag_prob"], "value": r["value_score"]})
        return preds, single_ms, None

    info = json.loads(urllib.request.urlopen(url + "/v1/models", timeout=3).read())
    batchable = info["models"][0].get("engine") == "laya-mlx"
    if batchable:
        t0 = time.perf_counter()
        results = []
        for i in range(0, len(rows), 50):
            chunk = rows[i:i + 50]
            body = post(url, {"states": [deal.product_state(p) for p in chunk], "questions": questions})
            results.extend(body["results"])
        batch_ms = (time.perf_counter() - t0) * 1000 / len(rows)
        answers = [r["answers"] for r in results]
        for p in rows[:LATENCY_SAMPLE]:
            t0 = time.perf_counter()
            post(url, {"state": deal.product_state(p), "questions": questions})
            single_ms.append((time.perf_counter() - t0) * 1000)
    else:
        answers = []
        for p in rows:
            t0 = time.perf_counter()
            body = post(url, {"model": deal.MODEL, "state": deal.product_state(p), "questions": questions})
            single_ms.append((time.perf_counter() - t0) * 1000)
            answers.append(body["answers"])
    for a in answers:
        preds.append({"probs": a["verdict"]["probabilities"], "red": a["red_flags"]["noul"],
                      "value": a["value"]["score"]})
    return preds, single_ms, batch_ms


def cmd_run(name, url, splits=("test", "catalog")):
    OUT.mkdir(parents=True, exist_ok=True)
    for split in splits:
        rows = load_split(split)
        t0 = time.time()
        preds, single_ms, batch_ms = run_engine(url, rows)
        rec = {"engine": name, "url": url, "split": split, "n": len(rows),
               "ids": [p["id"] for p in rows], "preds": preds,
               "gold": [p["label"] for p in rows],
               "single_ms": single_ms, "batch_ms_per_listing": batch_ms,
               "wall_s": round(time.time() - t0, 1)}
        (OUT / f"{name}-{split}.json").write_text(json.dumps(rec))
        m = metrics(rec)
        print(f"{name:14s} {split:8s} n={len(rows):4d}  acc={m['acc']:.3f}  macroF1={m['f1']:.3f}  "
              f"scam AUROC={m['auroc']:.3f} recall@.5={m['recall']:.2f}  "
              f"p50={m['p50']:.0f}ms  ({rec['wall_s']}s)")


def metrics(rec):
    gold = np.array([VERDICTS.index(g["verdict"]) for g in rec["gold"]])
    probs = np.array([[p["probs"][v] for v in VERDICTS] for p in rec["preds"]])
    pred = probs.argmax(1)
    scam = np.array([g["scam"] for g in rec["gold"]], dtype=int)
    red = np.array([p["red"] for p in rec["preds"]])
    f1s = []
    for k in range(3):
        tp = ((pred == k) & (gold == k)).sum()
        prec = tp / max((pred == k).sum(), 1)
        rec_k = tp / max((gold == k).sum(), 1)
        f1s.append(0 if tp == 0 else 2 * prec * rec_k / (prec + rec_k))
    flagged = red >= 0.5
    return {
        "acc": float((pred == gold).mean()),
        "f1": float(np.mean(f1s)),
        "ece": expected_calibration_error(probs, gold),
        "auroc": auroc(red, scam) if 0 < scam.sum() < len(scam) else float("nan"),
        "recall": float(flagged[scam == 1].mean()) if scam.sum() else float("nan"),
        "false_flags": int((flagged & (scam == 0)).sum()),
        "p50": statistics.median(rec["single_ms"]),
        "p95": float(np.percentile(rec["single_ms"], 95)),
        "batch": rec["batch_ms_per_listing"],
        "correct": (pred == gold),
    }


def cmd_report():
    recs = {}
    for f in sorted(OUT.glob("*.json")):
        r = json.loads(f.read_text())
        recs[(r["engine"], r["split"])] = r
    lines = ["# Deal-scoring benchmark", "",
             "Measured on this M4 Pro / 24 GB. Regenerate with `scripts/bench.py report`.", "",
             "## How to read this table", "",
             "* `demo` is the price-ratio rule that *generated* the labels, not a model. It scores",
             "  1.000 because it is the answer key. Read it as the ceiling, never as a baseline.",
             "* The storefront catalog's labels come from that same rule. So `laya-ft` scoring 25/25",
             "  means \"applied the learned rule to 25 hand-written listings from unseen products\".",
             "  It is not independent ground truth. A human-labelled set is the missing piece.",
             "* `laya-ft` = `models/laya-deals`, built by `scripts/finetune_laya.py` (2 epochs, 6,146 s).",
             "* `laya-zs` = `aac6fef/laya-typed-decisions-mlx` as shipped, and `kev-0.8b` on MPS.",
             "* Latency is per listing, for the ~500-token states this project sends with 3 questions.",
             "  The upstream Laya benchmark reports 13 ms for much shorter single-question states;",
             "  that number does not transfer here.", ""]
    for split, title in [("test", "Synthetic test split (400 listings, 8 held-out products)"),
                         ("catalog", "Storefront catalog (25 hand-written listings, never trained on)")]:
        lines += [f"## {title}", "",
                  "| engine | verdict acc | macro-F1 | ECE | scam AUROC | scam recall @0.5 | false flags | p50 / p95 ms (1 listing) | batched ms / listing |",
                  "|---|---|---|---|---|---|---|---|---|"]
        rows = {e: r for (e, s), r in recs.items() if s == split}
        ms = {e: metrics(r) for e, r in rows.items()}
        for e, m in ms.items():
            batch = f"{m['batch']:.1f}" if m["batch"] else "—"
            lines.append(f"| {e} | {m['acc']:.3f} | {m['f1']:.3f} | {m['ece']:.3f} | {m['auroc']:.3f} | "
                         f"{m['recall']:.2f} | {m['false_flags']} | {m['p50']:.0f} / {m['p95']:.0f} | {batch} |")
        base = "kev-0.8b" if "kev-0.8b" in ms else None
        if base and split == "test":
            lines += ["", f"Paired bootstrap of verdict accuracy against `{base}` (10,000 resamples):", ""]
            for e, m in ms.items():
                if e == base:
                    continue
                d = paired_bootstrap(m["correct"], ms[base]["correct"])
                lines.append(f"- `{e}` − `{base}`: {d['diff']:+.3f} (95% CI {d['lo']:+.3f} to {d['hi']:+.3f})")
        lines.append("")
    lines += ["## The two failures this project started with", "",
              "Kev 0.8B, the model the original demo shipped with, does not answer the scam question.",
              "Scam recall at the 0.5 threshold is 0.00 on both splits, while its red-flag *ranking* is",
              "good (AUROC 0.96). The signal is there but the operating point is wrong, and its verdict",
              "accuracy of 0.448 is close to the majority class. Zero-shot Laya is no better (0.338) and",
              "also never fires the scam question, which matches its own docs: the base checkpoint is near",
              "chance on typed decisions, and fine-tuning is where the value is.", "",
              "After fine-tuning, both failures are gone: 0.945 verdict accuracy on held-out products",
              "with an honestly fitted temperature (ECE 0.023), and every scam recalled with 3 false",
              "flags in 400.", "",
              "## The gate", "",
              "* Laya zero-shot as the default engine: **failed** (0.338 vs Kev's 0.448), so it was fine-tuned.",
              "* `laya-ft` as the default engine: accuracy 0.945 vs 0.448, paired bootstrap +0.497",
              "  (95% CI +0.445 to +0.550), scam recall 1.00 vs 0.00. **Passed on accuracy and recall.**",
              "* It **misses the 50 ms latency target** at ~100 ms per listing, batched or not.",
              "  Batching removes HTTP round trips, not per-listing compute: the encoder pass over a",
              "  ~500-token state with 3 questions is what costs 100 ms. That is still 7x faster than",
              "  Kev and fast enough to score a screen of cards while the shopper reads them, so Laya",
              "  is the default in `run.sh` with the miss stated rather than hidden.", ""]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "benchmark.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        cmd_run(sys.argv[2], sys.argv[3], tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ("test", "catalog"))
    else:
        cmd_report()
