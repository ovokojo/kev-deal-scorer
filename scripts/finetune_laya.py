"""Fine-tune laya-typed-decisions on the synthetic deal data, natively in MLX.

    uv run --project laya-server python scripts/finetune_laya.py            # → models/laya-deals

laya_mlx's DecisionModel is an ordinary mlx.nn.Module, so there is no PyTorch round trip
and no conversion step: train it, then save it in the exact layout laya_mlx.convert
writes, and `laya-server/serve.py --model models/laya-deals` serves it as-is.

Targets, per listing, for the three production questions in backend/deal.py:
  verdict   -> the rule label (see scripts/make_eval.py)
  red_flags -> 1 if scam
  value     -> 0 for a scam, else round(2 + (1 - ratio) * 6) clipped to 0..4
After training, a temperature per question bucket is refit on dev and written into the
checkpoint config; results are printed with and without it.
"""
import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import laya_mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from laya_mlx.agent import collate_items
from laya_mlx.common import temp_bucket
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
import deal  # noqa: E402

QUESTIONS = deal.build_questions()
VERDICTS = list(QUESTIONS["verdict"]["criteria"])


def rows(split):
    return [json.loads(l) for l in (ROOT / "data" / "eval" / f"{split}.jsonl").read_text().splitlines()]


def targets(p):
    ratio = p["price"] / p["typical_price"]
    value = 0 if p["label"]["scam"] else max(0, min(4, round(2 + (1 - ratio) * 6)))
    return {"verdict": VERDICTS.index(p["label"]["verdict"]),
            "red_flags": int(p["label"]["scam"]), "value": value}


def items_for(agent, listings):
    """One training item per (listing, question): token ids, marker positions, target."""
    out = []
    for p in listings:
        its, _ = agent.prepare(deal.product_state(p), QUESTIONS)
        tg = targets(p)
        for qid, it in zip(QUESTIONS, its):
            out.append({**it, "qid": qid, "target": tg[qid], "listing": p["id"]})
    return out


def batch_arrays(agent, chunk):
    b = collate_items(chunk, agent.tok.pad_token_id, pad_to_multiple=32,
                      max_length=agent.cfg.get("max_len", 512))
    t = mx.array([it["target"] for it in chunk])
    return {k: mx.array(v) for k, v in b.items()}, t


def loss_fn(model, batch, target):
    logits, _ = model(**batch)
    return nn.losses.cross_entropy(logits, target, reduction="mean")


def raw_logits(agent, items, bs=64):
    """Logits per item, sliced to that item's option count, no temperature applied."""
    out = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        b, _ = batch_arrays(agent, chunk)
        logits, _ = agent.model(**b)
        logits = np.asarray(logits)
        out.extend(logits[r, :len(it["markers"])] for r, it in enumerate(chunk))
    return out


def softmax(z, t=1.0):
    z = z / t
    e = np.exp(z - z.max())
    return e / e.sum()


def fit_temperature(logit_rows, targets_):
    """Grid-search T minimising NLL (a 1-D problem; a grid is exact enough and cannot diverge)."""
    grid = np.exp(np.linspace(math.log(0.3), math.log(5.0), 200))
    nll = [-np.mean([math.log(max(softmax(z, t)[y], 1e-12)) for z, y in zip(logit_rows, targets_)])
           for t in grid]
    return float(grid[int(np.argmin(nll))])


def evaluate(agent, items, temps=None):
    temps = temps or {}
    logits = raw_logits(agent, items)
    by_q = {}
    for it, z in zip(items, logits):
        p = softmax(z, temps.get(it["qid"], 1.0))
        by_q.setdefault(it["qid"], []).append((p, it["target"]))
    v = by_q["verdict"]
    acc = np.mean([p.argmax() == y for p, y in v])
    nll = -np.mean([math.log(max(p[y], 1e-12)) for p, y in v])
    red = by_q["red_flags"]
    scams = [p[1] for p, y in red if y == 1]
    clean = [p[1] for p, y in red if y == 0]
    return {"verdict_acc": round(float(acc), 3), "verdict_nll": round(float(nll), 3),
            "scam_recall@.5": round(float(np.mean([s >= 0.5 for s in scams])), 3),
            "false_flags": int(sum(c >= 0.5 for c in clean))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="aac6fef/laya-typed-decisions-mlx")
    ap.add_argument("--out", default=str(ROOT / "models" / "laya-deals"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    random.seed(0)
    mx.random.seed(0)
    agent = laya_mlx.load(args.base, dtype="float32")
    model = agent.model
    train, dev = items_for(agent, rows("train")), items_for(agent, rows("dev"))
    print(f"train items {len(train)}, dev items {len(dev)}")
    print("dev before:", evaluate(agent, dev))

    steps = args.epochs * math.ceil(len(train) / args.batch)
    warmup = max(1, steps // 20)
    sched = optim.join_schedules([optim.linear_schedule(0, args.lr, warmup),
                                  optim.cosine_decay(args.lr, steps - warmup)], [warmup])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.01)
    step_fn = nn.value_and_grad(model, loss_fn)

    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        random.shuffle(train)
        for i in range(0, len(train), args.batch):
            b, t = batch_arrays(agent, train[i:i + args.batch])
            loss, grads = step_fn(model, b, t)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        print(f"dev after epoch {epoch}:", evaluate(agent, dev), flush=True)

    # refit one temperature per production question on dev, then store it in the
    # checkpoint's bucket table so laya_mlx applies it at inference
    logits = raw_logits(agent, dev)
    temps = {}
    for qid in QUESTIONS:
        sel = [(z, it["target"]) for it, z in zip(dev, logits) if it["qid"] == qid]
        temps[qid] = fit_temperature([z for z, _ in sel], [y for _, y in sel])
    print("fitted temperatures:", {k: round(v, 3) for k, v in temps.items()})
    print("dev at T=1:   ", evaluate(agent, dev))
    print("dev at fitted:", evaluate(agent, dev, temps))

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(agent.model_dir / "tokenizer", out / "tokenizer")
    (out / "encoder").mkdir()
    (out / "encoder" / "config.json").write_text(json.dumps(agent.encoder_cfg, indent=2) + "\n")
    cfg = dict(agent.cfg)
    buckets = dict(cfg.get("temperature_by_options", {}))
    qtype = {"choice": 0, "score": 1, "noul": 2}
    for qid, q in QUESTIONS.items():
        k = len(q.get("criteria") or {}) if q["type"] != "noul" else 2
        buckets[temp_bucket(qtype[q["type"]], k)] = temps[qid]
    cfg.update(temperature_by_options=buckets, model_name="laya-deals",
               fine_tuned_on="kev-deal-scorer synthetic deals (scripts/make_eval.py)")
    (out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    params = {k: v.astype(mx.float16) for k, v in tree_flatten(model.parameters())}
    mx.save_safetensors(str(out / "model.safetensors"), params)
    (out / "mlx_config.json").write_text(json.dumps(
        {"format": "laya-mlx", "format_version": 1, "dtype": "float16", "source": args.base}, indent=2))
    print(f"saved {out} in {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
