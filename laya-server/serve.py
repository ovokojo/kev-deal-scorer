"""Laya on MLX, served over the System One protocol Kev speaks.

    uv run serve.py                                    # laya-typed-decisions, :8010
    uv run serve.py --model aac6fef/laya-mlx           # the base checkpoint
    uv run serve.py --model ../models/laya-deals       # a local fine-tuned checkpoint

POST /v1/systemone takes Kev's request ({"state", "questions"}) and returns Kev's
response shape. It also takes {"states": [...], "questions"} and answers every
listing in one batched pass through the encoder: {"results": [{answers, usage}, ...]}.
"""
import argparse
import time
from typing import Any

import laya_mlx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from laya_mlx.agent import collate_items
from laya_mlx.common import confidence_from_probs, temp_bucket
from pydantic import BaseModel

app = FastAPI()
agent: laya_mlx.Agent = None
model_name = ""


class Request(BaseModel):
    model: str | None = None
    state: Any = None
    states: list[Any] | None = None
    questions: dict[str, dict]


def answer(q, qtype, k, logits_row, act_row):
    """One question's answer from its logits. Mirrors laya_mlx.Agent.system_one so that
    batched and single-listing requests return identical numbers."""
    scale = agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype])
    z = logits_row[:k] / scale
    p = np.exp(z - z.max())
    p /= p.sum()
    out = {
        "type": q["t"],
        "confidence": round(confidence_from_probs(p, k), 4),
        "action": {"act_probability": round(float(act_row[0]), 4)},
    }
    if q["t"] == "choice":
        labels = list(q["crit"])
        out.update(choice=labels[int(p.argmax())],
                   probabilities={l: round(float(v), 4) for l, v in zip(labels, p)})
    elif q["t"] == "score":
        out.update(score=round(float((np.arange(k) * p).sum()), 4),
                   legend={str(i): v for i, v in enumerate(q["crit"])},
                   probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)})
    else:
        out.update(noul=round(float(p[1]), 4),
                   confidence=round(max(float(p[1]), 1.0 - float(p[1])), 4))
    return out


def predict_many(states, questions):
    """Every (listing, question) pair goes through the encoder in shared batches."""
    qids = list(questions)
    items, meta = [], []
    for s_idx, state in enumerate(states):
        its, internal = agent.prepare(state, questions)
        for qid, it, q in zip(qids, its, internal):
            items.append(it)
            meta.append((s_idx, qid, q))
    results = [{"answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}} for _ in states]
    for start in range(0, len(items), agent.batch_size):
        chunk = items[start:start + agent.batch_size]
        batch = collate_items(chunk, agent.tok.pad_token_id, pad_to_multiple=agent.pad_to_multiple,
                              max_length=agent.cfg.get("max_len", 512))
        logits, act = agent.forward(batch)
        logits, act = np.asarray(logits), np.asarray(act)
        if not np.isfinite(logits).all():
            raise FloatingPointError("non-finite model outputs; retry with --dtype float32")
        act = np.exp(act - act.max(axis=-1, keepdims=True))
        act /= act.sum(axis=-1, keepdims=True)
        for row, it in enumerate(chunk):
            s_idx, qid, q = meta[start + row]
            results[s_idx]["answers"][qid] = answer(q, it["qtype"], len(it["markers"]),
                                                   logits[row], act[row])
            results[s_idx]["usage"]["input_tokens"] += len(it["ids"])
    return results


@app.get("/v1/models")
def models():
    return {"models": [{"id": "laya", "run": model_name, "engine": "laya-mlx",
                        "dtype": str(agent.dtype), "batch_size": agent.batch_size}]}


@app.post("/v1/systemone")
def systemone(req: Request):
    if (req.state is None) == (req.states is None):
        raise HTTPException(400, "send exactly one of `state` or `states`")
    t0 = time.perf_counter()
    try:
        results = predict_many(req.states if req.states is not None else [req.state], req.questions)
    except ValueError as e:
        raise HTTPException(400, str(e))
    ms = round((time.perf_counter() - t0) * 1000, 1)
    if req.states is None:
        return {"model": model_name, **results[0], "latency_ms": ms}
    per = round(ms / max(len(results), 1), 1)
    return {"model": model_name, "latency_ms": ms,
            "results": [{**r, "latency_ms": per} for r in results]}


def main():
    global agent, model_name
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="aac6fef/laya-typed-decisions-mlx")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=48)
    args = ap.parse_args()

    model_name = args.model
    t0 = time.time()
    agent = laya_mlx.load(args.model, dtype=args.dtype, batch_size=args.batch_size,
                          pad_to_multiple=32)
    # warm up so the first real request does not pay for kernel compilation
    predict_many(["warm up"], {"w": {"type": "noul", "instructions": "Is this a test?"}})
    print(f"laya ready: {args.model} ({args.dtype}) in {time.time() - t0:.1f}s")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
