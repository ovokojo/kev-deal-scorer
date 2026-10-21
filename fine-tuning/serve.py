"""A local System One-style decision endpoint.

    uv run python serve.py                      # serves adapters/decision on :8033
    uv run python serve.py --port 8010 --temperature-file models/temperature.json

The API shape follows TypeSafe's System One, the same shape Kev implements, so a client
written against one can be pointed at the other. The request carries the questions; the
server returns calibrated probabilities, not generated text.

    POST /v1/systemone   {"state": "...", "questions": {...}}
    GET  /v1/models
    GET  /health

**The prompt is part of the artifact.** This server renders prompts with the exact template
the adapter was trained on. If you change the template, the instructions, or the option
names, you must retrain — the fine-tune binds those strings. That is not a limitation of
this implementation; it is true of every fine-tuned classifier.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from readout import LabelTokenError, label_token_ids  # noqa: E402

# Must match tickets.PROMPT_TEMPLATE exactly. Duplicated deliberately: this file is meant to
# be readable as a standalone service, and the duplication is one of the things lesson 04
# warns you about. If you change one, change the other and retrain.
PROMPT_TEMPLATE = """<ticket>
{state}
</ticket>

<question>{instructions}</question>
<options>{options}</options>
<answer>"""


class Question(BaseModel):
    type: str = Field(pattern="^(choice|noul|score)$")
    instructions: str
    criteria: dict | list | None = None      # option descriptions, or the level legend
    options: list[str] | None = None         # optional explicit option order


class Request(BaseModel):
    state: str
    model: str | None = None
    questions: dict[str, Question]


app = FastAPI(title="decision-model", version="0.1.0")

STATE: dict = {"model": None, "tokenizer": None, "temperature": {}}


def options_for(question: Question) -> list[str]:
    """The label strings for a question, in the order the model was trained on."""
    if question.type == "noul":
        return ["false", "true"]
    if question.options:
        return question.options
    if isinstance(question.criteria, dict):
        return list(question.criteria)
    if isinstance(question.criteria, list):
        # score levels: index them, because words are usually multi-token
        return [str(i) for i in range(len(question.criteria))]
    raise HTTPException(400, "choice/score questions need `criteria` or `options`")


def readout(state_text: str, question: Question) -> tuple[dict, float]:
    """One forward pass; softmax over the label tokens at the final position.

    Thread affinity matters here. MLX keeps its compute streams in thread-local storage, so
    a model created on the main thread cannot run on one of FastAPI's worker threads; you
    get `RuntimeError: There is no Stream(cpu, 0) in current thread`. The endpoint below is
    therefore `async def`, which keeps execution on the event loop's thread — the same one
    that loaded the model. A `def` endpoint would be dispatched to a threadpool and fail.
    """
    options = options_for(question)
    try:
        token_ids = label_token_ids(STATE["tokenizer"], options)
    except LabelTokenError as e:
        raise HTTPException(400, str(e))

    prompt = PROMPT_TEMPLATE.format(
        state=state_text.strip(),
        instructions=question.instructions,
        options=" | ".join(options),
    )
    ids = mx.array(STATE["tokenizer"].encode(prompt))[None]
    temperature = STATE["temperature"].get(question.instructions, 1.0)
    probs = mx.softmax(STATE["model"](ids)[:, -1, :] / temperature, axis=-1)[0]

    raw = {label: probs[i].item() for label, i in token_ids.items()}
    mass = sum(raw.values())
    if mass <= 0:
        raise HTTPException(500, "label_mass is zero — the model put no probability on these labels")
    return {label: v / mass for label, v in raw.items()}, mass


@app.post("/v1/systemone")
async def systemone(request: Request) -> dict:
    """`async def` on purpose — see the note in `readout`.

    This blocks the event loop for the duration of the forward passes, which serialises
    requests. For a tutorial that is the behaviour you want: one shared model, one GPU
    queue, no interleaving. Server 01 is a benchmark harness, not a production gateway —
    section 6 of the notebook discusses what you would change.
    """
    started = time.perf_counter()
    answers: dict[str, dict] = {}

    for name, question in request.questions.items():
        probabilities, _mass = readout(request.state, question)

        if question.type == "noul":
            answers[name] = {"type": "noul", "noul": probabilities.get("true", 0.0)}
        elif question.type == "score":
            best = max(probabilities, key=probabilities.get)
            answers[name] = {
                "type": "score",
                "score": sum(int(k) * v for k, v in probabilities.items()),
                "confidence": probabilities[best],
                "probabilities": {k: round(v, 6) for k, v in probabilities.items()},
                "legend": question.criteria if isinstance(question.criteria, list) else None,
            }
        else:
            best = max(probabilities, key=probabilities.get)
            answers[name] = {
                "type": "choice",
                "choice": best,
                "confidence": round(probabilities[best], 6),
                "probabilities": {k: round(v, 6) for k, v in probabilities.items()},
            }

    return {
        "model": request.model or "triage-latest",
        "answers": answers,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@app.get("/v1/models")
def models() -> dict:
    return {"data": [{"id": "triage-latest", "object": "model"}]}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": STATE["model"] is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter-path", default="adapters/decision")
    parser.add_argument("--temperature-file", default=None,
                        help="JSON mapping question instructions -> fitted temperature")
    parser.add_argument("--host", default="127.0.0.1")
    # 8033 by default: 8008/8009 are what the Kev reference server uses, and you may
    # well have one of those already running.
    parser.add_argument("--port", type=int, default=8033)
    args = parser.parse_args()

    from mlx_lm import load

    print(f"loading {args.model} + {args.adapter_path} ...")
    STATE["model"], STATE["tokenizer"] = load(args.model, adapter_path=args.adapter_path)
    if args.temperature_file and Path(args.temperature_file).exists():
        STATE["temperature"] = json.loads(Path(args.temperature_file).read_text())
        print(f"applied {len(STATE['temperature'])} fitted temperatures")
    print(f"ready on http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
