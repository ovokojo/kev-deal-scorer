# ---
# jupyter:
#   kernelspec:
#     display_name: Fine-tuning tutorial (uv)
#     language: python
#     name: fine-tuning-tutorial
# ---

# %% [markdown]
# # 06 · Serving it
#
# You have a model that answers three typed questions with probabilities. Now something
# else needs to call it.
#
# "Serving a model" means two quite different things, and picking the wrong one is a
# common source of wasted effort:
#
# **A generation endpoint.** Give it a prompt, it writes text back, token by token. This is
# what `mlx_lm.server` gives you, and it is the right thing when your application genuinely
# needs generated text — a chat assistant, a summariser, a code helper.
#
# **A decision endpoint.** Give it a piece of text and a set of questions, it returns
# probabilities. No tokens generated. This is what our model is for, and it is what
# `mlx_lm.server` *cannot* do, because an OpenAI chat completion has no way to express
# "answer these four questions, but don't let them see each other's answers, and give me
# the distribution over the options."
#
# This lesson builds the second one. `serve.py` in the project root is a complete, runnable
# decision endpoint in about 180 lines, and the API shape follows TypeSafe's System One —
# the same one Kev implements — so a client written against either will work with the other.
#
# **Time: about 10 minutes**, most of it waiting for the model to load.

# %%
import sys

sys.path.insert(0, "lib")

import json
import subprocess
import time
from pathlib import Path

import httpx
import pandas as pd

PORT = 8033
BASE = f"http://127.0.0.1:{PORT}"

# %% [markdown]
# ## 1. What the endpoint looks like
#
# The whole design is in the docstring of `serve.py`, but here is the shape. A request
# carries the ticket and the questions:

# %%
REQUEST = {
    "state": "Shoes arrived two weeks late and in the wrong size. I want a refund and I "
             "have already emailed twice with no reply.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "returns": "Exchanges, refunds, wrong or damaged items",
                "shipping": "Delivery status, delays, lost packages",
                "billing": "Charges, invoices, payment problems",
                "other": "Anything else",
            },
        },
        "escalate": {
            "type": "noul",
            "instructions": "Does this ticket need a human to look at it right now?",
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["calm", "annoyed", "angry"],
        },
    },
}
print(json.dumps(REQUEST, indent=2)[:700] + " ...")

# %% [markdown]
# Notice what the request does *not* contain: any mention of prompt formatting, delimiters,
# or option ordering. The client says what it wants decided; the server owns the plumbing.
#
# **But** — and this is the sharp edge — the `instructions` and option names are not free
# text. The fine-tune bound those exact strings during training. Send a request with
# different wording and you get a model answering a question it has never seen, with
# confidently wrong probabilities and no error message. We will demonstrate that in
# section 6; it is the single most common way a deployment like this breaks.
#
# ## 2. Start the server

# %%
log = Path("/tmp/decision-server.log")
log.unlink(missing_ok=True)

server = subprocess.Popen(
    [sys.executable, "serve.py", "--port", str(PORT), "--adapter-path", "adapters/decision"],
    stdout=log.open("w"), stderr=subprocess.STDOUT,
)

# Wait for the model to load. 0.6B takes ~20 s cold; the weights are cached by now.
deadline = time.time() + 180
ready = False
while time.time() < deadline:
    try:
        if httpx.get(f"{BASE}/health", timeout=2).json().get("model_loaded"):
            ready = True
            break
    except Exception:
        time.sleep(2)

print("ready" if ready else f"did not come up; see {log}")
if not ready:
    print(log.read_text()[-2000:])

# %% [markdown]
# ## 3. Ask it something

# %%
response = httpx.post(f"{BASE}/v1/systemone", json=REQUEST, timeout=120).json()
print(json.dumps(response, indent=2))

# %% [markdown]
# ### Reading the three answer types
#
# The three question types return deliberately different shapes, because they mean
# different things downstream:
#
# **`choice`** — a label plus the full distribution. `confidence` is the top probability,
# but the *distribution* is the useful part: 0.45/0.35/0.20 between two options is a very
# different situation from 0.80/0.05/0.05, and a bare label hides that entirely.
#
# **`noul`** — a single number, `P(true)`. Not a label. Whether to act on 0.55 is a policy
# decision, and you want the number, not a coin flip made inside the server.
#
# **`score`** — an *ordinal* answer, so it returns an expected value (`score`) plus the
# distribution and the legend. If the model says 53% level 1 and 41% level 2, the expected
# value is about 1.4. Rounding that to "2" throws away the fact that the model is genuinely
# unsure between adjacent levels — and for an ordinal scale, being off by one matters much
# less than being off by two, which a plain label cannot express.

# %%
answers = response["answers"]
for name, answer in answers.items():
    if answer["type"] == "choice":
        dist = "  ".join(f"{k}={v:.2f}" for k, v in answer["probabilities"].items())
        print(f"{name:12s} {answer['choice']:9s} (conf {answer['confidence']:.2f})   {dist}")
    elif answer["type"] == "noul":
        print(f"{name:12s} P(true) = {answer['noul']:.3f}")
    else:
        dist = "  ".join(f"{k}={v:.2f}" for k, v in answer["probabilities"].items())
        print(f"{name:12s} score = {answer['score']:.2f}   {dist}")

# %% [markdown]
# ## 4. How fast is it?
#
# Latency is one forward pass per question, plus the prompt tokens. We measure from the
# client side so the number includes the HTTP round trip.

# %%
def time_request(request: dict, repeats: int = 5) -> dict:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        httpx.post(f"{BASE}/v1/systemone", json=request, timeout=120)
        timings.append((time.perf_counter() - started) * 1000)
    n_questions = len(request["questions"])
    return {
        "questions": n_questions,
        "mean ms": round(sum(timings) / len(timings), 1),
        "min ms": round(min(timings), 1),
        "per question ms": round(sum(timings) / len(timings) / n_questions, 1),
    }


# Warm up first: the very first request includes Metal kernel compilation.
time_request(REQUEST, repeats=2)

rows = [
    time_request({"state": REQUEST["state"], "questions": {"department": REQUEST["questions"]["department"]}}),
    time_request({"state": REQUEST["state"], "questions": REQUEST["questions"]}),
]
latency = pd.DataFrame(rows)
print(f"ticket length: {len(REQUEST['state'])} characters\n")
latency

# %% [markdown]
# Two things to take from this table.
#
# **Cost is per question, not per ticket.** Each question is a separate forward pass over
# the same ticket text. Three questions cost roughly three times one question. This is the
# main architectural difference from Kev, which packs the shared ticket prefix and all the
# question branches into a *single* forward pass with a block-causal mask, so the ticket is
# encoded once no matter how many questions you ask. That is the optimisation to reach for
# if you scale this up — it is a meaningful constant factor, not a different algorithm.
#
# **Compare against a hosted model.** At these latencies, 50,000 tickets a day is a few
# hours on one Mac, with no per-token cost, and the data never leaves the machine. That is
# usually the whole argument for building this instead of calling an API.

# %% [markdown]
# ## 5. Wiring it into an application
#
# The response is plain JSON over HTTP, so any client works. Three common patterns:

# %%
# 1. Direct HTTP, for anything that can make a request.
print("--- httpx ---")
r = httpx.post(f"{BASE}/v1/systemone", json={
    "state": "Where is my parcel? Tracking has not updated in a week.",
    "questions": {"department": REQUEST["questions"]["department"]},
}, timeout=60).json()
print(r["answers"]["department"]["choice"], r["answers"]["department"]["confidence"])

# %%
# 2. The TypeSafe SDK, which is what Kev's README points at. Same API shape, so the same
#    client code works against either server by changing the base_url.
print("""
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8033", model="triage-latest")
result = client.evaluate(
    state=ticket_text,
    questions={
        "department": Choice("Which team should handle this ticket?",
                             returns="Exchanges and refunds", shipping="Deliveries",
                             billing="Charges and invoices", other="Anything else"),
        "escalate": Noul("Does this ticket need a human right now?"),
        "frustration": Score("How frustrated is the customer?", ["calm", "annoyed", "angry"]),
    },
)
""")

# %%
# 3. As a tool for an agent. The probabilities are what make this useful: an agent can be
#    given a hard rule ("only auto-route above 0.9 confidence") rather than being asked to
#    guess whether the model sounded sure.

# %% [markdown]
# ## 6. The failure mode to expect
#
# Now the sharp edge, demonstrated rather than described. We send the *same question* with
# the instructions reworded slightly — semantically identical, textually different:

# %%
rewritten = {
    "state": REQUEST["state"],
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which department is best suited to resolve this?",  # reworded
            "criteria": {"returns": "refunds", "shipping": "delivery", "billing": "charges", "other": "misc"},
        },
    },
}
r = httpx.post(f"{BASE}/v1/systemone", json=rewritten, timeout=120).json()["answers"]["department"]
print("reworded instruction:")
print(f"  -> {r['choice']} at {r['confidence']:.2f}   {r['probabilities']}")

r0 = httpx.post(f"{BASE}/v1/systemone", json={
    "state": REQUEST["state"], "questions": {"department": REQUEST["questions"]["department"]},
}, timeout=120).json()["answers"]["department"]
print("\ntraining instruction:")
print(f"  -> {r0['choice']} at {r0['confidence']:.2f}   {r0['probabilities']}")

# %% [markdown]
# The rewording changes the answer, and **nothing in the response tells you it happened.**
# No error, no warning, no drop in confidence necessarily. The model simply answers a
# question it was never trained on, with whatever it can improvise.
#
# Two defences, and you want both:
#
# 1. **Version the prompt.** Treat the question text as part of the model artifact — store
#    it, version it, and include the version in the response. Kev's skill makes the same
#    point: the fine-tune binds those strings.
# 2. **Monitor the label mass.** Our readout exposes it for exactly this reason. A request
#    that puts almost no probability on the label tokens is a request the model cannot
#    answer, and that is detectable at runtime without a ground-truth label.
#
# ## 7. Concurrency and batching
#
# One thing worth knowing before you put this behind a load balancer: the endpoint is
# `async def` but does blocking work, so requests are served **one at a time**. That is
# deliberate — a single MLX model with one GPU queue gains little from interleaving, and
# serialising makes latency predictable.
#
# It also avoids a real trap. MLX stores its compute streams in **thread-local** storage, so
# a model loaded on the main thread cannot run on one of FastAPI's worker threads. A plain
# `def` endpoint gets dispatched to a threadpool and dies with:
#
# ```
# RuntimeError: There is no Stream(cpu, 0) in current thread
# ```
#
# That is exactly the error this notebook's server hit the first time it was built. The fix
# is `async def` (stay on the loading thread), or creating a stream per worker thread.
#
# To go faster, the order of leverage is:
#
# 1. **Batch the readout.** All three questions share the ticket prefix. Compute the prefix
#    once, or batch the questions into one forward pass — a 2-4x win for free.
# 2. **Batch across tickets.** Serve several tickets per forward pass. This is where the
#    real throughput is, and it is what `mlx_lm.server`'s `--prompt-concurrency` does for
#    generation.
# 3. **Quantise.** 4-bit weights are ~4x smaller and materially faster per forward pass.
#    Lesson 07 measures what you give up.
# 4. **Use a smaller model.** 0.6B over 1.7B, if the accuracy holds.

# %% [markdown]
# ## 8. Shutting down

# %%
server.terminate()
try:
    server.wait(timeout=10)
    print("server stopped")
except subprocess.TimeoutExpired:
    server.kill()
    print("server killed")

# %% [markdown]
# ## 9. The generation alternative, for contrast
#
# If what you actually need is generated text, `mlx_lm.server` gives you an
# OpenAI-compatible endpoint in one command, and the adapter is applied on the fly:
#
# ```bash
# uv run mlx_lm.server --model Qwen/Qwen3-0.6B --adapter-path adapters/decision --port 8034
# curl -s localhost:8034/v1/chat/completions -H 'content-type: application/json' -d '{
#   "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32 }'
# ```
#
# Useful when your application wants text. For triage it is the wrong tool: it is slower,
# it can emit a label that was never an option, and it gives you no calibrated
# distribution — you would have to parse a string and pretend the model's confidence was 1.
#
# ---
#
# ### Exercises
#
# 1. **Measure the prompt-prefix win.** Time a request with one question versus four. How
#    close is the ratio to 4x? What does the gap tell you about how much of the cost is the
#    shared ticket prefix?
#
# 2. **Add a `label_mass` field to the response.** The server currently discards it. Return
#    it, then find an input that makes it collapse — an empty string, a ticket in another
#    language, or an instruction that mentions no options. That value is your runtime
#    detector for out-of-distribution requests.
#
# 3. **Add a confidence threshold.** Change the server so `choice` answers return
#    `choice: null` when confidence is below a threshold, plus the raw distribution. Then
#    argue for a threshold using lesson 05's coverage-at-error numbers, not a guess.
#
# 4. **Batch it.** Modify `readout` to build all of a ticket's question prompts at once and
#    run them as a single batched forward pass with left-padding. Measure the speedup. This
#    is the single highest-leverage optimisation on this server.

# %% [markdown]
# ---
# **Next:** [07 · Packaging and hosting](07_packaging.ipynb) — fusing the adapter,
# quantising, running it under Ollama, and publishing to Hugging Face.
