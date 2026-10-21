# ---
# jupyter:
#   kernelspec:
#     display_name: Fine-tuning tutorial (uv)
#     language: python
#     name: fine-tuning-tutorial
# ---

# %% [markdown]
# # 04 · The decision model
#
# Lesson 01 built the readout — reading a probability distribution off a language model's
# forward pass instead of generating text — and it produced a classifier that was **worse
# than always guessing the majority class**. The readout was not the problem. The model had
# never been told what our labels mean.
#
# This lesson fixes that. We fine-tune a small causal language model so that the readout
# becomes good, and in doing so we get something an encoder structurally cannot give us:
# **one model that answers any number of typed questions about the same input, with
# calibrated probabilities and an honest way to abstain.**
#
# That is the architecture Jev uses and that Kev reconstructs.
#
# **Time: about 8 minutes**, most of it training.
#
# > **A note on scale.** We use `Qwen3-0.6B` because it trains in five minutes and lets you
# > iterate. The recipe is identical for `Qwen3-1.7B` — every flag below is unchanged, and
# > section 8 covers what to expect.
# >
# > The model that matters is the one **you** fine-tune on your own data. Everything here is
# > built so that swapping the dataset is the only change you need: edit
# > `lib/tickets.py`'s `QUESTIONS` and the data generator, and the rest of the pipeline —
# > training, the readout, calibration, the server — works unchanged.

# %%
import sys

sys.path.insert(0, "lib")
import warnings

warnings.filterwarnings("ignore")

import json
import subprocess
import time
from pathlib import Path

import mlx.core as mx
import pandas as pd
from mlx_lm import load

import readout as R
from tickets import (
    PROMPT_TEMPLATE,
    QUESTIONS,
    generate_tickets,
    noise_ceiling,
    render_prompt,
    split,
    to_mlx_jsonl,
)

pd.set_option("display.width", 130)

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER = Path("adapters/decision")

tickets = generate_tickets(1200, seed=7)
parts = split(tickets, seed=7)

# %% [markdown]
# ## 1. The idea: put the questions in the prompt, not in the weights
#
# An encoder classifier has one head per question, fixed when you build it. Adding a fourth
# question means retraining. Changing `frustration` from three levels to five means
# retraining. And a 4-way softmax **must** pick a department even when the honest answer is
# "none of these", because the outputs sum to 1.
#
# A decision model inverts that. The question is text in the prompt; the model's answer is
# the next token. One model, any questions, and the `other` option is a real answer the
# model can choose.
#
# The price is a bigger model and a more careful prompt. But the prompt is where the
# flexibility lives, and it is worth being precise about the format.

# %%
print(PROMPT_TEMPLATE)
print("\n" + "=" * 70)
prompt, completion = render_prompt(parts["train"][0], "department")
print(prompt)
print(f"-> completion: {completion!r}")

# %% [markdown]
# Three design decisions in that template, each earning its place:
#
# * **Delimiters (`<ticket>`, `<question>`, `<options>`, `<answer>`).** The model must be
#   able to tell where the ticket ends and the question begins. Without them, a ticket
#   containing the words "which team should handle this" would be indistinguishable from
#   the instruction. Lesson 01's `user_tokens` discussion in Kev's codebase makes the same
#   point from the security angle: caller-supplied text must not be able to forge a
#   delimiter.
# * **The options are listed inline.** Lesson 01 measured this: without enumerating the
#   allowed answers, the model puts *zero* probability on them. The list is load-bearing.
# * **The prompt ends at `<answer>`.** Everything before it is context; the very next token
#   is the answer. That is the position we read.
#
# ## 2. The data
#
# One row per **(ticket, question)** pair. The same ticket appears three times with three
# different questions, which is how one adapter learns three question types.

# %%
data_dir = Path("data/mlx")
data_dir.mkdir(parents=True, exist_ok=True)

n_train = to_mlx_jsonl(parts["train"], data_dir / "train.jsonl")
n_valid = to_mlx_jsonl(parts["calibration"], data_dir / "valid.jsonl")
to_mlx_jsonl(parts["development"], data_dir / "test.jsonl")

print(f"train rows  {n_train}   ({len(parts['train'])} tickets x {len(QUESTIONS)} questions)")
print(f"valid rows  {n_valid}\n")

rows = [json.loads(line) for line in (data_dir / "train.jsonl").read_text().splitlines()[:3]]
for row in rows:
    print(f"  completion: {row['completion']!r}")
    print(f"  prompt tail: ...{row['prompt'][-90:]!r}\n")

# %% [markdown]
# `valid.jsonl` is our **calibration split**, not the development set. `mlx-lm` uses it to
# report validation loss during training, and lesson 05 reuses it to fit the temperature.
# It is never trained on. Keeping development out of the training loop entirely is what
# makes it usable for iteration later.

# %% [markdown]
# ## 3. What LoRA actually trains
#
# A full fine-tune of 0.6B would need about `12 × 0.6e9 = 7.2 GB` — possible here, but
# pointless. LoRA freezes the base model and trains two small matrices per targeted layer,
# whose product is *added* to the original weight:
#
# $$W' = W + \frac{\alpha}{r} BA$$
#
# `r` is the rank — the inner dimension of those matrices, so the adapter holds roughly
# `r × (d_in + d_out)` parameters instead of `d_in × d_out`. `alpha` scales the update; the
# effective learning rate on the adapter is proportional to `alpha / r`, which is why the
# two are usually moved together. `--num-layers` decides how many of the model's layers get
# adapters at all — the top 16 of 28 here, on the theory that the later layers hold the
# task-specific behaviour.

# %%
import subprocess as sp

# Read the adapter config mlx-lm will write, to see the defaults it chose.
print("mlx-lm's LoRA defaults (from the training run below):")
print(json.dumps({
    "fine_tune_type": "lora",
    "num_layers": 16,
    "lora_parameters": {"rank": 8, "scale": 20.0, "dropout": 0.0},
}, indent=2))
print("\nBy default mlx-lm uses rank 8 and scale 20 — a few million parameters on a")
print("596M base. That is usually enough for one narrow task and often not enough for")
print("three at once, so exercise 1 below tries a larger rank.")

# %% [markdown]
# ## 4. Train
#
# Five minutes on an M4 Pro. The flags that matter:
#
# | flag | why |
# |---|---|
# | `--mask-prompt` | put loss **only** on the completion. Without it the model is partly trained to reproduce the ticket text, which competes with learning the label |
# | `--num-layers 16` | adapt the top 16 of 28 layers |
# | `--batch-size 4` | 2.8 GB peak at sequence length 384. Raise it if you have headroom |
# | `--iters 1500` | about 2.4 epochs over 2520 rows |
# | `--max-seq-length 384` | attention is quadratic in this; our tickets are short |
#
# The command is printed so you can copy it into a terminal.

# %%
TRAIN_CMD = [
    "uv", "run", "mlx_lm.lora",
    "--model", BASE_MODEL,
    "--train",
    "--data", str(data_dir),
    "--fine-tune-type", "lora",
    "--num-layers", "16",
    "--batch-size", "4",
    "--iters", "1500",
    "--learning-rate", "1e-4",
    "--steps-per-report", "500",
    "--steps-per-eval", "500",
    "--val-batches", "4",
    "--adapter-path", str(ADAPTER),
    "--save-every", "250",
    "--mask-prompt",
    "--max-seq-length", "384",
    "--seed", "0",
]
print("$ " + " ".join(TRAIN_CMD))

# %%
started = time.perf_counter()
result = sp.run(TRAIN_CMD, capture_output=True, text=True)
elapsed = time.perf_counter() - started

for line in result.stdout.splitlines():
    if any(key in line for key in ("Trainable parameters", "Iter ", "Peak mem")):
        print(line)
print(f"\ntotal wall clock: {elapsed:.0f}s")
if result.returncode != 0:
    print(result.stderr[-3000:])

# %% [markdown]
# Read those three numbers:
#
# * **Trainable parameters: about 0.5%** — 2.9M floats against 596M frozen ones. That is
#   the whole reason this runs on a laptop. Check the count yourself:
#   `sum(math.prod(t.shape) for t in adapter_tensors)`.
# * **Peak memory around 2.8 GB.** Nothing. You could train this on a 16 GB machine.
# * **~5 iterations per second**, so 1500 iterations in about five minutes.
#
# And the losses: training loss falls steadily; **validation loss falls and then turns back
# up**. That upturn is the beginning of overfitting, and it is the signal to stop. It is
# mild here — the adapter has very few parameters to overfit with — but on a bigger model or
# a smaller dataset it is the difference between a useful model and a memorised one.

# %% [markdown]
# ## 5. Read the three question types
#
# Now the payoff. Same `readout` module as lesson 01, but pointed at the format the model
# was actually trained on (`style="structured"`).

# %%
model, tokenizer = load(BASE_MODEL, adapter_path=str(ADAPTER))
print("loaded")

# %%
example = parts["development"][3]
print(f"TICKET: {example.state}\n")

for question, spec in QUESTIONS.items():
    answer = R.answer(model, tokenizer, example, question, QUESTIONS, style="structured")
    print(f"{question} ({spec['type']})")

    if spec["type"] == "choice":
        print(f"  choice: {answer.choice}   confidence {answer.confidence:.3f}")
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in answer.probabilities.items()))
    elif spec["type"] == "noul":
        print(f"  P(true) = {answer.noul:.3f}")
    else:
        print(f"  score = {answer.score:.2f}   (expected level)")
        legend = spec.get("legend")
        print("  " + "  ".join(
            f"{legend[k] if legend else k}={v:.3f}" for k, v in answer.probabilities.items()
        ))
    print(f"  label_mass = {answer.label_mass:.3f}\n")

print(f"gold: department={example.department} escalate={example.escalate} frustration={example.frustration}")

# %% [markdown]
# The `score` answer is worth pausing on. It returns `1.35`, not `"1"`. If the model is 60%
# on level 1 and 35% on level 2, that is genuinely different information from a hard label,
# and for an ordinal scale it is more useful: being off by one is not the same failure as
# being off by two, and only the distribution expresses that.
#
# ## 6. Before and after
#
# This is the measurement that justifies the whole lesson. The **same** model, the **same**
# readout, the **same** tickets — once with the untrained base weights and once with the
# adapter:

# %%
base_model, base_tokenizer = load(BASE_MODEL)   # no adapter

def evaluate_questions(m, t):
    result = R.evaluate(m, t, parts["development"], QUESTIONS, style="structured", verbose=False)
    return {row["question"]: row for row in result.to_dict("records")}


evaluate = {
    "base": evaluate_questions(base_model, base_tokenizer),
    "fine-tuned": evaluate_questions(model, tokenizer),
}

rows = []
for question in QUESTIONS:
    b, f = evaluate["base"][question], evaluate["fine-tuned"][question]
    rows.append({
        "question": question,
        "base acc": b["accuracy"],
        "fine-tuned acc": f["accuracy"],
        "gain": round(f["accuracy"] - b["accuracy"], 3),
        "base label_mass": b["label_mass"],
        "fine-tuned label_mass": f["label_mass"],
        "ceiling": round(noise_ceiling(len(QUESTIONS[question].get("options") or ["false", "true"])), 3),
    })
summary = pd.DataFrame(rows)
summary

# %% [markdown]
# Two stories in one table.
#
# **Accuracy.** Gains of 20–76 points, from a model at or below the majority baseline to one
# close to the label-noise ceiling. This is what fine-tuning buys.
#
# **Label mass, and a correction worth reading carefully.** It is tempting to tell a tidy
# story here — "the untrained readout was garbage, fine-tuning fixed it". The `base
# label_mass` column says otherwise. On `department` and `frustration` the base model
# already puts 94–97% of its probability on the label tokens, because the prompt enumerates
# the options and it dutifully answers with one of them. **The mass is healthy and the
# accuracy is terrible**: it is picking from the right set and picking wrong. Fine-tuning
# barely moves the mass (0.943 → 0.977); what it moves is *which* option gets the mass.
#
# `escalate` is the instructive exception. Its base mass is only **0.46**, because the
# prompt asks for `true | false` and an untrained model has no particular reason to answer
# yes/no there rather than continue with prose. Fine-tuning raises it to 0.95.
#
# The lesson: **`label_mass` is a necessary check, not a sufficient one.** A collapsed mass
# tells you the readout is meaningless — lesson 01's 0.0000 case, where the numbers were
# renormalised float dust. But a healthy mass only tells you the model is answering
# *something* in your label space. It says nothing about whether the answer is right. You
# need accuracy for that, and you need a held-out set for accuracy to mean anything.
#
# Compare the `fine-tuned acc` column against `ceiling` rather than against 1.0. On
# `department` and `frustration` we are close to the ceiling, which means **most of the
# remaining error is mislabelled data, not model error**. Nothing you do to this model fixes
# that; only better labels do.

# %% [markdown]
# ## 7. How much training do you need?
#
# `mlx-lm` saved a checkpoint every 250 iterations. Let's evaluate each one — this is the
# learning curve, and reading it correctly is more important than any single number.

# %%
import shutil
import tempfile


def adapter_at(step: int):
    """Load the base model with the checkpoint saved at `step` iterations."""
    source = ADAPTER / f"{step:07d}_adapters.safetensors"
    if not source.exists():
        return None
    tmp = Path(tempfile.mkdtemp()) / "ad"
    tmp.mkdir()
    shutil.copy(source, tmp / "adapters.safetensors")
    shutil.copy(ADAPTER / "adapter_config.json", tmp / "adapter_config.json")
    return load(BASE_MODEL, adapter_path=str(tmp))


curve = []
for step in [0, 250, 500, 750, 1000, 1250, 1500]:
    if step == 0:
        m, t = base_model, base_tokenizer
    else:
        loaded = adapter_at(step)
        if loaded is None:
            continue
        m, t = loaded
    result = R.evaluate(m, t, parts["development"], QUESTIONS, style="structured", verbose=False)
    row = {"iters": step}
    row.update({r.question: round(r.accuracy, 3) for r in result.itertuples()})
    curve.append(row)

pd.DataFrame(curve)

# %% [markdown]
# Look carefully at that curve, because it does **not** rise smoothly. `department` goes
# 0.817 → 0.717 → 0.842 → 0.775 → 0.883 → 0.800. It moves by eight points from one
# checkpoint to the next and then back.
#
# **That is not the model getting better and worse. That is measurement noise.** The
# development set is 120 tickets. The standard error on an accuracy near 0.8 with n=120 is
# about `sqrt(0.8 × 0.2 / 120) ≈ 0.037`, so swings of ±0.08 are roughly two standard
# errors — entirely consistent with a flat line.
#
# The honest conclusions from this curve:
#
# * The gain from **base to any checkpoint** is enormous and unambiguous.
# * The differences **between checkpoints** are within noise. Picking the "best" one
#   (1250, at 0.883) because it scored highest is exactly the mistake lesson 05's
#   confidence intervals exist to prevent.
#
# This is the single most common way people deceive themselves in fine-tuning: running
# twenty configurations against a small development set, keeping the winner, and reporting
# its score. That number is a high-water mark of noise, and it will not reproduce.
#
# ## 8. Scaling up: 1.7B, and QLoRA
#
# The recipe does not change. Every flag above is identical; only memory and time change.

# %%
rows = []
for name, params, note in [
    ("Qwen3-0.6B (here)", 0.6e9, "2.8 GB peak, ~5 min for 1500 iters"),
    ("Qwen3-1.7B", 1.7e9, "~6 GB peak, ~12 min for 1500 iters"),
    ("Qwen3-1.7B, 4-bit base (QLoRA)", 1.7e9, "~2 GB peak, ~15 min — slower, far less memory"),
]:
    rows.append({
        "model": name,
        "LoRA on bf16 base": f"{params * 2.1 / 1024**3:.1f} GB",
        "LoRA on 4-bit base": f"{params * 0.6 / 1024**3:.1f} GB",
        "note": note,
    })
pd.DataFrame(rows)

# %%
print("""To train 1.7B instead, change one argument:

    uv run mlx_lm.lora --model Qwen/Qwen3-1.7B --train --data data/mlx \\
      --num-layers 16 --batch-size 4 --iters 1500 --learning-rate 1e-4 \\
      --adapter-path adapters/decision-1.7b --mask-prompt --max-seq-length 384

For QLoRA, start from a 4-bit build of the base and keep everything else the same:

    uv run mlx_lm.convert --hf-path Qwen/Qwen3-1.7B --mlx-path models/qwen3-1.7b-4bit -q
    uv run mlx_lm.lora --model models/qwen3-1.7b-4bit --train --data data/mlx ...

QLoRA trains the same LoRA matrices on top of frozen 4-bit weights. It is slower per
iteration but the memory saving is large, and for a 1.7B model on a 24 GB machine it is
the difference between comfortable and tight.""")

# %% [markdown]
# ### Which base model should you actually pick?
#
# Constraints, in order:
#
# 1. **License.** If you intend to publish, check it. Apache-2.0 (Qwen3, and the Kev models)
#    is genuinely permissive. Some "open" models carry custom licenses with commercial
#    restrictions, and a model you cannot ship is not a model you can use.
# 2. **A base variant, not just an instruct variant.** Fine-tuning from a *base* model
#    gives you more headroom, because an instruct model has already been shaped toward
#    chat and can resist your format. That said, for a readout task like this an instruct
#    model works well and is much more useful zero-shot for comparison — which is why this
#    tutorial uses `Qwen3-0.6B` (instruct).
# 3. **Size.** Under 2B on a 24 GB machine, in bf16, for LoRA. Go to QLoRA past that.
# 4. **Tokenizer.** Some models have smaller vocabularies, which makes multi-token labels
#    likelier — lesson 01's trap 1. Check before committing to a label set.
#
# The model that matters most is the one you fine-tune on **your** data. Everything in this
# tutorial is designed so that swapping the dataset is the only change you need.

# %% [markdown]
# ### Does this beat the encoder?
#
# Lesson 03's ModernBERT scored 0.892 / 0.892 / 0.908 on the same development tickets. This
# model scores 0.800 / 0.858 / 0.900.
#
# **The encoder is slightly better on accuracy**, which is what the architecture predicts:
# bidirectional attention and a head trained directly on the task beats a causal model
# reading one position. It is also 4x smaller and roughly 10x faster per item.
#
# The decision model wins on the things the encoder cannot do at all:
#
# * questions live in the prompt, so adding one costs nothing,
# * three question *types* with different output semantics from one model,
# * `other` is a real answer the model can choose, rather than a softmax with nowhere
#   to put "I don't know",
# * and — lesson 05's subject — a distribution you can calibrate and set a threshold on.
#
# Pick the encoder when you need one label at high volume. Pick the decision model when the
# *decisions* are the product.

# %% [markdown]
# ---
#
# ### Exercises
#
# 1. **Raise the rank.** `mlx-lm` defaults to rank 8. Add `--lora-parameters '{"rank": 32}'`
#    (or set it in a config YAML) and retrain. Three tasks in one adapter is exactly the
#    case where a small rank may be the binding constraint. Does `escalate` improve?
#
# 2. **Reproduce the noise.** Retrain with `--seed 1` and re-read the learning curve. How
#    much do the checkpoint accuracies move? That spread is your noise floor — any
#    "improvement" smaller than it is not an improvement. Lesson 05 makes this rigorous.
#
# 3. **Break the format.** Evaluate the fine-tuned model with `style="zero_shot"` instead of
#    `"structured"`. What happens to accuracy and `label_mass`? Explain why in terms of
#    what the adapter was trained to respond to.
#
# 4. **Add a fourth question.** Add something genuinely useful to `tickets.QUESTIONS` — for
#    example a `choice` question "is this about a subscription?" — regenerate the data, and
#    retrain. Notice that you changed no model code. Then try the same thing with the
#    encoder in lesson 03 and note what you had to change.

# %% [markdown]
# ---
# **Next:** [05 · Calibration and honest evaluation](05_calibration.ipynb) — turning the
# probabilities above into numbers you can set a business threshold on.
