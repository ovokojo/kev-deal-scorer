# ---
# jupyter:
#   kernelspec:
#     display_name: Fine-tuning tutorial (uv)
#     language: python
#     name: fine-tuning-tutorial
# ---

# %% [markdown]
# # 01 · From text to probability
#
# A language model appears to write text. Underneath, it does one thing: given a sequence
# of tokens, it produces a score for every token in its vocabulary. Everything else —
# chat, code, reasoning — is built on that.
#
# This lesson shows that raw mechanism and then uses it to get something a chat model
# cannot give you: **a probability distribution over labels you choose, in one forward
# pass, with no text generated at all.**
#
# That is the core trick behind Jev, and behind Kev. By the end of this notebook you will
# have implemented it yourself, measured it honestly, and seen exactly why it is not
# enough on its own.
#
# **Time: about 5 minutes.**

# %%
import sys

sys.path.insert(0, "lib")

import mlx.core as mx
import numpy as np

from mlx_lm import load
from tickets import QUESTIONS, generate_tickets, split
import readout as R

model, tokenizer = load("Qwen/Qwen3-0.6B")

# %% [markdown]
# ## 1. Tokens are the unit of everything
#
# The model never sees characters. Text is chopped into **tokens** by a tokenizer, and
# every token has an integer id. The vocabulary is typically 100k–200k entries.
#
# This is not trivia. Three of the traps in this lesson come from tokenization, and the
# first one bites the moment you pick label names carelessly.

# %%
ticket = split(generate_tickets(1200, seed=7), seed=7)["development"][0]
print(f"TEXT\n{ticket.state}\n")

ids = tokenizer.encode(ticket.state)
print(f"{len(ticket.state)} characters -> {len(ids)} tokens\n")
for i in ids[:14]:
    print(f"  {i:>7}  {tokenizer.decode([i])!r}")
print("  ...")

# %% [markdown]
# ### Trap 1: your labels must be exactly one token
#
# Watch what happens with a perfectly reasonable set of label names.

# %%
print("How many tokens is each label?\n")
for label in ["returns", "shipping", "billing", "other", "true", "false", "0", "1", "2",
              "calm", "annoyed", "angry"]:
    ids = tokenizer.encode(label)
    pieces = [tokenizer.decode([i]) for i in ids]
    flag = "ok" if len(ids) == 1 else "<-- NOT ONE TOKEN"
    print(f"  {label!r:10s} {len(ids)} token(s)  {pieces}  {flag}")

# %% [markdown]
# `calm`, `annoyed` and `angry` are **multi-token**. If you read "the last token of the
# label", you are reading a fragment — for `annoyed` you would read the piece that
# completes the word, at a position where the word has not started. The number you get
# back is meaningless, and it will look plausible.
#
# This is precisely why Kev's API takes numeric levels (`0`, `1`, `2`) for score questions
# and carries the words in a separate `legend` field. The words are for humans. The model
# only ever sees the digits.
#
# Our dataset follows that convention. `lib/readout.py` refuses to run on a multi-token
# label rather than silently returning nonsense:

# %%
try:
    R.label_token_ids(tokenizer, ["calm", "annoyed", "angry"])
except R.LabelTokenError as e:
    print("refused, as intended:\n")
    print(f"  {e}")

print("\nwith the numeric convention:")
print(f"  {R.label_token_ids(tokenizer, ['0', '1', '2'])}")

# %% [markdown]
# ### Trap 2: leading whitespace is a different token
#
# `"angry"` and `" angry"` are different ids. A model that has just written a space puts
# its probability on the space-prefixed variant. Score the wrong one and you get
# near-zero mass and a distribution made of rounding error.

# %%
for label in ["angry", " angry", "Angry", " Angry"]:
    ids = tokenizer.encode(label)
    print(f"  {label!r:9s} -> {ids}   {[tokenizer.decode([i]) for i in ids]}")

# %% [markdown]
# ## 2. One forward pass produces a score for every token
#
# Now the mechanism. Feed tokens in, get **logits** out: one raw, unbounded score per
# position, per vocabulary entry.
#
# Let's start with the laziest possible prompt, to see what the model does when we *don't*
# tell it what we want.

# %%
lazy_prompt = f"Ticket: {ticket.state}\nDepartment:"
ids = mx.array(tokenizer.encode(lazy_prompt))[None]
logits = model(ids)
print(f"input ids shape   {ids.shape}")
print(f"logits shape      {logits.shape}   (batch, sequence, vocabulary)")

last = logits[0, -1, :]
print(f"\ntop 5 tokens after 'Department:':")
for i in mx.argsort(last)[::-1][:5].tolist():
    print(f"  {last[i].item():7.3f}   {tokenizer.decode([i])!r}")

# %% [markdown]
# A newline and `the` — generic continuations. The model has no idea we wanted a
# department name, because we never said so.
#
# ### Logits are not probabilities
#
# Logits are unbounded reals. To treat them as beliefs, pass them through a **softmax**,
# which exponentiates and normalises so they sum to 1:
#
# $$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
#
# `T` is **temperature**. It does not change the ranking, only the sharpness: `T=1` is the
# model's native distribution, `T>1` flattens it, `T<1` sharpens it. Lesson 05 fits `T` on
# held-out data, and that one number is often the difference between a model you can
# automate on and one you cannot.

# %%
z = np.array([2.0, 1.0, 0.1])
for T in [0.5, 1.0, 2.0, 10.0]:
    e = np.exp(z / T)
    p = e / e.sum()
    print(f"  T={T:>4}   " + "   ".join(f"{v:.3f}" for v in p))

# %% [markdown]
# Same ranking every time, very different confidence. Keep this in mind: **a model's
# stated confidence is a property of the temperature as much as of the model.**

# %% [markdown]
# ## 3. The readout
#
# Here is the whole technique, and it is about ten lines:
#
# 1. Build a prompt that ends exactly where the answer goes.
# 2. One forward pass. Take the logits at the final position.
# 3. Keep only the entries for the tokens that spell your labels.
# 4. Softmax over *those*, renormalising so they sum to 1.
#
# First, the prompt. We state the question **and enumerate the allowed answers** — the
# next section shows why that second part is not optional.

# %%
ASK = R.zero_shot_ask("department", QUESTIONS["department"])
print(ASK)

prompt = R.build_prompt(tokenizer, ticket.state, ASK)
print("\n" + "=" * 70 + "\n")
print(prompt)
print("-" * 70)

# %%
ids = mx.array(tokenizer.encode(prompt))[None]
last = model(ids)[0, -1, :]

print("top 5 tokens now:")
for i in mx.argsort(last)[::-1][:5].tolist():
    print(f"  {last[i].item():7.3f}   {tokenizer.decode([i])!r}")

# %% [markdown]
# The model is now proposing **department names**, because we told it those were the
# options. Now do steps 3 and 4 by hand, so nothing is hidden:

# %%
LABELS = ["returns", "shipping", "billing", "other"]
label_ids = R.label_token_ids(tokenizer, LABELS)

probs_all = mx.softmax(last, axis=-1)
raw = {label: probs_all[i].item() for label, i in label_ids.items()}
mass = sum(raw.values())
normalised = {k: v / mass for k, v in raw.items()}

print(f"{'label':10s} {'raw p':>10s} {'renormalised':>14s}")
for label in LABELS:
    print(f"{label:10s} {raw[label]:10.6f} {normalised[label]:14.3f}")
print(f"\nlabel_mass (share of total probability on our labels): {mass:.3f}")
print(f"\nprediction: {max(normalised, key=normalised.get)}")
print(f"gold:       {ticket.department}")

# %% [markdown]
# Read that `label_mass` carefully — it is your sanity check. The next section shows what
# happens when it collapses to zero and the renormalised numbers become confident-looking
# garbage. **A readout whose mass you have not checked is not a measurement.**

# %% [markdown]
# ### Trap 3: the prompt decides whether this works at all
#
# Four versions of the *same* question, measured on the *same* ticket. The number is the
# share of the model's total probability that lands on our four label tokens:

# %%
def mass_for(prompt_text: str) -> float:
    ids_ = mx.array(tokenizer.encode(prompt_text))[None]
    lp = mx.softmax(model(ids_)[0, -1, :], axis=-1)
    return sum(lp[i].item() for i in label_ids.values())

variants = {
    'raw "Department:" only': f"Ticket: {ticket.state}\nDepartment:",
    "raw + the question": f"Ticket: {ticket.state}\n\nWhich team should handle this ticket?",
    "raw + question + Answer:": f"Ticket: {ticket.state}\n\n{ASK}\nAnswer:",
    "chat template, question only": R.build_prompt(
        tokenizer, ticket.state, "Which team should handle this ticket?"
    ),
    "chat template + question + options": R.build_prompt(tokenizer, ticket.state, ASK),
}
for name, text in variants.items():
    print(f"  label_mass {mass_for(text):8.4f}   {name}")

# %% [markdown]
# Two independent things have to be true, and neither alone is enough:
#
# * **The chat template** puts the model in the instruction-following regime.
# * **Enumerating the allowed answers in the prompt** is what actually moves probability
#   onto your label tokens. Stating the question without listing the options leaves the
#   mass at zero — the model is still happily considering every word in its vocabulary.
#
# The failure is silent. Renormalising `[0.0000001, 0.0000003, 0.0000002]` gives you
# `[0.17, 0.50, 0.33]` — a confident-looking distribution built out of nothing. This is
# why the readout returns `label_mass` alongside the probabilities and why you should look
# at it every single time.

# %% [markdown]
# ## 4. So how good is this, actually?
#
# We now have a working classifier with no training at all. Let's measure it properly on
# the development split — 120 tickets the model has never been tuned on.

# %%
import time

dev = split(generate_tickets(1200, seed=7), seed=7)["development"]
print(f"{len(dev)} development tickets\n")

t0 = time.perf_counter()
baseline = R.evaluate(model, tokenizer, dev, QUESTIONS)
elapsed = time.perf_counter() - t0

# The majority-class baseline: what you would score by always answering with the most
# common label. Never report an accuracy without this number next to it — on a skewed
# label distribution it can be well above 50%, and it is the bar that matters.
import collections

MAJORITY = {}
for question, spec in QUESTIONS.items():
    labels = spec.get("options") or ["true", "false"]
    counts = collections.Counter(
        R.gold(t, question, spec) for t in dev
    )
    MAJORITY[question] = counts.most_common(1)[0][1] / len(dev)

baseline.insert(4, "majority_class", baseline["question"].map(MAJORITY).round(3))
baseline.insert(5, "beats_majority?", baseline["accuracy"] > baseline["majority_class"])
baseline

# %%
print(f"scored {len(dev)} tickets x {len(QUESTIONS)} questions in {elapsed:.1f}s "
      f"({len(dev) / elapsed:.0f} tickets/s, no generation)")

# %% [markdown]
# This is the honest starting line, and it is not flattering. Compare each accuracy against
# its `majority_class` column above:
#
# * **`department`** — only a few points above the majority baseline. Barely better than
#   saying "returns" to every ticket.
# * **`escalate`** — **below** its majority baseline. Worse than always answering "false".
# * **`frustration`** — also **below** its majority baseline, i.e. worse than a constant.
#
# And look at the confidence columns. `conf|wrong` is about as high as `conf|right`, and
# `department` is confidently wrong about 80% of the time. **The model does not know when
# it does not know.**
#
# None of this is a bug in our code. It is the actual state of an off-the-shelf small model
# on a task with *your* label conventions:
#
# 1. It has never been told that "other" is the right bin for a wholesale enquiry, or that
#    a ticket mentioning both a late parcel and a double charge is a `billing` ticket.
# 2. It has no idea what you mean by the level `2`. We chose that convention; the model
#    would have to be taught it.
# 3. Nothing has aligned its confidence with its accuracy, because nothing has measured it.
#
# ### The three things fine-tuning will change
#
# | Problem now | Fixed by |
# |---|---|
# | Model does not know your labels or your tie-break rules | Lesson 04 — LoRA on your data |
# | No single model answers three question types at once | Lesson 04 — one model, three questions |
# | Confidence is meaningless, so you cannot automate | Lesson 05 — temperature scaling |
#
# A fair caveat before you conclude that small models are bad: **better prompting and a
# bigger model would both improve these numbers**, and that is a legitimate alternative to
# fine-tuning. The point is not that zero-shot is worthless — it is that you cannot know
# which is better without measuring, and now you have the measurement.
#
# ---
#
# ### Exercises
#
# 1. **Make the readout fail deliberately.** Call `R.readout` with the labels
#    `["calm", "annoyed", "angry"]` and see what `lib/readout.py` does. Then find a prompt
#    where `label_mass` falls below 0.01 while the renormalised probabilities still look
#    decisive. Why is that dangerous in production?
#
# 2. **Improve the zero-shot prompt.** `zero_shot_ask` in `lib/readout.py` builds the
#    phrasing used above. Rewrite it to add a one-line description of each option (e.g.
#    "shipping = delivery status, delays, lost packages"). Re-run section 4. How much of
#    the gap was prompt quality rather than model capability?
#
# 3. **Break the label mass.** Score a label that is a multi-token word and print
#    `label_mass`. Then prefix it with a space and try again. Explain the difference in
#    terms of what the model has just emitted.
#
# 4. **Cost of the readout.** We scored 120 tickets in a few seconds on a 0.6B model. If
#    your support queue were 50,000 tickets a day, how long would that take, and how much
#    would the same volume cost through a hosted frontier model?

# %% [markdown]
# ---
# **Next:** [02 · Building the dataset](02_build_a_dataset.ipynb) — where accuracy actually
# comes from, and where the labels, the splits and the label noise come from.
