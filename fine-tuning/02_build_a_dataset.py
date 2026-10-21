# ---
# jupyter:
#   kernelspec:
#     display_name: Fine-tuning tutorial (uv)
#     language: python
#     name: fine-tuning-tutorial
# ---

# %% [markdown]
# # 02 · Building the dataset
#
# Lesson 01 ended with a model that scored at or below the majority-class baseline. This
# lesson is about the thing that actually fixes that, and it is not the model.
#
# **The dataset is the product.** In every practical fine-tuning project, the difference
# between a useful classifier and a useless one is the quality of a few hundred labelled
# rows: whether the label space is well defined, whether the splits are honest, and whether
# the labels agree with each other. Model choice and hyperparameters are rounding errors
# next to those.
#
# This lesson has no training in it. It is the part you should not skip.
#
# **Time: about 10 minutes.**

# %%
import sys

sys.path.insert(0, "lib")

import collections

import pandas as pd

from tickets import (
    QUESTIONS,
    SPLIT_FRACTIONS,
    generate_tickets,
    render_prompt,
    split,
    split_holdout_template,
    template_registry,
    to_mlx_jsonl,
    to_records_jsonl,
)

pd.set_option("display.width", 120)

# %% [markdown]
# ## 1. The task, and the shape of the data
#
# We are building a triage model for an e-commerce support queue. For every incoming
# ticket it answers three questions of three different types — the same three types a Jev
# or Kev model answers:

# %%
for name, spec in QUESTIONS.items():
    print(f"{name:12s} type={spec['type']:7s} {spec['instructions']}")
    if spec.get("options"):
        print(f"{'':12s} options={spec['options']}")
    if spec.get("legend"):
        print(f"{'':12s} legend={spec['legend']}")
    print()

# %% [markdown]
# Those three types are worth naming, because each has different statistics and different
# failure modes:
#
# * **`choice`** — pick one of N named options. Reported as a distribution over options.
# * **`noul`** ("no-ul" = yes/no) — a single probability. Reported as `P(true)`.
# * **`score`** — an *ordinal* level. Reported as a distribution over levels **and** an
#   expected value, because "60% level 1, 40% level 2" should not be flattened to "2".
#
# ### Where this data comes from
#
# In this tutorial, `lib/tickets.py` generates it from sentence templates. That makes the
# whole thing reproducible and offline. **In your real project you will not do this** —
# section 7 covers the three real sources. Read the generator anyway: it is a worked
# example of the two design decisions that matter, which are the label space and the
# label noise.

# %%
tickets = generate_tickets(1200, seed=7)
print(f"{len(tickets)} tickets\n")
for t in tickets[:5]:
    print(f"[{t.department:8s} escalate={str(t.escalate):5s} frustration={t.frustration}]")
    print(f"  {t.state}\n")

# %%
print("the same ticket, rendered for each of the three questions:\n")
example = tickets[0]
for question in QUESTIONS:
    prompt, completion = render_prompt(example, question)
    print(f"--- {question} ---")
    print(prompt[-200:].replace(example.state, "<ticket text>"))
    print(f"  -> {completion!r}\n")

# %% [markdown]
# ## 2. Designing the label space
#
# This is the highest-leverage hour in the project, and it is a product decision, not a
# machine-learning one. Three rules:
#
# **Mutually exclusive.** Every ticket gets exactly one label. If two people on your team
# would disagree about a ticket, the label space is underspecified, not the model.
#
# **Exhaustive.** There is always a "none of the above". We have `other`, and it is not
# a dumping ground — it is the *abstain* signal. A model that confidently answers `other`
# is telling you it should not be automated, which lesson 05 turns into a business metric.
#
# **A documented tie-break rule.** Our generator deliberately produces tickets that mention
# two things at once:

# %%
multi = [t for t in tickets if "Please send an invoice" in t.state or "I want to return the other" in t.state][:2]
for t in multi:
    print(f"label = {t.department}")
    print(f"  {t.state}\n")

# %% [markdown]
# That first ticket is about a late delivery *and* asks for an invoice. We labelled it
# `shipping`, because our tie-break rule is **"the label follows the primary ask — the
# first complaint, not every subject mentioned."**
#
# Write that rule down. Put it in the prompt you give your labellers, and put it in the
# prompt you give an LLM if you use one. A large share of the "model errors" you will
# diagnose later are really labelling disagreements, and no amount of LoRA fixes those.

# %%
dist = collections.Counter(t.department for t in tickets)
frust = collections.Counter(t.frustration for t in tickets)
esc = collections.Counter(t.escalate for t in tickets)

print("department distribution")
for k, v in dist.most_common():
    print(f"  {k:10s} {v:5d}  {v/len(tickets):6.1%}   <- always predicting this scores {v/len(tickets):.1%}")
print("\nfrustration distribution")
for k in sorted(frust):
    print(f"  level {k}   {frust[k]:5d}  {frust[k]/len(tickets):6.1%}   <- majority class scores {max(frust.values())/len(tickets):.1%}")
print(f"\nescalate: true {esc[True]} ({esc[True]/len(tickets):.1%})  false {esc[False]} ({esc[False]/len(tickets):.1%})  "
      f"<- majority class scores {max(esc.values())/len(tickets):.1%}")

# %% [markdown]
# **Always compute these baselines first.** In lesson 01 the zero-shot model scored *below*
# the always-say-false baseline on `escalate`. Without the baseline number in front of you,
# 0.48 looks like a working model. With it, you know it is worse than doing nothing.
#
# Note also how *uneven* these are. That is deliberate and realistic — real queues are
# skewed, and a skewed label distribution is why **accuracy is a bad headline metric**.
# A model that predicts the majority class every time gets a good score on `escalate` while
# being completely useless. Lesson 05 introduces the metrics that do not have this hole.

# %% [markdown]
# ## 3. Label noise sets your ceiling
#
# Our generator deliberately mislabels 10% of tickets, because real ticketing data always
# does — people misclick, copy-paste the wrong row, or simply disagree about a borderline
# case. On a task like this, two competent humans reading the same ticket choose differently
# maybe 8–15% of the time.
#
# This has a consequence that surprises people: **you cannot train your way past label
# noise.** If 10% of your labels are wrong, a perfect model that has learned the true
# underlying function still disagrees with your labels. The best accuracy you can ever
# measure is bounded:

# %%
from tickets import LABEL_NOISE, noise_ceiling

print(f"label noise dialled into the generator: {LABEL_NOISE:.0%}\n")
rows = []
for name, n_classes in [("department", 4), ("escalate", 2), ("frustration", 3)]:
    rows.append({
        "question": name,
        "classes": n_classes,
        "best achievable accuracy": f"{noise_ceiling(n_classes):.3f}",
    })
pd.DataFrame(rows)

# %%
# Sanity-check the formula by simulation: a "perfect" model that always predicts the true
# label, measured against a label file corrupted at that rate.
import random

rng = random.Random(0)
n = 20000
rows = []
for noise in [0.0, 0.02, 0.05, 0.10, 0.20]:
    truth = [rng.randrange(4) for _ in range(n)]
    corrupted = [rng.randrange(4) if rng.random() < noise else t for t in truth]
    rows.append({
        "label noise": f"{noise:.0%}",
        "simulated agreement": f"{sum(a == b for a, b in zip(truth, corrupted)) / n:.3f}",
        "formula": f"{noise_ceiling(4, noise):.3f}",
    })
pd.DataFrame(rows)

# %% [markdown]
# Read that table before you celebrate or despair over any score in this tutorial. A perfect
# model measures about 0.93 against a 10%-noisy 4-class label file. So when lesson 03 shows
# an encoder at 0.88, the correct reading is **"about five points of headroom"**, not
# "terrible" — and when you see 0.99 somewhere, the correct reading is **"that measurement
# is broken, because it is above the ceiling."**
#
# This is why the single most valuable response to a disappointing result is to **read the
# errors**, not to reach for a bigger model. If the "errors" are all cases where you
# disagree with your own labels, you have found the bug, and it is in the data.
#
# Kev's evaluation workflow makes this a first-class step: every run writes an
# `errors.jsonl` with the wrong answers sorted by confidence, so the most confident
# mistakes — usually the mislabels — are at the top.

# %% [markdown]
# ## 4. Four splits, not two
#
# Everyone knows train and test. For a model that outputs *probabilities you intend to
# automate on*, you need four, and mixing them up is the most common way to fool yourself.
#
# | Split | Share | What it is for | Who may look at it |
# |---|---|---|---|
# | `train` | 70% | the only data the optimizer sees | the optimizer |
# | `calibration` | 15% | fitting the temperature (lesson 05) | the calibration procedure |
# | `development` | 10% | iterating: read errors, change data, retrain | **you**, as often as you like |
# | `test` | 5% | the final number you report | **you, once** |
#
# The rule that matters: **the calibration split exists so that you never fit the
# temperature on the data you are iterating against.** If you tune the temperature on
# `development`, your reported calibration is optimistic, exactly like tuning
# hyperparameters on a test set.
#
# And the `test` split is consumed by looking at it. Read it once, at the end. If you go
# back and change something, it is no longer a test set and you no longer have an honest
# number.

# %%
parts = split(tickets, seed=7)
for name, rows_ in parts.items():
    frac = SPLIT_FRACTIONS[name]
    print(f"  {name:12s} {len(rows_):5d} rows   (target {frac:.0%})")

# %% [markdown]
# ### The trap: leakage
#
# Now the most important measurement in this lesson. Our tickets are built from 60 sentence
# templates, each reused about 20 times. A *random* split therefore puts near-identical
# tickets on both sides of the boundary:

# %%
train_templates = {t.template_id for t in parts["train"]}
train_texts = {t.state for t in parts["train"]}

rows = []
for name in ["calibration", "development", "test"]:
    held = parts[name]
    same_template = sum(1 for t in held if t.template_id in train_templates) / len(held)
    exact_dup = sum(1 for t in held if t.state in train_texts) / len(held)
    rows.append({
        "split": name,
        "rows": len(held),
        "shares a template with train": f"{same_template:.0%}",
        "exact duplicate of a train row": f"{exact_dup:.0%}",
    })
pd.DataFrame(rows)

# %% [markdown]
# **Every single development ticket shares a sentence pattern with training, and 22% are
# character-for-character duplicates of a training row.**
#
# This is not a quirk of synthetic data. It is the normal state of real datasets. Support
# tickets come in from a template-driven help widget; scraped text has boilerplate; LLM-
# generated training data reuses phrasings. When this happens, your held-out score measures
# **memorisation, not generalisation**, and it will be badly optimistic about production.
#
# The fix is to split on the thing that repeats — the template, the customer, the source,
# the date — not on rows:

# %%
strict = split_holdout_template(tickets, holdout_frac=0.30, seed=7)

rows = []
for name in ["train", "calibration", "development", "test"]:
    rows.append({"split": name, "rows": len(strict[name])})
pd.DataFrame(rows)

# %%
strict_train_templates = {t.template_id for t in strict["train"]}
strict_train_texts = {t.state for t in strict["train"]}
held = strict["development"]
print(f"strict development set: {len(held)} tickets")
print(f"  shares a template with train:    {sum(1 for t in held if t.template_id in strict_train_templates) / len(held):.0%}")
print(f"  exact duplicate of a train row:  {sum(1 for t in held if t.state in strict_train_texts) / len(held):.0%}")

# %% [markdown]
# Zero on both. A model scoring well on *this* set has generalised to new phrasings, which
# is the only claim that survives contact with production.
#
# **How to use both.** Train on the normal split (more data, and duplicates act as a mild
# regulariser). Report the strict number. If the two diverge a lot, you learned something
# important: your model is pattern-matching surface forms rather than the task. In lesson
# 05 we will measure exactly that gap.
#
# In a real project the discipline is the same, and the thing you split on is whatever
# repeats:
#
# * tickets from the same **customer** → split by customer
# * data collected by the same **agent** → split by agent
# * anything time-ordered → split by **date**, never randomly

# %% [markdown]
# ## 5. Where labels really come from
#
# Three sources, best first.
#
# **1. Existing labels.** Your ticketing system already routes tickets to departments. Your
# CRM already has escalation flags. Export a few thousand rows, join on the ticket id, done.
# This is the best source by a wide margin: the labels reflect what your business actually
# does, including its quirks.
#
# The catch: *historical labels encode historical policy*, not ground truth. If your team
# systematically over-escalates, you will train a model to over-escalate. That is usually
# fine — you are automating an existing process — but decide deliberately.
#
# **2. An LLM as a labelling function.** Fast, cheap, and how most teams bootstrap. Give it
# the ticket and the label space, ask for one label, and validate a sample by hand.
#
# The trap is severe and worth stating plainly: **if you generate your training data with
# a teacher model and then evaluate on data generated by the same teacher, you are
# measuring agreement with the teacher, not correctness.** You will get a beautiful score
# and learn nothing about your real queue. Always keep a slice of *real, human-labelled*
# tickets for evaluation, even if it is only 100 rows. That slice is your ground truth.
#
# Here is the shape of the prompt you would send. Note that it includes the tie-break rule
# from section 2 — that is what makes the labels consistent:

# %%
LABELLING_PROMPT = """You are labelling support tickets for a triage model.

Ticket:
{ticket}

Answer with exactly one label per line, in this format:
department: one of returns, shipping, billing, other
escalate: true or false
frustration: 0, 1, or 2

Rules:
- The label follows the PRIMARY ask (the first complaint), not every subject mentioned.
- "other" is for anything that is not a return, a delivery problem, or a payment problem.
- escalate=true only if a human is needed now: threats of chargeback, fraud, legal
  action, or a customer who says they have already been ignored more than once.
- frustration: 0 = calm and factual, 1 = annoyed or repeating themselves, 2 = angry,
  threatening, or all-caps."""

print(LABELLING_PROMPT.format(ticket=multi[0].state))

# %% [markdown]
# **3. Hand-labelling.** Slow, and the only way to get a trustworthy evaluation set. Budget
# for it. Two hours of a domain expert labelling 200 tickets is worth more than any amount
# of prompt tuning, because it is the only measurement that cannot be gamed by the model
# you are trying to measure.

# %% [markdown]
# ## 6. Writing the files
#
# Two formats, matching the two layers of this tutorial.
#
# **Prompt/completion JSONL** is what `mlx-lm` wants. One row per (ticket, question) pair,
# so 1200 tickets become 3600 rows. The `prompt` is everything up to where the answer
# starts; `completion` is the label. Training with `--mask-prompt` puts loss only on the
# completion, so the model is graded on the answer and not on reproducing the ticket.

# %%
from pathlib import Path

DATA = Path("data")
DATA.mkdir(exist_ok=True)

n_rows = to_mlx_jsonl(parts["train"], DATA / "train.jsonl")
to_mlx_jsonl(parts["calibration"], DATA / "calibration.jsonl")
to_mlx_jsonl(parts["development"], DATA / "development.jsonl")
print(f"data/train.jsonl        {n_rows} rows ({len(parts['train'])} tickets x {len(QUESTIONS)} questions)")

import json

first = json.loads((DATA / "train.jsonl").read_text().splitlines()[0])
print(f"\nprompt:\n{first['prompt']}")
print(f"completion: {first['completion']!r}")

# %% [markdown]
# **Jev/Kev-style records** keep one row per ticket with all questions together. This is the
# shape the System One API takes and the shape lesson 06's serving layer uses.

# %%
to_records_jsonl(parts["development"], DATA / "development_records.jsonl")
rec = json.loads((DATA / "development_records.jsonl").read_text().splitlines()[0])
print(json.dumps(rec, indent=2))

# %% [markdown]
# ## 7. How much data do you actually need?
#
# The honest answer is "measure it", and the way to measure it is a **learning curve**:
# train on 25%, 50%, 75%, 100% of your training set and plot development accuracy against
# size. Lesson 04 does this for real. What to look for:
#
# * **Still climbing at 100%** → more data will help. Collect it.
# * **Flat well before 100%** → you are data-limited by *label noise or label definition*,
#   not by volume. More rows will not help; better labels will.
# * **Huge gap between 25% and 100%** → you are in the steep part of the curve, and every
#   extra hundred rows is worth more than any hyperparameter change.
#
# For a rough prior, Kev's own tooling makes this concrete. Its `plan_size.py` answers
# "how many records do I need to detect a +5 point accuracy gain at 80% power?" and lands
# around **1000 records for three questions per record**. Its author's measured experience
# on a real workload: 400 records gave +0.6 points with a ±6 point confidence interval —
# i.e. nothing — while 1050 records gave +5.9 points with a CI of [+2.3, +9.7].
#
# Note what that implies: **below a few hundred records you often cannot tell whether
# fine-tuning helped at all.** The gain is real, but it is smaller than the noise in your
# measurement. Plan for ~1000 labelled examples, and if you only have 200, expect to
# measure a lot of noise before you measure a signal.
#
# ---
#
# ### Exercises
#
# 1. **Find the mislabels.** The generator flips 2% of department labels. Write a few lines
#    to find tickets whose label disagrees with the other signals — for example, label
#    `shipping` on a ticket that never mentions delivery or tracking. Those are the rows a
#    model will learn to get wrong.
#
# 2. **Break the split.** Change `split(..., seed=...)` to a different value and re-run
#    section 4. Does the exact-duplicate rate change much? What does that tell you about
#    relying on a single random seed for your held-out estimate?
#
# 3. **Design a stricter split for your own domain.** Template-held-out is right for this
#    synthetic data. What repeats in *your* data — customers, agents, time, source system?
#    Sketch how you would split on it.
#
# 4. **Write your tie-break rule.** Take five tickets from your own work that you would find
#    genuinely hard to label. Write the rule that resolves each one. That rule is the most
#    important artifact in the project, more than any model.

# %% [markdown]
# ---
# **Next:** [03 · A conventional classifier](03_encoder_classifier.ipynb) — the other way to
# solve this task, with an encoder instead of a decoder. It is faster, smaller, and often
# more accurate, and you should know when to reach for it.
