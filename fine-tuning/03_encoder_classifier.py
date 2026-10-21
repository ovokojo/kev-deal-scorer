# ---
# jupyter:
#   kernelspec:
#     display_name: Fine-tuning tutorial (uv)
#     language: python
#     name: fine-tuning-tutorial
# ---

# %% [markdown]
# # 03 · A conventional classifier
#
# Lessons 01 and 02 built a readout over a language model and then the dataset to train it
# on. Before we fine-tune a decoder, it is worth asking whether a decoder is even the right
# tool. There is a second architecture for this problem, and for a lot of real tasks it wins.
#
# An **encoder** (BERT-family) reads the entire input at once, in both directions, and puts
# a small classification head on top. It is smaller, faster, and usually more accurate on
# classification than a same-sized decoder.
#
# The two architectures differ in one structural way, and everything else follows from it:
#
# | | Encoder (`ModernBERT`, `DeBERTa`) | Decoder (`Qwen`, `Llama`) |
# |---|---|---|
# | Attention | **bidirectional** — every token sees every other | **causal** — tokens only see the past |
# | Built for | understanding a fixed input | generating a continuation |
# | Natural output | one label per head | any sequence, incl. probabilities via readout |
# | Params for a good result | 100M–400M | 600M–2B |
# | Latency per item on M4 | ~1–3 ms | ~20–60 ms |
# | Multiple typed questions | one head per question, fixed at training | any questions you can prompt |
#
# The one-line version: **encoders are better classifiers; decoders are better decision
# models.** If you need one label at high volume, an encoder is usually the right answer.
# If you need several typed questions with calibrated probabilities and an abstain option,
# that is the decoder architecture — lesson 04.
#
# **Time: about 4 minutes**, most of it training.

# %%
import sys

sys.path.insert(0, "lib")
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

import encoder as E
from tickets import (
    LABEL_NOISE,
    generate_tickets,
    noise_ceiling,
    split,
    split_holdout_template,
)

pd.set_option("display.width", 130)


def sh(text: str, n: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


# %% [markdown]
# ## 1. The data, and where the baseline sits
#
# Same splits as before. We will train on `train`, read `development`, and keep the stricter
# template-held-out set for an honest second opinion.

# %%
tickets = generate_tickets(1200, seed=7)
parts = split(tickets, seed=7)
strict = split_holdout_template(tickets, holdout_frac=0.30, seed=7)

print(f"train        {len(parts['train']):5d}")
print(f"calibration  {len(parts['calibration']):5d}")
print(f"development  {len(parts['development']):5d}")
print(f"strict dev   {len(strict['development']):5d}  (no template overlap with train)\n")
print(f"label-noise ceiling: dept {noise_ceiling(4):.3f}  esc {noise_ceiling(2):.3f}  frus {noise_ceiling(3):.3f}")

# %% [markdown]
# ## 2. Before the neural network: the baseline you must beat
#
# This step is skipped constantly and it is the most common way to waste a month of work.
# Routing a support ticket into one of four departments is largely a **keyword** problem.
# A bag-of-words model — TF-IDF features into plain logistic regression — takes seconds to
# train, runs in microseconds, and is embarrassingly hard to beat.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X_train = vectorizer.fit_transform([t.state for t in parts["train"]])
X_dev = vectorizer.transform([t.state for t in parts["development"]])

baseline_rows = []
baseline_models = {}
for task, label_of in [
    ("department", lambda t: t.department),
    ("escalate", lambda t: str(t.escalate).lower()),
    ("frustration", lambda t: str(t.frustration)),
]:
    model = LogisticRegression(max_iter=2000).fit(X_train, [label_of(t) for t in parts["train"]])
    baseline_models[task] = model
    baseline_rows.append({
        "question": task,
        "TF-IDF + logistic regression": round(model.score(X_dev, [label_of(t) for t in parts["development"]]), 3),
    })
baseline_table = pd.DataFrame(baseline_rows)
print(f"vocabulary size: {X_train.shape[1]:,} n-grams from {X_train.shape[0]} tickets\n")
baseline_table

# %% [markdown]
# Remember those numbers. They took about three seconds to produce and no GPU.
#
# The most useful thing you can do here is **train this first and see whether it is good
# enough.** For a large class of real routing and triage problems it is, and the correct
# engineering decision is to ship the logistic regression and spend your time on the labels.
# Everything below has to justify itself against this table.

# %% [markdown]
# ## 3. The model
#
# Now the encoder. Three design choices worth noticing:
#
# **One shared trunk, one head per question.** All three questions are trained jointly on
# the same ticket text. The trunk learns a representation that serves all three, which is
# cheaper than three separate models and usually better, because the tasks share signal —
# an angry ticket about a lost parcel is also an escalation.
#
# **The CLS vector is the summary.** ModernBERT prepends a special `[CLS]` token; after 12
# layers of bidirectional attention its hidden state is a contextual summary of the whole
# ticket, and that is what the heads read.
#
# **The heads start random.** The pretrained checkpoint has no classification head — you
# saw the load report warn about that. Pre-training learned language; the mapping from
# language to *your* labels is what we are about to fit.

# %%
model = E.MultiTaskEncoder()
n_params = sum(p.numel() for p in model.parameters())
n_trunk = sum(p.numel() for p in model.backbone.parameters())
print(f"trunk (ModernBERT-base)  {n_trunk/1e6:8.1f} M parameters")
print(f"heads (3 linear layers)  {(n_params - n_trunk)/1e6:8.2f} M parameters")
print(f"total                    {n_params/1e6:8.1f} M parameters")
print(f"\nfull fine-tune memory ~= 12 bytes/param = {n_params * 12 / 1024**3:.2f} GB")

# %% [markdown]
# At 150M parameters this fits comfortably, so we fine-tune the **whole** model rather than
# using LoRA. LoRA becomes the right call past roughly 1B parameters, or when you need many
# task-specific variants sharing one frozen base. Section 7 shows that version.

# %%
model, tokenizer, history = E.train(parts["train"], epochs=4, batch_size=16, lr=3e-5)

# %% [markdown]
# Roughly 90 seconds on an M4 Pro for four epochs over 840 tickets. Compare that with what a
# 2B decoder would cost you, and you can see why the encoder deserves its reputation.

# %% [markdown]
# ## 4. How did it do?

# %%
print("development set (shares templates with train):")
dev_report = E.evaluate(model, tokenizer, parts["development"])

# %%
print("strict development set (no template overlap):")
strict_report = E.evaluate(model, tokenizer, strict["development"])

# %%
rows = []
for task, n_classes in [("department", 4), ("escalate", 2), ("frustration", 3)]:
    rows.append({
        "question": task,
        "encoder acc": round(dev_report[task]["accuracy"], 3),
        "TF-IDF acc": float(baseline_table.loc[baseline_table["question"] == task, "TF-IDF + logistic regression"].iloc[0]),
        "ceiling": round(noise_ceiling(n_classes), 3),
        "conf|right": round(dev_report[task]["mean_conf_right"], 3),
        "conf|wrong": round(dev_report[task]["mean_conf_wrong"], 3),
    })
comparison = pd.DataFrame(rows)
comparison

# %% [markdown]
# **Read that table honestly.** The fine-tuned 150M transformer ties a three-second
# logistic regression on `department` and `frustration`, and beats it by about 2.5 points
# on `escalate` — the one task that needs a phrase rather than a keyword ("reported this to
# my bank", "speak to a manager").
#
# That is not a failure of the encoder. It is the finding: for a task where the signal is
# largely lexical, a linear model over n-grams captures most of it. The encoder's advantage
# shows up exactly where you would predict — on the task that requires reading a phrase in
# context rather than spotting a word. It would show up much more strongly on a task needing
# negation or word order, such as distinguishing "I have received a refund" from "I have not
# received a refund", which a bag of words cannot represent at all.
#
# So why fine-tune at all? Three honest reasons:
#
# 1. **You cannot know in advance which regime you are in.** The reason to run the baseline
#    is to find out. Here it ties; on your data it may not.
# 2. **The encoder generalises further.** It is less brittle to unseen phrasings, which
#    matters as your traffic drifts.
# 3. **The decoder architecture gives you something the encoder structurally cannot** —
#    arbitrary questions, typed answers, calibrated distributions, and an abstain option.
#    That is lesson 04.
#
# Also compare against the ceiling column. For `department` we are at 0.892 against a
# ceiling of 0.925, so most of the remaining error is **mislabelled data, not model error**.
# For `escalate` there is more headroom, which matches what we designed: escalation needs
# the model to recognise behaviours like "reported this to my bank" in a polite sentence.

# %% [markdown]
# ### Where it goes wrong

# %%
pairs = list(zip(dev_report["department"]["preds"], dev_report["department"]["gold"]))
classes = dev_report["department"]["classes"]
confusion = pd.DataFrame(
    [[pairs.count((p, g)) for g in range(len(classes))] for p in range(len(classes))],
    index=[f"pred {c}" for c in classes],
    columns=[f"gold {c}" for c in classes],
)
confusion

# %%
probs = dev_report["department"]["probs"]
confidence = probs.max(axis=1)
wrong = dev_report["department"]["preds"] != dev_report["department"]["gold"]
order = np.argsort(-confidence)

print("the most confidently WRONG tickets — read these, not the loss curve:\n")
shown = 0
for i in order:
    if not wrong[i]:
        continue
    ticket = parts["development"][i]
    pred = classes[dev_report["department"]["preds"][i]]
    gold = classes[dev_report["department"]["gold"][i]]
    print(f"  conf={confidence[i]:.3f}  predicted {pred}, labelled {gold}")
    print(f"    {sh(ticket.state, 100)}\n")
    shown += 1
    if shown == 5:
        break

# %% [markdown]
# This listing — wrong answers sorted by confidence — is the highest-value artifact a
# training run produces, and it is what Kev writes to `errors.jsonl` on every run. The
# pattern to look for is **confident mistakes**: a model that is wrong and unsure is
# telling you the question is hard; a model that is wrong and certain is usually telling you
# the label is wrong, or that the tie-break rule is being applied inconsistently.
#
# In our case some of these are just the 10% injected label noise. If you go looking, you
# can catch them: a ticket labelled `shipping` that never mentions delivery or tracking is
# a flipped label, and no model will learn it, because it is not true.

# %% [markdown]
# ### The leakage check
#
# In lesson 02 we found 100% template overlap between train and development, and 22% exact
# duplicates. The expectation was that this inflates the held-out score. Let's see whether
# it did — the strict set has no template overlap at all:

# %%
rows = []
for task in ["department", "escalate", "frustration"]:
    rows.append({
        "question": task,
        "dev (leaky)": round(dev_report[task]["accuracy"], 3),
        "strict dev (held-out templates)": round(strict_report[task]["accuracy"], 3),
        "difference": round(strict_report[task]["accuracy"] - dev_report[task]["accuracy"], 3),
    })
pd.DataFrame(rows)

# %% [markdown]
# **The expected collapse did not happen.** The strict numbers are as high as the leaky ones,
# sometimes higher.
#
# That is a result, and a more useful one than a collapse would have been. It tells us the
# model is reading the *content* of the ticket rather than memorising sentence patterns —
# which makes sense, because the signal really is in the words. Had the strict set dropped
# by 20 points, the opposite conclusion would follow, and we would need to go back to the
# data before trusting anything.
#
# The lesson is not "leakage does not matter". It is that **you cannot predict the size of
# this effect, in either direction, without measuring it.** Every dataset is different, and
# the measurement costs you one extra split.

# %% [markdown]
# ## 5. Saving the model
#
# We save it here because lesson 05 needs these probabilities, and the logits behind them, to calibrate.

# %%
path = E.save(model, tokenizer, "models/encoder")
print(f"saved to {path}")

# %% [markdown]
# ## 6. What the encoder cannot do
#
# Everything above works, and it is a dead end for one reason: **the questions are baked
# into the weights.** There are three heads, fixed at construction time, each predicting
# labels fixed at training time.
#
# Want to add a fourth question — "is this about a subscription?" — you retrain. Want to
# change `frustration` from three levels to five? Retrain. Want the model to say "this
# ticket does not fit any of my labels"? There is no such output: a 4-way softmax always
# sums to 1, so the model must pick a department even when the honest answer is "I do not
# know." You can bolt on a confidence threshold, but the model has no native way to abstain.
#
# Those are exactly the properties a Jev-style decision model has, and they come from the
# decoder architecture: the questions are *text in the prompt*, not weights in a head.

# %% [markdown]
# ## 7. If your encoder is bigger: LoRA
#
# At 150M parameters, full fine-tuning is cheap and there is no reason to do anything else.
# If your best encoder is a 1B+ model, or you need one frozen base with many swappable
# task variants, switch to LoRA. The training loop is identical; only the parameter set
# that receives gradients changes:

# %%
# The LoRA version, for when the trunk is too big to fine-tune outright.
LORA_SNIPPET = '''
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                    # rank: how much capacity the adapter has
    lora_alpha=32,           # scaling; the effective step size is alpha/r
    target_modules=["Wqkv", "Wo"],   # ModernBERT attention projections
    lora_dropout=0.05,
    bias="none",
    task_type="FEATURE_EXTRACTION",
)
model.backbone = get_peft_model(model.backbone, config)
model.backbone.print_trainable_parameters()
# ...then the same training loop as above, unchanged
'''
print(LORA_SNIPPET)

# %% [markdown]
# Note the `task_type`: we attach LoRA to the **backbone** and keep our own heads fully
# trainable, because the heads are ours and start random anyway. `FEATURE_EXTRACTION` is
# what tells `peft` that the wrapped model returns hidden states rather than logits.
#
# With `r=16` targeting the attention projections you typically train 0.5–2% of parameters
# and get within a point or two of full fine-tuning, at a fraction of the memory. That is
# the same trade lesson 04 makes for the decoder.

# %% [markdown]
# ---
#
# ### Exercises
#
# 1. **Beat the baseline.** The logistic regression uses `ngram_range=(1,2)`. Try `(1,3)`
#    and tune `C`. Can you push it past the encoder? (This is a real research result, not a
#    trick question — for keyword-ish tasks, tuned linear models are extremely strong.)
#
# 2. **Find a task where the encoder wins.** Construct a test that needs word order, such as
#    "I have received a refund" vs "I have not received a refund." Train both models on
#    such data. Which copes, and why? This is the clearest demonstration of what
#    bidirectional attention buys you.
#
# 3. **Break the abstain.** Feed the encoder a ticket that belongs in no department, e.g.
#    "I'd like to apply for a job in your warehouse." Look at the softmax. It sums to 1 by
#    construction, so the model must pick something. What confidence does it report? Why is
#    this dangerous if you automate on a confidence threshold?
#
# 4. **Vary the seed.** Retrain with `seed=1` and compare. How much does development
#    accuracy move? That movement is your noise floor — improvements smaller than it are
#    not real. Lesson 05 makes this rigorous with a bootstrap.

# %% [markdown]
# ---
# **Next:** [04 · The decision model](04_decision_model.ipynb) — fine-tuning the readout from
# lesson 01, so that one model answers all three questions.
