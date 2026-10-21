# ---
# jupyter:
#   kernelspec:
#     display_name: Fine-tuning tutorial (uv)
#     language: python
#     name: fine-tuning-tutorial
# ---

# %% [markdown]
# # 05 · Calibration and honest evaluation
#
# You now have two models that both claim to be about 89% accurate on `department`:
# a ModernBERT encoder (lesson 03) and a fine-tuned Qwen3-0.6B decision model (lesson 04).
# Both emit probabilities. The question this lesson answers is whether those probabilities
# **mean anything**.
#
# Often they do not. A softmax output is a ranking device: it is good at putting the right
# answer first, and its sense of *how sure* it should be is a separate question — modern
# networks are routinely described as saying 0.99 when they are right about 90% of the time.
# If your plan is "auto-route anything above 0.95 confidence", that gap is the difference
# between a working system and an incident report.
#
# That is the textbook claim. **This lesson measures it rather than repeating it**, and what
# we find is more interesting than the textbook — including the case where the standard fix
# does not help.
#
# This lesson is short and it is the one that gets a model deployed, because it produces
# the only number a business can act on: **how much of the workload can we automate at an
# acceptable error rate?**
#
# **Time: about 10 minutes.**

# %%
import sys

sys.path.insert(0, "lib")
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import calibrate as C
import encoder as E
import readout as R
from tickets import QUESTIONS, generate_tickets, split

pd.set_option("display.width", 130)

tickets = generate_tickets(1200, seed=7)
parts = split(tickets, seed=7)
print(f"calibration split: {len(parts['calibration'])} tickets   "
      f"development split: {len(parts['development'])} tickets")

# %% [markdown]
# ## 1. Load both models
#
# The decision model comes from lesson 04's adapter; the encoder from lesson 03.

# %%
from mlx_lm import load

decision_model, decision_tok = load("Qwen/Qwen3-0.6B", adapter_path="adapters/decision")
encoder_model, encoder_tok = E.load("models/encoder")

print("decision model: Qwen3-0.6B + LoRA")
print("encoder:        ModernBERT-base, fine-tuned")

# %% [markdown]
# ## 2. Are the probabilities honest?
#
# Two measurements. The **reliability curve** buckets predictions by confidence and
# compares each bucket's mean confidence with its actual accuracy — a perfectly calibrated
# model lies on the diagonal. **ECE** collapses that curve into one number: the
# population-weighted average gap.
#
# We do this for both models on `department`, using the *development* split — the data we
# are allowed to look at.

# %%
def probs_for_question(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


decision_dev_logits = {
    q: R.collect_logits(decision_model, decision_tok, parts["development"], q, QUESTIONS)
    for q in QUESTIONS
}
decision_gold = {q: R.gold_indices(parts["development"], q, QUESTIONS) for q in QUESTIONS}

encoder_report = E.evaluate(encoder_model, encoder_tok, parts["development"], verbose=False)

rows = []
for question in QUESTIONS:
    dp = probs_for_question(decision_dev_logits[question])
    dy = decision_gold[question]
    ep = encoder_report[question]["probs"]
    ey = encoder_report[question]["gold"]
    rows.append({
        "question": question,
        "decision acc": round(float((dp.argmax(1) == dy).mean()), 3),
        "decision ECE": round(C.expected_calibration_error(dp, dy), 3),
        "encoder acc": round(float((ep.argmax(1) == ey).mean()), 3),
        "encoder ECE": round(C.expected_calibration_error(ep, ey), 3),
    })
pd.DataFrame(rows)

# %% [markdown]
# ECE around 0.05–0.10 means the model's confidence is off by five to ten points on
# average. It says "90% sure" and is right about 80–85% of the time. That is not
# catastrophic — and for a model trained with cross-entropy it is expected, because
# cross-entropy rewards honest probabilities — but it is more than enough to break a
# threshold you picked by intuition.
#
# Let's look at the shape of the error rather than just its average:

# %%
dp = probs_for_question(decision_dev_logits["department"])
C.reliability_curve(dp, decision_gold["department"], n_bins=8)

# %% [markdown]
# Read the `gap` column: negative means overconfident (confidence above accuracy),
# positive means underconfident. The bins with the most traffic are the ones that matter,
# because they are where your decisions will actually be taken.
#
# ## 3. Temperature scaling
#
# The fix is embarrassingly simple and it is the standard one. Instead of
# `softmax(logits)`, use `softmax(logits / T)` for a single scalar `T`. It cannot change
# which option wins — dividing by a positive number preserves order — so **accuracy is
# untouched**. It only moves probability mass around.
#
# `T` is fitted by minimising NLL on the **calibration split**, which is why lesson 02
# insisted on a fourth split. Fitting it on development would make our reported calibration
# optimistically biased, exactly like tuning hyperparameters on a test set.

# %%
temperatures, calibrated_rows = {}, []
for question in QUESTIONS:
    cal_logits = R.collect_logits(decision_model, decision_tok, parts["calibration"], question, QUESTIONS)
    cal_gold = R.gold_indices(parts["calibration"], question, QUESTIONS)

    T = C.fit_temperature(cal_logits, cal_gold)
    temperatures[question] = T

    raw = probs_for_question(decision_dev_logits[question])
    scaled = C.apply_temperature(decision_dev_logits[question], T)
    y = decision_gold[question]

    calibrated_rows.append({
        "question": question,
        "T": round(T, 3),
        "accuracy": round(float((scaled.argmax(1) == y).mean()), 3),
        "ECE before": round(C.expected_calibration_error(raw, y), 3),
        "ECE after": round(C.expected_calibration_error(scaled, y), 3),
        "NLL before": round(C.nll(raw, y), 3),
        "NLL after": round(C.nll(scaled, y), 3),
    })

calibration = pd.DataFrame(calibrated_rows)
calibration

# %% [markdown]
# **Read this table sceptically, because it does not say what the textbooks imply.**
#
# 1. **Accuracy is identical before and after.** It has to be — dividing logits by a
#    positive constant cannot reorder them. Any tutorial showing accuracy change from
#    temperature scaling is showing you a bug.
# 2. **On this data, temperature scaling made calibration *worse* on two of three
#    questions — and made NLL worse too**, even though NLL is precisely what `T` was fitted
#    to minimise.
#
# That second point deserves a proper look, because it is not a bug in the code. Let's
# measure where the temperature was fitted against where we are testing it:

# %%
print("Where does the fitted temperature actually help?\n")
for question in QUESTIONS:
    cal_logits = R.collect_logits(decision_model, decision_tok, parts["calibration"], question, QUESTIONS)
    cal_gold = R.gold_indices(parts["calibration"], question, QUESTIONS)
    T = temperatures[question]

    def nll_at(logits, y, t):
        return C.nll(C.apply_temperature(logits, t), y)

    cal_before, cal_after = nll_at(cal_logits, cal_gold, 1.0), nll_at(cal_logits, cal_gold, T)
    dev_before = nll_at(decision_dev_logits[question], decision_gold[question], 1.0)
    dev_after = nll_at(decision_dev_logits[question], decision_gold[question], T)

    print(f"{question:12s} T={T:.3f}")
    print(f"   calibration split (T was fitted here):  NLL {cal_before:.4f} -> {cal_after:.4f}  "
          f"{'improved' if cal_after < cal_before else 'worse'}")
    print(f"   development split (we measure here):     NLL {dev_before:.4f} -> {dev_after:.4f}  "
          f"{'improved' if dev_after < dev_before else 'worse'}")

# %% [markdown]
# The optimizer is doing its job: **on the split it was fitted to, NLL improves every
# time.** The problem is that this is a *different split* from the one we measure on.
#
# The mechanism is visible in the accuracies. Our calibration split happens to be a
# **harder sample** than development — about 0.75 against 0.80 on `department`. A model that
# is less accurate ought to be less confident, so the optimizer asks for a softer
# distribution (`T = 1.43`); development, being easier, would prefer something nearer 1.2.
# The fitted value is right for the data it saw, and slightly wrong for the data it did not.
#
# Three things to take from this:
#
# 1. **Calibration parameters are fitted parameters.** A few hundred calibration rows is
#    marginal; a few thousand is comfortable. Fit and *validate* on different splits, or
#    you are reporting the fit's optimism rather than its transfer.
# 2. **A model trained with cross-entropy is often already close to calibrated.** Softmax
#    outputs are overconfident in general, but this model's ECE was 0.06–0.12 before any
#    correction. There was little for a single scalar to fix, and the noise in estimating it
#    was larger than the error it removes.
# 3. **Where temperature scaling earns its reputation is on models that are *badly*
#    miscalibrated** — one trained on very few examples, or one whose training pushed
#    confidence up without pushing accuracy with it. There `T` comes out well above 1 and
#    the improvement is large. One measurement tells you which regime you are in.
#
# So the honest summary is not "temperature scaling fixes confidence". It is: **temperature
# scaling is one fitted scalar, it is not free, and you must measure whether it helped.**

# %%
# The reliability curve after scaling, for the question that improved least.
question = "department"
scaled = C.apply_temperature(decision_dev_logits[question], temperatures[question])
print(f"{question}: T = {temperatures[question]:.3f}\n")
print("before:")
print(C.reliability_curve(probs_for_question(decision_dev_logits[question]), decision_gold[question], n_bins=6).to_string(index=False))
print("\nafter:")
print(C.reliability_curve(scaled, decision_gold[question], n_bins=6).to_string(index=False))

# %% [markdown]
# ## 4. The number that gets a model deployed
#
# Accuracy is not actionable. "89% accurate" does not tell anyone what to do. The
# actionable framing is **selective prediction**: the model may abstain, a human handles
# what it declines, and you ask how much of the queue can be cleared automatically while
# keeping the error rate among *accepted* items under a budget.
#
# That is what `coverage_at_error` computes: sort by confidence, accept the most confident
# first, and find the largest set you can accept while staying under budget.

# %%
rows = []
for question in QUESTIONS:
    scaled = C.apply_temperature(decision_dev_logits[question], temperatures[question])
    y = decision_gold[question]
    for budget in [0.02, 0.05, 0.10]:
        cov = C.coverage_at_error(scaled, y, target_error=budget)
        rows.append({
            "question": question,
            "error budget": f"{budget:.0%}",
            "coverage": f"{cov['coverage']:.1%}",
            "threshold to apply": round(cov["threshold"], 3),
            "actual error": f"{cov['error_rate']:.1%}",
        })
pd.DataFrame(rows)

# %% [markdown]
# This is the sentence you take to the business: *"At a 5% error budget we can auto-route
# X% of tickets, by acting on every prediction the model is at least 0.87 confident about."*
# That is a decision someone can make. "89% accurate" is not.
#
# Two warnings:
#
# * **These numbers come from 120 development tickets, and from a temperature we just showed
#   does not transfer well.** A coverage estimate from 120 rows is noisy, and the threshold
#   it implies is noisier still. On a real workload you would use a few thousand rows for
#   both, and re-check that the fitted temperature actually helps the split you are
#   reporting on.
# * **The error budget must come from the business, not from the model.** A wrong refund
#   routing and a wrong tone-of-voice score do not have the same cost. One model, several
#   budgets, several thresholds.

# %%
C.coverage_curve(
    C.apply_temperature(decision_dev_logits["escalate"], temperatures["escalate"]),
    decision_gold["escalate"],
    steps=12,
)

# %% [markdown]
# ## 5. Is the difference between the two models real?
#
# Lesson 03 left an open question: the encoder and the decision model have similar accuracy.
# Is the encoder actually better, or is the gap smaller than the noise in a 120-row
# evaluation?
#
# The right tool is a **paired bootstrap**. Resample the development rows with replacement,
# recompute the difference in accuracy each time, and look at the spread. Pairing matters:
# both models saw the same rows, so we resample rows rather than models.

# %%
rows = []
for question in QUESTIONS:
    decision_pred = C.apply_temperature(
        decision_dev_logits[question], temperatures[question]
    ).argmax(axis=1)
    encoder_pred = encoder_report[question]["probs"].argmax(axis=1)
    y = decision_gold[question]

    # align the encoder's label order to the readout's
    assert list(encoder_report[question]["classes"]) == list(
        QUESTIONS[question].get("options") or ["false", "true"]
    ), question

    result = C.paired_bootstrap(encoder_pred, decision_pred, y)
    rows.append({
        "question": question,
        "encoder - decision": round(result["difference"], 3),
        "95% CI": f"[{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]",
        "real difference?": "yes" if result["significant"] else "no — CI spans zero",
    })
pd.DataFrame(rows)

# %% [markdown]
# **A confidence interval that spans zero means you have not measured a difference**, no
# matter how large the point estimate looks. This is the discipline that stops you from
# shipping a change because it gained 0.4 points on a 120-row evaluation.
#
# Kev's tooling bakes the same idea in from the other direction: its `plan_size.py` takes a
# target effect ("detect a +5 point gain at 80% power") and tells you how many labelled
# records you need before running anything. Its author's measured result on a real workload
# is worth remembering: 400 records gave **+0.6 points with a ±6 point interval** — a
# measurement that cannot distinguish a real gain from nothing — while 1050 records gave
# **+5.9 points with a [+2.3, +9.7] interval**, which can. Fine-tuning gains are usually
# real but modest, and you need enough data to see them at all.

# %% [markdown]
# ## 6. Three ways to fool yourself
#
# **Fit the temperature on the data you report.** The most common. If you tune `T` on
# development and then quote ECE on development, your calibration number is optimistic by
# however much the fit absorbed. Use the calibration split.
#
# **Quote ECE without its bin count.** ECE depends on binning. With two bins it is almost
# always small; with 50 bins it is almost always large. Report the bin count, and show the
# reliability curve so the reader can see the shape.
#
# **Assume calibration transfers.** `T` is fitted to a distribution. Change the traffic —
# a new product line, a different customer segment, tickets in a new language — and the
# calibration is fitted to the old one. This is why the notebook keeps both the leaky and
# the template-held-out development sets: they are two different distributions, and you can
# watch the same `T` behave differently on each. Re-fit calibration whenever the input
# distribution moves, and monitor ECE in production where you can.

# %% [markdown]
# ---
#
# ### Exercises
#
# 1. **Break the temperature.** Fit `T` on `development` instead of `calibration` and
#    compare the resulting ECE on `development`. How much optimism did you buy? Repeat
#    a few times with different splits to see the size of the effect.
#
# 2. **Vary the bins.** Compute ECE for the same predictions with 5, 15 and 40 bins. How
#    much does the number move? Now you know why a bare ECE figure is not comparable across
#    papers unless the binning matches.
#
# 3. **Calibrate the encoder too.** The encoder is already reasonably calibrated (its
#    cross-entropy training saw to that). Fit a `T` anyway and compare the ECE improvement
#    against the decision model's. Which one needed help, and does that match your
#    expectation about which model was overfitting?
#
# 4. **Cost-weighted thresholds.** Rework section 4 so a wrong `escalate=true` costs 10x a
#    wrong `escalate=false` (a human review is expensive). What error budget does that
#    imply, and what coverage do you get? This is the analysis that turns a model into a
#    decision.

# %% [markdown]
# ---
# **Next:** [06 · Serving it](06_serving.ipynb) — putting this behind an HTTP endpoint.
