"""Calibration and honest evaluation for classifiers that output probabilities.

A classifier's softmax output is not a probability in the sense you need. It is a number
between 0 and 1 that ranks options correctly and is almost always **too extreme**: modern
networks say 0.99 when they are right about 90% of the time. If you plan to automate a
decision whenever confidence > 0.95, that gap is the difference between a safe system and
an incident.

This module provides the standard tools, all of them small:

    expected_calibration_error  how far confidence is from accuracy, per bin
    reliability_curve           the same thing as data, for plotting
    fit_temperature             find the single scalar that fixes most of the problem
    brier_score / nll           proper scoring rules (lower is better)
    coverage_at_error           the business number: how much can you automate at X% error
    paired_bootstrap            is model A actually better than model B on this test set

Everything takes probabilities of shape (n, n_classes) and integer labels of shape (n,).
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------
# How wrong is the confidence?
# --------------------------------------------------------------------------------------

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE: the average gap between confidence and accuracy, weighted by bin population.

    For each bin of confidence, compare the mean confidence in that bin with the actual
    accuracy there, then take a population-weighted average of the absolute gaps.

    Two caveats worth knowing before you quote a number:
      * ECE depends on the bin count and on the binning scheme; it is not a single ground
        truth. Report the bin count alongside it.
      * It is a *summary*. A model can have good ECE overall and still be catastrophically
        overconfident in the highest-confidence bin — which is exactly the bin you were
        going to automate on. Always look at the reliability curve too.
    """
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        # right-closed bins, so confidence exactly 1.0 lands in the last bin
        in_bin = (confidence > lo) & (confidence <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
        ece += (count / n) * gap
    return float(ece)


def reliability_curve(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> "object":
    """Per-bin confidence, accuracy and population — the data behind a reliability plot."""
    import pandas as pd

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == labels).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        count = int(in_bin.sum())
        rows.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "count": count,
            "mean confidence": round(float(confidence[in_bin].mean()), 3) if count else None,
            "accuracy": round(float(correct[in_bin].mean()), 3) if count else None,
            "gap": round(float(correct[in_bin].mean() - confidence[in_bin].mean()), 3) if count else None,
        })
    return pd.DataFrame(rows)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between the predicted distribution and the one-hot truth.

    A proper scoring rule: it rewards being both right and honest about it. Range [0, 2]
    for multiclass; lower is better. Unlike accuracy it never saturates, so it can still
    distinguish two models that are both 100% accurate but differently confident.
    """
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Negative log-likelihood of the true label. Lower is better; infinite if a model
    ever assigns zero probability to the truth, which is why clipping matters."""
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.log(p).mean())


# --------------------------------------------------------------------------------------
# Fixing it: temperature scaling
# --------------------------------------------------------------------------------------

def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax over logits / T. T > 1 softens, T < 1 sharpens. Ranking never changes."""
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find the single temperature that minimises NLL on this data.

    Fit this on your **calibration split**, never on the data you are iterating against and
    never on the test set. It is a fitted parameter like any other, and fitting it on the
    set you report makes your calibration number optimistic.

    Because it is one scalar fitted to minimise a smooth objective, it cannot overfit much —
    but "not much" is not "none", and the discipline is free.
    """
    from scipy.optimize import minimize_scalar

    def objective(log_t: float) -> float:
        return nll(apply_temperature(logits, float(np.exp(log_t))), labels)

    result = minimize_scalar(objective, bounds=(np.log(0.01), np.log(100.0)), method="bounded")
    return float(np.exp(result.x))


# --------------------------------------------------------------------------------------
# The business number
# --------------------------------------------------------------------------------------

def coverage_at_error(probs: np.ndarray, labels: np.ndarray, target_error: float = 0.05) -> dict:
    """How much of the workload can be automated at a given error budget?

    This is the metric that actually gets a model deployed, and it is not accuracy. The
    procedure: sort by confidence, accept the most confident predictions first, and find the
    largest set you can accept while the error rate among accepted items stays under budget.

    Returns the threshold to apply in production, and what you get for it.
    """
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = predicted == labels
    order = np.argsort(-confidence)

    best = {"threshold": 1.0, "coverage": 0.0, "error_rate": 0.0, "accuracy": 0.0, "n_accepted": 0}
    running_correct = 0
    for k, idx in enumerate(order, start=1):
        running_correct += int(correct[idx])
        error_rate = 1.0 - running_correct / k
        if error_rate <= target_error:
            best = {
                "threshold": float(confidence[idx]),
                "coverage": k / len(labels),
                "error_rate": error_rate,
                "accuracy": running_correct / k,
                "n_accepted": k,
            }
    return best


def coverage_curve(probs: np.ndarray, labels: np.ndarray, steps: int = 40) -> "object":
    """Coverage vs error rate for every threshold — the trade-off you present to a business."""
    import pandas as pd

    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = predicted == labels
    order = np.argsort(-confidence)

    rows = []
    for frac in np.linspace(0.05, 1.0, steps):
        k = max(1, int(len(labels) * frac))
        accepted = order[:k]
        rows.append({
            "coverage": round(k / len(labels), 3),
            "threshold": round(float(confidence[order[k - 1]]), 3),
            "error rate": round(1.0 - correct[accepted].mean(), 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Is the difference real?
# --------------------------------------------------------------------------------------

def paired_bootstrap(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray,
                     n_resamples: int = 2000, seed: int = 0) -> dict:
    """Confidence interval for (accuracy of A) - (accuracy of B) on the same test set.

    The *paired* part matters. Both models saw the same rows, so the right question is not
    "how do their accuracies vary" but "how does their difference vary" when you resample
    rows. Pairing removes the variance that comes from some rows simply being harder, and
    it is far more sensitive as a result.

    A confidence interval that straddles zero means you have not measured a difference,
    however large the point estimate looks.
    """
    rng = np.random.default_rng(seed)
    correct_a = (preds_a == labels).astype(float)
    correct_b = (preds_b == labels).astype(float)
    observed = correct_a.mean() - correct_b.mean()

    n = len(labels)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        deltas[i] = correct_a[idx].mean() - correct_b[idx].mean()

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "difference": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }
