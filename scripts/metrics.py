"""Metrics for scripts/bench.py. ECE and the paired bootstrap are copied from the
fine-tuning tutorial (fine-tuning/lib/calibrate.py) so this repo stays
self-contained; the bootstrap here takes per-row correctness instead of predictions."""
import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Population-weighted gap between top-class confidence and accuracy, 15 equal bins."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.sum():
            ece += (in_bin.sum() / n) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def auroc(scores: np.ndarray, positives: np.ndarray) -> float:
    """Probability a random positive outranks a random negative (ties count half)."""
    pos, neg = scores[positives == 1], scores[positives == 0]
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def paired_bootstrap(correct_a: np.ndarray, correct_b: np.ndarray,
                     n_resamples: int = 10000, seed: int = 0) -> dict:
    """95% CI for accuracy(A) − accuracy(B) on the same rows. A CI spanning zero means no
    measured difference, however large the point estimate looks."""
    a, b = correct_a.astype(float), correct_b.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_resamples, len(a)))
    deltas = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"diff": float(a.mean() - b.mean()), "lo": float(lo), "hi": float(hi)}
