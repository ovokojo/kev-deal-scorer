# Deal-scoring benchmark

Measured on this M4 Pro / 24 GB. Regenerate with `scripts/bench.py report`.

## How to read this table

* `demo` is the price-ratio rule that *generated* the labels, not a model. It scores
  1.000 because it is the answer key. Read it as the ceiling, never as a baseline.
* The storefront catalog's labels come from that same rule. So `laya-ft` scoring 25/25
  means "applied the learned rule to 25 hand-written listings from unseen products".
  It is not independent ground truth. A human-labelled set is the missing piece.
* `laya-ft` = `models/laya-deals`, built by `scripts/finetune_laya.py` (2 epochs, 6,146 s).
* `laya-zs` = `aac6fef/laya-typed-decisions-mlx` as shipped, and `kev-0.8b` on MPS.
* Latency is per listing, for the ~500-token states this project sends with 3 questions.
  The upstream Laya benchmark reports 13 ms for much shorter single-question states;
  that number does not transfer here.

## Synthetic test split (400 listings, 8 held-out products)

| engine | verdict acc | macro-F1 | ECE | scam AUROC | scam recall @0.5 | false flags | p50 / p95 ms (1 listing) | batched ms / listing |
|---|---|---|---|---|---|---|---|---|
| demo | 1.000 | 1.000 | 0.174 | 1.000 | 1.00 | 0 | 0 / 0 | — |
| kev-0.8b | 0.448 | 0.274 | 0.057 | 0.960 | 0.00 | 0 | 706 / 717 | — |
| laya-ft | 0.945 | 0.941 | 0.023 | 0.999 | 1.00 | 3 | 102 / 103 | 92.5 |
| laya-zs | 0.338 | 0.239 | 0.119 | 0.946 | 0.00 | 0 | 106 / 109 | 94.0 |

Paired bootstrap of verdict accuracy against `kev-0.8b` (10,000 resamples):

- `demo` − `kev-0.8b`: +0.552 (95% CI +0.502 to +0.600)
- `laya-ft` − `kev-0.8b`: +0.497 (95% CI +0.445 to +0.550)
- `laya-zs` − `kev-0.8b`: -0.110 (95% CI -0.162 to -0.060)

## Storefront catalog (25 hand-written listings, never trained on)

| engine | verdict acc | macro-F1 | ECE | scam AUROC | scam recall @0.5 | false flags | p50 / p95 ms (1 listing) | batched ms / listing |
|---|---|---|---|---|---|---|---|---|
| demo | 1.000 | 1.000 | 0.214 | 1.000 | 1.00 | 0 | 0 / 0 | — |
| kev-0.8b | 0.480 | 0.284 | 0.105 | 1.000 | 0.00 | 0 | 705 / 715 | — |
| laya-ft | 1.000 | 1.000 | 0.048 | 1.000 | 1.00 | 0 | 102 / 110 | 101.2 |
| laya-zs | 0.520 | 0.339 | 0.051 | 1.000 | 0.00 | 0 | 106 / 120 | 106.3 |

## The two failures this project started with

Kev 0.8B, the model the original demo shipped with, does not answer the scam question.
Scam recall at the 0.5 threshold is 0.00 on both splits, while its red-flag *ranking* is
good (AUROC 0.96). The signal is there but the operating point is wrong, and its verdict
accuracy of 0.448 is close to the majority class. Zero-shot Laya is no better (0.338) and
also never fires the scam question, which matches its own docs: the base checkpoint is near
chance on typed decisions, and fine-tuning is where the value is.

After fine-tuning, both failures are gone: 0.945 verdict accuracy on held-out products
with an honestly fitted temperature (ECE 0.023), and every scam recalled with 3 false
flags in 400.

## The gate

* Laya zero-shot as the default engine: **failed** (0.338 vs Kev's 0.448), so it was fine-tuned.
* `laya-ft` as the default engine: accuracy 0.945 vs 0.448, paired bootstrap +0.497
  (95% CI +0.445 to +0.550), scam recall 1.00 vs 0.00. **Passed on accuracy and recall.**
* It **misses the 50 ms latency target** at ~100 ms per listing, batched or not.
  Batching removes HTTP round trips, not per-listing compute: the encoder pass over a
  ~500-token state with 3 questions is what costs 100 ms. That is still 7x faster than
  Kev and fast enough to score a screen of cards while the shopper reads them, so Laya
  is the default in `run.sh` with the miss stated rather than hidden.
