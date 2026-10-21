# Fine-tuning small models on a Mac

A hands-on tutorial for building your own small, open-source **decision model**: a model
that reads a piece of text and answers typed questions about it — which bin does this
belong in, yes or no, how much — with **calibrated probabilities** you can set a business
threshold on.

Everything here runs locally on an M4 Mac with 24 GB of memory, using models of 2B
parameters or fewer. No API keys, no cloud GPU, no Modal account.

The running example is an e-commerce support queue: for every incoming ticket, decide
which team should handle it, whether it needs a human right now, and how frustrated the
customer is. You will build that end to end, twice, with two different architectures —
and measure honestly which one is actually better.

---

## What you will be able to do at the end

* Estimate, before you start, whether a model will fit and how long training will take.
* Turn a language model into a classifier **without generating any text**, and know the
  three traps that silently produce garbage numbers.
* Build a labelled dataset: design a label space, write a tie-break rule, split it four
  ways, and measure how much your labels — not your model — are limiting you.
* Fine-tune with LoRA on Apple Silicon, and know when an encoder is the better answer.
* Calibrate a model so its confidence means something, and report the number your
  business actually cares about: *coverage at an error budget*.
* Serve the result as a local HTTP endpoint, and package it for other people to run.

## Who this is for

You should be comfortable with Python and the command line, and you should know roughly
what a train/test split and overfitting are. You do **not** need any prior PyTorch or
Hugging Face experience — the notebooks build that up. If you took a machine learning
course at some point and have since worked mostly as an engineer, that is exactly the
right background.

---

## How to run it

```bash
cd fine-tuning
uv sync                       # installs everything into .venv
uv run jupyter lab            # open the notebooks
```

Then open `00_setup.ipynb` and pick the **"Fine-tuning tutorial (uv)"** kernel.

Each lesson builds on the previous one and later notebooks load models that earlier ones
trained, so run them in order the first time.

`uv sync` pulls MLX (Apple's array framework), PyTorch, `transformers`, `peft`,
`scikit-learn` and Jupyter — a few GB. The models download on first use; `Qwen3-0.6B` is
about 1.5 GB and `ModernBERT-base` about 600 MB.

**Notebooks are generated from `.py` files.** Each `NN_name.py` is the readable source and
`NN_name.ipynb` is the same thing as an executable notebook, produced with
[jupytext](https://jupytext.readthedocs.io). Edit whichever you prefer:

```bash
uv run jupytext --to notebook --output 01_logits_to_probabilities.ipynb 01_logits_to_probabilities.py
```

Notebook outputs are committed, so you can read them on GitHub without running anything.

---

## The lessons

| | Lesson | What it covers | Time |
|---|---|---|---|
| **00** | [Setup and the memory arithmetic](00_setup.ipynb) | Hardware check, how much memory a model costs, first local generation | 5 min |
| **01** | [From text to probability](01_logits_to_probabilities.ipynb) | Tokens, logits, softmax, and the readout that turns a language model into a classifier. The three traps. A zero-shot baseline that is worse than guessing | 5 min |
| **02** | [Building the dataset](02_build_a_dataset.ipynb) | Label space design, tie-break rules, **label noise and the accuracy ceiling**, the four splits, leakage, where labels really come from | 10 min |
| **03** | [A conventional classifier](03_encoder_classifier.ipynb) | ModernBERT with one head per question, MPS training, and the TF-IDF baseline that ties it | 5 min |
| **04** | [The decision model](04_decision_model.ipynb) | LoRA fine-tuning of a causal LM on MLX. Three question types — `choice`, `noul`, `score` — from one model, one forward pass | 15 min |
| **05** | [Calibration and honest evaluation](05_calibration.ipynb) | Reliability curves, ECE, temperature scaling, coverage at an error budget, paired bootstrap | 10 min |
| **06** | [Serving it](06_serving.ipynb) | A local HTTP endpoint, the System One request shape, latency and batching | 10 min |
| **07** | [Packaging and hosting](07_packaging.ipynb) | Fusing the adapter, quantising, GGUF and Ollama, publishing to Hugging Face | 10 min |
| — | [Appendix](APPENDIX.md) | PyTorch/MPS + PEFT path, glossary, further reading, troubleshooting | — |

### The five ideas worth keeping

If you remember nothing else from this tutorial, remember these. Every one of them is
measured in a notebook rather than asserted.

1. **A language model is a probability distribution over tokens, and nothing else.**
   Classification is reading that distribution at the right position over the right
   tokens. No generation required — one forward pass per question.

2. **The prompt format is part of the model.** Measured in lesson 01: the *same* question,
   on the *same* ticket, put anywhere from 0.0000 to 0.9807 of its probability on the
   label tokens depending on how the prompt was built. Change the prompt after training
   and you have a different, worse model.

3. **Your labels set the ceiling, and the ceiling is lower than you think.** Two competent
   humans disagree on 8–15% of support tickets. A perfect model measured against that
   label file cannot score above about 0.93 on a 4-class task. Most "model failures" are
   label failures.

4. **Accuracy is not the number that gets a model deployed.** Calibrated probabilities plus
   a confidence threshold give you *coverage at an error budget* — measured in lesson 05,
   "act above 0.59 confidence and you clear 67% of routing at a 4.9% error rate" — which is
   a sentence you can take to a business. Note the same model gives only 10% coverage at the
   same budget on `escalate`, because that question is genuinely harder: one model, several
   thresholds.

5. **Fine-tuning is often not the highest-leverage thing you can do.** In lesson 03 a
   three-second TF-IDF logistic regression ties a fine-tuned transformer. Run the baseline
   first; the tutorial shows you how to find out rather than guess.

### Two architectures, and when to use which

The tutorial builds the same task both ways, so the comparison is concrete rather than
theoretical:

| | Encoder classifier (lesson 03) | Decision model (lesson 04) |
|---|---|---|
| Model | ModernBERT-base, 149M | Qwen3-0.6B, 596M |
| Attention | bidirectional | causal |
| Questions | fixed heads, set at training time | written in the prompt, any time |
| Output | one label per head | distributions over each question, plus `score` as an expected value |
| Abstain | impossible — the softmax must pick something | the `other` bin is a first-class answer |
| Fine-tuning on M4 | full fine-tune, ~90 s | LoRA, ~5 min |
| Peak memory | 1.7 GB | 2.8 GB |

Neither wins outright. **Encoders are better classifiers; decoders are better decision
models.** If you need one label at high volume, reach for the encoder. If you need several
typed questions with calibrated probabilities and an honest "I don't know", that is what
the decoder architecture is for.

---

## Hardware: what to expect on an M4 Pro / 24 GB

Measured on this machine, and the numbers the notebooks compare against:

| | 0.6B | 1.7B |
|---|---|---|
| Generation, MLX bf16 | ~150 tok/s | ~50 tok/s |
| LoRA training step (batch 4, seq 384) | ~5 it/s, 2.8 GB peak | ~2 it/s, ~4 GB peak |
| Full fine-tune | not needed | ~19 GB — tight, use LoRA |
| Readout (classification, no generation) | 15–50 tickets/s | 5–20 tickets/s |

The rule that governs everything: **full fine-tuning costs about 12 bytes per parameter;
LoRA on a quantised base costs about 0.5.** That 24x difference is why this works on a
laptop at all.

---

## What this is not

* **Not a leaderboard.** The dataset is synthetic, generated by `lib/tickets.py`, with
  deliberate label noise. Absolute numbers here mean nothing. The *methods* — the readout,
  the splits, the ceiling, the calibration — transfer directly.
* **Not a guide to training a foundation model.** We only ever adapt a pretrained one.
* **Not the Kev architecture.** Kev packs all questions into one forward pass with a
  block-causal mask and a pointer readout over option boundaries; this tutorial uses one
  forward pass per question and reads the label token. That is simpler, slightly slower,
  and much easier to reason about. If you want the real thing, read the
  [Kev repository](https://github.com/jaredpalmer/kev) — and note that its fine-tuning
  path targets a Modal H100, not a laptop.

---

## If something breaks

| Symptom | Cause |
|---|---|
| `label_mass = 0.000` and confident predictions | Your labels are not single tokens, or the prompt does not enumerate the answers. Lesson 01, trap 3 |
| Accuracy far above the ceiling in lesson 02 | Your held-out split leaks. Split on the thing that repeats |
| `RuntimeError: MPS backend out of memory` | Reduce `--batch-size` or `--max-seq-length`. On MPS, a model that exceeds memory can also *deadlock* rather than error — if a cell hangs, interrupt it and lower the batch size |
| Model scores near chance on a fine-tuned prompt | You are using the zero-shot prompt format against a model trained on the structured one. The prompt is part of the artifact |
| Loss goes down, accuracy does not | You are at the label-noise ceiling. Re-read lesson 02, section 3 |
