# ---
# jupyter:
#   kernelspec:
#     display_name: Fine-tuning tutorial (uv)
#     language: python
#     name: fine-tuning-tutorial
# ---

# %% [markdown]
# # 00 · Setup, and the only arithmetic you need
#
# Before any training, you need one piece of mental math: **how much memory does a
# model take?** Every failure you will hit on a 24 GB Mac is that arithmetic going wrong,
# so this lesson makes it concrete and then measures it on your actual machine.
#
# By the end you will have:
#
# 1. confirmed your hardware and toolchain,
# 2. a rule for estimating memory that you can apply to any model size,
# 3. run a real model locally and seen its generation speed.
#
# **Time: about 5 minutes**, most of it downloading a small model.

# %% [markdown]
# ## 1. What are we actually working with?
#
# A language model is a big pile of numbers (weights) plus a program that multiplies them.
# "Fine-tuning" means nudging some of those numbers using your labelled examples. The
# entire cost of doing that — in money, time, and memory — comes down to *how many numbers
# you are touching and how precisely you store them*.

# %%
import platform
import subprocess
import sys

print(f"Python           {sys.version.split()[0]}")
print(f"Machine          {platform.machine()}")
print(f"macOS            {platform.mac_ver()[0]}")

chip = subprocess.run(
    ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
).stdout.strip()
mem_gb = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 1024**3
print(f"Chip             {chip}")
print(f"Unified memory   {mem_gb:.0f} GB")

# %% [markdown]
# Two things about Apple Silicon matter for everything that follows:
#
# * **Unified memory.** The CPU and GPU share one pool. There is no separate VRAM to
#   copy tensors into, so the "will it fit" question has a single answer for both.
# * **No discrete GPU.** You get roughly 1/10th to 1/5th of an NVIDIA card's training
#   throughput for the same money. Small models are how you stay inside that budget.
#
# On a 24 GB machine you should assume you have **about 18 GB** to play with. macOS and
# your apps take the rest, and the memory allocator fragments. Every estimate below is
# against that 18 GB, not 24.

# %%
USABLE_GB = 18

# %% [markdown]
# ## 2. The arithmetic
#
# The number of weights is written in the model name: `Qwen3-0.6B` has about 0.6 billion
# parameters. Each parameter costs a fixed number of bytes depending on how you store it:
#
# | Precision | Bytes per weight | 0.6B model | 1.7B model | 8B model |
# |---|---|---|---|---|
# | fp32 (full precision) | 4 | 2.4 GB | 6.8 GB | 32 GB |
# | bf16 / fp16 (training default) | 2 | 1.2 GB | 3.4 GB | 16 GB |
# | 8-bit quantised | 1 | 0.6 GB | 1.7 GB | 8 GB |
# | 4-bit quantised | 0.5 | 0.3 GB | 0.85 GB | 4 GB |
#
# That is the *weights only*. Training adds more, and this is where people get caught out:
#
# * **Gradients** — one extra copy of every weight you are training. Same dtype.
# * **Optimizer state** — Adam keeps two running averages per trained weight, usually in
#   fp32. That is 8 bytes per weight, four times the cost of the weight itself.
# * **Activations** — the intermediate values saved for the backward pass. This is the
#   one that scales with batch size and sequence length, and the one that surprises people.
#
# So a **full fine-tune** of an N-parameter model needs roughly `2N (weights) + 2N
# (gradients) + 8N (Adam) = 12N` bytes — twelve times the parameter count. That is why
# full fine-tuning a 2B model wants 24 GB before you have done any work.

# %%
import sys

sys.path.insert(0, "lib")
from pretty import budget, memory_table

for name, n_params in [("0.6B", 0.6e9), ("1.7B", 1.7e9), ("8B", 8e9)]:
    print(f"\n=== {name} parameters ===")
    display(memory_table(budget(n_params), n_params))  # noqa: F821

# %% [markdown]
# Read the table for the 8B column: a full fine-tune needs ~96 GB. That is not a tuning
# problem, it is impossible on this machine. Meanwhile LoRA on a 4-bit 8B base fits in
# about 4.5 GB.
#
# > **The rule to remember:** full fine-tuning costs ~12 bytes per parameter, LoRA on a
# > quantised base costs ~0.5 bytes per parameter. That is a **24x** difference, and it is
# > the entire reason this tutorial can exist on a laptop.
#
# Why LoRA is so much cheaper: instead of touching all the weights, you freeze the model
# and train two small matrices per layer whose product is added to the original weight.
# Those matrices have a tiny *rank* (typically 8–32), so they hold maybe 0.1–1% of the
# parameter count. Fewer trainable numbers means smaller gradients and a much smaller
# optimizer state. Lesson 04 covers the mechanics properly.

# %% [markdown]
# ## 3. Confirm the toolchain

# %%
import importlib

for mod in ["mlx.core", "mlx_lm", "torch", "transformers", "peft", "sklearn", "pandas"]:
    try:
        m = importlib.import_module(mod)
        print(f"  ok   {mod:14s} {getattr(m, '__version__', '')}")
    except ImportError as e:
        print(f"  MISSING {mod:14s} {e}")
        raise

import torch

print(f"\n  torch MPS backend available: {torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    raise SystemExit("MPS not available — you are probably on an Intel Mac; the rest of this tutorial assumes Apple Silicon.")

# %% [markdown]
# If anything is missing, from a terminal in this folder:
#
# ```bash
# uv sync
# ```
#
# Everything in this tutorial runs locally. No API keys, no cloud GPU, no Modal account.

# %% [markdown]
# ## 4. Run a real model, and time it
#
# Now let's put a model in memory and watch it generate. `Qwen3-0.6B` is a good first
# choice: genuinely capable, small enough to iterate on in seconds, and released under
# Apache-2.0 so you can publish what you build.
#
# First run downloads about 1.5 GB.

# %%
import time

from mlx_lm import generate, load

model, tokenizer = load("Qwen/Qwen3-0.6B")

messages = [{"role": "user", "content": "In one sentence, what is a support ticket?"}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(repr(prompt))

# %% [markdown]
# **A gotcha worth meeting on day one.** Qwen3 is a *reasoning* model: left alone it opens
# a `<think>` block and writes its reasoning before the answer. That is great for hard
# problems and terrible for everything in this tutorial — it costs tokens, adds latency,
# and makes output parsing a chore. Every Qwen3 model takes an `enable_thinking` flag in
# its chat template. Turn it off unless you actually want a chain of thought.

# %%
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
)

t0 = time.perf_counter()
out = generate(model, tokenizer, prompt=prompt, max_tokens=60, verbose=False)
elapsed = time.perf_counter() - t0

print(out.strip())
print(f"\n--- {elapsed:.2f}s wall clock")

# %% [markdown]
# ### Measuring speed properly
#
# Wall-clock time for one short answer tells you little, because it includes loading and
# prompt processing. The figure that matters for serving is **tokens per second during
# generation**, and the figure that matters for training is **iterations per second**
# (which you will see in lesson 04).
#
# On an M4 Pro with 24 GB, expect roughly:
#
# | Model | Generation (MLX, 4-bit) | LoRA training step |
# |---|---|---|
# | 0.6B | 100–150 tok/s | ~25 it/s |
# | 1.7B | 40–70 tok/s | ~8 it/s |
# | 8B | 10–20 tok/s | ~2 it/s |
#
# These are within a factor of two for most tasks. If you see numbers 10x worse, something
# is wrong — usually you are accidentally running on CPU, or another process is competing
# for memory.

# %%
t0 = time.perf_counter()
long_out = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)
elapsed = time.perf_counter() - t0
n_tokens = len(tokenizer.encode(long_out))
print(f"generated ~{n_tokens} tokens in {elapsed:.2f}s  ->  {n_tokens / elapsed:.1f} tok/s")

# %% [markdown]
# ## 5. Two ideas to carry forward
#
# **Idea 1 — everything is a memory budget.** Before you pick a model, compute
# `params × bytes-per-param` for the mode you want. If it exceeds ~18 GB, drop to a
# smaller model, a lower bit-width, or LoRA. There is no clever workaround; this is the
# constraint that shapes every other decision in this tutorial.
#
# **Idea 2 — you do not need a big model for classification.** The tasks in this tutorial
# are "read this text, pick from a small set of labels". A 0.6B model can do that. The
# hard part is never the model size — it is the *data* and the *calibration*, which is
# what lessons 02 and 04 are about.
#
# ---
#
# ### Exercises
#
# 1. **Estimate before you measure.** Pick a model you might want to use
#    (e.g. `Qwen/Qwen3-1.7B`, `google/gemma-3-1b-it`, `meta-llama/Llama-3.2-1B`). Using the
#    rules above, write down how much memory LoRA training on it needs. Then check whether
#    the estimate holds by running lesson 04 on it.
#
# 2. **Find your usable budget.** Open Activity Monitor while the cell above runs. How much
#    memory is free? Replace `USABLE_GB` above with your real number and re-run section 2.
#
# 3. **Break it on purpose.** Try loading a model much larger than your budget, e.g.
#    `load("Qwen/Qwen3-8B")`, and watch what happens. (You can interrupt the cell.) Knowing
#    what memory pressure looks like — swap, beachball, or a hard allocation error — saves
#    you from misdiagnosing it later as a code bug.

# %% [markdown]
# ---
# **Next:** [01 · From text to probability](01_logits_to_probabilities.ipynb) — what the
# model actually outputs, and how to turn that into a classifier without generating a
# single token.
