# ---
# jupyter:
#   kernelspec:
#     display_name: Fine-tuning tutorial (uv)
#     language: python
#     name: fine-tuning-tutorial
# ---

# %% [markdown]
# # 07 · Packaging and hosting
#
# Right now your model is three things: a base model from the Hub, an adapter directory
# with a few million floats in it, and a prompt template that lives in your head. That is
# not something you can hand to anyone.
#
# This lesson turns it into a single artifact you can ship, and along the way answers the
# question that decides *how* you ship it: **what does quantisation cost you?** We measure
# rather than guess.
#
# The main artifact is an **MLX** model — for running on this Mac, and for publishing to
# Hugging Face. We also try to produce a **GGUF**, the format llama.cpp, Ollama and LM Studio
# read, and hit a limitation that is worth meeting early rather than on shipping day.
#
# **Time: about 10 minutes.**

# %%
import sys

sys.path.insert(0, "lib")
import warnings

warnings.filterwarnings("ignore")

import json
import subprocess
from pathlib import Path

import pandas as pd

import readout as R
from tickets import QUESTIONS, generate_tickets, split

tickets = generate_tickets(1200, seed=7)
dev = split(tickets, seed=7)["development"]

BASE_MODEL = "Qwen/Qwen3-0.6B"
ADAPTER = "adapters/decision"
FUSED = Path("models/triage-0.6b")
QUANTISED = Path("models/triage-0.6b-4bit")


def run(cmd: list[str]) -> str:
    """Run a command, show it, and return its tail."""
    print(f"$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError(f"exit {result.returncode}")
    return result.stdout


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2


# %% [markdown]
# ## 1. Fuse the adapter into the base
#
# LoRA keeps the base weights frozen and trains two small matrices per layer. **Fusing**
# multiplies those into the base weights once, producing an ordinary model directory that
# needs no adapter at load time.
#
# The trade is straightforward:
#
# | | adapter + base | fused |
# |---|---|---|
# | Files to ship | two (base is a dependency) | one |
# | Load time | base + adapter apply | base only |
# | Swap tasks at runtime | yes — load a different adapter | no |
#
# Keep adapters separate while you are iterating on several tasks from one base. Fuse for
# the artifact you ship.

# %%
if FUSED.exists():
    print(f"{FUSED} already exists, skipping")
else:
    run(["uv", "run", "mlx_lm.fuse", "--model", BASE_MODEL,
         "--adapter-path", ADAPTER, "--save-path", str(FUSED)])

print(f"\nfused model: {dir_size_mb(FUSED):.0f} MB")
print(sorted(p.name for p in FUSED.iterdir()))

# %% [markdown]
# ## 2. Quantise, and measure what it costs
#
# A bf16 weight costs 2 bytes. At 4 bits it costs 0.5 — a 4x reduction — which is the
# difference between a model that fits comfortably and one that does not, and it is a
# straight speedup on memory-bound hardware like Apple Silicon.
#
# The cost is accuracy, and the point of this section is that **you measure it rather than
# assume it**. For a small model on an easy task it is often nothing at all.

# %%
if QUANTISED.exists():
    print(f"{QUANTISED} already exists, skipping")
else:
    run(["uv", "run", "mlx_lm.convert", "--hf-path", str(FUSED), "--mlx-path", str(QUANTISED),
         "--quantize", "--q-bits", "4", "--q-group-size", "64"])

rows = [
    {"variant": "bf16 (fused)", "size MB": round(dir_size_mb(FUSED), 1)},
    {"variant": "4-bit", "size MB": round(dir_size_mb(QUANTISED), 1)},
]
pd.DataFrame(rows)

# %% [markdown]
# Now the part that matters. Same development tickets, same readout, three models: the
# original base, the fused fine-tune, and the 4-bit quantised fine-tune.

# %%
from mlx_lm import load

variants = {}
variants["base (no fine-tune)"] = load(BASE_MODEL)
variants["fine-tuned bf16"] = load(str(FUSED))
variants["fine-tuned 4-bit"] = load(str(QUANTISED))

# %%
rows = []
for name, (model, tokenizer) in variants.items():
    result = R.evaluate(model, tokenizer, dev, QUESTIONS, style="structured")
    row = {"variant": name}
    for r in result.itertuples():
        row[r.question] = round(r.accuracy, 3)
    rows.append(row)
pd.DataFrame(rows)

# %% [markdown]
# **Read that table before you assume 4-bit is free.** The size win is real — 1148 MB down
# to 331 MB, a 3.5x reduction — but the cost is not evenly spread:
#
# * `frustration` is **unchanged** (0.900 both ways).
# * `escalate` loses 3 points (0.858 → 0.825).
# * `department` loses **11 points** (0.800 → 0.692).
#
# Is the `department` drop real, or noise? With 120 development tickets and an accuracy near
# 0.75, the standard error is about 4 points, so an 11-point gap is roughly 2.5 standard
# errors — likely real, but not overwhelming. This is exactly the situation lesson 05's
# paired bootstrap exists for, and the honest answer is "measure it on more data before
# betting a deployment on it".
#
# A plausible mechanism: `department` is the 4-way question, so it depends on the most
# finely-tuned distinctions, and 4-bit weights blur small logit differences first. Binary and
# coarse ordinal questions survive quantisation better. That is a pattern worth expecting,
# not a law.
#
# What to do if you hit it, in order of cost:
#
# 1. **8-bit** instead of 4 — roughly 2x smaller than bf16 rather than 3.5x, and usually
#    lossless.
# 2. **A smaller group size** (`--q-group-size 32` instead of 64) — more scale factors, so
#    less error, at a little more size.
# 3. **Keep the question that matters unquantised** — you can serve a bf16 model and accept
#    the memory, or split heads across differently-quantised copies.
#
# The point is not that quantisation is bad. It is that **the decision is made from a
# measurement on held-out data**, not from a blog post claiming 4-bit is "basically
# lossless". The comparison above runs through the *same* `readout` code, so the difference
# is in the weights, not the pipeline.

# %% [markdown]
# ## 3. The prompt is part of the artifact
#
# Before shipping, write down what you trained. A model directory with no record of its
# prompt is a model nobody can use, including you in six months.

# %%
card = {
    "base_model": BASE_MODEL,
    "adapter": ADAPTER,
    "prompt_template": (
        "<ticket>\n{state}\n</ticket>\n\n"
        "<question>{instructions}</question>\n"
        "<options>{options}</options>\n"
        "<answer>"
    ),
    "questions": {
        name: {
            "type": spec["type"],
            "instructions": spec["instructions"],
            "options": spec.get("options") or ["false", "true"],
            "legend": spec.get("legend"),
        }
        for name, spec in QUESTIONS.items()
    },
    "notes": "Instructions and option names are bound by the fine-tune. Changing them requires retraining.",
}
(FUSED / "decision_card.json").write_text(json.dumps(card, indent=2))
print((FUSED / "decision_card.json").read_text()[:600])

# %% [markdown]
# ## 4. GGUF, and running it under Ollama
#
# GGUF is llama.cpp's format, and it is what Ollama, LM Studio and most desktop tooling
# read. `mlx_lm.fuse` advertises a `--export-gguf` flag, so let's try it.

# %%
GGUF = Path("models/triage-0.6b-f16.gguf")
GGUF.parent.mkdir(parents=True, exist_ok=True)

export = subprocess.run(
    ["uv", "run", "mlx_lm.fuse", "--model", BASE_MODEL,
     "--adapter-path", ADAPTER, "--save-path", "/tmp/gguf-fuse",
     "--export-gguf", "--gguf-path", str(GGUF)],
    capture_output=True, text=True,
)
print(f"exit code: {export.returncode}")
for line in (export.stdout + export.stderr).splitlines():
    if "Error" in line or "error" in line or "not supported" in line:
        print("  " + line.strip())

# %% [markdown]
# **It fails, and the failure is instructive.** `mlx_lm`'s GGUF exporter supports exactly
# three architectures — you can read the allowlist yourself:

# %%
import re

fuse_source = Path(
    subprocess.run(
        [sys.executable, "-c", "import mlx_lm.fuse, inspect; print(inspect.getsourcefile(mlx_lm.fuse))"],
        capture_output=True, text=True,
    ).stdout.strip()
)
match = re.search(r"model_type not in (\[[^\]]+\])", fuse_source.read_text())
print(f"architectures mlx-lm can export to GGUF: {match.group(1) if match else '?'}")

# %% [markdown]
# Qwen3 is not on that list, and the exporter would not work even if you bypassed the check:
# it writes `general.architecture = "llama"` into the metadata regardless. So the flag
# exists, and it does not do what its name suggests for most modern models.
#
# This is worth pausing on, because it is the ordinary texture of this work. A tool
# advertises a capability, the capability has a narrow allowlist, and the allowlist is not
# in the `--help`. The fix is not exotic:

# %%
print('''Use llama.cpp's own converter instead. It supports far more architectures:

    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
    pip install -r requirements.txt
    python convert_hf_to_gguf.py /path/to/models/triage-0.6b --outfile triage-0.6b-f16.gguf
    ./llama-quantize triage-0.6b-f16.gguf triage-0.6b-Q4_K_M.gguf Q4_K_M

Or skip the conversion entirely: Qwen3 GGUFs are widely published, so for distribution you
can ship your adapter plus a pointer to an existing quantised base rather than a whole
model. The adapter is a few megabytes; the base is a gigabyte you did not need to build.''')

# %% [markdown]
# ### A caveat that matters more than the format
#
# Once you leave MLX you leave the readout behind. Ollama and llama.cpp serve **generation**.
# They will happily load a GGUF of your model and chat with it, but they will not give you
# `P(returns) = 0.47, P(shipping) = 0.28, P(billing) = 0.25` over a label set in one forward
# pass.
#
# So the honest framing for this section is: **GGUF is how you distribute the base model
# for other uses; the decision endpoint is what you distribute for triage.**
#
# For a decision model, the thing to publish is the MLX model plus `serve.py` and the
# decision card. Someone who wants a chatbot can take the GGUF. Ollama *can* be driven into
# a classifier with a `/api/generate` call and `logprobs`, but you would be reimplementing
# the readout badly on top of a slower runtime — worth knowing it exists, not worth doing.

# %%
# If Ollama is installed, this is what importing a custom model looks like.
OLLAMA_SNIPPET = '''
# A Modelfile is a two-line recipe: which weights, which prompt template.
cat > Modelfile <<'EOF'
FROM ./models/triage-0.6b-f16.gguf
TEMPLATE """{{ .Prompt }}"""
PARAMETER stop "<|im_end|>"
EOF

ollama create triage-0.6b -f Modelfile
ollama run triage-0.6b "Ticket: my order is late. Department:"
'''
print(OLLAMA_SNIPPET)

# %%
ollama = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
if ollama.returncode == 0:
    print(f"detected: {ollama.stdout.strip()}")
    print(f"installed models:\n{subprocess.run(['ollama', 'list'], capture_output=True, text=True).stdout}")
else:
    print("ollama not found on PATH")

# %% [markdown]
# ## 5. Publishing to Hugging Face
#
# `mlx_lm` can upload directly, so this is one flag. Two things to do first:
#
# * **Authenticate:** `hf auth login` (the `huggingface-cli` name is retired).
# * **Decide public or private.** For a model trained on customer text, private. Note that
#   publishing is not reversible in the way you might hope — a public repo can be cloned,
#   mirrored and cached within minutes, so treat the upload itself as the point of no
#   return.
#
# ```bash
# uv run mlx_lm.fuse --model Qwen/Qwen3-0.6B \
#   --adapter-path adapters/decision \
#   --save-path models/triage-0.6b \
#   --upload-repo your-name/triage-0.6b
# ```
#
# The repo gets the model, the tokenizer, and any `decision_card.json` you left in the
# directory — which is the reason section 3 exists. A model card that states the exact
# prompt is the difference between a model someone can use and a directory of tensors.
#
# What to put in the model card, at minimum:
#
# * the base model and the adapter,
# * the exact prompt template and the exact instruction strings,
# * the label space per question, with the legend for score questions,
# * the measured accuracy **and the label-noise ceiling it should be read against**,
# * the calibration temperature, and which split it was fitted on.

# %% [markdown]
# ## 6. What to ship, and to whom
#
# | Audience | Artifact | Why |
# |---|---|---|
# | Your own app, on this Mac | adapter + `serve.py` | smallest, loads fastest, swap adapters freely |
# | A colleague with a Mac | fused MLX model + `serve.py` + decision card | self-contained, no adapter confusion |
# | Someone who wants a chatbot | GGUF | runs in Ollama/LM Studio with no Python |
# | The public | MLX model on the Hub + card | reproducible, and the card prevents prompt drift |
#
# The one thing to avoid is shipping the adapter without the prompt template. It is the
# most common failure of published fine-tunes, and it is silent: the model loads, answers,
# and is wrong.
#
# ---
#
# ### Exercises
#
# 1. **Find the quantisation cliff.** Re-run section 2 at 8-bit, 4-bit and 3-bit. Where does
#    accuracy actually break? Compare against the label-noise ceiling from lesson 02 —
#    is the drop real, or inside the noise?
#
# 2. **Measure the size/speed tradeoff.** Time the readout for the bf16 and 4-bit models.
#    What is the actual speedup on this machine? Is it 4x, or less?
#
# 3. **Write the model card.** Write a real one for this model, following section 5's list.
#    Then hand it to someone and ask them to reproduce one prediction. If they cannot, the
#    card is incomplete.
#
# 4. **Ship it to a friend.** Export the GGUF, write the Modelfile, and get the model
#    running under Ollama on another machine. Then work out what you would have to add
#    before that person could reproduce your *decision* endpoint rather than a chatbot.

# %% [markdown]
# ---
# You have reached the end of the tutorial.
#
# **[Appendix](APPENDIX.md)** covers the PyTorch/MPS path (which is what the Kev repository
# uses), a glossary, further reading, and troubleshooting.
