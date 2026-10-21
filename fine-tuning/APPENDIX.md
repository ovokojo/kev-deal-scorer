# Appendix

Reference material for the tutorial: the PyTorch path, a glossary, what to read next, and
the errors you are most likely to hit.

---

## A · The PyTorch + MPS + PEFT path

Every notebook in this tutorial uses **MLX**, Apple's array framework, because on an M4 it
is the fastest way to train and serve a small model. But MLX is not the only option, and
you may already have a reason to prefer PyTorch:

* **Your existing code is PyTorch.** The [Kev](https://github.com/jaredpalmer/kev)
  repository trains with `torch` + `transformers` + `peft` on MPS. Code you write here
  transfers to it directly; MLX code does not.
* **You need the Hugging Face ecosystem.** `Trainer`, `datasets`, `evaluate`, `accelerate`,
  and the enormous pile of tooling built on them.
* **You want the same script to run on a CUDA box later.** MLX is Apple-only.

The cost is throughput. MPS runs a Transformer training step roughly 3–5x slower than MLX
on the same Mac, and it has gaps: some kernels are missing, and **MPS can deadlock rather
than raise an out-of-memory error** — the process simply hangs, with no traceback. The Kev
repository documents this for its 4B model: MPS does not run out of memory, it stops
responding. Interrupt and lower the batch size.

### PEFT/RoRA on the decision readout

This is the PyTorch equivalent of what lesson 04 does with `mlx_lm.lora`. It is close to
what `kev/train.py` does, minus the block-causal packing.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.bfloat16
).to(device)

# LoRA: freeze the base, train two small matrices per targeted layer.
config = LoraConfig(
    r=16,                          # rank — capacity of the adapter
    lora_alpha=32,                 # scaling; effective step size is alpha / r
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()   # expect ~0.2-1% of the base

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4
)

def encode(prompt: str, completion: str):
    """Tokenise the pair and mask the prompt, so loss lands only on the label.

    This is the analogue of mlx-lm's --mask-prompt. Getting it wrong trains the model to
    reproduce the ticket, which is the most common silent mistake in classification SFT.
    """
    p = tokenizer(prompt, add_special_tokens=False).input_ids
    c = tokenizer(completion, add_special_tokens=False).input_ids
    ids = torch.tensor([p + c])
    labels = ids.clone()
    labels[:, : len(p)] = -100          # -100 is ignored by cross_entropy
    return {"input_ids": ids, "labels": labels}

trainable = [p for p in model.parameters() if p.requires_grad]
model.train()
for step, (prompt, completion) in enumerate(loader):
    batch = {k: v.to(device) for k, v in encode(prompt, completion).items()}
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step % 50 == 0:
        print(step, round(loss.item(), 4))

model.save_pretrained("adapters/decision-torch")
```

To quantise the base first (QLoRA), add a `BitsAndBytesConfig` — though note that
`bitsandbytes` had no first-class Apple Silicon support for a long time, so on a Mac QLoRA
is the one place MLX is clearly ahead rather than merely faster.

### Reading probabilities out of a PyTorch model

The same readout, in torch:

```python
@torch.no_grad()
def label_probs(prompt: str, labels: list[str]) -> dict[str, float]:
    ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    logits = model(ids).logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    token_ids = [tokenizer.encode(l)[-1] for l in labels]   # verify single-token first!
    raw = {l: probs[i].item() for l, i in zip(labels, token_ids)}
    mass = sum(raw.values())
    return {k: v / mass for k, v in raw.items()}, mass
```

Every trap from lesson 01 applies unchanged: check the labels are single tokens, check the
`label_mass`, and make sure the prompt at inference matches the prompt at training.

---

## B · Glossary

**Adapter** — the small set of trained weights LoRA produces, applied on top of a frozen
base model. A few MB rather than a few GB.

**Batch size** — how many examples are averaged into one gradient step. Bigger is more
stable and uses more memory.

**bf16 / fp16 / fp32** — numeric formats holding 16, 16 and 32 bits per weight. bf16 is the
default for training; fp32 for optimiser state.

**Calibration** — whether a stated confidence matches the observed frequency. A model that
says 0.9 and is right 90% of the time is calibrated.

**Causal attention** — each token attends only to earlier tokens. What makes a decoder able
to generate left to right.

**Coverage at error** — the share of inputs you can handle automatically while keeping the
error rate among those you accepted under a budget. The business-facing metric.

**Decision model** — a model that answers typed questions about an input rather than
generating text. What Jev and Kev are.

**ECE (Expected Calibration Error)** — the population-weighted average gap between
confidence and accuracy across bins. Depends on the binning; always report the bin count.

**Encoder** — a bidirectional model (BERT family) that reads a whole input at once. Better
at classification, incapable of generation.

**Epoch** — one pass over the training set.

**Fine-tuning** — continuing to train a pretrained model on your own data.

**Full fine-tuning** — updating every weight. Costs ~12 bytes per parameter with Adam.

**Fused model** — a base model with its LoRA adapter multiplied permanently into the
weights. No adapter needed at load time.

**GGUF** — llama.cpp's weight format, read by Ollama and LM Studio. For generation, not for
the readout.

**Gradient** — the direction and magnitude in which a weight should move to reduce loss.

**LoRA (Low-Rank Adaptation)** — freeze the base model, train two small matrices per layer
whose product is *added* to the original weight. ~0.1–1% trainable parameters.

**Logit** — a raw, unbounded score from the model. Softmax turns a vector of logits into
probabilities.

**Label mass** — the total probability the model assigns to your label tokens before
renormalising. A health check on the readout; near zero means the numbers are meaningless.

**MPS** — Apple's Metal Performance Shaders backend for PyTorch. Slower than MLX here, and
can deadlock instead of raising OOM.

**NLL (Negative Log-Likelihood)** — a proper scoring rule; what temperature scaling
minimises.

**Noul** — Jev's name for a yes/no question, answered as `P(true)`.

**Precision / recall** — for the minority class, precision is "of the ones I flagged, how
many were right"; recall is "of the ones that were right, how many did I flag".

**Prompt template** — the exact formatting around the input and question. **Part of the
trained artifact**, not a cosmetic choice.

**QLoRA** — LoRA on top of a quantised base. ~0.5 bytes per parameter.

**Quantisation** — storing weights in fewer bits (8, 4, 3). ~4x smaller at 4-bit.

**Readout** — our term for reading a distribution over chosen tokens from a forward pass,
with no generation.

**Reliability curve** — confidence versus accuracy, per bin. The visual form of calibration.

**Score (question type)** — an ordinal question, answered as a distribution over levels
plus an expected value.

**Selective prediction** — letting a model abstain so a human handles the uncertain cases.

**Softmax** — turns logits into a probability distribution that sums to 1.

**Temperature** — the scalar dividing logits before the softmax. Fitting it on held-out data
is the standard calibration fix.

**Token** — the unit a model reads and writes. Roughly ¾ of a word in English.

**Validation / development / test split** — the three held-out sets. Iterate on
development, calibrate on a separate calibration set, touch test once.

**Zero-shot** — using a model on a task it was not trained for.

---

## C · What to read next

Ordered roughly by value to someone who has just finished this tutorial.

**Calibration and evaluation**

* Guo, Pleiss, Sun & Weinberger (2017) — *On Calibration of Modern Neural Networks*. The
  temperature scaling paper. Short, readable, and the source of everything in lesson 05.
* Naemi, Cooper & Hauskrecht (2015) — *Obtaining Well Calibrated Probabilities Using
  Bayesian Binning*. Where ECE comes from.
* Geifman & El-Yaniv (2017) — *Selective Classification for Deep Neural Networks*. The
  formal version of coverage at an error budget.
* Ovadia et al. (2019) — *Can You Trust Your Model's Uncertainty?* Shows how badly
  calibration degrades under distribution shift. Sobering, and relevant if your traffic
  drifts.
* Brier (1950) — *Verification of forecasts expressed in terms of probability*. A one-page
  paper that invented the Brier score, and still the clearest explanation of why proper
  scoring rules matter.

**Fine-tuning**

* Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*. Read this to
  understand what the `r` and `alpha` in your config actually do.
* Dettmers et al. (2023) — *QLoRA: Efficient Finetuning of Quantized LLMs*. Where the
  4-bit-base trick comes from.
* Mitchell et al. (2019) — *Model Cards for Model Reporting*. Short, and the reason
  section 3 of lesson 07 exists.
* Gebru et al. (2021) — *Datasheets for Datasets*. Read alongside the model card paper. The
  questions in it are the ones lesson 02 asks.

**Practice and judgement**

* Sculley et al. (2015) — *Hidden Technical Debt in Machine Learning Systems*. Explains why
  the dataset, not the model, is where projects die.
* The [Kev repository](https://github.com/jaredpalmer/kev) — the reference implementation
  this tutorial is a simplified version of. Read `kev/model.py` for the packed block-causal
  version of the readout, and `PLAN.md` for a genuine research log with negative results
  included.
* *Jev's Architecture Unmasked* (archerhume.com) — the write-up that Kev reconstructs. Good
  for understanding why anyone would build a decision model instead of prompting.

---

## D · Troubleshooting

### The model answers confidently and is wrong

Check `label_mass` first. If it is near zero, the readout is returning renormalised float
noise. Causes, in order of likelihood:

1. Labels that are not single tokens. Run `R.label_token_ids(tokenizer, labels)` — it
   raises rather than guessing. Remember `"calm"` is two tokens but `" calm"` is one.
2. The prompt does not enumerate the allowed answers. Lesson 01 measured this: 0.0000
   without the option list, 0.9807 with it.
3. You are using the zero-shot prompt format against a model trained on the structured
   one, or vice versa. The prompt is part of the artifact.

### Everything is slower than expected

* **Two training processes on MPS slow each other by roughly 10x.** Run one at a time. On
  MLX the contention is less severe but still real.
* Lower `--max-seq-length`. Attention cost is quadratic in sequence length, and the padding
  in a fixed-length batch is pure waste.
* Confirm you are actually on the GPU. In MLX, `mx.default_device()` should be `gpu`.

### Training hangs with no error

Almost always MPS memory pressure, which deadlocks instead of raising OOM. Interrupt, then
reduce `--batch-size` or `--max-seq-length`. Adding `--grad-checkpoint` trades compute for
memory if you cannot reduce the batch further.

### Loss goes down, development accuracy does not move

You are at the label-noise ceiling. Lesson 02, section 3. More training will not help;
better labels will. Read the errors sorted by confidence before changing anything else.

### Development accuracy is suspiciously high

Above the ceiling from lesson 02 means the measurement is broken, not the model. Check for
leakage: split on the thing that repeats (template, customer, agent, date) rather than on
rows.

### Fine-tuning made it worse

* **Learning rate too high.** 1e-4 is a LoRA default; for a further fine-tune of an already
  fine-tuned model, drop to 2e-5. Kev's own guidance for delta training is `lr <= 2e-5`.
* **Too few examples for the number of questions.** Three questions on 100 tickets is 300
  rows; expect noise. Kev's tooling puts the threshold for a measurable gain around 1000
  records.
* **You forgot to mask the prompt.** Without `--mask-prompt`, the model is partly being
  trained to reproduce the ticket text, which competes with learning the label.

### `RuntimeError: There is no Stream(cpu, 0) in current thread`

MLX's compute streams are thread-local. A model loaded on one thread cannot run on another.
In a web server this bites when FastAPI dispatches a `def` endpoint to a threadpool — make
the endpoint `async def` so it stays on the loading thread (lesson 06, section 7), or create
a stream per worker thread.

### `huggingface-cli` no longer works

The CLI was renamed. Use `hf auth login`, `hf upload`, `hf download`.

### `IncompleteSnapshotError: ... 3 file(s) are missing (.gitattributes, LICENSE, README.md)`

Hit this in lesson 07 while fusing. The cached copy of the base model is missing files that
`mlx_lm.fuse` wants, and `huggingface_hub` refuses to fetch them when it has been told not
to reach the network. Fix:

```bash
uv run hf download Qwen/Qwen3-0.6B
```

Worth knowing because the error names the missing files rather than the real problem, and
because a *partially* populated cache is easy to end up with if you ever downloaded under a
restricted network or with `HF_HUB_OFFLINE=1` set.

### `ValueError: Model type qwen3 not supported for GGUF conversion`

`mlx_lm.fuse --export-gguf` only handles `llama`, `mixtral` and `mistral` — it writes
llama-architecture metadata regardless, so the flag does not generalise. Use llama.cpp's
`convert_hf_to_gguf.py` instead (lesson 07, section 4), or ship the adapter and point at an
existing quantised GGUF of the base.
