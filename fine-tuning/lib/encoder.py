"""A conventional multi-task encoder classifier: ModernBERT + one linear head per question.

This is the "boring" answer to the triage task, and for a lot of real problems it is the
right one. An encoder reads the whole input bidirectionally and puts small classification
heads on top. Compared with the decoder + readout approach:

    encoder classifier          decoder + readout
    ------------------          -----------------
    bidirectional attention     causal (left-to-right only)
    150M params                 600M-2B params
    ~1-3 ms per ticket          ~20-60 ms per ticket
    fixed heads, trained per task   any number of questions, any type
    cannot be prompted          works zero-shot if you prompt it well
    one label per head          calibrated distributions + abstain

The tradeoff in one line: **encoders are better classifiers, decoders are better decision
models.** One label at high volume: use an encoder. Several typed questions, with
calibrated probabilities, from one shared input: that is what lesson 05 is for.

The architecture here deliberately mirrors the decision model, so the comparison is fair:
one shared trunk, one head per question, all questions trained jointly on the same tickets.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

BASE = "answerdotai/ModernBERT-base"
MAX_LENGTH = 128

# Question name -> (number of labels, label lookup)
TASKS: dict[str, tuple[int, list[str]]] = {
    "department": (4, ["returns", "shipping", "billing", "other"]),
    "escalate": (2, ["false", "true"]),
    "frustration": (3, ["0", "1", "2"]),
}


def labels_for(ticket, question: str) -> str:
    """The ticket's label for a question, as a string in the question's label space."""
    if question == "department":
        return ticket.department
    if question == "escalate":
        return str(ticket.escalate).lower()
    if question == "frustration":
        return str(ticket.frustration)
    raise KeyError(question)


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class MultiTaskEncoder(nn.Module):
    """Shared ModernBERT trunk, one linear head per question."""

    def __init__(self, base: str = BASE):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base)
        hidden = self.backbone.config.hidden_size
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden, n) for name, (n, _) in TASKS.items()}
        )
        # A fresh head starts at chance. Nothing clever here — the trunk does the work.
        for head in self.heads.values():
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # ModernBERT prepends a CLS token at position 0; that is the sequence summary.
        pooled = out.last_hidden_state[:, 0]
        return {name: head(pooled) for name, head in self.heads.items()}


def load_tokenizer(base: str = BASE):
    return AutoTokenizer.from_pretrained(base)


def encode(tokenizer, texts: list[str]):
    return tokenizer(
        texts, truncation=True, max_length=MAX_LENGTH, padding="max_length", return_tensors="pt"
    )


def _targets(tickets) -> dict[str, torch.Tensor]:
    out = {}
    for question, (_, label_list) in TASKS.items():
        index = {label: i for i, label in enumerate(label_list)}
        out[question] = torch.tensor([index[labels_for(t, question)] for t in tickets])
    return out


def train(tickets, epochs: int = 4, batch_size: int = 16, lr: float = 3e-5,
          seed: int = 0, weights: dict[str, float] | None = None, verbose: bool = True):
    """Fine-tune trunk and heads jointly on all three questions.

    Full fine-tuning costs ~12 bytes/param, so ~1.8 GB for this model — cheap at 150M.
    Past ~1B parameters you would switch to LoRA with `peft`: identical loop, but the
    trunk is frozen and only the adapters get gradients.
    """
    torch.manual_seed(seed)
    tokenizer = load_tokenizer()
    model = MultiTaskEncoder().to(device())
    weights = weights or {name: 1.0 for name in TASKS}

    enc = encode(tokenizer, [t.state for t in tickets])
    targets = _targets(tickets)
    loader = DataLoader(
        TensorDataset(enc["input_ids"], enc["attention_mask"], *targets.values()),
        batch_size=batch_size, shuffle=True,
    )
    names = list(TASKS)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    history: list[float] = []
    model.train()
    t0 = time.perf_counter()
    for epoch in range(epochs):
        epoch_losses = []
        for batch in loader:
            ids, mask = batch[0].to(device()), batch[1].to(device())
            ys = [b.to(device()) for b in batch[2:]]
            logits = model(ids, mask)
            loss = sum(
                weights[name] * nn.functional.cross_entropy(logits[name], y)
                for name, y in zip(names, ys)
            ) / sum(weights.values())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            history.append(loss.item())
            epoch_losses.append(loss.item())
        if verbose:
            print(f"  epoch {epoch + 1}/{epochs}  mean loss {np.mean(epoch_losses):.4f}")
    if verbose:
        print(f"  {epochs} epochs in {time.perf_counter() - t0:.1f}s on {device()}")
    model.eval()
    return model, tokenizer, history


@torch.no_grad()
def predict_logits(model, tokenizer, texts: list[str], batch_size: int = 32) -> dict[str, np.ndarray]:
    """Raw per-question logits, each of shape (n_texts, n_labels).

    Temperature scaling operates on logits, not probabilities, so we keep them: dividing a
    probability by T does not give the same answer as dividing its logit by T.
    """
    dev = device()
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in TASKS}
    for i in range(0, len(texts), batch_size):
        enc = encode(tokenizer, texts[i : i + batch_size])
        out = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
        for name, logit in out.items():
            chunks[name].append(logit.float().cpu().numpy())
    return {name: np.concatenate(v, axis=0) for name, v in chunks.items()}


def predict_proba(model, tokenizer, texts: list[str], batch_size: int = 32) -> dict[str, np.ndarray]:
    """Per-question softmax probabilities, each of shape (n_texts, n_labels)."""
    logits = predict_logits(model, tokenizer, texts, batch_size=batch_size)
    return {name: torch.softmax(torch.from_numpy(v), dim=-1).numpy() for name, v in logits.items()}


def evaluate(model, tokenizer, tickets, verbose: bool = True) -> dict:
    """Accuracy and confidence split by right/wrong, per question."""
    texts = [t.state for t in tickets]
    probs = predict_proba(model, tokenizer, texts)
    logits = predict_logits(model, tokenizer, texts)
    report: dict[str, dict] = {}
    for question, (n_classes, label_list) in TASKS.items():
        index = {label: i for i, label in enumerate(label_list)}
        gold = np.array([index[labels_for(t, question)] for t in tickets])
        pred = probs[question].argmax(axis=1)
        correct = pred == gold
        conf = probs[question].max(axis=1)
        report[question] = {
            "n": len(tickets),
            "accuracy": float(correct.mean()),
            "mean_conf_right": float(conf[correct].mean()) if correct.any() else float("nan"),
            "mean_conf_wrong": float(conf[~correct].mean()) if (~correct).any() else float("nan"),
            "probs": probs[question],
            "logits": logits[question],
            "preds": pred,
            "gold": gold,
            "classes": label_list,
        }
        if verbose:
            print(
                f"  {question:12s} acc={report[question]['accuracy']:.3f}  "
                f"conf|right={report[question]['mean_conf_right']:.3f}  "
                f"conf|wrong={report[question]['mean_conf_wrong']:.3f}"
            )
    return report


def save(model, tokenizer, path: str | Path = "models/encoder") -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "base": BASE}, path / "model.pt")
    tokenizer.save_pretrained(path)
    return path


def load(path: str | Path = "models/encoder"):
    path = Path(path)
    checkpoint = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    model = MultiTaskEncoder(base=checkpoint["base"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device()).eval()
    return model, load_tokenizer(checkpoint["base"])
