"""The decision readout: turn a language model into a classifier without generating text.

This is the single most important idea in the tutorial, so it lives in one place and
every notebook imports it.

A causal language model's only output is a score for every token in its vocabulary. To
classify, you:

1. put the input and the question in a prompt that ends where the answer goes,
2. run ONE forward pass (no generation),
3. look at the scores at the final position,
4. softmax over *only the tokens that spell your labels*, renormalising so they sum to 1.

Step 4 is where the subtleties live, and they are not cosmetic:

* **Your labels must be single tokens.** "calm" is two tokens and "annoyed" is three in
  the Qwen tokenizer, so there is no one position to read. Check, do not assume.
* **Leading whitespace changes the token.** " angry" and "angry" are different ids, and a
  model that has just emitted a space will put its mass on the space-prefixed one.
* **Renormalising can amplify noise.** If the labels carry no probability mass at all,
  dividing by their sum manufactures confident-looking numbers out of float dust. Always
  look at `label_mass` — what share of the model's total probability the labels hold —
  before trusting a readout.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


class LabelTokenError(ValueError):
    """Raised when a label does not tokenise to exactly one token."""


def label_token_ids(tokenizer, labels: list[str]) -> dict[str, int]:
    """Map each label to its single token id, refusing labels that are not one token.

    Raises with a helpful message rather than silently reading the wrong position — the
    failure mode this function exists to prevent.
    """
    out: dict[str, int] = {}
    for label in labels:
        ids = tokenizer.encode(label)
        if len(ids) != 1:
            pieces = [tokenizer.decode([i]) for i in ids]
            raise LabelTokenError(
                f"label {label!r} is {len(ids)} tokens {ids} -> {pieces}. "
                f"A readout needs a single token per label. Rename the label, or prefix it "
                f"with a space, or use the numeric-level convention (0/1/2) with a legend."
            )
        out[label] = ids[0]
    return out


def build_prompt(tokenizer, state: str, ask: str, system: str | None = None) -> str:
    """Chat-template a ticket plus a question, ending where the answer should begin.

    Both halves matter, and we measured it. For the same question and ticket, the share of
    probability landing on the label tokens was:

        raw "Department:" only ................ 0.0000
        raw + instruction ..................... 0.0007
        chat template, no label enumeration ... 0.0000
        chat template + "one of: a, b, c" ..... 0.9807

    So it is not enough to use the instruct format, and not enough to state the question.
    The model needs the allowed answers *enumerated in the prompt* before it will put mass
    on them. `zero_shot_ask` below does that.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": f"Ticket: {state.strip()}\n\n{ask}"})
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )


def readout(model, tokenizer, prompt: str, labels: list[str], temperature: float = 1.0):
    """One forward pass; returns (probabilities over labels, label_mass).

    `temperature` divides the logits before the softmax. T > 1 flattens the distribution,
    T < 1 sharpens it. Fitting T on held-out data is the whole of lesson 04.
    """
    ids = mx.array(tokenizer.encode(prompt))[None]
    logits = model(ids)[:, -1, :] / temperature
    probs = mx.softmax(logits, axis=-1)[0]

    tok_ids = label_token_ids(tokenizer, labels)
    raw = {label: probs[i].item() for label, i in tok_ids.items()}
    mass = sum(raw.values())
    if mass <= 0:
        return {label: float("nan") for label in labels}, 0.0
    return {label: v / mass for label, v in raw.items()}, mass


def label_logits(model, tokenizer, prompt: str, labels: list[str]) -> "np.ndarray":
    """The raw logits at the final position, restricted to the label tokens.

    Temperature scaling has to work on these rather than on the softmaxed probabilities:
    dividing a probability by T is not the same transformation as dividing its logit by T,
    and only the latter leaves the normalisation intact.
    """
    import numpy as np

    ids = mx.array(tokenizer.encode(prompt))[None]
    row = model(ids)[0, -1, :]
    token_ids = label_token_ids(tokenizer, labels)
    return np.array([row[i].item() for i in token_ids.values()], dtype=np.float64)


def collect_logits(model, tokenizer, tickets, question: str, questions: dict,
                   style: str = "structured") -> "np.ndarray":
    """Label logits for every ticket, one row per ticket, for a single question."""
    import numpy as np

    spec = questions[question]
    labels = spec.get("options") or ["false", "true"]
    rows = []
    for ticket in tickets:
        prompt = (
            structured_prompt(tokenizer, ticket, question)
            if style == "structured"
            else build_prompt(tokenizer, ticket.state, zero_shot_ask(question, spec))
        )
        rows.append(label_logits(model, tokenizer, prompt, labels))
    return np.stack(rows)


def gold_indices(tickets, question: str, questions: dict) -> "np.ndarray":
    """Integer gold labels, aligned to the option order of the question."""
    import numpy as np

    spec = questions[question]
    labels = spec.get("options") or ["false", "true"]
    index = {label: i for i, label in enumerate(labels)}
    return np.array([index[gold(t, question, spec)] for t in tickets])


@dataclass
class Answer:
    """One question's answer, shaped like Kev's System One response."""

    question: str
    qtype: str
    probabilities: dict[str, float]
    label_mass: float
    temperature: float = 1.0

    @property
    def choice(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def confidence(self) -> float:
        return self.probabilities[self.choice]

    @property
    def noul(self) -> float:
        """P(true) — the convention for yes/no questions."""
        return self.probabilities.get("true", float("nan"))

    @property
    def score(self) -> float:
        """Expected level, e.g. 1.44 rather than a hard label. Ordinal-aware: the model
        saying 60/40 between levels 1 and 2 should not be reported as a bare '2'."""
        return sum(int(k) * v for k, v in self.probabilities.items())


def structured_prompt(tokenizer, ticket, question: str) -> str:
    """The delimited format the decision model is fine-tuned on.

    Lesson 01 measured the zero-shot mass on this format as ~0.000 — an untrained model
    has no reason to put probability on `<answer>returns`. After LoRA training on
    prompt/completion pairs in exactly this shape, the same measurement jumps to ~1.0. That
    contrast is the whole point of fine-tuning a readout.
    """
    from tickets import render_prompt

    prompt, _ = render_prompt(ticket, question)
    return prompt


def answer(model, tokenizer, ticket, question: str, questions: dict, temperature: float = 1.0,
           style: str = "zero_shot") -> Answer:
    """Answer one question about one ticket.

    `style="zero_shot"` uses the chat template and a natural-language phrasing — right for
    an untrained model. `style="structured"` uses the delimited prompt the fine-tuned
    decision model was trained on. Use the wrong one against a fine-tuned model and you
    will get confident nonsense; the prompt format is part of the artifact.
    """
    spec = questions[question]
    labels = spec.get("options") or ["false", "true"]
    if style == "structured":
        prompt = structured_prompt(tokenizer, ticket, question)
    else:
        prompt = build_prompt(tokenizer, ticket.state, zero_shot_ask(question, spec))
    probs, mass = readout(model, tokenizer, prompt, labels, temperature=temperature)
    return Answer(question, spec["type"], probs, mass, temperature)


def gold(ticket, question: str, spec: dict) -> str:
    """The label a ticket carries for a question, in the label space (not the legend)."""
    if question == "department":
        return ticket.department
    if question == "escalate":
        return str(ticket.escalate).lower()
    if question == "frustration":
        return str(ticket.frustration)
    raise KeyError(question)


def evaluate(model, tokenizer, tickets, questions: dict, temperature: float = 1.0,
             style: str = "zero_shot", verbose: bool = False) -> "object":
    """Score a model on the dev split. Returns a tidy DataFrame, one row per question.

    Accuracy alone is never the report — the mean confidence split by right/wrong answers
    is what tells you whether the model knows when it does not know.
    """
    import pandas as pd

    rows = []
    for question, spec in questions.items():
        n = right = 0
        conf_right: list[float] = []
        conf_wrong: list[float] = []
        masses: list[float] = []
        for t in tickets:
            a = answer(model, tokenizer, t, question, questions, temperature=temperature, style=style)
            is_right = a.choice == gold(t, question, spec)
            n += 1
            right += is_right
            (conf_right if is_right else conf_wrong).append(a.confidence)
            masses.append(a.label_mass)
        rows.append(
            {
                "question": question,
                "type": spec["type"],
                "accuracy": round(right / n, 3),
                "conf|right": round(sum(conf_right) / max(len(conf_right), 1), 3),
                "conf|wrong": round(sum(conf_wrong) / max(len(conf_wrong), 1), 3),
                "label_mass": round(sum(masses) / len(masses), 3),
                "n": n,
            }
        )
        if verbose:
            r = rows[-1]
            print(f"  {r['question']:12s} acc={r['accuracy']:.3f}  "
                  f"conf|right={r['conf|right']:.3f}  conf|wrong={r['conf|wrong']:.3f}  "
                  f"label_mass={r['label_mass']:.3f}")
    return pd.DataFrame(rows)


def zero_shot_ask(question: str, spec: dict) -> str:
    """A natural-language phrasing of the question, for the zero-shot baseline.

    Fine-tuning binds the exact prompt strings, so the trained model uses the structured
    format from `tickets.render_prompt` instead. This one exists so lesson 01 can ask an
    untrained model a fair question."""
    options = spec.get("options") or ["true", "false"]
    legend = spec.get("legend")
    if legend:
        opts = ", ".join(f"{k} = {v}" for k, v in legend.items())
    else:
        opts = ", ".join(options)
    return f"{spec['instructions']} Reply with exactly one of: {opts}."
