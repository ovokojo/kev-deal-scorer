"""Synthetic e-commerce support tickets for the fine-tuning tutorial.

Everything in this tutorial runs on the same running example: routing and triaging
customer-support tickets. This module generates that dataset deterministically, so
every notebook sees exactly the same rows and your results are reproducible.

The generator is template-based on purpose. It needs no API key, no network, and no
teacher model, so the tutorial works offline. Lesson 02 shows how to replace it with
a real LLM generator or, better, your own labelled logs.

Three questions are asked of every ticket, mirroring the Jev/Kev question types:

    department   choice  -> returns | shipping | billing | other
    escalate     noul    -> true | false          (does a human need to look now?)
    frustration  score   -> 0 | 1 | 2             (calm | annoyed | angry)

Design notes that matter for the lessons:

* The phrasing carries the signal. Frustration and escalation are expressed in the
  *wording*, not handed to the model as a field, so the task is learnable from text.
* Tickets mention multiple things. A late delivery that caused a double charge names
  both shipping and billing; the label follows the *primary* ask. Lesson 02 uses these
  to show why ambiguous labels cap your ceiling and why you need a tie-break rule.
* Some tickets are genuinely unlabelable. `department == "other"` is the abstain bin,
  and lesson 04 treats a confident "other" as the model's way of abstaining.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Department = Literal["returns", "shipping", "billing", "other"]

DEPARTMENTS: list[str] = ["returns", "shipping", "billing", "other"]

# Instructions are part of the trained artifact: the strings below are the exact strings
# production must send. Change them and you must retrain. (Kev's skill makes the same point.)
QUESTIONS: dict[str, dict] = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "options": ["returns", "shipping", "billing", "other"],
    },
    "escalate": {
        "type": "noul",
        "instructions": "Does this ticket need a human to look at it right now?",
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        # Score questions answer with the level INDEX, not the word. This is not arbitrary:
        # "calm" tokenises to two tokens and "annoyed" to three, so a readout over the words
        # has no single position to read. "0"/"1"/"2" are one token each. The words live in
        # `legend` and are only ever shown to humans. Kev's API does exactly this.
        "options": ["0", "1", "2"],
        "legend": {"0": "calm", "1": "annoyed", "2": "angry"},
    },
}

# --------------------------------------------------------------------------------------
# Slot values
# --------------------------------------------------------------------------------------

ITEMS = [
    "running shoes", "winter jacket", "linen dress", "leather boots", "wool sweater",
    "denim jeans", "rain coat", "silk scarf", "flannel shirt", "canvas backpack",
    "yoga pants", "down vest", "cotton t-shirt", "suede loafers", "cashmere socks",
]

REASONS = [
    "they're too small", "they're too large", "the colour is wrong", "it arrived damaged",
    "the stitching came apart", "it doesn't match the photo", "I changed my mind",
    "it's not what I ordered", "the zipper is broken", "it shrank after one wash",
]

# The share of tickets whose label is deliberately wrong, standing in for the disagreement
# you get between two competent human labellers. Lesson 02 and 03 both measure what this
# does to the best score any model can reach.
LABEL_NOISE = 0.10


def noise_ceiling(n_classes: int, noise: float = LABEL_NOISE) -> float:
    """Best accuracy achievable by a model that has learned the true label function.

    A model that predicts the true label is still wrong whenever the recorded label was
    corrupted, which happens with probability `noise`; when it was corrupted the wrong
    label is correct with probability 1/n_classes. So:

        ceiling = (1 - noise) + noise / n_classes
    """
    return (1 - noise) + noise / n_classes


DIGITS = "0123456789"


def _order_id(rng: random.Random) -> str:
    return "#" + "".join(rng.choice(DIGITS) for _ in range(6))


# --------------------------------------------------------------------------------------
# Phrasing banks, keyed by (department, frustration level)
#
# Each template is a format string. `{item}`, `{order}`, `{reason}`, `{days}` are filled in.
# The register of the sentence is what encodes frustration: level 0 is polite and factual,
# 1 is clipped and shows effort already spent, 2 is heated and often threatens escalation.
# --------------------------------------------------------------------------------------

TEMPLATES: dict[str, dict[int, list[str]]] = {
    "shipping": {
        0: [
            "Hi, could you check on order {order}? The tracking hasn't updated in {days} days.",
            "Hello, my order {order} was expected on Tuesday but hasn't arrived yet.",
            "Good morning, I'd like an update on the delivery of {order}, please.",
            "Could you confirm when {order} will ship? It still shows as processing.",
            "My parcel for {order} appears to be stuck at the sorting facility.",
        ],
        1: [
            "Order {order} still hasn't arrived and it's been {days} days now. Where is it?",
            "I've checked the tracking three times today and {order} hasn't moved at all.",
            "This is the second time I've had to ask about {order}. It was due last week.",
            "The tracking for {order} says delivered but nothing came. Please look into this.",
            "I paid for express delivery on {order} and it's now {days} days late.",
        ],
        2: [
            "Where is my order {order}?! It's {days} days late and nobody has told me anything.",
            "This is completely unacceptable. {order} was meant to arrive last week and tracking is dead.",
            "I have now contacted you four times about {order} with no reply. I want answers today.",
            "Your courier left {order} at a strangers house and I cannot get anyone to respond.",
            "If {order} is not found today I am disputing the charge with my bank.",
        ],
    },
    "returns": {
        0: [
            "Hello, I'd like to return the {item} from order {order} because {reason}.",
            "Could you send me a return label for the {item}? {reason_cap}.",
            "I received the {item} today and would like to exchange it, {reason}.",
            "What is the process for returning the {item}? {reason_cap}.",
            "I'd like to start a return for the {item} in {order}, please.",
        ],
        1: [
            "I want to return the {item} from {order}. {reason_cap} and I shouldn't pay return shipping.",
            "The {item} arrived and {reason}. I've asked about this once already.",
            "Second request: please send a return label for the {item}, {reason}.",
            "I need to return the {item} but your portal keeps rejecting my order number.",
            "The {item} is faulty because {reason}, and I'd like a refund rather than a replacement.",
        ],
        2: [
            "The {item} arrived completely broken. This is the third faulty item I've had from you.",
            "I have asked twice for a return label for the {item} and been ignored both times.",
            "Your site will not let me return the {item} and there is no phone number anywhere.",
            "I am not paying to return a {item} that {reason}. Sort this out today.",
            "Refund me for the {item} now or I am opening a chargeback. {reason_cap}.",
        ],
    },
    "billing": {
        0: [
            "Hi, I noticed a charge of $ {amount} on my card for {order} that I don't recognise.",
            "Could you send an invoice for {order}? I can't find it in my account.",
            "I'd like to update the card on file for my subscription, please.",
            "Hello, I think I may have been charged twice for {order}.",
            "Can you confirm the total for {order}? It looks slightly different from my quote.",
        ],
        1: [
            "I've been charged twice for {order} and need the duplicate refunded.",
            "My invoice for {order} is $ {amount} higher than the price I agreed. Please explain.",
            "I was billed for express shipping on {order} but the order came standard.",
            "I cancelled the subscription last month and have just been charged again.",
            "The discount code I used on {order} never applied but you charged the full amount.",
        ],
        2: [
            "You have charged me {amount} dollars for an order I cancelled. I want it back today.",
            "This is the third time I have been overcharged. I am reporting this to my bank.",
            "I am being billed for a subscription I cancelled twice. This is fraud.",
            "Take the duplicate charge off {order} immediately or I will dispute every charge.",
            "I have emailed about this charge three times and each time been told to wait.",
        ],
    },
    "other": {
        0: [
            "Hello, do you ship to Portugal? I'd like to place an order.",
            "Is the {item} available in a size 12? Happy to wait if it's restocking.",
            "Could you tell me whether the {item} is machine washable?",
            "I'd like to open a wholesale account for my shop.",
            "Do you have a physical store I could visit?",
        ],
        1: [
            "I've emailed twice asking whether you ship to Portugal and had no reply.",
            "Your size guide contradicts the product page for the {item}.",
            "I want to close my account but there is no option anywhere on the site.",
            "The gift card I bought is not being accepted at checkout.",
            "Nobody has answered my question about the {item} sizing from last week.",
        ],
        2: [
            "I have been trying to close my account for a month and you keep ignoring me.",
            "Your gift card took my money and does not work. I want a person on the phone now.",
            "I am being passed between agents with no answer. This is a complaint.",
            "The {item} I was sent is not what the site describes and nobody will help.",
            "Every message I send gets an automated reply. I want an actual response.",
        ],
    },
}

# Frustration level implied by choosing a template at that level. Kept as an explicit
# function so lesson 02 can show you how the labels were derived and swap in your own.
AMOUNTS = ["12.99", "45.00", "89.50", "23.75", "156.00", "9.99", "67.20", "240.00"]


def _fill(template: str, rng: random.Random) -> str:
    reason = rng.choice(REASONS)
    return template.format(
        item=rng.choice(ITEMS),
        order=_order_id(rng),
        reason=reason,
        reason_cap=reason[0].upper() + reason[1:],
        days=rng.choice([3, 4, 5, 6, 7, 9, 11, 14]),
        amount=rng.choice(AMOUNTS),
    )


def _append_clause(text: str, clause: str) -> str:
    """Add a sentence without producing '?.' or a lowercase sentence start."""
    text = text.rstrip()
    clause = clause[0].upper() + clause[1:]
    if not clause.endswith("."):
        clause += "."
    return text + " " + clause if text.endswith((".", "?", "!")) else text + ". " + clause


# A fraction of tickets get a trailing secondary complaint. This is what makes the task
# non-trivial: the model must find the *primary* ask, not just any keyword present.
SECONDARY_CLAUSES = [
    "I was charged twice for it",
    "the tracking never updated",
    "I want to return the other item too",
    "please send an invoice",
    "the packaging was damaged",
    "I'd like a different colour as well",
    "I still have not had a refund",
    "the delivery estimate has changed twice",
]

# Escalation is decided by whether the ticket shows these behaviours, NOT by how cross the
# customer sounds. A calm ticket about fraud still needs a human; a furious ticket about a
# late parcel usually does not. The cues below are the observable signals.
ESCALATION_CLAUSES = [
    "If this is not resolved today I am opening a chargeback.",
    "I have already contacted you twice about this with no reply.",
    "I have reported this to my bank.",
    "I want to speak to a manager.",
    "This is a formal complaint.",
    "I have been ignored for over a week now.",
    "I am disputing every charge on this account.",
    "My solicitor has advised me to put this in writing.",
    "I will be contacting trading standards.",
    "I want my account closed today.",
]

# Words that mark a ticket as needing a human even when the register is polite. Used to
# derive the escalation label; the model has to learn to recognise them from the text.
HARD_CUES = [
    "chargeback", "dispute", "disputing", "solicitor", "trading standards",
    "formal complaint", "manager", "account closed", "reported this to my bank",
]


@dataclass
class Ticket:
    state: str
    department: str
    escalate: bool
    frustration: int
    # Which sentence template produced this ticket. This is metadata, not a label: it is
    # here so lesson 02 can measure leakage between splits. Two tickets built from the same
    # template are near-duplicates of each other, and if one is in train and the other in
    # development, your held-out score is inflated.
    template_id: int = -1

    def to_record(self) -> dict:
        return {
            "state": self.state,
            "answers": {
                "department": self.department,
                "escalate": self.escalate,
                "frustration": self.frustration,
            },
        }


def template_registry() -> list[tuple[int, str, int, str]]:
    """Flatten TEMPLATES into (template_id, department, frustration, text) rows.

    Lesson 02 uses this to build a template-held-out split: a stricter test than a random
    split, because no sentence pattern in the development set has been seen in training.
    """
    out: list[tuple[int, str, int, str]] = []
    tid = 0
    for dept, by_level in TEMPLATES.items():
        for level, templates in by_level.items():
            for text in templates:
                out.append((tid, dept, level, text))
                tid += 1
    return out


def generate_tickets(n: int, seed: int = 0) -> list[Ticket]:
    """Generate `n` tickets with a reproducible mix of departments and difficulties.

    The department mix is deliberately uneven (real ticket queues are), and the
    frustration mix is skewed toward calm, so that a model which always predicts the
    majority class looks deceptively good on accuracy. Lesson 04 fixes that.
    """
    rng = random.Random(seed)
    dept_weights = [0.30, 0.30, 0.28, 0.12]        # returns, shipping, billing, other
    frust_weights = [0.55, 0.30, 0.15]             # calm, annoyed, angry

    registry: dict[str, int] = {}
    for tid, dept, level, text in template_registry():
        registry.setdefault(text, tid)

    tickets: list[Ticket] = []
    for _ in range(n):
        dept = rng.choices(DEPARTMENTS, weights=dept_weights)[0]
        frust = rng.choices([0, 1, 2], weights=frust_weights)[0]
        template = rng.choice(TEMPLATES[dept][frust])
        text = _fill(template, rng)

        # A secondary complaint pulls attention to a different department. At 35% this is
        # the main source of genuine difficulty in the routing task: the model has to work
        # out which complaint is the primary ask rather than keying on any single noun.
        if rng.random() < 0.35:
            text = _append_clause(text, rng.choice(SECONDARY_CLAUSES))

        # Escalation is a separate judgement from frustration, and it is determined by
        # observable behaviour: has this customer threatened a chargeback, asked for a
        # manager, been ignored repeatedly? A calm fraud report escalates; a furious
        # complaint about a late parcel often does not.
        escalate = False
        if rng.random() < 0.28:
            text = _append_clause(text, rng.choice(ESCALATION_CLAUSES))
            escalate = True
        # A few tickets escalate for reasons the text does not spell out (a VIP customer, a
        # phone call nobody logged). Unlearnable by design — every real dataset has some.
        if not escalate and rng.random() < 0.06:
            escalate = True

        # Label noise. Real labelling has disagreement: two competent people reading the
        # same ticket choose differently maybe 8-12% of the time on a task like this. This
        # is the single most important number in the dataset, because it caps the accuracy
        # any model can reach and it is invisible unless you go looking for it.
        NOISE = 0.10
        if rng.random() < NOISE:
            dept = rng.choice(DEPARTMENTS)
        if rng.random() < NOISE:
            frust = rng.choice([0, 1, 2])
        if rng.random() < NOISE:
            escalate = not escalate

        tickets.append(Ticket(text, dept, escalate, frust, registry.get(template, -1)))
    return tickets


# --------------------------------------------------------------------------------------
# Splits
#
# Four splits, not two. This is the part people get wrong, so the tutorial repeats it:
#
#   train        - the only data the optimizer ever sees
#   calibration  - held out; fits the temperature. Optimizer never sees it either.
#   development  - the split you iterate against: read errors, change the data, retrain
#   test         - touched ONCE, at the end. If you tune on it, it is no longer a test set.
# --------------------------------------------------------------------------------------

SPLIT_FRACTIONS = {"train": 0.70, "calibration": 0.15, "development": 0.10, "test": 0.05}


def split_holdout_template(
    tickets: list[Ticket], holdout_frac: float = 0.3, seed: int = 0
) -> dict[str, list[Ticket]]:
    """A stricter split that keeps whole sentence templates on one side of the boundary.

    A random split leaks: with 20 tickets per template, a randomly chosen development
    ticket is almost certain to have a near-identical sibling in training. That inflates
    your held-out score, because the model can memorise the sentence and never learn the
    task.

    Here, entire templates go to either the "seen" side (train + calibration) or the
    "unseen" side (development + test). A model that scores well here has generalised to
    new phrasings, which is the only claim that survives contact with production.
    """
    rng = random.Random(seed)
    tids = sorted({t.template_id for t in tickets})
    rng.shuffle(tids)
    n_holdout = max(1, int(len(tids) * holdout_frac))
    holdout = set(tids[:n_holdout])

    seen = [t for t in tickets if t.template_id not in holdout]
    unseen = [t for t in tickets if t.template_id in holdout]

    seen_parts = split(seen, seed=seed)
    unseen_parts = split(unseen, seed=seed)
    return {
        "train": seen_parts["train"],
        "calibration": seen_parts["calibration"],
        "development": unseen_parts["train"] + unseen_parts["calibration"],
        "test": unseen_parts["development"] + unseen_parts["test"],
    }


def split(tickets: list[Ticket], seed: int = 0) -> dict[str, list[Ticket]]:
    shuffled = list(tickets)
    random.Random(seed).shuffle(shuffled)
    out: dict[str, list[Ticket]] = {}
    start = 0
    names = list(SPLIT_FRACTIONS)
    for i, name in enumerate(names):
        if i == len(names) - 1:
            out[name] = shuffled[start:]
        else:
            end = start + int(len(shuffled) * SPLIT_FRACTIONS[name])
            out[name] = shuffled[start:end]
            start = end
    return out


# --------------------------------------------------------------------------------------
# Prompt / completion rendering, for MLX LoRA training
#
# mlx-lm wants {"prompt": ..., "completion": ...} and `--mask-prompt` so the loss lands
# only on the answer. One row per (ticket, question) pair: the same ticket appears three
# times with three different questions. That is how one model answers three question types.
# --------------------------------------------------------------------------------------

PROMPT_TEMPLATE = """<ticket>
{state}
</ticket>

<question>{instructions}</question>
<options>{options}</options>
<answer>"""


def render_prompt(ticket: Ticket, question: str) -> tuple[str, str]:
    """Return (prompt, completion) for one ticket and one question."""
    q = QUESTIONS[question]
    options = q.get("options")
    options_text = " | ".join(options) if options else "true | false"
    prompt = PROMPT_TEMPLATE.format(
        state=ticket.state.strip(),
        instructions=q["instructions"],
        options=options_text,
    )
    if question == "escalate":
        completion = "true" if ticket.escalate else "false"
    elif question == "frustration":
        # score questions answer with the level index, matching Kev's convention
        completion = str(ticket.frustration)
    else:
        completion = ticket.department
    return prompt, completion


def to_mlx_jsonl(tickets: list[Ticket], path: Path, questions: list[str] | None = None) -> int:
    """Write mlx-lm prompt/completion JSONL. Returns the number of rows written."""
    questions = questions or list(QUESTIONS)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for t in tickets:
            for q in questions:
                prompt, completion = render_prompt(t, q)
                f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")
                n += 1
    return n


def to_records_jsonl(tickets: list[Ticket], path: Path, with_labels: bool = True) -> int:
    """Write the Jev/Kev-shaped record form: one row per ticket, all questions together."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for t in tickets:
            rec: dict = {"state": t.state, "questions": {}}
            if with_labels:
                rec["questions"] = t.to_record()["answers"]
            f.write(json.dumps(rec) + "\n")
    return len(tickets)


if __name__ == "__main__":
    ts = generate_tickets(1200, seed=7)
    parts = split(ts, seed=7)
    for name, rows in parts.items():
        print(f"{name:12s} {len(rows):5d} rows")
    print()
    print("sample ticket:")
    print(" ", parts["train"][0].state)
    print("  labels:", parts["train"][0].to_record()["answers"])
