"""Small display helpers shared by the notebooks. Nothing here is essential to the
machine learning — it exists so the notebooks print readable tables instead of raw dicts.
"""

from __future__ import annotations

import json
from pathlib import Path


def gb(n_bytes: float) -> float:
    return n_bytes / 1024**3


def memory_table(rows: list[tuple[str, int, int]], n_params: float) -> "object":
    """rows: list of (mode, bytes, note). Returns a DataFrame scaled to GB."""
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "mode": mode,
                "GB needed": round(gb(nbytes), 2),
                "fits in 24 GB?": "yes" if gb(nbytes) < 24 else "NO",
                "note": note,
            }
            for mode, nbytes, note in rows
        ]
    )


def budget(n_params: float) -> list[tuple[str, int, int]]:
    """Rough memory cost of the standard training modes for a dense transformer.

    These are rules of thumb, not exact: they ignore activations (which depend on batch
    size and sequence length) and fragmentation. Treat the output as a lower bound.
    """
    p = n_params
    return [
        ("inference, bf16", int(2 * p), "weights only"),
        ("inference, 4-bit", int(0.5 * p), "quantised weights only"),
        ("LoRA training, bf16 base", int(2 * p + 0.1 * p + 4 * 0.01 * p), "frozen base + small adapter + Adam on the adapter"),
        ("QLoRA training, 4-bit base", int(0.5 * p + 4 * 0.01 * p), "quantised base + adapter"),
        ("full fine-tune, bf16", int(2 * p + 2 * p + 8 * p), "weights + grads + Adam moments (fp32)"),
    ]


def markdown_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def truncate(text: str, n: int = 120) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def show_jsonl(path: str | Path, n: int = 3) -> None:
    """Print the first n rows of a JSONL file, pretty-printed."""
    path = Path(path)
    rows = path.read_text().splitlines()
    print(f"{path}  ({len(rows)} rows)")
    for line in rows[:n]:
        print(json.dumps(json.loads(line), indent=2)[:1200])
        print("-" * 60)
