"""Deal scoring via Kev (TypeSafe System One protocol), with a demo-mode fallback.

When a Kev server is reachable at KEV_URL, listings are scored by the model.
Otherwise a transparent heuristic is used so the visual PoC works with zero setup.
"""
import json
import time
import urllib.request
import urllib.error

KEV_URL = "http://127.0.0.1:8009/v1/systemone"
KEV_MODELS_URL = "http://127.0.0.1:8009/v1/models"
KEV_MODEL = "kev-latest"
TIMEOUT_S = 10

VALUE_LABELS = ["Terrible value", "Poor value", "Fair", "Good value", "Exceptional steal"]


def product_state(p):
    """State sent to the model. Note: market_price is deliberately excluded --
    the model should judge from its own knowledge, like a real shopper would."""
    return "\n".join([
        f"Listing: {p['title']}",
        f"Asking price: ${p['price']}",
        f"Condition: {p.get('condition', 'Not specified')}",
        f"Category: {p.get('category', '')}",
        f"Seller: {p.get('seller', '')}",
        f"Details: {p.get('blurb', '')}",
    ])


def build_questions():
    return {
        "verdict": {
            "type": "choice",
            "instructions": (
                "Compared to the typical market price for this item, is this listing "
                "a great deal, a fair price, or a bad deal?"
            ),
            "criteria": {
                "great_deal": "Priced clearly below typical market value. A buyer should act fast.",
                "fair_price": "Priced around typical market value. Neither a steal nor a rip-off.",
                "bad_deal": "Priced clearly above typical market value, or poor value for what you get. Skip it.",
            },
        },
        "value": {
            "type": "score",
            "instructions": "Rate the overall value for money of this listing.",
            "criteria": VALUE_LABELS,
        },
        "red_flags": {
            "type": "noul",
            "instructions": (
                "Does this listing show red flags: suspiciously vague details, "
                "a price too good to be true, or signs of a scam?"
            ),
        },
    }


def kev_available():
    try:
        req = urllib.request.Request(KEV_MODELS_URL)
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def kev_run_name():
    """Best-effort name of the loaded Kev checkpoint, e.g. 'jaredpalmer/kev-4b'."""
    try:
        with urllib.request.urlopen(KEV_MODELS_URL, timeout=2) as r:
            info = json.loads(r.read().decode())
        models = info.get("models") if isinstance(info, dict) else None
        first = models[0] if isinstance(models, list) and models else {}
        return first.get("run") or first.get("id") or None
    except Exception:
        return None


def score_with_kev(product):
    payload = {
        "state": product_state(product),
        "model": KEV_MODEL,
        "questions": build_questions(),
    }
    req = urllib.request.Request(
        KEV_URL, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        body = json.loads(r.read().decode())
    round_trip_ms = int((time.time() - t0) * 1000)

    answers = body["answers"]
    verdict = answers["verdict"]
    value = answers["value"]
    red = answers["red_flags"]
    score_idx = max(0, min(4, int(round(value["score"]))))

    return {
        "mode": "kev",
        "latency_ms": round_trip_ms,
        "verdict": verdict["choice"],
        "probabilities": verdict["probabilities"],
        "confidence": round(verdict.get("confidence", 0), 2),
        "value_score": round(value["score"], 2),
        "value_label": VALUE_LABELS[score_idx],
        "red_flag_prob": round(red["noul"], 2),
        # full request/response trace for the educational lab page
        "trace": {
            "mode": "kev",
            "request": {
                "url": KEV_URL,
                "model": KEV_MODEL,
                "state": payload["state"],
                "questions": payload["questions"],
            },
            "response": {"answers": answers, "usage": body.get("usage", {})},
            "timing": {
                "model_ms": body.get("latency_ms"),
                "round_trip_ms": round_trip_ms,
            },
        },
    }


def score_demo(product):
    """Transparent heuristic fallback so the UI works with no model installed."""
    t0 = time.time()
    ratio = product["price"] / max(product["market_price"], 1)

    if ratio <= 0.70:
        verdict = "great_deal"
        probs = {"great_deal": 0.82, "fair_price": 0.15, "bad_deal": 0.03}
    elif ratio <= 0.90:
        verdict = "great_deal"
        probs = {"great_deal": 0.58, "fair_price": 0.34, "bad_deal": 0.08}
    elif ratio <= 1.10:
        verdict = "fair_price"
        probs = {"great_deal": 0.18, "fair_price": 0.66, "bad_deal": 0.16}
    elif ratio <= 1.30:
        verdict = "bad_deal"
        probs = {"great_deal": 0.06, "fair_price": 0.32, "bad_deal": 0.62}
    else:
        verdict = "bad_deal"
        probs = {"great_deal": 0.02, "fair_price": 0.13, "bad_deal": 0.85}

    value_score = round(max(0.0, min(4.0, (1.3 - ratio) * 5)), 2)
    red_flag_prob = 0.85 if ratio < 0.35 else 0.05
    confidence = round(max(probs.values()), 2)
    elapsed_ms = int((time.time() - t0) * 1000)

    # same answer shape Kev returns, so the lab page can treat both alike
    answers = {
        "verdict": {
            "type": "choice", "choice": verdict,
            "confidence": confidence, "probabilities": probs,
        },
        "value": {
            "type": "score", "score": value_score,
            "legend": {str(i): l for i, l in enumerate(VALUE_LABELS)},
        },
        "red_flags": {"type": "noul", "noul": red_flag_prob},
    }

    return {
        "mode": "demo",
        "latency_ms": elapsed_ms,
        "verdict": verdict,
        "probabilities": probs,
        "confidence": confidence,
        "value_score": value_score,
        "value_label": VALUE_LABELS[max(0, min(4, int(round(value_score))))],
        "red_flag_prob": red_flag_prob,
        "trace": {
            "mode": "demo",
            "request": {
                "url": None,
                "model": "heuristic (no Kev server on :8009)",
                "state": product_state(product),
                "questions": build_questions(),
            },
            "response": {
                "answers": answers,
                "usage": {},
                "heuristic": {
                    "ratio": round(ratio, 3),
                    "rule": f"asking {ratio:.2f}x market price → {verdict}",
                    "note": "price compared against market_price — no model involved",
                },
            },
            "timing": {"model_ms": elapsed_ms, "round_trip_ms": elapsed_ms},
        },
    }


def score_product(product):
    if kev_available():
        try:
            return score_with_kev(product)
        except Exception:
            pass  # fall through to demo mode
    return score_demo(product)
