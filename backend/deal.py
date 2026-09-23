"""Deal scoring over the TypeSafe System One protocol, with a demo-mode fallback.

Two local engines speak the same protocol, and the first one that answers is used:

  laya  — laya-server/ on :8010, a 421M ModernBERT decision encoder on MLX (~35 ms/listing batched)
  kev   — kev.serve on :8009, a Qwen decision model (0.8B on MPS: 0.1–0.7 s per listing)

With neither up, a transparent heuristic scores listings so the UI works with zero setup.
A live engine that errors is reported as an error, never quietly replaced by the heuristic.
"""
import json
import os
import re
import time
import urllib.request

ENGINES = [
    # name, base url, request timeout (s, per listing for batch engines), accepts a batch of states
    ("laya", os.environ.get("LAYA_URL", "http://127.0.0.1:8010"), 0.5, True),
    ("kev", os.environ.get("KEV_URL", "http://127.0.0.1:8009"), 20, False),
]
MODEL = "kev-latest"      # Kev ignores unknown names; the Laya sidecar ignores this field
DETECT_TTL_S = 10

VALUE_LABELS = ["Terrible value", "Poor value", "Fair", "Good value", "Exceptional steal"]
SCAM_TERMS = re.compile(r"wire transfer|gift card|telegram|whatsapp|no returns|bulk|liquidation", re.I)


def product_state(p):
    """State sent to the model. It includes the 90-day price history a deal tracker
    would show, so the model compares against a reference instead of recalling
    2026 street prices from memory."""
    return "\n".join([
        f"Listing: {p['title']}",
        f"Asking price: ${p['price']:,}",
        f"List price (MSRP): ${p['list_price']:,}",
        f"Price history (90 days): typical ${p['typical_price']:,}, "
        f"low ${p['low_90d']:,}, high ${p['high_90d']:,}",
        f"Condition: {p.get('condition', 'Not specified')}",
        f"Category: {p.get('category', '')}",
        f"Seller: {p['seller']} ({p['seller_feedback']}% positive, {p['seller_ratings']:,} ratings)",
        f"Details: {p.get('blurb', '')}",
    ])


def build_questions():
    return {
        "verdict": {
            "type": "choice",
            "instructions": (
                "Compare the asking price with the typical price in the price history. "
                "Is this listing a great deal, a fair price, or a bad deal?"
            ),
            "criteria": {
                "great_deal": "Asking price is clearly below the typical price (20% or more under). A buyer should act fast.",
                "fair_price": "Asking price is close to the typical price. Neither a steal nor a rip-off.",
                "bad_deal": "Asking price is clearly above the typical price (15% or more over). Skip it.",
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
                "Is this listing likely a scam? Signs: a price far below the typical price, "
                "a seller with low feedback or very few ratings, payment by wire transfer, "
                "gift card or off-platform chat, no returns, or bulk-lot wording."
            ),
        },
    }


# ---------- engine detection ----------

_detected = {"at": 0.0, "engine": None}


def _get_json(url, timeout):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return json.loads(r.read().decode())


def detect_engine(force=False):
    """First engine whose /v1/models answers, cached for DETECT_TTL_S seconds."""
    now = time.time()
    if not force and now - _detected["at"] < DETECT_TTL_S:
        return _detected["engine"]
    engine = None
    for name, base, timeout, batch in ENGINES:
        try:
            info = _get_json(base + "/v1/models", timeout=1.5)
        except Exception:
            continue
        models = info.get("models") if isinstance(info, dict) else None
        first = models[0] if isinstance(models, list) and models else {}
        engine = {
            "name": name, "url": base + "/v1/systemone", "timeout": timeout, "batch": batch,
            "run": first.get("run") or first.get("id") or name,
        }
        break
    _detected.update(at=now, engine=engine)
    return engine


# ---------- scoring ----------

def _post(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _result(mode, engine_name, run, state, answers, usage, model_ms, round_trip_ms, extra=None):
    """One shape for every engine. `confidence` is always the top verdict probability,
    because each engine defines its own `confidence` differently (Kev: rescaled top
    probability, Laya: 1 - normalised entropy). The raw values stay in the trace."""
    verdict, value, red = answers["verdict"], answers["value"], answers["red_flags"]
    probs = verdict["probabilities"]
    score_idx = max(0, min(4, int(round(value["score"]))))
    response = {"answers": answers, "usage": usage}
    if extra:
        response.update(extra)
    return {
        "mode": mode,
        "engine": engine_name,
        "run": run,
        "latency_ms": round_trip_ms,
        "verdict": verdict["choice"],
        "probabilities": probs,
        "confidence": round(max(probs.values()), 2),
        "value_score": round(value["score"], 2),
        "value_label": VALUE_LABELS[score_idx],
        "red_flag_prob": round(red["noul"], 2),
        # full request/response trace for the educational lab page
        "trace": {
            "mode": mode,
            "engine": engine_name,
            "request": {"model": run, "state": state, "questions": build_questions()},
            "response": response,
            "timing": {"model_ms": model_ms, "round_trip_ms": round_trip_ms},
        },
    }


def _error(product, engine, message):
    return {"id": product["id"], "mode": "error", "engine": engine["name"],
            "run": engine["run"], "error": message}


def score_with_engine(engine, products):
    """Score listings with a live engine. Batch engines get one request for all of them."""
    states = [product_state(p) for p in products]
    questions = build_questions()
    if engine["batch"]:
        t0 = time.time()
        body = _post(engine["url"], {"model": MODEL, "states": states, "questions": questions},
                     3 + engine["timeout"] * len(states))
        rt = int((time.time() - t0) * 1000)
        per = body["results"]
        batch_info = {"batch_size": len(products), "batch_model_ms": body.get("latency_ms"),
                      "batch_round_trip_ms": rt}
        out = []
        for p, s, r in zip(products, states, per):
            res = _result("live", engine["name"], engine["run"], s, r["answers"], r.get("usage", {}),
                          r.get("latency_ms"), rt, extra={"batch": batch_info})
            res["id"] = p["id"]
            out.append(res)
        return out

    out = []
    for p, s in zip(products, states):
        t0 = time.time()
        try:
            body = _post(engine["url"], {"model": MODEL, "state": s, "questions": questions},
                         engine["timeout"])
        except Exception as e:
            out.append(_error(p, engine, f"{engine['name']} request failed: {e}"))
            continue
        rt = int((time.time() - t0) * 1000)
        res = _result("live", engine["name"], engine["run"], s, body["answers"],
                      body.get("usage", {}), body.get("latency_ms"), rt)
        res["id"] = p["id"]
        out.append(res)
    return out


def scam_signals(p):
    """Seller-side signals the heuristic (and the eval labels) treat as scam evidence."""
    signals = []
    if p["seller_feedback"] < 90:
        signals.append(f"seller feedback {p['seller_feedback']}%")
    if p["seller_ratings"] < 50:
        signals.append(f"only {p['seller_ratings']} seller ratings")
    m = SCAM_TERMS.search(p.get("blurb", "") + " " + p["title"])
    if m:
        signals.append(f'"{m.group(0).lower()}"')
    return signals


def _softmax(z):
    import math
    m = max(z.values())
    e = {k: math.exp(v - m) for k, v in z.items()}
    s = sum(e.values())
    return {k: round(v / s, 2) for k, v in e.items()}


def score_demo(product):
    """Transparent heuristic: asking price vs the 90-day typical price, plus seller signals."""
    t0 = time.time()
    ratio = product["price"] / max(product["typical_price"], 1)
    probs = _softmax({
        "great_deal": (0.80 - ratio) * 12,
        "fair_price": 0.0,
        "bad_deal": (ratio - 1.15) * 12,
    })
    verdict = max(probs, key=probs.get)
    signals = scam_signals(product)
    scam = ratio < 0.4 and bool(signals)
    red_flag_prob = 0.9 if scam else 0.35 if ratio < 0.4 else 0.04
    value_score = round(max(0.0, min(4.0, 2 + (1 - ratio) * 6)), 2)
    elapsed_ms = int((time.time() - t0) * 1000)

    # same answer shape the live engines return, so the lab page treats all alike
    answers = {
        "verdict": {"type": "choice", "choice": verdict,
                    "confidence": max(probs.values()), "probabilities": probs},
        "value": {"type": "score", "score": value_score,
                  "legend": {str(i): l for i, l in enumerate(VALUE_LABELS)}},
        "red_flags": {"type": "noul", "noul": red_flag_prob},
    }
    res = _result(
        "demo", "demo", "heuristic", product_state(product), answers, {}, elapsed_ms, elapsed_ms,
        extra={"heuristic": {
            "ratio": round(ratio, 3),
            "rule": f"asking {ratio:.2f}× the 90-day typical price → {verdict}",
            "scam_signals": signals,
            "note": "no model involved — price ratio and seller signals only",
        }},
    )
    res["id"] = product["id"]
    return res


def score_products(products):
    engine = detect_engine()
    if engine is None:
        return [score_demo(p) for p in products]
    try:
        return score_with_engine(engine, products)
    except Exception as e:
        detect_engine(force=True)   # the engine may have gone away; re-probe next time
        return [_error(p, engine, f"{engine['name']} request failed: {e}") for p in products]
