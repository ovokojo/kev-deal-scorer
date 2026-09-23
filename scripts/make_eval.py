"""Generate the synthetic deal-judging dataset: data/eval/{train,dev,test}.jsonl.

THE LABEL RULE (the whole answer key, in one place):

  ratio = asking price / 90-day typical price
  verdict = great_deal if ratio <= 0.80, bad_deal if ratio >= 1.15, else fair_price
  scam    = ratio < 0.40 AND at least one seller signal (deal.scam_signals):
            feedback < 90%, fewer than 50 ratings, or wire transfer / gift card /
            Telegram / WhatsApp / no returns / bulk / liquidation wording

Listings are generated from 40 product templates, and the split is BY TEMPLATE:
no product in test appears in train or dev. The 25 storefront listings in
data/products.json are separate products again and are never trained on.

    python3 scripts/make_eval.py            # writes ~2,400 train / 200 dev / 400 test
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from deal import scam_signals  # noqa: E402

# name, category, typical 2026 street price, MSRP
TEMPLATES = [
    ("Apple Watch Series 10 45mm GPS", "Wearables", 329, 429),
    ("Garmin Forerunner 265", "Wearables", 349, 449),
    ("Fitbit Charge 6", "Wearables", 129, 159),
    ("Samsung Galaxy Watch 7 44mm", "Wearables", 229, 329),
    ("Sonos Era 100 Speaker", "Audio", 199, 249),
    ("JBL Flip 6 Bluetooth Speaker", "Audio", 99, 129),
    ("Sennheiser Momentum 4 Wireless", "Audio", 249, 379),
    ("Beats Studio Pro Headphones", "Audio", 199, 349),
    ("Samsung Galaxy S25 128GB Unlocked", "Phones", 699, 799),
    ("iPhone 16 128GB Unlocked", "Phones", 729, 829),
    ("OnePlus 13 256GB", "Phones", 799, 899),
    ("Motorola Edge 2025", "Phones", 399, 549),
    ("Dell XPS 14 Core Ultra 7, 16GB/512GB", "Computers", 1299, 1699),
    ("Lenovo ThinkPad X1 Carbon Gen 12", "Computers", 1449, 2199),
    ("Microsoft Surface Pro 11 Snapdragon X Plus", "Computers", 899, 999),
    ("Mac mini M4 16GB/256GB", "Computers", 549, 599),
    ("Kindle Paperwhite 16GB", "Computers", 139, 159),
    ("Samsung T7 Shield 2TB Portable SSD", "Computers", 149, 229),
    ("Shark Navigator Lift-Away Vacuum", "Home", 179, 229),
    ("iRobot Roomba Combo j9+", "Home", 699, 1399),
    ("Nest Learning Thermostat 4th Gen", "Home", 249, 279),
    ("Ring Video Doorbell Pro 2", "Home", 179, 249),
    ("Philips Hue White & Color Starter Kit", "Home", 159, 199),
    ("Levoit Core 400S Air Purifier", "Home", 179, 219),
    ("Vitamix E310 Explorian Blender", "Kitchen", 299, 349),
    ("Instant Pot Duo Plus 6-Qt", "Kitchen", 99, 139),
    ("Nespresso Vertuo Next", "Kitchen", 129, 179),
    ("Le Creuset 5.5-Qt Dutch Oven", "Kitchen", 399, 460),
    ("Cuisinart 14-Cup Food Processor", "Kitchen", 199, 249),
    ("Secretlab Titan Evo Gaming Chair", "Furniture", 499, 549),
    ("Uplift V2 Standing Desk 60x30", "Furniture", 699, 799),
    ("Branch Ergonomic Chair", "Furniture", 449, 499),
    ("Xbox Series X 1TB", "Gaming", 499, 599),
    ("Steam Deck OLED 512GB", "Gaming", 549, 549),
    ("Meta Quest 3 512GB", "Gaming", 499, 649),
    ("Nintendo Switch 2", "Gaming", 449, 449),
    ("Samsung 65\" QN90D Neo QLED TV", "TVs", 1399, 2599),
    ("TCL 55\" QM7 Mini-LED TV", "TVs", 599, 999),
    ("Canon EOS R50 Mirrorless Kit", "Cameras", 679, 799),
    ("GoPro HERO13 Black", "Cameras", 349, 399),
]

GOOD_SELLERS = ["TechOutlet", "PrimeGoods", "DirectDeals", "CityElectronics", "MainStreet Supply",
                "ReNew Store", "Harbor Retail", "OpenBox Pros", "NorthStar Shop", "Bright Buy"]
SHADY_SELLERS = ["deals_king_77", "bestprice_outlet_x", "mega_liquidators", "shop4less_2026",
                 "wholesale_direct_9", "super_saver_hub"]
CONDITIONS = ["New", "New, sealed", "Open box", "Used, like new", "Used, good",
              "Manufacturer refurbished"]
LEGIT_BLURBS = [
    "Includes all original accessories. Ships free within 2 business days.",
    "Tested and fully working. 30-day returns accepted.",
    "Retail box, full manufacturer warranty. Tracked shipping.",
    "Minor cosmetic wear, fully functional. Returns within 14 days.",
    "Sealed in original packaging. Ships from a US warehouse.",
]
CLEARANCE_BLURBS = [
    "Store closing clearance, open box, pick up in store only.",
    "Display model, fully working, final markdown before the new model.",
]
SCAM_BLURBS = [
    "100% original, bulk wholesale lot. No returns. Wire transfer preferred.",
    "Brand new sealed units. Payment by gift card only. Ships from overseas.",
    "Warehouse liquidation. Message me on Telegram before buying.",
    "Contact on WhatsApp for fast deal. No returns, final sale.",
]


def verdict_for(ratio):
    return "great_deal" if ratio <= 0.80 else "bad_deal" if ratio >= 1.15 else "fair_price"


def make_listing(rng, tpl, idx):
    name, cat, typical, msrp = tpl
    r = rng.random()
    if r < 0.15:
        ratio = rng.uniform(0.15, 0.39)            # too cheap: scam or clearance
    elif r < 0.43:
        ratio = rng.uniform(0.45, 0.79)
    elif r < 0.71:
        ratio = rng.uniform(0.82, 1.12)
    elif r < 0.93:
        ratio = rng.uniform(1.16, 1.8)
    else:
        ratio = rng.uniform(0.76, 0.84) if rng.random() < 0.5 else rng.uniform(1.11, 1.19)

    price = max(5, round(typical * ratio))
    ratio = price / typical

    shady = (ratio < 0.4 and rng.random() < 0.7) or (ratio >= 0.4 and rng.random() < 0.12)
    if shady:
        seller = rng.choice(SHADY_SELLERS)
        feedback = round(rng.uniform(40, 89), 1)
        ratings = rng.randint(2, 45)
        blurb = rng.choice(SCAM_BLURBS) if ratio < 0.4 or rng.random() < 0.5 else rng.choice(LEGIT_BLURBS)
    else:
        seller = rng.choice(GOOD_SELLERS)
        feedback = round(rng.uniform(95, 99.9), 1)
        ratings = rng.randint(300, 20000)
        blurb = rng.choice(CLEARANCE_BLURBS) if ratio < 0.4 else rng.choice(LEGIT_BLURBS)

    p = {
        "id": f"e{idx}",
        "title": name,
        "category": cat,
        "price": price,
        "list_price": msrp,
        "typical_price": typical,
        "low_90d": round(typical * rng.uniform(0.85, 0.95)),
        "high_90d": round(typical * rng.uniform(1.03, 1.12)),
        "condition": rng.choice(CONDITIONS),
        "seller": seller,
        "seller_feedback": feedback,
        "seller_ratings": ratings,
        "blurb": blurb,
    }
    p["label"] = {"verdict": verdict_for(ratio), "scam": ratio < 0.4 and bool(scam_signals(p))}
    return p


def main():
    rng = random.Random(7)
    templates = TEMPLATES[:]
    rng.shuffle(templates)
    splits = {"test": (templates[:8], 50), "dev": (templates[8:12], 50), "train": (templates[12:], 86)}
    out_dir = ROOT / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for split, (tpls, per) in splits.items():
        rows = []
        for tpl in tpls:
            for _ in range(per):
                rows.append(make_listing(rng, tpl, idx))
                idx += 1
        rng.shuffle(rows)
        with open(out_dir / f"{split}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        n_scam = sum(r["label"]["scam"] for r in rows)
        counts = {v: sum(r["label"]["verdict"] == v for r in rows)
                  for v in ("great_deal", "fair_price", "bad_deal")}
        print(f"{split:5s} {len(rows):5d} rows, {len(tpls)} templates, {counts}, scams={n_scam}")


if __name__ == "__main__":
    main()
