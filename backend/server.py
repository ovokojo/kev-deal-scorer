#!/usr/bin/env python3
"""Deal Scorer backend: serves the frontend and /api/* endpoints. Zero dependencies."""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import deal

ROOT = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.normpath(os.path.join(ROOT, "..", "frontend"))
DATA = os.path.normpath(os.path.join(ROOT, "..", "data", "products.json"))

with open(DATA) as f:
    PRODUCTS = json.load(f)
BY_ID = {p["id"]: p for p in PRODUCTS}

CONTENT_TYPES = {
    "html": "text/html", "css": "text/css", "js": "application/javascript",
    "jpg": "image/jpeg", "png": "image/png", "svg": "image/svg+xml", "json": "application/json",
}
# the eval labels stay server-side; the storefront never sees the answer key
PUBLIC = [{k: v for k, v in p.items() if k != "label"} for p in PRODUCTS]
try:
    with open(os.path.normpath(os.path.join(ROOT, "..", "data", "photo-credits.json"))) as f:
        CREDITS = json.load(f)
except FileNotFoundError:
    CREDITS = {}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            engine = deal.detect_engine()
            return self._send(200, json.dumps({
                "engine": engine["name"] if engine else "demo",
                "run": engine["run"] if engine else None,
                "batch": bool(engine and engine["batch"]),
                "products": len(PRODUCTS),
            }))
        if path == "/api/products":
            return self._send(200, json.dumps(PUBLIC))
        if path == "/api/credits":
            return self._send(200, json.dumps(CREDITS))

        rel = "index.html" if path == "/" else path.lstrip("/")
        fpath = os.path.normpath(os.path.join(FRONTEND, rel))
        if fpath.startswith(FRONTEND) and os.path.isfile(fpath):
            ctype = CONTENT_TYPES.get(fpath.rsplit(".", 1)[-1], "application/octet-stream")
            with open(fpath, "rb") as f:
                return self._send(200, f.read(), ctype)
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/api/score":
            return self._send(404, "not found", "text/plain")
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            return self._send(400, json.dumps({"error": "invalid JSON"}))
        # {"id": "p1"} scores one listing; {"ids": [...]} scores several in one call
        ids = payload.get("ids") or [payload.get("id")]
        products = [BY_ID.get(i) for i in ids]
        if not ids or None in products:
            return self._send(404, json.dumps({"error": "unknown product id"}))
        try:
            results = deal.score_products(products)
        except Exception as e:  # never crash the demo on a model hiccup
            return self._send(502, json.dumps({"error": f"scoring failed: {e}"}))
        body = {"results": results} if "ids" in payload else results[0]
        return self._send(200, json.dumps(body))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    engine = deal.detect_engine()
    where = f"{engine['name']} · {engine['run']}" if engine else "no engine — demo mode"
    print(f"Deal scorer on http://127.0.0.1:{port}  ({where})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
