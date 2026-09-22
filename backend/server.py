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

CONTENT_TYPES = {"html": "text/html", "css": "text/css", "js": "application/javascript"}


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
            kev = deal.kev_available()
            return self._send(200, json.dumps({
                "kev": kev,
                "run": deal.kev_run_name() if kev else None,
                "products": len(PRODUCTS),
            }))
        if path == "/api/products":
            return self._send(200, json.dumps(PRODUCTS))

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
        product = BY_ID.get(payload.get("id"))
        if not product:
            return self._send(404, json.dumps({"error": "unknown product id"}))
        try:
            result = deal.score_product(product)
        except Exception as e:  # never crash the demo on a model hiccup
            return self._send(502, json.dumps({"error": f"scoring failed: {e}"}))
        result["id"] = product["id"]
        return self._send(200, json.dumps(result))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    kev = deal.kev_available()
    print(f"Deal scorer on http://127.0.0.1:{port}  (kev: {'up' if kev else 'down — demo mode'})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
