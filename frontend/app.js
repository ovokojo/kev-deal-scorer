/* app.js — the shop: engine strip, hero tally, product drawer, cart.
   The grid, scoring queue, sort and filters live in catalog.js. */

const $ = (id) => document.getElementById(id);
const { esc, money, pct } = Catalog;

const shop = {
  health: null,
  open: null,          // product id in the detail drawer
  cart: loadCart(),
};

boot();

async function boot() {
  let health = { engine: "demo" };
  try { health = await fetch("/api/health").then((r) => r.json()); } catch (e) { /* offline */ }
  shop.health = health;
  renderAnnounce(health);

  await Catalog.init({
    health,
    els: { chips: $("chips"), search: $("search"), results: $("results-line"), sort: $("sort"), hide: $("hide-bad") },
    onCardClick: (p) => openProduct(p.id),
    quickAdd: (p, btn) => { addToCart(p.id); btn.textContent = "Added ✓"; btn.classList.add("added"); },
    // eslint-disable-next-line no-unused-vars
    onResult: (p) => { if (shop.open === p.id) renderProduct(p.id); if (shop.cart.includes(p.id)) renderCart(); },
    onTally: renderTally,
  });
  renderCartCount();
  renderCredits();
  bindDrawers();
}

/* ---------- engine strip + tally ---------- */

function engineName(h) {
  const run = (h.run || "").split("/").pop();
  if (h.engine === "laya") return `Laya · ${run.replace(/-mlx$/, "")} on MLX`;
  if (h.engine === "kev") return `Kev · ${run} on Apple GPU`;
  return "heuristic";
}

function renderAnnounce(h) {
  const el = $("announce"), text = $("announce-text");
  el.classList.remove("live", "demo");
  if (h.engine === "demo") {
    el.classList.add("demo");
    text.innerHTML = `<b>Demo mode</b> — no model is running, so a price-ratio heuristic stands in.
      <span class="mono">./run.sh</span> prints how to start one.`;
  } else {
    el.classList.add("live");
    text.innerHTML = `<b>Model live</b> · <span class="mono">${esc(engineName(h))}</span> ·
      ${h.batch ? "scores everything on screen in one pass" : "scores one listing at a time"}`;
  }
}

function renderTally(t) {
  $("t-great").textContent = t.great_deal;
  $("t-fair").textContent = t.fair_price;
  $("t-bad").textContent = t.bad_deal;
  $("t-scam").textContent = t.scam;
  const live = Object.values(Catalog.results).filter((r) => r.mode === "live");
  const ms = live.map((r) => r.trace.timing.model_ms).filter((x) => x != null);
  const avg = ms.length ? Math.round(ms.reduce((a, b) => a + b, 0) / ms.length) : null;
  const extra = [
    t.unsure ? `${t.unsure} not sure` : "",
    t.error ? `${t.error} failed` : "",
    avg != null ? `avg <span class="mono">${avg} ms</span> of model time per listing` : "",
  ].filter(Boolean).join(" · ");
  $("tally-note").innerHTML = t.scored
    ? `<span class="mono">${t.scored}/${t.total}</span> listings checked${extra ? " · " + extra : ""}`
    : "Waiting for the first listing…";
}

/* ---------- product drawer ---------- */

function openProduct(id) {
  shop.open = id;
  renderProduct(id);
  openDrawer("pd");
}

function renderProduct(id) {
  const p = Catalog.product(id);
  const r = Catalog.results[id];
  const d = Catalog.display(r);
  const vsTypical = (p.price - p.typical_price) / p.typical_price;
  const fbLow = p.seller_feedback < 90;

  $("pd-body").innerHTML = `
    <div class="pd-media"><img src="${esc(p.image)}" alt="${esc(p.title)}"></div>
    <h3 class="pd-title">${esc(p.title)}</h3>
    <div class="pd-sub">
      <span>★ ${p.rating.toFixed(1)} · ${p.reviews.toLocaleString()} reviews</span>
      <span>${esc(p.condition)}</span>
      <span>${esc(p.category)}</span>
    </div>
    <div class="pd-price">
      <span class="price">${money(p.price)}</span>
      ${p.list_price > p.price ? `<s class="list">${money(p.list_price)} MSRP</s>` : ""}
    </div>
    <p class="pd-blurb">${esc(p.blurb)}</p>

    <div class="section-label">${Catalog.judge()}’s take</div>
    ${renderTake(p, r, d)}

    <div class="section-label">Price history · last 90 days</div>
    ${renderRange(p, d)}
    <div class="range-legend">
      <span>typical <span class="mono">${money(p.typical_price)}</span></span>
      <span>this listing is <b>${Math.abs(Math.round(vsTypical * 100))}% ${vsTypical < 0 ? "under" : "over"}</b> typical</span>
    </div>

    <div class="section-label">Seller</div>
    <div class="seller-box">
      <div class="avatar">${esc(p.seller[0].toUpperCase())}</div>
      <div><b>${esc(p.seller)}</b><span>${p.seller_ratings.toLocaleString()} ratings</span></div>
      <span class="fb ${fbLow ? "low" : ""}">${p.seller_feedback}% positive</span>
    </div>`;

  const inCart = shop.cart.includes(id);
  $("pd-foot").innerHTML = inCart
    ? `<div class="btn-row"><button class="btn secondary" id="pd-view-cart">In your cart — view cart</button></div>`
    : d.cls === "scam"
      ? `<div class="btn-row"><button class="btn secondary" data-close>Keep browsing</button>
         <button class="btn warn" id="pd-add">Add anyway</button></div>`
      : `<button class="btn" id="pd-add">Add to cart · ${money(p.price)}</button>`;
  const add = $("pd-add");
  if (add) add.onclick = () => { addToCart(id); renderProduct(id); };
  const view = $("pd-view-cart");
  if (view) view.onclick = () => { closeDrawers(); openCart(); };
  $("pd-foot").querySelectorAll("[data-close]").forEach((b) => (b.onclick = closeDrawers));

  requestAnimationFrame(() => requestAnimationFrame(() => {
    $("pd-body").querySelectorAll(".bar-fill").forEach((f) => (f.style.width = f.dataset.w + "%"));
  }));
}

function takeSentence(p, r, d) {
  const off = Math.round(Math.abs(p.price - p.typical_price) / p.typical_price * 100);
  const top = pct(r.probabilities[r.verdict]);
  switch (d.cls) {
    case "scam":
      return `Probably not a real listing: the scam risk is ${pct(r.red_flag_prob)}%.`;
    case "great_deal":
      return `${top}% sure this is a great deal, ${off}% under what it usually sells for.`;
    case "fair_price":
      return `Close to its usual price. ${top}% sure it’s a fair deal.`;
    case "bad_deal":
      return `${top}% sure you’d overpay. It usually sells for ${off}% less.`;
    case "unsure":
      return `Too close to call. Best guess is ${Catalog.VERDICT_BADGES[r.verdict].toLowerCase()} at ${top}%. Worth a closer look.`;
    default:
      return "";
  }
}

function renderTake(p, r, d) {
  if (!r) {
    return `<div class="take checking"><div class="take-head"><span class="vchip checking"><i></i>${Catalog.judge()} is checking</span></div>
      <p class="take-sentence">Reading the listing…</p></div>`;
  }
  if (d.cls === "error") {
    return `<div class="take"><div class="take-head"><span class="vchip error"><i></i>Scorer error</span></div>
      <p class="take-sentence">The model didn’t answer for this listing.</p>
      <p class="gate-note">${esc(r.error || "")}</p></div>`;
  }
  const bars = [["great_deal", "Great deal"], ["fair_price", "Fair price"], ["bad_deal", "Overpriced"]].map(([k, label]) => {
    const w = pct(r.probabilities[k]);
    return `<div class="bar-row"><span>${label}</span><span class="bar-track"><span class="bar-fill ${k}" data-w="${w}"></span></span><span class="bar-pct">${w}%</span></div>`;
  }).join("");
  const value = Math.round(r.value_score);
  const red = pct(r.red_flag_prob);
  const checks = [];
  if (p.seller_feedback < 90) checks.push(`seller feedback is ${p.seller_feedback}%`);
  if (p.seller_ratings < 50) checks.push(`the seller has only ${p.seller_ratings} ratings`);
  if (p.price < p.low_90d * 0.6) checks.push(`the price is far below the 90-day low of ${money(p.low_90d)}`);
  const engine = r.mode === "demo" ? "heuristic · no model" : `${r.engine} · ${Math.round(r.trace.timing.model_ms)} ms`;

  return `
    <div class="take">
      <div class="take-head">
        <span class="vchip ${d.cls}"><i></i>${d.label}</span>
        <span class="take-engine">${esc(engine)}</span>
      </div>
      <p class="take-sentence">${esc(takeSentence(p, r, d))}</p>
      ${d.cls === "scam" && checks.length ? `<p class="scam-note">Things you can check yourself: ${esc(checks.join("; "))}.</p>` : ""}
      <div class="bars">${bars}</div>
      <div class="take-grid">
        <div class="stat"><span>Value for money</span><b>${esc(r.value_label)}</b>
          <div class="pips">${[0, 1, 2, 3, 4].map((i) => `<i class="${i <= value ? "on" : ""}"></i>`).join("")}</div></div>
        <div class="stat"><span>Red flags</span><b class="mono">${red}%</b>
          <div class="redmeter ${red >= 50 ? "hot" : ""}"><i style="width:${red}%"></i></div></div>
      </div>
      ${d.cls === "unsure" ? `<p class="gate-note">The top probability is under the 60% gate, so Kev hands this decision back to you.</p>` : ""}
      <a class="take-link" href="/lab.html#${esc(p.id)}">See how ${Catalog.judge()} decided →</a>
    </div>`;
}

function renderRange(p, d) {
  const lo = Math.min(p.price, p.low_90d) * 0.9;
  const hi = Math.max(p.price, p.high_90d) * 1.06;
  const x = (v) => ((v - lo) / (hi - lo)) * 100;
  // a label centred on a point near either end would hang off the edge, so pin it inward
  const at = (v) => {
    const pos = x(v);
    if (pos < 15) return "left:0;transform:none";
    if (pos > 85) return "right:0;transform:none";
    return `left:${pos}%;transform:translateX(-50%)`;
  };
  const color = { great_deal: "var(--great)", fair_price: "var(--fair)", bad_deal: "var(--bad)", scam: "var(--scam)" }[d.cls] || "var(--ink-2)";
  // keep the two top labels apart when the asking price sits on top of the typical tick
  const clash = Math.abs(x(p.price) - x(p.typical_price)) < 16;
  return `
    <div class="range">
      <div class="track"></div>
      <div class="band" style="left:${x(p.low_90d)}%;width:${x(p.high_90d) - x(p.low_90d)}%"></div>
      <div class="tick" style="left:${x(p.typical_price)}%"></div>
      <div class="ask" style="left:${x(p.price)}%;color:${color}"></div>
      <span class="lbl up" style="${at(p.price)};color:${color};font-weight:600">asking ${money(p.price)}</span>
      ${clash ? "" : `<span class="lbl up" style="${at(p.typical_price)}">typical</span>`}
      <span class="lbl down" style="${at(p.low_90d)}">low ${money(p.low_90d)}</span>
      <span class="lbl down" style="${at(p.high_90d)}">high ${money(p.high_90d)}</span>
    </div>`;
}

/* ---------- cart ---------- */

function loadCart() {
  try { return JSON.parse(localStorage.getItem("dealscan-cart") || "[]"); } catch (e) { return []; }
}
function saveCart() {
  try { localStorage.setItem("dealscan-cart", JSON.stringify(shop.cart)); } catch (e) { /* private mode */ }
}

function addToCart(id) {
  if (!shop.cart.includes(id)) shop.cart.push(id);
  Catalog.check(id);          // judge it now, even if it was never on screen
  saveCart();
  renderCartCount();
  const btn = $("cart-btn");
  btn.classList.remove("bump"); void btn.offsetWidth; btn.classList.add("bump");
}

function removeFromCart(id) {
  shop.cart = shop.cart.filter((x) => x !== id);
  saveCart();
  renderCartCount();
  renderCart();
  const qa = Catalog.cardEl(id).querySelector(".quick-add");
  if (qa) { qa.textContent = "Add to cart"; qa.classList.remove("added"); }
}

function renderCartCount() {
  shop.cart = shop.cart.filter((id) => Catalog.product(id));
  $("cart-count").textContent = shop.cart.length;
}

function openCart() { renderCart(); openDrawer("cart"); }

function renderCart() {
  const items = shop.cart.map((id) => Catalog.product(id)).filter(Boolean);
  if (!items.length) {
    $("cart-body").innerHTML = `<div class="cart-empty"><p>Your cart is empty.</p>Kev will check anything you add.</div>`;
    $("cart-foot").innerHTML = `<button class="btn secondary" data-close>Keep browsing</button>`;
    $("cart-foot").querySelector("[data-close]").onclick = closeDrawers;
    return;
  }
  const rows = items.map((p) => {
    const d = Catalog.display(Catalog.results[p.id]);
    return `<div class="line-item">
      <img src="${esc(p.image)}" alt="">
      <div><h4>${esc(p.title)}</h4><span class="vchip ${d.cls}"><i></i>${d.label}</span></div>
      <div class="right"><span class="price">${money(p.price)}</span><button class="rm" data-rm="${esc(p.id)}">Remove</button></div>
    </div>`;
  }).join("");

  const cls = items.map((p) => Catalog.display(Catalog.results[p.id]).cls);
  const unchecked = items.filter((p) => !Catalog.results[p.id]);
  const good = cls.filter((c) => c === "great_deal" || c === "fair_price").length;
  const flagged = items.filter((p, i) => cls[i] === "scam");
  const over = items.filter((p, i) => cls[i] === "bad_deal");
  const saved = items.filter((p, i) => cls[i] !== "scam")
    .reduce((s, p) => s + Math.max(0, p.typical_price - p.price), 0);
  const warn = flagged.length || over.length;
  const parts = [`<b>${good} of ${items.length}</b> ${items.length === 1 ? "item is" : "items are"} fair or better.`];
  if (unchecked.length) parts.push(`${unchecked.length} still being checked.`);
  if (flagged.length) parts.push(`<b>${flagged.length} flagged</b> as a likely scam: ${esc(flagged.map((p) => p.title.split(" —")[0]).join(", "))}.`);
  if (over.length) parts.push(`${over.length} overpriced.`);
  if (saved) parts.push(`You’re ${money(saved)} under typical prices on the rest.`);

  const total = items.reduce((s, p) => s + p.price, 0);
  $("cart-body").innerHTML = `<div class="kev-check ${warn ? "warn" : ""}">${Catalog.judge()}’s cart check · ${parts.join(" ")}</div>${rows}`;
  $("cart-foot").innerHTML = `
    <div class="subtotal"><span>Subtotal</span><span class="price">${money(total)}</span></div>
    <button class="btn" id="checkout">Checkout — demo store, nothing is charged</button>`;
  $("cart-body").querySelectorAll("[data-rm]").forEach((b) => (b.onclick = () => removeFromCart(b.dataset.rm)));
  $("checkout").onclick = (e) => { e.target.textContent = "This is a demo — no orders are placed ✓"; };
}

/* ---------- drawers ---------- */

function openDrawer(which) {
  $("scrim").classList.add("open");
  $(which).classList.add("open");
  $(which).setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
}

function closeDrawers() {
  shop.open = null;
  $("scrim").classList.remove("open");
  ["pd", "cart"].forEach((d) => { $(d).classList.remove("open"); $(d).setAttribute("aria-hidden", "true"); });
  document.body.style.overflow = "";
}

function bindDrawers() {
  $("cart-btn").onclick = openCart;
  $("scrim").onclick = closeDrawers;
  document.querySelectorAll(".drawer-head [data-close]").forEach((b) => (b.onclick = closeDrawers));
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawers(); });
}

/* ---------- photo credits (Unsplash license asks for attribution) ---------- */

async function renderCredits() {
  try {
    const credits = await fetch("/api/credits").then((r) => r.json());
    const names = [...new Map(Object.values(credits).map((c) => [c.username, c])).values()];
    if (!names.length) return;
    $("credits").innerHTML = "Product photos from Unsplash by " + names.map((c) =>
      `<a href="${esc(c.photo_url)}" target="_blank" rel="noopener">${esc(c.photographer)}</a>`).join(", ") + ".";
  } catch (e) { /* credits are optional */ }
}
