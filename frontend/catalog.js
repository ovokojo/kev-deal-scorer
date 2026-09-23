/* catalog.js — shared storefront: product grid, scroll-triggered scoring queue,
   verdict rendering, sort and filters. Used by the shop (app.js) and the lab (lab.js).

   Scoring follows the engine. Laya (and the demo heuristic) answer a whole screen of
   listings in one batched request, so visible cards are collected for a moment and sent
   together. Kev answers one listing at a time, so cards go through a FIFO and light up
   in sequence, like a register. */

window.Catalog = (() => {
  let products = [];
  const results = {};          // id -> score response once scored
  const queued = new Set();    // ids waiting for, or in, a scoring request
  let pending = [];            // ids waiting to be sent
  let draining = false;
  let flushTimer = null;
  let paused = false;          // the lab gates scoring until a live engine answers
  let batchMode = true;
  let hooks = {};
  let els = {};

  let judge = "the model";      // whoever is answering: Laya, Kev, or the heuristic
  let activeCategory = "all";
  let searchText = "";
  let sortKey = "featured";
  let hideBad = false;

  const CONFIDENCE_GATE = 0.6;  // top probability below this → "not sure"
  const RED_FLAG_GATE = 0.5;    // red-flag probability at or above this → "likely scam"
  const BATCH_MAX = 30;
  const MIN_REVEAL_MS = 250;    // batched answers land in ~0.1 s; give the reveal a beat

  const LABELS = {
    great_deal: "Great deal",
    fair_price: "Fair price",
    bad_deal: "Overpriced",
    scam: "Likely scam",
    unsure: "Not sure",
    error: "Scorer error",
    checking: "Checking",
  };
  const VERDICT_BADGES = LABELS;  // the lab page reads this name
  const ORDER = ["great_deal", "fair_price", "bad_deal"];

  /* What the shopper is shown for a result. The scam flag overrides the price verdict:
     a $39 AirPods listing is not a "great deal" if it is probably not real. */
  function display(r) {
    if (!r) return { cls: "checking", label: LABELS.checking };
    if (r.mode === "error" || r.error) return { cls: "error", label: LABELS.error };
    if (r.red_flag_prob >= RED_FLAG_GATE) return { cls: "scam", label: LABELS.scam };
    if (r.confidence < CONFIDENCE_GATE) return { cls: "unsure", label: LABELS.unsure };
    return { cls: r.verdict, label: LABELS[r.verdict] };
  }

  /* ---------- init ---------- */

  async function init(opts = {}) {
    hooks = opts;
    els = opts.els || {};
    if (opts.health) setEngine(opts.health);
    products = await fetch("/api/products").then((r) => r.json());

    renderGrid();
    if (els.chips) renderChips();
    if (els.search) els.search.addEventListener("input", (e) => {
      searchText = e.target.value.trim().toLowerCase();
      applyFilters();
    });
    if (els.sort) els.sort.addEventListener("change", (e) => { sortKey = e.target.value; applySort(); });
    if (els.hide) els.hide.addEventListener("change", (e) => { hideBad = e.target.checked; applyFilters(); });
    if (els.rescore) els.rescore.onclick = () => rescoreAll();
    applyFilters();

    const io = new IntersectionObserver(onVisible, { rootMargin: "200px", threshold: 0.05 });
    products.forEach((p) => io.observe(cardEl(p.id)));
    return products;
  }

  function setEngine(health) {
    batchMode = health.engine !== "kev";
    judge = health.engine === "laya" ? "Laya" : health.engine === "kev" ? "Kev" : "the model";
    document.querySelectorAll(".js-judge").forEach((el) => (el.textContent = judge));
    // cards that already say "<engine> is checking…" should follow a mid-session switch
    products.forEach((p) => {
      const card = cardEl(p.id);
      if (card && !results[p.id]) card.querySelector(".kev-text").textContent = `${judge} is checking…`;
    });
  }

  /* ---------- grid rendering ---------- */

  function renderGrid() {
    const grid = document.getElementById("grid");
    products.forEach((p, i) => {
      const el = document.createElement("article");
      el.className = "card";
      el.id = "card-" + p.id;
      el.dataset.id = p.id;
      el.dataset.cat = p.category || "";
      el.dataset.haystack = (p.title + " " + p.seller + " " + p.category).toLowerCase();
      el.style.animationDelay = Math.min(i * 45, 600) + "ms";
      el.innerHTML = `
        <div class="media">
          <img src="${esc(p.image)}" alt="${esc(p.title)}" loading="lazy" width="500" height="625">
          <div class="badge-zone"></div>
          ${hooks.quickAdd ? `<button class="quick-add" type="button">Add to cart</button>` : ""}
        </div>
        <div class="body">
          <div class="meta-row">
            <span class="seller">${esc(p.seller)}</span>
            <span class="stars">★ ${p.rating.toFixed(1)} <i>(${p.reviews.toLocaleString()})</i></span>
          </div>
          <h3 class="title" title="${esc(p.title)}">${esc(p.title)}</h3>
          <div class="price-row">
            <span class="price">${money(p.price)}</span>
            ${p.list_price > p.price ? `<s class="list">${money(p.list_price)}</s>` : ""}
            <span class="cond">· ${esc(p.condition)}</span>
          </div>
          <div class="kev-line">
            <span class="meter checking"><i class="m-great_deal"></i><i class="m-fair_price"></i><i class="m-bad_deal"></i></span>
            <span class="kev-text">${judge} is checking…</span>
          </div>
        </div>`;
      el.addEventListener("click", (e) => {
        if (e.target.closest(".quick-add")) {
          e.stopPropagation();
          hooks.quickAdd(p, e.target);
          return;
        }
        hooks.onCardClick && hooks.onCardClick(p);
      });
      grid.appendChild(el);
      setChip(el, null);
    });
  }

  function setChip(card, r, animate) {
    const d = display(r);
    card.querySelector(".badge-zone").innerHTML =
      `<span class="vchip ${d.cls}${animate ? " pop" : ""}"><i></i>${d.label}</span>`;
    card.classList.toggle("is-scam", d.cls === "scam");
    card.dataset.verdict = d.cls;
  }

  function renderResult(card, r) {
    setChip(card, r, true);
    const meter = card.querySelector(".meter");
    const text = card.querySelector(".kev-text");
    meter.classList.remove("checking");
    text.classList.remove("scam");
    const d = display(r);
    if (d.cls === "error") {
      ORDER.forEach((k) => (meter.querySelector(".m-" + k).style.width = "0"));
      text.textContent = "Couldn’t reach the model";
      return;
    }
    requestAnimationFrame(() => requestAnimationFrame(() => {
      ORDER.forEach((k) => (meter.querySelector(".m-" + k).style.width =
        Math.round((r.probabilities[k] || 0) * 100) + "%"));
    }));
    const top = pct(r.probabilities[r.verdict]);
    if (d.cls === "scam") {
      text.classList.add("scam");
      text.innerHTML = `⚠ <span class="mono">${pct(r.red_flag_prob)}%</span> scam risk`;
    } else if (d.cls === "unsure") {
      text.innerHTML = `Unsure · <span class="mono">${top}%</span> ${short(r.verdict)}`;
    } else {
      text.innerHTML = `<span class="mono">${top}%</span> <b>${short(r.verdict)}</b>`;
    }
  }

  function short(v) {
    return { great_deal: "great deal", fair_price: "fair price", bad_deal: "overpriced" }[v] || v;
  }

  function resetCard(card) {
    setChip(card, null);
    card.querySelector(".meter").classList.add("checking");
    card.querySelectorAll(".meter i").forEach((i) => (i.style.width = "0"));
    const text = card.querySelector(".kev-text");
    text.classList.remove("scam");
    text.textContent = `${judge} is checking…`;
  }

  /* ---------- filters + sort ---------- */

  function renderChips() {
    const counts = {};
    products.forEach((p) => (counts[p.category] = (counts[p.category] || 0) + 1));
    for (const c of ["all", ...Object.keys(counts)]) {
      const b = document.createElement("button");
      b.className = "chip" + (c === activeCategory ? " active" : "");
      b.textContent = c === "all" ? "All" : c;
      b.dataset.cat = c;
      b.onclick = () => {
        activeCategory = c;
        els.chips.querySelectorAll(".chip").forEach((el) =>
          el.classList.toggle("active", el.dataset.cat === c));
        applyFilters();
      };
      els.chips.appendChild(b);
    }
  }

  function applyFilters() {
    let shown = 0;
    for (const p of products) {
      const el = cardEl(p.id);
      const inCat = activeCategory === "all" || el.dataset.cat === activeCategory;
      const inSearch = !searchText || el.dataset.haystack.includes(searchText);
      const hidden = hideBad && ["bad_deal", "scam"].includes(el.dataset.verdict);
      const show = inCat && inSearch && !hidden;
      el.classList.toggle("filtered", !show);
      if (show) shown++;
    }
    const empty = document.getElementById("empty");
    if (empty) empty.classList.toggle("hidden", shown > 0);
    if (els.results) {
      els.results.textContent = `${shown} of ${products.length} listings` +
        (hideBad ? " · overpriced and flagged listings hidden" : "");
    }
    updateTally();
  }

  function dealScore(p) {
    const r = results[p.id];
    if (!r || r.error) return -0.5;
    if (display(r).cls === "scam") return -3;
    return (r.probabilities.great_deal || 0) - (r.probabilities.bad_deal || 0);
  }

  function applySort() {
    const grid = document.getElementById("grid");
    const order = products.slice();
    if (sortKey === "deal") order.sort((a, b) => dealScore(b) - dealScore(a));
    if (sortKey === "price-asc") order.sort((a, b) => a.price - b.price);
    if (sortKey === "price-desc") order.sort((a, b) => b.price - a.price);
    order.forEach((p) => {
      const el = cardEl(p.id);
      el.style.animation = "none";
      grid.appendChild(el);
    });
  }

  /* ---------- the scoring queue ---------- */

  function onVisible(entries) {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      enqueue(e.target.dataset.id);
    }
  }

  /* Score one listing even if it is off screen (used when it lands in the cart). */
  function check(id) {
    if (results[id] || queued.has(id)) return;
    enqueue(id);
  }

  function enqueue(id) {
    if (paused || results[id] || queued.has(id)) return;
    queued.add(id);
    pending.push(id);
    clearTimeout(flushTimer);
    flushTimer = setTimeout(drain, batchMode ? 60 : 0);
  }

  async function drain() {
    if (draining) return;
    draining = true;
    while (pending.length) {
      const ids = pending.splice(0, batchMode ? BATCH_MAX : 1);
      await scoreIds(ids);
    }
    draining = false;
  }

  async function scoreIds(ids) {
    const started = performance.now();
    let list;
    try {
      const res = await fetch("/api/score", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ ids }),
      }).then((r) => r.json());
      if (res.error) throw new Error(res.error);
      list = res.results;
    } catch (e) {
      list = ids.map((id) => ({ id, mode: "error", error: String(e) }));
    }
    const wait = MIN_REVEAL_MS - (performance.now() - started);
    if (batchMode && wait > 0) await new Promise((r) => setTimeout(r, wait));

    list.forEach((r, i) => {
      const p = products.find((x) => x.id === r.id);
      results[r.id] = r;
      queued.delete(r.id);
      // stagger the reveal a little so a batch reads as a sweep, not a flash
      setTimeout(() => {
        if (!results[r.id]) return;   // re-scored in the meantime
        renderResult(cardEl(r.id), r);
        if (hooks.onResult) hooks.onResult(p, r);
        updateTally();
        if (hideBad) applyFilters();
        if (sortKey === "deal") applySort();
      }, batchMode ? i * 55 : 0);
    });
  }

  /* ---------- tally + rescore ---------- */

  function tally() {
    const t = { total: products.length, scored: 0, great_deal: 0, fair_price: 0, bad_deal: 0, scam: 0, unsure: 0, error: 0 };
    for (const r of Object.values(results)) {
      t.scored++;
      t[display(r).cls]++;
    }
    return t;
  }

  function updateTally() {
    if (hooks.onTally) hooks.onTally(tally());
  }

  function rescoreAll() {
    for (const p of products) {
      delete results[p.id];
      resetCard(cardEl(p.id));
    }
    queued.clear();
    pending = [];
    for (const id of Object.keys(results)) delete results[id];
    updateTally();
    for (const p of products) {
      const el = cardEl(p.id);
      if (!el.classList.contains("filtered") && isShown(el)) enqueue(p.id);
    }
  }

  function isShown(card) {
    const r = card.getBoundingClientRect();
    return r.bottom > -200 && r.top < window.innerHeight + 200;
  }

  /* ---------- utils ---------- */

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function money(n) { return "$" + Number(n).toLocaleString(); }
  function pct(x) { return Math.round((x || 0) * 100); }
  function cardEl(id) { return document.getElementById("card-" + id); }

  return {
    init,
    setEngine,
    check,
    judge: () => judge,
    rescoreAll,
    cardEl,
    display,
    tally,
    esc,
    money,
    pct,
    setPaused(v) {
      paused = v;
      if (!v) {
        // score whatever is on screen now; the observer only fires on changes
        for (const p of products) {
          const el = cardEl(p.id);
          if (!el.classList.contains("filtered") && isShown(el)) enqueue(p.id);
        }
      }
    },
    products: () => products,
    product: (id) => products.find((p) => p.id === id),
    results,             // live object: id -> response
    CONFIDENCE_GATE,
    RED_FLAG_GATE,
    VERDICT_BADGES,
  };
})();
