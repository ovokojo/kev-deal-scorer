/* catalog.js — shared storefront: product grid, scroll-triggered scoring queue,
   verdict rendering. Used by both the catalog page (app.js) and the lab page
   (lab.js). The Kev server answers one request at a time, so scoring runs
   through a client-side FIFO: cards light up in sequence, like a register. */

window.Catalog = (() => {
  let products = [];
  const results = {};          // id -> score response (or {error}) once scored
  const queued = new Set();    // ids waiting in or in the scoring queue
  const scoreQueue = [];       // ids waiting to be scored (FIFO)
  let draining = false;
  let paused = false;          // lab page gates scoring until Kev is live
  let hooks = {};
  let els = {};

  let activeCategory = "all";
  let searchText = "";

  const VERDICT_BADGES = {
    great_deal: "🔥 Great deal",
    fair_price: "🤝 Fair price",
    bad_deal: "🚫 Bad deal",
  };
  const BAR_LABELS = { great_deal: "Great", fair_price: "Fair", bad_deal: "Bad" };
  const BAR_ORDER = ["great_deal", "fair_price", "bad_deal"];
  const THINK_STEPS = [
    "Reading the listing…",
    "Judging price vs. market…",
    "Scanning for red flags…",
    "Calibrating confidence…",
  ];
  const CONFIDENCE_GATE = 0.6;   // below this, show "not sure" instead of a badge

  /* ---------- init ---------- */

  async function init(opts = {}) {
    hooks = opts;
    els = opts.els || {};
    const res = await fetch("/api/products");
    products = await res.json();

    renderGrid();
    if (els.chips) renderChips();
    if (els.search) els.search.addEventListener("input", (e) => {
      searchText = e.target.value.trim().toLowerCase();
      applyFilters();
    });
    if (els.rescore) els.rescore.onclick = () => rescoreAll();

    updateTally();

    const io = new IntersectionObserver(onVisible, { rootMargin: "220px", threshold: 0.05 });
    products.forEach((p) => io.observe(cardEl(p.id)));

    return products;
  }

  /* ---------- grid rendering ---------- */

  function renderGrid() {
    const grid = document.getElementById("grid");
    products.forEach((p, i) => {
      const el = document.createElement("article");
      el.className = "card enter";
      el.id = "card-" + p.id;
      el.dataset.id = p.id;
      el.dataset.cat = p.category || "";
      el.dataset.haystack = (p.title + " " + p.seller + " " + p.category).toLowerCase();
      el.style.animationDelay = Math.min(i * 40, 500) + "ms";
      el.innerHTML = `
        <div class="media tint-${i % 6}">
          <span class="emoji">${p.icon || "🛍️"}</span>
          <div class="badge-zone"></div>
        </div>
        <div class="body">
          <div class="vendor">${esc(p.seller)}</div>
          <h3 class="title" title="${esc(p.title)}">${esc(p.title)}</h3>
          <div class="price-row">
            <span class="price">$${esc(String(p.price))}</span>
            <span class="cond">${esc(p.condition)}</span>
          </div>
          <div class="analysis">
            <div class="thinking">
              <div class="think-row"><span class="spin"></span><span class="think-msg">Reading the listing…</span></div>
              <div class="skel"><i></i><i></i><i></i></div>
            </div>
            <div class="verdict hidden"></div>
          </div>
        </div>`;
      el.addEventListener("click", () => hooks.onCardClick && hooks.onCardClick(p));
      grid.appendChild(el);
      setBadge(el, "thinking");
    });
  }

  function setBadge(card, state) {
    const zone = card.querySelector(".badge-zone");
    if (state === "thinking") {
      zone.innerHTML = `<span class="badge thinking"><span class="mini-spin"></span>evaluating…</span>`;
    } else if (state === "unsure") {
      zone.innerHTML = `<span class="badge unsure pop">❓ Not sure</span>`;
    } else {
      zone.innerHTML = `<span class="badge ${state} pop">${VERDICT_BADGES[state]}</span>`;
    }
  }

  /* ---------- filters ---------- */

  function renderChips() {
    const cats = [...new Set(products.map((p) => p.category).filter(Boolean))];
    const chips = els.chips;
    for (const c of ["all", ...cats]) {
      const b = document.createElement("button");
      b.className = "chip" + (c === activeCategory ? " active" : "");
      b.textContent = c === "all" ? "All" : c;
      b.dataset.cat = c;
      b.onclick = () => {
        activeCategory = c;
        chips.querySelectorAll(".chip").forEach((el) =>
          el.classList.toggle("active", el.dataset.cat === c));
        applyFilters();
      };
      chips.appendChild(b);
    }
  }

  function applyFilters() {
    let shown = 0;
    for (const p of products) {
      const el = cardEl(p.id);
      const inCat = activeCategory === "all" || el.dataset.cat === activeCategory;
      const inSearch = !searchText || el.dataset.haystack.includes(searchText);
      const show = inCat && inSearch;
      el.classList.toggle("filtered", !show);
      if (show) shown++;
    }
    const empty = document.getElementById("empty");
    if (empty) empty.classList.toggle("hidden", shown > 0);
    updateTally();
  }

  /* ---------- the scoring queue ---------- */

  function onVisible(entries) {
    if (paused) return;
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const id = e.target.dataset.id;
      if (results[id] || queued.has(id)) continue;
      enqueue(id);
    }
  }

  function enqueue(id) {
    queued.add(id);
    scoreQueue.push(id);
    updateQueueNote();
    drain();
  }

  async function drain() {
    if (draining) return;
    draining = true;
    while (scoreQueue.length) {
      const id = scoreQueue.shift();
      updateQueueNote();
      await scoreCard(id);
    }
    draining = false;
    updateQueueNote();
  }

  function updateQueueNote() {
    const note = document.getElementById("queue-note");
    if (!note) return;
    const n = scoreQueue.length;
    note.classList.toggle("hidden", n === 0);
    if (n) note.textContent = `⏳ ${n} in queue`;
  }

  /* Kev judges for real in 0.2–0.8 s; demo mode answers instantly.
     Either way, hold each card in "thinking" long enough to read. */
  const MIN_THINK_MS = () => 1100 + Math.random() * 700;

  function scoreCard(id) {
    const p = products.find((x) => x.id === id);
    const card = cardEl(id);
    const msgTimer = startThinkMessages(card);

    const waitAll = Promise.all([
      fetch("/api/score", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ id }),
      }).then((r) => r.json()),
      new Promise((res) => setTimeout(res, MIN_THINK_MS())),
    ]);

    return waitAll
      .then(([r]) => {
        if (r.error) throw new Error(r.error);
        results[id] = r;
        renderVerdict(card, r);
        if (hooks.onResult) hooks.onResult(p, r);
      })
      .catch(() => {
        results[id] = { error: true };
        card.querySelector(".analysis .thinking").classList.add("hidden");
        setBadge(card, "unsure");
        const v = card.querySelector(".verdict");
        v.classList.remove("hidden");
        v.innerHTML =
          `<p class="err">Couldn’t reach the scorer — hit “Re-score all” to retry.</p>`;
      })
      .finally(() => {
        clearInterval(msgTimer);
        queued.delete(id);
        updateTally();
      });
  }

  function startThinkMessages(card) {
    const msg = card.querySelector(".think-msg");
    let i = 0;
    msg.textContent = THINK_STEPS[0];
    return setInterval(() => {
      i = (i + 1) % THINK_STEPS.length;
      msg.textContent = THINK_STEPS[i];
    }, 550);
  }

  /* ---------- verdict rendering ---------- */

  function renderVerdict(card, r) {
    // NB: ".thinking" also matches the waiting badge ("badge thinking"), and the
    // badge precedes the analysis block in tree order — always scope to .analysis.
    card.querySelector(".analysis .thinking").classList.add("hidden");
    setBadge(card, r.confidence < CONFIDENCE_GATE ? "unsure" : r.verdict);

    const v = card.querySelector(".verdict");
    v.classList.remove("hidden");
    v.classList.add("reveal");

    const bars = BAR_ORDER.map((k) => {
      const pct = Math.round((r.probabilities[k] || 0) * 100);
      return `<div class="bar-row">
        <span class="bar-name">${BAR_LABELS[k]}</span>
        <span class="bar-track"><span class="bar-fill ${k}" style="width:0%"></span></span>
        <span class="bar-pct">${pct}%</span>
      </div>`;
    }).join("");

    const unsure = r.confidence < CONFIDENCE_GATE;
    const chips = [
      `<span class="chiplet${unsure ? " warn" : ""}">${unsure ? "low confidence" : Math.round(r.confidence * 100) + "% confident"}</span>`,
      `<span class="chiplet">⚡ ${r.latency_ms} ms</span>`,
      r.red_flag_prob >= 0.5
        ? `<span class="chiplet alert">⚠ scam signals ${Math.round(r.red_flag_prob * 100)}%</span>`
        : `<span class="chiplet">${esc(r.value_label)}</span>`,
    ].join("");

    v.innerHTML =
      (unsure ? `<p class="unsure-note">Below the 60% confidence gate — review this one yourself.</p>` : "") +
      `<div class="bars">${bars}</div><div class="chips">${chips}</div>`;

    // animate fills after paint so the width transition runs
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        v.querySelectorAll(".bar-fill").forEach((f, i) => {
          const pct = Math.round((r.probabilities[BAR_ORDER[i]] || 0) * 100);
          f.style.width = pct + "%";
        });
      }));
  }

  /* ---------- tally + rescore ---------- */

  function updateTally() {
    if (!els.countLine) return;
    const scored = Object.values(results).filter((r) => r && !r.error);
    const total = products.length;
    const g = scored.filter((r) => r.verdict === "great_deal").length;
    const f = scored.filter((r) => r.verdict === "fair_price").length;
    const b = scored.filter((r) => r.verdict === "bad_deal").length;
    const u = total - scored.length;

    const tally = scored.length
      ? `${scored.length} evaluated — ${g} great · ${f} fair · ${b} skip`
      : "none evaluated yet";
    els.countLine.textContent =
      `${total} products · ${tally}` + (u ? ` · ${u} to go` : "");
  }

  function rescoreAll() {
    for (const p of products) {
      const card = cardEl(p.id);
      delete results[p.id];
      const v = card.querySelector(".verdict");
      v.classList.add("hidden");
      v.innerHTML = "";
      v.classList.remove("reveal");
      card.querySelector(".analysis .thinking").classList.remove("hidden");
      setBadge(card, "thinking");
      // re-enqueue only what's on screen; the observer catches the rest on scroll
      if (!card.classList.contains("filtered") && isShown(card)) enqueue(p.id);
    }
    updateTally();
  }

  function isShown(card) {
    const r = card.getBoundingClientRect();
    return r.bottom > 0 && r.top < window.innerHeight;
  }

  /* ---------- utils ---------- */

  function esc(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  /* ---------- public API ---------- */

  function cardEl(id) { return document.getElementById("card-" + id); }

  return {
    init,
    rescoreAll,
    updateTally,
    cardEl,
    setPaused(v) {
      paused = v;
      if (!v) {
        // score whatever is on screen now — the observer only fires on changes
        for (const p of products) {
          const el = cardEl(p.id);
          if (!results[p.id] && !queued.has(p.id) && !el.classList.contains("filtered") && isShown(el)) {
            enqueue(p.id);
          }
        }
      }
    },
    products: () => products,
    results,             // live object: id -> response
    CONFIDENCE_GATE,
    VERDICT_BADGES,
  };
})();
