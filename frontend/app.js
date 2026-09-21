let products = [], idx = 0;

const $ = (id) => document.getElementById(id);
const VERDICT_NAMES = { great_deal: "🔥 Great deal", fair_price: "🤝 Fair price", bad_deal: "🚫 Bad deal" };

async function init() {
  const [pRes, hRes] = await Promise.all([
    fetch("/api/products"), fetch("/api/health"),
  ]);
  products = await pRes.json();
  const health = await hRes.json();

  const pill = $("mode-pill");
  if (health.kev) {
    pill.textContent = "● kev live";
    pill.className = "pill live";
  } else {
    pill.textContent = "● demo mode (no model)";
    pill.className = "pill demo";
  }
  $("total").textContent = products.length;
  showCard(0);
  bindSwipe();
  bindKeys();
  $("next").onclick = () => advance(1);
  $("skip").onclick = () => advance(1);
}

function showCard(i) {
  idx = ((i % products.length) + products.length) % products.length;
  const p = products[idx];
  $("cat").textContent = p.icon || "🛍️";
  $("title").textContent = p.title;
  $("price").textContent = "$" + p.price;
  $("cond").textContent = p.condition;
  $("blurb").textContent = p.blurb;
  $("seller").textContent = "Sold by " + p.seller;
  $("idx").textContent = idx + 1;

  $("verdict").classList.add("hidden");
  $("scoring").classList.remove("hidden");
  const card = $("card");
  card.style.transform = "";
  card.style.opacity = "";

  score(p);
}

async function score(p) {
  try {
    const res = await fetch("/api/score", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: p.id }),
    });
    const r = await res.json();
    if (r.error) throw new Error(r.error);
    renderVerdict(r);
  } catch (e) {
    $("scoring").innerHTML = "<span>scoring failed — check the backend</span>";
  }
}

function renderVerdict(r) {
  $("scoring").classList.add("hidden");
  const v = $("verdict");
  v.classList.remove("hidden");

  const label = $("verdict-label");
  label.textContent = VERDICT_NAMES[r.verdict] || r.verdict;
  label.className = "verdict-label " + r.verdict;

  const bars = $("bars");
  bars.innerHTML = "";
  const order = ["great_deal", "fair_price", "bad_deal"];
  for (const k of order) {
    const pct = Math.round((r.probabilities[k] || 0) * 100);
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML =
      `<span class="name">${k.replace("_", " ")}</span>` +
      `<span class="bar-track"><span class="bar-fill" style="width:0%"></span></span>` +
      `<span class="pct">${pct}%</span>`;
    bars.appendChild(row);
    requestAnimationFrame(() =>
      requestAnimationFrame(() => (row.querySelector(".bar-fill").style.width = pct + "%"))
    );
  }

  $("latency").textContent = `⚡ ${r.latency_ms} ms`;
  $("confidence").textContent = `confidence ${Math.round(r.confidence * 100)}%`;
  const rf = $("redflag");
  if (r.red_flag_prob >= 0.5) {
    rf.textContent = `⚠️ red flags ${Math.round(r.red_flag_prob * 100)}%`;
    rf.className = "warn";
  } else {
    rf.textContent = `${r.value_label} (${r.value_score}/4)`;
    rf.className = "";
  }
}

function advance(dir) {
  const card = $("card");
  card.style.transform = `translateX(${dir * 120}vw) rotate(${dir * 18}deg)`;
  card.style.opacity = "0";
  setTimeout(() => showCard(idx + dir), 240);
}

function bindSwipe() {
  const card = $("card");
  let startX = 0, dx = 0, dragging = false;

  card.addEventListener("pointerdown", (e) => {
    dragging = true; startX = e.clientX; dx = 0;
    card.style.transition = "none";
    card.setPointerCapture(e.pointerId);
  });
  card.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    dx = e.clientX - startX;
    card.style.transform = `translateX(${dx}px) rotate(${dx / 18}deg)`;
  });
  const end = () => {
    if (!dragging) return;
    dragging = false;
    card.style.transition = "";
    if (Math.abs(dx) > 110) advance(dx > 0 ? 1 : -1);
    else { card.style.transform = ""; }
  };
  card.addEventListener("pointerup", end);
  card.addEventListener("pointercancel", end);
}

function bindKeys() {
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") advance(1);
    if (e.key === "ArrowLeft") advance(-1);
  });
}

init();
