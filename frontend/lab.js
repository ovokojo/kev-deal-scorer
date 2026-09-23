/* lab.js — the "How it decides" lesson panel.
   Auto-follows live scoring runs and walks through the six stages of a real
   System One request: listing → state → typed questions → the model pass →
   structured output → verdict + timing. Requires a live engine (Laya on :8010 or
   Kev on :8009); the gate card polls and flips the page to live automatically. */

const $ = (id) => document.getElementById(id);

const STAGES = [
  { title: "The listing", render: renderListing },
  { title: "The state", render: renderState },
  { title: "Three typed questions", render: renderQuestions },
  { title: "The model pass", render: renderPass },
  { title: "The structured output", render: renderOutput },
  { title: "Verdict + timing", render: renderVerdictTiming },
];
const AUTO_MS = 3400;

const lab = {
  runs: {},        // id -> {product, result}
  pending: [],     // ids waiting to be played
  current: null,   // product id on stage
  stage: 0,
  auto: true,
  timer: null,
  live: false,
};

/* ---------- boot ---------- */

const deepLink = location.hash.slice(1) || null;   // /lab.html#p7 from the shop's product drawer

renderDots();
bindControls();
Catalog.setPaused(true);       // no scoring until a live engine answers
Catalog.init({
  els: { chips: $("chips"), results: $("results-line"), rescore: $("rescore") },
  onResult,
  onCardClick: (p) => { if (lab.runs[p.id]) { setAuto(false); loadRun(p.id); } },
  onTally: (t) => {
    $("results-line").textContent = t.scored
      ? `${t.scored} of ${t.total} listings evaluated`
      : `${t.total} listings · none evaluated yet`;
  },
});
healthLoop();

function onResult(product, result) {
  lab.runs[product.id] = { product, result };
  if (!lab.live) return;
  if (product.id === deepLink && !lab.deepLinked) {
    lab.deepLinked = true;
    setAuto(false);
    loadRun(product.id);
    Catalog.cardEl(product.id).scrollIntoView({ block: "center", behavior: "smooth" });
    return;
  }
  lab.pending.push(product.id);
  tryNext();
}

/* ---------- health gate ---------- */

async function healthLoop() {
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    setAnnounce(h);
    Catalog.setEngine(h);
    setLive(h.engine !== "demo");
  } catch (e) { /* backend down — stay gated */ }
  setTimeout(healthLoop, lab.live ? 10000 : 3000);
}

function setAnnounce(h) {
  const el = $("announce"), text = $("announce-text");
  el.classList.remove("live", "demo");
  const run = (h.run || "").split("/").pop();
  if (h.engine === "demo") {
    el.classList.add("demo");
    text.innerHTML = "<b>No model running</b> · the lab waits for a live engine";
  } else {
    el.classList.add("live");
    text.innerHTML = `<b>Model live</b> · <span class="mono">${esc(h.engine)} · ${esc(run)}</span>`;
  }
}

function setLive(on) {
  $("gate").classList.toggle("hidden", on);
  if (on === lab.live) return;
  lab.live = on;
  if (on) {
    lab.runs = {};
    lab.pending = [];
    Catalog.setPaused(false);
    Catalog.rescoreAll();       // everything re-scores with the real model
    if (deepLink && Catalog.cardEl(deepLink)) Catalog.cardEl(deepLink).scrollIntoView({ block: "center" });
  } else {
    Catalog.setPaused(true);
  }
}

/* ---------- lesson flow ---------- */

function tryNext() {
  if (!lab.auto || !lab.live) return;
  // busy mid-lesson? chain when it finishes (scheduleNext calls back here)
  if (lab.current && lab.stage < STAGES.length - 1) return;
  const id = lab.pending.shift();
  if (id) loadRun(id);
}

function loadRun(id) {
  document.querySelectorAll(".card.on-stage").forEach((c) => c.classList.remove("on-stage"));
  Catalog.cardEl(id).classList.add("on-stage");
  lab.current = id;
  lab.stage = 0;
  renderStage();
  scheduleNext();
}

function scheduleNext() {
  clearTimeout(lab.timer);
  if (!lab.auto) return;
  lab.timer = setTimeout(() => {
    if (lab.stage < STAGES.length - 1) {
      lab.stage++;
      renderStage();
      scheduleNext();
    } else {
      tryNext();               // lesson finished — play the next queued run
    }
  }, AUTO_MS);
}

function setAuto(on) {
  lab.auto = on;
  $("auto-toggle").textContent = on ? "⏸ auto" : "▶ auto";
  if (on) { tryNext(); scheduleNext(); }
  else clearTimeout(lab.timer);
}

function gotoStage(n) {
  setAuto(false);
  lab.stage = Math.max(0, Math.min(STAGES.length - 1, n));
  renderStage();
}

function bindControls() {
  $("prev-stage").onclick = () => gotoStage(lab.stage - 1);
  $("next-stage").onclick = () => gotoStage(lab.stage + 1);
  $("auto-toggle").onclick = () => setAuto(!lab.auto);
}

function renderDots() {
  const dots = $("dots");
  dots.innerHTML = STAGES.map((s, i) =>
    `<button class="dot${i === lab.stage && lab.current ? " on" : ""}" data-i="${i}" title="${s.title}"></button>`
  ).join("");
  dots.querySelectorAll(".dot").forEach((d) =>
    d.onclick = () => gotoStage(+d.dataset.i));
}

function renderStage() {
  const run = lab.runs[lab.current];
  const stage = $("stage");
  if (!run) {
    stage.innerHTML = `<p class="stage-wait">Scoring hasn’t started yet — scroll the catalog and the
      lesson will follow the first listing that gets evaluated.</p>`;
    renderDots();
    return;
  }
  const s = STAGES[lab.stage];
  const demoBanner = run.result.mode === "error"
    ? `<div class="demo-banner">⚠ The model didn’t answer this run: ${esc(run.result.error || "")}</div>`
    : run.result.mode === "demo"
      ? `<div class="demo-banner">⚠ The heuristic answered this run, not a model.</div>`
      : "";
  if (run.result.mode === "error") {
    stage.innerHTML = demoBanner;
    renderDots();
    return;
  }
  stage.innerHTML = `
    <div class="stage-kicker">Stage ${lab.stage + 1} of ${STAGES.length} — ${esc(s.title)}</div>
    ${demoBanner}
    ${s.render(run)}`;
  stage.classList.remove("reveal");
  void stage.offsetWidth;      // restart the entrance animation
  stage.classList.add("reveal");
  renderDots();
  $("run-label").textContent =
    `${run.product.title.slice(0, 44)}${run.product.title.length > 44 ? "…" : ""}` +
    (run.result.mode === "live" ? ` · live ${run.result.engine} run` : " · heuristic");
  stage.scrollTop = 0;
}

/* ---------- stage renderers ---------- */

function renderListing({ product: p }) {
  return `
    <div class="mini-product">
      <img src="${esc(p.image)}" alt="">
      <div class="mini-info">
        <span class="mini-meta">${esc(p.seller)} · ${p.seller_feedback}% positive</span>
        <strong>${esc(p.title)}</strong>
        <span class="mini-meta">${Catalog.money(p.price)} · ${esc(p.condition)} · ${esc(p.category)}</span>
      </div>
    </div>
    <p class="explain">The shopper sees a listing. The backend turns it into a ticket for the model.</p>
    <div class="callout">The ticket includes the item’s <strong>90-day price history</strong>
    (typical ${Catalog.money(p.typical_price)}, low ${Catalog.money(p.low_90d)}, high ${Catalog.money(p.high_90d)}),
    the same numbers a deal tracker shows you. The model doesn’t have to remember what a
    ${esc(p.category.toLowerCase())} item costs in 2026. It compares, and it reads the seller.</div>`;
}

function renderState({ result: r }) {
  const st = esc(r.trace.request.state).replace(/^(Price history.*)$/m, "<mark>$1</mark>");
  return `
    <p class="explain">The backend flattens the listing into one text block, the
    <strong>state</strong>. This string is everything the model will ever see.</p>
    <pre class="code">${st}</pre>
    <div class="callout">No chat history, no system prompt, no tools. Plain text in,
    decisions out. That is the whole input contract.</div>`;
}

function renderQuestions({ result: r }) {
  const qs = r.trace.request.questions;
  const cards = Object.entries(qs).map(([id, q]) => {
    let crit;
    if (q.type === "choice") {
      crit = Object.entries(q.criteria).map(([k, v]) =>
        `<li><b>${esc(k)}</b> — ${esc(v || "")}</li>`).join("");
    } else if (q.type === "score") {
      crit = q.criteria.map((v, i) => `<li><b>${i}</b> — ${esc(v)}</li>`).join("");
    } else {
      crit = `<li><b>true</b> — flags present</li><li><b>false</b> — looks clean</li>`;
    }
    return `
      <div class="q-card">
        <div class="q-head"><span class="q-id">${esc(id)}</span><span class="q-type t-${q.type}">${q.type}</span></div>
        <p class="q-instructions">${esc(q.instructions)}</p>
        <ul class="q-criteria">${crit}</ul>
      </div>`;
  }).join("");
  return `
    <p class="explain">Three typed questions ride along with the state, each with its
    own rubric. Every answer comes back as a number, never a sentence.</p>
    ${cards}
    <div class="callout">The keys (<code>verdict</code>, <code>value</code>, <code>red_flags</code>) are ours.
    The model never sees them, only the instructions and options.</div>`;
}

function renderPass({ result: r }) {
  const usage = r.trace.response.usage || {};
  const inTok = usage.input_tokens != null ? usage.input_tokens : "–";
  const branches = ["verdict · choice", "value · score", "red_flags · noul"].map((label) => `
    <div class="flow-q">
      <span class="flow-q-name">${label}</span>
      <div class="flow-opts"><i></i><i></i><i></i></div>
      <div class="flow-decide">${r.engine === "laya" ? "option markers" : "&lt;decide&gt;"}</div>
    </div>`).join("");
  const batch = r.trace.response.batch;

  if (r.engine === "laya") {
    return `
      <div class="flow">
        <div class="flow-head">one batched call · ${batch ? batch.batch_size * 3 : 3} sequences side by side</div>
        <div class="flow-branches">${branches}</div>
        <div class="flow-state">bidirectional encoder · ModernBERT-large, 421M<span class="tok">${inTok} input tokens across the three sequences</span></div>
        <div class="flow-head">scorer head over each option marker · softmax → probabilities</div>
      </div>
      <p class="explain">Laya is an <strong>encoder</strong>, not a chat model. Each question is packed
      with its options and the state into its own sequence, and every token sees every other one.
      Nothing is generated. A small head scores each option marker and a softmax turns the scores into probabilities.</p>
      <p class="explain">All three questions for ${batch ? `all ${batch.batch_size} listings on screen` : "this listing"} go through
      the model in <strong>one batched call</strong>. That is why the shop can check a whole screen at once.</p>
      <div class="chiplets">
        <span class="chiplet">in: ${inTok} tokens</span><span class="chiplet">out: 0 tokens</span>
        ${batch ? `<span class="chiplet">batch: ${batch.batch_size} listings</span>` : ""}
      </div>`;
  }
  return `
    <div class="flow">
      <div class="flow-state">state, encoded <em>once</em><span class="tok">${inTok} input tokens</span></div>
      <div class="flow-branches">${branches}</div>
      <div class="flow-head">pointer head · softmax → probabilities</div>
    </div>
    <p class="explain">Kev puts the state and all three questions into <strong>one token sequence</strong>.
    An attention mask lets each question read the state, but never the other questions.</p>
    <p class="explain">The pointer head scores every option against each question’s
    <code>&lt;decide&gt;</code> token, and a softmax turns the scores into probabilities.
    <strong>One forward pass, three answers.</strong></p>
    <div class="chiplets"><span class="chiplet">in: ${inTok} tokens</span><span class="chiplet">passes: 1</span></div>`;
}

function renderOutput({ result: r }) {
  const answers = r.trace.response.answers || {};
  const heur = r.trace.response.heuristic;
  return `
    <p class="explain">What came back, verbatim, keyed by our question ids:</p>
    <pre class="code json">${hi(JSON.stringify(answers, null, 2))}</pre>
    ${heur ? `<div class="callout warn">Heuristic run: ${esc(heur.rule)}. ${esc(heur.note || "")}</div>` : ""}
    <div class="callout">Each engine reports its own <code>confidence</code> (Kev: rescaled top
    probability; Laya: one minus normalised entropy). The shop ignores both and gates on the
    <strong>top probability</strong>, so a 60% gate means the same thing whichever engine answered.</div>`;
}

function renderVerdictTiming({ result: r }) {
  const d = Catalog.display(r);
  const confPct = Math.round(r.confidence * 100);
  const unsure = r.confidence < Catalog.CONFIDENCE_GATE;
  const red = Math.round(r.red_flag_prob * 100);

  const t = r.trace.timing || {};
  const batch = r.trace.response.batch;
  const total = batch ? batch.batch_round_trip_ms : (t.round_trip_ms != null ? t.round_trip_ms : r.latency_ms);
  const model = batch ? Math.round(batch.batch_model_ms) : (t.model_ms != null ? Math.round(t.model_ms) : total);
  const overhead = Math.max(0, total - model);
  const modelW = total > 0 ? Math.min(100, Math.round((model / total) * 100)) : 100;

  const timingNote = r.engine === "laya"
    ? `<p class="explain">This listing was scored in a batch of ${batch ? batch.batch_size : 1}, so the bar is the
       whole batch: <span class="mono">${model} ms</span> of model time for all of them, about
       <span class="mono">${Math.round(t.model_ms)} ms</span> each.</p>`
    : r.engine === "kev"
      ? `<p class="explain">“Model” is Kev’s own measure of the forward pass. The rest is HTTP and backend overhead.</p>`
      : `<p class="explain">The heuristic answers in about 0 ms because no model is involved.</p>`;

  return `
    <p class="explain">The shop reads the structured answer directly:</p>
    <div class="final-badge">
      <span class="vchip ${d.cls} pop"><i></i>${d.label}</span>
      <span class="mini-meta">${d.cls === "scam" ? "red flags ≥ 50% override the price verdict" : "argmax of the verdict distribution"}</span>
    </div>
    <div class="gate-meter">
      <div class="gm-bar"><span style="width:${confPct}%"></span><i class="gm-gate" title="confidence gate"></i></div>
      <div class="gm-labels"><span>top probability ${confPct}%</span><span>gate 60%</span></div>
      ${unsure ? `<p class="explain">Below the gate. The badge steps aside and hands this one to a human.</p>` : ""}
    </div>
    <div class="chiplets">
      <span class="chiplet">${esc(r.value_label)} (${Number(r.value_score).toFixed(1)}/4)</span>
      <span class="chiplet${red >= 50 ? " alert" : ""}">red flags ${red}%</span>
    </div>
    <p class="explain" style="margin-top:14px">How long it took, end to end:</p>
    <div class="wf">
      <span class="wf-model" style="width:${modelW}%">model ${model} ms</span>
      <span class="wf-net" style="width:${100 - modelW}%">+${overhead} ms</span>
    </div>
    <div class="gm-labels"><span>${esc(r.engine)} ${batch ? "batch" : "forward pass"}</span><span>total ${total} ms</span></div>
    ${timingNote}
    <div class="callout">That is the System One pitch: a fast, typed, calibrated judgment,
    cheap enough to run on every product your shopper scrolls past.</div>`;
}

/* ---------- utils ---------- */

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function hi(json) {
  return esc(json).replace(
    /("(\\u[a-f0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(\.\d+)?([eE][+-]?\d+)?)/g,
    (m) => {
      let cls = "num";
      if (/^"/.test(m)) cls = /:$/.test(m) ? "key" : "str";
      else if (/^(true|false|null)$/.test(m)) cls = "bool";
      return `<span class="j-${cls}">${m}</span>`;
    });
}
