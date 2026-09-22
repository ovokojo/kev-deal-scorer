/* lab.js — the "How Kev thinks" lesson panel.
   Auto-follows live scoring runs and walks through the six stages of a real
   System One request: listing → state → typed questions → one forward pass →
   structured output → verdict + timing. Requires a live Kev server; the gate
   card waits for :8009 and flips the page to live automatically. */

const $ = (id) => document.getElementById(id);

const STAGES = [
  { title: "The listing", render: renderListing },
  { title: "The state", render: renderState },
  { title: "Three typed questions", render: renderQuestions },
  { title: "One forward pass", render: renderPass },
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

renderDots();
bindControls();
Catalog.setPaused(true);       // no scoring until a live Kev server answers
Catalog.init({
  els: {
    chips: $("chips"),
    search: $("search"),
    countLine: $("count-line"),
    rescore: $("rescore"),
  },
  onResult,
  onCardClick: (p) => { if (lab.runs[p.id]) { setAuto(false); loadRun(p.id); } },
});
healthLoop();

function onResult(product, result) {
  lab.runs[product.id] = { product, result };
  if (!lab.live) return;
  lab.pending.push(product.id);
  tryNext();
}

/* ---------- health gate ---------- */

async function healthLoop() {
  try {
    const h = await fetch("/api/health").then((r) => r.json());
    setPill(h);
    setLive(h.kev && !!h.run);
  } catch (e) { /* backend down — stay gated */ }
  setTimeout(healthLoop, lab.live ? 10000 : 3000);
}

function setPill(health) {
  const pill = $("mode-pill"), text = $("mode-text");
  pill.classList.remove("live", "demo");
  if (health.kev) {
    const short = (health.run || "kev").split("/").pop();
    text.textContent = `Kev live · ${short}`;
    pill.classList.add("live");
  } else {
    text.textContent = "Demo mode · heuristic";
    pill.classList.add("demo");
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
  const demoBanner = run.result.mode === "demo"
    ? `<div class="demo-banner">⚠ Demo fallback answered this run — not Kev. Start the Kev server for real lessons.</div>`
    : "";
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
    (run.result.mode === "kev" ? " · live Kev run" : " · demo fallback");
  stage.scrollTop = 0;
}

/* ---------- stage renderers ---------- */

function renderListing({ product: p }) {
  return `
    <div class="mini-product">
      <div class="mini-media tint-0"><span class="emoji">${p.icon || "🛍️"}</span></div>
      <div class="mini-info">
        <div class="vendor">${esc(p.seller)}</div>
        <strong>${esc(p.title)}</strong>
        <span class="mini-meta">$${esc(String(p.price))} · ${esc(p.condition)} · ${esc(p.category || "")}</span>
      </div>
    </div>
    <p class="explain">The shopper sees a listing. The backend turns it into a ticket for Kev.</p>
    <div class="callout">🔒 The page knows this item’s typical market price
    ($${esc(String(p.market_price))}) — but it is <strong>never sent</strong> to the model.
    Kev has to judge the deal from its own knowledge, like a real shopper would.</div>`;
}

function renderState({ result: r }) {
  const st = r.trace.request.state;
  return `
    <p class="explain">The backend flattens the listing into one text block — the
    <strong>state</strong>. This string is everything the model will ever see.</p>
    <pre class="code">${esc(st)}</pre>
    <div class="callout">No chat history, no system prompt, no tools. Plain text in,
    decisions out — that is the whole input contract.</div>`;
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
      crit = `<li><b>yes</b> — flags present</li><li><b>no</b> — looks clean</li>`;
    }
    return `
      <div class="q-card">
        <div class="q-head"><span class="q-id">${esc(id)}</span><span class="q-type t-${q.type}">${q.type}</span></div>
        <p class="q-instructions">${esc(q.instructions)}</p>
        <ul class="q-criteria">${crit}</ul>
      </div>`;
  }).join("");
  return `
    <p class="explain">Three typed questions ride along with the state — each with its
    own rubric. Every answer comes back as a number, never a sentence.</p>
    ${cards}
    <div class="callout">The keys (<code>verdict</code>, <code>value</code>, <code>red_flags</code>) are ours —
    the model never sees them. Only instructions and options reach it.</div>`;
}

function renderPass({ result: r }) {
  const usage = r.trace.response.usage || {};
  const inTok = usage.input_tokens != null ? usage.input_tokens : "–";
  const outTok = usage.output_tokens != null ? usage.output_tokens : "–";
  return `
    <div class="flow">
      <div class="flow-state">state — encoded <em>once</em><span class="tok">${inTok} input tokens</span></div>
      <div class="flow-branches">
        ${["verdict · choice", "value · score", "red_flags · noul"].map((label) => `
          <div class="flow-q">
            <span class="flow-q-name">${label}</span>
            <div class="flow-opts"><i></i><i></i><i></i></div>
            <div class="flow-decide">&lt;decide&gt;</div>
          </div>`).join("")}
      </div>
      <div class="flow-head">pointer head · softmax → probabilities</div>
    </div>
    <p class="explain">The state and all three questions go into <strong>one token sequence</strong>.
    An attention mask lets each question read the state — but never the other questions.</p>
    <p class="explain">The pointer head scores every option against each question’s
    <code>&lt;decide&gt;</code> token; a softmax turns scores into calibrated probabilities.
    <strong>One forward pass, three answers.</strong></p>
    <div class="chips">
      <span class="chiplet">in: ${inTok} tokens</span>
      <span class="chiplet">out: ${outTok} tokens</span>
      <span class="chiplet">passes: 1</span>
    </div>`;
}

function renderOutput({ result: r }) {
  const answers = r.trace.response.answers || {};
  const usage = r.trace.response.usage || {};
  const heur = r.trace.response.heuristic;
  return `
    <p class="explain">What came back, verbatim — keyed by our question ids:</p>
    <pre class="code json">${hi(JSON.stringify(answers, null, 2))}</pre>
    ${heur ? `<div class="callout warn">Heuristic run: ${esc(heur.rule)} — ${esc(heur.note || "")}</div>` : ""}
    <div class="chips">
      ${usage.input_tokens != null ? `<span class="chiplet">usage: ${usage.input_tokens} in / ${usage.output_tokens} out</span>` : ""}
      <span class="chiplet">types: choice + score + noul</span>
    </div>
    <div class="callout">Typed decisions with calibrated probabilities — the code consumes
    them directly. No prose to parse, nothing to hallucinate.</div>`;
}

function renderVerdictTiming({ product: p, result: r }) {
  const probs = r.probabilities || {};
  const winner = Object.entries(probs).sort((a, b) => b[1] - a[1])[0];
  const badge = Catalog.VERDICT_BADGES[winner ? winner[0] : r.verdict] || r.verdict;
  const confPct = Math.round(r.confidence * 100);
  const unsure = r.confidence < Catalog.CONFIDENCE_GATE;

  const t = r.trace.timing || {};
  const total = t.round_trip_ms != null ? t.round_trip_ms : r.latency_ms;
  const model = t.model_ms != null ? t.model_ms : total;
  const overhead = Math.max(0, total - model);
  const modelW = total > 0 ? Math.min(100, Math.round((model / total) * 100)) : 100;

  const timingNote = r.mode === "kev"
    ? `<p class="explain">“Model time” is Kev’s own measure of the forward pass; the rest is HTTP
       and backend overhead. Chat-LLM baselines for the same judgment run thousands of ms.</p>`
    : `<p class="explain">The heuristic answers in ~0 ms — which is why the catalog enforces a minimum
       “thinking” time. Real Kev runs land in the 100–800 ms range on Apple Silicon.</p>`;

  return `
    <p class="explain">The UI reads the structured answer directly:</p>
    <div class="final-badge">
      <span class="badge ${r.verdict} ${unsure ? "" : "pop"}">${unsure ? "❓ Not sure" : badge}</span>
      <span class="mini-meta">argmax of the probability distribution</span>
    </div>
    <div class="gate-meter">
      <div class="gm-bar"><span style="width:${confPct}%"></span><i class="gm-gate" title="confidence gate"></i></div>
      <div class="gm-labels"><span>confidence ${confPct}%</span><span>gate 60%</span></div>
      ${unsure ? `<p class="explain">Below the gate — the badge steps aside and hands this one to a human.</p>` : ""}
    </div>
    <div class="chips">
      <span class="chiplet">${esc(r.value_label)} (${Number(r.value_score).toFixed(1)}/4)</span>
      <span class="chiplet${r.red_flag_prob >= 0.5 ? " alert" : ""}">red flags ${Math.round(r.red_flag_prob * 100)}%</span>
    </div>
    <p class="explain" style="margin-top:14px">How long it took, end to end:</p>
    <div class="wf">
      <span class="wf-model" style="width:${modelW}%">model ${model} ms</span>
      <span class="wf-net" style="width:${100 - modelW}%">+${overhead} ms</span>
    </div>
    <div class="gm-labels"><span>Kev forward pass</span><span>total ${total} ms</span></div>
    ${timingNote}
    <div class="callout">That is the System One pitch: a fast, typed, calibrated judgment —
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
