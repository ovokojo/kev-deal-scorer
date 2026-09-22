/* app.js — catalog page: mode pill + wiring. The grid, scoring queue,
   filters, tally and verdict rendering live in catalog.js. */

const $ = (id) => document.getElementById(id);

renderPill();
Catalog.init({
  els: {
    chips: $("chips"),
    search: $("search"),
    countLine: $("count-line"),
    rescore: $("rescore"),
  },
});

function renderPill() {
  fetch("/api/health").then((r) => r.json()).then((health) => {
    const pill = $("mode-pill"), text = $("mode-text");
    if (health.kev) {
      const short = (health.run || "kev").split("/").pop();
      text.textContent = `Kev live · ${short}`;
      pill.classList.add("live");
    } else {
      text.textContent = "Demo mode · heuristic";
      pill.classList.add("demo");
    }
  });
}
