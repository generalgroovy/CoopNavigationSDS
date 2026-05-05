const componentOrder = [
  "Foundations and Architectures",
  "ASR and Speech Input",
  "NLU and State Tracking",
  "Dialog Management",
  "Response Generation",
  "Speech Output and Prosody",
  "End-to-End Spoken Models",
  "Human Evaluation",
  "Automatic Evaluation",
  "Evaluation Frameworks",
  "User Simulation",
  "Datasets and Shared Tasks"
];

const componentColors = {
  "Foundations and Architectures": "#0f766e",
  "ASR and Speech Input": "#2563eb",
  "NLU and State Tracking": "#7c3aed",
  "Dialog Management": "#c2410c",
  "Response Generation": "#be123c",
  "Speech Output and Prosody": "#a16207",
  "End-to-End Spoken Models": "#4338ca",
  "Human Evaluation": "#15803d",
  "Automatic Evaluation": "#0369a1",
  "Evaluation Frameworks": "#b45309",
  "User Simulation": "#9333ea",
  "Datasets and Shared Tasks": "#475569"
};

const literature = window.SDS_LITERATURE
  .slice()
  .sort((a, b) => {
    const componentDelta = componentOrder.indexOf(a.component) - componentOrder.indexOf(b.component);
    return componentDelta || a.year - b.year || a.title.localeCompare(b.title);
  });

const state = {
  query: "",
  component: "All",
  evaluation: "All",
  type: "All"
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function unique(values) {
  return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
}

function populateFilters() {
  const componentSelect = $("#componentFilter");
  const typeSelect = $("#typeFilter");
  unique(literature.map((item) => item.component)).forEach((component) => {
    componentSelect.append(new Option(component, component));
  });
  unique(literature.map((item) => item.type)).forEach((type) => {
    typeSelect.append(new Option(type, type));
  });
}

function filteredItems() {
  const q = state.query.trim().toLowerCase();
  return literature.filter((item) => {
    const haystack = [item.title, item.authors, item.venue, item.note, item.component, item.type].join(" ").toLowerCase();
    const queryOk = !q || haystack.includes(q);
    const componentOk = state.component === "All" || item.component === state.component;
    const evaluationOk = state.evaluation === "All" || item.evaluation.includes(state.evaluation);
    const typeOk = state.type === "All" || item.type === state.type;
    return queryOk && componentOk && evaluationOk && typeOk;
  });
}

function renderStats(items) {
  const years = literature.map((item) => item.year);
  const human = literature.filter((item) => item.evaluation.includes("human")).length;
  const automatic = literature.filter((item) => item.evaluation.includes("automatic")).length;
  $("#stats").innerHTML = `
    <div><strong>${literature.length}</strong><span>curated records</span></div>
    <div><strong>${Math.min(...years)}-${Math.max(...years)}</strong><span>field coverage</span></div>
    <div><strong>${human}</strong><span>human-eval links</span></div>
    <div><strong>${automatic}</strong><span>automatic-eval links</span></div>
    <div><strong>${items.length}</strong><span>visible now</span></div>
  `;
}

function renderCards(items) {
  const grouped = items.reduce((groups, item) => {
    if (!groups.has(item.component)) groups.set(item.component, []);
    groups.get(item.component).push(item);
    return groups;
  }, new Map());
  $("#records").innerHTML = componentOrder
    .filter((component) => grouped.has(component))
    .map((component) => {
      const cards = grouped.get(component).map((item) => `
        <article class="paper-card" data-year="${item.year}">
          <div class="paper-meta">
            <span class="year">${item.year}</span>
            <span>${item.type}</span>
            ${item.evaluation.map((tag) => `<span>${tag}</span>`).join("")}
          </div>
          <h3><a href="${item.url}" target="_blank" rel="noreferrer">${item.title}</a></h3>
          <p class="authors">${item.authors}</p>
          <p class="venue">${item.venue}</p>
          <p>${item.note}</p>
        </article>
      `).join("");
      return `
        <section class="component-section">
          <header>
            <span class="component-dot" style="--dot:${componentColors[component] || "#64748b"}"></span>
            <h2>${component}</h2>
            <p>${grouped.get(component).length} records</p>
          </header>
          <div class="card-grid">${cards}</div>
        </section>
      `;
    }).join("");
}

function renderTimeline(items) {
  const minYear = Math.min(...literature.map((item) => item.year));
  const maxYear = Math.max(...literature.map((item) => item.year));
  const width = 1100;
  const rowHeight = 34;
  const padding = { left: 210, right: 24, top: 28, bottom: 30 };
  const rows = componentOrder.filter((component) => literature.some((item) => item.component === component));
  const height = padding.top + rows.length * rowHeight + padding.bottom;
  const x = (year) => padding.left + ((year - minYear) / (maxYear - minYear)) * (width - padding.left - padding.right);
  const y = (component) => padding.top + rows.indexOf(component) * rowHeight + 14;
  const activeIds = new Set(items.map((item) => item.id));

  const yearTicks = [];
  for (let year = 1980; year <= maxYear; year += 10) yearTicks.push(year);

  $("#timeline").innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Literature timeline">
      ${yearTicks.map((year) => `
        <line x1="${x(year)}" y1="10" x2="${x(year)}" y2="${height - 22}" class="tick-line"></line>
        <text x="${x(year)}" y="${height - 8}" class="tick-text">${year}</text>
      `).join("")}
      ${rows.map((component) => `
        <text x="12" y="${y(component) + 4}" class="row-label">${component}</text>
        <line x1="${padding.left}" y1="${y(component)}" x2="${width - padding.right}" y2="${y(component)}" class="row-line"></line>
      `).join("")}
      ${literature.map((item) => `
        <a href="${item.url}" target="_blank" rel="noreferrer">
          <circle cx="${x(item.year)}" cy="${y(item.component)}" r="${activeIds.has(item.id) ? 6 : 3}" 
            fill="${componentColors[item.component] || "#64748b"}" 
            opacity="${activeIds.has(item.id) ? 0.95 : 0.18}">
            <title>${item.year}: ${item.title}</title>
          </circle>
        </a>
      `).join("")}
    </svg>
  `;
}

function renderMatrix(items) {
  const dimensions = [
    ["Task success", "Goal completion, exact answer, inform/success, API-call validity"],
    ["Efficiency", "Turns, elapsed time, barge-in rate, repair count"],
    ["Robustness", "WER/SER sensitivity, out-of-domain handling, recovery"],
    ["Interaction quality", "Naturalness, coherence, grounding, initiative, latency"],
    ["User experience", "Satisfaction, trust, cognitive load, preference, accessibility"],
    ["Safety and reliability", "Hallucination, privacy, bias, abuse resistance, escalation"]
  ];
  $("#matrix").innerHTML = dimensions.map(([name, detail]) => `
    <div class="matrix-row">
      <strong>${name}</strong>
      <span>${detail}</span>
    </div>
  `).join("");

  const humanCount = items.filter((item) => item.evaluation.includes("human")).length;
  const automaticCount = items.filter((item) => item.evaluation.includes("automatic")).length;
  const bothCount = items.filter((item) => item.evaluation.includes("human") && item.evaluation.includes("automatic")).length;
  $("#evalBars").innerHTML = [
    ["Human", humanCount],
    ["Automatic", automaticCount],
    ["Hybrid", bothCount]
  ].map(([label, value]) => {
    const pct = items.length ? Math.round((value / items.length) * 100) : 0;
    return `<div class="bar"><span>${label}</span><div><i style="width:${pct}%"></i></div><b>${value}</b></div>`;
  }).join("");
}

function renderExport(items) {
  const csv = [
    ["year", "component", "type", "evaluation", "title", "authors", "venue", "url", "note"],
    ...items.map((item) => [
      item.year,
      item.component,
      item.type,
      item.evaluation.join("|"),
      item.title,
      item.authors,
      item.venue,
      item.url,
      item.note
    ])
  ].map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = $("#downloadCsv");
  link.href = url;
  link.download = "speech-dialog-literature.csv";
}

function render() {
  const items = filteredItems();
  renderStats(items);
  renderCards(items);
  renderTimeline(items);
  renderMatrix(items);
  renderExport(items);
}

function bindEvents() {
  $("#search").addEventListener("input", (event) => {
    state.query = event.target.value;
    render();
  });
  $("#componentFilter").addEventListener("change", (event) => {
    state.component = event.target.value;
    render();
  });
  $("#evalFilter").addEventListener("change", (event) => {
    state.evaluation = event.target.value;
    render();
  });
  $("#typeFilter").addEventListener("change", (event) => {
    state.type = event.target.value;
    render();
  });
  $$(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".tab").forEach((node) => node.classList.remove("active"));
      $$(".view").forEach((node) => node.classList.remove("active"));
      tab.classList.add("active");
      $(`#${tab.dataset.target}`).classList.add("active");
    });
  });
}

populateFilters();
bindEvents();
render();
