import "./style.css";
import { Space3D, MOOD } from "./space3d.js";

const POSTER_BASE = "https://image.tmdb.org/t/p/w342";

const state = {
  movies: [],
  tagNames: [],
  arcTypes: [],
  selected: null, // movie index (string) or null
  highlighted: new Set(),
  // Axis bounds match the slider range, which is wider than [-1, 1] because
  // the display transform pushes the most extreme films past the axis ring.
  filters: {
    levity: [-1.7, 1.7],
    threat: [-1.7, 1.7],
    intimacy: [-1.7, 1.7],
    rating: [1, 10],
  },
  // Curated lists. Picking one scopes the whole app to its films — the cloud,
  // search, and the recommendations narrow together, and the three axis
  // filters keep working inside that smaller world.
  //
  // `listNN` is why this needs pipeline support rather than a client-side
  // filter: a ~100-film list is ~2% of the catalog, so a movie's global top-10
  // neighbors contain roughly none of it. export_web.py re-ranks every channel
  // over members only and fuses them with the same weights, so the
  // recommendations are the same model looking at a smaller world.
  lists: [],
  activeList: null, // slug, or null for the whole catalog
  listMembers: null, // Set of movie indices, or null
  listNN: null, // movie index (string) -> in-list neighbors, or null
};

// Aspect steering: the user picks tags they liked about the selected movie;
// recommendations re-rank by genome similarity with those dimensions boosted.
// The quantized genome matrix is lazy-fetched on first use.
const steer = {
  tags: new Set(), // selected tag indices
  matrix: null, // Uint8Array, rows * dim
  scales: null, // Float32Array per-row dequantization scale
  rows: [], // genome row -> movie index
  rowOf: new Map(), // movie index -> genome row
  dim: 0,
  loading: false,
};

const STEER_BOOST = 6;

let space;

// Percentile rank per axis, computed once at boot. The 3D coordinate is a
// tail-stretched value that runs past ±1 so outliers separate visually, which
// makes it useless as a readout — the panel should say "how does this compare
// to every other film", and that's the rank. The stretch is monotone, so
// ranking the shipped values recovers it exactly.
const axisPct = { levity: [], threat: [], intimacy: [] };

function computeAxisPercentiles() {
  const n = state.movies.length;
  for (const axis of ["levity", "threat", "intimacy"]) {
    const order = state.movies
      .map((m, i) => [m[axis], i])
      .sort((a, b) => a[0] - b[0]);
    const out = new Array(n);
    order.forEach(([, i], r) => (out[i] = Math.round(((r + 0.5) / n) * 100)));
    axisPct[axis] = out;
  }
}

async function loadSteeringMatrix() {
  if (steer.matrix || steer.loading) return;
  steer.loading = true;
  const resp = await fetch(import.meta.env.BASE_URL + "data/genome_q8.bin");
  const buf = await resp.arrayBuffer();
  const head = new DataView(buf);
  const n = head.getUint32(0, true);
  steer.dim = head.getUint32(4, true);
  steer.scales = new Float32Array(buf, 8, n);
  steer.matrix = new Uint8Array(buf, 8 + n * 4);
  steer.loading = false;
  if (state.selected !== null) selectMovie(Number(state.selected));
}

function steeredNeighbors(idx, k = 10) {
  const gi = steer.rowOf.get(idx);
  if (gi === undefined || !steer.matrix) return null;
  const { matrix, scales, dim, rows } = steer;
  const ref = new Float32Array(dim);
  const base = gi * dim;
  for (let d = 0; d < dim; d++) ref[d] = matrix[base + d];
  for (const t of steer.tags) ref[t] *= 1 + STEER_BOOST;

  const scored = [];
  for (let r = 0; r < rows.length; r++) {
    if (r === gi) continue;
    if (state.listMembers && !state.listMembers.has(rows[r])) continue;
    const off = r * dim;
    let s = 0;
    for (let d = 0; d < dim; d++) s += ref[d] * matrix[off + d];
    scored.push([s * scales[r], r]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  return scored.slice(0, k).map(([, r]) => rows[r]);
}

// One recommendation model: the precomputed fusion of story, vibe, and
// audience (export_web.py), optionally re-ranked by the user's picked tags.
function neighborsOf(idx) {
  if (steer.tags.size > 0) {
    const steered = steeredNeighbors(idx);
    if (steered) return steered;
  }
  // Inside a list, use its own in-list neighbors — never the global ones,
  // which point almost entirely outside the list.
  if (state.listNN) return state.listNN[String(idx)] ?? [];
  return state.movies[idx].nn;
}

// ---- selection ----

function clearSelection() {
  state.selected = null;
  state.highlighted.clear();
  steer.tags.clear();
  document.getElementById("detail").hidden = true;
  space?.syncSelection();
  updateFilterCount();
}

function selectMovie(idx) {
  const key = String(idx);
  if (state.selected !== key) steer.tags.clear(); // steering is per-movie
  state.selected = key;
  state.highlighted = new Set([key]);
  for (const nb of neighborsOf(idx)) state.highlighted.add(String(nb));
  renderDetail(idx);
  space?.syncSelection();
  updateFilterCount();
}

function flyTo(idx) {
  space?.flyTo(idx);
}

// ---- detail panel ----

// Tags two movies have in common, strongest first (strength = the weaker of
// the two relevances — both movies must actually have the quality).
function sharedTags(a, b, n = 3) {
  if (!a.tags.length || !b.tags.length) return [];
  const inA = new Map(a.tags);
  return b.tags
    .filter(([t]) => inA.has(t))
    .map(([t, s]) => [t, Math.min(s, inA.get(t))])
    .sort((x, y) => y[1] - x[1])
    .slice(0, n)
    .map(([t]) => state.tagNames[t]);
}

// The tension curve over runtime, from subtitle timing (silence and distress
// vocabulary up, chatter down). Drawn as a filled sparkline: left edge is the
// opening, right edge the finale.
function arcSparkline(m) {
  if (!m.arc) return "";
  const w = 268, h = 34, pad = 3;
  const lo = Math.min(...m.arc), hi = Math.max(...m.arc);
  const span = Math.max(hi - lo, 0.001);
  const pts = m.arc.map((v, i) => {
    const x = pad + (i / (m.arc.length - 1)) * (w - pad * 2);
    const y = h - pad - ((v - lo) / span) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const label = state.arcTypes[m.arcType] ?? "";
  return `<div class="arc">
    <svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
      <polygon points="${pad},${h} ${pts.join(" ")} ${w - pad},${h}" />
      <polyline points="${pts.join(" ")}" />
    </svg>
    <div class="arc-label"><span>${escapeHtml(label)}</span><span>tension · start → end</span></div>
  </div>`;
}

function renderDetail(idx) {
  const m = state.movies[idx];
  const el = document.getElementById("detail-body");
  const neighbors = neighborsOf(idx);

  const poster = m.poster
    ? `<img class="poster" src="${POSTER_BASE}${m.poster}" alt="" loading="lazy" />`
    : "";
  const directors = m.directors.length ? ` · ${m.directors.join(", ")}` : "";
  const axisReadout = `<div class="axis-readout">
    <span>Levity <b class="levity">${axisPct.levity[idx]}</b></span>
    <span>Threat <b class="threat">${axisPct.threat[idx]}</b></span>
    <span>Intimacy <b class="intimacy">${axisPct.intimacy[idx]}</b></span>
  </div>`;
  const chips = m.tags.length
    ? `<div class="chips" role="group" aria-label="What did you like about this?">${m.tags
        .slice(0, 8)
        .map(([t]) => `<button class="chip${steer.tags.has(t) ? " active" : ""}"
          aria-pressed="${steer.tags.has(t)}"
          data-tag="${t}">${escapeHtml(state.tagNames[t])}</button>`)
        .join("")}</div>
      <p class="chips-hint">pick what you liked — recommendations follow</p>`
    : "";
  const neighborItems = neighbors
    .map((nb) => {
      const n = state.movies[nb];
      const shared = sharedTags(m, n);
      return `<div class="neighbor" data-idx="${nb}">
        <div class="nrow">
          <span>${escapeHtml(n.title)}</span><span class="year">${n.year ?? ""}</span>
        </div>
        ${shared.length ? `<div class="shared">${shared.map(escapeHtml).join(" · ")}</div>` : ""}
      </div>`;
    })
    .join("");
  const steering = steer.tags.size > 0 && steer.rowOf.has(idx);
  const heading = steering
    ? `More ${[...steer.tags].map((t) => escapeHtml(state.tagNames[t])).join(" + ")}`
    : "Recommended";

  el.innerHTML = `
    ${poster}
    <h2>${escapeHtml(m.title)}</h2>
    <p class="meta">${m.year ?? "—"} · ${m.genres.join(", ")}${directors}
      · ★ ${m.rating.toFixed(1)}</p>
    ${axisReadout}
    ${arcSparkline(m)}
    ${chips}
    <p class="overview">${escapeHtml(m.overview)}</p>
    <h3>${heading}${steering && steer.loading ? " (loading…)" : ""}</h3>
    ${neighborItems}
  `;

  el.querySelectorAll(".chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const t = Number(btn.dataset.tag);
      if (steer.tags.has(t)) steer.tags.delete(t);
      else steer.tags.add(t);
      if (steer.tags.size > 0) loadSteeringMatrix();
      selectMovie(idx); // re-rank + re-highlight with steering applied
    });
  });

  el.querySelectorAll(".neighbor").forEach((div) => {
    div.addEventListener("click", () => {
      const nb = Number(div.dataset.idx);
      selectMovie(nb);
      flyTo(nb);
    });
  });

  document.getElementById("detail").hidden = false;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

// ---- search ----

function setupSearch() {
  const input = document.getElementById("search");
  const results = document.getElementById("search-results");
  let focused = -1; // index into the current <li> list, -1 = none

  function close() {
    results.hidden = true;
    results.innerHTML = "";
    focused = -1;
  }

  function setFocused(i) {
    const items = results.querySelectorAll("li");
    if (!items.length) return;
    focused = (i + items.length) % items.length;
    items.forEach((li, j) => li.classList.toggle("focused", j === focused));
    items[focused].scrollIntoView({ block: "nearest" });
  }

  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    if (q.length < 2) return close();
    const hits = [];
    for (let i = 0; i < state.movies.length && hits.length < 12; i++) {
      if (state.movies[i].title.toLowerCase().includes(q)) hits.push(i);
    }
    focused = -1;
    results.innerHTML = hits
      .map(
        (i) => `<li data-idx="${i}">
          <span>${escapeHtml(state.movies[i].title)}</span>
          <span class="year">${state.movies[i].year ?? ""}</span>
        </li>`
      )
      .join("");
    results.hidden = hits.length === 0;
    results.querySelectorAll("li").forEach((li) => {
      li.addEventListener("click", () => {
        const idx = Number(li.dataset.idx);
        close();
        // Don't echo the title back into the box — picking a dot or a
        // neighbor changes the selection too, and a stale query would then
        // name the wrong movie. The detail panel is the source of truth.
        input.value = "";
        selectMovie(idx);
        flyTo(idx);
      });
    });
  });

  input.addEventListener("keydown", (e) => {
    if (results.hidden) return;
    if (e.key === "Escape") return close();
    if (e.key === "ArrowDown") {
      e.preventDefault();
      return setFocused(focused + 1);
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      return setFocused(focused - 1);
    }
    if (e.key === "Enter") {
      const item = results.querySelectorAll("li")[focused] ?? results.querySelector("li");
      if (item) item.click();
    }
  });

  document.addEventListener("click", (e) => {
    if (!results.contains(e.target) && e.target !== input) close();
  });
}

// ---- legend ----

function renderLegend() {
  const entries = [
    ["Playful", MOOD.levity.hex],
    ["Tense", MOOD.threat.hex],
    ["Intimate", MOOD.intimacy.hex],
    ["Neutral", MOOD.neutral.hex],
  ];
  const legend = document.getElementById("legend");
  legend.innerHTML =
    `<button class="legend-toggle" aria-expanded="false">Mood</button>
     <div class="legend-items">` +
    entries.map(([g, c]) => `<div><span class="swatch" style="background:${c}"></span>${g}</div>`).join("") +
    `<div class="legend-note">hues blend · size = rating</div></div>`;
  legend.querySelector(".legend-toggle").addEventListener("click", () => {
    const open = legend.classList.toggle("expanded");
    legend.querySelector(".legend-toggle").setAttribute("aria-expanded", String(open));
  });
}

// ---- axis/rating filters ----

function updateFilterCount() {
  const text = `${space?.visibleCount ?? state.movies.length} of ${state.movies.length} shown`;
  document.getElementById("filter-count").textContent = text;
  // Mobile-collapsed filters still need a hint of what's active without
  // opening the sheet.
  document.getElementById("filter-count-mini").textContent =
    space && space.visibleCount < state.movies.length ? `· ${space.visibleCount}` : "";
}

function setupFilters() {
  const toggle = document.getElementById("filters-toggle");
  const body = document.getElementById("filters-body");
  toggle.addEventListener("click", () => {
    const open = body.classList.toggle("open");
    toggle.setAttribute("aria-expanded", String(open));
  });

  for (const row of document.querySelectorAll("#filters .frow")) {
    const axis = row.dataset.axis;
    const lo = row.querySelector(".lo");
    const hi = row.querySelector(".hi");
    const fill = row.querySelector(".fill");
    const min = parseFloat(lo.min);
    const span = parseFloat(lo.max) - min;

    const update = () => {
      const a = parseFloat(lo.value);
      const b = parseFloat(hi.value);
      state.filters[axis] = [Math.min(a, b), Math.max(a, b)];
      fill.style.left = `${((Math.min(a, b) - min) / span) * 100}%`;
      fill.style.right = `${100 - ((Math.max(a, b) - min) / span) * 100}%`;
      space?.restyle();
      updateFilterCount();
    };
    lo.addEventListener("input", update);
    hi.addEventListener("input", update);
    update();
  }
}

// ---- boot ----

async function main() {
  const resp = await fetch(import.meta.env.BASE_URL + "data/movies.json");
  const data = await resp.json();
  state.movies = data.movies;
  state.tagNames = data.tags;
  state.arcTypes = data.arcTypes ?? [];
  computeAxisPercentiles();
  steer.rows = data.genome_rows;
  data.genome_rows.forEach((mi, r) => steer.rowOf.set(mi, r));

  space = new Space3D(document.getElementById("space"), state, {
    onPick: (idx) => {
      selectMovie(idx);
      flyTo(idx); // center the idle orbit on the picked movie
    },
    onClear: clearSelection,
    neighborsOf,
  });
  space.show();
  space.restyle();

  document.getElementById("detail-close").addEventListener("click", clearSelection);

  setupSearch();
  setupFilters();
  renderLegend();
  document.body.classList.add("ready");
  setTimeout(() => document.getElementById("loading")?.remove(), 800);
}

main().catch((err) => {
  document.querySelector("#loading .load-status").textContent =
    "couldn't load the map — check your connection and reload";
  console.error(err);
});
