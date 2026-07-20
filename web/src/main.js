import "./style.css";
import Graph from "graphology";
import Sigma from "sigma";
import { Space3D, MOOD } from "./space3d.js";

// Dataviz reference palette (dark column), assigned semantically to the most
// common primary genres; everything else folds into muted "Other". Color is
// only a grouping cue — identity always comes from labels, hover, and the
// detail panel.
const GENRE_COLORS = {
  Drama: "#3987e5", // blue
  Comedy: "#c98500", // yellow
  Action: "#d95926", // orange
  Adventure: "#199e70", // aqua
  Horror: "#e66767", // red
  Thriller: "#9085e9", // violet
  "Science Fiction": "#008300", // green
  Animation: "#d55181", // magenta
};
const OTHER_COLOR = "#898781";
const DIM_COLOR = "#2c2c2a";
const EDGE_COLOR = "#52514e";

const POSTER_BASE = "https://image.tmdb.org/t/p/w342";

const state = {
  movies: [],
  tagNames: [],
  lens: "best", // "best" | "text" | "vibe" | "als"
  view: "map", // "map" | "galaxy" | "space"
  selected: null, // node key (string) or null
  highlighted: new Set(),
  filters: {
    levity: [-1, 1],
    threat: [-1, 1],
    intimacy: [-1, 1],
    rating: [1, 10],
  },
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
    const off = r * dim;
    let s = 0;
    for (let d = 0; d < dim; d++) s += ref[d] * matrix[off + d];
    scored.push([s * scales[r], r]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  return scored.slice(0, k).map(([, r]) => rows[r]);
}

let graph;
let renderer;
let space; // 3D view (lazy-initialized on first open)

function primaryGenre(genres) {
  for (const g of genres) if (GENRE_COLORS[g]) return g;
  return null;
}

function nodeColor(movie) {
  const g = primaryGenre(movie.genres);
  return g ? GENRE_COLORS[g] : OTHER_COLOR;
}

function neighborsOf(idx) {
  if (steer.tags.size > 0) {
    const steered = steeredNeighbors(idx);
    if (steered) return steered;
  }
  const m = state.movies[idx];
  if (state.lens === "best") return m.nn_best;
  if (state.lens === "vibe") return m.nn_vibe;
  if (state.lens === "als") return m.nn_als;
  return m.nn_text;
}

const LENS_LABELS = {
  best: "Recommended",
  text: "Similar story",
  vibe: "Similar vibe",
  als: "Similar audience",
};

const LENS_EMPTY = {
  vibe: "This movie isn't in the MovieLens tag genome — try the story lens.",
  als: `Too few MovieLens ratings for the audience lens —
        coverage is thin for recent releases. Try the story lens.`,
};

// ---- selection ----

function clearSelection() {
  state.selected = null;
  state.highlighted.clear();
  steer.tags.clear();
  graph.clearEdges();
  document.getElementById("lens-vibe").classList.remove("no-data");
  document.getElementById("lens-als").classList.remove("no-data");
  document.getElementById("detail").hidden = true;
  if (state.view !== "space") renderer.refresh();
  space?.syncSelection();
}

function selectMovie(idx) {
  const key = String(idx);
  if (state.selected !== key) steer.tags.clear(); // steering is per-movie
  state.selected = key;
  state.highlighted = new Set([key]);
  graph.clearEdges();
  for (const nb of neighborsOf(idx)) {
    state.highlighted.add(String(nb));
    graph.addEdge(key, String(nb), { color: EDGE_COLOR, size: 1 });
  }
  const m = state.movies[idx];
  document.getElementById("lens-vibe").classList.toggle("no-data", m.nn_vibe.length === 0);
  document.getElementById("lens-als").classList.toggle("no-data", m.nn_als.length === 0);
  renderDetail(idx);
  if (state.view !== "space") renderer.refresh();
  space?.syncSelection();
}

function flyTo(idx) {
  if (state.view === "space") {
    space.flyTo(idx);
    return;
  }
  const pos = renderer.getNodeDisplayData(String(idx));
  if (pos) {
    renderer.getCamera().animate({ x: pos.x, y: pos.y, ratio: 0.25 }, { duration: 600 });
  }
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

function renderDetail(idx) {
  const m = state.movies[idx];
  const el = document.getElementById("detail-body");
  const neighbors = neighborsOf(idx);

  const poster = m.poster
    ? `<img class="poster" src="${POSTER_BASE}${m.poster}" alt="" loading="lazy" />`
    : "";
  const directors = m.directors.length ? ` · ${m.directors.join(", ")}` : "";
  const chips = m.tags.length
    ? `<div class="chips">${m.tags
        .slice(0, 8)
        .map(([t]) => `<button class="chip${steer.tags.has(t) ? " active" : ""}"
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
  const neighborBlock =
    neighbors.length > 0
      ? neighborItems
      : `<p class="no-lens">${LENS_EMPTY[state.lens] ?? ""}</p>`;
  const steering = steer.tags.size > 0 && steer.rowOf.has(idx);
  const heading = steering
    ? `More ${[...steer.tags].map((t) => escapeHtml(state.tagNames[t])).join(" + ")}`
    : LENS_LABELS[state.lens];

  el.innerHTML = `
    ${poster}
    <h2>${escapeHtml(m.title)}</h2>
    <p class="meta">${m.year ?? "—"} · ${m.genres.join(", ")}${directors}
      · ★ ${m.rating.toFixed(1)}</p>
    ${chips}
    <p class="overview">${escapeHtml(m.overview)}</p>
    <h3>${heading}${steering && steer.loading ? " (loading…)" : ""}</h3>
    ${neighborBlock}
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

  function close() {
    results.hidden = true;
    results.innerHTML = "";
  }

  input.addEventListener("input", () => {
    const q = input.value.trim().toLowerCase();
    if (q.length < 2) return close();
    const hits = [];
    for (let i = 0; i < state.movies.length && hits.length < 12; i++) {
      if (state.movies[i].title.toLowerCase().includes(q)) hits.push(i);
    }
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
        input.value = state.movies[idx].title;
        selectMovie(idx);
        flyTo(idx);
      });
    });
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close();
    if (e.key === "Enter") {
      const first = results.querySelector("li");
      if (first) first.click();
    }
  });

  document.addEventListener("click", (e) => {
    if (!results.contains(e.target) && e.target !== input) close();
  });
}

// ---- view switcher (map <-> galaxy) ----

let tweenRaf = null;

function animatePositions(target, duration = 1100) {
  cancelAnimationFrame(tweenRaf);
  const from = state.movies.map((_, i) => ({
    x: graph.getNodeAttribute(String(i), "x"),
    y: graph.getNodeAttribute(String(i), "y"),
  }));
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const start = performance.now();

  function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const k = ease(t);
    state.movies.forEach((m, i) => {
      const [tx, ty] = target(m);
      graph.mergeNodeAttributes(String(i), {
        x: from[i].x + (tx - from[i].x) * k,
        y: from[i].y + (ty - from[i].y) * k,
      });
    });
    if (t < 1) tweenRaf = requestAnimationFrame(frame);
  }
  tweenRaf = requestAnimationFrame(frame);
}

function setupView() {
  const buttons = {
    map: document.getElementById("view-map"),
    galaxy: document.getElementById("view-galaxy"),
    space: document.getElementById("view-space"),
  };
  let sigmaLayout = "map"; // which coordinate set the sigma graph currently holds

  function applySigmaLayout(view) {
    const galaxy = view === "galaxy";
    graph.forEachNode((node, attrs) => {
      if (attrs.marker) graph.setNodeAttribute(node, "hidden", !galaxy);
    });
    if (sigmaLayout !== view) {
      animatePositions(galaxy ? (m) => [m.gx, m.gy] : (m) => [m.x, m.y]);
      sigmaLayout = view;
    }
    renderer.getCamera().animate({ x: 0.5, y: 0.5, ratio: 1 }, { duration: 1100 });
  }

  for (const [view, btn] of Object.entries(buttons)) {
    btn.addEventListener("click", () => {
      if (state.view === view) return;
      const prev = state.view;
      state.view = view;
      for (const [v, b] of Object.entries(buttons)) {
        b.classList.toggle("active", v === view);
      }

      const filters = document.getElementById("filters");
      if (view === "space") {
        document.getElementById("map").style.display = "none";
        space ??= new Space3D(document.getElementById("space"), state, {
          onPick: (idx) => {
            selectMovie(idx);
            flyTo(idx); // center the idle orbit on the picked movie
          },
          onClear: clearSelection,
          neighborsOf,
        });
        space.show();
        space.restyle();
        filters.hidden = false;
        updateFilterCount();
        renderLegend("mood");
      } else {
        filters.hidden = true;
        renderLegend("genre");
        if (prev === "space") {
          space.hide();
          document.getElementById("map").style.display = "";
          renderer.refresh();
        }
        applySigmaLayout(view);
      }
    });
  }
}

// ---- lens toggle ----

function setupLens() {
  const buttons = {
    best: document.getElementById("lens-best"),
    text: document.getElementById("lens-text"),
    vibe: document.getElementById("lens-vibe"),
    als: document.getElementById("lens-als"),
  };
  for (const [lens, btn] of Object.entries(buttons)) {
    btn.addEventListener("click", () => {
      if (state.lens === lens) return;
      state.lens = lens;
      for (const [l, b] of Object.entries(buttons)) {
        b.classList.toggle("active", l === lens);
      }
      if (state.selected !== null) selectMovie(Number(state.selected));
    });
  }
}

// ---- legend ----

function renderLegend(mode = "genre") {
  let entries;
  let note = "";
  if (mode === "mood") {
    entries = [
      ["Playful", MOOD.levity.hex],
      ["Tense", MOOD.threat.hex],
      ["Intimate", MOOD.intimacy.hex],
      ["Neutral", MOOD.neutral.hex],
    ];
    note = `<div class="legend-note">hues blend · size = rating</div>`;
  } else {
    const counts = {};
    for (const m of state.movies) {
      const g = primaryGenre(m.genres) ?? "Other";
      counts[g] = (counts[g] ?? 0) + 1;
    }
    entries = Object.keys(GENRE_COLORS)
      .filter((g) => counts[g])
      .map((g) => [g, GENRE_COLORS[g]]);
    entries.push(["Other", OTHER_COLOR]);
  }
  document.getElementById("legend").innerHTML =
    entries
      .map(([g, c]) => `<div><span class="swatch" style="background:${c}"></span>${g}</div>`)
      .join("") + note;
}

// ---- axis/rating filters (3D view) ----

function updateFilterCount() {
  const el = document.getElementById("filter-count");
  el.textContent = `${space?.visibleCount ?? state.movies.length} of ${state.movies.length} shown`;
}

function setupFilters() {
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
  steer.rows = data.genome_rows;
  data.genome_rows.forEach((mi, r) => steer.rowOf.set(mi, r));

  const maxVotes = Math.max(...state.movies.map((m) => m.votes));
  graph = new Graph();
  state.movies.forEach((m, i) => {
    graph.addNode(String(i), {
      x: m.x,
      y: m.y,
      size: 1.5 + 5 * Math.sqrt(m.votes / maxVotes),
      label: m.title,
      color: nodeColor(m),
    });
  });

  // Decade ring labels for the galaxy view (hidden on the flat map).
  for (const d of data.decades) {
    graph.addNode(`decade-${d.label}`, {
      x: 0,
      y: d.r,
      size: 1,
      color: "#383835",
      label: d.label,
      marker: true,
      hidden: true,
      forceLabel: true,
    });
  }

  renderer = new Sigma(graph, document.getElementById("map"), {
    // #map is display:none while the 3D view is active, so its own resize
    // observer sees a zero-width container and throws without this.
    allowInvalidContainer: true,
    labelColor: { color: "#a6a59c" },
    labelFont: "Geist, system-ui, sans-serif",
    labelWeight: "500",
    labelSize: 12,
    labelRenderedSizeThreshold: 5,
    zIndex: true,
    nodeReducer(node, data) {
      if (data.marker) return data; // decade rings never dim
      if (state.selected === null) return data;
      // A selection with no neighbors in the active lens (no ratings data)
      // shouldn't black out the map — just spotlight the selected node.
      if (state.highlighted.size <= 1) {
        return node === state.selected ? { ...data, zIndex: 2, highlighted: true } : data;
      }
      if (state.highlighted.has(node)) {
        return { ...data, zIndex: 2, highlighted: node === state.selected };
      }
      return { ...data, color: DIM_COLOR, label: null, zIndex: 0 };
    },
  });

  renderer.on("clickNode", ({ node }) => {
    if (graph.getNodeAttribute(node, "marker")) return;
    selectMovie(Number(node));
  });
  renderer.on("clickStage", clearSelection);
  const mapEl = document.getElementById("map");
  renderer.on("enterNode", () => (mapEl.style.cursor = "pointer"));
  renderer.on("leaveNode", () => (mapEl.style.cursor = ""));
  document.getElementById("detail-close").addEventListener("click", clearSelection);

  setupSearch();
  setupLens();
  setupView();
  setupFilters();
  renderLegend();
  document.body.classList.add("ready");
  setTimeout(() => document.getElementById("loading")?.remove(), 800);
}

main().catch((err) => {
  document.querySelector("#loading .load-status").textContent =
    "no map data — run the pipeline first (see README)";
  console.error(err);
});
