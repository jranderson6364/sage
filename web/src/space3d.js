// 3D semantic-axes view: levity x threat x intimacy, dot size = rating.
// The app's only view — a three.js point cloud. Selection state is shared
// (main.js owns it, this module renders it).

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const EDGE_COLOR = "#52514e";
const AXIS_COLOR = "#2c2c2a";

// Mood-blend palette: each positive pole owns a hue, colors mix by how far a
// movie sits along each axis, and neutral films settle into slate.
export const MOOD = {
  levity: { r: 1.0, g: 0.66, b: 0.2, hex: "#ffa833" },
  threat: { r: 0.96, g: 0.3, b: 0.3, hex: "#f54d4d" },
  intimacy: { r: 0.98, g: 0.45, b: 0.78, hex: "#fa73c6" },
  neutral: { r: 0.42, g: 0.47, b: 0.58, hex: "#6b7894" },
};

// Positions sit outside the point cloud's reach. The display transform pushes
// the outer fifth of films past the ±1 ring, the most extreme landing near
// ±1.7 — labels have to clear that or outliers render on top of them.
const AXIS_LABELS = [
  { text: "Playful", pos: [1.85, 0, 0], color: MOOD.levity.hex },
  { text: "Somber", pos: [-1.85, 0, 0] },
  { text: "Tense", pos: [0, 1.85, 0], color: MOOD.threat.hex },
  { text: "Safe", pos: [0, -1.85, 0] },
  { text: "Intimate", pos: [0, 0, 1.85], color: MOOD.intimacy.hex },
  { text: "Detached", pos: [0, 0, -1.85] },
];

function moodColor(m) {
  const f = Math.max(0, m.levity);
  const s = Math.max(0, m.threat);
  const r = Math.max(0, m.intimacy);
  const w = f + s + r;
  if (w < 1e-6) return [MOOD.neutral.r, MOOD.neutral.g, MOOD.neutral.b];
  const mix = [
    (f * MOOD.levity.r + s * MOOD.threat.r + r * MOOD.intimacy.r) / w,
    (f * MOOD.levity.g + s * MOOD.threat.g + r * MOOD.intimacy.g) / w,
    (f * MOOD.levity.b + s * MOOD.threat.b + r * MOOD.intimacy.b) / w,
  ];
  // Mild movies stay slate-ish; only strong moods reach the vivid pole hues.
  const k = Math.min(1, w / 1.2);
  return [
    MOOD.neutral.r + (mix[0] - MOOD.neutral.r) * k,
    MOOD.neutral.g + (mix[1] - MOOD.neutral.g) * k,
    MOOD.neutral.b + (mix[2] - MOOD.neutral.b) * k,
  ];
}

const VERT = /* glsl */ `
  attribute float size;
  attribute vec3 pointColor;
  varying vec3 vColor;
  void main() {
    vColor = pointColor;
    if (size <= 0.0) {
      // Filtered out: GPUs clamp gl_PointSize 0 to a 1px dot, so move the
      // vertex outside clip space instead of relying on a zero size.
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      gl_PointSize = 0.0;
    } else {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = size * (520.0 / -mv.z);
      gl_Position = projectionMatrix * mv;
    }
  }
`;

const FRAG = /* glsl */ `
  varying vec3 vColor;
  void main() {
    vec2 c = gl_PointCoord - 0.5;
    float d = length(c);
    if (d > 0.5) discard;
    float a = smoothstep(0.5, 0.4, d);
    gl_FragColor = vec4(vColor, a * 0.92);
  }
`;

export class Space3D {
  /**
   * @param container  element to render into (may be display:none now)
   * @param state      shared app state ({movies, selected, highlighted, lens})
   * @param hooks      { onPick(idx), onClear(), neighborsOf(idx) }
   */
  constructor(container, state, hooks) {
    this.container = container;
    this.state = state;
    this.hooks = hooks;
    this.ready = false;
    this.running = false;
    this.fly = null;
  }

  init() {
    const { movies } = this.state;
    const n = movies.length;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#09090b");

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 50);
    this.camera.position.set(3.1, 2.4, 3.1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 0.4;
    this.controls.maxDistance = 8;

    // Point cloud: position = axes, size = rating, color = genre.
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const size = new Float32Array(n);
    this.baseColors = new Float32Array(n * 3);
    movies.forEach((m, i) => {
      pos.set([m.levity, m.threat, m.intimacy], i * 3);
      col.set(moodColor(m), i * 3);
      // vote_average lives in ~[4, 9]; map to world-space point size
      const r = Math.min(Math.max(m.rating, 4), 9);
      size[i] = 0.012 + ((r - 4) / 5) * 0.03;
    });
    this.baseColors.set(col);
    this.baseSizes = size.slice();

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("pointColor", new THREE.BufferAttribute(col, 3));
    geo.setAttribute("size", new THREE.BufferAttribute(size, 1));
    this.points = new THREE.Points(
      geo,
      new THREE.ShaderMaterial({
        vertexShader: VERT,
        fragmentShader: FRAG,
        transparent: true,
        depthWrite: false,
      })
    );
    this.scene.add(this.points);

    // Axis guide lines through the origin.
    const axisGeo = new THREE.BufferGeometry().setFromPoints(
      [
        [-1.7, 0, 0], [1.7, 0, 0],
        [0, -1.7, 0], [0, 1.7, 0],
        [0, 0, -1.7], [0, 0, 1.7],
      ].map((p) => new THREE.Vector3(...p))
    );
    this.scene.add(
      new THREE.LineSegments(axisGeo, new THREE.LineBasicMaterial({ color: AXIS_COLOR }))
    );

    // Selection edges (selected -> neighbors), rebuilt in syncSelection.
    this.edgeGeo = new THREE.BufferGeometry();
    this.edges = new THREE.LineSegments(
      this.edgeGeo,
      new THREE.LineBasicMaterial({ color: EDGE_COLOR, transparent: true, opacity: 0.85 })
    );
    this.edges.visible = false;
    this.scene.add(this.edges);

    // Axis end labels + hover tooltip as positioned overlay divs.
    this.labelEls = AXIS_LABELS.map(({ text, pos, color }) => {
      const el = document.createElement("span");
      el.className = "axis-label";
      el.textContent = text;
      if (color) el.style.color = color;
      this.container.appendChild(el);
      return { el, v: new THREE.Vector3(...pos) };
    });
    this.tooltip = document.createElement("div");
    this.tooltip.id = "tooltip3d";
    this.tooltip.hidden = true;
    this.container.appendChild(this.tooltip);

    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points.threshold = 0.02;
    this.pointer = new THREE.Vector2();
    this.hovered = null;
    // On touch, pointermove only fires while dragging (there's no hover) —
    // showing a hover tooltip there would just flicker mid-orbit. Tap still
    // picks via pointerdown/up regardless; touch users see detail on tap.
    this.isCoarsePointer = window.matchMedia("(pointer: coarse)").matches;

    const canvas = this.renderer.domElement;
    canvas.addEventListener("pointerdown", (e) => {
      this.downAt = [e.clientX, e.clientY];
    });
    canvas.addEventListener("pointerup", (e) => {
      if (!this.downAt) return;
      const dx = e.clientX - this.downAt[0];
      const dy = e.clientY - this.downAt[1];
      this.downAt = null;
      if (dx * dx + dy * dy > 36) return; // drag, not click
      const idx = this.pick(e);
      if (idx !== null) this.hooks.onPick(idx);
      else this.hooks.onClear();
    });
    canvas.addEventListener("pointermove", (e) => {
      this.pointerEvent = e;
    });

    window.addEventListener("resize", () => this.resize());
    this.ready = true;
  }

  pick(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const size = this.points.geometry.getAttribute("size");
    for (const hit of this.raycaster.intersectObject(this.points)) {
      if (size.array[hit.index] > 0) return hit.index; // skip filtered-out
    }
    return null;
  }

  inFilter(m, i) {
    // List membership gates first, then the axis/rating sliders narrow
    // further — the two compose, so you can filter by mood inside a list.
    const members = this.state.listMembers;
    if (members && !members.has(i)) return false;
    const f = this.state.filters;
    return (
      m.levity >= f.levity[0] && m.levity <= f.levity[1] &&
      m.threat >= f.threat[0] && m.threat <= f.threat[1] &&
      m.intimacy >= f.intimacy[0] && m.intimacy <= f.intimacy[1] &&
      m.rating >= f.rating[0] && m.rating <= f.rating[1]
    );
  }

  // One pass computing every point's color + size from base style, the axis/
  // rating filters, and the selection spotlight. With a selection active,
  // everything except the selected movie and its neighbors is removed
  // outright — hidden points are unclickable and unhoverable, not dimmed.
  restyle() {
    if (!this.ready) return;
    const { selected, highlighted, movies } = this.state;
    const col = this.points.geometry.getAttribute("pointColor");
    const size = this.points.geometry.getAttribute("size");
    const selActive = selected !== null && highlighted.size > 1;

    let visible = 0;
    for (let i = 0; i < movies.length; i++) {
      const hi = highlighted.has(String(i));
      const shown = selActive ? hi : this.inFilter(movies[i], i);
      if (!shown) {
        size.array[i] = 0;
        continue;
      }
      visible++;
      col.array.set(this.baseColors.subarray(i * 3, i * 3 + 3), i * 3);
      size.array[i] = selActive
        ? this.baseSizes[i] * (String(i) === selected ? 1.7 : 1.25)
        : this.baseSizes[i];
    }
    this.visibleCount = visible;

    // Idle orbit around the selection — a slow cinematic drift.
    this.controls.autoRotate = selActive;
    this.controls.autoRotateSpeed = 0.7;

    if (selActive) {
      const sel = movies[Number(selected)];
      const verts = [];
      for (const nb of this.hooks.neighborsOf(Number(selected))) {
        const m = movies[nb];
        verts.push(sel.levity, sel.threat, sel.intimacy, m.levity, m.threat, m.intimacy);
      }
      this.edgeGeo.setAttribute("position",
        new THREE.BufferAttribute(new Float32Array(verts), 3));
      this.edges.visible = verts.length > 0;
    } else {
      this.edges.visible = false;
    }
    col.needsUpdate = true;
    size.needsUpdate = true;
  }

  syncSelection() {
    this.restyle();
  }

  flyTo(idx) {
    if (!this.ready) return;
    const m = this.state.movies[idx];
    const target = new THREE.Vector3(m.levity, m.threat, m.intimacy);
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this.fly = {
      start: performance.now(),
      duration: 700,
      fromT: this.controls.target.clone(),
      toT: target,
      fromP: this.camera.position.clone(),
      toP: target.clone().add(dir.multiplyScalar(1.1)),
    };
  }

  show() {
    if (!this.ready) this.init();
    this.container.style.display = "block";
    this.resize();
    this.running = true;
    this.loop();
  }

  hide() {
    this.running = false;
    this.container.style.display = "none";
  }

  resize() {
    if (!this.ready) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  loop() {
    if (!this.running) return;
    requestAnimationFrame(() => this.loop());

    if (this.fly) {
      const t = Math.min(1, (performance.now() - this.fly.start) / this.fly.duration);
      const k = 1 - Math.pow(1 - t, 3);
      this.controls.target.lerpVectors(this.fly.fromT, this.fly.toT, k);
      this.camera.position.lerpVectors(this.fly.fromP, this.fly.toP, k);
      if (t >= 1) this.fly = null;
    }
    this.controls.update();

    // Hover: raycast at most once per frame. Skipped on touch (see above).
    if (this.pointerEvent && !this.isCoarsePointer) {
      const idx = this.pick(this.pointerEvent);
      if (idx !== this.hovered) {
        this.hovered = idx;
        this.renderer.domElement.style.cursor = idx !== null ? "pointer" : "grab";
        this.tooltip.hidden = idx === null;
        if (idx !== null) {
          const m = this.state.movies[idx];
          // Percentile rank, not the raw coordinate — the display transform
          // pushes extreme films past ±1, so mapping the coordinate onto 0-100
          // reads over 100 (and below 0) at the tails. main.js owns the ranks.
          const pct = this.state.axisPct;
          this.tooltip.innerHTML =
            `<strong>${m.title.replaceAll("<", "&lt;")}</strong>` +
            `${m.year ? ` (${m.year})` : ""} · ★ ${m.rating}` +
            `<span class="t3-scores">levity ${pct.levity[idx]} · threat ${pct.threat[idx]}` +
            ` · intimacy ${pct.intimacy[idx]}</span>`;
        }
      }
      if (idx !== null) {
        this.tooltip.style.left = `${this.pointerEvent.clientX + 14}px`;
        this.tooltip.style.top = `${this.pointerEvent.clientY + 10}px`;
      }
      this.pointerEvent = null;
    }

    this.renderer.render(this.scene, this.camera);

    // Project axis labels to screen space.
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    for (const { el, v } of this.labelEls) {
      const p = v.clone().project(this.camera);
      const visible = p.z < 1;
      el.style.display = visible ? "block" : "none";
      if (visible) {
        el.style.left = `${(p.x * 0.5 + 0.5) * w}px`;
        el.style.top = `${(-p.y * 0.5 + 0.5) * h}px`;
      }
    }
  }
}
