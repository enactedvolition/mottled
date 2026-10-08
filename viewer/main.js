/* Mottled web viewer — raw WebGL2, no dependencies. World is z-up to match
 * the .mtj scene convention (terrain z = height, trajectory xyz draped). */
"use strict";

// ---------------------------------------------------------------- constants
// Incision data palette — accent blue, live teal, risk amber, loss red,
// payout green, then lightened variants. Mirrors ui.py's _MARBLE_COLORS.
const PALETTE = ["#4B7CF3", "#00CCA8", "#D4934A", "#E05050", "#38B07A",
                 "#8FA7F7", "#5CE0C6", "#E6B884", "#F08A8A", "#7FD0AC"];
// draw/skip fine-segment run lengths, cycled per run: solid, dash, dot, longdash, dashdot
const DASH_CYCLE = [null, [5, 3], [1, 2], [9, 3], [5, 2, 1, 2]];
const SEG = 8;              // Catmull-Rom subdivisions per layer span
const OVERLAY_ALPHA = 0.55; // runs after the first
const GEN_ALPHA = 0.55;     // extra fade on decoded-token ("+") trajectory lines
// Pick tolerance in px, converted to a world-space ray radius. Much tighter
// than the 14px the old marker-picking used: a trajectory is a continuous
// line, so it needs no slack to be grabbable, and because the ray returns the
// *front-most* segment within the radius, the radius is also the worst case
// by which a pick can land on a neighbouring line instead of the one under
// the cursor. Measured over the sample scenes, 6px keeps that under ~8px in
// the densest bundle and near zero in a single-run scene.
const PICK_RADIUS = 6;
const FOVY = 0.9;           // vertical field of view (rad), shared by the
                            // projection and the pixels->world pick radius
// Terrain potential ramp: void -> surfaces -> precision blue -> light blue.
// Mirrors ui.py's _TERRAIN_COLORSCALE.
const TERRAIN_RAMP = [[0.016, 0.024, 0.055], [0.031, 0.047, 0.102], [0.047, 0.063, 0.125],
                      [0.078, 0.118, 0.271], [0.110, 0.165, 0.333], [0.184, 0.333, 0.722],
                      [0.294, 0.486, 0.953], [0.475, 0.616, 0.965], [0.784, 0.831, 0.984]];

function terrainColor(t) {
  t = Math.min(1, Math.max(0, t)) * (TERRAIN_RAMP.length - 1);
  const i = Math.min(TERRAIN_RAMP.length - 2, Math.floor(t)), f = t - i;
  const a = TERRAIN_RAMP[i], b = TERRAIN_RAMP[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function hexRGB(h) {
  return [parseInt(h.slice(1, 3), 16) / 255, parseInt(h.slice(3, 5), 16) / 255,
          parseInt(h.slice(5, 7), 16) / 255];
}

// ---------------------------------------------------------------- mat4 (column-major)
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0,
                           0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}
function lookAt(eye, c, up) {
  let zx = eye[0] - c[0], zy = eye[1] - c[1], zz = eye[2] - c[2];
  let l = Math.hypot(zx, zy, zz); zx /= l; zy /= l; zz /= l;
  let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
  l = Math.hypot(xx, xy, xz) || 1; xx /= l; xy /= l; xz /= l;
  const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
  return new Float32Array([xx, yx, zx, 0, xy, yy, zy, 0, xz, yz, zz, 0,
                           -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
                           -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
                           -(zx * eye[0] + zy * eye[1] + zz * eye[2]), 1]);
}
function matMul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++)
    o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}
function projectPoint(m, p, w, h) { // -> [px, py, clipW] in CSS pixels
  const cw = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15];
  const x = (m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / cw;
  const y = (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / cw;
  return [(x * 0.5 + 0.5) * w, (0.5 - y * 0.5) * h, cw];
}
// Inverse of a column-major 4x4, in float64 — the MVP is a Float32Array and
// its near/far planes differ by 2000x, so the solve is done at double width.
function invertMat4(m) {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3],
        a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7],
        a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11],
        a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10,
        b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11,
        b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12,
        b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30,
        b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31,
        b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return null;
  det = 1 / det;
  return new Float64Array([
    (a11 * b11 - a12 * b10 + a13 * b09) * det, (a02 * b10 - a01 * b11 - a03 * b09) * det,
    (a31 * b05 - a32 * b04 + a33 * b03) * det, (a22 * b04 - a21 * b05 - a23 * b03) * det,
    (a12 * b08 - a10 * b11 - a13 * b07) * det, (a00 * b11 - a02 * b08 + a03 * b07) * det,
    (a32 * b02 - a30 * b05 - a33 * b01) * det, (a20 * b05 - a22 * b02 + a23 * b01) * det,
    (a10 * b10 - a11 * b08 + a13 * b06) * det, (a01 * b08 - a00 * b10 - a03 * b06) * det,
    (a30 * b04 - a31 * b02 + a33 * b00) * det, (a21 * b02 - a20 * b04 - a23 * b00) * det,
    (a11 * b07 - a10 * b09 - a12 * b06) * det, (a00 * b09 - a01 * b07 + a02 * b06) * det,
    (a31 * b01 - a30 * b03 - a32 * b00) * det, (a20 * b03 - a21 * b01 + a22 * b00) * det]);
}
function unprojectPoint(inv, px, py, ndcZ, w, h) { // exact inverse of projectPoint
  const x = (px / w) * 2 - 1, y = 1 - (py / h) * 2;
  const cw = inv[3] * x + inv[7] * y + inv[11] * ndcZ + inv[15];
  return [(inv[0] * x + inv[4] * y + inv[8] * ndcZ + inv[12]) / cw,
          (inv[1] * x + inv[5] * y + inv[9] * ndcZ + inv[13]) / cw,
          (inv[2] * x + inv[6] * y + inv[10] * ndcZ + inv[14]) / cw];
}
// World-space ray through a CSS-pixel cursor position: from the near plane to
// the far plane of the current view. Origin is the near-plane point, so ray
// parameters compare consistently across every run in the scene.
function cameraRay(mvp, px, py, w, h) {
  const inv = invertMat4(mvp);
  if (!inv) return null;
  const near = unprojectPoint(inv, px, py, -1, w, h);
  const far = unprojectPoint(inv, px, py, 1, w, h);
  const dir = [far[0] - near[0], far[1] - near[1], far[2] - near[2]];
  const len = Math.hypot(dir[0], dir[1], dir[2]);
  if (!Number.isFinite(len) || len < 1e-12 || !Number.isFinite(near[0] + near[1] + near[2]))
    return null;
  return { origin: near, dir };
}

// ---------------------------------------------------------------- GL helpers
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl2", { antialias: true });

function makeProgram(vsSrc, fsSrc) {
  const mk = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  };
  const p = gl.createProgram();
  gl.attachShader(p, mk(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
function makeVAO(program, buffers) { // buffers: [{name, size, data|length}]
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const out = { vao, bufs: {} };
  for (const b of buffers) {
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, b.data || b.length * 4, b.data ? gl.STATIC_DRAW : gl.DYNAMIC_DRAW);
    const loc = gl.getAttribLocation(program, b.name);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, b.size, gl.FLOAT, false, 0, 0);
    out.bufs[b.name] = buf;
  }
  gl.bindVertexArray(null);
  return out;
}

const meshProg = gl ? makeProgram(
  `#version 300 es
   in vec3 pos; in vec3 nrm; in vec3 col;
   uniform mat4 mvp; out vec3 vc; out vec3 vn;
   void main() { gl_Position = mvp * vec4(pos, 1.0); vc = col; vn = nrm; }`,
  `#version 300 es
   precision highp float; in vec3 vc; in vec3 vn; out vec4 o;
   void main() {
     float d = 0.55 + 0.45 * max(dot(normalize(vn), normalize(vec3(0.35, 0.25, 1.0))), 0.0);
     o = vec4(vc * d, 1.0);
   }`) : null;

const lineProg = gl ? makeProgram(
  `#version 300 es
   in vec3 pos; in vec4 col;
   uniform mat4 mvp; out vec4 vc;
   void main() { gl_Position = mvp * vec4(pos, 1.0); vc = col; }`,
  `#version 300 es
   precision highp float; in vec4 vc; out vec4 o;
   void main() { o = vc; }`) : null;

const pointProg = gl ? makeProgram(
  `#version 300 es
   in vec3 pos; in vec4 col; in float size;
   uniform mat4 mvp; out vec4 vc;
   void main() { gl_Position = mvp * vec4(pos, 1.0); gl_PointSize = size; vc = col; }`,
  `#version 300 es
   precision highp float; in vec4 vc; uniform float rim; out vec4 o;
   void main() {
     float r = length(gl_PointCoord - 0.5);
     if (r > 0.5) discard;
     if (rim > 1.5 && r < 0.34) discard;   // rim 2: a ring alone, in vc
     vec3 c = (rim > 0.5 && rim < 1.5 && r > 0.40) ? vec3(1.0) : vc.rgb;
     o = vec4(c, vc.a * smoothstep(0.5, 0.45, r));
   }`) : null;

// ---------------------------------------------------------------- app state
const state = {
  scene: null,        // MTJ.loadScene result
  runs: [],           // per-run render data
  terrain: null,
  attn: null,         // attention line buffer
  bounds: null,
  layerF: 0, L: 1,
  playing: false, speed: 2,
  showAttention: false,
  showUncertainty: false,
  visible: [],
  // `hover` is what the cursor is over right now; `pinned` is the pick a click
  // froze (it survives the cursor moving away, until Escape or a click on
  // empty space). `pick` is whichever of the two the panel currently shows.
  pick: null, hover: null, pinned: null, pickPinned: false,
  // blast scenes only: which layout is drawn, the label pellets are coloured
  // by, and the label the reader picked (null until they pick one)
  blast: null,
  cam: { theta: -2.2, phi: 0.65, dist: 10, target: [0, 0, 0] },
  mouse: null,
};

function catmullRom(pts, L) { // pts: Float32Array (L*3) -> densified Float32Array
  const n = (L - 1) * SEG + 1, out = new Float32Array(n * 3);
  const P = (i) => { i = Math.min(L - 1, Math.max(0, i)); return [pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]]; };
  for (let s = 0; s < L - 1; s++) {
    const p0 = P(s - 1), p1 = P(s), p2 = P(s + 1), p3 = P(s + 2);
    for (let k = 0; k < SEG; k++) {
      const t = k / SEG, t2 = t * t, t3 = t2 * t;
      for (let d = 0; d < 3; d++)
        out[(s * SEG + k) * 3 + d] = 0.5 * ((2 * p1[d]) + (-p0[d] + p2[d]) * t +
          (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2 + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3);
    }
  }
  out.set(P(L - 1), (n - 1) * 3);
  return out;
}

// Uncertainty ramp for the density standard-error overlay: confident terrain
// stays its own colour (blended out at se≈0), rising error washes toward a
// warning amber. Kept distinct from the potential ramp so the two readings
// never get confused.
const SE_COLOR = [0.878, 0.576, 0.290]; // #E0934A, the risk-amber data colour

function buildTerrain(t) {
  const W = t.x.shape[0], H = t.y.shape[0], xs = t.x.data, ys = t.y.data, zs = t.z.data;
  const se = t.se ? t.se.data : null;
  let seMax = 0;
  if (se) for (const v of se) { if (v > seMax) seMax = v; }
  seMax = seMax || 1;
  let zmin = Infinity, zmax = -Infinity;
  for (const v of zs) { if (v < zmin) zmin = v; if (v > zmax) zmax = v; }
  const zr = (zmax - zmin) || 1;
  const pos = new Float32Array(W * H * 3), col = new Float32Array(W * H * 3),
        seCol = new Float32Array(W * H * 3), nrm = new Float32Array(W * H * 3);
  for (let i = 0; i < H; i++) for (let j = 0; j < W; j++) {
    const v = i * W + j, z = zs[v];
    pos.set([xs[j], ys[i], z], v * 3);
    const shade = terrainColor((z - zmin) / zr);
    // faint contour-band feel: darken near iso-lines of height
    const iso = Math.abs(((z - zmin) / zr * 14) % 1 - 0.5);
    const k = iso < 0.06 ? 0.82 : 1.0;
    col.set([shade[0] * k, shade[1] * k, shade[2] * k], v * 3);
    // uncertainty shading: blend the base colour toward SE_COLOR by the
    // cell's relative bootstrap standard error (confident cells unchanged)
    const u = se ? Math.min(1, se[v] / seMax) : 0;
    const um = 0.15 + 0.85 * u; // keep a floor so confident terrain reads too
    seCol.set([shade[0] * k * (1 - um) + SE_COLOR[0] * um,
               shade[1] * k * (1 - um) + SE_COLOR[1] * um,
               shade[2] * k * (1 - um) + SE_COLOR[2] * um], v * 3);
    const dzx = (zs[i * W + Math.min(j + 1, W - 1)] - zs[i * W + Math.max(j - 1, 0)]) /
                ((xs[Math.min(j + 1, W - 1)] - xs[Math.max(j - 1, 0)]) || 1);
    const dzy = (zs[Math.min(i + 1, H - 1) * W + j] - zs[Math.max(i - 1, 0) * W + j]) /
                ((ys[Math.min(i + 1, H - 1)] - ys[Math.max(i - 1, 0)]) || 1);
    const nl = Math.hypot(dzx, dzy, 1);
    nrm.set([-dzx / nl, -dzy / nl, 1 / nl], v * 3);
  }
  const idx = new Uint32Array((W - 1) * (H - 1) * 6);
  let q = 0;
  for (let i = 0; i < H - 1; i++) for (let j = 0; j < W - 1; j++) {
    const a = i * W + j;
    idx.set([a, a + 1, a + W, a + 1, a + W + 1, a + W], q); q += 6;
  }
  const vaoInfo = makeVAO(meshProg, [{ name: "pos", size: 3, data: pos },
                                     { name: "nrm", size: 3, data: nrm },
                                     { name: "col", size: 3, data: col }]);
  gl.bindVertexArray(vaoInfo.vao);
  const ib = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
  gl.bindVertexArray(null);

  // sparse wireframe, lifted a hair above the surface
  const lift = zr * 0.004;
  const wire = [];
  for (let i = 0; i < H; i += 8) for (let j = 0; j < W - 1; j++)
    wire.push(xs[j], ys[i], zs[i * W + j] + lift, xs[j + 1], ys[i], zs[i * W + j + 1] + lift);
  for (let j = 0; j < W; j += 8) for (let i = 0; i < H - 1; i++)
    wire.push(xs[j], ys[i], zs[i * W + j] + lift, xs[j], ys[i + 1], zs[(i + 1) * W + j] + lift);
  const wirePos = new Float32Array(wire);
  const wireCol = new Float32Array(wirePos.length / 3 * 4);
  for (let i = 0; i < wireCol.length; i += 4) wireCol.set([1, 1, 1, 0.06], i);
  const wireVao = makeVAO(lineProg, [{ name: "pos", size: 3, data: wirePos },
                                     { name: "col", size: 4, data: wireCol }]);
  return { vao: vaoInfo.vao, count: idx.length, wireVao: wireVao.vao, wireCount: wirePos.length / 3,
           zmin, zmax, hasSE: !!se, seMax, colBuf: vaoInfo.bufs.col, baseCol: col, seCol };
}

// Swap the terrain colour attribute between the potential shading and the
// uncertainty (density standard-error) shading in place — no re-tessellation.
function setTerrainColors(showUncertainty) {
  const ter = state.terrain;
  if (!ter || !ter.colBuf) return;
  gl.bindBuffer(gl.ARRAY_BUFFER, ter.colBuf);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, showUncertainty && ter.hasSE ? ter.seCol : ter.baseCol);
}

// `look` (blast scenes only) overrides the palette: colour(j) is pellet j's
// label colour, ring(j) whether its marble is ringed, and trail scales the
// alpha of lines and layer dots so the marbles at the scrubbed layer read
// over 25 layers of trails drawn in one plane.
function buildRun(run, runIdx, colorBase, dashUnit, look) {
  const [N, L] = run.points.shape, pts = run.points.data;
  const alpha = runIdx === 0 ? 1.0 : OVERLAY_ALPHA;
  const trail = look ? look.trail : 1;
  // every GL object made here, so a rebuilt run can free the one it replaces
  const owned = [];
  const own = (v) => { owned.push(v.vao, ...Object.values(v.bufs)); return v; };
  const dash = DASH_CYCLE[runIdx % DASH_CYCLE.length];
  const patLen = dash ? dash.reduce((a, b) => a + b, 0) * dashUnit : 0;
  // decode boundary: trajectories j >= genFrom trace generated ("+") tokens
  // (-1 when the run has no decode record, or trajectories aren't 1:1 tokens)
  const genFrom = MTJ.generationBoundary(run);
  // spatial index for picking: one BVH over this run's layer-to-layer chords
  // (segment j*(L-1)+l joins layer l to l+1 of trajectory j, meta [j, l]).
  // Built once per scene load; the camera ray is cast against it every frame.
  const bvh = BVH.fromPoints(pts, N, L);
  const trajs = [], lineVerts = [], lineCols = [];
  for (let j = 0; j < N; j++) {
    const rgb = hexRGB(look ? look.colour(j) : PALETTE[(colorBase + j) % PALETTE.length]);
    const generated = genFrom >= 0 && j >= genFrom;
    const lineAlpha = (generated ? alpha * GEN_ALPHA : alpha) * trail;
    const fine = catmullRom(pts.subarray(j * L * 3, (j + 1) * L * 3), L);
    const nFine = fine.length / 3 - 1;
    const emit = (x0, y0, z0, x1, y1, z1) => {
      lineVerts.push(x0, y0, z0, x1, y1, z1);
      lineCols.push(rgb[0], rgb[1], rgb[2], lineAlpha, rgb[0], rgb[1], rgb[2], lineAlpha);
    };
    // Dashes are cut by accumulated arc length (world units), so the
    // pattern stays even whether a layer span covers millimetres or
    // hundreds of units (real-model scenes).
    let dpos = 0;
    for (let s = 0; s < nFine; s++) {
      const x0 = fine[s * 3], y0 = fine[s * 3 + 1], z0 = fine[s * 3 + 2];
      const x1 = fine[s * 3 + 3], y1 = fine[s * 3 + 4], z1 = fine[s * 3 + 5];
      if (!dash) { emit(x0, y0, z0, x1, y1, z1); continue; }
      const segLen = Math.hypot(x1 - x0, y1 - y0, z1 - z0);
      let done = 0;
      while (done < segLen - 1e-9) {
        let p = dpos % patLen, di = 0;
        while (di < dash.length && p >= dash[di] * dashUnit) { p -= dash[di] * dashUnit; di++; }
        if (di >= dash.length) { di = 0; p = 0; }  // float residue at the cycle seam
        // the minimum step keeps float rounding from ever stalling the walk
        const chunk = Math.min(Math.max(dash[di] * dashUnit - p, dashUnit * 1e-3),
                               segLen - done);
        if (di % 2 === 0) {
          const t0 = done / segLen, t1 = (done + chunk) / segLen;
          emit(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0, z0 + (z1 - z0) * t0,
               x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1, z0 + (z1 - z0) * t1);
        }
        done += chunk; dpos += chunk;
      }
    }
    trajs.push({ fine, rgb, generated, ring: !!(look && look.ring(j)),
                 label: (generated ? "+" : "") +
                        (run.trajectoryLabels[j] != null ? String(run.trajectoryLabels[j]) : `#${j}`) });
  }
  const linePos = new Float32Array(lineVerts), lineCol = new Float32Array(lineCols);
  const lineVao = own(makeVAO(lineProg, [{ name: "pos", size: 3, data: linePos },
                                         { name: "col", size: 4, data: lineCol }]));
  // small dot at every stored layer point (the "markers" of the reference);
  // decoded-token trajectories get a rimmed (open) dot — the viewer's version
  // of the explorer's open-diamond markers for the decode axis
  const dotSets = { base: { pos: [], col: [], size: [] }, gen: { pos: [], col: [], size: [] } };
  for (let j = 0; j < N; j++) for (let l = 0; l < L; l++) {
    const d = trajs[j].generated ? dotSets.gen : dotSets.base;
    const o = (j * L + l) * 3;
    d.pos.push(pts[o], pts[o + 1], pts[o + 2]);
    d.col.push(trajs[j].rgb[0], trajs[j].rgb[1], trajs[j].rgb[2], alpha * trail);
    d.size.push(trajs[j].generated ? 6 : 5);
  }
  const mkDots = (d) => d.pos.length ? own(makeVAO(pointProg, [
    { name: "pos", size: 3, data: new Float32Array(d.pos) },
    { name: "col", size: 4, data: new Float32Array(d.col) },
    { name: "size", size: 1, data: new Float32Array(d.size) }])).vao : null;
  return { run, N, L, trajs, alpha, bvh, owned,
           lineVao: lineVao.vao, lineCount: linePos.length / 3,
           dotVao: mkDots(dotSets.base), dotCount: dotSets.base.size.length,
           genDotVao: mkDots(dotSets.gen), genDotCount: dotSets.gen.size.length };
}

// dynamic point buffer (marbles + hover highlight)
let dynPoints = null;
function ensureDynPoints(maxPts) {
  dynPoints = { max: maxPts, ...makeVAO(pointProg, [
    { name: "pos", size: 3, length: maxPts * 3 },
    { name: "col", size: 4, length: maxPts * 4 },
    { name: "size", size: 1, length: maxPts }]) };
}
// dynamic attention line buffer
let attnBuf = null;
function ensureAttnBuf(maxVerts) {
  attnBuf = { max: maxVerts, count: 0, ...makeVAO(lineProg, [
    { name: "pos", size: 3, length: maxVerts * 3 },
    { name: "col", size: 4, length: maxVerts * 4 }]) };
}

function setScene(scene) {
  state.scene = scene;
  state.L = scene.runs[0].points.shape[1];
  state.layerF = 0;
  state.playing = false;
  state.pick = state.hover = state.pinned = null;
  state.pickPinned = false;
  ui.infoPanel.hidden = true;
  ui.infoPanel.classList.remove("pinned");
  state.visible = scene.runs.map(() => true);
  // A blast opens at its deepest layer: at layer 0 the pellets can all be one
  // point, and the spread with depth is what the scene is for.
  state.blast = scene.blast ? { layout: 0, colour: null, chosen: null, shownLayer: -1 } : null;
  if (state.blast) { state.layerF = state.L - 1; applyBlastLayout(); }

  // Real-model scenes span hundreds of units in x/y while terrain height is
  // normalized 0..1 — relief flattens into invisibility and marbles sink
  // into the surface. Exaggerate z uniformly (a pure view transform applied
  // identically to terrain and trajectories, so all spatial relationships
  // are preserved) until relief reads at ~10% of the horizontal span.
  const tzArr = scene.terrain.z.data, txArr = scene.terrain.x.data, tyArr = scene.terrain.y.data;
  let tzmin = Infinity, tzmax = -Infinity;
  for (const v of tzArr) { if (v < tzmin) tzmin = v; if (v > tzmax) tzmax = v; }
  const spanXY = Math.hypot(txArr[txArr.length - 1] - txArr[0],
                            tyArr[tyArr.length - 1] - tyArr[0]) || 1;
  const relief = tzmax - tzmin;
  const zScale = relief > 1e-9 ? Math.max(1, 0.10 * spanXY / relief) : 1;
  if (zScale > 1) {
    for (let i = 0; i < tzArr.length; i++) tzArr[i] *= zScale;
    for (const run of scene.runs) {
      const p = run.points.data;
      for (let i = 2; i < p.length; i += 3) p[i] *= zScale;
    }
  }

  state.terrain = buildTerrain(scene.terrain);
  setTerrainColors(state.showUncertainty);
  state.runs = [];
  const dashUnit = spanXY / 500;  // world-unit length of one dash-pattern tick
  let colorBase = 0, totalTrajs = 0, maxAttn = 0;
  scene.runs.forEach((run, i) => {
    state.runs.push(buildRun(run, i, colorBase, dashUnit, i === 0 ? blastLook() : null));
    colorBase += run.points.shape[0];
    totalTrajs += run.points.shape[0];
    if (run.attention) maxAttn += run.attention.shape[1] * 3 * 2;
  });
  ensureDynPoints(2 * totalTrajs + 1);  // marbles, the pick, a ring per marble at most
  ensureAttnBuf(Math.max(maxAttn, 2));

  // bounds: terrain footprint + trajectory extents
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const acc = (x, y, z) => {
    lo[0] = Math.min(lo[0], x); hi[0] = Math.max(hi[0], x);
    lo[1] = Math.min(lo[1], y); hi[1] = Math.max(hi[1], y);
    lo[2] = Math.min(lo[2], z); hi[2] = Math.max(hi[2], z);
  };
  const t = scene.terrain;
  acc(t.x.data[0], t.y.data[0], state.terrain.zmin);
  acc(t.x.data[t.x.data.length - 1], t.y.data[t.y.data.length - 1], state.terrain.zmax);
  // frame the trajectories (the terrain usually extends far beyond the action)
  const tlo = [Infinity, Infinity, Infinity], thi = [-Infinity, -Infinity, -Infinity];
  for (const r of scene.runs) {
    const p = r.points.data;
    for (let i = 0; i < p.length; i += 3) {
      acc(p[i], p[i + 1], p[i + 2]);
      for (let d = 0; d < 3; d++) {
        tlo[d] = Math.min(tlo[d], p[i + d]); thi[d] = Math.max(thi[d], p[i + d]);
      }
    }
  }
  state.bounds = { lo, hi };
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 1;
  const tdiag = Math.hypot(thi[0] - tlo[0], thi[1] - tlo[1], thi[2] - tlo[2]) || 1;
  const target = [(tlo[0] + thi[0]) / 2, (tlo[1] + thi[1]) / 2, (tlo[2] + thi[2]) / 2];
  state.cam = { theta: -2.35, phi: 0.55, dist: Math.min(diag * 1.05, tdiag * 1.7), target };
  state.diag = diag;
  // a blast's terrain is flat, so this would fall back to a fixed lift that
  // floats each marble visibly off its own trail; its run already sits just
  // above the plane
  state.zEps = state.blast ? 0 : (state.terrain.zmax - state.terrain.zmin) * 0.05 || 0.02;

  buildUI(scene);
  if (state.blast) frameBlast();  // after buildUI: it fits the pellets around the panels
  rebuildAttention();
  hideMessage();
}

function rebuildAttention() {
  if (!attnBuf) return;
  attnBuf.count = 0;
  const layer = Math.floor(state.layerF);
  if (!state.showAttention || layer < 1) return;
  const pos = [], col = [];
  state.runs.forEach((rd, i) => {
    const att = rd.run.attention;
    if (!state.visible[i] || !att || layer - 1 >= att.shape[0]) return;
    const T = att.shape[1], w = att.data, base = (layer - 1) * T * T;
    const pts = rd.run.points.data, L = rd.L;
    for (let dst = 0; dst < T && dst < rd.N; dst++) {
      const row = Array.from({ length: T }, (_, s) => s).sort((a, b) => w[base + dst * T + b] - w[base + dst * T + a]);
      for (const src of row.slice(0, 3)) {
        if (src === dst || w[base + dst * T + src] < 0.1 || src >= rd.N) continue;
        pos.push(pts[(src * L + layer) * 3], pts[(src * L + layer) * 3 + 1], pts[(src * L + layer) * 3 + 2],
                 pts[(dst * L + layer) * 3], pts[(dst * L + layer) * 3 + 1], pts[(dst * L + layer) * 3 + 2]);
        col.push(0.506, 0.561, 0.722, 0.55 * rd.alpha,
                 0.506, 0.561, 0.722, 0.55 * rd.alpha);
      }
    }
  });
  attnBuf.count = Math.min(pos.length / 3, attnBuf.max);
  gl.bindBuffer(gl.ARRAY_BUFFER, attnBuf.bufs.pos);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(pos.slice(0, attnBuf.max * 3)));
  gl.bindBuffer(gl.ARRAY_BUFFER, attnBuf.bufs.col);
  gl.bufferSubData(gl.ARRAY_BUFFER, 0, new Float32Array(col.slice(0, attnBuf.max * 4)));
}

// ---------------------------------------------------------------- rendering
function currentMVP() {
  const c = state.cam;
  const eye = [c.target[0] + c.dist * Math.cos(c.phi) * Math.cos(c.theta),
               c.target[1] + c.dist * Math.cos(c.phi) * Math.sin(c.theta),
               c.target[2] + c.dist * Math.sin(c.phi)];
  const proj = perspective(FOVY, canvas.clientWidth / Math.max(canvas.clientHeight, 1), c.dist * 0.01, c.dist * 20);
  return matMul(proj, lookAt(eye, c.target, [0, 0, 1]));
}

function marblePositions() {
  const out = [];
  const f = state.layerF * SEG;
  state.runs.forEach((rd, i) => {
    if (!state.visible[i]) return;
    for (const t of rd.trajs) {
      const n = t.fine.length / 3;
      const i0 = Math.min(Math.floor(f), n - 1), i1 = Math.min(i0 + 1, n - 1), fr = f - Math.floor(f);
      out.push({
        p: [t.fine[i0 * 3] + (t.fine[i1 * 3] - t.fine[i0 * 3]) * fr,
            t.fine[i0 * 3 + 1] + (t.fine[i1 * 3 + 1] - t.fine[i0 * 3 + 1]) * fr,
            t.fine[i0 * 3 + 2] + (t.fine[i1 * 3 + 2] - t.fine[i0 * 3 + 2]) * fr + state.zEps],
        rgb: t.rgb, alpha: rd.alpha, ring: t.ring,
      });
    }
  });
  return out;
}

let lastT = 0;
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - lastT) / 1000, 0.1); lastT = now;
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  gl.viewport(0, 0, w, h);
  gl.clearColor(0.031, 0.043, 0.094, 1);  // --color-base #080B18
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  if (!state.scene) return;

  if (state.playing) {
    state.layerF += dt * state.speed;
    if (state.layerF > state.L - 1) state.layerF = 0;
    ui.layerSlider.value = state.layerF;
    onLayerChanged(false);
  }

  const mvp = currentMVP();
  gl.enable(gl.DEPTH_TEST);

  // A blast's terrain is the writer's flat 2 x 2 stand-in for viewers that
  // predate the record. Drawn, it is a slab whose edges cut across the
  // pellets and whose surface means nothing, so a blast skips it.
  if (!state.blast) {
    gl.useProgram(meshProg);
    gl.uniformMatrix4fv(gl.getUniformLocation(meshProg, "mvp"), false, mvp);
    gl.bindVertexArray(state.terrain.vao);
    gl.drawElements(gl.TRIANGLES, state.terrain.count, gl.UNSIGNED_INT, 0);
  }

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.depthMask(false);
  gl.useProgram(lineProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, "mvp"), false, mvp);
  if (!state.blast) {
    gl.bindVertexArray(state.terrain.wireVao);
    gl.drawArrays(gl.LINES, 0, state.terrain.wireCount);
  }
  state.runs.forEach((rd, i) => {
    if (!state.visible[i]) return;
    gl.bindVertexArray(rd.lineVao);
    gl.drawArrays(gl.LINES, 0, rd.lineCount);
  });
  if (attnBuf.count) {
    gl.bindVertexArray(attnBuf.vao);
    gl.drawArrays(gl.LINES, 0, attnBuf.count);
  }

  gl.useProgram(pointProg);
  gl.uniformMatrix4fv(gl.getUniformLocation(pointProg, "mvp"), false, mvp);
  const rimLoc = gl.getUniformLocation(pointProg, "rim");
  gl.uniform1f(rimLoc, 0);
  state.runs.forEach((rd, i) => {
    if (!state.visible[i] || !rd.dotCount) return;
    gl.bindVertexArray(rd.dotVao);
    gl.drawArrays(gl.POINTS, 0, rd.dotCount);
  });
  // decoded-token layer dots render rimmed ("open"), like the marbles
  gl.uniform1f(rimLoc, 1);
  state.runs.forEach((rd, i) => {
    if (!state.visible[i] || !rd.genDotCount) return;
    gl.bindVertexArray(rd.genDotVao);
    gl.drawArrays(gl.POINTS, 0, rd.genDotCount);
  });

  // marbles + hover highlight through the dynamic buffer
  const marbles = marblePositions();
  const pick = updatePick(mvp);
  const pos = new Float32Array(dynPoints.max * 3), col = new Float32Array(dynPoints.max * 4),
        size = new Float32Array(dynPoints.max);
  let n = 0;
  for (const m of marbles) {
    pos.set(m.p, n * 3); col.set([...m.rgb, m.alpha], n * 4); size[n] = 13 * dpr; n++;
  }
  if (pick) {
    pos.set([pick.p[0], pick.p[1], pick.p[2] + state.zEps], n * 3);
    col.set([...pick.rgb, 1], n * 4); size[n] = 17 * dpr; n++;
  }
  // rings (blast scenes: pellets whose labels break the pairing most pellets
  // follow, MTJ.blastRings) go after the marbles and the pick, and draw in a
  // pass of their own, so they sit on top
  const ringFrom = n;
  for (const m of marbles) {
    if (!m.ring) continue;
    pos.set(m.p, n * 3); col.set([...RING_RGB, 1], n * 4); size[n] = 23 * dpr; n++;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, dynPoints.bufs.pos); gl.bufferSubData(gl.ARRAY_BUFFER, 0, pos);
  gl.bindBuffer(gl.ARRAY_BUFFER, dynPoints.bufs.col); gl.bufferSubData(gl.ARRAY_BUFFER, 0, col);
  gl.bindBuffer(gl.ARRAY_BUFFER, dynPoints.bufs.size); gl.bufferSubData(gl.ARRAY_BUFFER, 0, size);
  gl.uniform1f(rimLoc, 1);
  // Marbles and the pick highlight always face the camera whole — depth
  // testing a screen-space sprite against the surface it rides on slices it
  // into a half-dome (the Plotly reference draws markers on top as well).
  gl.disable(gl.DEPTH_TEST);
  gl.bindVertexArray(dynPoints.vao);
  gl.drawArrays(gl.POINTS, 0, ringFrom);
  if (n > ringFrom) {
    gl.uniform1f(rimLoc, 2);
    gl.drawArrays(gl.POINTS, ringFrom, n - ringFrom);
  }
  gl.enable(gl.DEPTH_TEST);

  gl.depthMask(true);
  gl.disable(gl.BLEND);
  gl.bindVertexArray(null);
}

// ---------------------------------------------------------------- picking
// True 3-D picking: the cursor is unprojected to a world-space camera ray and
// cast at each visible run's segment BVH, keeping the front-most hit across
// runs. This replaces the old O(N·L)-per-frame screen-space scan over stored
// layer points — the cursor now grabs anywhere *along* a trajectory, and the
// reading it returns carries the fractional layer it landed at.

// PICK_RADIUS screen pixels as a world-space tolerance at the orbit target's
// depth, so the grab feels the same at every zoom level and scene scale.
function pickRadiusWorld(h) {
  return PICK_RADIUS * 2 * Math.tan(FOVY / 2) * state.cam.dist / Math.max(h, 1);
}

// Position on the drawn curve at a fractional layer. The BVH indexes the
// straight layer-to-layer chords, but the line the user sees is the
// Catmull-Rom densification of the same points — read the highlight off that
// so it never floats off the ribbon it is meant to be riding.
function finePoint(fine, layerFrac) {
  const n = fine.length / 3, f = layerFrac * SEG;
  const i0 = Math.min(n - 1, Math.max(0, Math.floor(f))), i1 = Math.min(i0 + 1, n - 1);
  const fr = f - i0;
  return [fine[i0 * 3] + (fine[i1 * 3] - fine[i0 * 3]) * fr,
          fine[i0 * 3 + 1] + (fine[i1 * 3 + 1] - fine[i0 * 3 + 1]) * fr,
          fine[i0 * 3 + 2] + (fine[i1 * 3 + 2] - fine[i0 * 3 + 2]) * fr];
}

// BVH hit -> the pick shape the inspector and the highlight already speak.
// The segment's meta is [traj, layer-of-its-start], and how far along the
// segment the ray landed gives the fractional layer; `layer` rounds that to
// the nearer endpoint, which is the layer every per-(layer, token) readout
// (entropy, features, quality, top-k) is indexed by.
function hitToPick(hit, rd, runIdx, multi) {
  const m = BVH.metaOf(rd.bvh, hit.index);
  if (!m) return null;
  const traj = m[0];
  const layerFrac = m[1] + BVH.segmentParam(rd.bvh, hit.index, hit.point);
  const layer = Math.min(rd.L - 1, Math.max(0, Math.round(layerFrac)));
  const t = rd.trajs[traj];
  if (!t) return null;
  return { runIdx, traj, layer, layerFrac, p: finePoint(t.fine, layerFrac),
           rgb: t.rgb, label: (multi ? rd.run.label + " · " : "") + t.label };
}

// In a blast every trail lies in one plane, so the trail nearest the camera
// under the cursor is arbitrary: it was often another pellet's, at another
// layer, rather than the marble drawn on top. The marbles at the current
// layer are what a reader points at, so they are tried first.
const MARBLE_PICK_PX = 10;
function blastMarblePick(mvp, w, h) {
  const rd = state.runs[0];
  if (!rd || !state.visible[0]) return null;
  const f = state.layerF * SEG;
  let best = null, bestD = MARBLE_PICK_PX;
  rd.trajs.forEach((t, j) => {
    const n = t.fine.length / 3;
    const i0 = Math.min(Math.floor(f), n - 1), i1 = Math.min(i0 + 1, n - 1), fr = f - Math.floor(f);
    const p = [0, 1, 2].map((a) => t.fine[i0 * 3 + a] + (t.fine[i1 * 3 + a] - t.fine[i0 * 3 + a]) * fr);
    const s = projectPoint(mvp, p, w, h);
    if (!s || s[2] <= 0) return;
    const d = Math.hypot(s[0] - state.mouse[0], s[1] - state.mouse[1]);
    if (d < bestD) { bestD = d; best = { j, p }; }
  });
  if (!best) return null;
  const t = rd.trajs[best.j];
  return { runIdx: 0, traj: best.j, layer: Math.min(rd.L - 1, Math.round(state.layerF)),
           layerFrac: state.layerF, p: best.p, rgb: t.rgb, label: t.label };
}

// The reading under the cursor right now, or null.
function hoverPick(mvp) {
  if (!state.mouse || !state.scene) return null;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (state.blast) {
    const marble = blastMarblePick(mvp, w, h);
    if (marble) return marble;
  }
  const ray = cameraRay(mvp, state.mouse[0], state.mouse[1], w, h);
  if (!ray) return null;
  const radius = pickRadiusWorld(h), multi = state.scene.runs.length > 1;
  let best = null, bestRd = null, bestIdx = -1, bestT = Infinity;
  state.runs.forEach((rd, i) => {
    if (!state.visible[i] || !rd.bvh || !rd.bvh.length) return;
    // maxDist is the best t so far, so a farther run can never take the pick
    const hit = BVH.rayPick(rd.bvh, ray.origin, ray.dir, { radius, maxDist: bestT });
    if (hit) { bestT = hit.t; best = hit; bestRd = rd; bestIdx = i; }
  });
  return best ? hitToPick(best, bestRd, bestIdx, multi) : null;
}

function updatePick(mvp) {
  state.hover = hoverPick(mvp);
  // a pinned reading holds the panel; hovering only previews when nothing is
  const shown = state.pinned || state.hover;
  setPickInfo(shown, !!state.pinned);
  return shown;
}

function setPickInfo(pick, pinned) {
  // Re-render only when the displayed reading would actually change —
  // layerFrac is compared at the one decimal the panel prints.
  const same = (a, b) => a && b && a.runIdx === b.runIdx && a.traj === b.traj &&
                         a.layer === b.layer &&
                         Math.round(a.layerFrac * 10) === Math.round(b.layerFrac * 10);
  pinned = !!pinned;
  if (pinned === state.pickPinned && (same(pick, state.pick) || (!pick && !state.pick))) {
    state.pick = pick; return;
  }
  state.pick = pick; state.pickPinned = pinned;
  if (!pick) { ui.infoPanel.hidden = true; ui.infoPanel.classList.remove("pinned"); return; }
  const run = state.scene.runs[pick.runIdx];
  const T = run.tokens.length;
  const tok = pick.traj < T ? pick.traj : T - 1; // trajectory j reads out token j (last token if fewer)
  // decoded tokens read "+"-prefixed, matching the explorer's convention
  const genTok = MTJ.isGeneratedToken(run.generation, tok);
  // between two layers the ray hit reads fractionally ("layer 8.4"); the
  // readouts below still key off the nearer integer layer
  const lf = pick.layerFrac;
  const between = typeof lf === "number" && Math.abs(lf - pick.layer) >= 0.05;
  const layerHtml = `layer <b>${between ? lf.toFixed(1) : pick.layer}</b>`;
  let html = `<div class="ip-title">${esc(pick.label)}</div>`;
  const pellet = state.scene.blast && pick.runIdx === 0
    ? MTJ.blastPellet(state.scene.blast, pick.traj, pick.layer, 140) : null;
  if (pellet) {
    // a pellet is a prompt, not a token: its text, labels and exact distance
    // from this layer's centroid stand where the token line would
    html += `<div>${layerHtml}</div><div class="ip-text">${esc(pellet.text)}</div>` +
            `<div class="mono">${pellet.labels.map(([k, v]) => `${esc(k)} ${esc(v)}`).join(" · ")}</div>`;
    if (pellet.range !== null)
      html += `<div>range <b>${pellet.range.toFixed(3)}</b><span class="dim"> exact, ` +
              `from this layer's centroid in mean state norms</span></div>`;
  } else {
    html += `<div>${layerHtml} · token <span class="mono">'${esc((genTok ? "+" : "") + (run.tokens[tok] ?? "?"))}'</span></div>`;
  }
  const step = genTok ? MTJ.generationStep(run.generation, tok) : null;
  if (step) {
    const bits = [];
    if (typeof step.p === "number") bits.push(`p ${(step.p * 100).toFixed(1)}%`);
    if (typeof step.entropy === "number") bits.push(`entropy ${step.entropy.toFixed(2)} nats`);
    if (bits.length)
      html += `<div><span class="dim">decode step</span> ${bits.join(" · ")}</div>`;
  }
  if (run.entropy) {
    const [L, Te] = run.entropy.shape;
    if (pick.layer < L && tok < Te)
      html += `<div>entropy ${run.entropy.data[pick.layer * Te + tok].toFixed(2)} nats</div>`;
  }
  if (run.features) {
    // dominant SAE feature at this (layer, token); layers where the
    // dictionary doesn't fit (recon error > 50% of norm) are flagged
    const feat = MTJ.featureAt(run.features, pick.layer, tok);
    if (feat) {
      const re = run.features.recon_error;
      const extrap = re && pick.layer < re.data.length && re.data[pick.layer] > 0.5;
      html += `<div><span class="mono">feature f${feat.id} · ${feat.act.toFixed(2)}</span>` +
              (extrap ? `<span class="dim"> (extrapolation)</span>` : "") + `</div>`;
    }
  }
  if (run.quality) {
    const [Lq, Tq] = run.quality.shape;
    if (pick.layer < Lq && tok < Tq) {
      const pres = run.quality.data[pick.layer * Tq + tok];
      html += `<div>nbhd preserved <b>${(pres * 100).toFixed(0)}%</b>` +
              `<span class="dim"> (2-D fidelity)</span></div>`;
    }
  }
  if (run.topk && run.topk[pick.layer] && run.topk[pick.layer][tok]) {
    const rows = run.topk[pick.layer][tok].slice(0, 5).map(([t, p]) =>
      `<div><span class="bar" style="width:${Math.max(2, p * 120)}px"></span>` +
      `<span class="mono">'${esc(t)}'</span> <span class="dim">${(p * 100).toFixed(1)}%</span></div>`).join("");
    html += `<div class="topk"><span class="dim">top-k readout</span>${rows}</div>`;
  }
  if (run.inspector) {
    // Nearest vocabulary tokens to this hidden state (cosine in embedding
    // space, nearest first) — the explorer's representation-space neighbors,
    // resolved at export because the (V, D) embedding matrix dwarfs the
    // scene. Bars echo the top-k rows but read as similarity, not
    // probability, so they are drawn in a muted accent.
    const nbrs = MTJ.neighborsAt(run.inspector, pick.layer, tok, 5);
    if (nbrs.length) {
      const rows = nbrs.map((n) =>
        `<div><span class="bar" style="width:${Math.max(2, Math.max(0, n.sim) * 120)}px"></span>` +
        `<span class="mono">'${esc(n.token)}'</span> <span class="dim">cos ${n.sim.toFixed(3)}</span></div>`).join("");
      html += `<div class="topk nbrs"><span class="dim">nearest tokens</span>${rows}</div>`;
    }
    // Attention/MLP split of the block write that produced this state. Layer 0
    // is the embedding stream — written by no block — so the section is simply
    // absent there, as it is for runs captured without residual components.
    const share = MTJ.componentShareAt(run.inspector, pick.layer, tok);
    const sum = share ? share.attn + share.mlp : 0;
    if (share && Number.isFinite(sum) && sum > 1e-9) {
      const a = Math.min(1, Math.max(0, share.attn / sum)) * 100;
      html += `<div class="resid"><span class="dim">residual write</span>` +
              `<div class="resid-bar"><span class="attn" style="width:${a.toFixed(1)}%"></span>` +
              `<span class="mlp" style="width:${(100 - a).toFixed(1)}%"></span></div>` +
              `<div class="mono">attn ${a.toFixed(0)}%<span class="dim"> · </span>` +
              `mlp ${(100 - a).toFixed(0)}%</div></div>`;
    }
  }
  if (pinned) html += `<div class="ip-pin">pinned · Esc to clear</div>`;
  ui.infoPanel.innerHTML = html;
  ui.infoPanel.classList.toggle("pinned", pinned);
  ui.infoPanel.hidden = false;
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------------------------------------------------------- UI
const ui = {
  overlay: document.getElementById("overlay"),
  overlayMsg: document.getElementById("overlay-msg"),
  dropShade: document.getElementById("drop-shade"),
  meta: document.getElementById("meta"),
  runsPanel: document.getElementById("runs-panel"),
  runsList: document.getElementById("runs-list"),
  comparisons: document.getElementById("comparisons"),
  attnRow: document.getElementById("attn-row"),
  attnToggle: document.getElementById("attnToggle"),
  uncertaintyRow: document.getElementById("uncertainty-row"),
  uncertaintyToggle: document.getElementById("uncertaintyToggle"),
  infoPanel: document.getElementById("info-panel"),
  readingPanel: document.getElementById("reading-panel"),
  readingBody: document.getElementById("reading-body"),
  hudBottom: document.getElementById("hud-bottom"),
  playBtn: document.getElementById("playBtn"),
  speedSel: document.getElementById("speedSel"),
  layerSlider: document.getElementById("layerSlider"),
  layerLabel: document.getElementById("layerLabel"),
  openBtn: document.getElementById("openBtn"),
  fileInput: document.getElementById("fileInput"),
  hudTop: document.getElementById("hud-top"),
  runsHead: document.getElementById("runs-head"),
  blastPanel: document.getElementById("blast-panel"),
  blastLayouts: document.getElementById("blastLayouts"),
  blastColour: document.getElementById("blastColour"),
  blastLegend: document.getElementById("blastLegend"),
  blastReadout: document.getElementById("blastReadout"),
  blastCaption: document.getElementById("blastCaption"),
  blastCaptionBody: document.getElementById("blastCaptionBody"),
};

function showMessage(msg, isError) {
  ui.overlayMsg.textContent = msg;
  ui.overlay.classList.toggle("error", !!isError);
  ui.overlay.style.display = "flex";
}
function hideMessage() { ui.overlay.style.display = "none"; }

function buildUI(scene) {
  const meta = scene.meta || {};
  ui.meta.textContent = [meta.model && `model: ${meta.model}`, meta.backend && `backend: ${meta.backend}`]
    .filter(Boolean).join("  ·  ");
  ui.runsPanel.hidden = false;
  ui.hudBottom.hidden = false;
  ui.layerSlider.max = state.L - 1;
  ui.layerSlider.value = state.layerF;
  onLayerChanged(false);

  ui.runsList.innerHTML = "";
  let colorBase = 0;
  scene.runs.forEach((run, i) => {
    const sw = PALETTE[colorBase % PALETTE.length];
    colorBase += run.points.shape[0];
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = true;
    cb.addEventListener("change", () => {
      state.visible[i] = cb.checked;
      // a pin on a run that just went invisible would leave the highlight
      // floating over nothing
      if (!cb.checked && state.pinned && state.pinned.runIdx === i) state.pinned = null;
      rebuildAttention();
    });
    label.appendChild(cb);
    label.insertAdjacentHTML("beforeend",
      `<span class="swatch" style="background:${sw}"></span><b>${esc(run.label)}</b>` +
      (run.model ? `<span class="model mono" title="${esc(run.model)}">${esc(run.model)}</span>` : "") +
      `<span class="prompt" title="${esc(run.prompt)}">${esc(run.prompt)}</span>`);
    if (run.generation) {
      // decode summary: mode (+temperature when sampled), token count, and
      // the decoded continuation — the "+" trajectories on the scene
      const cont = MTJ.continuationText(run.generation, meta.backend);
      label.insertAdjacentHTML("beforeend",
        `<span class="gen-summary" title="${esc(cont)}">${esc(MTJ.decodeSummary(run.generation))}` +
        (cont ? ` &#8594; <span class="mono">${esc(cont)}</span>` : "") + `</span>`);
    }
    if (run.features) {
      // SAE feature-layer summary: dictionary, best-fitting layer, and how
      // much of the norm its reconstruction still misses there
      const fitLine = MTJ.featureFitSummary(run.features);
      if (fitLine) label.insertAdjacentHTML("beforeend",
        `<span class="feat-summary" title="${esc(run.features.hook || "")}">${esc(fitLine)}</span>`);
    }
    ui.runsList.appendChild(label);
  });

  if (scene.comparisons.length) {
    ui.comparisons.innerHTML = "<table><tr><th></th><th>hausdorff</th><th>dtw</th><th>shared</th></tr>" +
      scene.comparisons.map((c) =>
        `<tr><td><b>${esc(c.label ?? "?")}</b></td><td>${fmt(c.hausdorff)}</td>` +
        `<td>${fmt(c.dtw_normalized)}</td><td>${c.shared_tokens ?? "–"}</td></tr>`).join("") + "</table>";
  } else ui.comparisons.innerHTML = "";

  const hasAttn = scene.runs.some((r) => r.attention);
  ui.attnRow.hidden = !hasAttn;
  ui.attnToggle.checked = state.showAttention = false;

  // Uncertainty defaults ON when the scene carries a standard-error field:
  // the confidence of the terrain should be visible before the viewer trusts
  // its shape, not something a user has to opt into.
  const hasSE = !!(scene.terrain && scene.terrain.se);
  ui.uncertaintyRow.hidden = !hasSE;
  ui.uncertaintyToggle.checked = state.showUncertainty = hasSE;
  setTerrainColors(state.showUncertainty);

  // the caption is how a blast is read, so it starts open where there is
  // room for it; on a phone it would bury the scene
  if (scene.blast) ui.blastCaption.open = !window.matchMedia("(max-width: 720px)").matches;
  renderBlastPanel();
  renderReading(scene);
}

/* The panel that makes a shared scene self-describing: how much of it
 * survived the projection, what the terrain does and does not mean, and what
 * produced it. Collapsed by default — it is the answer to a question the
 * reader has, not a wall between them and the picture. */
function renderReading(scene, keepOpen) {
  const R = window.Reading;
  if (!R) { ui.readingPanel.hidden = true; return; }
  const summary = R.sceneSummary(scene);
  const rows = [];

  const fidelity = R.fidelityNote(summary.fidelity);
  if (fidelity) {
    const low = summary.fidelity.low_fraction > 0.25 ? " warn" : "";
    rows.push(`<p class="reading-fidelity${low}">${esc(fidelity)}</p>`);
  }
  if (scene.blast) {
    // A blast has no density terrain, so the terrain note and the
    // density-uncertainty line would describe something the scene does not
    // have. The fidelity line above is the current layout's.
    for (const [heading, body] of BLAST_NOTES)
      rows.push(`<p><b>${esc(heading)}.</b> ${esc(body)}</p>`);
    for (const [heading, body] of R.NOTES)
      if (!/terrain/i.test(heading)) rows.push(`<p><b>${esc(heading)}.</b> ${esc(body)}</p>`);
  } else {
    rows.push(`<p>${esc(R.uncertaintyNote(summary.hasUncertainty))}</p>`);
    for (const [heading, body] of R.NOTES)
      rows.push(`<p><b>${esc(heading)}.</b> ${esc(body)}</p>`);
  }

  const p = summary.provenance;
  if (p) {
    const models = p.models.map((m) =>
      esc(m.id) + (m.revision ? ` <span class="mono dim">@${esc(m.revision.slice(0, 7))}</span>` : "")
    ).join(", ");
    const bits = [`${p.prompts} prompt${p.prompts === 1 ? "" : "s"}`];
    if (p.projection) bits.push(`${esc(p.projection)} projection`);
    if (p.density) bits.push(`${esc(p.density)} density`);
    if (p.seed !== null) bits.push(`seed ${p.seed}`);
    if (p.sae && p.sae.source) bits.push(`SAE ${esc(p.sae.source)}`);
    rows.push(
      `<p class="reading-prov"><b>What produced this.</b> ${models} · ` +
      `${bits.join(" · ")}${p.created ? ` · ${esc(p.created)}` : ""}` +
      `${p.mottled ? ` · mottled ${esc(p.mottled)}` : ""}. ` +
      `This is the parameterization a reproduction needs, not evidence that ` +
      `the run reproduces.</p>`);
  } else {
    rows.push('<p class="reading-prov">This scene carries no analysis record, ' +
              'so it cannot say what produced it. Scenes exported by a current ' +
              'Mottled do.</p>');
  }

  ui.readingBody.innerHTML = rows.join("");
  ui.readingPanel.hidden = false;
  if (!keepOpen) ui.readingPanel.open = false;
}

const fmt = (v) => (typeof v === "number" ? v.toFixed(3) : "–");

// ---------------------------------------------------------------- blast
// A blast scene (blast.py) is one pellet per prompt, drawn flat under one of
// several layouts. Run 0 holds the first layout; switching rewrites its x, y
// in place (z keeps the writer's small lift) and rebuilds everything derived
// from those points. What a colour, a ring or a readout says is decided by
// the MTJ.blast* helpers, which node --test pins; this section only draws.
const BLAST_TRAIL = 0.4;   // trail alpha; the marbles at the scrubbed layer stay at 1
const RING_RGB = hexRGB(MTJ.BLAST_COLOURS.ring);

// docs/validity.md's own wording for a blast monitor split
const BLAST_NOTES = [
  ["No terrain",
   "A blast scene estimates no density, so there is no height to read and no " +
   "density uncertainty to show."],
  ["What a split is",
   "A split under a monitor layout means these labelled prompts are linearly " +
   "separable, held out, at that layer. Not what the model computes, " +
   "and not yet a monitor: check wording, contrast pairs and surface features " +
   "first, then confirm on held-out prompts."],
];

function blastLayout() {
  return state.scene.blast.layouts[state.blast.layout];
}

function applyBlastLayout() {
  const run = state.scene.runs[0], lay = blastLayout();
  const p = run.points.data, q = lay.positions.data;
  // positions (N, L, 2) and points (N, L, 3) share their pellet-major order
  for (let i = 0, n = q.length / 2; i < n; i++) {
    p[i * 3] = q[i * 2]; p[i * 3 + 1] = q[i * 2 + 1];
  }
  // the inspector's nbhd line and the reading panel's fidelity are this layout's
  run.quality = lay.quality;
  state.blast.colour = MTJ.blastColourLabel(state.scene.blast, lay,
                                            state.blast.chosen, state.blast.colour);
}

function blastLook() {
  const b = state.scene.blast;
  if (!b || !state.blast) return null;
  const name = state.blast.colour;
  const rings = MTJ.blastRings(b, name, blastLayout().driver);
  return { colour: (j) => MTJ.blastPelletColour(b.pellets[j], name),
           ring: (j) => !!(rings && rings.ringed[j]), trail: BLAST_TRAIL };
}

function rebuildBlastRun() {
  for (const o of state.runs[0].owned)
    if (o instanceof WebGLVertexArrayObject) gl.deleteVertexArray(o); else gl.deleteBuffer(o);
  state.runs[0] = buildRun(state.scene.runs[0], 0, 0, 1, blastLook());
  // a pin follows its pellet into the new layout instead of floating where
  // the old one drew it; clearing state.pick makes the panel re-read it
  if (state.pinned) {
    const t = state.runs[0].trajs[state.pinned.traj];
    state.pinned = { ...state.pinned, p: finePoint(t.fine, state.pinned.layerFrac), rgb: t.rgb };
  }
  state.pick = null;
}

// The part of the canvas no panel covers. Each panel cuts the free rectangle
// from whichever side keeps the most of it — the loader from the top, the
// desktop pellet panel from the right, the phone one from the bottom — so
// the framing follows the CSS layout without restating its breakpoints.
function freeRect() {
  let f = { l: 0, t: 0, r: canvas.clientWidth, b: canvas.clientHeight };
  const area = (c) => Math.max(0, c.r - c.l) * Math.max(0, c.b - c.t);
  for (const el of [ui.hudTop, ui.runsPanel, ui.hudBottom]) {
    if (el.hidden) continue;
    const r = el.getBoundingClientRect();
    if (r.right <= f.l || r.left >= f.r || r.bottom <= f.t || r.top >= f.b) continue;
    f = [{ ...f, t: r.bottom }, { ...f, b: r.top }, { ...f, l: r.right }, { ...f, r: r.left }]
      .reduce((a, c) => (area(c) > area(a) ? c : a));
  }
  return f;
}

// A blast is drawn flat (x, y per layer and a hair of z), so it is read from
// above: theta -pi/2 puts +x to the right and +y up, and the distance fits
// the pellets' footprint into the free part of the canvas, not a 3-D
// diagonal into all of it.
function frameBlast() {
  const p = state.scene.runs[0].points.data;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < p.length; i += 3) {
    x0 = Math.min(x0, p[i]); x1 = Math.max(x1, p[i]);
    y0 = Math.min(y0, p[i + 1]); y1 = Math.max(y1, p[i + 1]);
  }
  const W = Math.max(canvas.clientWidth, 1), H = Math.max(canvas.clientHeight, 1);
  let f = freeRect();
  if (f.r - f.l < 80 || f.b - f.t < 80) f = { l: 0, t: 0, r: W, b: H };
  const hx = Math.max((x1 - x0) / 2, 1e-6), hy = Math.max((y1 - y0) / 2, 1e-6);
  const s = Math.min((f.r - f.l) / (2 * hx), (f.b - f.t) / (2 * hy)) / 1.15; // px per unit
  // offset the orbit target so the pellets centre in the free rectangle
  state.cam = { theta: -Math.PI / 2, phi: 1.45, dist: H / (2 * s * Math.tan(FOVY / 2)),
                target: [(x0 + x1) / 2 - ((f.l + f.r) / 2 - W / 2) / s,
                         (y0 + y1) / 2 + ((f.t + f.b) / 2 - H / 2) / s, 0] };
  state.diag = Math.hypot(x1 - x0, y1 - y0) || 1;
}

function setBlastLayout(k) {
  state.blast.layout = k;
  applyBlastLayout();
  rebuildBlastRun();
  renderBlastPanel();
  frameBlast();
  renderReading(state.scene, true);
}

function setBlastColour(name) {
  state.blast.chosen = state.blast.colour = name;
  rebuildBlastRun();
  renderBlastPanel();
}

// The pellet panel takes the runs panel's place: one run of N pellets has
// nothing to toggle, and the layout, colouring and caption are what a blast
// is read by.
function renderBlastPanel() {
  const b = state.scene && state.scene.blast;
  ui.runsPanel.classList.toggle("blast", !!b);
  ui.blastPanel.hidden = !b;
  ui.runsList.hidden = !!b;
  ui.runsHead.textContent = b ? `${b.pellets.length} pellets` : "runs";
  if (!b) return;
  const lay = blastLayout();
  ui.blastLayouts.innerHTML = "";
  b.layouts.forEach((l, k) => {
    const btn = document.createElement("button");
    btn.className = "seg-btn";
    btn.textContent = l.name;
    btn.setAttribute("aria-pressed", String(k === state.blast.layout));
    btn.addEventListener("click", () => { if (k !== state.blast.layout) setBlastLayout(k); });
    ui.blastLayouts.appendChild(btn);
  });
  const names = MTJ.blastLabelNames(b);
  // a family without labels has nothing to colour by
  ui.blastColour.closest(".blast-colour").hidden = !names.length;
  ui.blastColour.innerHTML = names
    .map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
  ui.blastColour.value = state.blast.colour;
  const lg = MTJ.blastLegend(b, lay, state.blast.colour);
  ui.blastLegend.innerHTML = !lg ? "" : lg.rows.map((r) =>
    `<span class="lg"><span class="blast-swatch" style="background:${r.colour}"></span>` +
    `${esc(r.text)} <span class="dim">(${r.count})</span></span>`).join("") +
    (lg.ring ? `<span class="lg"><span class="blast-swatch ring"></span>${esc(lg.ring.text)} ` +
               `<span class="dim">(${lg.ring.count})</span></span>` : "");
  ui.blastCaptionBody.innerHTML = MTJ.blastCaption(b, lay).map((c) =>
    `<p><span class="cap-kind">${esc(c.kind)}</span> ${esc(c.text)}</p>`).join("");
  renderBlastReadout();
}

// the readout at the scrubbed layer; onLayerChanged calls this only when the
// integer layer changes, so playback does not rewrite the DOM every frame
function renderBlastReadout() {
  const l = Math.floor(state.layerF);
  state.blast.shownLayer = l;
  const r = MTJ.blastReadout(state.scene.blast, blastLayout(), l);
  // each " · " part is kept whole, so a narrow panel breaks the line between
  // the AUROC and its null band rather than inside "5–95%"
  const parts = r ? r.text.split(" \u00b7 ").map((t) => `<span class="part">${esc(t)}</span>`) : [];
  ui.blastReadout.innerHTML = `<span class="part dim">layer ${l}</span> ` +
    (r ? parts.join(' <span class="dim">\u00b7</span> ')
       : `<span class="dim">no readout recorded</span>`) +
    (r && r.flag ? `<div class="blast-flag">${esc(r.flag)}</div>` : "");
}

function onLayerChanged(fromSlider) {
  if (fromSlider) state.layerF = parseFloat(ui.layerSlider.value);
  const l = Math.floor(state.layerF);
  ui.layerLabel.textContent = `layer ${l} / ${state.L - 1}`;
  if (l !== state._lastAttnLayer) { state._lastAttnLayer = l; rebuildAttention(); }
  if (state.blast && l !== state.blast.shownLayer) renderBlastReadout();
}

ui.layerSlider.addEventListener("input", () => { state.playing = false; ui.playBtn.innerHTML = "&#9654;"; onLayerChanged(true); });
ui.playBtn.addEventListener("click", () => {
  state.playing = !state.playing;
  ui.playBtn.innerHTML = state.playing ? "&#10074;&#10074;" : "&#9654;";
});
ui.speedSel.addEventListener("change", () => { state.speed = parseFloat(ui.speedSel.value); });
ui.attnToggle.addEventListener("change", () => { state.showAttention = ui.attnToggle.checked; rebuildAttention(); });
ui.uncertaintyToggle.addEventListener("change", () => {
  state.showUncertainty = ui.uncertaintyToggle.checked;
  setTerrainColors(state.showUncertainty);
});
ui.blastColour.addEventListener("change", () => setBlastColour(ui.blastColour.value));
ui.openBtn.addEventListener("click", () => ui.fileInput.click());
ui.fileInput.addEventListener("change", () => {
  if (ui.fileInput.files[0]) loadBlob(ui.fileInput.files[0], ui.fileInput.files[0].name);
});

// camera controls
//
// Pointer events rather than mouse events, so a finger and a mouse take the
// same path. What touch needs beyond that is a second finger: there is no
// wheel to zoom with and no right button to pan with, so two pointers do
// both — the distance between them zooms, their midpoint pans. (The canvas
// also sets `touch-action: none`; without it the browser claims the drag as
// a page scroll and the pointer stream stops mid-gesture.)
let drag = null;
const touches = new Map();   // active pointers, for pinch/two-finger pan
let pinch = null;

const pinchState = () => {
  const [a, b] = [...touches.values()];
  return {
    dist: Math.hypot(a.x - b.x, a.y - b.y),
    cx: (a.x + b.x) / 2, cy: (a.y + b.y) / 2,
  };
};

function zoomBy(factor) {
  const d = state.diag || 10;
  state.cam.dist = Math.min(d * 12, Math.max(d * 0.03, state.cam.dist * factor));
}

canvas.addEventListener("pointerdown", (e) => {
  touches.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (touches.size === 2) {
    // Second finger down: this gesture is a pinch, not an orbit. Drop the
    // orbit in progress so the view does not lurch as the fingers settle.
    drag = null;
    pinch = pinchState();
    return;
  }
  if (touches.size > 2) return;
  drag = { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, moved: false,
           pan: e.button === 2 || e.shiftKey };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (touches.has(e.pointerId)) touches.set(e.pointerId, { x: e.clientX, y: e.clientY });

  if (pinch && touches.size === 2) {
    const now = pinchState();
    if (now.dist > 0 && pinch.dist > 0) zoomBy(pinch.dist / now.dist);
    const c = state.cam, k = c.dist * 0.0012;
    const dx = now.cx - pinch.cx, dy = now.cy - pinch.cy;
    const st = Math.sin(c.theta), ct = Math.cos(c.theta);
    const sp = Math.sin(c.phi), cp = Math.cos(c.phi);
    const right = [-st, ct, 0], up = [-sp * ct, -sp * st, cp];
    for (let d = 0; d < 3; d++) c.target[d] += (-dx * right[d] + dy * up[d]) * k;
    pinch = now;
    return;
  }

  state.mouse = [e.clientX, e.clientY];
  if (!drag) return;
  // past a few pixels this is an orbit/pan, not a click — so it can't pin
  if (Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) > 3) drag.moved = true;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  const c = state.cam;
  if (drag.pan) {
    const k = c.dist * 0.0012;
    const st = Math.sin(c.theta), ct = Math.cos(c.theta), sp = Math.sin(c.phi), cp = Math.cos(c.phi);
    const right = [-st, ct, 0], up = [-sp * ct, -sp * st, cp]; // camera basis, world z up
    for (let d = 0; d < 3; d++) c.target[d] += (-dx * right[d] + dy * up[d]) * k;
  } else {
    c.theta -= dx * 0.006;
    c.phi = Math.min(1.55, Math.max(-1.55, c.phi + dy * 0.006));
  }
});
// A click that didn't orbit or pan pins whatever is under the cursor, and
// clears the pin when that is empty space. The ray is re-cast here rather
// than reusing the last frame's hover so a move-then-click lands where the
// cursor actually is.
canvas.addEventListener("pointerup", (e) => {
  touches.delete(e.pointerId);
  if (touches.size < 2) pinch = null;
  const click = drag && !drag.moved && !drag.pan && e.button === 0;
  drag = null;
  if (click && state.scene) state.pinned = hoverPick(currentMVP());
});
// A cancelled pointer (the browser taking over, a call arriving) must clear
// the same state an up would, or the next touch starts mid-gesture.
canvas.addEventListener("pointercancel", (e) => {
  touches.delete(e.pointerId);
  if (touches.size < 2) pinch = null;
  drag = null;
});
canvas.addEventListener("pointerleave", (e) => {
  touches.delete(e.pointerId);
  if (touches.size < 2) pinch = null;
  state.mouse = null;
});
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.pinned) { state.pinned = null; e.preventDefault(); }
});
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  zoomBy(Math.exp(e.deltaY * 0.0012));
}, { passive: false });

// drag & drop
let dragDepth = 0;
window.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; ui.dropShade.hidden = false; });
window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; ui.dropShade.hidden = true; } });
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0; ui.dropShade.hidden = true;
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (!f) return;
  // Weights and scenes both arrive by drop. Dispatch on the name rather than
  // trying a scene parse first: a 600 MB model failing as "corrupt scene" is
  // a confusing way to learn you dropped the right file in the right place.
  if (/\.(mwt|gguf)$/i.test(f.name)) {
    f.arrayBuffer()
      .then((buf) => useModelBuffer(buf, f.name))
      .catch((err) => setModelStatus(`could not read ${f.name}: ${err.message}`, true));
  } else {
    loadBlob(f, f.name);
  }
});

// ---------------------------------------------------------------- loading
function loadBuffer(buf, name) {
  try {
    setScene(MTJ.loadScene(buf));
  } catch (err) {
    showMessage(`${name}: ${err.message}`, err.kind !== "trajectory");
    console.warn(err);
  }
}
function loadBlob(blob, name) {
  blob.arrayBuffer().then((buf) => loadBuffer(buf, name))
    .catch((err) => showMessage(`could not read ${name}: ${err.message}`, true));
}
async function loadURL(url, quiet) {
  try {
    const res = await fetch(url); // relative to the page
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    loadBuffer(await res.arrayBuffer(), url);
    return true;
  } catch (err) {
    if (!quiet) showMessage(`could not fetch ${url}: ${err.message}`, true);
    return false;
  }
}

// ------------------------------------------------- in-browser capture
// The viewer can run the model itself: weights are fetched into the page
// (cached, so the second visit is instant), the prompt is tokenized here, and
// the forward pass, projection and terrain are all computed client-side. The
// result is the object MTJ.loadScene produces, so nothing below setScene knows
// whether a scene was captured or loaded.
const local = { weights: null, ops: null, name: null, gpu: null, busy: false };

const el = (id) => document.getElementById(id);

function setModelStatus(text, isError) {
  const s = el("modelStatus");
  s.textContent = text;
  s.classList.toggle("err", !!isError);
  el("model-progress").hidden = !text;
}

function setProgress(fraction) {
  el("model-bar").style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
}

/* Prefer WebGPU, fall back to the CPU reference.
 *
 * The CPU path is the one pinned against HuggingFace, so falling back is a
 * slowdown and never a correctness change — but on a real model it is a large
 * enough slowdown to be worth saying out loud rather than letting someone
 * wonder why a capture is taking a minute. */
async function ensureOps() {
  if (local.gpu !== null) return;
  if (typeof MottledOpsWebGPU === "undefined") { local.gpu = false; }
  else {
    try { local.ops = await MottledOpsWebGPU.create(); local.gpu = true; }
    catch (err) {
      console.warn("WebGPU unavailable, using the CPU reference:", err.message);
      local.gpu = false;
    }
  }
  const badge = el("gpuBadge");
  badge.textContent = local.gpu ? "webgpu" : "cpu";
  badge.classList.toggle("gpu", local.gpu);
  badge.title = local.gpu
    ? "Running on the GPU."
    : "No WebGPU here, so the CPU reference runs instead — same numbers, slower.";
}

async function useModelBuffer(buf, name) {
  const isGGUF = new DataView(buf).getUint32(0, true) === 0x46554747;
  // A GGUF that this stack cannot fully implement throws here rather than
  // loading and being quietly wrong; let that message reach the user as-is.
  local.weights = isGGUF ? MottledGGUF.open(buf) : MottledWeights.open(buf);
  local.name = name;
  await ensureOps();

  const c = local.weights.config;
  if (!local.weights.tokenizer) {
    throw new Error(
      `${name} carries no tokenizer, so a typed prompt cannot be encoded. ` +
      `GGUF files embed one; a .mwt needs to be exported with it.`);
  }

  el("capture-row").hidden = false;
  el("captureModel").textContent =
    `${c.numLayers}x${c.hiddenSize} · ${c.architecture || "mwt"}`;
  el("captureHint").textContent = "⌘/Ctrl+Enter to capture";
  setModelStatus(`${name} ready`);
  setProgress(1);
  renderModelList();
}

async function loadModelFrom(url, name) {
  if (local.busy) return;
  local.busy = true;
  renderModelList();
  try {
    setProgress(0);
    setModelStatus(`fetching ${name}…`);
    let lastPaint = 0;
    const buf = await MottledModels.fetchWeights(url, (got, total, fromCache) => {
      if (fromCache) {
        setProgress(1);
        setModelStatus(`${name} — already downloaded, loading from cache`);
        return;
      }
      const now = performance.now();
      if (now - lastPaint < 100) return;   // repainting per chunk is wasted work
      lastPaint = now;
      setProgress(total ? got / total : 0);
      setModelStatus(total
        ? `downloading ${name} — ${MottledModels.humanBytes(got)} of ` +
          `${MottledModels.humanBytes(total)}`
        : `downloading ${name} — ${MottledModels.humanBytes(got)}`);
    });
    setModelStatus(`preparing ${name}…`);
    // Yield so the status paints before the parse blocks the thread.
    await new Promise((r) => setTimeout(r, 0));
    await useModelBuffer(buf, name);
  } catch (err) {
    setModelStatus(`could not load ${name}: ${err.message}`, true);
    setProgress(0);
    console.warn(err);
  } finally {
    local.busy = false;
    renderModelList();
  }
}

/* The picker. Each entry says its size before anything is downloaded —
 * several hundred megabytes is a thing a person should get to decline — and
 * says when a model is already cached, which is what makes a return visit
 * feel different from a first one. */
let cachedURLs = [];
function renderModelList() {
  const list = el("model-list");
  list.innerHTML = "";
  for (const m of MottledModels.CATALOG) {
    const b = document.createElement("button");
    b.className = "model-card" + (local.name === m.name ? " active" : "");
    b.disabled = local.busy;
    const here = cachedURLs.includes(m.url);
    b.innerHTML =
      `<div class="row1"><span class="nm"></span>` +
      `<span class="sz"></span></div>` +
      `<div class="row2"></div>`;
    b.querySelector(".nm").textContent = m.name;
    b.querySelector(".sz").textContent = here
      ? "cached" : MottledModels.humanBytes(m.bytes);
    if (here) b.querySelector(".sz").classList.add("ready");
    b.querySelector(".row2").textContent = m.detail;
    b.title = `${m.note}\n${m.license} · ${m.url}`;
    b.addEventListener("click", () => loadModelFrom(m.url, m.name));
    list.appendChild(b);
  }
}

async function refreshCached() {
  cachedURLs = await MottledModels.listCached();
  renderModelList();
}

async function captureLocally(prompts) {
  if (local.busy || !local.weights) return;
  local.busy = true;
  const btn = el("captureBtn");
  btn.disabled = true;
  const label = btn.textContent;
  try {
    const scene = await MottledCapture.capture(local.weights, prompts, {
      ops: local.ops,
      modelName: local.name,
      onProgress: (phase, a, b) => {
        btn.textContent = phase === "readout" ? `Readout ${a}/${b}`
          : phase === "forward" ? `Forward ${a + 1}/${b}` : "Building scene";
        // Let the button repaint between layers; without this the whole
        // capture is one frozen frame.
        return new Promise((r) => setTimeout(r, 0));
      },
    });
    setScene(scene);
    el("captureHint").textContent =
      `${scene.runs.length} run${scene.runs.length > 1 ? "s" : ""} captured`;
  } catch (err) {
    el("captureHint").textContent = `capture failed: ${err.message}`;
    console.warn(err);
  } finally {
    local.busy = false;
    btn.disabled = false;
    btn.textContent = label;
    renderModelList();
  }
}

function promptsFromBox() {
  return el("capturePrompts").value.split("\n")
    .map((s) => s.trim()).filter(Boolean);
}

(function wireLocalCapture() {
  // The panel starts collapsed: the page already shows a sample scene, and a
  // model download should be something asked for rather than arrived at.
  el("liveToggle").addEventListener("click", () => {
    const panel = el("live-panel");
    panel.hidden = !panel.hidden;
    el("liveToggle").textContent = panel.hidden
      ? "Run a model in this page \u203a" : "Run a model in this page \u2039";
    if (!panel.hidden) { ensureOps(); refreshCached(); }
  });

  const examples = el("examples");
  for (const ex of MottledModels.EXAMPLES) {
    const b = document.createElement("button");
    b.className = "example";
    b.textContent = ex.label;
    b.title = ex.note;
    b.addEventListener("click", () => {
      el("capturePrompts").value = ex.prompts.join("\n");
    });
    examples.appendChild(b);
  }

  el("modelBtn").addEventListener("click", () => {
    const url = el("modelUrl").value.trim();
    if (url) loadModelFrom(url, url.split("/").pop() || url);
  });
  el("modelUrl").addEventListener("keydown", (e) => {
    if (e.key === "Enter") el("modelBtn").click();
  });

  el("captureBtn").addEventListener("click", () => {
    if (!local.weights) return;   // the server path binds its own handler
    const p = promptsFromBox();
    if (p.length) captureLocally(p);
  });
  el("capturePrompts").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      const p = promptsFromBox();
      if (p.length) captureLocally(p);
    }
  });

  renderModelList();
})();


// Capture backend (serve.py) discovery: when /api/health answers, the
// browser can generate new scenes directly instead of only loading files.
(async function discoverBackend() {
  let health;
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return;
    health = await res.json();
  } catch { return; }  // plain static hosting: no backend, no form
  const row = document.getElementById("capture-row");
  const promptsEl = document.getElementById("capturePrompts");
  const btn = document.getElementById("captureBtn");
  row.hidden = false;
  document.getElementById("captureModel").textContent = `model: ${health.model}`;
  btn.addEventListener("click", async () => {
    const prompts = promptsEl.value.split("\n").map((s) => s.trim()).filter(Boolean);
    if (!prompts.length) return;
    btn.disabled = true; btn.textContent = "Capturing";
    try {
      const res = await fetch("/api/scene", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompts }),
      });
      if (!res.ok) throw new Error((await res.json()).error || `HTTP ${res.status}`);
      loadBuffer(await res.arrayBuffer(), prompts[0]);
    } catch (err) {
      showMessage(`capture failed: ${err.message}`, true);
    } finally {
      btn.disabled = false; btn.textContent = "Capture";
    }
  });
})();

(async function boot() {
  if (!gl) { showMessage("WebGL2 is not available in this browser", true); return; }
  requestAnimationFrame(frame);
  const param = new URLSearchParams(location.search).get("file");
  if (param) { await loadURL(param, false); return; }
  await loadURL("samples/scene-abc.mtj", true); // graceful fallback to the drop prompt
})();
