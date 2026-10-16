/* scene.js — build a Mottled scene in the browser.
 *
 * A port of the Python scene pipeline (`projection.py` → `density.py` →
 * `terrain.py`, assembled by `pipeline.py`): projected coordinates, a density
 * landscape, a terrain height map, and draped trajectories. Until now that
 * pipeline only existed in Python, so a scene could only be produced by a
 * server (`serve.py`) and the viewer merely drew the result. An in-browser
 * model producer has no server to call, so scene assembly has to exist here.
 *
 * This module is deliberately *producer-agnostic*: it consumes hidden states
 * as plain Float32Arrays and knows nothing about how they were computed
 * (WebGPU, WASM, ONNX, or a fixture). Whatever runs the forward pass, the
 * scene is built the same way and stays comparable to a Python-built one.
 *
 * Numerical contract: the outputs match the Python modules to float64
 * agreement (`tests/test_scene_conformance.py` pins it). Where a convention
 * had to be chosen — PCA component signs, scipy's `reflect` edge handling —
 * this file follows scikit-learn / scipy exactly and says so at the site.
 *
 * UMD: `window.Scene` in the browser, `module.exports` under Node. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Scene = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // ---- linear algebra -------------------------------------------------

  /* Eigendecomposition of a symmetric matrix by the cyclic Jacobi method.
   * Returns eigenvalues in descending order with matching eigenvectors as
   * columns. Jacobi is chosen over a Lanczos/QR implementation because the
   * matrices here are small (the Gram matrix is n_states x n_states) and
   * Jacobi is unconditionally stable for symmetric input — accuracy matters
   * more than asymptotics at this size. */
  function jacobiEigh(Ain, n, sweeps = 100) {
    const A = Float64Array.from(Ain);
    const V = new Float64Array(n * n);
    for (let i = 0; i < n; i++) V[i * n + i] = 1;

    for (let sweep = 0; sweep < sweeps; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++)
        for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
      if (off <= 1e-30) break;

      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          const apq = A[p * n + q];
          if (Math.abs(apq) < 1e-300) continue;
          const app = A[p * n + p], aqq = A[q * n + q];
          const theta = (aqq - app) / (2 * apq);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) {
            const akp = A[k * n + p], akq = A[k * n + q];
            A[k * n + p] = c * akp - s * akq;
            A[k * n + q] = s * akp + c * akq;
          }
          for (let k = 0; k < n; k++) {
            const apk = A[p * n + k], aqk = A[q * n + k];
            A[p * n + k] = c * apk - s * aqk;
            A[q * n + k] = s * apk + c * aqk;
          }
          for (let k = 0; k < n; k++) {
            const vkp = V[k * n + p], vkq = V[k * n + q];
            V[k * n + p] = c * vkp - s * vkq;
            V[k * n + q] = s * vkp + c * vkq;
          }
        }
      }
    }

    const idx = Array.from({ length: n }, (_, i) => i)
      .sort((a, b) => A[b * n + b] - A[a * n + a]);
    const values = new Float64Array(n);
    const vectors = new Float64Array(n * n);
    idx.forEach((src, dst) => {
      values[dst] = A[src * n + src];
      for (let k = 0; k < n; k++) vectors[k * n + dst] = V[k * n + src];
    });
    return { values, vectors };
  }

  // ---- projection -----------------------------------------------------

  /* PCA over the union of runs — the port of `projection.project_joint`.
   *
   * Computed in the *dual* (Gram) form: the covariance is D x D and D is the
   * model's hidden width (up to several thousand), while the number of states
   * N is layers x tokens (hundreds). Eigendecomposing the N x N Gram matrix
   * is the same decomposition at a fraction of the cost, and the component
   * vectors are recovered as Xc^T W / s.
   *
   * `hidden` is a flat Float32Array/Array of N*D values, row-major by state. */
  function pca(hidden, n, d, nComponents = 2) {
    const X = hidden instanceof Float64Array ? hidden : Float64Array.from(hidden);
    const mean = new Float64Array(d);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < d; j++) mean[j] += X[i * d + j];
    for (let j = 0; j < d; j++) mean[j] /= n;

    const Xc = new Float64Array(n * d);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < d; j++) Xc[i * d + j] = X[i * d + j] - mean[j];

    // Gram matrix G = Xc Xc^T (symmetric, n x n).
    const G = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        let acc = 0;
        for (let k = 0; k < d; k++) acc += Xc[i * d + k] * Xc[j * d + k];
        G[i * n + j] = acc;
        G[j * n + i] = acc;
      }
    }

    const k = Math.min(nComponents, n, d);
    const { values, vectors } = jacobiEigh(G, n);

    let totalVar = 0;
    for (let i = 0; i < n; i++) totalVar += G[i * n + i];

    // Component vectors in hidden space: V = Xc^T U / s (D x k, column-major
    // by component). Needed for transform/inverse_transform and the PCA
    // residual that `projection_quality` reports — and, first, for the sign.
    const sv = new Float64Array(k);
    const components = new Float64Array(k * d);
    for (let c = 0; c < k; c++) {
      sv[c] = Math.sqrt(Math.max(values[c], 0));
      if (sv[c] <= 1e-12) continue;
      for (let j = 0; j < d; j++) {
        let acc = 0;
        for (let i = 0; i < n; i++) acc += Xc[i * d + j] * vectors[i * n + c];
        components[c * d + j] = acc / sv[c];
      }
    }

    /* Sign is fixed as `projection.PCAProjection` pins it, which is
     * scikit-learn >= 1.5's `svd_flip(u_based_decision=False)`: for each
     * component, the entry of V with the largest magnitude is made positive
     * (first index wins ties, matching np.argmax). This once followed U, as
     * scikit-learn < 1.5 did, and browser-built scenes came out mirrored
     * against Python-built ones on most inputs. Scores are U * s, flipped
     * with their component. */
    const scores = new Float64Array(n * k);
    for (let c = 0; c < k; c++) {
      let best = 0, bestAbs = -1;
      for (let j = 0; j < d; j++) {
        const a = Math.abs(components[c * d + j]);
        if (a > bestAbs) { bestAbs = a; best = j; }
      }
      const sign = Math.sign(components[c * d + best]) || 1;
      for (let j = 0; j < d; j++) components[c * d + j] *= sign;
      for (let i = 0; i < n; i++) scores[i * k + c] = vectors[i * n + c] * sv[c] * sign;
    }

    let kept = 0;
    for (let c = 0; c < k; c++) kept += Math.max(values[c], 0);

    return {
      coords: scores,
      nComponents: k,
      mean,
      components,
      explainedVariance: totalVar > 0 ? kept / totalVar : 0,
      transform(vec) {
        const out = new Float64Array(k);
        for (let c = 0; c < k; c++) {
          let acc = 0;
          for (let j = 0; j < d; j++) acc += (vec[j] - mean[j]) * components[c * d + j];
          out[c] = acc;
        }
        return out;
      },
    };
  }

  // ---- density --------------------------------------------------------

  /* Gaussian KDE with the Scott's-rule bandwidth `density.KDEEstimator`
   * uses: scale = mean per-axis population std, h = scale * n^(-1/(d+4)).
   * The kernel normalisation constant is included for fidelity with
   * scikit-learn even though `computeDensity` divides it out again. */
  function kdeBandwidth(pts, n, d) {
    const mean = new Float64Array(d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += pts[i * d + j];
    for (let j = 0; j < d; j++) mean[j] /= n;
    let scale = 0;
    for (let j = 0; j < d; j++) {
      let v = 0;
      for (let i = 0; i < n; i++) { const t = pts[i * d + j] - mean[j]; v += t * t; }
      scale += Math.sqrt(v / n);            // population std (numpy default)
    }
    scale = scale / d || 1.0;
    const h = n > 1 ? scale * Math.pow(n, -1 / (d + 4)) : 1.0;
    return Math.max(h, 1e-6);
  }

  function kdeEvaluate(query, m, pts, n, h) {
    const out = new Float64Array(m);
    const inv = 1 / (2 * h * h);
    const norm = 1 / (n * 2 * Math.PI * h * h);   // d = 2
    for (let q = 0; q < m; q++) {
      const qx = query[q * 2], qy = query[q * 2 + 1];
      let acc = 0;
      for (let i = 0; i < n; i++) {
        const dx = qx - pts[i * 2], dy = qy - pts[i * 2 + 1];
        acc += Math.exp(-(dx * dx + dy * dy) * inv);
      }
      out[q] = acc * norm;
    }
    return out;
  }

  /* Port of `density.compute_density` (KDE path): a grid spanning the point
   * cloud plus padding, normalised so the peak is 1. */
  function computeDensity(coords, n, opts = {}) {
    const gridSize = opts.gridSize ?? 64;
    const padding = opts.padding ?? 0.2;

    let loX = Infinity, loY = Infinity, hiX = -Infinity, hiY = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = coords[i * 2], y = coords[i * 2 + 1];
      if (x < loX) loX = x; if (x > hiX) hiX = x;
      if (y < loY) loY = y; if (y > hiY) hiY = y;
    }
    const spanX = Math.max(hiX - loX, 1e-6), spanY = Math.max(hiY - loY, 1e-6);
    loX -= padding * spanX; hiX += padding * spanX;
    loY -= padding * spanY; hiY += padding * spanY;

    const gridX = new Float64Array(gridSize), gridY = new Float64Array(gridSize);
    for (let i = 0; i < gridSize; i++) {
      const t = gridSize === 1 ? 0 : i / (gridSize - 1);
      gridX[i] = loX + t * (hiX - loX);
      gridY[i] = loY + t * (hiY - loY);
    }

    // np.meshgrid + ravel order: y outer, x inner.
    const gridPts = new Float64Array(gridSize * gridSize * 2);
    for (let r = 0; r < gridSize; r++)
      for (let c = 0; c < gridSize; c++) {
        gridPts[(r * gridSize + c) * 2] = gridX[c];
        gridPts[(r * gridSize + c) * 2 + 1] = gridY[r];
      }

    const h = kdeBandwidth(coords, n, 2);
    const density = kdeEvaluate(gridPts, gridSize * gridSize, coords, n, h);
    const pointDensity = kdeEvaluate(coords, n, coords, n, h);

    let peak = 0;
    for (let i = 0; i < density.length; i++) if (density[i] > peak) peak = density[i];
    if (peak > 0) {
      for (let i = 0; i < density.length; i++) density[i] /= peak;
      for (let i = 0; i < pointDensity.length; i++) pointDensity[i] /= peak;
    }
    return { gridX, gridY, density, pointDensity, gridSize };
  }

  // ---- terrain --------------------------------------------------------

  /* scipy.ndimage.gaussian_filter's edge handling: mode='reflect' is
   * half-sample symmetric — index -1 maps to 0 and index n maps to n-1. */
  function reflectIndex(i, n) {
    if (n === 1) return 0;
    const period = 2 * n;
    let k = ((i % period) + period) % period;
    if (k >= n) k = period - 1 - k;
    return k;
  }

  function gaussianFilter1d(src, rows, cols, sigma, axis) {
    const radius = Math.floor(4.0 * sigma + 0.5);   // scipy truncate=4.0
    const size = 2 * radius + 1;
    const w = new Float64Array(size);
    let sum = 0;
    for (let i = 0; i < size; i++) {
      const x = i - radius;
      w[i] = Math.exp(-0.5 * (x * x) / (sigma * sigma));
      sum += w[i];
    }
    for (let i = 0; i < size; i++) w[i] /= sum;

    const out = new Float64Array(src.length);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        let acc = 0;
        for (let k = 0; k < size; k++) {
          const off = k - radius;
          const rr = axis === 0 ? reflectIndex(r + off, rows) : r;
          const cc = axis === 1 ? reflectIndex(c + off, cols) : c;
          acc += src[rr * cols + cc] * w[k];
        }
        out[r * cols + c] = acc;
      }
    }
    return out;
  }

  /* Port of `terrain.height_map`. */
  function heightMap(density, gridSize, opts = {}) {
    const sigma = opts.smoothSigma ?? 1.5;
    const scale = opts.heightScale ?? 1.0;
    const invert = opts.invert ?? false;

    let z = gaussianFilter1d(density, gridSize, gridSize, sigma, 0);
    z = gaussianFilter1d(z, gridSize, gridSize, sigma, 1);

    if (invert) {
      let mx = -Infinity;
      for (let i = 0; i < z.length; i++) if (z[i] > mx) mx = z[i];
      for (let i = 0; i < z.length; i++) z[i] = mx - z[i];
    }
    let mn = Infinity;
    for (let i = 0; i < z.length; i++) if (z[i] < mn) mn = z[i];
    for (let i = 0; i < z.length; i++) z[i] -= mn;
    let peak = 0;
    for (let i = 0; i < z.length; i++) if (z[i] > peak) peak = z[i];
    if (peak > 0) for (let i = 0; i < z.length; i++) z[i] /= peak;
    for (let i = 0; i < z.length; i++) z[i] *= scale;
    return z;
  }

  /* Port of `terrain.surface_height` / `drape`: bilinear interpolation on the
   * (y, x) grid, out-of-range points clamped to the grid's minimum height —
   * the `fill_value` RegularGridInterpolator is constructed with. */
  function surfaceHeight(terrain, x, y) {
    const { gridX, gridY, z, gridSize } = terrain;
    if (x < gridX[0] || x > gridX[gridSize - 1] ||
        y < gridY[0] || y > gridY[gridSize - 1]) {
      let mn = Infinity;
      for (let i = 0; i < z.length; i++) if (z[i] < mn) mn = z[i];
      return mn;
    }
    const stepX = gridX[1] - gridX[0], stepY = gridY[1] - gridY[0];
    let cx = stepX > 0 ? Math.floor((x - gridX[0]) / stepX) : 0;
    let cy = stepY > 0 ? Math.floor((y - gridY[0]) / stepY) : 0;
    cx = Math.min(Math.max(cx, 0), gridSize - 2);
    cy = Math.min(Math.max(cy, 0), gridSize - 2);
    const tx = (x - gridX[cx]) / (gridX[cx + 1] - gridX[cx]);
    const ty = (y - gridY[cy]) / (gridY[cy + 1] - gridY[cy]);
    const z00 = z[cy * gridSize + cx], z01 = z[cy * gridSize + cx + 1];
    const z10 = z[(cy + 1) * gridSize + cx], z11 = z[(cy + 1) * gridSize + cx + 1];
    return z00 * (1 - tx) * (1 - ty) + z01 * tx * (1 - ty) +
           z10 * (1 - tx) * ty + z11 * tx * ty;
  }

  function drape(terrain, coords, n, lift = 0.04) {
    const out = new Float64Array(n * 3);
    for (let i = 0; i < n; i++) {
      const x = coords[i * 2], y = coords[i * 2 + 1];
      out[i * 3] = x;
      out[i * 3 + 1] = y;
      out[i * 3 + 2] = surfaceHeight(terrain, x, y) + lift;
    }
    return out;
  }

  // ---- readout --------------------------------------------------------

  /* Softmax entropy (nats) and top-k over a logit row — the readout the
   * inspector shows. Mirrors `models/synthetic._entropy_and_topk` and the
   * transformers capture path: shift by the max before exponentiating, and
   * treat p = 0 as contributing nothing to the entropy sum. */
  function entropyAndTopK(logits, k, vocab) {
    const n = logits.length;
    let mx = -Infinity;
    for (let i = 0; i < n; i++) if (logits[i] > mx) mx = logits[i];
    const p = new Float64Array(n);
    let sum = 0;
    for (let i = 0; i < n; i++) { p[i] = Math.exp(logits[i] - mx); sum += p[i]; }
    let entropy = 0;
    for (let i = 0; i < n; i++) {
      p[i] /= sum;
      if (p[i] > 0) entropy -= p[i] * Math.log(p[i]);
    }
    const order = Array.from({ length: n }, (_, i) => i)
      .sort((a, b) => p[b] - p[a] || a - b)
      .slice(0, k);
    return {
      entropy,
      topk: order.map((i) => [vocab ? vocab[i] : i, p[i]]),
    };
  }

  // ---- assembly -------------------------------------------------------

  /* Assemble a viewer-ready scene from one or more captured runs.
   *
   * `runs` is a list of { hidden: Float32Array (L*T*D), nLayers, nTokens,
   * dim, tokens, model }. All runs are projected into ONE shared space
   * (fitted on their union, as `projection.project_joint` does) so their
   * geometry is comparable, and the terrain is built from the union of all
   * states — a scene is one landscape, not several overlaid.
   *
   * The result mirrors the `.mtj` scene bundle's shape closely enough for the
   * viewer to draw it directly; `docs/mtj-format.md` remains the spec. */
  function buildScene(runs, opts = {}) {
    if (!runs.length) throw new Error("buildScene: no runs");
    const dim = runs[0].dim;
    let total = 0;
    for (const r of runs) {
      if (r.dim !== dim) throw new Error("buildScene: runs differ in hidden width");
      total += r.nLayers * r.nTokens;
    }

    const union = new Float64Array(total * dim);
    let off = 0;
    for (const r of runs) {
      union.set(r.hidden.subarray ? r.hidden.subarray(0, r.nLayers * r.nTokens * dim)
                                  : r.hidden, off);
      off += r.nLayers * r.nTokens * dim;
    }

    const projector = pca(union, total, dim, 2);
    const landscape = computeDensity(projector.coords, total, opts);
    const z = heightMap(landscape.density, landscape.gridSize, opts);
    const terrain = {
      gridX: landscape.gridX, gridY: landscape.gridY,
      z, gridSize: landscape.gridSize,
    };

    const out = [];
    let cursor = 0;
    for (const r of runs) {
      const count = r.nLayers * r.nTokens;
      const coords = projector.coords.subarray(cursor * 2, (cursor + count) * 2);
      out.push({
        model: r.model,
        tokens: r.tokens,
        nLayers: r.nLayers,
        nTokens: r.nTokens,
        coords,
        draped: drape(terrain, coords, count, opts.lift ?? 0.04),
      });
      cursor += count;
    }

    return {
      runs: out,
      terrain,
      landscape,
      explainedVariance: projector.explainedVariance,
      projector,
    };
  }

  return {
    jacobiEigh, pca, kdeBandwidth, computeDensity, heightMap,
    surfaceHeight, drape, entropyAndTopK, buildScene,
  };
});
