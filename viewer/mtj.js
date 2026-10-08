/* mtj.js — reader for Mottled `.mtj` files (see docs/mtj-format.md).
 * UMD: `window.MTJ` in the browser, `module.exports` under Node. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.MTJ = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const SUPPORTED_VERSION = 1;
  const HEADER_BYTES = 12;

  function decodeFloat16(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i];
      const s = (h & 0x8000) ? -1 : 1;
      const e = (h >> 10) & 0x1f;
      const m = h & 0x3ff;
      if (e === 0) out[i] = s * m * Math.pow(2, -24);
      else if (e === 31) out[i] = m ? NaN : s * Infinity;
      else out[i] = s * (1 + m / 1024) * Math.pow(2, e - 15);
    }
    return out;
  }

  // Known dtypes; anything else is ignored for forward compatibility.
  const DTYPES = {
    float32: { itemsize: 4, read: (dv, off, n) => new Float32Array(dv.buffer, dv.byteOffset + off, n) },
    int32: { itemsize: 4, read: (dv, off, n) => new Int32Array(dv.buffer, dv.byteOffset + off, n) },
    float16: {
      itemsize: 2,
      read: (dv, off, n) => decodeFloat16(new Uint16Array(dv.buffer, dv.byteOffset + off, n)),
    },
  };

  function toUint8(input) {
    if (input instanceof ArrayBuffer) return new Uint8Array(input);
    if (ArrayBuffer.isView(input)) {
      // Node Buffers can sit unaligned inside a pool; copy so typed-array
      // views at 16-byte-aligned offsets are always constructible.
      if (input.byteOffset % 16 !== 0) return new Uint8Array(input.slice ? input.slice() : input);
      return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    throw new TypeError("MTJ.parse expects an ArrayBuffer or typed array");
  }

  function parse(input) {
    const u8 = toUint8(input);
    if (u8.byteLength < HEADER_BYTES) throw new Error("not a .mtj file: too short for header");
    if (u8[0] !== 0x4d || u8[1] !== 0x54 || u8[2] !== 0x52 || u8[3] !== 0x4a)
      throw new Error('not a .mtj file: bad magic (expected "MTRJ")');
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    const version = dv.getUint32(4, true);
    if (version !== SUPPORTED_VERSION)
      throw new Error(`unsupported .mtj version ${version} (this reader supports ${SUPPORTED_VERSION})`);
    const manifestLen = dv.getUint32(8, true);
    if (HEADER_BYTES + manifestLen > u8.byteLength)
      throw new Error("corrupt .mtj file: manifest extends past end of file");

    let manifest;
    try {
      manifest = JSON.parse(new TextDecoder("utf-8").decode(u8.subarray(HEADER_BYTES, HEADER_BYTES + manifestLen)));
    } catch (e) {
      throw new Error("corrupt .mtj file: manifest is not valid JSON (" + e.message + ")");
    }

    const blobStart = HEADER_BYTES + manifestLen; // spec: 12+M is 16-byte aligned
    const blobLen = u8.byteLength - blobStart;
    const arrays = {};
    for (const [name, ref] of Object.entries(manifest.arrays || {})) {
      const dt = DTYPES[ref && ref.dtype];
      if (!dt || !Array.isArray(ref.shape)) continue; // unknown dtype/shape: ignore
      const count = ref.shape.reduce((a, b) => a * b, 1);
      const bytes = count * dt.itemsize;
      if (typeof ref.offset !== "number" || ref.offset < 0 || ref.offset + bytes > blobLen)
        throw new Error(`corrupt .mtj file: array "${name}" is out of range`);
      if (typeof ref.length === "number" && ref.length !== bytes)
        throw new Error(`corrupt .mtj file: array "${name}" length ${ref.length} != shape*itemsize ${bytes}`);
      arrays[name] = {
        dtype: ref.dtype,
        shape: ref.shape.slice(),
        offset: ref.offset,
        data: dt.read(dv, blobStart + ref.offset, count),
      };
    }
    return { manifest, arrays };
  }

  function loadScene(input) {
    const { manifest, arrays } = parse(input);
    if (manifest.kind === "trajectory") {
      const err = new Error(
        'this is a full "trajectory" capture, not a viewer scene — ' +
        're-export it with kind "scene" (analysis baked in) for the web viewer'
      );
      err.kind = "trajectory";
      throw err;
    }
    if (manifest.kind !== "scene")
      throw new Error(`unsupported .mtj kind "${manifest.kind}" (expected "scene")`);

    const resolve = (name, what) => {
      const a = arrays[name];
      if (!a) throw new Error(`corrupt scene: ${what} references missing array "${name}"`);
      return a;
    };

    // Optional per-run SAE feature layer (writers >= scene-v5): dominant
    // feature id/activation per (layer, token) plus the dictionary's median
    // relative reconstruction error per layer. Strictly additive — a missing,
    // malformed, or dangling-ref record reads as "no features" so it can
    // never break a scene that renders without it (same contract as the
    // `generation` record).
    const resolveFeatures = (f) => {
      if (!f || typeof f !== "object" || Array.isArray(f)) return null;
      const reconError = typeof f.recon_error === "string" ? arrays[f.recon_error] : null;
      const topId = typeof f.top_id === "string" ? arrays[f.top_id] : null;
      const topAct = typeof f.top_act === "string" ? arrays[f.top_act] : null;
      if (!reconError || !topId || !topAct) return null;
      return {
        source: f.source != null ? String(f.source) : null,
        hook: f.hook != null ? String(f.hook) : null,
        best_layer: typeof f.best_layer === "number" ? f.best_layer : null,
        recon_error: reconError,  // (L,) float32
        top_id: topId,            // (L, T) int32
        top_act: topAct,          // (L, T) float32
      };
    };

    // Optional per-run inspector layer (writers >= scene-v6): readouts a
    // viewer cannot recompute from a scene alone, because their inputs (the
    // (V, D) embedding matrix, the 2 x (L-1) x T x D residual components) are
    // far larger than the scene itself — the nearest vocabulary tokens to
    // each hidden state, and the attention/MLP split of every block's write
    // to the residual stream. Members are independent: a capture without
    // residual components carries no `component_shares`, a producer with no
    // embedding matrix carries no `idx`/`sim`. A missing, non-string, or
    // dangling ref drops just that member; when nothing resolves the record
    // reads null (same contract as `features` and `generation`).
    const resolveInspector = (e) => {
      if (!e || typeof e !== "object" || Array.isArray(e)) return null;
      const idx = typeof e.idx === "string" ? arrays[e.idx] || null : null;
      const sim = typeof e.sim === "string" ? arrays[e.sim] || null : null;
      const shares = typeof e.component_shares === "string"
        ? arrays[e.component_shares] || null : null;
      if (!idx && !sim && !shares) return null;
      return {
        // compact string table the neighbor indices point into
        tokens: Array.isArray(e.tokens) ? e.tokens.map(String) : [],
        idx,                        // (L, T, k) int32, nearest first
        sim,                        // (L, T, k) float32 cosine
        component_shares: shares,   // (L-1, T, 2) float32, blocks on axis 0
      };
    };

    const t = manifest.terrain || {};
    const terrain = {
      x: resolve(t.x, "terrain.x"),
      y: resolve(t.y, "terrain.y"),
      z: resolve(t.z, "terrain.z"),
      // optional uncertainty layers (writers >= scene-v3)
      density: t.density ? resolve(t.density, "terrain.density") : null,
      se: t.se ? resolve(t.se, "terrain.se") : null,
    };
    if (terrain.z.shape.length !== 2 ||
        terrain.z.shape[0] !== terrain.y.shape[0] || terrain.z.shape[1] !== terrain.x.shape[0])
      throw new Error("corrupt scene: terrain z shape does not match x/y axes");

    const runs = (manifest.runs || []).map((r, i) => {
      const points = resolve(r.points, `run ${i}`);
      if (points.shape.length !== 3 || points.shape[2] !== 3)
        throw new Error(`corrupt scene: run ${i} points must have shape (N, L, 3)`);
      return {
        label: r.label != null ? String(r.label) : String.fromCharCode(65 + i),
        prompt: r.prompt || "",
        tokens: r.tokens || [],
        trajectoryLabels: r.trajectory_labels || r.tokens || [],
        points,
        entropy: r.entropy ? resolve(r.entropy, `run ${i} entropy`) : null,
        quality: r.quality ? resolve(r.quality, `run ${i} quality`) : null,
        attention: r.attention ? resolve(r.attention, `run ${i} attention`) : null,
        topk: r.topk || null,
        // optional decode record (writers >= scene-v4): prompt/continuation
        // boundary plus per-step token, id, p, entropy
        generation: (r.generation && typeof r.generation === "object" &&
                     !Array.isArray(r.generation)) ? r.generation : null,
        // optional SAE feature record (writers >= scene-v5)
        features: resolveFeatures(r.features),
        // optional inspector record (writers >= scene-v6): nearest vocabulary
        // tokens per state + attn/MLP share of each block's residual write
        inspector: resolveInspector(r.inspector),
        // which model produced this run. Scene-level `meta` describes run 0,
        // which is simply wrong for a scene whose runs are different models.
        model: typeof r.model === "string" ? r.model : null,
      };
    });
    if (!runs.length) throw new Error("corrupt scene: no runs");

    return { manifest, arrays, meta: manifest.meta || {}, terrain, runs,
             comparisons: manifest.comparisons || [],
             blast: resolveBlast(manifest.blast, manifest.meta, arrays, runs[0]) };
  }

  // Optional `blast` record (blast.py): one pellet per prompt, with every
  // layout the writer built. The scene's run 0 is the pellets drawn at the
  // first layout, so a viewer without this code still shows them; this
  // record adds the other layouts and what each one's axes mean. A layout
  // whose positions are missing or the wrong shape is dropped; when none
  // survives, or the pellets do not match run 0, the record reads null
  // (same contract as `features` and `inspector`).
  function resolveBlast(b, meta, arrays, run0) {
    if (!b || typeof b !== "object" || Array.isArray(b)) return null;
    const n = run0.points.shape[0], L = run0.points.shape[1];
    const pellets = meta && Array.isArray(meta.pellets) ? meta.pellets : null;
    if (!pellets || pellets.length !== n) return null;
    const arr = (ref, shape) => {
      const a = typeof ref === "string" ? arrays[ref] : null;
      if (!a || a.shape.length !== shape.length) return null;
      return a.shape.every((s, i) => shape[i] == null || s === shape[i]) ? a : null;
    };
    const strings = (x) => (Array.isArray(x) ? x.map(String) : []);
    const layouts = (Array.isArray(b.layouts) ? b.layouts : []).map((lay) => {
      if (!lay || typeof lay !== "object") return null;
      const positions = arr(lay.positions, [n, L, 2]);
      if (!positions) return null;
      const extra = {};
      const refs = lay.arrays && typeof lay.arrays === "object" ? lay.arrays : {};
      for (const k of Object.keys(refs)) {
        const a = typeof refs[k] === "string" ? arrays[refs[k]] : null;
        if (a) extra[k] = a;
      }
      return {
        name: lay.name != null ? String(lay.name) : String(lay.method || "layout"),
        method: lay.method != null ? String(lay.method) : null,
        driver: typeof lay.driver === "string" ? lay.driver : null,
        positions,                              // (N, L, 2) float32
        quality: arr(lay.quality, [L, n]),      // (L, N) or null
        exact: strings(lay.exact), fitted: strings(lay.fitted),
        projected: strings(lay.projected),
        arrays: extra,                          // monitor: auroc, null05, null95 (L,), labelled (N,)
        params: lay.params && typeof lay.params === "object" ? lay.params : {},
      };
    }).filter(Boolean);
    if (!layouts.length) return null;
    return {
      schema: typeof b.schema === "string" ? b.schema : null,
      pellets: pellets.map((p) => ({
        id: p && p.id != null ? String(p.id) : "",
        text: p && p.text != null ? String(p.text) : "",
        labels: p && p.labels && typeof p.labels === "object" ? p.labels : {},
      })),
      layouts,
      skipped: strings(b.skipped),
      range: arr(b.range, [n, L]),             // (N, L) exact distance from origin
      spread: arr(b.spread, [L]),              // (L,) RMS of range
      norm: arr(b.norm, [L]),                  // (L,) the unit: mean state norm
    };
  }

  // ------------------------------------------------ generation (decode) helpers
  // A run's optional `generation` record marks where the prompt ends and the
  // decoded continuation begins (token indices >= prompt_tokens). Everything
  // here is null-safe: an absent or malformed record reads as "no generation"
  // so pre-decode scenes behave exactly as before.

  function isGeneratedToken(generation, index) {
    return !!generation && typeof generation === "object" &&
           typeof generation.prompt_tokens === "number" &&
           typeof index === "number" && index >= generation.prompt_tokens;
  }

  function generationStep(generation, index) {
    // per-step decode record ({token, id, p, entropy}) for token `index`
    if (!isGeneratedToken(generation, index) || !Array.isArray(generation.steps)) return null;
    const step = generation.steps[index - generation.prompt_tokens];
    return step && typeof step === "object" ? step : null;
  }

  function continuationText(generation, backend) {
    if (!generation || !Array.isArray(generation.steps)) return "";
    const toks = generation.steps.map((s) => (s && s.token != null ? String(s.token) : ""));
    // synthetic tokens are bare words; BPE/SentencePiece pieces carry their
    // own leading spaces
    return toks.join(backend === "synthetic" ? " " : "");
  }

  function decodeSummary(generation) {
    // "greedy · +3 tokens" / "sample · T=0.8 · +5 tokens" (continuation text
    // is rendered separately so callers can style it as data)
    if (!generation || typeof generation !== "object") return "";
    const mode = generation.mode != null ? String(generation.mode) : "?";
    const n = typeof generation.new_tokens === "number" ? generation.new_tokens
            : Array.isArray(generation.steps) ? generation.steps.length : 0;
    let s = mode;
    if (mode === "sample" && typeof generation.temperature === "number")
      s += ` · T=${generation.temperature.toFixed(1)}`;
    return `${s} · +${n} token${n === 1 ? "" : "s"}`;
  }

  function generationBoundary(run) {
    // First generated trajectory index for a loaded run, or -1 when the run
    // has no usable decode record or trajectories aren't 1:1 with tokens
    // (single-token exports): per-trajectory decode styling only makes sense
    // when trajectory i reads out token i.
    if (!run || !run.generation) return -1;
    const g = run.generation;
    if (typeof g.prompt_tokens !== "number" || g.prompt_tokens < 0) return -1;
    const n = run.points && Array.isArray(run.points.shape) ? run.points.shape[0] : -1;
    const tokens = run.tokens || [], labels = run.trajectoryLabels || [];
    if (n <= 0 || n !== tokens.length || labels.length !== n) return -1;
    for (let i = 0; i < n; i++)
      if (String(labels[i]) !== String(tokens[i])) return -1;
    return g.prompt_tokens < n ? g.prompt_tokens : -1;
  }

  // ------------------------------------------------ SAE feature helpers
  // A run's optional `features` record carries the dominant SAE feature per
  // (layer, token) and the dictionary's per-layer reconstruction error.
  // Everything here is null-safe: an absent or malformed record reads as
  // "no features" so pre-feature scenes behave exactly as before.

  function featureAt(features, layer, tokenIdx) {
    // dominant feature {id, act} at (layer, tokenIdx), or null when the
    // record is absent or the indices fall outside the arrays' shapes
    if (!features || typeof features !== "object") return null;
    const ids = features.top_id, acts = features.top_act;
    if (!ids || !acts || !Array.isArray(ids.shape) || ids.shape.length !== 2) return null;
    if (!Number.isInteger(layer) || !Number.isInteger(tokenIdx)) return null;
    const [L, T] = ids.shape;
    if (layer < 0 || layer >= L || tokenIdx < 0 || tokenIdx >= T) return null;
    const o = layer * T + tokenIdx;
    if (!ids.data || !acts.data || o >= ids.data.length || o >= acts.data.length) return null;
    return { id: ids.data[o], act: acts.data[o] };
  }

  function featureFitSummary(features) {
    // "features: jbloom/GPT2-Small-SAEs-Reformatted · best fit layer 8 ·
    // 22% err" — appends " · EXTRAPOLATION" when even the best layer's
    // dictionary misses more than half the norm (features are guesswork)
    if (!features || typeof features !== "object") return "";
    const re = features.recon_error;
    if (!re || !re.data || !re.data.length) return "";
    let best = typeof features.best_layer === "number" ? features.best_layer : -1;
    if (!Number.isInteger(best) || best < 0 || best >= re.data.length) {
      best = 0; // record omitted/out of range: recompute the argmin
      for (let l = 1; l < re.data.length; l++) if (re.data[l] < re.data[best]) best = l;
    }
    const err = re.data[best];
    let s = `features: ${features.source != null ? features.source : "?"}` +
            ` · best fit layer ${best} · ${(err * 100).toFixed(0)}% err`;
    if (err > 0.5) s += " · EXTRAPOLATION";
    return s;
  }

  // ------------------------------------------------ inspector helpers
  // A run's optional `inspector` record carries the nearest vocabulary tokens
  // to every hidden state and the attention/MLP split of each block's write to
  // the residual stream. Everything here is null-safe: an absent, partial, or
  // malformed record reads as "not available" so scenes without it — and
  // scenes carrying only one of the two layers — behave exactly as before.

  function neighborsAt(inspector, layer, tokenIdx, limit) {
    // the nearest vocabulary tokens to the state at (layer, tokenIdx) as
    // [{token, sim}, …] nearest-first (at most `limit`), or [] when the
    // record is absent or the indices fall outside the arrays' shapes
    if (!inspector || typeof inspector !== "object") return [];
    const idx = inspector.idx, sim = inspector.sim;
    // the pair is only readable together: ids without similarities (or the
    // reverse) is not a neighbor list
    if (!idx || !sim || !Array.isArray(idx.shape) || idx.shape.length !== 3) return [];
    if (!idx.data || !sim.data) return [];
    if (!Number.isInteger(layer) || !Number.isInteger(tokenIdx)) return [];
    const [L, T, K] = idx.shape;
    if (layer < 0 || layer >= L || tokenIdx < 0 || tokenIdx >= T) return [];
    const tokens = Array.isArray(inspector.tokens) ? inspector.tokens : [];
    const n = Number.isInteger(limit) ? Math.min(Math.max(limit, 0), K) : K;
    const base = (layer * T + tokenIdx) * K;
    const out = [];
    for (let j = 0; j < n; j++) {
      const o = base + j;
      if (o >= idx.data.length || o >= sim.data.length) break;
      const ti = idx.data[o];
      if (!(ti >= 0) || ti >= tokens.length) continue; // index off the table
      out.push({ token: String(tokens[ti]), sim: sim.data[o] });
    }
    return out;
  }

  function componentShareAt(inspector, layer, tokenIdx) {
    // {attn, mlp} share of the block write that produced the state at
    // (layer, tokenIdx), or null. The array's first axis is *blocks*: block b
    // writes the state at layer b+1, so layer l reads index l-1 and layer 0 —
    // the embedding stream, written by no block — has no share at all.
    if (!inspector || typeof inspector !== "object") return null;
    const cs = inspector.component_shares;
    if (!cs || !cs.data || !Array.isArray(cs.shape) || cs.shape.length !== 3) return null;
    if (!Number.isInteger(layer) || !Number.isInteger(tokenIdx)) return null;
    const [B, T, C] = cs.shape;
    if (C < 2) return null;
    const block = layer - 1;
    if (block < 0 || block >= B || tokenIdx < 0 || tokenIdx >= T) return null;
    const o = (block * T + tokenIdx) * C;
    if (o + 1 >= cs.data.length) return null;
    return { attn: cs.data[o], mlp: cs.data[o + 1] };
  }

  return { parse, loadScene, decodeFloat16, SUPPORTED_VERSION,
           isGeneratedToken, generationStep, continuationText, decodeSummary,
           generationBoundary, featureAt, featureFitSummary,
           neighborsAt, componentShareAt };
});
