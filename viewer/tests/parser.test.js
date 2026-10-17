"use strict";
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const MTJ = require("../mtj.js");

const SAMPLES = path.resolve(__dirname, "..", "samples");
const read = (name) => fs.readFileSync(path.join(SAMPLES, name));

// Build a synthetic .mtj in memory (mirrors the writer layout in the spec).
function buildMtj({ manifest, blob = new Uint8Array(0), version = 1, magic = "MTRJ" }) {
  let json = Buffer.from(JSON.stringify(manifest), "utf-8");
  const pad = (16 - ((12 + json.length) % 16)) % 16;
  json = Buffer.concat([json, Buffer.alloc(pad, 0x20)]);
  const header = Buffer.alloc(12);
  header.write(magic, 0, "ascii");
  header.writeUInt32LE(version, 4);
  header.writeUInt32LE(json.length, 8);
  return Buffer.concat([header, json, Buffer.from(blob.buffer || blob)]);
}

function f32blob(values) {
  const blob = new Uint8Array(Math.ceil((values.length * 4) / 16) * 16);
  new Uint8Array(blob.buffer, 0, values.length * 4).set(new Uint8Array(new Float32Array(values).buffer));
  return blob;
}

test("rejects bad magic", () => {
  const buf = Buffer.from(read("single.mtj"));
  buf[0] = 0x58; // 'X'
  assert.throws(() => MTJ.parse(buf), /magic/i);
  assert.throws(() => MTJ.parse(buildMtj({ manifest: {}, magic: "GLTF" })), /magic/i);
});

test("rejects unsupported version", () => {
  const buf = buildMtj({ manifest: { format: "mottled-trajectory", version: 2, kind: "scene" }, version: 2 });
  assert.throws(() => MTJ.parse(buf), /version 2/i);
});

test("rejects truncated files and out-of-range arrays", () => {
  assert.throws(() => MTJ.parse(new Uint8Array(4)), /too short/i);
  const bad = buildMtj({
    manifest: { arrays: { a: { dtype: "float32", shape: [1000], offset: 0, length: 4000 } } },
    blob: f32blob([1, 2, 3]),
  });
  assert.throws(() => MTJ.parse(bad), /out of range/i);
});

test("parses single.mtj manifest and arrays", () => {
  const { manifest, arrays } = MTJ.parse(read("single.mtj"));
  assert.strictEqual(manifest.format, "mottled-trajectory");
  assert.strictEqual(manifest.version, 1);
  assert.strictEqual(manifest.kind, "scene");
  assert.strictEqual(manifest.runs.length, 1);

  for (const [name, ref] of Object.entries(manifest.arrays)) {
    assert.strictEqual(ref.offset % 16, 0, `${name} offset 16-byte aligned`);
    assert.ok(arrays[name], `${name} resolved`);
    const count = ref.shape.reduce((a, b) => a * b, 1);
    assert.strictEqual(arrays[name].data.length, count, `${name} element count`);
    assert.deepStrictEqual(arrays[name].shape, ref.shape);
    assert.strictEqual(arrays[name].dtype, ref.dtype);
  }
  assert.ok(arrays["terrain.z"].data instanceof Float32Array);
  assert.deepStrictEqual(arrays["run0.points"].shape, [5, 13, 3]);
  assert.ok(arrays["run0.points"].data.every(Number.isFinite));
});

test("parses scene-abc.mtj manifest and arrays", () => {
  const { manifest, arrays } = MTJ.parse(read("scene-abc.mtj"));
  assert.strictEqual(manifest.kind, "scene");
  assert.strictEqual(manifest.runs.length, 3);
  for (let i = 0; i < 3; i++) {
    assert.ok(arrays[`run${i}.points`], `run${i}.points present`);
    assert.deepStrictEqual(arrays[`run${i}.attention`].shape, [12, 5, 5]);
    assert.strictEqual(arrays[`run${i}.entropy`].dtype, "float32");
  }
});

test("loadScene resolves terrain, runs, topk and comparisons", () => {
  const scene = MTJ.loadScene(read("scene-abc.mtj"));
  assert.strictEqual(scene.terrain.x.shape[0], 64);
  assert.strictEqual(scene.terrain.y.shape[0], 64);
  assert.deepStrictEqual(scene.terrain.z.shape, [64, 64]);

  assert.deepStrictEqual(scene.runs.map((r) => r.label), ["A", "B", "C"]);
  for (const run of scene.runs) {
    assert.strictEqual(run.tokens.length, 5);
    assert.strictEqual(run.trajectoryLabels.length, 5);
    assert.deepStrictEqual(run.points.shape, [5, 13, 3]);
    assert.ok(run.points.data instanceof Float32Array);
    assert.deepStrictEqual(run.entropy.shape, [13, 5]);
    assert.deepStrictEqual(run.attention.shape, [12, 5, 5]);
    assert.strictEqual(run.topk.length, 13); // [L][T][k]
    assert.strictEqual(run.topk[0].length, 5);
    assert.ok(Array.isArray(run.topk[0][0][0] !== undefined ? run.topk[0][0] : null));
  }
  assert.strictEqual(scene.comparisons.length, 2);
  const b = scene.comparisons[0];
  assert.strictEqual(b.label, "B");
  for (const k of ["hausdorff", "dtw_normalized"]) assert.strictEqual(typeof b[k], "number");
  assert.strictEqual(typeof b.shared_tokens, "number");
});

test("loadScene rejects kind trajectory with a friendly error", () => {
  const buf = buildMtj({
    manifest: {
      format: "mottled-trajectory", version: 1, kind: "trajectory",
      tokens: ["a"], arrays: { hidden: { dtype: "float32", shape: [1, 1, 2], offset: 0, length: 8 } },
    },
    blob: f32blob([0, 1]),
  });
  assert.doesNotThrow(() => MTJ.parse(buf)); // parse is kind-agnostic
  assert.throws(() => MTJ.loadScene(buf), (e) => e.kind === "trajectory" && /scene/.test(e.message));
});

test("forward compatibility: unknown fields and dtypes are ignored", () => {
  const buf = buildMtj({
    manifest: {
      format: "mottled-trajectory", version: 1, kind: "scene",
      future_field: { nested: true },
      terrain: { x: "x", y: "y", z: "z" },
      runs: [{ label: "A", prompt: "p", tokens: ["a"], trajectory_labels: ["a"],
               points: "pts", future_run_field: 42 }],
      arrays: {
        x: { dtype: "float32", shape: [2], offset: 0, length: 8 },
        y: { dtype: "float32", shape: [2], offset: 16, length: 8 },
        z: { dtype: "float32", shape: [2, 2], offset: 32, length: 16 },
        pts: { dtype: "float32", shape: [1, 2, 3], offset: 48, length: 24 },
        exotic: { dtype: "float64", shape: [2], offset: 80, length: 16 }, // unknown dtype
      },
    },
    blob: (() => {
      const b = new Uint8Array(96);
      const put = (off, vals) => new Uint8Array(b.buffer, off, vals.length * 4)
        .set(new Uint8Array(new Float32Array(vals).buffer));
      put(0, [0, 1]); put(16, [0, 1]); put(32, [0, 0.5, 0.5, 1]); put(48, [0, 0, 0, 1, 1, 1]);
      return b;
    })(),
  });
  const { manifest, arrays } = MTJ.parse(buf);
  assert.deepStrictEqual(manifest.future_field, { nested: true });
  assert.strictEqual(arrays.exotic, undefined, "unknown dtype skipped");
  assert.ok(arrays.pts);

  const scene = MTJ.loadScene(buf); // unknown fields must not break scene resolution
  assert.strictEqual(scene.runs.length, 1);
  assert.deepStrictEqual(scene.runs[0].points.shape, [1, 2, 3]);
  assert.strictEqual(scene.runs[0].entropy, null);
});

test("loadScene resolves optional uncertainty layers (terrain se/density, run quality)", () => {
  const buf = buildMtj({
    manifest: {
      format: "mottled-trajectory", version: 1, kind: "scene",
      terrain: { x: "x", y: "y", z: "z", density: "d", se: "se" },
      runs: [{ label: "A", prompt: "p", tokens: ["a", "b"], trajectory_labels: ["a", "b"],
               points: "pts", quality: "q" }],
      arrays: {
        x: { dtype: "float32", shape: [2], offset: 0, length: 8 },
        y: { dtype: "float32", shape: [2], offset: 16, length: 8 },
        z: { dtype: "float32", shape: [2, 2], offset: 32, length: 16 },
        d: { dtype: "float32", shape: [2, 2], offset: 48, length: 16 },
        se: { dtype: "float32", shape: [2, 2], offset: 64, length: 16 },
        pts: { dtype: "float32", shape: [2, 1, 3], offset: 80, length: 24 },
        q: { dtype: "float32", shape: [1, 2], offset: 112, length: 8 },
      },
    },
    blob: (() => {
      const b = new Uint8Array(128);
      const put = (off, vals) => new Uint8Array(b.buffer, off, vals.length * 4)
        .set(new Uint8Array(new Float32Array(vals).buffer));
      put(0, [0, 1]); put(16, [0, 1]); put(32, [0, 0.5, 0.5, 1]);
      put(48, [1, 0.5, 0.5, 0]); put(64, [0.1, 0.2, 0.05, 0.3]);
      put(80, [0, 0, 0, 1, 1, 1]); put(112, [0.75, 0.25]);
      return b;
    })(),
  });
  const scene = MTJ.loadScene(buf);
  assert.deepStrictEqual(scene.terrain.se.shape, [2, 2]);
  assert.deepStrictEqual(scene.terrain.density.shape, [2, 2]);
  assert.ok(scene.terrain.se.data instanceof Float32Array);
  assert.deepStrictEqual(scene.runs[0].quality.shape, [1, 2]);
  assert.deepStrictEqual(Array.from(scene.runs[0].quality.data), [0.75, 0.25]);
});

test("loadScene resolves layer_fidelity, and reads a malformed one as null", () => {
  const scene = (lf) => MTJ.loadScene(buildMtj({
    manifest: {
      format: "mottled-trajectory", version: 1, kind: "scene",
      terrain: { x: "x", y: "y", z: "z" },
      runs: [{ label: "A", tokens: ["a", "b"], points: "pts", quality: "q",
               layer_fidelity: lf }],
      arrays: {
        x: { dtype: "float32", shape: [2], offset: 0, length: 8 },
        y: { dtype: "float32", shape: [2], offset: 16, length: 8 },
        z: { dtype: "float32", shape: [2, 2], offset: 32, length: 16 },
        pts: { dtype: "float32", shape: [2, 1, 3], offset: 48, length: 24 },
        q: { dtype: "float32", shape: [1, 2], offset: 80, length: 8 },
        lf: { dtype: "float32", shape: [1, 2], offset: 96, length: 8 },
      },
    },
    blob: (() => {               // float32 index = byte offset / 4
      const v = new Array(28).fill(0);
      v.splice(0, 2, 0, 1); v.splice(4, 2, 0, 1); v.splice(8, 4, 0, 0.5, 0.5, 1);
      v.splice(12, 6, 0, 0, 0, 1, 1, 1); v.splice(20, 2, 1, 1); v.splice(24, 2, 0.2, 0.6);
      return f32blob(v);
    })(),
  })).runs[0].layerFidelity;

  const lf = scene({ preservation: "lf", k: 1, chance: 1 / 3 });
  assert.deepStrictEqual(lf.preservation.shape, [1, 2]);
  assert.ok(Math.abs(lf.preservation.data[0] - 0.2) < 1e-6);
  assert.strictEqual(lf.k, 1);
  assert.strictEqual(lf.chance, 1 / 3);
  assert.strictEqual(scene({ preservation: "pts" }), null);      // not (L, T)
  assert.strictEqual(scene({ preservation: "nope" }), null);     // dangling
  assert.strictEqual(scene(undefined), null);                    // older scene
  assert.strictEqual(scene({ preservation: "lf", chance: "x" }).chance, null);
});

test("loadScene tolerates scenes with no uncertainty layers (older writers)", () => {
  // A pre-v3 scene: terrain has only x/y/z, runs have no quality array.
  const buf = buildMtj({
    manifest: {
      format: "mottled-trajectory", version: 1, kind: "scene",
      terrain: { x: "x", y: "y", z: "z" },
      runs: [{ label: "A", prompt: "p", tokens: ["a"], trajectory_labels: ["a"], points: "pts" }],
      arrays: {
        x: { dtype: "float32", shape: [2], offset: 0, length: 8 },
        y: { dtype: "float32", shape: [2], offset: 16, length: 8 },
        z: { dtype: "float32", shape: [2, 2], offset: 32, length: 16 },
        pts: { dtype: "float32", shape: [1, 2, 3], offset: 48, length: 24 },
      },
    },
    blob: (() => {
      const b = new Uint8Array(80);
      const put = (off, vals) => new Uint8Array(b.buffer, off, vals.length * 4)
        .set(new Uint8Array(new Float32Array(vals).buffer));
      put(0, [0, 1]); put(16, [0, 1]); put(32, [0, 0.5, 0.5, 1]); put(48, [0, 0, 0, 1, 1, 1]);
      return b;
    })(),
  });
  const scene = MTJ.loadScene(buf);
  assert.strictEqual(scene.terrain.se, null);
  assert.strictEqual(scene.terrain.density, null);
  assert.strictEqual(scene.runs[0].quality, null);
});

test("float16 arrays decode to Float32Array", () => {
  // 1.0 = 0x3C00, -2.0 = 0xC000, 0.5 = 0x3800
  const blob = new Uint8Array(16);
  new Uint16Array(blob.buffer, 0, 3).set([0x3c00, 0xc000, 0x3800]);
  const buf = buildMtj({
    manifest: { format: "mottled-trajectory", version: 1, kind: "scene",
                arrays: { h: { dtype: "float16", shape: [3], offset: 0, length: 6 } } },
    blob,
  });
  const { arrays } = MTJ.parse(buf);
  assert.ok(arrays.h.data instanceof Float32Array);
  assert.deepStrictEqual(Array.from(arrays.h.data), [1, -2, 0.5]);
});
