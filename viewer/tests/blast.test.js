"use strict";
const test = require("node:test");
const assert = require("node:assert");

const MTJ = require("../mtj.js");

// Same in-memory writer as features.test.js: header, padded manifest, blob.
function buildMtj(manifest, blob) {
  let json = Buffer.from(JSON.stringify(manifest), "utf-8");
  const pad = (16 - ((12 + json.length) % 16)) % 16;
  json = Buffer.concat([json, Buffer.alloc(pad, 0x20)]);
  const header = Buffer.alloc(12);
  header.write("MTRJ", 0, "ascii");
  header.writeUInt32LE(1, 4);
  header.writeUInt32LE(json.length, 8);
  return Buffer.concat([header, json, Buffer.from(blob.buffer)]);
}

// A blast scene: 2 pellets x 2 layers. The run is the pellets at the first
// layout (what an older viewer draws); the record adds the layouts.
//   pos  = positions (2, 2, 2)  float32   q = quality (2, 2)
//   au   = auroc (2,)           sp = spread (2,)   rg = range (2, 2)
function blastScene(blast, pellets) {
  const blob = new Uint8Array(192);
  const put = (off, vals) => new Uint8Array(blob.buffer, off, vals.length * 4)
    .set(new Uint8Array(new Float32Array(vals).buffer));
  put(0, [0, 1]); put(16, [0, 1]); put(32, [0, 0, 0, 0]);
  put(48, [0, 0, 0, 0.1, 0.2, 0, 0, 0, 0, -0.1, 0.3, 0]);
  put(96, [0, 0, 0.1, 0.2, 0, 0, -0.1, 0.3]);
  put(128, [1, 1, 0.5, 0.5]); put(144, [NaN, 0.9]); put(160, [0, 0.2]);
  put(176, [0, 0.22, 0, 0.32]);
  const f = (shape, offset) => ({ dtype: "float32", shape, offset,
                                  length: 4 * shape.reduce((a, b) => a * b, 1) });
  return buildMtj({
    format: "mottled-trajectory", version: 1, kind: "scene",
    meta: { axis: "pellets", pellets },
    terrain: { x: "x", y: "y", z: "z" },
    runs: [{ label: "A", points: "pts", tokens: ["p0", "p1"] }],
    blast,
    arrays: { x: f([2], 0), y: f([2], 16), z: f([2, 2], 32), pts: f([2, 2, 3], 48),
              pos: f([2, 2, 2], 96), q: f([2, 2], 128), au: f([2], 144),
              sp: f([2], 160), rg: f([2, 2], 176) },
  }, blob);
}

const PELLETS = [{ id: "p0", text: "first", labels: { c: 1 } },
                 { id: "p1", text: "second", labels: { c: 0 } }];
const RECORD = {
  schema: "mottled-blast/1", range: "rg", spread: "sp", norm: "sp", skipped: [],
  layouts: [{ name: "monitor · c", method: "monitor", driver: "c", positions: "pos",
              quality: "q", arrays: { auroc: "au" }, exact: ["origin"],
              fitted: ["x: cross-fitted"], projected: ["y"], params: { folds: 2 } }],
};

test("a blast record resolves its layouts, arrays and pellets", () => {
  const b = MTJ.loadScene(blastScene(RECORD, PELLETS)).blast;
  assert.strictEqual(b.schema, "mottled-blast/1");
  assert.strictEqual(b.layouts.length, 1);
  const lay = b.layouts[0];
  assert.deepStrictEqual([lay.name, lay.method, lay.driver], ["monitor · c", "monitor", "c"]);
  assert.deepStrictEqual(lay.positions.shape, [2, 2, 2]);
  assert.ok(Math.abs(lay.positions.data[2] - 0.1) < 1e-7);
  assert.deepStrictEqual(lay.quality.shape, [2, 2]);
  assert.ok(Number.isNaN(lay.arrays.auroc.data[0]));          // the muzzle reads nothing
  assert.deepStrictEqual([lay.fitted, lay.params.folds], [["x: cross-fitted"], 2]);
  assert.deepStrictEqual(b.pellets[1], { id: "p1", text: "second", labels: { c: 0 } });
  assert.deepStrictEqual([b.range.shape, b.spread.shape], [[2, 2], [2]]);
});

test("a malformed blast record reads null and the scene still loads", () => {
  const bad = [
    null, "blast", [RECORD],
    { ...RECORD, layouts: [] },
    { ...RECORD, layouts: [{ ...RECORD.layouts[0], positions: "missing" }] },
    { ...RECORD, layouts: [{ ...RECORD.layouts[0], positions: "q" }] },   // wrong shape
  ];
  for (const rec of bad) {
    const s = MTJ.loadScene(blastScene(rec, PELLETS));
    assert.strictEqual(s.blast, null);
    assert.strictEqual(s.runs.length, 1);
  }
  // pellets that do not match the run cannot label it
  assert.strictEqual(MTJ.loadScene(blastScene(RECORD, PELLETS.slice(1))).blast, null);
  assert.strictEqual(MTJ.loadScene(blastScene(RECORD, undefined)).blast, null);
});

test("a bad optional member drops only that member", () => {
  const rec = { ...RECORD, range: "missing",
                layouts: [{ ...RECORD.layouts[0], quality: "pos", arrays: { auroc: "nope" } }] };
  const b = MTJ.loadScene(blastScene(rec, PELLETS)).blast;
  assert.strictEqual(b.range, null);
  assert.strictEqual(b.layouts[0].quality, null);              // wrong shape
  assert.deepStrictEqual(Object.keys(b.layouts[0].arrays), []);
});
