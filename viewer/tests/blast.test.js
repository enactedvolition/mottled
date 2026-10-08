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

// ---------------------------------------------------------------- viewer helpers
// What main.js draws for a blast is decided by these pure helpers; the tests
// pin the colouring, the rings and the exact readout wording.

const fs = require("node:fs");
const path = require("node:path");

const BLAST = { pellets: [
  { id: "a", text: "one", labels: { harmful: 1, refused: 1 } },
  { id: "b", text: "two", labels: { harmful: 1, refused: 0 } },   // discordant
  { id: "c", text: "three", labels: { harmful: 0, refused: 0 } },
  { id: "d", text: "four", labels: { harmful: 0 } },              // no refused label
], skipped: [], spread: { shape: [3], data: new Float32Array([0, 0.2, 0.3]) },
   range: { shape: [4, 3], data: new Float32Array([0, 0.1, 0.2, 0, 0.3, 0.4,
                                                    0, 0.5, 0.6, 0, 0.7, 0.8]) } };
const MON = { name: "monitor · harmful", method: "monitor", driver: "harmful",
              exact: ["origin"], fitted: ["x: held out"], projected: ["y: residual"],
              arrays: { auroc: { data: new Float32Array([NaN, 0.2, 0.5, 0.97]) },
                        null05: { data: new Float32Array([NaN, 0.31, 0.31, 0.31]) },
                        null95: { data: new Float32Array([NaN, 0.67, 0.67, 0.67]) } } };
const OPEN = { name: "open", method: "open", driver: null, exact: ["origin", "radius"],
               fitted: [], projected: ["camera"],
               arrays: { shown: { data: new Float32Array([1, 0.354, 0.5]) } } };

test("the blast colours are design tokens, as viewer/style.css mirrors them", () => {
  // style.css is pinned to design_tokens.py by tests/test_tokens.py, so
  // holding these to the CSS holds them to the one home of every colour
  const css = fs.readFileSync(path.join(__dirname, "..", "style.css"), "utf-8");
  const v = (name) => (css.match(new RegExp(`--${name}:\\s*(#[0-9A-Fa-f]{6})`)) || [])[1];
  const C = MTJ.BLAST_COLOURS;
  assert.deepStrictEqual(
    [C.one, C.zero, C.none, C.ring].map((c) => c.toLowerCase()),
    ["color-amber", "color-accent", "color-fg-2", "color-fg-1"].map((n) => v(n).toLowerCase()));
});

test("a label reads 1, 0 or missing, and nothing else is guessed at", () => {
  const val = (x) => MTJ.blastLabelValue({ k: x }, "k");
  assert.deepStrictEqual([1, true, 0, false].map(val), [1, 1, 0, 0]);
  assert.deepStrictEqual([null, undefined, "1", 2, "true"].map(val), [null, null, null, null, null]);
  assert.strictEqual(MTJ.blastLabelValue(null, "k"), null);
  assert.deepStrictEqual(MTJ.blastLabelNames(BLAST), ["harmful", "refused"]);
  assert.deepStrictEqual(MTJ.blastLabelNames(null), []);
});

test("colour label: a pick sticks, a monitor uses its driver, open keeps the current one", () => {
  assert.strictEqual(MTJ.blastColourLabel(BLAST, MON, null, null), "harmful");
  assert.strictEqual(MTJ.blastColourLabel(BLAST, MON, "refused", "harmful"), "refused");
  assert.strictEqual(MTJ.blastColourLabel(BLAST, OPEN, null, "refused"), "refused");
  assert.strictEqual(MTJ.blastColourLabel(BLAST, OPEN, null, null), "harmful");  // first name
  // a pick or driver that names no label falls through
  assert.strictEqual(MTJ.blastColourLabel(BLAST, { ...MON, driver: "gone" }, "nope", null), "harmful");
  assert.strictEqual(MTJ.blastColourLabel({ pellets: [{ labels: {} }] }, OPEN, null, null), null);
});

test("pellets are coloured by label value, missing labels muted", () => {
  const C = MTJ.BLAST_COLOURS;
  assert.deepStrictEqual(BLAST.pellets.map((p) => MTJ.blastPelletColour(p, "refused")),
                         [C.one, C.zero, C.zero, C.none]);
  assert.strictEqual(MTJ.blastPelletColour(undefined, "refused"), C.none);
});

test("discordance needs both labels: true where they differ, null where one is missing", () => {
  const d = (lab, drv) => BLAST.pellets.map((p) => MTJ.blastDiscordant(p, lab, drv));
  assert.deepStrictEqual(d("refused", "harmful"), [false, true, false, null]);
  assert.deepStrictEqual(d("harmful", "harmful"), [false, false, false, false]);
});

test("rings mark the pellets that break the labels' prevailing pairing", () => {
  // mostly agreeing labels: the one pellet where they differ is ringed
  const r = MTJ.blastRings(BLAST, "refused", "harmful");
  assert.deepStrictEqual(r, { ringed: [false, true, false, false], count: 1,
                              text: "ringed: refused \u2260 harmful" });
  // mostly differing labels (a surface feature riding the other class): the
  // pellets where they match are the ones that tell the two monitors apart
  const anti = { pellets: [
    { labels: { neg: 1, f: 0 } }, { labels: { neg: 1, f: 0 } }, { labels: { neg: 0, f: 1 } },
    { labels: { neg: 0, f: 0 } }, { labels: { neg: 1 } }] };
  assert.deepStrictEqual(MTJ.blastRings(anti, "neg", "f"),
                         { ringed: [false, false, false, true, false], count: 1,
                           text: "ringed: neg = f" });
  // a tie rings disagreement
  const tie = { pellets: [{ labels: { a: 1, b: 1 } }, { labels: { a: 1, b: 0 } }] };
  assert.deepStrictEqual(MTJ.blastRings(tie, "a", "b").ringed, [false, true]);
  // nothing to ring: colouring by the driver, or a layout without one (open)
  assert.strictEqual(MTJ.blastRings(BLAST, "harmful", "harmful"), null);
  assert.strictEqual(MTJ.blastRings(BLAST, "harmful", null), null);
});

test("the legend says what 1 and 0 are as the data writes them, and what a ring means", () => {
  const lg = MTJ.blastLegend(BLAST, MON, "refused");
  assert.deepStrictEqual(lg.rows.map((r) => [r.text, r.count]),
                         [["refused = 1", 1], ["refused = 0", 2], ["no refused label", 1]]);
  assert.deepStrictEqual(lg.ring, { text: "ringed: refused ≠ harmful", count: 1 });
  assert.strictEqual(MTJ.blastLegend(BLAST, MON, "harmful").ring, null);    // colour is the driver
  assert.strictEqual(MTJ.blastLegend(BLAST, OPEN, "refused").ring, null);   // open has no driver
  assert.strictEqual(MTJ.blastLegend(BLAST, MON, "harmful").rows.length, 2);
  const boolish = { pellets: [{ labels: { neg: true } }, { labels: { neg: false } }] };
  assert.deepStrictEqual(MTJ.blastLegend(boolish, OPEN, "neg").rows.map((r) => r.text),
                         ["neg = true", "neg = false"]);
});

test("the monitor readout prints the record's AUROC beside its shuffle null", () => {
  const r = (l) => MTJ.blastReadout(BLAST, MON, l);
  assert.deepStrictEqual(r(0), { text: "layer is one point: no direction", flag: null });
  assert.deepStrictEqual(r(3), {
    text: "held-out AUROC 0.97 · shuffle null 5–95%: 0.31–0.67", flag: null });
  assert.match(r(1).flag, /^below the null band: check for near-duplicate or paired prompts/);
  assert.match(r(2).flag, /^inside the null band/);
  // without a null band there is nothing to compare against, and no flag
  const bare = { ...MON, arrays: { auroc: MON.arrays.auroc } };
  assert.deepStrictEqual(MTJ.blastReadout(BLAST, bare, 3), { text: "held-out AUROC 0.97", flag: null });
  assert.strictEqual(r(9), null);
  assert.strictEqual(r(1.5), null);
});

test("the open readout is the share of the layer's spread the camera keeps", () => {
  const r = (l) => MTJ.blastReadout(BLAST, OPEN, l);
  assert.strictEqual(r(1).text, "camera shows 35% of this layer's spread");
  assert.strictEqual(r(0).text, "layer is one point: every pellet sits at the origin");
  assert.strictEqual(MTJ.blastReadout(BLAST, { ...OPEN, arrays: {} }, 1), null);
});

test("the caption is the record's own lines, the frame note, and what was skipped", () => {
  const cap = MTJ.blastCaption({ ...BLAST, skipped: ["refused: too few"] }, MON);
  assert.deepStrictEqual(cap.map((c) => c.kind),
                         ["exact", "fitted", "projected", "frame", "not drawn"]);
  assert.deepStrictEqual(cap.slice(0, 3).map((c) => c.text), ["origin", "x: held out", "y: residual"]);
  assert.match(cap[3].text, /each layer has its own frame/);
  assert.match(cap[3].text, /two different frames/);
  assert.deepStrictEqual(MTJ.blastCaption(BLAST, null), []);
});

test("the inspector reads the pellet's exact range at a layer", () => {
  const p = MTJ.blastPellet(BLAST, 2, 1);
  assert.deepStrictEqual([p.id, p.text, p.labels], ["c", "three", [["harmful", "0"], ["refused", "0"]]]);
  assert.ok(Math.abs(p.range - 0.5) < 1e-7);                 // (N, L): pellet 2, layer 1
  assert.strictEqual(MTJ.blastPellet(BLAST, 2, 7).range, null);
  assert.strictEqual(MTJ.blastPellet({ ...BLAST, range: null }, 2, 1).range, null);
  assert.strictEqual(MTJ.blastPellet(BLAST, 9, 1), null);
});

test("long pellet text keeps both ends", () => {
  assert.strictEqual(MTJ.truncateMiddle("short  text", 40), "short text");
  const s = "What is the question that this long prompt asks the model to answer? -> " +
            "The answer the state was read at the end of";
  const t = MTJ.truncateMiddle(s, 60);
  assert.ok(t.length <= 62, t);
  assert.ok(t.startsWith("What is the"), t);
  assert.ok(t.endsWith("read at the end of"), t);
  assert.ok(t.includes(" … "), t);
});

// The shipped samples, read the way the viewer reads them.
const SAMPLE = (name) => MTJ.loadScene(
  fs.readFileSync(path.join(__dirname, "..", "samples", "blast", name)));

test("refusal sample: colouring by refused under the harmful monitor rings only h23", () => {
  const b = SAMPLE("refusal.mtj").blast;
  const mon = b.layouts.find((l) => l.name === "monitor · harmful");
  const ringed = b.pellets.filter((p) => MTJ.blastDiscordant(p, "refused", mon.driver));
  assert.deepStrictEqual(ringed.map((p) => p.id), ["h23"]);
  assert.deepStrictEqual(MTJ.blastLegend(b, mon, "refused").ring,
                         { text: "ringed: refused ≠ harmful", count: 1 });
  assert.strictEqual(MTJ.blastReadout(b, mon, 0).text, "layer is one point: no direction");
  assert.match(MTJ.blastReadout(b, mon, 1).text, /^held-out AUROC 0\.77 /);
  assert.match(MTJ.blastReadout(b, mon, 16).text, /^held-out AUROC 1\.00 /);
  const open = b.layouts.find((l) => l.method === "open");
  assert.strictEqual(MTJ.blastColourLabel(b, open, null, null), "harmful");
  assert.match(MTJ.blastReadout(b, open, 24).text, /^camera shows \d+% of this layer's spread$/);
});

test("truth sample: colouring by negation under false_claim rings the unconfounded pellets", () => {
  // negation words sit on 12 of 24 true answers and 2 of 24 false ones, so
  // the labels mostly differ; the 14 pellets where they match are ringed
  const b = SAMPLE("truth.mtj").blast;
  const mon = b.layouts.find((l) => l.name === "monitor · false_claim");
  const r = MTJ.blastRings(b, "negation", mon.driver);
  assert.strictEqual(r.text, "ringed: negation = false_claim");
  assert.strictEqual(r.count, 14);
  const ringed = b.pellets.filter((p, i) => r.ringed[i]);
  assert.ok(ringed.every((p) => p.labels.negation === p.labels.false_claim));
});
