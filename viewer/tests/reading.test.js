"use strict";
const test = require("node:test");
const assert = require("node:assert");

const R = require("../reading.js");

// A scene as mtj.loadScene hands it over: arrays already resolved to
// {data, shape}, the raw manifest still attached.
function scene({ quality = [0.9, 0.2, 0.8, 0.1], se = true, analysis } = {}) {
  const runs = [{ label: "A", quality: quality === null ? null
                                     : { data: Float32Array.from(quality),
                                         shape: [2, quality.length / 2] } }];
  return {
    runs,
    terrain: { z: { data: new Float32Array(4), shape: [2, 2] },
               se: se ? { data: new Float32Array(4), shape: [2, 2] } : null },
    manifest: analysis === undefined ? {} : { analysis },
  };
}

const RECORD = {
  schema: "mottled-analysis/1",
  created: "2026-09-16T00:00:00Z",
  mottled: "0.2.0",
  config: { projection: "pca", density: "kde", seed: 7 },
  prompts: ["a", "b"],
  models: [{ id: "gpt2", revision: "e7da7f2abc", device: "cpu", dtype: "float32" }],
};

test("fidelitySummary counts the states the projection may have moved", () => {
  const s = R.fidelitySummary([0.9, 0.2, 0.8, 0.1]);
  assert.strictEqual(s.n, 4);
  assert.ok(Math.abs(s.mean - 0.5) < 1e-12);
  assert.strictEqual(s.low_fraction, 0.5);
  assert.strictEqual(s.low_fidelity, 0.5);
});

test("fidelitySummary refuses to summarize nothing", () => {
  assert.throws(() => R.fidelitySummary([]), /no preservation values/);
  assert.throws(() => R.fidelitySummary(null), /no preservation values/);
});

test("scene fidelity pools states rather than averaging run means", () => {
  // one long run and one short one: a mean of means would weight them alike
  const long = { quality: { data: Float32Array.from([1, 1, 1, 1, 1, 1]), shape: [6] } };
  const short = { quality: { data: Float32Array.from([0]), shape: [1] } };
  const pooled = R.sceneFidelity([long, short]);
  assert.strictEqual(pooled.n, 7);
  assert.ok(Math.abs(pooled.mean - 6 / 7) < 1e-9);
});

test("a scene with no quality layer reports no fidelity rather than zero", () => {
  assert.strictEqual(R.sceneFidelity([{ label: "A" }]), null);
  assert.strictEqual(R.sceneSummary(scene({ quality: null })).fidelity, null);
});

test("sceneSummary reads uncertainty presence off the terrain", () => {
  assert.strictEqual(R.sceneSummary(scene({ se: true })).hasUncertainty, true);
  assert.strictEqual(R.sceneSummary(scene({ se: false })).hasUncertainty, false);
});

test("uncertainty note states the lower bound, or its absence", () => {
  assert.match(R.uncertaintyNote(true), /lower bound/);
  assert.match(R.uncertaintyNote(false), /no density uncertainty/);
});

test("provenance comes from the analysis record the scene carries", () => {
  const p = R.sceneSummary(scene({ analysis: RECORD })).provenance;
  assert.strictEqual(p.models[0].id, "gpt2");
  assert.strictEqual(p.models[0].revision, "e7da7f2abc");
  assert.strictEqual(p.prompts, 2);
  assert.strictEqual(p.projection, "pca");
  assert.strictEqual(p.seed, 7);
  assert.strictEqual(p.mottled, "0.2.0");
});

test("the same model across runs is listed once", () => {
  // models is one entry per run; a two-prompt scene on one model would
  // otherwise read "gpt2, gpt2", which counts runs rather than weights
  const record = { ...RECORD, models: [RECORD.models[0], RECORD.models[0]] };
  const p = R.sceneSummary(scene({ analysis: record })).provenance;
  assert.strictEqual(p.models.length, 1);
});

test("a cross-model scene keeps every distinct set of weights", () => {
  const record = { ...RECORD, models: [
    { id: "gpt2", revision: "aaa" }, { id: "gpt2", revision: "bbb" },
    { id: "Qwen/Qwen2.5-0.5B", revision: "aaa" }] };
  const p = R.sceneSummary(scene({ analysis: record })).provenance;
  assert.deepStrictEqual(p.models.map((m) => m.id + "@" + m.revision),
                         ["gpt2@aaa", "gpt2@bbb", "Qwen/Qwen2.5-0.5B@aaa"]);
});

test("a scene without a record says so instead of inventing one", () => {
  assert.strictEqual(R.sceneSummary(scene()).provenance, null);
});

test("a missing hub commit stays absent — absent is not 'latest'", () => {
  const record = { ...RECORD, models: [{ id: "local/path" }] };
  const p = R.sceneSummary(scene({ analysis: record })).provenance;
  assert.strictEqual(p.models[0].revision, null);
});

test("seed 0 survives, since 0 is a real seed", () => {
  const record = { ...RECORD, config: { seed: 0 } };
  assert.strictEqual(R.sceneSummary(scene({ analysis: record })).provenance.seed, 0);
});

test("a record without a lens predates the tuned lens: it was the raw lens", () => {
  assert.strictEqual(R.sceneSummary(scene({ analysis: RECORD })).provenance.lens, "logit");
  const tuned = { ...RECORD, config: { ...RECORD.config, lens: "tuned" } };
  assert.strictEqual(R.sceneSummary(scene({ analysis: tuned })).provenance.lens, "tuned");
});

test("the inspector names the readout, and cautions the raw lens", () => {
  assert.match(R.readoutLabel(scene({ analysis: RECORD })), /^logit lens .*early layers/);
  // no record at all: every producer that writes none reads through the raw lens
  assert.match(R.readoutLabel(scene()), /^logit lens .*early layers/);
  // a lens directory is a tuned lens too
  const dir = { ...RECORD, config: { ...RECORD.config, lens: "/lenses/qwen" } };
  assert.strictEqual(R.readoutLabel(scene({ analysis: dir })), "tuned lens");
});

test("the notes keep docs/validity.md's vocabulary", () => {
  const text = R.NOTES.map(([h, b]) => h + " " + b).join(" ");
  assert.match(text, /state concentration region/);
  assert.match(text, /representation-space neighbors/);
  assert.match(text, /readout diagnostic/);
  assert.match(text, /biased, most at early layers/);
  // and never claim the stronger thing
  assert.doesNotMatch(text, /\battractor\b/);
  assert.match(text, /does not show|not establish|Mechanism\./);
});

test("fidelity note is null when there is no fidelity to report", () => {
  assert.strictEqual(R.fidelityNote(null), null);
  assert.match(R.fidelityNote(R.fidelitySummary([0.9, 0.1])), /50% of states/);
});
