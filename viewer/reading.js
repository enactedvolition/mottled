/* reading.js — what this scene is, how much of it survived the projection,
 * and what produced it.
 *
 * The explorer has always said this next to its figure: projection fidelity
 * inline, ✕ markers on low-fidelity states, the validity contract under the
 * scene. The viewer said none of it — and the viewer is the surface a scene
 * gets *shared* on, where the reader has no session, no config and no author
 * to ask. An attractive picture with nothing attached is exactly the
 * research-validity risk docs/validity.md exists to name.
 *
 * Everything here is derived from what the scene file already carries: per-run
 * neighborhood preservation (within-layer where the scene has it, pooled
 * across layers where it does not), the density standard-error field, and the
 * analysis record (provenance.py) that `.mtj` has carried since the manifest
 * work. Nothing is recomputed and nothing is assumed — a scene that lacks a
 * layer says so rather than showing a blank.
 *
 * `fidelitySummary` is a port of `projection.fidelity_summary`, pinned by
 * tests/test_reading_conformance.py: the two surfaces must not report
 * different fidelity for the same run.
 *
 * UMD: `window.Reading` in the browser, `module.exports` under Node. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Reading = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // projection.LOW_FIDELITY — one home, mirrored here and pinned by the
  // conformance test; the explorer's ✕ markers use the same cut.
  const LOW_FIDELITY = 0.5;

  // projection.low_fidelity_cut: a within-layer score is cut at its chance
  // level (no better than a random placement); a score with none — pooled,
  // or an older scene — at the plain LOW_FIDELITY.
  function lowFidelityCut(chance, lowFidelity) {
    const lf = lowFidelity === undefined ? LOW_FIDELITY : lowFidelity;
    return chance != null && chance > 0 ? chance : lf;
  }

  /** Mean preservation and the share of states the projection may have moved. */
  function fidelitySummary(preservation, lowFidelity, chance) {
    const cut = lowFidelityCut(chance, lowFidelity);
    const values = preservation && preservation.data ? preservation.data : preservation;
    if (!values || values.length === 0)
      throw new Error("no preservation values to summarize");
    let sum = 0, low = 0;
    for (let i = 0; i < values.length; i++) {
      sum += values[i];
      if (values[i] < cut) low++;
    }
    return {
      mean: sum / values.length,
      low_fraction: low / values.length,
      low_fidelity: cut,
      chance: chance == null ? 0 : chance,
      n: values.length,
    };
  }

  /** Fidelity across every run that carries it — the scene-level headline.
   *
   * Each run is read through what the explorer's ✕ flag reads
   * (projection.fidelity_basis): its within-layer score, against that run's
   * own chance level, when the scene carries one; otherwise `quality` with
   * the plain cut. For a scene written before within-layer scores, or a run
   * too short for one, `quality` pools neighbors across layers, and on GPT-2
   * cannot tell PCA from a random projection — `metric` says which was read.
   *
   * Runs are pooled rather than averaged over per-run means: the runs in a
   * scene can have different token counts, and a mean of means would quietly
   * weight a three-token prompt like a thirty-token one. */
  function sceneFidelity(runs) {
    let sum = 0, low = 0, n = 0;
    const metrics = new Set(), cuts = [], chances = [];
    for (const run of runs || []) {
      const lf = run && run.layerFidelity;
      const q = lf ? lf.preservation : run && run.quality;
      const values = q && q.data ? q.data : null;
      if (!values || !values.length) continue;
      const cut = lowFidelityCut(lf ? lf.chance : 0);
      for (let i = 0; i < values.length; i++) {
        sum += values[i];
        if (values[i] < cut) low++;
      }
      n += values.length;
      metrics.add(lf ? "within-layer" : "pooled");
      cuts.push(cut);
      if (lf && lf.chance != null) chances.push(lf.chance);
    }
    if (!n) return null;
    return {
      mean: sum / n,
      low_fraction: low / n,
      n,
      metric: metrics.size === 1 ? [...metrics][0] : "mixed",
      // a range, because each run is cut against its own chance level
      low_fidelity: [Math.min(...cuts), Math.max(...cuts)],
      chance: chances.length ? [Math.min(...chances), Math.max(...chances)] : null,
    };
  }

  /** The provenance lines, from the analysis record the .mtj carries. */
  function provenanceSummary(scene) {
    const record = scene && scene.manifest && scene.manifest.analysis;
    if (!record) return null;
    const config = record.config || {};
    // one entry per run, so a two-prompt scene on one model lists it twice;
    // distinct weights are what a reader needs, not a run count
    const seen = new Set();
    const models = [];
    for (const m of record.models || []) {
      const id = m.id || "unknown";
      // a hub commit when the weights came from a snapshot; absent for a
      // local path or a model built in-process, and absent is not "latest"
      const revision = m.revision || null;
      const key = id + "@" + revision;
      if (seen.has(key)) continue;
      seen.add(key);
      models.push({ id, revision, device: m.device || null, dtype: m.dtype || null });
    }
    return {
      models,
      created: record.created || null,
      mottled: record.mottled || null,
      prompts: (record.prompts || []).length,
      projection: config.projection || null,
      density: config.density || null,
      seed: config.seed === undefined ? null : config.seed,
      // a record without the field predates the tuned lens: every scene
      // written before it read its layers through the raw logit lens
      lens: config.lens || "logit",
      sae: record.sae ? { source: record.sae.source || null,
                          hook: record.sae.hook || null } : null,
    };
  }

  /** Everything the panel needs, as data. Rendering is main.js's business —
   * this stays a pure function so `node --test` can hold it to account. */
  function sceneSummary(scene) {
    const runs = (scene && scene.runs) || [];
    return {
      runs: runs.length,
      fidelity: sceneFidelity(runs),
      hasUncertainty: !!(scene && scene.terrain && scene.terrain.se),
      provenance: provenanceSummary(scene),
    };
  }

  // The vocabulary is docs/validity.md's and is deliberate: a basin is a state
  // concentration region, not an attractor; the logit lens is a readout
  // diagnostic, not the model's belief; density_se is a lower bound because
  // the bootstrap treats dependent states as independent.
  const NOTES = [
    ["What the terrain is",
     "A density field over the projected states themselves. Height means " +
     "“many states landed here” — it is not an external " +
     "landscape the model is moving through, and a basin is a state " +
     "concentration region, not a circuit computing."],
    ["What the readouts are",
     "Nearest tokens are representation-space neighbors, not semantic ones. " +
     "The logit lens is a readout diagnostic — what the output head " +
     "would say if pointed at an intermediate state — not the model’s " +
     "belief at that layer. The raw lens is also biased, most at early " +
     "layers; a scene captured with --lens tuned reads each layer through " +
     "a tuned lens, which corrects much of that and is still a readout."],
    ["What it does not show",
     "Mechanism. A scene generates hypotheses about representation geometry; " +
     "establishing a cause takes activation or path patching and held-out " +
     "prompts. docs/validity.md is the inferential contract."],
  ];

  /** The inspector's name for a scene's readout, with the raw lens's
   * caveat attached where a reader meets it rather than in a panel. */
  function readoutLabel(scene) {
    const p = provenanceSummary(scene);
    if (p && p.lens !== "logit") return "tuned lens";
    return "logit lens — biased at early layers";
  }

  // 0.5, or 0.61–0.67 when the runs of a scene were cut at different levels
  const span = (r) => {
    const [a, b] = Array.isArray(r) ? r : [r, r];
    const f = (x) => String(Number(x.toFixed(2)));
    return a === b ? f(a) : `${f(a)}–${f(b)}`;
  };

  /** One sentence sizing the projection's trustworthiness, naming the
   * measure it read, or null. */
  function fidelityNote(fidelity) {
    if (!fidelity) return null;
    const pct = (fidelity.low_fraction * 100).toFixed(0);
    const mean = fidelity.mean.toFixed(2);
    let what = `Mean neighborhood preservation ${mean}.`;
    if (fidelity.metric === "within-layer")
      what = `Mean within-layer neighborhood preservation ${mean}` +
        (fidelity.chance ? `; a random placement keeps ${span(fidelity.chance)}.` : ".");
    else if (fidelity.metric === "pooled")
      what = `Mean neighborhood preservation ${mean}, pooled across layers: this ` +
        `scene carries no within-layer score (it predates them, or its prompts ` +
        `are too short for one). The pooled score mixes in neighbors from ` +
        `other layers, and on GPT-2 it cannot tell PCA from a random projection.`;
    else if (fidelity.metric === "mixed")
      what = `Mean neighborhood preservation ${mean}: within each layer for runs ` +
        `long enough to have the score, pooled across layers for the rest.`;
    // a within-layer score is cut at its chance level (projection.low_fidelity_cut)
    let low = `low-fidelity (below ${span(fidelity.low_fidelity)})`;
    if (fidelity.chance && fidelity.metric === "within-layer")
      low = `no better than a random placement (below chance ${span(fidelity.chance)})`;
    else if (fidelity.chance && fidelity.metric === "mixed")
      low = `low-fidelity: no better than a random placement within a layer ` +
        `(below chance ${span(fidelity.chance)}), below ${LOW_FIDELITY} where pooled`;
    return `${what} ${pct}% of states are ${low} — drawn where the projection ` +
      `could put them, not where they truly are.`;
  }

  function uncertaintyNote(hasUncertainty) {
    return hasUncertainty
      ? "This scene carries a bootstrap standard error for the density. It is " +
        "a lower bound: the bootstrap treats dependent states as independent."
      : "This scene carries no density uncertainty, so nothing on the terrain " +
        "marks where its shape is unstable — read it as suggestive.";
  }

  return { LOW_FIDELITY, lowFidelityCut, fidelitySummary, sceneFidelity, provenanceSummary,
           sceneSummary, fidelityNote, uncertaintyNote, readoutLabel, NOTES };
});
