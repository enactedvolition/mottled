# experiments/

Research code that is not part of Mottled's public API. `pyproject.toml`
lists its modules and packages explicitly and this directory is not among
them, so nothing here ships in a wheel; interfaces may change without notice
until the research requirements settle. Run from the repository root
(`python -m experiments.<name>`), which puts the top-level modules on the path.

| File | What it is |
|---|---|
| `patching.py` | Activation patching at one residual-stream site, read from the model's own output |
| `lens_patch_predictiveness.py` | Do raw/tuned lens readouts anticipate patch effects? (pre-registered below) |
| `sotu_corpus.jsonl` | 80 State of the Union excerpts, the experiment's corpus |
| `make_sotu_corpus.py` | Rebuilds `sotu_corpus.jsonl` from the npm package, byte for byte |

## `patching.py`

`patch(model, input_ids, layer, position, value, readout)` overwrites the
residual state at one site and returns the model's own final logits
(`out.logits`, not a lens) at the readout positions, for the clean pass and
the patched one, as float32. The site uses the `StateTrajectory` convention
(layer 0 = embedding stream, layer k = output of block k-1) and indexes the
exact `input_ids` given: nothing is re-tokenized. The result also carries the
state at the site before and after the write, and a provenance record (model,
resolved hub commit, family, dtype, device, torch/transformers versions,
input ids, site, readout, and the written value's norm and hash). It is built
on `capture.HookCapture` and `intervene.SetState`, so a patch is the same
write-edit every other Mottled intervention uses. Only the readout positions'
logits are computed and kept (`logits_to_keep`), never an `(L, T, V)` stack.

Pinned by `tests/test_patching.py`: a self-patch (`h <- h`) gives bit-identical
logits; the clean logits are bit-identical to the same call with no hooks
installed; patching position t leaves every position before t bit-identical;
at the last layer the lens prediction equals the observed change.

## `lens_patch_predictiveness.py` — pre-registered analysis plan

Registered in the commit that added this file, before any GPT-2 number from
this experiment existed. Changes after that commit are listed under
[Deviations](#deviations-from-the-plan) with their reasons.

### Question

PR #58 showed the tuned lens agrees better with GPT-2's output *distribution*
than the raw logit lens does (layer-0 mean KL 40.49 nats raw vs 4.69 tuned).
Better distributional agreement does not show that a readout better explains
what an intermediate state *does*. This asks a causal-predictive version:
when the residual state at one site is replaced, does a lens reading of the
replaced state at that layer anticipate the change the model's own final
output then shows — and does the tuned lens anticipate it better than the raw
lens?

### Design

**Model.** GPT-2 small, hub id `gpt2` (openai-community/gpt2), revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. Registered as an assumption (the
commit PR #54 recorded; the container this plan was written in had no hub
access) and since confirmed: a GitHub Actions run loading `gpt2` logged that
commit as resolved (Evidence run 38985813312, job 117256900327). The model
and tokenizer are loaded at that revision and the run refuses, before any
forward pass, if the resolved `config._commit_hash` differs. float32, CPU,
eval mode, `torch.set_num_threads(4)` (recorded),
`torch.use_deterministic_algorithms(True)`.

**Tuned lens.** `tuned_lens.from_hub("gpt2")` from the AlignmentResearch/
tuned-lens Space. Its commit is unknown offline, so `--lens-revision`
defaults to the Space's main at run time; the resolved commit, the lens's
recorded `base_model_revision` and a SHA-256 of its translator weights are
recorded, and the revision is pinned after the first run.

**Corpus.** `sotu_corpus.jsonl` (provenance [below](#corpus-provenance)):
80 excerpts of about 1,000 characters, one per address, 1790–2015. The run
refuses if the file's SHA-256 differs from the one recorded here.

**Items.** Each excerpt is tokenized with the GPT-2 tokenizer at the pinned
revision with `add_special_tokens=False` (HF GPT-2 prepends no BOS, and none
is added). Windows are token spans `[32m, 32m+32)` for m = 0, 1: two
non-overlapping 32-token windows per document, 160 items, every item's exact
token ids recorded.

**Split.** Documents sorted by id (which sorts them by year); even rank →
train (40 documents, 80 items), odd rank → test (40, 80). Splitting documents
rather than items keeps the two windows of one address on the same side.

**Site and readout.** Patch site = position 31 (the window's last token) at
every layer k = 0..12. Readout = the model's final logits at that same
position, i.e. its prediction of the token after the window.

**Conditions**, per (item, layer):

- *donor*: h ← h_donor, the same-layer, same-position clean state of the
  item's partner. Partner = the document one rank later in the same split
  (cyclically), same window index — always a different document.
- *norm-matched random*: h ← h + r, ‖r‖ = ‖h_donor − h‖, direction seeded
  (`intervene._norm_matched_random`, seed `item·13 + k`).
- *self*: h ← h.

**Targets.** Primary y = the clean run's final top-1 token at the readout.
Secondary y = the partner's clean final top-1 token.

**Observed change.** Δobs(y) = LO_patched(y) − LO_clean(y), where LO(y) =
log p_y − log(1 − p_y) = z_y − logsumexp_{j≠y} z_j over the final logits z,
computed in float64 (the second form is the same quantity with no
cancellation as p_y → 1).

**Predicted change.** For lens λ ∈ {raw, tuned}: Δpred_λ(y) = the same
log-odds change read through λ at layer k, on the patched state at the site
versus the clean state at the site. Raw = final norm + unembedding
(`capture.logit_lens`); tuned = `tuned_lens.translate` then the same. Δpred
uses only the state at the site; it never sees the downstream computation.

**Controls** — the run exits non-zero if either fails, after writing its
outputs:

- *Self-patch identity*: over every item and layer, the self-patched final
  logits must equal the clean ones bit for bit (max |Δ| exactly 0), and
  Δobs = 0 exactly.
- *Positive control*: at k = 12 both lenses are the head itself, so Δpred
  must equal Δobs. Max |Δpred − Δobs| over items, the donor and random
  conditions, both targets and both lenses must be ≤ 1e-3 nats. Basis: 2.8e-5
  measured on a randomly initialised GPT-2-small-shaped model with its logits
  scaled to ±350 (the residue is float32 rounding between a (1, D) and a
  (1, T, D) head call).

**Held-out evaluation**, per cell (lens × layer × condition ∈ {donor, random}
× target): fit Δobs ≈ a + b·Δpred by least squares on train items. On test
items report calibrated MAE and RMSE, uncalibrated MAE (|Δobs − Δpred|), and
the null predictor's MAE (|Δobs − mean of train Δobs|). A lens *beats the
null* in a cell only if the 95% interval of MAE_cal − MAE_null lies below 0;
otherwise that cell is a null result for that lens.

**Rank correlation.** Spearman ρ(Δpred, Δobs) on test items per cell;
Pearson r as a secondary point estimate.

**Uncertainty.** Cluster bootstrap over the 40 test documents (documents
drawn with replacement, both windows of each kept together), B = 2000,
seed 0, percentile 95% intervals. Every statistic and every cell uses the
same resampled document sets, so lens comparisons are paired. Calibration
coefficients and the null's train mean are fixed from train; the intervals
describe test-set sampling only. Intervals: ρ_raw, ρ_tuned, ρ_tuned − ρ_raw,
MAE_cal(tuned) − MAE_cal(raw), and MAE_cal − MAE_null per lens.

### Hypothesis and decision rules

**H1 (primary).** In the donor condition, for the primary target, at layers
1–11, Δpred from the tuned lens rank-correlates with Δobs more strongly than
Δpred from the raw lens: ρ_tuned − ρ_raw > 0.

- **Per layer** k = 1..11, from the paired 95% interval of ρ_tuned − ρ_raw:
  **tuned better** if it lies above 0, **raw better** if it lies below 0,
  **not distinguishable** if it spans 0.
- **Pooled**: the mean of ρ_tuned − ρ_raw over layers 1–11, its interval
  taken over the same bootstrap replicates, with the same three verdicts.
- The 11 per-layer intervals are unadjusted. Under no true difference about
  0.55 of them are expected to read "distinguishable" by chance; an isolated
  per-layer verdict should be read with that in mind.
- Layer 0 (the embedding stream, where a donor patch amounts to swapping the
  input token's embedding) and layer 12 (the positive control) are reported
  and are outside H1.

Everything else — the secondary target, the random condition, the MAE
comparisons, beats-null per cell — is reported descriptively, with no
confirmatory claim. Every cell is in `summary.json` whatever it shows: null
and reversed results are reported with the same prominence as positive ones.

### Sample size

Fixed at 80 documents × 2 windows by run cost, then checked for what it can
detect.

*Cost.* One hooked, patched pass of a GPT-2-small-shaped model (random init:
same shapes and FLOPs as GPT-2) over 32 tokens with logits kept at one
position took a median 55 ms here (4-core Intel Xeon @ 2.8 GHz, torch 2.13,
4 threads; 71 ms keeping all positions' logits; 24 tokens took 52 ms, so the
pass is mostly weight traffic and shorter windows buy little). Per item:
1 clean + 13 layers × 3 conditions = 40 passes; 160 items = 6,400 passes ≈
5.9 min, about 7 min with loading, lens readouts and the bootstrap. For a
4-vCPU GitHub runner this is an *estimate*, 6–13 min, assuming it is up to 2×
slower; it was not measured there. (The tiny test model is no basis: its
7.7 ms per pass is per-call overhead, so a FLOP ratio from it would mislead.)

*Detectable difference.* 80 test items in 40 documents. Assuming ρ ≈ 0.5,
correlation ≈ 0.6 between the two lenses' predictions, and an intraclass
correlation ≈ 0.2 between a document's two windows (design effect 1.2,
effective n ≈ 67), the Meng–Rosenthal–Rubin approximation for two dependent
correlations sharing a variable gives SE(ρ_tuned − ρ_raw) ≈ 0.10 per layer:
a 95% half-width ≈ 0.19, and 80% power at α = 0.05 for a per-layer
difference ≈ 0.27. If the layer-wise differences correlate ≈ 0.5, the
pooled mean's SE is ≈ 0.07 (half-width ≈ 0.14, 80% power at ≈ 0.20).
These are planning assumptions, not measurements. **A "not distinguishable"
verdict therefore means no difference larger than roughly 0.2–0.3 was
detected — not that the lenses are equivalent.**

### Interpretation

In the vocabulary of `docs/validity.md`:

- A lens is a **readout diagnostic**. Δpred is what the head (with or without
  a translator) says about the patched state at layer k. A lens that predicts
  Δobs well anticipates the downstream effect of that replacement on this
  corpus, under these conditions; that is not the model's belief at layer k.
- A patch effect shows **sufficiency** of that state replacement to move the
  final readout under the tested conditions, not the mechanism that normally
  produces the output. Donor states are real states from other contexts;
  random ones are norm-matched and may lie off the activation manifold.
- "Tuned better" would mean the tuned readout of an intermediate state ranks
  patch effects better than the raw one here. Generalization is limited to
  this model revision, this corpus (formal political English, 1790–2015),
  32-token windows, the last position, and top-1 targets.

### Outputs

`--out results.json`: schema string, configuration, provenance (git commit,
model and lens resolved revisions, corpus SHA-256, environment versions,
threads), and one record per item (document id, split, window, exact token
ids, site, partner, targets, and per layer and condition Δobs and Δpred
raw/tuned). `--summary summary.json`: per-cell metrics, intervals, the H1
verdicts, controls, configuration, item counts and provenance, without the
items — sorted keys, floats rounded to 6 significant digits, no timestamps,
so a rerun produces the same bytes.
SHA-256 of both files goes to stderr at the end.

### Deviations from the plan

None so far.

### Notes after registration

Facts established after the plan was committed. None of them changes the
design above.

- **Revision.** The pinned GPT-2 commit is what the hub serves for `gpt2`
  (Evidence run 38985813312); see *Model*.
- **Full-size dry run.** `main()` on a randomly initialised model with
  GPT-2-small's exact shapes (12 × 768, vocabulary 50,257), 80 word-level
  documents, a 12-translator synthetic lens, 4 threads and B = 2000: the same
  6,400 forward passes and lens readouts as the real run. Measured here
  (4-core Xeon @ 2.8 GHz, while other processes held the load average at
  3.5–7): 724 s and 661 s for two runs (their own `timing`), 2.6–2.8 s per
  item in steady stretches, about 8–9 min uncontended. For a 4-vCPU GitHub
  runner that
  suggests 8–17 min, still an estimate. Controls: self-patch max |Δlogit| = 0
  over all 2,080 self-patches, positive-control max error 8.35e-07. The two
  runs wrote byte-identical `summary.json` (20,525 bytes) and `results.json`
  identical apart from `timing`. Its H1 table is meaningless (random
  weights, random translators) and is not evidence about lenses.

### Running it

    pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
    python -m experiments.lens_patch_predictiveness --out results.json --summary summary.json
    cat summary.json

From the repository root. Downloads GPT-2 and the tuned lens from the hub;
nothing beyond `requirements.lock` is needed (`huggingface_hub` arrives with
transformers). Status, the control values, the H1 table and both files'
SHA-256 go to stderr; the exit code is 1 if a control failed. `--corpus`
exists so the tests can point the run at a tiny corpus; the registered file
is checked by SHA-256 and any other is a deviation. `main(argv, loaded=(model,
tokenizer, lens))` runs everything but the download on models you pass:
`tests/test_lens_patch_predictiveness.py` uses it with a tiny model and a
lens of random translators (machinery only — its numbers mean nothing).
`pytest -m network tests/test_patching.py` checks the self-patch identity and
the last-layer control on real GPT-2 at the pinned revision in seconds, as a
preflight for the full run.

## Corpus provenance

`sotu_corpus.jsonl` is derived from the npm package
**`@stdlib/datasets-sotu@0.2.3`** (published 2026-02-08):

- tarball: `https://registry.npmjs.org/@stdlib/datasets-sotu/-/datasets-sotu-0.2.3.tgz`
- npm `dist.integrity`: `sha512-QbspCBwsy6S6io8mwmZoPh55YBCj4X3RwoGldDneFZU8WI9Vjrdq/l01beIF0qSiJBFlsngiIz8ySmmRjFm8aA==`
  (verified against the downloaded tarball; tarball SHA-256
  `ab03f909e80b99afe2aa74474f7ca53e976982efa9f8c77f2bf3c34f179c85c9`)
- licence: the package's README states the data files are under the ODC
  Public Domain Dedication & License 1.0 and their contents under CC0 1.0;
  State of the Union addresses are works of the US Government and not under
  copyright in the US. The package's code is Apache-2.0. None of its code was
  run: the tarball was extracted into an empty directory and only
  `data/*.json` and `package.json` were read.
- contents: 233 addresses, 1790–2021, one JSON file each (`year`, `name`,
  `party`, `text`).
- selection rule (`make_sotu_corpus.py`, by SHA-256 so it does not depend on
  any RNG implementation): the 80 addresses with the smallest
  `sha256("mottled-lens-patch-v1:doc:" + id)`; from each, whitespace
  collapsed, an excerpt starting after the first ". " at or past offset
  `sha256("mottled-lens-patch-v1:offset:" + id) mod (len − 1000)`, 1,000
  characters cut back to the last space. Each line records the address's
  id, year, name, party and the excerpt's character offset.
- committed file: 89,846 bytes, SHA-256
  `0e00dc3a326d362094e8c142418edbf6c1a5a42421ed9ecb6b3a108605b6d9a2`.

To rebuild: download and verify the tarball, extract it into an empty
directory, then
`python -I experiments/make_sotu_corpus.py <dir>/package experiments/sotu_corpus.jsonl`.
