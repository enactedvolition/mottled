# Research validity — what a Mottled picture can and cannot establish

**The boundary, in one sentence:** Mottled visualizes and quantitatively
summarizes representation-space behavior under declared analysis choices; it
generates mechanistic hypotheses that require full-dimensional, controlled,
and causally targeted validation.

**The 30-second version** — scan this before interpreting your first scene:

- A basin = states accumulated under *this* projection and estimator.
  Not an attractor. Not a circuit.
- A successful steer = sufficiency under the tested conditions. Not the
  mechanism that normally produces the behavior.
- Nearest tokens = representation-space neighbors. Not semantic neighbors.
- Logit lens = a readout diagnostic. Not the model's "belief" at that layer.
- SAE labels = auto-interp correlations. Not verified functions.
- `density_se` = a lower bound (the bootstrap treats dependent states as
  independent). Not a confidence statement.
- A blast monitor split = these labelled prompts are linearly separable,
  held out, at that layer. Not what the model computes, and not yet a
  monitor: check wording, contrast pairs and surface features first.
- A pretty picture ≠ a mechanism. Confirm on held-out prompts, in full
  hidden space, with causal methods.

The rest of this file is the full version of each line, for paper authors
and methodologists.

---

This file is the project's inferential contract. The README says what the
tool does; this says what a result from it *licenses you to claim*. The
main research-validity risk of a tool like this is specific: an attractive
within-run visualization can be mistaken for evidence of a model mechanism.
Every section below names that gap for one part of the stack, states what
the existing machinery already measures, and marks what remains open.

Rule of the file: for each artifact the tool produces, know which of these
you are looking at —

1. **A rendering choice** (colors, smoothing, camera, spline densify).
2. **A measurement of the projection**, not the model (density terrain,
   basin membership, projected step lengths).
3. **A measurement of the model under a declared readout** (logit lens,
   entropy, SAE activations, neighbor lists).
4. **A causal test** (perturb-and-replay with controls).

Only (4) supports causal language, and even it supports less than it seems
to (see [Interventions](#interventions-sufficiency-not-mechanism)).

---

## The inferential boundary

The central object is a trajectory through residual-stream states after a
large compression into 2-D or 3-D, followed by a density estimate and a
terrain rendering. A visible basin therefore establishes, at most, that
**projected sampled states concentrate under a chosen projection and
density estimator**. It does not by itself establish a dynamical attractor,
a computational subroutine, a semantic variable, or a causal mechanism.

The default technical term for what the terrain shows is a **state
concentration region**. "Attractor" survives in the API (`attractor.py`)
and the UI as descriptive geometry — where this run's states accumulate —
and is reserved, as a technical claim, for an explicitly operational,
counterfactually tested definition: stability under perturbation and
recurrence under a defined evolution rule. Nothing in the current tool
performs that test, so nothing in the current tool establishes an attractor
in the dynamical-systems sense.

## Projection is a transformation of geometry, not a window

PCA and UMAP can materially alter geometry: relative distances, apparent
curvature, neighborhood structure, density, and clustering can all change
when hundreds or thousands of dimensions become two or three.

**What is measured now.** `projection.projection_quality` reports per-state
neighborhood preservation, PCA reconstruction residual, and explained
variance; the explorer prints the fidelity header inline and flags
low-preservation states on the terrain; `attractor.explain` folds the
basin's own preservation into its prose.

**What that does not cover.** A good *average* fidelity score does not
validate the particular visually salient feature a reader interprets —
especially a ridge, a convergence, or a basin. Before presenting a visual
pattern as more than exploratory, put a **robustness envelope** around it:

- Re-run the claim across projection seeds, dimensions (2-D vs 3-D), and
  methods (PCA vs UMAP).
- Check whether the concentration region survives in the original hidden
  space, with kNN density or radius-based measures — `metrics.py` and
  `compare.py` already operate in full hidden space for exactly this
  reason; prefer their numbers over projected geometry for any claim.
- Require agreement between projected and full-dimensional measures before
  the pattern graduates from "exploratory."
- Treat UMAP output as descriptive topology, not metric geometry: apparent
  distances and path shapes under UMAP are not faithful quantitative
  trajectories unless independently validated. PCA at least admits an exact
  statement of what was discarded (the residual); UMAP does not.

## Density is estimated from dependent samples

The terrain is a density estimate over states from one or a few prompts —
not an independent sample from any distribution over model activity. A
peak can arise trivially because adjacent layers and nearby token positions
are autocorrelated and produce many similar states; it need not indicate a
special regime of the model.

**Known limit of the current bootstrap.** `compute_density(bootstrap=B)`
resamples *individual states* with replacement. Because layer-token states
within a run are highly dependent, that i.i.d. bootstrap understates
uncertainty: it treats dependent observations as exchangeable, so
`density_se` is a lower bound on the honest error, useful for spotting
bandwidth artifacts but not for confidence statements about the model.
The honest upgrades, in order of strength:

- **Block bootstrap** over prompts, token trajectories, or contiguous layer
  segments, so resampling respects the dependence structure.
- A **prompt ensemble** drawn from a declared corpus and distribution, so
  the terrain estimates something beyond the run that built it.
- **Null comparisons** that preserve sequence length and layer structure:
  matched random prompts, shuffled token assignments, or synthetic
  random-walk controls generated to match the run's shape.
- A **pre-specified operational criterion** for calling something a basin —
  threshold, null distribution, effect size — declared before looking.

None of these are implemented yet; until they are, a basin from a single
run is an observation about that run's projected states, full stop.

## "Dynamics" is three different things

A transformer forward pass is depth-indexed computation, not autonomous
time evolution of a state under a fixed transition rule. Generation adds a
time-like axis, but each new token changes the context and therefore the
computation being performed. "Trajectory" is a fine visualization term;
claims about settling, velocity, curvature, or attraction must say which
of these they mean:

| axis | what it is | what a claim on it can say |
|---|---|---|
| **Depth progression** | one forward computation through non-identical blocks | "the state's projected step length decreases after layer 9" |
| **Decode progression** | successive computations on *changing* contexts | "successive contexts produce nearby states" — not "the same system evolves" |
| **Dynamical-system behavior** | stability under perturbation + recurrence under a defined evolution rule | requires perturbation-stability and recurrence tests Mottled does not perform |

"The computation settles into an attractor" is a claim of the third kind
and needs the third kind's evidence. Everything `attractor.py` measures —
deceleration, membership, readout stabilization, entropy collapse — is a
claim of the first kind, and its prose is written to stay there.

## Logit lens is a readout diagnostic

Applying the final unembedding to intermediate residual states assumes
intermediate representations are comparable to the final readout space
without the later transformations that normally make them legible. That
assumption is known to mislead, especially in early layers. Per-layer
predictions and entropies are therefore **probe/readout diagnostics** —
"what the output head would say if pointed here" — not evidence that the
model "believes" or has committed to a token at that layer.

Stronger validation, none of it implemented here: tuned-lens or affine
probes fitted on held-out prompts, compared against the raw lens;
intermediate prediction stability compared against actual causal
predictiveness of those states; calibration and rank correlation between
intermediate readout scores and the final output distribution.

## A blast monitor is a held-out readout

The blast's **monitor** layout (`blast.py`) draws each prompt where a linear
readout for one label puts it. What it measures: whether the labelled prompts'
states at that layer separate along the diff-of-means direction **when each
prompt is scored on a direction fitted without it** (K-fold cross-fitting),
reported as an AUROC beside the same statistic under label shuffles. That is
the question a deployment monitor asks, answered for these prompts. It is not
evidence that the model represents the label as such, or uses it.

What the cross-fitting does not remove, each of which the samples in
`viewer/samples/blast/` were built to show:

- **Wording.** If the two classes differ in form, the readout can read the
  form. Refusal against Alpaca's harmless prompts read 0.94 at layer 1;
  against harmless prompts written in AdvBench's imperative style, 0.77.
- **Contrast pairs.** Two answers to one question share most of their state.
  Scored while its twin is in training, a held-out pellet lands on the twin's
  side: TruthfulQA pairs read below the shuffle null at 9 of 25 layers until
  each pair was held out together (`group`).
- **Surface features.** On those pairs a one-bit negation label reads as well
  as the true/false label it travels with (12 of 24 true answers carry a
  negation word, 2 of 24 false ones). Put the surface feature in as a second
  label and compare the two monitors.
- **Behaviour vs request.** A monitor for what the model *did* (refused) is
  only distinguishable from one for what it was *asked* (harmful) through the
  prompts where the two disagree; with one such prompt, it is not.

The **open** layout uses no labels, but its camera is built from every depth,
so structure from deeper layers can show at shallower ones (each pellet's own
bearing is left out of its camera, which reduces that, not removes it). Read a
split there as a lead for the monitor, not as an onset.

## Interventions: sufficiency, not mechanism

Perturb-and-replay (`intervene.py`) is the closest thing in Mottled to
causal evidence, and it is real evidence — of **sufficiency under the
tested conditions**. The right wording is: "this perturbation is sufficient
to alter the downstream readout under these conditions" — never "this
direction is the cause of the behavior." A sufficiently large off-manifold
perturbation can force a desired output without identifying the mechanism
that normally produces it.

**What is measured now.** `intervene.faithfulness` scores a steer against a
norm-matched random control, so the effect of the *direction* is separated
from the effect of the *push*. That is an effect size, not a circuit, and
the README says so.

**What the control does not resolve:** distribution shift (the edited state
may lie off the model's activation manifold), direction selection and
multiple testing (many candidate directions were available), and downstream
nonlinear amplification. For research-grade causal language, add:

- **On-manifold checks** — compare the intervened state to the empirical
  activation distribution at that layer.
- **Dose-response curves** — effects across perturbation magnitudes, not
  one successful magnitude. `dose.dose_sweep` runs one: a signed grid of
  doses relative to the injection layer's median residual norm, with
  seeded random controls orthogonal to the direction, a shuffled-label
  control for contrast directions, and a direct-path reference for token
  directions. Doses above 1 are flagged `replacement_regime`, since the push
  there outweighs the state it was added to. A curve that clears its
  controls is still sufficiency, and still one prompt set.
- **Direction specificity** — target direction vs random, semantically
  related, and matched-norm orthogonal directions.
- **Restoration tests** — ablate a candidate component, then restore it;
  removal-impairs plus replacement-rescues is far stronger than either.
- **Cross-prompt replication** — the effect must persist beyond the prompt
  used to derive the steering direction (`direction_from_contrast` derives
  from sets of runs, which helps, but replication is still on the user).

## SAE features: gates before claims

Feature labels are auto-interpreter descriptions (the explorer names the
model that wrote them), and a dictionary trained on differently
preprocessed activations can fit the captured distribution poorly —
`sae.fit_report` measures this because provenance is not calibration
(see field-notes trap #1). A sparse feature that reconstructs poorly on
the captured distribution cannot support a semantic or mechanistic claim.

The gates, per claim level — treat these as non-optional:

| claim level | minimum evidence |
|---|---|
| Feature visualization | SAE source, layer/site, preprocessing match, reconstruction metrics, firing statistics — `fit_report` + the provenance the UI already prints |
| Feature label | label provenance (`explained_by`), top activating examples, counterexamples, human review |
| Feature-function hypothesis | cross-prompt selectivity, causal ablation/activation, matched controls |
| Mechanistic feature claim | replication across seeds/checkpoints/prompts, plus necessity *and* sufficiency evidence |

Mottled's machinery covers the first row and part of the second; the third
and fourth rows are outside the tool. And no SAE decomposition is uniquely
"the" feature basis: dictionaries are non-identifiable up to meaningful
representational alternatives. Sparsity buys a usable coordinate system,
not metaphysical feature truth.

## Neighbors are representation-space neighbors

Nearest neighbors in residual or embedding space are not automatically
semantic neighbors. Proximity can reflect token frequency, syntax,
position, morphology, formatting, or shared downstream logit behavior.
The honest name for what `neighbors.py` returns is
**representation-space neighbors**: the k nearest token embeddings to a
hidden state under the declared normalization. The explorer labels them
that way, and any write-up should say which
space, which layer, which normalization, and that the index is token
*embeddings*, not observed hidden states.

To call them semantic: test whether displayed neighbors predict a declared
semantic relation better than frequency- and part-of-speech-matched
baselines. Absent that test, they are representation-space neighbors.

## Cross-model comparisons are readout-alignment diagnostics

Readout space (`crossmodel.py`) compares models' *output distributions*
over a shared vocabulary — a real shared coordinate system, and only that.
It does not compare hidden representations or computations, and the shared
subset plus `⟨unshared⟩` bucket can conceal tokenization and
probability-mass differences the projection cannot show. Depth
normalization, CKA layer alignment (which already reports whether its own
answer is identified), and generation comparison are all
**functional/readout alignment diagnostics**. "Similar outputs" and
"similar mechanisms" are different claims; stronger cross-model statements
need matched corpora, tokenizer-aware evaluation, and uncertainty across
prompts.

## Researcher degrees of freedom

The tool deliberately supports exploratory freedom: projection method,
density estimator, bandwidth, layer/token selection, prompts, SAE
features, and intervention magnitude can all be varied until a compelling
picture emerges. That is what an exploratory instrument is *for*, and it
is exactly what makes post-hoc visualization vulnerable. For any paper or
public claim built on Mottled output:

- Pre-register — or at minimum timestamp — a claim-specific analysis plan.
- Separate exploratory scene construction from held-out confirmatory
  prompts.
- Report all attempted parameterizations, or provide a reproducible sweep.
- Use prompt-level effect sizes and confidence intervals; a screenshot is
  an illustration, not evidence.
- Version-lock model weights, tokenizer, library versions, GPU precision,
  seeds, and SAE artifact hashes. Every exported scene carries this as its
  `analysis` record — the config, the environment, the resolved device and
  dtype, the hub commit the weights came from, and a content hash of the
  dictionary — and `mottled export-manifest scene.mtj` prints it as citable
  JSON for a methods section. What the record cannot know, say yourself:
  it reports the parameterization, not that the run reproduces.

## Where this leaves the project

Mottled's defensible position is an **auditable hypothesis-generation and
diagnostic environment**: it makes representations, reductions,
uncertainty, and interventions inspectable in one place, then directs its
user toward independent causal methods — activation patching, path
patching, circuit analysis, behavioral evaluation — for anything that
needs the word "mechanism." That restraint is not a disclaimer bolted on;
it is the project's methodological identity. A visualization that cannot
state its own inferential ceiling is worse than none.
