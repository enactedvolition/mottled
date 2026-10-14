# 🔮 Mottled

[![CI](https://github.com/enactedvolition/mottled/actions/workflows/ci.yml/badge.svg)](https://github.com/enactedvolition/mottled/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**A viewer for latent dynamics: where a model's hidden states travel, turn
and pile up as a prompt moves through its layers.**

Every 2-D picture of a residual stream is a lie of some size. Mottled's pitch
is that it *measures the size of the lie* and prints it on the picture:
per-state neighborhood preservation, a bootstrap confidence field on the
terrain, and an amber ✕ on every state whose local structure did not survive
the flattening. It is an instrument for *generating* hypotheses about
representation geometry, not for establishing mechanism. The inferential
contract is [`docs/validity.md`](docs/validity.md).

![The web viewer scrubbing three GPT-2 runs from layer 0 to layer 12 across
one density terrain](docs/images/viewer-scrub.gif)

*Three GPT-2 prompts (the capitals of France, Germany and Italy) on one
density terrain, swept from layer 0 to 12. The runs start apart and end in
the same high-density region. The amber overlay is the density's standard
error: it marks where the terrain rests on too few points to trust.*

## Try it without installing anything

**[enactedvolition.github.io/mottled](https://enactedvolition.github.io/mottled/)**
is the WebGL viewer, with real captures, no build step and no dependencies.
Hover anywhere along a trajectory for the inspector; click to pin it.

| scene | what it shows |
|---|---|
| [`qwen-capitals`](https://enactedvolition.github.io/mottled/viewer/?file=samples/qwen-capitals.mtj) | Qwen2.5-1.5B, 29 layers × 1536 |
| [`gpt2-decode`](https://enactedvolition.github.io/mottled/viewer/?file=samples/gpt2-decode.mtj) | GPT-2 *generating*: the decode axis |
| [`models-qwen-gpt2`](https://enactedvolition.github.io/mottled/viewer/?file=samples/models-qwen-gpt2.mtj) | two different models on one terrain |
| [`gpt2-features`](https://enactedvolition.github.io/mottled/viewer/?file=samples/gpt2-features.mtj) | real SAE features, with their measured fit |
| [`self-portrait`](https://enactedvolition.github.io/mottled/viewer/?file=samples/self-portrait.mtj) | Mottled pointed at itself (see below) |
| [`blast/refusal`](https://enactedvolition.github.io/mottled/viewer/?file=samples/blast/refusal.mtj) | 48 requests, one pellet each: where harmful and harmless separate, layer by layer |
| [`blast/truth`](https://enactedvolition.github.io/mottled/viewer/?file=samples/blast/truth.mtj) | true vs false answers, beside the negation readout that rivals it |

The page can also **run a model itself**. Open *Run a model in this page*,
pick one, and capture. The picker lists models checked to load completely,
with the download size shown before the download starts, and a model you
have fetched once is served from the browser's cache after that.

| model | size | architecture |
|---|---|---|
| SmolLM2 360M | 386 MB | llama |
| Qwen3 0.6B | 639 MB | qwen3: GQA + per-head q/k norms |
| Qwen2.5 0.5B | 675 MB | qwen2: attention bias |

## Install

```bash
pip install "mottled[models] @ git+https://github.com/enactedvolition/mottled"

mottled                                    # the Streamlit explorer
mottled serve --model gpt2                 # web viewer + capture API
mottled export "The capital of France is" -o scene.mtj
mottled export "The residual stream" --generate 8 -o decode.mtj
mottled export "The capital of France is" --models gpt2,distilgpt2 -o models.mtj
mottled export-blast items.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --chat -o blast.mtj
mottled export-manifest scene.mtj          # what produced a scene, as JSON
mottled parity                             # Mottled's capture vs HF, TL, NNsight
mottled smoke                              # does this install actually work?
```

### On the command line, the stages compose

`export` is a convenience. Underneath it are stages that read stdin and
write stdout, so they pipe like any Unix tool. Stdout carries only data;
status goes to stderr, and only when you ask with `-v`:

```bash
mottled capture "The capital of France is" "The capital of Germany is" \
  | mottled project > scene.mtj                 # = mottled export ... -o scene.mtj
cat prompts.txt | mottled capture --model gpt2 > runs.mtj   # one prompt per line
mottled project runs.mtj --projection umap --set grid_size=96 -o umap.mtj
mottled inspect scene.mtj                       # one-screen summary
mottled inspect scene.mtj --ndjson | jq -c 'select(.layer == 12) | {text, entropy}'
mottled validate *.mtj                          # silent when valid, exit 1 when not
```

`capture` writes one trajectory container per prompt (a `.mtj` stream;
`cat a.mtj b.mtj` is a stream too) and `project` turns any stream into one
scene. `capture | project` produces exactly the scene `export` does. Exit
status is 0 on success, 1 when the work failed or an input is invalid, 2
for a usage error. Errors are one line on stderr (`--debug` or
`MOTTLED_DEBUG=1` for the traceback). Nothing is overwritten without
`-f`, and binary is never written to a terminal. `NO_COLOR` is honoured.
The disk cache lives in the per-user cache directory (`$MOTTLED_CACHE_DIR`
overrides it), never in the working directory.

Every capture is a real model; `gpt2` is the default because it is the
smallest honest one. The extras are `models` (torch and transformers, for
capture), `remote` (stream weights you do not hold), `umap`, `faiss`,
`tlens`, `sae` and `nnsight`. A viewer, an analysis or a `.mtj` consumer
needs none of them. From a clone:
`pip install -r requirements.txt && streamlit run ui.py`.

The explorer's **Chat** switch puts a conversation on the left and its
latest turn on the right: the conversation as the model was sent it, plus
the reply as the decode axis, so the scene is the forward pass that reply
came from. A model with a chat template is sent the ids its own chat
tokenization produces; one without (GPT-2) gets a plain transcript.
Earlier replies are re-sent as text, as any chat re-sends its history, so
their tokens can differ from the ones generated at the time. The panel
shows exactly what was sent.

## What it is, and what it isn't

This matters more than the feature list, so it comes first.

- **A basin is states accumulating, not a circuit computing.** It is a
  *state concentration region*: where this run's projected states pile up
  under the chosen projection and density estimator. A pattern that exists
  only in the projection is a pattern about the projection.
- **The logit lens is a readout diagnostic**: what the output head would say
  if pointed at an intermediate state, not the model's belief at that
  layer. Nearest tokens are *representation-space* neighbors; whether they
  are semantic neighbors is a hypothesis the display does not test.
- **Fidelity is stated inline.** Every scene reports how much of each
  state's neighborhood survived the projection. The density's bootstrap
  standard error is a *lower* bound, because the bootstrap treats dependent
  states as independent.
- **An edit shows sufficiency, not mechanism.** A steer that beats a
  norm-matched random control, or a dose-response curve that clears its
  controls, shows the push is enough to move the readout under the tested
  conditions. It does not identify how the model normally produces the
  behavior.
- **An SAE is only interpretable on the distribution it was trained on**,
  and feature names are leads: Neuronpedia's auto-interp explanations,
  written by a language model, describe what a feature correlates with, not
  what it computes.
- **The knobs are researcher degrees of freedom.** A picture found by
  turning them is a hypothesis; confirming it takes held-out prompts and a
  criterion fixed before looking.

For verified causal claims (circuit discovery, path patching, activation
patching at scale) use a dedicated tool:
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens),
[circuit-tracer](https://pypi.org/project/circuit-tracer/), ACDC, EAP.
Mottled is the map you read *before* and *alongside* those, and it
interoperates both ways: a `HookedTransformer`, an `ActivationCache` you
already ran, or states from NNsight or your own hooks all become a
`StateTrajectory`, and a SAELens dictionary loads directly.

## One finding worth stealing even if you never run this

Public GPT-2 SAEs are trained on **TransformerLens-processed** residuals. TL
folds LayerNorm and centres weights, which changes residual *values* while
preserving the model's function. So the same trained SAE reads **~24%**
reconstruction error on a TransformerLens capture and **~342%** on raw
HuggingFace hidden states: same model, same SAE, same prompt
([`docs/field-notes.md`](docs/field-notes.md), trap 1).

Provenance is not calibration. `sae.fit_report` measures reconstruction error
and firing density per layer, and the explorer labels a bad fit as
extrapolation instead of drawing confident features on top of it. On a good
fit the measurement finds the SAE's training hook on its own: best layer 8,
where it was trained.

## What you get

- **Two time axes.** Layers within a forward pass, and decode steps within a
  generation. `generate_and_capture` decodes, then captures the finished
  sequence in one pass, which causal attention makes exact.
- **Comparison.** Prompts on one joint projection (Hausdorff distance,
  dynamic time warping, the layer where their readouts part). Models too, in
  *readout space*: the vocabulary they share becomes a shared coordinate
  system, with the mass spent outside it kept visible. `layer_similarity`
  says which layer of B matches layer *l* of A, and whether that answer is
  identified or noise.
- **Interventions.** Perturb, set, noise or freeze a state and replay the
  forward pass from there. `faithfulness` scores a steer against a
  norm-matched random control; `dose.dose_sweep` runs one over a signed grid
  of doses, with random, shuffled-label and direct-path controls.
- **SAE features**, applied and never trained, with their measured fit, and a
  feature field that names its largest regions: hue stays each feature's
  identity, and the name says what the region is about.
- **Uncertainty everywhere**: neighborhood preservation per state, explained
  variance, and the density's bootstrap standard-error field.
- **Measured explanations.** `attractor.explain` turns a basin's numbers into
  prose in which every sentence comes from a measurement.
- **A methods section in every scene.** An exported `.mtj` carries its
  analysis record: the config it ran under, the environment, the model's hub
  commit, device and dtype, SAE hashes, and each edit (a vector edit with
  its norm and a hash of its values). It is the parameterization a reproduction needs, not
  evidence that the run reproduces.

## Many prompts at once: the shotgun blast

A token-trajectory scene follows one prompt's tokens through one global
projection, and on real captures that projection spends its axes on what you
already know: which token, which layer. `mottled export-blast` takes a JSONL
file of labelled prompts instead and makes each prompt one **pellet**, read
where a deployed monitor would read it (the state before the model writes),
with every layer framed around the family's centroid. Prompts that end on the
same template token start as one point, so the picture is the spread.

```json
{"id": "h00", "text": "Explain how to …", "labels": {"harmful": 1, "refused": 1}}
{"id": "q00t", "messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}],
 "labels": {"false_claim": 0}, "group": "q00"}
```

The **monitor** layout puts on x the score a linear monitor for one label
would give each pellet, cross-fitted so the split you see is held out, with
its AUROC per layer next to a label-shuffle null band; the viewer toggles
between a monitor per label and an unsupervised **open** layout, colours by
any label, and rings the pellets whose labels disagree. A split is a readout
of these prompts, not a mechanism, and the samples in
[`viewer/samples/blast/`](viewer/samples/blast/README.md) show the confounds
to look for: wording that differs between the classes, contrast pairs (hold
them out together with `group`), and a surface feature, there negation, that
reads as well as the label you meant.

## Where the states come from

`StateTrajectory` (`trajectory.py`) is the only interchange: `hidden` is
`(L, T, D)`, layer 0 being the embedding stream. Producers emit one; analyses
and viewers consume one; neither knows about the other.

```
producers                         interchange                  viewers
─────────                         ───────────                  ───────
transformers capture (hooks)  ─┐                          ┌─ Streamlit explorer
Mamba (state-space)           ─┤                          ├─ WebGL viewer
TransformerLens               ─┼─► StateTrajectory ─► .mtj ─┤  (no dependencies)
states you already captured   ─┤                          └─ Jupyter (plain
API logprobs (degraded)       ─┤                             Plotly figures)
the browser's own forward pass─┤
streamed, for models too big  ─┘
```

- **Transformers.** `models/families.py` resolves Qwen, Llama, Mistral,
  Gemma, GPT-2 and NeoX layouts structurally. **Mamba** is the proof the
  abstraction is not transformer-shaped: block capture and the logit lens
  work unchanged, and the captures that do not apply to a state-space model
  (attention patterns, the attention/MLP split) refuse rather than return
  something plausible.
- **States you already captured.** `models.external.from_hidden_states`
  takes activations from anywhere (an NNsight trace, a vLLM hook, a run on
  another machine) and `models.hooked.from_cache` takes a TransformerLens
  cache. The readout is the same code path as a native capture, so the
  numbers match it rather than resemble it.
- **Closed models**, honestly bounded. `models/logprobs.py` turns a hosted
  API's top-k logprobs into a *degraded* trajectory: there is no residual
  stream, so depth is unavailable and the animated axis becomes decode time.
  The trajectory declares what it cannot see, and the explorer prints that
  as a banner.
- **The browser.** `viewer/model.js` runs a Llama/Qwen3-family forward pass
  that records the residual stream after every block, pinned against
  HuggingFace. `viewer/gguf.js` reads GGUF as published, including ternary
  builds, and refuses a file holding tensors it cannot place: a model that
  loads and silently skips a weight is a trajectory of a model that does not
  exist. WebGPU kernels accelerate the same op contract the CPU path defines.

### Models too big to hold

A forward pass with no gradients needs block *i*'s weights only while block
*i* runs. `stream.stream_capture` builds the model skeleton with no weights,
materialises each block immediately before it runs and releases it after, by
hooks around the model's own blocks, so the architecture stays
HuggingFace's. Peak memory is one block plus activations; a checkpoint that
stores each expert as its own tensor briefly holds a block's experts twice
while fusing them. The blocks are found by `models.families`, as for every
other producer, so GPT-2 and GPT-NeoX stream as the Llama layout does, and a
layout it cannot name is refused before a weight is read.

```python
from stream import stream_capture, stream_capture_batch

traj = stream_capture("/path/to/checkpoint", "The capital of France is")
traj = stream_capture("hf://org/model", "The capital of")     # a repo id or URL
many = stream_capture_batch(checkpoint, prompts)   # one pass, N trajectories
```

Point it at a repo id or URL and the weights never land on local disk in full
either: `remote.py` reads one layer's byte ranges out of the published
shards, writes them to a cache file, and deletes it once the block has been
read. Ranges rather than whole files, because shard boundaries are not layer
boundaries. Egress is then the binding cost, so `stream_capture_batch` puts
every prompt through each block while it is resident, and every remote
trajectory carries the bill (`meta.remote`: bytes fetched, requests made,
peak cache bytes).

A single-prompt streamed pass is bit-exact against an in-memory capture, and
the residency bound is asserted rather than assumed. A batched pass is a
tolerance, not a promise: it reshapes every matmul, so the last bits move by
an amount that depends on the machine. `capture(..., capture_routing=True)`
records which experts each token was routed to in a sparse-MoE model, and
refuses on a dense one. It also refuses a checkpoint that leaves any
parameter unloaded, in a block or outside one (only the common per-expert
gate/up/down layout is fused), and a host that answers a range request with
the wrong bytes: the whole file, or a range of the wrong length.

**None of this has been run at frontier scale.** The mechanism is proven on
small models; the open items are in [`ROADMAP.md`](ROADMAP.md) (M7).

## Programmatic API

```python
from capture import capture
from projection import project
from density import compute_density
from terrain import mesh

traj = capture("gpt2", "The capital of France is")      # StateTrajectory
coords, projector = project(traj.hidden, method="pca")
landscape = compute_density(coords, method="kde")
surface = mesh(landscape)
```

Everything downstream of `capture` is a pure function over the trajectory: no
torch, no transformer internals. The whole pipeline in one call, and its
Plotly figure:

```python
from config import MarbleConfig
from ui import render, run_pipeline

result = run_pipeline(MarbleConfig(), "The capital of France is")
fig = render(result["traj"], result["mesh"], result["trajectories"],
             result["fine_paths"])
```

`from ui import run_pipeline, run_scene, run_compare, run_intervention,
render` and the `attach_*` helpers are the flat API, kept working across
minor versions ([`RELEASING.md`](RELEASING.md)).

## Don't take this README's word for it

The tests prove the pipeline is self-consistent. A reader of a paper built on
Mottled has a different question: is the residual stream in the picture the
one the model actually computed?

```bash
mottled parity --models gpt2 -o parity.json --markdown parity.md
```

This runs one prompt through Mottled and through implementations a reviewer
already trusts, and reports the largest disagreement as a number: the
per-block states against HuggingFace's own `output_hidden_states`, the
deepest logit lens against the model's actual logits, NNsight, and
TransformerLens (only against `from_pretrained_no_processing`, since TL's
default processing moves the residual stream on purpose). It exits non-zero
when anything exceeds tolerance, and a skipped row is never counted as a
passed one.

## Models

Verified end to end, with the attention/MLP residual decomposition
reconciling exactly (`max |h[l+1] − (h[l] + attn + mlp)| = 0.0000`):
Qwen2.5-1.5B-Instruct (GQA, RoPE, SwiGLU, RMSNorm), GPT-2, DistilGPT-2,
Pythia-70m, and the in-browser backend. Llama-3.2 and Gemma-2 work the same
way but are licence-gated on the Hub, so they cannot back bundled samples or
offline CI.

Asked for the capital of France, Qwen answers `" Paris"` where GPT-2 says
`" the"`; [`models-qwen-gpt2.mtj`](viewer/samples/models-qwen-gpt2.mtj) puts
both on one terrain across a 29×1536 vs 13×768 divide.

### Self-portrait

[`self-portrait.mtj`](https://enactedvolition.github.io/mottled/viewer/?file=samples/self-portrait.mtj)
is Mottled pointed at itself: GPT-2 processing three of Mottled's own
self-descriptions (*"Mottled visualizes hidden-state evolution as
trajectories over a semantic manifold"*, *"the residual stream moves, turns,
and settles"*, *"StateTrajectory is the center of the project"*), captured
and rendered by Mottled. Given the second one, GPT-2's top continuation is
`" into"`. It finishes the thesis.

## Where things live

| | |
|---|---|
| `capture.py` | hooks on every block + the logit lens → `StateTrajectory` |
| `pipeline.py` · `render.py` · `ui.py` | the pipeline (pure), its Plotly figures, the Streamlit shell and flat API |
| `projection.py` · `density.py` · `terrain.py` | projection with per-state fidelity, density with its standard error, the terrain mesh |
| `compare.py` · `crossmodel.py` | comparison within a model, and across models in readout space |
| `intervene.py` · `dose.py` | edits and counterfactual replay; dose-response sweeps with controls |
| `sae.py` · `attractor.py` | SAE features and their fit; basin analysis as measured prose |
| `stream.py` · `remote.py` | capture for models too big to hold |
| `statefile.py` · `provenance.py` | the `.mtj` format and the analysis record it carries |
| `models/` | producers: model families, TransformerLens, external states, API logprobs |
| `viewer/` | the WebGL viewer and the in-browser capture stack, each JS file pinned to a Python reference by a conformance test |
| `mtjschema.py` · `cli.py` | the `.mtj` JSON Schemas and validator; the command-line stages |
| `design_tokens.py` · `codegen.py` | every colour and font; `python -m codegen` generates the viewer's `tokens.js` and `:root` CSS, the Streamlit theme and `docs/schema/` from Python, and a test fails on drift |

## Docs

- [`docs/validity.md`](docs/validity.md): the inferential contract, what each
  artifact licenses you to claim and the controls a research-grade claim
  needs on top
- [`docs/mtj-format.md`](docs/mtj-format.md): the `.mtj` interchange spec,
  with its generated JSON Schemas in [`docs/schema/`](docs/schema/)
- [`docs/field-notes.md`](docs/field-notes.md): a dated orientation to the
  interpretability landscape, and the traps this project has already paid for
- [`ROADMAP.md`](ROADMAP.md) · [`CHANGELOG.md`](CHANGELOG.md) ·
  [`RELEASING.md`](RELEASING.md) · [`CLAUDE.md`](CLAUDE.md) (working in the repo)

## Tests

```bash
pytest -m "not network" -q      # what CI runs, offline
node --test viewer/tests/       # the viewer; no npm install needed
pytest -m network               # the HuggingFace-hub tests CI skips
```

The offline suite includes torch mechanism tests on locally-built models:
hook captures pinned against HuggingFace's own `hidden_states`, and the
residual decomposition against the real attention and MLP outputs. CI has no
GPU, so the WebGPU kernels are checked by hand:
[`viewer/tests/parity.html`](viewer/tests/parity.html) runs both backends
over identical weights in a real browser. Re-run it after touching a kernel.

## Non-goals

No training or finetuning (SAEs are applied, never trained). No circuit
discovery, no distributed inference, no production auth. A single-machine
research tool.

## License

[Apache 2.0](LICENSE): permissive, with an explicit patent grant, matching
the ecosystem it interoperates with (TransformerLens, SAELens, nnsight). See
[`NOTICE`](NOTICE).
