# The `.mtj` format — Mottled Trajectory interchange, version 1

`StateTrajectory` is the center of Mottled: producers (transformer capture,
the browser's own forward pass, future diffusion / neuroscience recorders) emit
one, and viewers (the Streamlit app, the web viewer, notebooks) consume one.
`.mtj` is the stable on-disk form of that abstraction, designed to be
trivially parseable from any language — a JSON manifest plus raw
little-endian typed-array buffers, in the spirit of glTF's `.glb`.

## Container layout

| bytes | content |
|---|---|
| 0–3 | magic `MTRJ` (ASCII) |
| 4–7 | `uint32` (LE) format version, currently `1` |
| 8–11 | `uint32` (LE) manifest length in bytes (`M`) |
| 12 – 12+M | manifest: UTF-8 JSON, space-padded so `12+M` is a multiple of 16 |
| 12+M – EOF | binary blob: raw array data, each array 16-byte aligned |

Array `offset`s in the manifest are relative to the **start of the blob**
(byte `12+M`), not the file. All numeric data is little-endian.

A JavaScript reader needs only `DataView` + `TypedArray`; a Python reader
needs only `struct` + `numpy.frombuffer`. No compression, no dependencies.

## Manifest

```jsonc
{
  "format": "mottled-trajectory",
  "version": 1,
  "kind": "trajectory",              // or "scene"
  "meta": { "model": "gpt2", "prompt": "…", "backend": "transformers", … },
  "arrays": {                        // name -> array reference
    "hidden": { "dtype": "float32", "shape": [13, 6, 64], "offset": 0, "length": 199680 }
  },
  …kind-specific fields…
}
```

Array references always carry `dtype` (`float16` | `float32` | `int32`),
`shape`, `offset`, and `length` (bytes; equals the product of the shape and
the item size). Readers MUST ignore unknown manifest fields and unknown
arrays — that is how the format stays stable while growing.

## `analysis` — the record of what produced the file *(optional)*

Both kinds may carry an `analysis` object: the full parameterization of the
run, written by `provenance.record` (Python) via `pipeline.attach_manifest`.
`docs/validity.md` asks anyone publishing on Mottled output to version-lock
the model, tokenizer, library versions, precision, seeds and SAE artifact
hashes; this is where the file states them itself.

```jsonc
{
  "schema": "mottled-analysis/1",
  "created": "2026-09-15T21:40:00Z",   // UTC, when the record was made
  "mottled": "0.2.0",                  // null if the version is unknowable
  "config": { "projection": "pca", "seed": 0, … },  // every MarbleConfig field
  "prompts": ["The capital of France is"],
  "models": [                          // one entry per run, in run order
    { "id": "gpt2", "revision": "e7da7f2",  // hub commit, null if not from one
      "backend": "transformers", "family": "gpt2",
      "device": "cpu", "dtype": "float32" }
  ],
  "environment": {
    "python": "3.11.9", "platform": "Linux-6.8.0-x86_64",
    "packages": { "numpy": "2.1.0", "torch": "2.4.0", … }
  },
  "sae": {                             // only when a dictionary was applied
    "source": "jbloom/GPT2-Small-SAEs-Reformatted",
    "hook": "blocks.8.hook_resid_pre",
    "sha256": "…",                     // hash of the weights, not of a file
    "n_features": 24576
  },
  "interventions": [                   // only when a run was edited: one list
    [],                                // per run, in run order; the untouched
    [{ "kind": "perturb", "layer": 11, // baseline's is empty
       "token": -1, "norm": 30.0,
       "sha256": "…" }]                // hash of the vector's float32 values
  ]
}
```

`models` is a list because a cross-model scene has one identity per run;
`interventions` is per run for the same reason. Each edit records its
`kind`, `layer` and `token` (`null` for every position). A vector edit
(`perturb`, `set`) adds the vector's `norm` and a `sha256` of its values, and
`noise` adds its `scale` and `seed`, so two strengths or two directions of
the same edit read differently. `config` is the
configuration the runs were captured under, which is not always the one the
session asked for: an intervention replays the prompt pass only, so its
record says `generate_tokens: 0` even with generation on.
The record describes the parameterization, not the result: it is what a
reproduction attempt needs, not evidence that the run reproduces.

## `kind: "trajectory"` — one full StateTrajectory

Round-trips a `StateTrajectory` at full fidelity. Fields:

- `tokens`: list of token strings, length `T`.
- `vocab` *(optional)*: list of vocabulary strings, length `V`.
- `topk` *(optional)*: `[L][T][k]` list of `[token, probability]` pairs.
- Arrays (all optional except `hidden`):
  - `hidden` — `(L, T, D)` float32,
  - `logits` — `(L, T, V)` float16,
  - `entropy` — `(L, T)` float32,
  - `attention` — `(L-1, T, T)` float32,
  - `components.attn`, `components.mlp` — `(L-1, T, D)` float32,
  - `embedding_matrix` — `(V, D)` float32.

## `kind: "scene"` — a viewer-ready bundle

What a viewer needs to draw and inspect, with the analysis already done in
Python (projection, terrain, draping, comparisons). No hidden states are
required, so scene files are small enough for the web.

Fields:

- `terrain`: `{ "x": <array ref name>, "y": …, "z": … }` — grid axes
  `(W,)`, `(H,)` and heights `(H, W)`; `z[i][j]` is the height at
  `(x[j], y[i])`. Two optional keys carry the density and its uncertainty:
  - `density` — `(H, W)` float32 *(optional)*: the normalised `[0, 1]`
    density field the height map was built from.
  - `se` — `(H, W)` float32 *(optional)*: the per-cell bootstrap standard
    error of that density, in the same normalised units — the confidence
    field. Present only when the producer ran the density bootstrap.
- `runs`: list of runs drawn on that terrain. Each run:
  - `label`: display name (`"A"`, `"B"`, …),
  - `prompt`: the prompt text,
  - `tokens`: token strings, length `T`,
  - `trajectory_labels`: display label per drawn trajectory,
  - arrays (referenced by name from `arrays`):
    - `points` — `(N, L, 3)` float32 draped trajectory points
      (`N` trajectories × `L` layers × xyz). **Required.**
    - `entropy` — `(L, T)` float32 *(optional)*,
    - `quality` — `(L, T)` float32 *(optional)*: per-state projection
      fidelity — the fraction of each state's hidden-space nearest
      neighbors preserved in the 2-D projection (`1.0` = local structure
      intact). Tells a viewer where the flattened picture is trustworthy.
    - `attention` — `(L-1, T, T)` float32 *(optional)*,
  - `topk` *(optional)*: `[L][T][k]` list of `[token, probability]` pairs.
  - `generation` *(optional)*: the decode record when the run covers
    prompt + generated continuation: `{ "prompt_tokens", "new_tokens",
    "mode" ("greedy"|"sample"), "temperature", "seed", "steps": [{ "token",
    "id", "p", "entropy" }, …] }`. Token indices `>= prompt_tokens` were
    generated.
  - `inspector` *(optional)*: readouts a viewer cannot recompute, because
    the inputs (the `(V, D)` embedding matrix, the `2 x (L-1) x T x D`
    residual components) are far larger than the scene itself, so they are
    resolved before export: `{ "tokens": [str, …] (a compact table of only
    the strings that appear), "idx" -> `(L, T, k)` int32 into that table,
    "sim" -> `(L, T, k)` float32 cosine similarity (nearest first),
    "component_shares" -> `(L-1, T, 2)` float32 attention/MLP share of each
    block's residual write, each pair summing to 1 }`. Any member may be
    absent independently. Note the block axis: `component_shares[b]`
    describes the write producing layer `b+1`, so layer 0 has none.
  - `features` *(optional)*: an SAE feature layer with its **measured fit**
    attached: `{ "source", "hook", "best_layer", "recon_error" → `(L,)`
    float32 array ref, "top_id" → `(L, T)` int32 array ref (dominant
    feature per state), "top_act" → `(L, T)` float32 array ref }`.
    `recon_error[l]` is the dictionary's median relative reconstruction
    error at layer `l`; values ≫ `0.5` mean the dictionary does not fit
    those activations and features there are extrapolation — viewers
    should say so rather than display them as measurement.
- `comparisons` *(optional)*: for runs beyond the first, JSON summaries:
  `{ "label", "hausdorff", "dtw_normalized", "shared_tokens", "onset_layer",
     "readout_changed" }`.
- `blast` *(optional)*: a **pellet family** (`blast.py`, `mottled
  export-blast`): one pellet per prompt, read at one position, at every
  layer. The scene keeps every required field, so a reader without this
  record still draws something true: the terrain is flat (`2 x 2`, z = 0)
  and there is one run whose `N` trajectories are the `N` pellets drawn at
  the first layout, its `tokens` the pellet ids and its per-(layer, token)
  arrays (`entropy`, `topk`, `quality`) per (layer, pellet). Each pellet's
  `{ "id", "text", "labels": {name: 0 | 1} }` is in `meta.pellets`, in run
  order. The record:
  `{ "schema": "mottled-blast/1", "range" → (N, L), "spread" → (L,),
     "norm" → (L,), "skipped": [str, …], "layouts": [ … ] }`.
  Positions are in units of each layer's mean state norm (`norm`), around
  that layer's centroid, so each layer has its own frame. `range[i, l]` is
  pellet `i`'s exact full-space distance from the centroid in that unit,
  and `spread[l]` the family's RMS range. Each layout is
  `{ "name", "method": "monitor" | "open", "driver": label name or null,
     "positions" → (N, L, 2) float32, "quality" → (L, N) float32 k = 5
     neighbourhood preservation per layer, "exact" / "fitted" / "projected":
     [str, …] what each part of the picture is, "arrays": {name → ref},
     "params": {…} }`. `monitor` arrays: `auroc`, `null05`, `null95` (L,)
  — the cross-fitted readout AUROC of the driver label per layer and the
  5th / 95th percentiles of the same statistic under label shuffles (NaN
  where the layer is one point) — and `labelled` (N,) int32, 1 where the
  pellet's driver label was used. `open` arrays: `shown` (L,), the share of
  each layer's spread the camera keeps, and `mass` (N,), each pellet's
  weight in choosing it; `params.copies` / `params.near_copies` list the
  pellet pairs that shared a vote, and the pairs within 10% of that cut.
  `skipped` says why a label could not drive a monitor.

Array names for run `i` are conventionally prefixed `run{i}.` (e.g.
`run0.points`) but viewers MUST resolve them through the manifest references,
never by naming convention.

Viewers are expected to densify trajectory polylines themselves (e.g.
Catmull-Rom) for smooth animation — fine paths are derived data and are not
stored.

## Versioning

- The `version` header and manifest `version` change **only** on breaking
  layout changes.
- Additive evolution (new optional arrays, new manifest fields) does not
  bump the version; readers ignore what they don't know.
- Writers MUST NOT change the meaning of existing fields within a version.
