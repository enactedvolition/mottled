# Mottled web viewer

A self-contained WebGL2 viewer for `.mtj` **scene** files — no dependencies,
no build step, no network access beyond fetching the file itself.

## Run

From the repo root:

```sh
python -m http.server 8000
```

then open <http://localhost:8000/viewer/>. On startup the viewer tries
`samples/scene-abc.mtj`; if that is missing it falls back to a drop prompt.

Loading a file:

- drag a `.mtj` anywhere onto the page, or
- click **Open .mtj…**, or
- pass a page-relative URL: `http://localhost:8000/viewer/?file=samples/single.mtj`.

Files of `kind: "trajectory"` are recognized but not rendered — the viewer
asks you to export a `kind: "scene"` file instead (scenes carry the projected
terrain and draped trajectory points; raw trajectories don't).

## Controls

| input | action |
|---|---|
| left-drag | orbit |
| right-drag or shift-drag | pan |
| wheel | zoom |
| hover anywhere along a trajectory | highlight + inspector (run, layer — fractional between layers, token, entropy, neighborhood fidelity, features, top-k readout, nearest vocabulary tokens by cosine, and the attn/MLP split of the block's residual write) |
| click a trajectory | pin the inspector to that reading (Escape or a click on empty space clears it) |
| slider / play button | scrub or animate the marbles across layers |
| runs panel checkboxes | show / hide individual runs |
| attention flow toggle | draw top-3 attention edges (weight ≥ 0.1) at the current layer |
| uncertainty toggle | recolor the terrain by the density's bootstrap standard error (amber = less certain); appears only when the scene carries an `se` field |

The uncertainty controls surface two honesty signals the format now carries.
The **uncertainty toggle** washes the terrain toward amber where the density
estimate is least stable across bootstrap resamples — the height there is
bandwidth artifact more than measurement. The inspector's **nbhd preserved**
line reports how much of a state's hidden-space neighborhood survived the 2-D
projection at the point you hover: a low percentage means the flattened
position is not to be trusted.

Run A is drawn solid; runs B, C, … get dash patterns and reduced opacity,
with labels prefixed `B · token` etc. Comparison summaries (`hausdorff`,
`dtw_normalized`, `shared_tokens`) appear under the runs list when present.

## What the viewer expects from the format

Everything is per `docs/mtj-format.md`, version 1:

- container: `MTRJ` magic, u32 LE version `1`, u32 LE manifest length,
  UTF-8 JSON manifest padded so the blob starts 16-byte aligned, then raw
  little-endian arrays at 16-byte-aligned offsets relative to the blob start;
- manifest `kind: "scene"` with `terrain.{x,y,z}` array references
  (`(W,)`, `(H,)`, `(H, W)`; `z[i][j]` is the height at `(x[j], y[i])`) plus
  optional `terrain.{density,se}` `(H, W)` uncertainty layers;
- `runs[]`, each with a required `points` array `(N, L, 3)` and optional
  `entropy` `(L, T)`, `quality` `(L, T)` (projection fidelity),
  `attention` `(L-1, T, T)`, and manifest `topk`
  `[L][T][k]` of `[token, prob]` pairs;
- array references are resolved strictly through `manifest.arrays` — never by
  the `run{i}.` naming convention;
- unknown manifest fields and unknown dtypes are ignored (forward
  compatibility); supported dtypes are `float32`, `int32`, `float16`
  (decoded to `Float32Array`).

Trajectory polylines are densified in the viewer with Catmull-Rom splines
(8 segments per layer span) for smooth lines and marble animation, as the
spec prescribes — fine paths are not stored in the file.

## Blast scenes

A scene carrying a `blast` record (`mottled export-blast`, see `blast.py` and
`docs/mtj-format.md`) is one pellet per prompt rather than one trajectory per
token, and the viewer reads it differently:

- **Layout toggle.** The pellet panel (where the runs list would be) lists the
  record's layouts by name, `monitor · <label>` for each label that could
  drive a monitor and `open`; the first is drawn on load. Switching redraws
  the pellets at that layout's positions and reframes the view from above.
- **Colour by.** Pellets are coloured by one label: amber where it is 1,
  accent blue where it is 0, muted where the pellet has no such label (the
  legend prints the values as the file writes them, with counts). A monitor
  colours by its own driver until you pick a label, and a pick sticks across
  layouts.
- **Rings.** Colouring by a label other than the layout's driver rings the
  pellets whose two labels break the pairing most pellets follow — where the
  labels mostly agree, the ones where they differ, and where they mostly
  differ, the ones where they match. The legend says which (`ringed:
  refused ≠ harmful`, `ringed: negation = false_claim`).
- **Readout at the scrubbed layer.** For a monitor, the held-out AUROC beside
  the label-shuffle null's 5–95% band, flagged amber when it falls inside or
  below the band; for `open`, the share of the layer's spread the camera
  keeps. **What the axes are** prints the layout's own exact / fitted /
  projected lines from the file, and that each layer has its own frame.
- **Inspector.** A pellet's id, text (long text keeps both ends: the state is
  read by default at its last position), labels and exact distance from the layer's
  centroid, above the usual entropy, fidelity and top-k readout.

The scene's flat terrain is not drawn, and the reading panel drops the
terrain and density-uncertainty notes, since a blast estimates no density.
The colouring, rings and readout text are pure helpers in `mtj.js`
(`blastLegend`, `blastRings`, `blastReadout`, …), pinned by
`tests/blast.test.js`.

## Files

- `mtj.js` — parser (`MTJ.parse`, `MTJ.loadScene`); works in the browser and
  under plain Node for testing.
- `main.js`, `index.html`, `style.css` — the viewer app.
- `tests/parser.test.js` — run with `node --test viewer/tests/` from the
  repo root.
