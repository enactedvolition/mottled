# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -r requirements.txt        # requirements.lock pins the known-good closure
pytest -m "not network" -q             # what CI runs
node --test viewer/tests/              # viewer tests; no npm dependencies, none needed
```

Python 3.11+. `conftest.py` puts the repo root *and* `tests/` on `sys.path`, so
test modules import project modules and the shared fixture (`tests/tiny.py`) by
plain name from any invocation directory.

```bash
pytest tests/test_density.py                          # one file
pytest tests/test_density.py::test_grid_covers_points  # one test
pytest -m network                                     # the HF-hub downloads CI skips
```

Tests marked `network` download models from the HuggingFace hub and are
deselected in CI. Everything else, including the torch mechanism tests, runs
offline against tiny locally-built models.

Running the tool:

```bash
mottled                              # Streamlit explorer (ui.py)
mottled serve --model gpt2           # stdlib server: viewer + capture API
mottled capture "a" "b" | mottled project > s.mtj   # the stages, piped
mottled inspect s.mtj --ndjson       # per-state records; --json summary
mottled validate s.mtj               # schema + byte-layout check (exit 1 if bad)
mottled export "a prompt" -o s.mtj   # capture | project in one step
mottled export-blast items.jsonl     # one pellet per prompt -> blast scene
mottled export-manifest s.mtj        # the analysis record the scene carries
mottled parity                       # capture vs HF/TransformerLens/NNsight
mottled smoke                        # does this install work? (see RELEASING.md)
mottled export-weights gpt2          # write .mwt for in-browser inference
python -m http.server                # viewer alone at /viewer/ (static, no API)
python -m codegen                    # regenerate docs/schema/, viewer/tokens.js,
                                     # the style.css :root block, .streamlit/config.toml
```

Generated files are never edited by hand: change `mtjschema.py` or
`design_tokens.py` and rerun `python -m codegen` (`tests/test_codegen.py`
fails on drift). CLI rules (`cli.py`): stdout is data only, status goes to
stderr behind `-v`, errors are one line (`--debug` for a traceback), exit 1
runtime / 2 usage, never clobber without `-f`, never binary to a TTY.

## Architecture

### StateTrajectory is the waist of the hourglass

`trajectory.py` defines `StateTrajectory` — `hidden` is `(L, T, D)`, layer 0
being the embedding stream. Backends produce one; everything else is a pure
function over one. **Nothing above `trajectory.py` may reach into transformer
internals.** This is why `metrics.py`, `compare.py`, `attractor.py`,
`neighbors.py` and `sae.py` take trajectories and never models, and it is what
lets a browser forward pass, a TransformerLens hook (`models/hooked.py`), a
logprob-only backend (`models/logprobs.py`), or states captured in someone
else's process (`models/external.py`) substitute for `capture.py` without
touching anything downstream.

The pipeline is `capture -> project -> density -> terrain -> paths`, driven by
one `MarbleConfig` (`config.py`, hashable for `cache.py`). `pipeline.py` holds
it as pure functions with no Streamlit and no Plotly; `render.py` holds the
Plotly figures; `ui.py` is the Streamlit shell and re-exports both, so
`from ui import run_pipeline` is the documented flat API and must keep working.

`run_pipeline` is one prompt. `run_scene` joint-projects several prompts into
one space so their trajectories are comparable — runs on separate projections
are not. `run_intervention` runs the counterfactual pair.

### Two surfaces, one tool: every JS file has a Python reference

The viewer is dependency-free WebGL2 with no build step, and it reimplements a
lot of Python. A reimplementation is only worth having if it is the same
computation, so each pair is pinned by a conformance test. **If you change one
side of a pair, change the other and run its test.**

| Python reference | Browser port | Pinned by |
|---|---|---|
| `statefile.py` (`.mtj`) | `viewer/mtj.js` | `tests/test_mtj_conformance.py` |
| `bvh.py` | `viewer/bvh.js` | `tests/test_bvh_conformance.py` |
| `projection.py`/`density.py`/`terrain.py` | `viewer/scene.js` | `tests/test_scene_conformance.py` |
| HF `transformers` forward pass | `viewer/model.js` | `tests/test_model_conformance.py` |
| `mweights.py` (`.mwt`) | `viewer/weights.js` | `tests/test_weights_conformance.py` |
| the `gguf` package's dequantiser | `viewer/gguf.js` | `tests/test_gguf_conformance.py` |
| HF tokenizer | `viewer/tokenizer.js` | `tests/test_tokenizer_conformance.py` (network) |
| `design_tokens.py` | `.streamlit/config.toml`, `viewer/style.css` | `tests/test_tokens.py` |
| `projection.fidelity_summary` | `viewer/reading.js` | `tests/test_reading_conformance.py` |

`tests/test_browser_capture_e2e.py` runs the whole chain — export weights, load
in JS, tokenize, forward pass, logit lens, joint projection, density, terrain,
drape — and checks the result is something the renderer can draw.

These are the tests that catch the failures this codebase actually produces:
a quantised block unpacked *almost* right, or a tokenizer that near-misses,
does not crash. It silently changes what the model was asked, and every state,
basin and readout downstream becomes a picture of a different question.

### The CPU path is the source of truth; WebGPU is an accelerator

`viewer/model.js` exports `cpuOps`. `viewer/ops-webgpu.js` implements the same
four-op contract (matmul, rmsNorm, swiglu, add) in WGSL. `forward` is `async`
and awaits each op so one implementation serves both backends — keep it that
way.

**CI cannot verify the WGSL.** There is no GPU on the runner, and
`navigator.gpu` is undefined even under SwiftShader. `node --test` checks only
that each shader's bindings match what `dispatch()` supplies.
`viewer/tests/parity.html` is the real check: it runs both backends over
identical weights in a real browser and reports the largest disagreement.
It has been run on hardware and passed. **Re-run it after touching a kernel** —
nothing automated will catch an arithmetic regression there.

### Formats

`.mtj` (`docs/mtj-format.md`) is the interchange format: JSON manifest plus raw
little-endian buffers, glb-style, parseable from any language's standard
library. Two kinds — `trajectory` (full fidelity) and `scene` (projected,
draped, viewer-ready). The viewer renders only `scene`. `.mwt` is the same
container idea for weights; its header is padded to a 32-byte boundary so
typed-array views over the data section are aligned, which is load-bearing and
was once true only by luck.

## Conventions

**Inferential language is a correctness property here.** `docs/validity.md` is
the project's inferential contract, and its vocabulary is deliberate: a basin is
a **state concentration region**, not an attractor; nearest tokens are
**representation-space neighbors**, not semantic ones; the logit lens is a
**readout diagnostic**, not the model's belief; an intervention shows
**sufficiency**, not mechanism; `density_se` is a **lower bound**, because the
bootstrap treats dependent states as independent. `attractor.py` keeps its name
as descriptive geometry. Don't reintroduce stronger claims in docs, UI strings,
or comments — the main research-validity risk of this tool is an attractive
within-run picture being read as evidence of a model mechanism.

**Design tokens have one home.** `design_tokens.py` owns every color and font;
`.streamlit/config.toml` and `viewer/style.css` mirror it and `test_tokens.py`
fails on drift. Add a CSS variable to `viewer/style.css` only if it exists
there.

**GitHub Actions are pinned to commit SHAs** with the version in a trailing
comment, and workflows declare `permissions:` explicitly rather than inheriting
a repository default that isn't part of the file's review surface.

**Comments explain why, not what.** The existing code is written that way — it
names the failure mode a line prevents, the alternative that was tried, or the
reason a value is what it is. Match that density rather than annotating syntax.

## Workflow

Land pull requests as soon as CI is green rather than leaving them open for
their own sake. Hold only for a specific reason: unexplained red or flaky CI, a
merge conflict where both sides changed the same logic, a genuinely pending
human review, or a change that reaches outside the repo — a release, a publish,
a Pages deploy that can't be verified first.

Auto-merge is the intent, but the `enable_pr_auto_merge` tool available in
Claude Code sessions refuses this repo's PRs in *both* directions, so in
practice there is no window for it:

- while the one CI check is still running it reports the PR `unstable`, worded
  as "required checks are failing" even though nothing has failed;
- once that check passes it reports the PR already `clean` and says to merge
  directly.

So squash-merge directly once CI goes green. (Observed on #33 and #34. GitHub's
own auto-merge does accept a pending PR — this is a limitation of the tool, not
of the repository, and it may stop being true.)
