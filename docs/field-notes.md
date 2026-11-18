# Field notes — the interpretability landscape, dated

**Verified 2026-11-16** (previous check 2026-08-05). Orientation for anyone — human or agent — picking
this repo up cold. The field moves fast enough that remembered facts go stale
within a year, so:

**Rule of the file: every claim is either (a) dated and re-verifiable with the
commands at the bottom, or (b) marked secondary and cited.** If a claim has
neither, delete it rather than trust it.

---

## What the field is trying to do

Reverse-engineer what a network *computes*, not what it correlates with. The
working ontology, roughly in order of how settled it is:

| idea | what it claims | how settled |
|---|---|---|
| **Residual stream as a highway** | Blocks read from and write to a shared additive stream | Solid — and mechanically checkable (see traps) |
| **Logit lens** | Decoding an intermediate state with the output head shows what the model "would say" there | Useful, known to mislead in early layers |
| **Superposition** | Models pack more features than dimensions, so single neurons are polysemantic | Widely accepted framing |
| **Linear features** | Concepts are directions, recoverable by dictionary learning (SAEs) | **Contested — see below** |
| **Circuits** | Behaviour decomposes into identifiable subgraphs | Real, but expensive to establish per-behaviour |
| **Workspace lenses** | A state's corpus-averaged first-order effect on the last state (the Jacobian lens) reads what it is disposed to make the model say | New in 2026, changing monthly, evaluated mostly by its authors — see below |

The honest summary: **instrumentation is solid, decomposition is contested,
explanation is weakest.**

## The stack (verified versions 2026-11-16)

| library | version (released) | use it when |
|---|---|---|
| `transformer-lens` | 4.4.0 (2026-11-12) | Standard mech-interp on GPT-style models; activation patching, attention analysis |
| `sae-lens` | 6.55.5 (2026-11-13) | Training/loading SAEs. Inference works with plain PyTorch, not just TL |
| `nnsight` | 0.8.2 (2026-11-02) | Non-transformer architectures, or remote execution on models too big to host (NDIF) |
| `circuit-tracer` | 0.5.0 (2026-03-29, unchanged since the last check) | Attribution graphs / transcoders — the open implementation of Anthropic's circuit tracing |
| `pyvene` | 0.1.8 (2025-05-26, unchanged) | Intervention-focused, architecture-agnostic |
| `transformers` | 5.23.0 (2026-11-13) | The substrate everything else wraps |

**TransformerLens changed shape.** 3.0 (2026-04-17) introduced
`TransformerBridge`, which wraps the HF model behind an architecture
adapter instead of converting it into TL's own classes, and whose default
does no weight processing (no folding, no centring). 4.0 removed
`HookedTransformer`: in the 4.4.0 wheel, `from transformer_lens import
HookedTransformer` raises a message pointing at
`TransformerBridge.boot_transformers`. The bridge is close to HF's numerics
but not bit-identical: measured 2026-11-18 on 4.4.0, its residual stream
differs from HF's `hidden_states` by up to 6.1e-5 absolute on GPT-2
(2.5e-7 relative) and 1.3e-5 relative on Pythia-160m, where Mottled's own
capture agrees with HF exactly. Mottled's `tlens` extra requires `>=4`, and
the parity cell and `models/hooked.py` run on the bridge (until 2026-11-18
they were pinned `<4` on `HookedTransformer`).

Mottled deliberately depends on **none** of these at runtime. It reads the
residual stream through plain HF forward hooks and treats TL/SAELens as
optional adapters. That is a design choice about not inheriting anyone's
preprocessing (see trap #1) — not a claim they are unnecessary.

## Public dictionaries that actually exist

Checked on the Hub, 2026-11-16. **`gated` refers to the dictionary repo, not
the base model. They differ, and that catches people.** The base-model column
is checked too.

| suite | models | kind | gated (dict / base) | last updated |
|---|---|---|---|---|
| `Qwen/SAE-Res-Qwen3.5-{2B-Base,9B-Base,27B,35B-A3B-Base}-*` | Qwen3.5 2B–35B-A3B | residual SAEs | no / no | 2026-05-13 |
| `Qwen/SAE-Res-Qwen3-{1.7B,8B,30B-A3B}-Base-*` | Qwen3 1.7B–30B-A3B | residual SAEs | no / no | 2026-05-13 |
| `google/gemma-scope-2-{270m,1b,4b,12b,27b}-{pt,it}` | Gemma 3 270M–27B | SAEs (resid, attn, MLP), transcoders, crosscoders, CLTs | no / **manual** | 2026-01 |
| `mwhanna/qwen3-{0.6b,1.7b,4b,8b,14b}-transcoders*` | Qwen3 0.6B–14B | transcoders for `circuit-tracer` | no / no | 2026-10-28 |
| `OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B` | Qwen3 1.7B | transcoders + attention (CRM) | no / no | 2026-04-02 |
| `google/gemma-scope-2b-pt-res` | Gemma-2 2B | residual SAEs | no / manual | 2025-01-19 |
| `EleutherAI/sae-llama-3.1-8b-64x` | Llama-3.1 8B | residual SAEs | no / manual | 2025-07-22 |
| `jbloom/GPT2-Small-SAEs-Reformatted` | GPT-2 small | residual SAEs | no / no | 2024-08-02 |

**Correction to the 2026-08-05 table.** It listed Qwen3 and Gemma Scope 1
only. Qwen's Qwen3.5 SAEs (created 2026-04-27) and Gemma Scope 2 (released
2025-12-19) both existed then and were missed, so "Qwen3-1.7B-Base is the
only fully-ungated modern feature path" was already wrong. **Qwen remains the
fully-ungated route**, now through 27B dense and 35B-A3B MoE. Gemma Scope 2
is the most complete suite (every layer, every size, PT and IT, plus
multi-layer dictionaries), but Gemma 3 base weights are gated `manual`, so it
cannot back bundled samples or offline CI. Gemma Scope 2 was trained with a
Matryoshka loss and a frequency penalty *(secondary: the release post)*, so
its latents are not drop-in comparable with Gemma Scope 1's.

## The shape of the frontier gap

It is not one gap, and calling it a "chasm" flattens four very different
things:

1. **Instrumentation — no gap.** Hooks, logit lens, residual decomposition
   are architecture-agnostic. This repo's decomposition reconciles to
   `0.0000` on Qwen2.5-1.5B (GQA/RoPE/SwiGLU/RMSNorm). Scale costs money,
   not method.
2. **Trained dictionaries: about 1.5 orders of magnitude, and closing.**
   Public SAEs now reach 27B dense and 35B-A3B MoE (Qwen's, table above),
   and a 70B transcoder set exists (`FraunhoferHHI/llama-3.3-70b-instruct-transcoders`,
   2026-11-04, not evaluated here). The 2026-08-05 figure was "~8–9B
   publicly". Against ~1T frontier (Kimi K2's weights are 1,029 GB,
   measured), the gap is still a generation or so behind.
3. **Explanations — the widest gap, least discussed.** Neuronpedia's GPT-2
   labels were written by **gpt-3.5-turbo** (verified in the API response's
   `explanationModelName`). Auto-interp quality is capped by the explainer,
   so the semantic layer runs on models two generations stale. A different
   route opened in 2026: **natural language autoencoders** train the model
   to put its own activations into words, with no dictionary in between
   *(secondary: Anthropic, transformer-circuits.pub/2026/nla, 2026-05-07)*.
   Neuronpedia hosts them. They are explanations by a model about itself,
   with the same correlate-not-computation caveat as any label.
4. **Closed weights — a wall, not a gap.** No residual stream at any price.
   This is what `models/logprobs.py` exists for: a deliberately degraded
   producer that declares what it cannot see.

Also worth separating: *"the public ecosystem lags frontier"* is a different
claim from *"interpretability lags frontier."* Labs run frontier-scale
interpretability internally; what's thin is the **shared artifacts**.

## Contested ground — do not present as settled

SAEs are the field's dominant decomposition tool and are under real, current
criticism *(secondary — from the literature, not verified here)*:

- **Incomplete coverage.** A fixed latent budget captures high-frequency
  patterns and starves fine-grained ones; even the largest SAEs give an
  incomplete description of a model's representations.
- **Structured residual error.** Much SAE error is *linearly predictable*
  from the input activations — i.e. unlearned features, not noise — and
  reinserting reconstructions causes substantial substitution loss.
- **Low steerability.** Many latents are neither interpretable nor
  steerable; user-desired concepts are often simply absent.
- **Circuit tracing is partial.** Anthropic's attribution-graph work on
  Claude 3.5 Haiku reportedly gave satisfying insight on roughly a *quarter*
  of tested prompts.
- **The biggest SAE producer stepped back from SAE research.** Google
  DeepMind's interp team says in the Gemma Scope 2 announcement that it has
  deprioritised fundamental research on SAEs. It still released the largest
  open suite to date.
- **Dictionaries may not be needed for sparse circuits.** Arora et al.
  (2026) report circuits that are sparse in the plain MLP *neuron* basis,
  traced with gradient attribution and no learned dictionary.
- **Workspace lenses are evaluated mostly by their authors.** The J-lens,
  R-lens and J++ numbers below come from the teams that built them, on
  their own evals.

## Workspace lenses (new since the last check)

A line of readouts that came out of one paper in 2026 and has moved monthly
since. Mottled carries two of the three. All are **readout diagnostics**
under `docs/validity.md`, the same as the raw and tuned lens. Their causal
claims (patching a lens direction flips an answer) are the authors' claims
*(secondary, cited)*.

| lens | what changes | reported | in Mottled |
|---|---|---|---|
| **J-lens** (Gurnee et al. 2026, transformer-circuits.pub/2026/workspace) | Read h as head(norm(J h)), J = E[∂h_final/∂h_l] over a corpus | Coherent readouts mostly in the later half of layers | `experiments/jacobian_lens.py` |
| **R-lens** (LessWrong, 2026) | The same estimator with LRP rules in the backward: frozen RMSNorm factor, identity rule on the gate activation, half rule on gate × up | Earlier, cleaner readouts; the advantage grows with scale and is **absent on the smallest models tested**; R/J lenses published at `camilablank/workspace-lenses` | `--backward lrp` (unmeasured here; see below) |
| **J++** (Ayonrinde & Lindsey; Anthropic circuits update, October 2026) | R-lens rules, plus per-cluster Jacobians combined with learned weights that mostly drop clusters giving poor readouts, plus filtering of non-semantic readout tokens | 55.2% recall@10 on latent-variable extraction vs 35.7% (J) and 37.7% (R), across 9–284B models | Not implemented: the weighting method was not readable in what this check could fetch, and guessing it would be a different lens under J++'s name |

Mottled's own results with the J-lens, both on Qwen2.5-1.5B: the registered
`introspection_workspace` run did not support H1 (`experiments/README.md`).
Mottled has no R-lens measurement. On a 1.5B model the post's own scaling
result predicts little difference, so a null there would say little about
the method. A registered comparison should name its model with that in mind.

Mech interp was named one of MIT Technology Review's 10 Breakthrough
Technologies for 2026 — the field is ascendant *and* internally unsure. Both
are true; say both.

## Traps this project already hit

Hard-won and specific. Each cost real debugging time.

1. **Preprocessing is not provenance.** A trained SAE fits the activation
   distribution it was trained on — including the *preprocessing*.
   TransformerLens folds LayerNorm and centres weights, which changes
   residual *values* while preserving the model's function. The same
   jbloom SAE reads **~24% reconstruction error on a TL capture and ~342% on
   raw HF states.** Never trust an SAE because it names your model —
   measure the fit (`sae.fit_report`).
2. **CKA saturates on residual streams.** A few very-high-variance
   dimensions shared by every layer dominate the inner products, so raw CKA
   scores ~1.0 for *every* layer pair and its argmax is noise (measured:
   middle rows flat to within 0.001 on GPT-2 vs DistilGPT-2). Z-score per
   dimension first — but know the cost: exact rotation invariance is lost
   (~0.96 for a rotated copy).
3. **Auto-interp labels describe correlates, not computation.** Carry
   `explained_by` with the text. A feature name is a lead.
4. **Token *count* is not token *alignment*.** GPT-2 and Pythia-70m both
   split `"The capital of France is"` into 5 pieces with different
   tokenizers. Pair states on the token **strings**.
5. **`np.eye(V)` is a memory bomb.** Representing "each axis is a vocabulary
   entry" as an identity matrix is 7 GB at V=42k. It killed a process here.
6. **Entropy from truncated top-k is a lower bound**, never the true
   predictive entropy. Label it.
7. **A lens directory is not necessarily a tuned lens.** The viewer named
   any non-logit `config.lens` "tuned lens", so a Jacobian-lens scene
   claimed a KL-trained fit it never had. The run's recorded `lens.kind`
   decides now (`viewer/reading.js`, mirroring `tuned_lens.readout_name`).

## Where Mottled sits

Not a circuit-discovery tool and does not claim to be. It instruments
**latent dynamics** — where states travel and pile up — and measures how much
of that survives projection to 2-D. For verified causal claims, reach for
`circuit-tracer` (which now has Qwen3 transcoders, so it runs on fully
ungated weights), TransformerLens, ACDC/EAP. The design bet is that an honest
map you read *before* those tools is worth having, and that a visualization
which cannot state its own error is worse than none.

## Re-verify (run this before trusting the tables above)

```bash
python - <<'PY'
import json, urllib.request
from huggingface_hub import model_info
pypi = lambda n: json.load(urllib.request.urlopen(
    f'https://pypi.org/pypi/{n}/json', timeout=20))['info']['version']
for p in ['transformer-lens','sae-lens','nnsight','circuit-tracer','pyvene','transformers']:
    print(f"{p:20s} {pypi(p)}")
for r in ['Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_50','Qwen/Qwen3.5-2B-Base',
          'Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50','google/gemma-scope-2-1b-pt',
          'google/gemma-3-1b-pt','mwhanna/qwen3-1.7b-transcoders-lowl0',
          'google/gemma-scope-2b-pt-res','EleutherAI/sae-llama-3.1-8b-64x',
          'jbloom/GPT2-Small-SAEs-Reformatted','camilablank/workspace-lenses']:
    i = model_info(r); print(f"{r:48s} gated={i.gated} {str(i.last_modified)[:10]}")
PY
# TransformerLens >= 4 has no HookedTransformer
pip download --no-deps -q transformer-lens -d /tmp/tl && python -c "import glob, zipfile; \
  z = zipfile.ZipFile(glob.glob('/tmp/tl/*.whl')[0]); \
  print([n for n in z.namelist() if n.endswith('HookedTransformer.py')] or 'removed')"
```

Trap #1 is re-checkable in-repo with `pytest -m network -k "gpt2_and_distilgpt2 or real_labels"`,
and the decomposition claim with `pytest -k causally_exact`.

## Sources

Secondary claims above (SAE limitations, circuit tracing, field status) come
from:

- [Resurrecting the Salmon: Rethinking Mechanistic Interpretability with Domain-Specific Sparse Autoencoders](https://arxiv.org/pdf/2508.09363)
- [Interpretable and Steerable Concept Bottleneck Sparse Autoencoders](https://arxiv.org/abs/2512.10805)
- [Improving Dictionary Learning with Gated Sparse Autoencoders](https://arxiv.org/pdf/2404.16014)
- [The Circuits Research Landscape — Neuronpedia](https://www.neuronpedia.org/graph/info)
- [Circuit Tracing: A Step Closer to Understanding Large Language Models](https://towardsdatascience.com/circuit-tracing-a-step-closer-to-understanding-large-language-models/)
- [Understanding Mechanistic Interpretability in AI Models — IntuitionLabs](https://intuitionlabs.ai/articles/mechanistic-interpretability-ai-llms)
- [SAELens](https://github.com/decoderesearch/SAELens)
- [A Primer on the Inner Workings of Transformer-based Language Models](https://arxiv.org/pdf/2405.00208)

Added 2026-11-16:

- [Announcing Gemma Scope 2](https://www.lesswrong.com/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2) and [DeepMind's release post](https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior?hl=es) (2025-12-19)
- [R-lens: making J-lens more faithful on early layers](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers)
- [J++ Lens: Jacobian filtering enables more faithful workspace lenses](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace-lenses) and [Circuits Updates, October 2026](https://transformer-circuits.pub/2026/october-update/)
- [Natural Language Autoencoders (Anthropic, 2026-05-07)](https://transformer-circuits.pub/2026/nla/index.html)
- [TransformerLens 3.0 release notes](https://transformerlensorg.github.io/TransformerLens/content/news/release-3.0.html) and [Migrating to TransformerLens 3](https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v3.html)
- [Language Model Circuits Are Sparse in the Neuron Basis (Arora et al., 2026)](https://arxiv.org/html/2601.22594)
- [Neuronpedia: Interpretability Explorer and CRM graphs (2026-04-21)](https://www.neuronpedia.org/blog/interp-explorer)
