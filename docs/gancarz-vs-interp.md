# mottled: Gancarz's Unix philosophy vs. modern mech-interp practice

Branch `feat/gancarz-interp` (stacked on `feat/unix-cli`). Sources are listed at the end. Time was short, so I cited them from known references and did **not** re-fetch each URL in this pass. The Gancarz wording matches the standard list in his book.

## Gancarz's nine tenets (Mike Gancarz, *The UNIX Philosophy*, Digital Press, 1995)
1. Small is beautiful.
2. Make each program do one thing well.
3. Build a prototype as soon as possible.
4. Choose portability over efficiency.
5. Store data in flat text files.
6. Use software leverage to your advantage.
7. Use shell scripts to increase leverage and portability.
8. Avoid captive user interfaces.
9. Make every program a filter.

## Interp practice in 2025-26, in brief
- **Notebooks are the glue.** Work happens in Jupyter or Colab, in Python against hooked models: TransformerLens `run_with_cache`/hooks, and nnsight tracing contexts. Teaching material such as ARENA is built on notebooks.
- **Remote execution.** NDIF/nnsight runs the same intervention code against large models you can't host yourself.
- **Shared artifacts on the HF Hub.** SAEs are loaded through SAELens (e.g. Gemma Scope), feature dashboards and labels come from Neuronpedia, and tensors are stored as safetensors.
- **Attribution graphs and circuit tracing.** Anthropic's 2025 methods paper and the open `circuit-tracer`, with graphs explored interactively on Neuronpedia.
- **Interventions with controls.** Patching and steering are expected to report baselines and random or norm-matched controls, and to state metric and position choices (Zhang & Nanda 2023; Heimersheim & Nanda 2024). Steering-reliability work (Tan et al. 2024) shows effects vary a lot by input.
- **Reproducibility.** Pinned model revisions, seeds, dtype, and versions of the SAE and library.
- **Shareable interactive views.** A link someone else can open (Neuronpedia-style) counts as part of the result.

## Tenet by tenet
| # | Gancarz | Interp practice wants | Verdict | Resolution in mottled |
|---|---|---|---|---|
| 1 | Small is beautiful | One importable library you `import` in a notebook | Diverge (mild) | Keep one library and make each CLI verb a thin wrapper (<60 lines, no new math). Small at the interface, cohesive underneath. *Judgment call.* |
| 2 | One thing well | Composable primitives: capture, project, compare, steer, SAE | **Agree** | One verb per job: `capture`, `project`, `inspect`, `validate`, `compare`, `sae`, `intervene`, `dose`, `arrays`. |
| 3 | Prototype early | Fast exploratory loops, but results must be reproducible | Diverge | Prototype freely, but every artifact carries provenance (revision, seed, dtype, versions) in the `.mtj`. Gancarz's "throw one away" applies to code, not to results. |
| 4 | Portability over efficiency | GPUs, bf16, fused kernels, NDIF | **Conflict** | Make the data and interface portable, not the compute: the format is device-agnostic, CPU works everywhere, and `--device/--dtype` is opt-in. Speed never changes what a file means; dtype is recorded. *Judgment call.* |
| 5 | Flat text files | Large float tensors in safetensors, HDF5, zarr, npz | **Conflict** | Text for everything a person or script reads (JSON manifest, JSON/NDJSON outputs, JSON Schema). Binary for tensors, in the `.mtj` container, with `mottled arrays --format npz/safetensors` so they land in standard formats. A text dump of a (L,T,D) float tensor would be lossy or huge. |
| 6 | Software leverage | Reuse TL, nnsight, SAELens, Neuronpedia, HF Hub | **Agree** | Wrap, don't reimplement: SAEs load from SAELens/Hub, and arrays go out as safetensors for torch/numpy. The Neuronpedia converter (spec'd earlier) is the next leverage step. |
| 7 | Shell scripts as glue | Python notebooks as glue | **Conflict** | Support both. Every verb is scriptable from the shell (stdin/stdout, JSON, exit codes), and every verb is a library call too. Notebook route: `statefile.read_container(path)` returns (manifest, numpy arrays), or `safetensors.torch.load_file`. The CLI is for batch work and CI; notebooks are for exploration. *Judgment call:* notebooks win day to day, and the shell matters for reproducible pipelines. |
| 8 | Avoid captive UIs | Interactive visual exploration is the core product | **Conflict** | The UI stays, but it is no longer the default and never the only path. Bare `mottled` prints help, Streamlit is `mottled ui`, and every analysis in the UI has a non-interactive verb. The viewer reads the same `.mtj` (a shareable `?file=` link). |
| 9 | Every program a filter | Pipelines over runs; JSON to jq/pandas | **Agree** (with binary caveat) | Stages read stdin and write stdout. Analysis verbs print one JSON document. The pipe format between stages is binary `.mtj`, a filter over typed tensors rather than text lines. |

**Agreement:** 2, 6, 9 (and mostly 1). **Divergence:** 1, 3. **Real conflicts:** 4, 5, 7, 8.

## A point where interp practice is stricter than Gancarz
Gancarz says nothing about measurement validity. Interp practice requires controls, and a correct but misleading number is worse than a slow tool. Two bugs fell under this and are fixed on this branch:
- **Steering inflation.** A steer pushed at every position was scored only at the last one, while the control was also applied everywhere. It is now scored as the mean shift over all steered positions (`token=None`).
- **MoE layer labels.** Router logits for interleaved MoE stacks (Qwen-MoE `decoder_sparse_step`/`mlp_only_layers`, DeepSeek `first_k_dense_replace`/`moe_layer_freq`, Llama-4 `interleave_moe_layer_step`) were assigned to the tail blocks. They now come from the config, or from position when HF returns one entry per block.

## Sources
- Gancarz tenets: https://en.wikipedia.org/wiki/Unix_philosophy (Mike Gancarz section); M. Gancarz, *The UNIX Philosophy*, 1995.
- TransformerLens: https://github.com/TransformerLensOrg/TransformerLens
- nnsight / NDIF: https://nnsight.net , https://ndif.us
- SAELens: https://github.com/jbloomAus/SAELens ; Gemma Scope: https://huggingface.co/google/gemma-scope
- Neuronpedia: https://www.neuronpedia.org , https://github.com/hijohnnylin/neuronpedia
- Attribution graphs: https://transformer-circuits.pub/2025/attribution-graphs/methods.html ; circuit-tracer: https://github.com/safety-research/circuit-tracer
- Activation patching practice: Zhang & Nanda, https://arxiv.org/abs/2309.16042 ; Heimersheim & Nanda, https://arxiv.org/abs/2404.15255
- Steering reliability: Tan et al., https://arxiv.org/abs/2407.12404
- ARENA notebooks: https://github.com/callummcdougall/ARENA_3.0
- safetensors: https://huggingface.co/docs/safetensors
