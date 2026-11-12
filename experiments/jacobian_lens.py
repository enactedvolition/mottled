"""Fit a Jacobian lens: the corpus-averaged Jacobian from each state to the last.

Gurnee et al. 2026, "Verbalizable Representations Form a Global Workspace in
Language Models" (transformer-circuits.pub/2026/workspace) read a state h at
layer l as

    lens_l(h) = head(final_norm(J_l h)),     J_l = E[ d h_final / d h_l ]

The estimator is the one their reference implementation documents
(github.com/anthropics/jacobian-lens, jlens/fitting.py): for each output
dimension a one-hot cotangent is set at every valid target position at once,
so the gradient at source position p is sum_{p' >= p} d h_final[p'] / d h_l[p];
that is averaged over valid source positions, then over sequences. Valid
positions leave out the first SKIP_FIRST (attention sinks, whose statistics
are atypical) and the last (no next-token target).

The lens is written as a tuned-lens directory with translator weight J_l - I
and zero bias, so `tuned_lens.translate` computes h + (J_l - I) h = J_l h and
`capture --lens DIR` reads through it unchanged. The target is the last
block's output — the head's own input — so J_final = I, which is the format's
rule that the last layer needs no translator.

    python -m experiments.jacobian_lens --device xpu --dtype bfloat16 --out lens/

A Jacobian-lens readout is a readout diagnostic (docs/validity.md), as the raw
and tuned lens are: what a state is disposed, on average over this corpus, to
make the model say. It is not what the model believes at that layer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from capture import HookCapture, load_model

SKIP_FIRST = 16      # the reference implementation's default
MAX_TOKENS = 128     # the paper's sequence length


def valid_positions(n_tokens: int, skip_first: int = SKIP_FIRST) -> torch.Tensor:
    pos = torch.arange(skip_first, n_tokens - 1)
    if len(pos) == 0:
        raise ValueError(f"{n_tokens} tokens leave no valid position after skipping "
                         f"the first {skip_first} and the last")
    return pos


def jacobians(model, input_ids: torch.Tensor, skip_first: int = SKIP_FIRST,
              dim_batch: int = 8, layers=None) -> torch.Tensor:
    """J_l for each state index l in `layers` (default: every one below the
    last), from one sequence: (len(layers), D, D) float32 on CPU, row i =
    output dimension i.

    One forward of the sequence replicated `dim_batch` times, then one
    backward per `dim_batch` output dimensions against the retained graph:
    batch element b carries the one-hot cotangent for dimension start + b.
    The graph is rooted at the lowest requested state, so asking for one
    layer leaves its J unchanged and skips the backward through every block
    below it.
    """
    from models.families import resolve_family

    n = resolve_family(model).n_layers
    layers = list(range(n)) if layers is None else sorted(set(layers))
    if not layers or layers[0] < 0 or layers[-1] >= n:
        raise ValueError(f"layers must be state indices in [0, {n}), got {layers}")
    device = next(model.parameters()).device
    pos = valid_positions(len(input_ids), skip_first).to(device)
    live = {}

    def keep(layer):
        def edit(h):
            if layer == layers[0]:
                # a leaf here roots the graph, so the backward stops at the
                # lowest state asked for instead of walking to the embedding
                h = h.detach().requires_grad_(True)
            live[layer] = h
            return h
        return edit

    with torch.enable_grad(), HookCapture(model, state_edits={l: keep(l) for l in [*layers, n]}):
        model(input_ids=input_ids.to(device)[None].expand(dim_batch, -1), logits_to_keep=1)
    target, sources = live[n], [live[l] for l in layers]
    d = target.shape[-1]
    out = torch.zeros(len(layers), d, d)
    rows = torch.arange(dim_batch, device=device)
    cotangent = torch.zeros_like(target)
    for start in range(0, d, dim_batch):
        m = min(dim_batch, d - start)
        cotangent.zero_()
        cotangent[rows[:m, None], pos[None, :], start + rows[:m, None]] = 1
        grads = torch.autograd.grad(target, sources, grad_outputs=cotangent,
                                    retain_graph=start + dim_batch < d)
        for i, g in enumerate(grads):
            out[i, start:start + m] = g[:m, pos].float().mean(1).cpu()
    return out


def fit(model, sequences, skip_first: int = SKIP_FIRST, dim_batch: int = 8,
        layers=None, log=lambda s: None) -> torch.Tensor:
    """The mean of `jacobians` over token-id sequences."""
    total = None
    for i, ids in enumerate(sequences):
        t0 = time.time()
        j = jacobians(model, ids, skip_first, dim_batch, layers)
        total = j if total is None else total + j
        log(f"sequence {i + 1}/{len(sequences)}: {time.time() - t0:.0f}s")
    return total / len(sequences)


def save(path, J: torch.Tensor, record: dict, layers=None, n_blocks=None) -> str:
    """A tuned-lens directory (config.json + params.pt) holding J; returns
    the SHA-256 of params.pt, the bytes a later run should name.

    J fit at only some states (`layers`, of a model with `n_blocks`) is
    written with just those translators and a "layers" entry, which
    `tuned_lens.from_dir` refuses: a lens with no translator at a layer
    would read that layer as the raw logit lens under a Jacobian label.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    k, d, _ = J.shape
    layers = list(range(k)) if layers is None else sorted(layers)
    n_blocks = k if n_blocks is None else n_blocks
    eye = torch.eye(d)
    torch.save({f"{l}.weight": (J[i] - eye).contiguous() for i, l in enumerate(layers)},
               path / "params.pt")
    partial = {} if layers == list(range(n_blocks)) else {"layers": layers}
    (path / "config.json").write_text(json.dumps(
        {**record, **partial, "num_hidden_layers": n_blocks, "d_model": d, "bias": False,
         "lens_kind": "jacobian"}, indent=1))
    return hashlib.sha256((path / "params.pt").read_bytes()).hexdigest()


def main(argv=None, loaded=None):
    from experiments import introspection as ix

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=ix.MODEL)
    ap.add_argument("--revision", default=ix.MODEL_REVISION)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--skip-first", type=int, default=SKIP_FIRST)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--layers", default=None,
                    help="comma-separated state indices to fit (default: all below the last)")
    ap.add_argument("--docs", type=int, default=None, help="cap documents (tests)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)

    if loaded is None:
        model, tok = load_model(args.model, device=args.device, dtype=args.dtype,
                                revision=args.revision)
    else:
        model, tok = loaded
        model.to(args.device)
    # only the states need gradients; parameter grads would hold memory for nothing
    model.requires_grad_(False)

    # the introspection plan's train documents: the lens never sees a passage
    # that plan evaluates on
    train_docs, _ = ix.doc_split(ix.load_corpus())
    if args.docs:
        train_docs = train_docs[:args.docs]
    seqs = [torch.tensor(tok(d["text"], add_special_tokens=False)["input_ids"][:args.max_tokens])
            for d in train_docs]

    from models.families import resolve_family
    n_blocks = resolve_family(model).n_layers
    layers = None if args.layers is None else [int(x) for x in args.layers.split(",")]
    J = fit(model, seqs, args.skip_first, args.dim_batch, layers, log)
    record = {
        "base_model_name_or_path": args.model,
        "base_model_revision": getattr(model.config, "_commit_hash", None) or args.revision,
        "estimator": "sum over valid target positions, mean over valid source positions",
        "target": "last block output (pre final norm)",
        "corpus_sha256": ix.CORPUS_SHA256,
        "docs": [d["id"] for d in train_docs],
        "tokens": [len(s) for s in seqs],
        "skip_first": args.skip_first,
        "max_tokens": args.max_tokens,
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "git_commit": ix._git_commit(),
    }
    print(save(args.out, J, record, layers, n_blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
