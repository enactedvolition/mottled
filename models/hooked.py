"""Optional TransformerLens producer: a TransformerLens model -> StateTrajectory.

Works on TransformerLens 4's ``TransformerBridge`` and on a pre-4
``HookedTransformer``: both expose ``run_with_cache`` with the same
``resid_pre`` / ``resid_post`` cache keys, ``ln_final``, ``unembed`` and
``W_E``, which is all this module touches.

TransformerLens is *not* a dependency — import this module only if you have it
installed (`pip install transformer-lens`). Rather than route a
``HookedTransformer`` through Mottled's HF forward hooks, we use TL's own
``run_with_cache``: it exposes the residual stream directly (``resid_pre`` /
``resid_post``), which is the cleanest bridge and a second proof that the
producer contract is all a new backend needs — emit a ``StateTrajectory`` and
the entire projection / density / metrics / viewer stack works unchanged.

Two entry points, because a TransformerLens user arrives in two states:

    from_hooked_transformer(model, prompt)   # run the pass, then build
    from_cache(cache, model, prompt)         # a cache you already ran

``from_cache`` is the one that matters for an expensive model: a notebook that
has already called ``run_with_cache`` should not have to pay for the forward
pass twice to look at it. ``from_hooked_transformer`` is that plus the run.

The residual extraction (`residual_stack`) and the ``StateTrajectory`` assembly
(`_to_trajectory`) are separated from both so each is unit-testable without the
heavy library.
"""
from __future__ import annotations

import numpy as np

from capture import _clean_token, _entropy_topk
from trajectory import StateTrajectory


def _to_trajectory(hidden: np.ndarray, tokens: list[str], lens: np.ndarray,
                   vocab: list[str], embedding_matrix: np.ndarray,
                   top_k: int = 5, keep_logits: bool = True,
                   model_name: str = "hooked", prompt: str = "",
                   device: str | None = None,
                   dtype: str | None = None,
                   family: str = "HookedTransformer") -> StateTrajectory:
    """Assemble a StateTrajectory from an extracted residual stack + logit lens.

    hidden: (L, T, D) residual stream (layer 0 = embeddings), lens: (L, T, V)
    logit-lens logits for every state. Pure numpy — this is where the entropy
    and top-k are computed, shared with the HF capture path.
    """
    hidden = np.asarray(hidden, dtype=np.float32)
    lens = np.asarray(lens, dtype=np.float32)
    entropy, topk = _entropy_topk(lens, vocab, top_k)
    traj = StateTrajectory(
        hidden=hidden,
        tokens=list(tokens),
        logits=lens.astype(np.float16) if keep_logits else None,
        entropy=entropy,
        topk=topk,
        vocab=vocab,
        embedding_matrix=np.asarray(embedding_matrix, dtype=np.float32),
        meta={"backend": "transformer_lens", "model": model_name,
              "prompt": prompt, "family": family,
              # provenance.record reads these; TransformerLens carries no hub
              # commit of its own, so `revision` stays absent rather than being
              # filled with the name of the checkpoint TL converted from
              "device": device, "dtype": dtype},
    )
    traj.validate()
    return traj


def residual_stack(cache, n_layers: int):
    """(L, T, D) residual stream from a TransformerLens ``ActivationCache``.

    Layer 0 is ``resid_pre`` of block 0 — the embedding stream — and layer l+1
    is block l's ``resid_post``, which is Mottled's layer convention and the
    same stack ``capture.py`` records for a HuggingFace model. Takes anything
    that indexes like a cache, so it is testable without the library.
    """
    import torch

    resid = [cache["resid_pre", 0]] + [cache["resid_post", l] for l in range(n_layers)]
    stacked = torch.stack(resid, dim=0)
    return stacked[:, 0] if stacked.ndim == 4 else stacked   # drop the batch axis


def from_cache(cache, model, prompt, top_k: int = 5,
               keep_logits: bool = True) -> StateTrajectory:
    """Build a StateTrajectory from a cache you have already run.

    `cache` is the ``ActivationCache`` from ``model.run_with_cache(...)`` and
    `prompt` is what you ran it on (a string, or the token tensor — whatever
    ``to_str_tokens`` accepts). Only the readout touches the model: TL's final
    layer-norm and unembed, applied to every captured state as the logit lens.
    """
    import torch

    str_tokens = [_clean_token(t) for t in model.to_str_tokens(prompt)]
    with torch.no_grad():
        hidden = residual_stack(cache, int(model.cfg.n_layers))
        # a string is re-tokenized with TL's default BOS, so a cache run with
        # prepend_bos=False is one position shorter; say so here rather than
        # let validate() report a bare count mismatch after the readout
        if len(str_tokens) != hidden.shape[1]:
            raise ValueError(
                f"the cache has {hidden.shape[1]} positions but to_str_tokens(prompt) "
                f"gives {len(str_tokens)}; pass the token tensor you ran (e.g. "
                f"model.to_tokens(text, prepend_bos=False)) instead of the string")
        lens = model.unembed(model.ln_final(hidden))          # (L, T, V) logit lens

    lens_np = lens.float().cpu().numpy()
    vocab = [_clean_token(t)
             for t in model.tokenizer.convert_ids_to_tokens(range(lens_np.shape[-1]))]
    return _to_trajectory(
        hidden.float().cpu().numpy(), str_tokens, lens_np, vocab,
        model.W_E.detach().float().cpu().numpy(),
        top_k=top_k, keep_logits=keep_logits,
        model_name=getattr(model.cfg, "model_name", "hooked"),
        prompt=prompt if isinstance(prompt, str) else "",
        device=str(hidden.device), dtype=str(hidden.dtype).removeprefix("torch."),
        # a bridge capture labelled "HookedTransformer" would name a class
        # TransformerLens 4 no longer has
        family=type(model).__name__,
    )


def from_hooked_transformer(model, prompt: str, top_k: int = 5,
                            keep_logits: bool = True) -> StateTrajectory:
    """Capture a TransformerLens ``TransformerBridge`` (or pre-4
    ``HookedTransformer``) as a StateTrajectory.

    Uses ``run_with_cache`` to read the residual stream after every block, then
    hands the cache to `from_cache`. Call that directly if you already have one.
    """
    import torch

    with torch.no_grad():
        _, cache = model.run_with_cache(model.to_tokens(prompt))
    return from_cache(cache, model, prompt, top_k=top_k, keep_logits=keep_logits)
