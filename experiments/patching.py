"""Activation patching at one residual-stream site, read from the model's own output.

A patch overwrites the residual state at one (layer, position) and lets the
forward pass continue from it. What comes back is the model's own final
logits (`out.logits`), clean and patched, never a lens reading of them: a
lens is what a patching experiment is there to test, so it cannot also be
the instrument that scores it.

Sites use the StateTrajectory convention (layer 0 is the embedding stream,
layer k the output of block k-1), and positions index the exact `input_ids`
given. Nothing here tokenizes: a patch on a re-tokenized prompt can land on a
different token, and that does not crash, it answers a different question.

The write is `intervene.SetState` through `capture.HookCapture`, the same
edit every other Mottled intervention makes. Only the readout positions'
logits are computed (`logits_to_keep`): over items x layers x conditions, an
(L, T, V) stack per pass against GPT-2's 50k vocabulary is what a sweep
cannot hold.

A patch effect shows sufficiency under the tested conditions, not mechanism
(docs/validity.md).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

from capture import HookCapture, logit_lens
from intervene import SetState, _compile
from models.families import resolve_family


@dataclass(frozen=True)
class Run:
    """One unedited forward pass over exact token ids."""

    input_ids: tuple[int, ...]
    readout: tuple[int, ...]   # positions whose final logits were kept
    logits: np.ndarray         # (len(readout), V) float32: the model's out.logits
    hidden: np.ndarray         # (L, T, D) float32: the residual stream


@dataclass(frozen=True)
class Patch:
    clean: Run
    layer: int
    position: int
    logits: np.ndarray         # (len(readout), V) float32 under the patch
    state_before: np.ndarray   # (D,) the clean state at the site
    state_after: np.ndarray    # (D,) the state that propagated from the site
    provenance: dict


def load_model(name: str, revision: str | None):
    """`name` at exactly `revision`, float32 on CPU in eval mode, plus its
    tokenizer from the same revision.

    `revision` should be a full commit SHA. The hub resolves a branch or tag
    to whatever it points at today, so a resolved commit other than the one
    asked for is refused rather than recorded: the run would otherwise be a
    measurement of different weights under the requested name.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision,
                                                 dtype=torch.float32)
    model.to("cpu").eval()
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved is None and revision is not None and re.fullmatch(r"[0-9a-f]{40}", revision):
        # transformers 5.20 no longer sets _commit_hash; a full commit SHA is
        # immutable on the hub, so files fetched by it are that commit's
        resolved = revision
        model.config._commit_hash = revision
    if revision is not None and resolved != revision:
        raise ValueError(f"{name} at revision {revision} resolved to commit {resolved}")
    return model, tokenizer


def model_record(model) -> dict:
    """Which weights, in which arithmetic, under which library versions."""
    import transformers

    param = next(model.parameters())
    config = getattr(model, "config", None)
    return {
        "model": getattr(config, "name_or_path", type(model).__name__),
        # None for a local directory or a model built in-process
        "revision": getattr(config, "_commit_hash", None),
        "family": resolve_family(model).name,
        "dtype": str(param.dtype).removeprefix("torch."),
        "device": str(param.device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def clean(model, input_ids, readout=-1) -> Run:
    """The unedited pass: final logits at `readout` (an int or a sequence of
    positions; negative counts from the end) and every layer's state."""
    ids = _ids(input_ids)
    keep = _positions(readout, len(ids))
    logits, cap = _forward(model, ids, keep)
    return Run(input_ids=ids, readout=keep, logits=logits, hidden=cap.stacked().numpy())


def patch(model, input_ids, layer: int, position: int, value, readout=-1,
          clean_run: Run | None = None) -> Patch:
    """Write `value` (D,) at (`layer`, `position`) and read the final logits.

    `clean_run` reuses a baseline from `clean()` for the same ids and readout
    (a sweep patches one item many times); it is refused if it was made for
    other ids or positions, since the difference would then compare two
    different inputs.
    """
    ids = _ids(input_ids)
    keep = _positions(readout, len(ids))
    base = clean(model, ids, keep) if clean_run is None else clean_run
    if base.input_ids != ids or base.readout != keep:
        raise ValueError("clean_run was made for other input_ids or readout positions")
    n_layers, n_tokens, dim = base.hidden.shape
    if not 0 <= layer < n_layers:
        raise ValueError(f"layer {layer} outside 0..{n_layers - 1}")
    (position,) = _positions(position, n_tokens)
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (dim,):
        raise ValueError(f"value has shape {value.shape}; the residual stream is ({dim},)")

    edit = SetState(layer, value, token=position)
    state_edits, _ = _compile([edit])
    logits, cap = _forward(model, ids, keep, state_edits)
    return Patch(
        clean=base, layer=int(layer), position=position, logits=logits,
        state_before=base.hidden[layer, position],
        state_after=cap.states[layer][0, position].float().cpu().numpy(),
        provenance={**model_record(model), "input_ids": list(ids), "readout": list(keep),
                    "site": {"layer": int(layer), "position": position},
                    "edit": edit.record()})


def lens_logits(model, states, lens=None) -> np.ndarray:
    """Lens readouts of `states` laid out (L, n, D), row l read as layer l:
    the raw logit lens, or `lens` (a `tuned_lens.TunedLens`) translating
    first. Returns (L, n, V) float32."""
    h = torch.from_numpy(np.ascontiguousarray(states, dtype=np.float32))
    if lens is not None:
        from tuned_lens import translate

        h = translate(h, lens)
    return logit_lens(h, resolve_family(model)).numpy()


def log_odds(logits, target: int) -> np.ndarray:
    """log p - log(1 - p) of `target` under softmax over the last axis, float64.

    Computed as z_y - logsumexp over j != y of z_j, which is the same number
    as log_softmax_y - log(1 - p_y) without the cancellation the second term
    suffers as p_y -> 1, where the observed changes of a confident top-1
    token live.
    """
    z = np.asarray(logits, dtype=np.float64)
    rest = np.delete(z, target, axis=-1)
    top = rest.max(axis=-1, keepdims=True)
    return z[..., target] - (top[..., 0] + np.log(np.exp(rest - top).sum(axis=-1)))


def _forward(model, ids: tuple[int, ...], keep: tuple[int, ...], state_edits=None):
    device = next(model.parameters()).device
    model.eval()
    with HookCapture(model, state_edits=state_edits) as cap, torch.no_grad():
        out = model(torch.tensor([ids], device=device),
                    logits_to_keep=torch.tensor(keep, device=device), use_cache=False)
    return out.logits[0].float().cpu().numpy(), cap


def _ids(input_ids) -> tuple[int, ...]:
    a = np.asarray(input_ids)
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 1 or a.size == 0 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("input_ids must be a non-empty 1-D sequence of token ids; "
                         "tokenize once and pass the ids, so every pass sees the same tokens")
    return tuple(int(i) for i in a)


def _positions(positions, n_tokens: int) -> tuple[int, ...]:
    ps = [positions] if isinstance(positions, (int, np.integer)) else list(positions)
    if not ps or any(not -n_tokens <= int(p) < n_tokens for p in ps):
        raise ValueError(f"positions {ps} outside a {n_tokens}-token input")
    return tuple(int(p) % n_tokens for p in ps)
