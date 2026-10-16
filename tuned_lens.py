"""Tuned lens: a per-layer affine correction before the output head.

The raw logit lens (`capture.logit_lens`) points the final norm and
unembedding at every intermediate state, as if layer l spoke the final
layer's basis. It does not, and the error is a bias, not noise: early-layer
readouts lean toward the input token and away from what the model goes on to
predict (Belrose et al. 2023, "Eliciting Latent Predictions from Transformers
with the Tuned Lens"). A tuned lens fits one affine translator per layer so
the head reads that layer in the basis it was trained on:

    logits_l = head(final_norm(h_l + A_l h_l + b_l))

trained to minimise KL from the model's own output distribution. The last
layer needs no translator: it *is* the head's input.

This module loads the lenses that paper published (the `AlignmentResearch/
tuned-lens` Space: GPT-2, Pythia-deduped, Llama-2/3, OPT, ...) or any
directory in the same format — which is what `python -m tuned_lens train`
from the `tuned-lens` package writes, the route for a model nobody has
published a lens for. Mottled applies a lens; it does not train one.

A tuned readout is still a readout diagnostic (docs/validity.md): less
biased than the raw lens, not the model's belief at that layer.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HUB_REPO = "AlignmentResearch/tuned-lens"   # a Space, not a model repo

# Said wherever a raw-lens readout is shown. On GPT-2 (mean over four
# prompts) the raw lens sits 40.5 nats of KL from the model's own output
# distribution at layer 0 and 4.6-5.2 through layers 1-3, against 4.7 and
# 1.6-1.9 for the published tuned lens; the gap closes toward the last layer.
EARLY_LAYER_CAVEAT = ("the raw logit lens is biased, most at early layers; "
                      "capture with --lens tuned for a corrected readout")


@dataclass(frozen=True)
class TunedLens:
    weight: np.ndarray        # (n, D, D): translator i applies to captured layer i
    bias: np.ndarray          # (n, D)
    source: str               # hub path or local directory it was loaded from
    revision: str | None      # Space commit, when it came from the hub
    base_model_revision: str | None   # the weights it was trained against

    @property
    def n_layers(self) -> int:
        return int(self.weight.shape[0])

    def record(self) -> dict:
        """What a trajectory's meta says about the readout it carries."""
        return {"kind": "tuned", "source": self.source, "revision": self.revision,
                "base_model_revision": self.base_model_revision}


def readout_name(meta) -> str:
    """"tuned lens" or "logit lens", from a trajectory's meta. A capture
    that does not say is a raw logit lens: every backend before this one was."""
    kind = ((meta or {}).get("lens") or {}).get("kind", "logit")
    return "tuned lens" if kind == "tuned" else "logit lens"


def from_dir(path) -> TunedLens:
    """A lens directory in the tuned-lens format: config.json + params.pt."""
    import torch

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    # weights_only: params.pt is a plain state dict, and anything that needs
    # unpickling arbitrary objects to load is not one
    state = torch.load(path / "params.pt", map_location="cpu", weights_only=True)
    n = int(config["num_hidden_layers"])
    weight = np.stack([state[f"{i}.weight"].float().numpy() for i in range(n)])
    bias = (np.stack([state[f"{i}.bias"].float().numpy() for i in range(n)])
            if config.get("bias", True) else np.zeros(weight.shape[:2], np.float32))
    return TunedLens(weight=weight, bias=bias, source=str(path), revision=None,
                     base_model_revision=config.get("base_model_revision"))


@functools.lru_cache(maxsize=4)
def from_hub(model_name: str, revision: str | None = None) -> TunedLens:
    """The published lens for `model_name` (a hub id such as "gpt2")."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        files = [hf_hub_download(HUB_REPO, f"lens/{model_name}/{name}",
                                 repo_type="space", revision=revision)
                 for name in ("config.json", "params.pt")]
    except EntryNotFoundError:
        raise ValueError(
            f"no published tuned lens for {model_name!r} in {HUB_REPO}; train one "
            "with the tuned-lens package and pass its directory as the lens") from None
    lens = from_dir(Path(files[0]).parent)
    # the snapshot directory is named for the commit it resolved to: the
    # record names the lens that ran, not "whatever main was that day"
    parts = Path(files[0]).parts
    commit = parts[parts.index("snapshots") + 1] if "snapshots" in parts else None
    return TunedLens(weight=lens.weight, bias=lens.bias,
                     source=f"{HUB_REPO}/lens/{model_name}", revision=commit,
                     base_model_revision=lens.base_model_revision)


def resolve(lens: str, model_name: str) -> TunedLens | None:
    """`MarbleConfig.lens` -> a lens: None for "logit" (the raw lens),
    the published lens for the model on "tuned", else a lens directory."""
    if lens == "logit":
        return None
    if lens == "tuned":
        return from_hub(model_name)
    if Path(lens).is_dir():
        return from_dir(lens)
    raise ValueError(f"lens must be 'logit', 'tuned' or a tuned-lens directory, got {lens!r}")


def translate(hidden: "torch.Tensor", lens: TunedLens) -> "torch.Tensor":
    """Apply each layer's translator: (L, T, D) -> (L, T, D), float32."""
    import torch

    L, _, D = hidden.shape
    # a lens for another depth or width would run without error on a slice of
    # the wrong layers, so a mismatch is refused, not truncated
    if L != lens.n_layers + 1 or D != lens.weight.shape[-1]:
        raise ValueError(f"lens has {lens.n_layers} translators of width "
                         f"{lens.weight.shape[-1]}; this capture has {L} layers of "
                         f"width {D} (a lens needs one per layer but the last)")
    h = hidden.float()
    weight = torch.from_numpy(lens.weight)
    bias = torch.from_numpy(lens.bias)
    out = h.clone()
    out[:-1] = h[:-1] + torch.einsum("ltd,led->lte", h[:-1], weight) + bias[:, None, :]
    return out
