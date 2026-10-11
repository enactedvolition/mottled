"""Conditional wording scores and narrowly scoped residual interventions.

Scores are sums of actual final-output conditional token log probabilities.
They are not logit-lens estimates or a reconstruction of hidden causes.
Layer numbering follows ``capture.HookCapture``: 0 is the stream entering
the first block, and 1..N are block outputs. A patch replaces exactly the
final *prompt* position, including when scoring a multi-token continuation.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from numbers import Integral

import numpy as np


def _sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


class WordingBackend:
    """An already loaded, local Hugging Face causal LM and its tokenizer.

    No generation or model loading occurs here. Each candidate is evaluated
    separately with a full causal forward pass, no KV cache, and no gradients.
    Tokenization of prompt + continuation must preserve the prompt's token
    prefix; ambiguous boundary retokenization is rejected, not approximated.
    """

    def __init__(self, model, tokenizer):
        from capture import _require_torch
        from models.families import resolve_family

        _require_torch()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.adapter = resolve_family(model)
        if not len(self.adapter.blocks):
            raise ValueError("wording scoring requires at least one model block")

    @property
    def n_layers(self) -> int:
        """Number of residual sites, including the embedding stream."""
        return len(self.adapter.blocks) + 1

    def _layer(self, layer) -> int:
        if isinstance(layer, bool) or not isinstance(layer, Integral):
            raise ValueError("layer must be an integer residual-site index")
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer must be in [0, {self.n_layers})")
        return int(layer)

    def _encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise ValueError("prompt and continuations must be strings")
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in ids):
            raise ValueError("tokenizer must return a flat sequence of nonnegative token ids")
        return [int(i) for i in ids]

    def _inputs(self, ids):
        import torch

        device = next(self.adapter.embed.parameters()).device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def _edit(self, patch, position):
        import torch

        if not isinstance(patch, dict) or set(patch) != {"layer", "value"}:
            raise ValueError("patch must contain exactly layer and value")
        layer = self._layer(patch["layer"])
        value = np.asarray(patch["value"])
        if (value.ndim != 1 or not value.size or value.dtype.kind not in "fiu"
                or not np.isfinite(value).all()):
            raise ValueError("patch value must be a finite, nonempty real vector")
        value = value.copy()

        def replace(hidden):
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("patch expects a single-sequence residual stream")
            if value.shape != (hidden.shape[-1],):
                raise ValueError("patch value shape must match residual width")
            replacement = torch.as_tensor(value, device=hidden.device, dtype=hidden.dtype)
            if not torch.isfinite(replacement).all():
                raise ValueError("patch value is nonfinite in the model's dtype")
            edited = hidden.clone()
            edited[0, position, :] = replacement
            return edited

        return {layer: replace}

    def state(self, prompt: str, layer: int) -> np.ndarray:
        """Copy a donor residual from a prompt-only pass.

        Although attention is causal, floating-point kernels can depend on
        sequence shape. Use ``self_score`` for the instrumentation control,
        which captures and replaces within matching full scoring shapes.
        """
        import torch
        from capture import HookCapture

        layer = self._layer(layer)
        ids = self._encode(prompt)
        if not ids:
            raise ValueError("prompt must contain at least one token")
        self.model.eval()
        with torch.no_grad(), HookCapture(self.model) as captured:
            self.model(**self._inputs(ids), use_cache=False, return_dict=True)
            value = captured.states[layer][0, -1].float().cpu().numpy().copy()
        if not np.isfinite(value).all():
            raise ValueError("captured prompt state contains nonfinite values")
        return value

    def score(self, prompt: str, continuations: dict[str, str],
              patch: dict | None = None) -> dict:
        """Return per-label scores and the exact scored token sequences.

        ``logprob`` is unnormalized across alternatives. Different-length
        continuations retain their full sequence probability; no length
        normalization, EOS probability, or candidate-set softmax is applied.
        """
        return self._score(prompt, continuations, patch=patch)

    def self_score(self, prompt: str, continuations: dict[str, str], layer: int) -> dict:
        """Self-replace using each candidate's exact full input and shape.

        A capture pass followed by a replacement pass isolates hook/write
        instrumentation error from prefix-versus-full-pass numerical drift.
        The donor remains the final prompt position, never a forced token.
        This control does not measure numerical drift in prompt-only donor
        captures used by ``state``.
        """
        return self._score(prompt, continuations, self_layer=self._layer(layer))

    def _score(self, prompt, continuations, patch=None, self_layer=None):
        import torch
        from capture import HookCapture

        prompt_ids = self._encode(prompt)
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")
        if not isinstance(continuations, dict) or not continuations:
            raise ValueError("continuations must be a nonempty label-to-text mapping")
        encoded = {}
        seen = set()
        for label, text in continuations.items():
            if not isinstance(label, str) or not label:
                raise ValueError("continuation labels must be nonempty strings")
            if not isinstance(text, str) or not text:
                raise ValueError("continuations must be nonempty strings")
            full = self._encode(prompt + text)
            if full[:len(prompt_ids)] != prompt_ids:
                raise ValueError(f"continuation {label!r} retokenizes the prompt boundary")
            continuation_ids = tuple(full[len(prompt_ids):])
            if not continuation_ids:
                raise ValueError(f"continuation {label!r} contains no additional tokens")
            if continuation_ids in seen:
                raise ValueError("distinct alternatives must not have identical token ids")
            seen.add(continuation_ids)
            encoded[label] = (full, continuation_ids)

        edits = None if patch is None else self._edit(patch, len(prompt_ids) - 1)
        self.model.eval()
        records = {}
        for label, (full, continuation_ids) in encoded.items():
            inputs = self._inputs(full)
            if self_layer is not None:
                with torch.no_grad(), HookCapture(self.model) as captured:
                    self.model(**inputs, use_cache=False, return_dict=True)
                    # float64 preserves every native floating-point value,
                    # including float64 model states and bfloat16 tensors
                    # that cannot be converted directly to NumPy.
                    value = captured.states[self_layer][0, len(prompt_ids) - 1]
                    value = value.double().cpu().numpy().copy()
                edits = self._edit({"layer": self_layer, "value": value},
                                   len(prompt_ids) - 1)
            context = (nullcontext() if edits is None
                       else HookCapture(self.model, state_edits=edits))
            with torch.no_grad(), context:
                output = self.model(**inputs, use_cache=False, return_dict=True)
                # Logits at P-1 predict the first continuation token. The
                # model's output at the last forced token predicts nothing
                # included in this score, and is deliberately excluded.
                logits = output.logits[0, len(prompt_ids) - 1:len(full) - 1].double()
                if logits.shape[0] != len(continuation_ids) or not torch.isfinite(logits).all():
                    raise ValueError("model returned missing or nonfinite scoring logits")
                target = torch.tensor(continuation_ids, device=logits.device,
                                      dtype=torch.long)
                logprobs = logits.log_softmax(-1).gather(-1, target[:, None]).squeeze(-1)
                total = logprobs.sum()
                if not torch.isfinite(logprobs).all() or not torch.isfinite(total):
                    raise ValueError("model returned nonfinite conditional log probabilities")
                records[label] = {
                    "logprob": float(total.cpu()),
                    "token_ids": list(continuation_ids),
                    "token_logprobs": logprobs.cpu().tolist(),
                    "prompt_token_ids": prompt_ids.copy(),
                    "input_ids": full.copy(),
                }
                if self_layer is not None:
                    records[label]["self_patch_capture_policy"] = "matched_full_input_ids"
        return records

    def metadata(self) -> dict:
        """Record reproducibility identifiers without implying a weight hash.

        An unavailable revision remains null. Locally initialized models
        require their initializer/seed or saved checkpoint to be reproduced;
        a configuration hash alone does not identify their weights.
        """
        import torch

        config = self.model.config
        tokenizer_config = getattr(self.tokenizer, "init_kwargs", {})
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        tokenizer_material = {
            "backend": None if backend is None else backend.to_str(),
            "vocab": self.tokenizer.get_vocab(),
            "special_tokens": getattr(self.tokenizer, "special_tokens_map", {}),
        }
        parameter = next(self.model.parameters())
        return {
            "model_id": getattr(config, "_name_or_path", None) or None,
            "model_class": type(self.model).__name__,
            "model_revision": getattr(config, "_commit_hash", None),
            "model_config_sha256": _sha256(config.to_dict()),
            "model_weights_sha256": None,
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", None) or None,
            "tokenizer_revision": tokenizer_config.get("_commit_hash") or tokenizer_config.get("revision"),
            "tokenizer_sha256": _sha256(tokenizer_material),
            "tokenizer_hash_scope": "backend serialization, vocabulary, special tokens; no slow-tokenizer code",
            "device": str(parameter.device),
            "dtype": str(parameter.dtype),
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "n_residual_sites": self.n_layers,
            "inference": {
                "mode": "teacher_forced_conditional_logprob",
                "eval": True,
                "grad_enabled": False,
                "use_cache": False,
                "add_special_tokens": False,
                "score_dtype": "float64",
                "append_eos": False,
                "length_normalization": False,
                "patch_position": "last_prompt_token",
                "donor_capture_policy": "prompt_only",
                "self_patch_capture_policy": "matched_full_input_ids",
                "attention_implementation": getattr(config, "_attn_implementation", None),
            },
        }
