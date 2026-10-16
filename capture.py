"""Residual-stream capture via forward hooks.

`capture(model, prompt)` runs one forward pass and records the residual
stream after every transformer block (plus the initial embedding stream as
layer 0), then applies the logit lens (final norm + LM head) to every
captured state to obtain per-layer, per-token logits / entropy / top-k.

The result is a StateTrajectory — the only thing downstream modules see.
Transformers are one backend among several; any producer that emits a
StateTrajectory (models/logprobs.py, models/hooked.py, Mamba) plugs in the
same way.
"""

from __future__ import annotations

import numpy as np

from trajectory import StateTrajectory

try:  # torch/transformers are optional: not every producer needs them.
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False


def _require_torch():
    if not HAS_TORCH:
        raise ImportError(
            "torch is required for transformer capture: "
            'pip install "mottled[models]"')


def resolve_device(device: str = "auto") -> str:
    _require_torch()
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, device: str = "auto", dtype: str = "float32"):
    """Load a HF causal LM + tokenizer in eval mode."""
    _require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(device)
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch_dtype)
    model.to(device).eval()
    return model, tokenizer


class HookCapture:
    """Context manager that records — and optionally *edits* — the residual
    stream around every block.

    Layer 0 is taken from a forward *pre*-hook on the first block — the exact
    tensor entering the residual stream (this is also correct for Gemma,
    which scales embeddings after the embedding module).  Layers 1..N come
    from forward hooks on each block's output.

    Passing ``state_edits`` and/or ``frozen_blocks`` turns the capture into a
    *resumable/intervened* pass: the forward runs to a block, the residual is
    rewritten, and the model continues from the edited state — this is what
    makes perturb-and-replay possible.  With neither argument the behaviour is
    a pure read-only capture, identical to before.

        state_edits[k]  : callable(hidden) -> hidden applied to the state at
                          index k (0 = embeddings) *before* it is captured and
                          before it propagates downstream.
        frozen_blocks   : block indices whose update is skipped (output := input),
                          i.e. the residual stream passes through unchanged.

    Captured states always reflect the value that actually propagated, so a
    perturbation at layer k appears in ``hidden[k]`` and in every layer after.

    ``capture_components=True`` additionally hooks every block's attention and
    MLP submodules and records their outputs — the two additive writes to the
    residual stream, so for pre-norm architectures (Llama-style, GPT-2, NeoX)
    ``hidden[l+1] = hidden[l] + attn[l] + mlp[l]`` exactly.  A frozen block's
    submodules still run, but it writes nothing, so its components are zero.
    """

    def __init__(self, model, state_edits: dict | None = None,
                 frozen_blocks: set | None = None, capture_components: bool = False):
        from models.families import resolve_family

        self.adapter = resolve_family(model)
        self.states: dict[int, "torch.Tensor"] = {}
        self.components: dict[str, dict[int, "torch.Tensor"]] = {"attn": {}, "mlp": {}}
        self.state_edits = dict(state_edits or {})
        self.frozen_blocks = set(frozen_blocks or set())
        self.capture_components = capture_components
        self._handles = []

    def __enter__(self):
        blocks = list(self.adapter.blocks)

        def pre_hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            edit = self.state_edits.get(0)
            if edit is not None:
                hs = edit(hs)
                self.states[0] = hs.detach()
                if args:
                    return (hs,) + tuple(args[1:]), kwargs
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = hs
                return args, kwargs
            self.states[0] = hs.detach()
            return None

        self._handles.append(blocks[0].register_forward_pre_hook(pre_hook, with_kwargs=True))
        for i, block in enumerate(blocks):
            def hook(module, args, output, _block=i, _layer=i + 1):
                changed = False
                if _block in self.frozen_blocks:            # skip the update
                    out = args[0]
                    changed = True
                    # the submodules' outputs were discarded; recording them
                    # would give a block that wrote nothing an attn/MLP share
                    for writes in self.components.values():
                        if _block in writes:
                            writes[_block] = torch.zeros_like(writes[_block])
                else:
                    out = output[0] if isinstance(output, tuple) else output
                edit = self.state_edits.get(_layer)
                if edit is not None:
                    out = edit(out)
                    changed = True
                self.states[_layer] = out.detach()
                if not changed:
                    return None
                if isinstance(output, tuple):
                    return (out,) + tuple(output[1:])
                return out

            self._handles.append(block.register_forward_hook(hook))

        if self.capture_components:
            def component_hook(name: str, i: int):
                def hook(module, args, output):
                    out = output[0] if isinstance(output, tuple) else output
                    self.components[name][i] = out.detach()
                return hook

            for i, block in enumerate(blocks):
                attn, mlp = self.adapter.block_submodules(block)
                self._handles.append(attn.register_forward_hook(component_hook("attn", i)))
                self._handles.append(mlp.register_forward_hook(component_hook("mlp", i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def stacked(self) -> "torch.Tensor":
        """(L, T, D) float32 on CPU, layer 0 first, batch dim squeezed."""
        n = len(self.adapter.blocks) + 1
        missing = [i for i in range(n) if i not in self.states]
        if missing:
            raise RuntimeError(f"missing hook captures for layers {missing}")
        return torch.stack([self.states[i][0] for i in range(n)]).float().cpu()

    def stacked_components(self) -> dict[str, "torch.Tensor"]:
        """{"attn": (L-1, T, D), "mlp": (L-1, T, D)} float32 on CPU."""
        n = len(self.adapter.blocks)
        out = {}
        for name, per_block in self.components.items():
            missing = [i for i in range(n) if i not in per_block]
            if missing:
                raise RuntimeError(f"missing {name} captures for blocks {missing}")
            out[name] = torch.stack([per_block[i][0] for i in range(n)]).float().cpu()
        return out


def logit_lens(hidden: "torch.Tensor", adapter, chunk: int = 4) -> "torch.Tensor":
    """Apply final norm + LM head to every captured state: (L, T, V) logits."""
    _require_torch()
    if adapter.lm_head is None:
        raise ValueError("model exposes no LM head; cannot apply logit lens")
    device = next(adapter.lm_head.parameters()).device
    dtype = next(adapter.lm_head.parameters()).dtype
    outs = []
    with torch.no_grad():
        for i in range(0, hidden.shape[0], chunk):
            h = hidden[i : i + chunk].to(device=device, dtype=dtype)
            if adapter.final_norm is not None:
                h = adapter.final_norm(h)
            outs.append(adapter.lm_head(h).float().cpu())
    return torch.cat(outs, dim=0)


def _entropy_topk(logits: np.ndarray, vocab: list[str], k: int):
    # One layer at a time, and a partition rather than a sort of the whole
    # vocabulary: the (L, T, V) block in float64 plus a full argsort cost
    # 4.5 GB and 31 s on a 258-token GPT-2 capture, about three turns of a
    # chat that re-captures its whole conversation every turn.
    L, T, V = logits.shape
    k = min(k, V)
    ent = np.empty((L, T), dtype=np.float32)
    topk = []
    for layer in range(L):
        x = logits[layer].astype(np.float64)
        x -= x.max(axis=-1, keepdims=True)
        p = np.exp(x)
        p /= p.sum(axis=-1, keepdims=True)
        ent[layer] = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=-1)
        topk.append([[(vocab[j], float(row[j])) for j in _top_indices(row, k)]
                     for row in p])
    return ent, topk


def _top_indices(row: np.ndarray, k: int) -> np.ndarray:
    """The k largest entries, largest first, ties to the lower index: what a
    stable sort of the row gives, without sorting all of it."""
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    kth = np.partition(row, row.size - k)[row.size - k]
    above = np.flatnonzero(row > kth)
    idx = np.concatenate([above, np.flatnonzero(row == kth)[: k - above.size]])
    return idx[np.argsort(-row[idx], kind="stable")]


def capture(
    model,
    prompt: str,
    tokenizer=None,
    top_k: int = 5,
    device: str = "auto",
    dtype: str = "float32",
    keep_logits: bool = True,
    capture_components: bool = False,
    capture_attention: bool = False,
    capture_routing: bool = False,
    input_ids=None,
    positions: list[int] | None = None,
) -> StateTrajectory:
    """Run a forward pass and capture the residual stream at every layer.

    `model` may be a HF model instance (with `tokenizer` supplied) or a HF hub
    name.  Returns hidden[layer][token][dimension]
    wrapped in a StateTrajectory with logit-lens statistics attached.

    `input_ids`, when the prompt's exact token ids are known (a chat
    template's own tokenization), are captured as given; `prompt` then only
    labels the run.

    `capture_components=True` also records each block's attention and MLP
    outputs — the residual decomposition — in `StateTrajectory.components`.
    `capture_attention=True` records each block's head-averaged attention
    pattern (L-1, T, T) in `StateTrajectory.attention`.

    `capture_routing=True` records, for a sparse-MoE model, which experts each
    token was routed to per layer (`StateTrajectory.routing`). It refuses on a
    dense model rather than inventing a routing.

    `positions` keeps only those token positions (negative counts from the
    end), and runs the logit lens on them alone: the lens over every token
    against a 150k vocabulary is most of a chat prompt's capture time, and a
    reader of one position (`blast.pellets`) needs none of the rest. The
    trajectory's T axis is then those positions, recorded in
    meta["positions"]; the per-token-pair captures (components, attention,
    routing) are refused with it rather than sliced.
    """
    _require_torch()
    return _run(model, prompt, tokenizer=tokenizer, top_k=top_k, device=device,
                dtype=dtype, keep_logits=keep_logits,
                capture_components=capture_components,
                capture_attention=capture_attention,
                capture_routing=capture_routing,
                input_ids=None if input_ids is None
                else torch.as_tensor(input_ids).reshape(1, -1),
                positions=positions)


def generate_and_capture(
    model,
    prompt: str,
    max_new_tokens: int = 8,
    tokenizer=None,
    temperature: float = 0.0,
    seed: int | None = None,
    top_k: int = 5,
    device: str = "auto",
    dtype: str = "float32",
    keep_logits: bool = True,
    capture_components: bool = False,
    capture_attention: bool = False,
    input_ids=None,
) -> StateTrajectory:
    """Autoregressively decode, then capture the completed sequence.

    Attention is causal, so the hidden states of one forward pass over the
    finished sequence are *exactly* the states that existed at every decode
    step — the result is an ordinary StateTrajectory over prompt + generated
    tokens, not an approximation and not a new interchange type.  The decode
    itself is recorded in ``meta["generation"]``::

        {"prompt_tokens": P, "new_tokens": N,
         "mode": "greedy" | "sample", "temperature": t, "seed": s,
         "steps": [{"token": str, "id": int, "p": float, "entropy": float}]}

    ``p`` and ``entropy`` describe the actual sampling distribution at that
    step (temperature applied when sampling).  ``temperature=0`` decodes
    greedily; ``temperature>0`` samples with a seeded generator, so a run is
    reproducible given (prompt, temperature, seed).  Decoding stops early at
    the tokenizer's EOS token.  The decode loop runs one full forward pass
    per step (no KV cache) — transparent and exact, sized for the short
    continuations Mottled visualizes, not for bulk generation.

    `input_ids`, when the prompt's exact token ids are known (a chat
    template's own tokenization), are decoded from as given instead of
    re-tokenizing `prompt`, which then only labels the run: rendering a
    template and tokenizing the text again can add special tokens the
    template already wrote.
    """
    _require_torch()
    if isinstance(model, str):
        model, tokenizer = load_model(model, device=device, dtype=dtype)
    if tokenizer is None:
        raise ValueError("a tokenizer is required when passing a model instance")

    model_device = next(model.parameters()).device
    if input_ids is None:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    input_ids = torch.as_tensor(input_ids).reshape(1, -1).to(model_device)
    n_prompt = int(input_ids.shape[1])
    eos_id = getattr(tokenizer, "eos_token_id", None)

    gen = None
    if temperature > 0:
        gen = torch.Generator()
        gen.manual_seed(0 if seed is None else int(seed))

    steps: list[dict] = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids).logits[0, -1].float().cpu()
            scaled = logits / temperature if temperature > 0 else logits
            probs = torch.softmax(scaled, dim=-1)
            entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
            if temperature > 0:
                next_id = int(torch.multinomial(probs, 1, generator=gen))
            else:
                next_id = int(torch.argmax(logits))
            steps.append({
                "token": _clean_token(tokenizer.convert_ids_to_tokens([next_id])[0]),
                "id": next_id,
                "p": float(probs[next_id]),
                "entropy": entropy,
            })
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=model_device)], dim=1)
            if eos_id is not None and next_id == eos_id:
                break

    generation = {
        "prompt_tokens": n_prompt,
        "new_tokens": len(steps),
        "mode": "sample" if temperature > 0 else "greedy",
        "temperature": float(temperature),
        "seed": seed,
        "steps": steps,
    }
    return _run(model, prompt, tokenizer=tokenizer, top_k=top_k,
                keep_logits=keep_logits, extra_meta={"generation": generation},
                capture_components=capture_components,
                capture_attention=capture_attention,
                input_ids=input_ids)


def _run(model, prompt, tokenizer=None, top_k=5, device="auto", dtype="float32",
         keep_logits=True, state_edits: dict | None = None,
         frozen_blocks: set | None = None, extra_meta: dict | None = None,
         capture_components: bool = False,
         capture_attention: bool = False,
         capture_routing: bool = False,
         input_ids: "torch.Tensor | None" = None,
         logits_dtype: str = "float16",
         positions: list[int] | None = None) -> StateTrajectory:
    """Forward pass (optionally intervened) -> StateTrajectory. Shared by
    capture(), intervene() and generate_and_capture().

    `input_ids` (1, T) overrides tokenization of `prompt` — used when the
    exact token ids are already known (a decoded sequence must not be
    re-tokenized, since detokenize->tokenize can move token boundaries).

    `logits_dtype` is how the logits are stored. float16 halves a large
    array, but its rounding makes tiny KL steps and rank ties; `dose.py`
    asks for float32 so its smallest doses measure the model, not the cast.
    """
    if positions is not None and (capture_components or capture_attention
                                  or capture_routing):
        raise ValueError("positions keeps a subset of tokens; components, attention "
                         "and routing describe all of them and cannot be sliced to it")
    if isinstance(model, str):
        model, tokenizer = load_model(model, device=device, dtype=dtype)
    if tokenizer is None:
        raise ValueError("a tokenizer is required when passing a model instance")

    model_device = next(model.parameters()).device
    if input_ids is None:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
    input_ids = input_ids.to(model_device)
    tokens = [_clean_token(t) for t in tokenizer.convert_ids_to_tokens(input_ids[0].tolist())]

    model.eval()
    # sdpa/flash kernels never materialise the attention matrix; switch the
    # dispatch to eager for this pass so output_attentions actually returns.
    config = getattr(model, "config", None)
    prev_impl = getattr(config, "_attn_implementation", None)
    if capture_attention and prev_impl and prev_impl != "eager":
        config._attn_implementation = "eager"
    try:
        with HookCapture(model, state_edits=state_edits, frozen_blocks=frozen_blocks,
                         capture_components=capture_components) as cap, torch.no_grad():
            out = model(input_ids, output_attentions=capture_attention or None,
                        **({"output_router_logits": True} if capture_routing else {}))
    finally:
        if capture_attention and prev_impl and prev_impl != "eager":
            config._attn_implementation = prev_impl
    hidden = cap.stacked()  # (L, T, D)
    components = None
    if capture_components:
        components = {k: v.numpy() for k, v in cap.stacked_components().items()}
    attention = None
    if capture_attention:
        if not getattr(out, "attentions", None):
            raise ValueError("model returned no attention patterns; "
                             "this architecture does not support capture_attention")
        # (blocks, B, H, T, T) -> head-average, squeeze batch -> (L-1, T, T)
        attention = torch.stack([a.float().mean(dim=1)[0] for a in out.attentions]).cpu().numpy()

    routing = _routing_from(out, model, capture_routing)

    if positions is not None:
        T = hidden.shape[1]
        if not positions:
            raise ValueError("positions is empty: name at least one token to keep")
        if any(not -T <= p < T for p in positions):
            raise ValueError(f"positions {positions} outside a {T}-token input")
        positions = [p % T for p in positions]
        hidden = hidden[:, positions]
        tokens = [tokens[p] for p in positions]
        extra_meta = {**(extra_meta or {}), "positions": positions}

    logits_t = logit_lens(hidden, cap.adapter)
    vocab = [_clean_token(t) for t in tokenizer.convert_ids_to_tokens(range(logits_t.shape[-1]))]
    logits = logits_t.numpy()
    entropy, topk = _entropy_topk(logits, vocab, top_k)

    meta = {
        "backend": "transformers",
        "model": getattr(getattr(model, "config", None), "name_or_path", type(model).__name__),
        "prompt": prompt,
        "family": cap.adapter.name,
        # what a reproduction has to match beyond the name: the exact weights
        # (the hub commit they resolved to — absent for a local path or a model
        # built in-process) and the arithmetic they ran in.  provenance.record
        # reads these; nothing else in the pipeline does.
        "revision": getattr(config, "_commit_hash", None),
        "device": str(model_device),
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
    }
    if extra_meta:
        meta.update(extra_meta)
    return StateTrajectory(
        hidden=hidden.numpy(),
        tokens=tokens,
        logits=logits.astype(logits_dtype) if keep_logits else None,
        entropy=entropy,
        topk=topk,
        vocab=vocab,
        embedding_matrix=cap.adapter.embedding_weight().numpy(),
        components=components,
        attention=attention,
        routing=routing,
        meta=meta,
    )


def _clean_token(tok) -> str:
    """Make tokenizer pieces human-readable (SentencePiece/BPE markers)."""
    if tok is None:
        return "<unk>"
    return str(tok).replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\\n")


def moe_block_indices(cfg, n_blocks: int, logits: list) -> list[int]:
    """Which decoder block each non-None router-logits entry belongs to.

    HF returns router logits either one entry per block (dense blocks as
    None) or only for the MoE blocks. In the first case the position is the
    block. In the second, MoE blocks can be *interleaved* with dense ones
    (Qwen-MoE `decoder_sparse_step` / `mlp_only_layers`, DeepSeek
    `first_k_dense_replace` + `moe_layer_freq`, Llama-4
    `interleave_moe_layer_step`), so assuming they are the tail of the stack
    mislabels every layer. Config says which; the tail is the last resort.
    """
    present = [i for i, l in enumerate(logits) if l is not None]
    if len(logits) == n_blocks:
        return present
    n = len(present)
    moe = None
    if cfg is not None:
        mlp_only = set(getattr(cfg, "mlp_only_layers", None) or [])
        step = getattr(cfg, "decoder_sparse_step", None)
        step = step or getattr(cfg, "interleave_moe_layer_step", None)
        first_dense = getattr(cfg, "first_k_dense_replace", None)
        freq = getattr(cfg, "moe_layer_freq", None)
        if step:
            moe = [i for i in range(n_blocks)
                   if i not in mlp_only and (i + 1) % int(step) == 0]
        elif first_dense is not None:
            f = int(freq or 1)
            moe = [i for i in range(n_blocks) if i >= int(first_dense) and i % f == 0]
        elif mlp_only:
            moe = [i for i in range(n_blocks) if i not in mlp_only]
    if moe is not None and len(moe) == n:
        return moe
    first = max(0, n_blocks - n)
    return list(range(first, first + n))


def _routing_from(out, model, wanted: bool, spans=None):
    """Expert routing from a forward pass, or None.

    HF's sparse-MoE models return `router_logits` as one (T, n_experts)
    tensor per MoE layer when asked. Top-k over those is the routing decision
    itself, which is what makes it worth capturing: unlike an activation it
    needs no dictionary to read, and unlike attention it is a *discrete*
    choice the model made about this token.

    Dense models have no router. Rather than return an empty routing that
    later reads as "this token used no experts", this refuses — the same
    contract attention capture has on Mamba.

    `spans` un-batches the result. Router logits come back flattened over
    `(batch * tokens)`, so a batched pass needs to be told the padding: pass
    one `(row, start)` per sequence and get one `Routing` per sequence back,
    padding already dropped.
    """
    if not wanted:
        return None
    from trajectory import Routing

    logits = getattr(out, "router_logits", None)
    if not logits:
        raise ValueError(
            "this model returned no router logits — it is dense, or its "
            "implementation does not support output_router_logits. Routing "
            "capture only means something for a sparse-MoE model.")

    cfg = getattr(model, "config", None)
    cfg = getattr(cfg, "text_config", cfg)
    k = int(getattr(cfg, "num_experts_per_tok", 0) or 0)
    n_experts = int(getattr(cfg, "num_experts", 0)
                    or getattr(cfg, "n_routed_experts", 0) or 0)

    n_blocks = len(list(getattr(model, "model", model).layers)) \
        if hasattr(getattr(model, "model", model), "layers") else len(logits)
    blocks = moe_block_indices(cfg, n_blocks, list(logits))
    layers = [l for l in logits if l is not None]
    if not layers:
        raise ValueError("router logits present but empty")
    n_experts = n_experts or int(layers[0].shape[-1])
    k = k or min(2, n_experts)

    experts, weights, idx = [], [], []
    batch = len(spans) if spans else 1
    for m, lg in enumerate(layers):
        probs = torch.softmax(lg.float(), dim=-1)
        w, e = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
        if batch > 1:
            e = e.reshape(batch, -1, e.shape[-1])
            w = w.reshape(batch, -1, w.shape[-1])
        experts.append(e.cpu().numpy().astype(np.int32))
        weights.append((w / w.sum(-1, keepdim=True)).cpu().numpy().astype(np.float32))
        idx.append(blocks[m])

    experts, weights = np.stack(experts), np.stack(weights)
    layer_idx = np.asarray(idx, dtype=np.int32)
    if spans is None:
        return Routing(experts=experts, weights=weights,
                       layers=layer_idx, n_experts=n_experts)
    return [Routing(experts=experts[:, row, start:] if batch > 1 else experts[:, start:],
                    weights=weights[:, row, start:] if batch > 1 else weights[:, start:],
                    layers=layer_idx, n_experts=n_experts)
            for row, start in spans]
