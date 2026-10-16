"""Causal intervention: perturb-and-replay over a resumable forward pass.

Observation shows what a system *did*; intervention shows what it *would have
done*.  An intervention edits the process that generates a Trace and produces a
new **counterfactual StateTrajectory** — real DATA (the model genuinely ran and
produced it) that flows through the same projection / measurement / renderer
stack as the original.  The four layers never collapse: the counterfactual
trace is data, its distance from the baseline is a measurement, and any reading
of that ("commitment point") is a separate, defeasible narrative.

Supported edits (transformer backend):

    Perturb(layer, delta)    push a state by a vector  — the grab gesture
    SetState(layer, value)   overwrite a state
    InjectNoise(layer, scale) add Gaussian noise (seeded, reproducible)
    FreezeLayer(block)       skip a block's update (residual passes through)

All are applied by write-hooks during a single forward pass (`capture._run`),
so the model resumes from the edited residual and every downstream layer sees
the consequence.  Interventions require a torch model: the forward pass has
to be resumable from an edited state.

Fork semantics: `intervene(...)` returns a branch; `baseline` (an unperturbed
capture) plus one or more branches are what the UI overlays and diffs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from capture import _run
from trajectory import StateTrajectory


# --------------------------------------------------------------------------
# Intervention algebra
# --------------------------------------------------------------------------
@dataclass
class Intervention:
    """A single edit to the process at one point of the trajectory.

    `layer` indexes the state for state edits (0 = embeddings, k = output of
    block k-1) and the *block* for freezes.  `token=None` edits all positions.
    """

    layer: int
    kind: str                                   # perturb | set | noise | freeze
    token: int | None = None
    vector: np.ndarray | None = None            # delta (perturb) or value (set)
    scale: float = 0.0                          # noise magnitude
    seed: int | None = None

    def describe(self) -> str:
        where = "all tokens" if self.token is None else f"token {self.token}"
        return f"{self.kind}@layer{self.layer}[{where}]"

    def record(self) -> dict:
        """This edit as the analysis record states it (`provenance.record`).

        `describe()` reads the same for a push of 30 and one of 60, or for
        pushes toward two different tokens. A vector edit is named here by its
        norm and a hash of its exact values: enough to tell edits apart and to
        check a reproduction, without a D-dimensional array in the record.
        """
        rec = {"kind": self.kind, "layer": int(self.layer),
               "token": None if self.token is None else int(self.token)}
        if self.vector is not None:
            v = np.ascontiguousarray(self.vector, dtype="<f4")
            rec["norm"] = float(np.linalg.norm(v))
            # shape, then the float32 bytes, as provenance.sae_digest hashes
            rec["sha256"] = hashlib.sha256(
                repr(v.shape).encode("ascii") + v.tobytes()).hexdigest()
        if self.kind == "noise":
            rec["scale"] = float(self.scale)
            rec["seed"] = None if self.seed is None else int(self.seed)
        return rec


def Perturb(layer: int, delta, token: int | None = None) -> Intervention:
    """Add `delta` (a vector in state space) to the state at `layer`."""
    return Intervention(layer, "perturb", token=token, vector=np.asarray(delta, dtype=np.float32))


def SetState(layer: int, value, token: int | None = None) -> Intervention:
    """Replace the state at `layer` with `value`."""
    return Intervention(layer, "set", token=token, vector=np.asarray(value, dtype=np.float32))


def InjectNoise(layer: int, scale: float, token: int | None = None, seed: int | None = 0) -> Intervention:
    """Add zero-mean Gaussian noise of std `scale` to the state at `layer`."""
    return Intervention(layer, "noise", token=token, scale=float(scale), seed=seed)


def FreezeLayer(block: int) -> Intervention:
    """Skip block `block`'s update: hidden[block+1] := hidden[block]."""
    return Intervention(block, "freeze")


# --------------------------------------------------------------------------
# Torch edit closures (built lazily so numpy-only imports stay clean)
# --------------------------------------------------------------------------
def _make_edit(interventions: list[Intervention]):
    """Compose the state edits targeting one layer into a single fn(hidden)."""
    import torch

    def fn(hidden):
        h = hidden.clone()
        tok = slice(None)
        for iv in interventions:
            t = slice(None) if iv.token is None else iv.token
            if iv.kind == "perturb":
                d = torch.as_tensor(iv.vector, dtype=h.dtype, device=h.device)
                h[:, t] = h[:, t] + d
            elif iv.kind == "set":
                v = torch.as_tensor(iv.vector, dtype=h.dtype, device=h.device)
                h[:, t] = v
            elif iv.kind == "noise":
                g = torch.Generator(device=h.device)
                if iv.seed is not None:
                    g.manual_seed(int(iv.seed))
                shape = h[:, t].shape
                noise = torch.randn(shape, generator=g, dtype=h.dtype, device=h.device)
                h[:, t] = h[:, t] + iv.scale * noise
        return h

    return fn


def _compile(interventions: list[Intervention]) -> tuple[dict, set]:
    """Split interventions into per-layer state edits and frozen block ids."""
    frozen = {iv.layer for iv in interventions if iv.kind == "freeze"}
    by_layer: dict[int, list[Intervention]] = {}
    for iv in interventions:
        if iv.kind != "freeze":
            by_layer.setdefault(iv.layer, []).append(iv)
    state_edits = {layer: _make_edit(ivs) for layer, ivs in by_layer.items()}
    return state_edits, frozen


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def intervene(
    model,
    prompt: str,
    interventions: list[Intervention],
    tokenizer=None,
    top_k: int = 5,
    device: str = "auto",
    dtype: str = "float32",
    keep_logits: bool = True,
    capture_attention: bool = False,
    capture_components: bool = False,
) -> StateTrajectory:
    """Run a counterfactual forward pass under `interventions`.

    Returns a StateTrajectory whose `meta["interventions"]` records the edits
    (`Intervention.record`) and `meta["counterfactual"]` is True.  Requires a
    torch model (a HF instance with `tokenizer`, or a hub name).  Pass the
    baseline's `capture_attention`: it moves the pass onto the eager attention
    kernel, and a branch on another kernel differs from its baseline by
    rounding before any edit takes effect.  `capture_components` records each
    block's attention and MLP writes, as `capture` does.
    """
    if not interventions:
        raise ValueError("no interventions given; use capture() for a baseline pass")

    state_edits, frozen = _compile(interventions)
    return _run(
        model, prompt, tokenizer=tokenizer, top_k=top_k, device=device, dtype=dtype,
        keep_logits=keep_logits, state_edits=state_edits, frozen_blocks=frozen,
        capture_attention=capture_attention,
        capture_components=capture_components,
        extra_meta={
            "counterfactual": True,
            "interventions": [iv.record() for iv in interventions],
        },
    )


# --------------------------------------------------------------------------
# Branch comparison
# --------------------------------------------------------------------------
@dataclass
class Divergence:
    """How and where a branch separates from its baseline."""

    profile: np.ndarray          # (L,) state-space distance per layer
    onset: int                   # first layer whose distance exceeds the band
    readout_changed: int | None  # first layer whose top-1 prediction differs, or None
    token: int


def divergence(baseline: StateTrajectory, branch: StateTrajectory,
               token: int = -1, rel: float = 0.05) -> Divergence:
    """Layerwise separation of `branch` from `baseline` for one unit.

    `profile[l]` is the state-space (L2) distance between the two trajectories
    at layer l.  `onset` is the first layer whose distance exceeds `rel` times
    the maximum separation — a rough "when did the intervention start to
    matter" that is *deliberately* just a measurement, not a claimed cause.
    `readout_changed` is the first layer where the top-1 logit-lens token
    differs, when logits are available.
    """
    if baseline.hidden.shape != branch.hidden.shape:
        raise ValueError("baseline and branch must share shape (same prompt/model)")
    t = int(token) % baseline.n_tokens
    profile = np.linalg.norm(baseline.hidden[:, t] - branch.hidden[:, t], axis=-1)

    peak = float(profile.max())
    if peak <= 0:
        onset = len(profile) - 1
    else:
        exceed = np.flatnonzero(profile > rel * peak)
        onset = int(exceed[0]) if exceed.size else len(profile) - 1

    readout_changed = None
    if baseline.logits is not None and branch.logits is not None:
        ba = baseline.logits[:, t].astype(np.float32).argmax(axis=-1)
        br = branch.logits[:, t].astype(np.float32).argmax(axis=-1)
        diff = np.flatnonzero(ba != br)
        readout_changed = int(diff[0]) if diff.size else None

    return Divergence(profile=profile.astype(np.float32), onset=onset,
                      readout_changed=readout_changed, token=t)


# --------------------------------------------------------------------------
# Steering directions — deltas from data, not hand-picked numbers
# --------------------------------------------------------------------------
# A `Perturb` takes a raw delta vector; where that vector comes from is what
# separates a principled steer from a magic number. These helpers derive the
# direction from the model's own representations: a token's embedding
# direction, or a diff-of-means contrast between two sets of runs. The caller
# still chooses the magnitude (how far to push), but the *direction* is
# measured, so the intervention is reproducible and interpretable.

def _as_trajectories(x) -> list[StateTrajectory]:
    if isinstance(x, StateTrajectory):
        return [x]
    xs = list(x)
    if not xs:
        raise ValueError("expected at least one StateTrajectory")
    return xs


def direction_from_token(traj: StateTrajectory, token_id: int,
                         normalize: bool = True) -> np.ndarray:
    """Unit direction of a vocabulary token's embedding (the readout axis).

    For a model with tied embeddings (GPT-2) this row is also the unembedding
    direction, so pushing a late state along it moves the logit lens toward
    that token.
    """
    if traj.embedding_matrix is None:
        raise ValueError("trajectory has no embedding_matrix; capture with a "
                         "torch model (keep_logits=True) to get one")
    d = np.asarray(traj.embedding_matrix[int(token_id)], dtype=np.float32)
    if normalize:
        d = d / max(float(np.linalg.norm(d)), 1e-12)
    return d


def direction_from_contrast(pos, neg, layer: int, token: int = -1,
                            normalize: bool = True) -> np.ndarray:
    """Diff-of-means steering direction between two groups of runs.

    `pos` and `neg` are each a StateTrajectory or a list of them (e.g. runs
    that do vs. don't exhibit a behavior). The direction is the difference of
    the groups' mean hidden state at `(layer, token)` — the classic
    activation-steering construction. Backend-agnostic: it reads only
    `hidden`, so it works on any producer's runs.
    """
    pos_t, neg_t = _as_trajectories(pos), _as_trajectories(neg)

    def _mean(group: list[StateTrajectory]) -> np.ndarray:
        vecs = []
        for t in group:
            tok = int(token) % t.n_tokens
            vecs.append(np.asarray(t.hidden[int(layer), tok], dtype=np.float64))
        return np.mean(vecs, axis=0)

    d = (_mean(pos_t) - _mean(neg_t)).astype(np.float32)
    if normalize:
        d = d / max(float(np.linalg.norm(d)), 1e-12)
    return d


# --------------------------------------------------------------------------
# Faithfulness — how much of a steer's effect is the *direction*, not the norm
# --------------------------------------------------------------------------
@dataclass
class Faithfulness:
    """Effect-size of a targeted steer against a norm-matched random control.

    A perturbation of any direction, if large enough, disturbs the readout;
    the question that matters is whether *this* direction specifically drives
    the intended `target`. We answer it by comparing the steer to a random
    delta of identical norm:

        steer_shift   — logit-lens shift toward `target` under the steer
        control_shift — the same shift under a random delta of equal norm
        effect        — steer_shift - control_shift (direction-specific gain)

    A large positive `effect` means the direction, not merely the magnitude,
    is what moved the model. `steer_hits_target` reports whether the steered
    branch's final top-1 actually became the target. This is a measurement,
    not a mechanistic proof — it quantifies a counterfactual, it does not
    isolate a circuit.
    """

    target: int
    target_token: str | None
    steer_shift: float
    control_shift: float
    effect: float
    steer_hits_target: bool
    scale: float
    token: int


def target_logit_shift(baseline: StateTrajectory, branch: StateTrajectory,
                       target: int, token: int = -1, layer: int = -1) -> float:
    """Logit-lens shift toward `target` from baseline to branch at `layer`.

    Defaults to the final layer — the readout `faithfulness()` scores — but
    any captured layer can be read, which is what lets
    `persistence_profile()` trace the same effect down the stack instead of
    only at the end.
    """
    if baseline.logits is None or branch.logits is None:
        raise ValueError("both trajectories need logits (keep_logits=True)")
    t = int(token) % baseline.n_tokens
    l = int(layer) % baseline.n_layers
    base = float(baseline.logits[l, t, int(target)])
    new = float(branch.logits[l, t, int(target)])
    return new - base


def _norm_matched_random(delta: np.ndarray, seed: int = 0) -> np.ndarray:
    """A random delta with the same L2 norm as `delta` (the control push)."""
    delta = np.asarray(delta, dtype=np.float32)
    rng = np.random.default_rng(seed)
    r = rng.normal(size=delta.shape).astype(np.float32)
    r *= float(np.linalg.norm(delta)) / max(float(np.linalg.norm(r)), 1e-12)
    return r


def score_against_control(model, prompt: str, baseline: StateTrajectory,
                          branch: StateTrajectory, delta: np.ndarray, layer: int,
                          target: int, *, token: int | None = -1,
                          scale: float | None = None, tokenizer=None, seed: int = 0,
                          device: str = "auto", dtype: str = "float32",
                          top_k: int = 5,
                          capture_attention: bool = False) -> Faithfulness:
    """Score an already-computed steer against a norm-matched random control.

    Given `baseline` and the `branch` produced by applying `delta` at
    `(layer, token)`, apply a random delta of *identical* L2 norm and compare
    how far each moves the logit lens toward `target`. Factored out so both
    `faithfulness()` and `ui.run_intervention` — which already holds
    `baseline`/`branch` and must not pay for a second branch forward pass —
    construct the control the same way. `token=None` means the steer was
    applied at every position: both shifts are then averaged over all
    positions (the control is applied at every position too), and `token`
    in the result is the last position, whose top-1 `steer_hits_target`
    reads. `scale` records the steer magnitude and
    defaults to ``||delta||``. `capture_attention` is the one `baseline` was
    captured with, so the control runs on the same attention kernel (see
    `intervene`).
    """
    delta = np.asarray(delta, dtype=np.float32)
    r = _norm_matched_random(delta, seed)
    control = intervene(model, prompt, [Perturb(layer, r, token=token)],
                        tokenizer=tokenizer, top_k=top_k, device=device,
                        dtype=dtype, keep_logits=True,
                        capture_attention=capture_attention)

    # A steer applied at every position (token=None) moves every position,
    # so its effect is the mean shift over all of them; reading only the last
    # one credits the steer with what the earlier pushes did downstream.
    positions = (range(baseline.n_tokens) if token is None else [token])
    steer_shift = float(np.mean([target_logit_shift(baseline, branch, target, p)
                                 for p in positions]))
    control_shift = float(np.mean([target_logit_shift(baseline, control, target, p)
                                   for p in positions]))
    t = (baseline.n_tokens - 1) if token is None else int(token) % baseline.n_tokens
    hits = bool(int(np.asarray(branch.logits[-1, t]).argmax()) == int(target))
    target_token = None
    if baseline.vocab is not None and 0 <= int(target) < len(baseline.vocab):
        target_token = baseline.vocab[int(target)]

    return Faithfulness(
        target=int(target), target_token=target_token,
        steer_shift=steer_shift, control_shift=control_shift,
        effect=steer_shift - control_shift, steer_hits_target=hits,
        scale=float(np.linalg.norm(delta) if scale is None else scale), token=t,
    )


def faithfulness(model, prompt: str, layer: int, direction: np.ndarray,
                 target: int, *, token: int = -1, scale: float = 1.0,
                 tokenizer=None, baseline: StateTrajectory | None = None,
                 seed: int = 0, device: str = "auto", dtype: str = "float32",
                 top_k: int = 5) -> Faithfulness:
    """Score a targeted steer against a norm-matched random control.

    Normalizes `direction`, applies `scale * direction` as a `Perturb` at
    `(layer, token)`, then applies a random delta of the *same* L2 norm, and
    compares how far each moves the logit lens toward `target`. Requires a
    torch model. `baseline` is captured if not supplied.
    """
    from capture import capture

    direction = np.asarray(direction, dtype=np.float32)
    unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
    delta = (scale * unit).astype(np.float32)

    if baseline is None:
        baseline = capture(model, prompt, tokenizer=tokenizer, top_k=top_k,
                           device=device, dtype=dtype, keep_logits=True)

    branch = intervene(model, prompt, [Perturb(layer, delta, token=token)],
                       tokenizer=tokenizer, top_k=top_k, device=device,
                       dtype=dtype, keep_logits=True)

    return score_against_control(
        model, prompt, baseline, branch, delta, layer, target, token=token,
        scale=scale, tokenizer=tokenizer, seed=seed, device=device, dtype=dtype,
        top_k=top_k)


# --------------------------------------------------------------------------
# Persistence — is an injected effect carried forward, or rebuilt downstream?
# --------------------------------------------------------------------------
@dataclass
class PersistenceRecord:
    """One layer of a persistence profile.

    `effect_size` is the faithfulness computation — steer logit-lens shift
    toward the target minus the shift under a norm-matched random push —
    evaluated at this layer's logit lens instead of only the final one.
    `attn_share` / `mlp_share` are the baseline pass's residual-write
    decomposition for the block that wrote *into* this state
    (`metrics.component_shares` row `layer - 1`; NaN at layer 0, which no
    block writes).
    """

    layer: int
    effect_size: float
    steer_shift: float
    control_shift: float
    attn_share: float
    mlp_share: float


@dataclass
class PersistenceProfile:
    """Layer-by-layer fate of one injected direction — a measurement.

    One steer at `inject_layer` and one norm-matched random control, read at
    every downstream layer's logit lens. A curve that decays smoothly (or
    holds) says the residual stream carried the effect forward; a curve that
    drops and comes back, aligned with a spike in `mlp_share` at the layer
    where it returns, says a later block rewrote it. Like `Divergence`, this
    is *deliberately* just a measurement of what happened downstream of the
    edit — it locates where the effect lives, it does not identify the
    circuit that computes it.

    The final record is exactly `faithfulness()`'s scoring (same steer, same
    control, same readout), so the profile strictly generalizes it rather
    than reimplementing it. Indexing is by absolute layer:
    ``profile[inject_layer]`` is the first record and ``profile[-1]`` the
    final-layer one.
    """

    records: list[PersistenceRecord]
    inject_layer: int
    target: int
    target_token: str | None
    scale: float
    token: int
    seed: int

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, layer: int) -> PersistenceRecord:
        n = self.inject_layer + len(self.records)   # layers covered: inject..n-1
        l = int(layer) % n
        if l < self.inject_layer:
            raise IndexError(f"layer {l} precedes inject_layer {self.inject_layer}; "
                             f"the profile covers layers {self.inject_layer}..{n - 1}")
        return self.records[l - self.inject_layer]

    @property
    def layers(self) -> np.ndarray:
        return np.array([r.layer for r in self.records], dtype=int)

    @property
    def effect_sizes(self) -> np.ndarray:
        return np.array([r.effect_size for r in self.records], dtype=np.float32)

    @property
    def mlp_shares(self) -> np.ndarray:
        return np.array([r.mlp_share for r in self.records], dtype=np.float32)


def persistence_profile(model, prompt: str, direction: np.ndarray,
                        inject_layer: int, target: int, tokenizer=None, *,
                        token: int = -1, scale: float = 1.0,
                        baseline: StateTrajectory | None = None,
                        branch: StateTrajectory | None = None,
                        seed: int = 0, device: str = "auto",
                        dtype: str = "float32",
                        top_k: int = 5,
                        capture_attention: bool = False) -> PersistenceProfile:
    """Trace an injected direction's effect through every downstream layer.

    Applies `scale * unit(direction)` as a `Perturb` at `(inject_layer,
    token)` — exactly as `faithfulness()` does — plus a norm-matched random
    control, then evaluates the faithfulness effect (steer shift minus
    control shift toward `target`) at *each* layer from `inject_layer` to the
    final one, pairing every layer with the baseline pass's attention/MLP
    write shares from `metrics.component_shares`. Requires a torch model
    (inherited from `intervene`: the forward pass must be resumable).

    `baseline` is captured (with `capture_components=True`) if not supplied;
    a supplied baseline must carry the residual decomposition. `branch` — the
    already-computed steer at these exact settings — can be passed by callers
    that hold one (`pipeline.run_intervention`) to skip a forward pass, as
    with `score_against_control`. `capture_attention` applies to every pass
    run here and must match a supplied baseline's (see `intervene`).
    """
    from capture import capture
    from metrics import component_shares

    direction = np.asarray(direction, dtype=np.float32)
    unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
    delta = (scale * unit).astype(np.float32)

    if baseline is None:
        baseline = capture(model, prompt, tokenizer=tokenizer, top_k=top_k,
                           device=device, dtype=dtype, keep_logits=True,
                           capture_components=True,
                           capture_attention=capture_attention)
    shares = component_shares(baseline, token=token)          # (L-1, 2)
    inject_layer = int(inject_layer) % baseline.n_layers

    if branch is None:
        branch = intervene(model, prompt, [Perturb(inject_layer, delta, token=token)],
                           tokenizer=tokenizer, top_k=top_k, device=device,
                           dtype=dtype, keep_logits=True,
                           capture_attention=capture_attention)
    control = intervene(model, prompt,
                        [Perturb(inject_layer, _norm_matched_random(delta, seed),
                                 token=token)],
                        tokenizer=tokenizer, top_k=top_k, device=device,
                        dtype=dtype, keep_logits=True,
                        capture_attention=capture_attention)

    records = []
    for layer in range(inject_layer, baseline.n_layers):
        steer = target_logit_shift(baseline, branch, target, token=token, layer=layer)
        ctrl = target_logit_shift(baseline, control, target, token=token, layer=layer)
        attn, mlp = ((float(shares[layer - 1, 0]), float(shares[layer - 1, 1]))
                     if layer > 0 else (float("nan"), float("nan")))
        records.append(PersistenceRecord(
            layer=layer, effect_size=steer - ctrl,
            steer_shift=steer, control_shift=ctrl,
            attn_share=attn, mlp_share=mlp))

    target_token = None
    if baseline.vocab is not None and 0 <= int(target) < len(baseline.vocab):
        target_token = baseline.vocab[int(target)]

    return PersistenceProfile(
        records=records, inject_layer=inject_layer, target=int(target),
        target_token=target_token, scale=float(np.linalg.norm(delta)),
        token=int(token) % baseline.n_tokens, seed=int(seed))
