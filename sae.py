"""Phase 3 — sparse-autoencoder features over the residual stream.

An SAE re-expresses a hidden state as a sparse combination of learned
dictionary directions ("features"):

    f = ReLU((h - b_dec) @ w_enc + b_enc)        # (…, F) activations
    h ≈ f @ w_dec + b_dec                        # reconstruction

This module *applies* SAEs — it never trains them (a non-goal).  Weights
come from `load_npz` (export any pretrained SAE — SAELens, dictionary-
learning runs — to the four arrays below), or from `demo_sae`, an untrained
random dictionary that exercises the whole feature pipeline without a
download.  Demo activations are sparse projections, NOT interpretable
features; they exist so overlays, tests and UI development work offline.

Everything is a pure numpy function over StateTrajectories, consistent with
the architecture: no torch, no transformer internals, any backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trajectory import StateTrajectory


@dataclass
class SAE:
    """A sparse autoencoder as plain arrays.

    w_enc : (D, F)   encoder weights
    b_enc : (F,)     encoder bias
    w_dec : (F, D)   decoder dictionary (rows are feature directions)
    b_dec : (D,)     decoder bias (subtracted before encoding)
    labels: optional human-readable name per feature
    """

    w_enc: np.ndarray
    b_enc: np.ndarray
    w_dec: np.ndarray
    b_dec: np.ndarray
    labels: list[str] | None = None

    @property
    def dim(self) -> int:
        return self.w_enc.shape[0]

    @property
    def n_features(self) -> int:
        return self.w_enc.shape[1]

    def validate(self) -> None:
        D, F = self.w_enc.shape
        if self.b_enc.shape != (F,):
            raise ValueError(f"b_enc shape {self.b_enc.shape} != {(F,)}")
        if self.w_dec.shape != (F, D):
            raise ValueError(f"w_dec shape {self.w_dec.shape} != {(F, D)}")
        if self.b_dec.shape != (D,):
            raise ValueError(f"b_dec shape {self.b_dec.shape} != {(D,)}")
        if self.labels is not None and len(self.labels) != F:
            raise ValueError(f"labels ({len(self.labels)}) != F ({F})")

    # ------------------------------------------------------------- transforms
    def encode(self, hidden: np.ndarray) -> np.ndarray:
        """(…, D) states -> (…, F) non-negative sparse activations."""
        h = np.asarray(hidden, dtype=np.float32)
        pre = (h - self.b_dec) @ self.w_enc + self.b_enc
        return np.maximum(pre, 0.0)

    def decode(self, acts: np.ndarray) -> np.ndarray:
        """(…, F) activations -> (…, D) reconstructed states."""
        return np.asarray(acts, dtype=np.float32) @ self.w_dec + self.b_dec

    def reconstruct(self, hidden: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(hidden))

    def reconstruction_error(self, hidden: np.ndarray) -> np.ndarray:
        """Relative L2 reconstruction error per state: (…,)."""
        h = np.asarray(hidden, dtype=np.float32)
        err = np.linalg.norm(h - self.reconstruct(h), axis=-1)
        return err / np.maximum(np.linalg.norm(h, axis=-1), 1e-12)

    def feature_label(self, i: int) -> str:
        if self.labels is not None:
            return self.labels[i]
        return f"f{i}"


# --------------------------------------------------------------------- IO
def save_npz(sae: SAE, path: str | Path) -> None:
    """Persist an SAE as a portable .npz (the interchange format)."""
    sae.validate()
    arrays = {"w_enc": sae.w_enc, "b_enc": sae.b_enc,
              "w_dec": sae.w_dec, "b_dec": sae.b_dec}
    if sae.labels is not None:
        arrays["labels"] = np.asarray(sae.labels)
    np.savez_compressed(path, **arrays)


def load_npz(path: str | Path) -> SAE:
    """Load an SAE saved by `save_npz` (or exported from any trainer)."""
    with np.load(path, allow_pickle=False) as z:
        sae = SAE(
            w_enc=z["w_enc"].astype(np.float32),
            b_enc=z["b_enc"].astype(np.float32),
            w_dec=z["w_dec"].astype(np.float32),
            b_dec=z["b_dec"].astype(np.float32),
            labels=[str(s) for s in z["labels"]] if "labels" in z else None,
        )
    sae.validate()
    return sae


def _to_numpy(x) -> np.ndarray:
    """Array from a numpy array or a torch tensor, without importing torch."""
    if hasattr(x, "detach"):          # torch.Tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _orient(w, rows: int, cols: int, name: str) -> np.ndarray:
    """Return `w` as (rows, cols), transposing if it arrived the other way.

    Trainers disagree on weight orientation; we pin ours to Mottled's
    convention (w_enc is (D, F), w_dec is (F, D)) and accept either layout.
    """
    w = _to_numpy(w)
    if w.shape == (rows, cols):
        return w
    if w.shape == (cols, rows):
        return np.ascontiguousarray(w.T)
    raise ValueError(f"{name} shape {w.shape} is neither {(rows, cols)} nor its transpose")


def from_state_dict(state_dict, labels: list[str] | None = None) -> SAE:
    """Build an SAE from a pretrained weight mapping (the real-weights path).

    Accepts the standard four-array convention shared by SAELens and most
    dictionary-learning runs — keys `W_enc`/`b_enc`/`W_dec`/`b_dec` (case
    tolerant) — as numpy arrays or torch tensors. `b_dec` fixes the hidden
    dimension D, `b_enc` the feature count F; encoder/decoder weights are
    oriented to Mottled's (D, F) / (F, D) layout automatically. This applies
    a *trained* dictionary — the whole point of the feature overlay: real
    features instead of `demo_sae`'s random directions.
    """
    def _pick(*names):
        for n in names:
            for key in (n, n.lower(), n.upper()):
                if key in state_dict:
                    return state_dict[key]
        raise KeyError(f"state dict is missing any of {names}; keys: {list(state_dict)[:8]}")

    b_dec = _to_numpy(_pick("b_dec"))
    b_enc = _to_numpy(_pick("b_enc"))
    D, F = int(b_dec.shape[0]), int(b_enc.shape[0])
    w_enc = _orient(_pick("W_enc", "w_enc"), D, F, "W_enc")
    w_dec = _orient(_pick("W_dec", "w_dec"), F, D, "W_dec")
    sae = SAE(w_enc=w_enc, b_enc=b_enc, w_dec=w_dec, b_dec=b_dec, labels=labels)
    sae.validate()
    return sae


def from_sae_lens(sae, labels: list[str] | None = None) -> SAE:
    """Convert a loaded SAELens SAE object into a Mottled SAE.

    SAELens' *standard* SAE stores `W_enc` (d_in, d_sae), `b_enc`, `W_dec`
    (d_sae, d_in), `b_dec` and runs the same ReLU forward Mottled applies, so
    conversion is a direct array copy. Two config options change that forward,
    and we handle each explicitly rather than silently mis-encoding a nominally
    "standard" SAE:

    * ``apply_b_dec_to_input=False`` — the source encodes ``h @ W_enc + b_enc``
      without the ``b_dec`` subtraction Mottled always does. That constant folds
      exactly into ``b_enc`` (``b_enc += b_dec @ W_enc``), so the converted
      encoder stays numerically identical.
    * ``normalize_activations`` — a runtime input rescaling Mottled does not do
      and cannot fold into a fixed dictionary, so it is rejected loudly.

    Non-standard architectures (gated / jumprelu / top-k) use a different
    nonlinearity than our plain ReLU and are rejected — detected from
    ``cfg.architecture`` / ``activation_fn`` and, so the gate cannot fail open
    when a config is missing, from the tell-tale weights they carry.
    """
    cfg = getattr(sae, "cfg", None)

    def _cfg(name):
        v = getattr(cfg, name, None) if cfg is not None else None
        return v() if callable(v) else v

    _require_standard(_cfg("architecture"),
                      _cfg("activation_fn") or _cfg("activation_fn_str"),
                      _cfg("normalize_activations"))
    # Fail closed even if the config is absent: these weights exist only on
    # gated SAEs, whose forward is not our ReLU.
    for attr, kind in (("W_gate", "gated"), ("r_mag", "gated")):
        if getattr(sae, attr, None) is not None:
            raise ValueError(
                f"this looks like a {kind} SAE (carries {attr!r}); Mottled applies "
                "a plain ReLU. Convert it with its own tooling first.")
    out = from_state_dict({"W_enc": sae.W_enc, "b_enc": sae.b_enc,
                           "W_dec": sae.W_dec, "b_dec": sae.b_dec}, labels=labels)
    if _cfg("apply_b_dec_to_input") is False:
        out.b_enc = _fold_b_dec(out)
        out.validate()
    return out


def _require_standard(arch, act, norm) -> None:
    """Reject SAE configs whose forward is not Mottled's plain ReLU."""
    if arch is not None and str(arch).lower() not in {"standard", "relu"}:
        raise ValueError(
            f"unsupported SAE architecture {arch!r}: Mottled applies a plain ReLU "
            "(standard) SAE. Gated / JumpReLU / top-k SAEs encode differently and "
            "must be converted with their own tooling first.")
    if act is not None and str(act).lower() not in {"relu", "", "none"}:
        raise ValueError(
            f"unsupported SAE activation_fn {act!r}: Mottled applies a plain ReLU. "
            "Top-k / other activations must be converted with their own tooling.")
    if norm not in (None, False, "none", "None"):
        raise ValueError(
            f"this SAE normalizes activations (normalize_activations={norm!r}), an "
            "input rescaling Mottled does not apply; its features would not match. "
            "Export it without activation normalization first.")


def _fold_b_dec(out: SAE) -> np.ndarray:
    """b_enc with the missing b_dec subtraction folded in, exactly:
    (h - b_dec) @ W_enc + (b_enc + b_dec @ W_enc) == h @ W_enc + b_enc."""
    return (out.b_enc + out.b_dec @ out.w_enc).astype(np.float32)


def fetch_from_hub(
    repo_id: str = "jbloom/GPT2-Small-SAEs-Reformatted",
    subfolder: str = "blocks.8.hook_resid_pre",
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> SAE:
    """Download a trained SAE from a SAELens-format Hugging Face repo.

    The real-weights path with no ``sae-lens`` dependency: SAELens releases
    store ``sae_weights.safetensors`` (the standard four arrays) next to a
    ``cfg.json`` per hook point. This fetches both through
    ``huggingface_hub`` (cached under the standard HF cache, integrity
    checked against the repo's manifest), applies the same fail-closed
    gates as ``from_sae_lens`` (standard ReLU forward only, no activation
    normalization, exact ``apply_b_dec_to_input`` fold), and returns a
    ready SAE.

    The default is Joseph Bloom's GPT-2-small residual-stream release —
    ~150 MB on first fetch, then cached. Remember what an SAE is calibrated
    on: this dictionary was trained at one hook point (layer 8's
    ``resid_pre``); applying it at other layers is extrapolation, and the
    UI says so.

    Needs ``huggingface_hub`` + ``safetensors`` (both ship with the
    ``models`` extra via transformers).
    """
    import json

    try:
        from huggingface_hub import hf_hub_download
        from safetensors.numpy import load_file
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fetching a trained SAE needs huggingface_hub and safetensors "
            '(pip install "mottled[models]")') from exc

    common = {"repo_id": repo_id, "subfolder": subfolder,
              "revision": revision, "cache_dir": cache_dir}
    cfg_path = hf_hub_download(filename="cfg.json", **common)
    cfg = json.loads(Path(cfg_path).read_text())
    _require_standard(cfg.get("architecture"),
                      cfg.get("activation_fn_str") or cfg.get("activation_fn"),
                      cfg.get("normalize_activations"))

    weights_path = hf_hub_download(filename="sae_weights.safetensors", **common)
    out = from_state_dict(dict(load_file(weights_path)))
    if cfg.get("apply_b_dec_to_input") is False:
        out.b_enc = _fold_b_dec(out)
        out.validate()
    return out


@dataclass
class SAEFit:
    """Measured fit of a dictionary to one capture — not assumed, computed.

    recon_error : (L,) median relative L2 reconstruction error per layer
    active_frac : (L,) mean fraction of features firing per layer
    best_layer  : layer where the dictionary reconstructs best
    """

    recon_error: np.ndarray
    active_frac: np.ndarray
    best_layer: int

    @property
    def best_error(self) -> float:
        return float(self.recon_error[self.best_layer])


def fit_report(sae: SAE, traj: StateTrajectory) -> SAEFit:
    """How well does this dictionary fit this capture's activations?

    An SAE is only calibrated to the activation distribution it was trained
    on — a specific model, hook point, *and preprocessing* (TransformerLens'
    weight folding/centering changes residual values while preserving the
    function, so a TL-trained SAE mis-fits raw HF states). This measures the
    fit instead of trusting provenance: reconstruction error and firing
    density per layer. A best-layer error near the SAE's training loss
    (~0.1–0.3 for public residual SAEs) means calibrated; errors ≫ 1 mean
    the features shown are extrapolation, and the UI says so.
    """
    hidden = traj.hidden.astype(np.float32)
    err = np.median(sae.reconstruction_error(hidden), axis=-1)     # (L,)
    active = (sae.encode(hidden) > 0).mean(axis=(1, 2))            # (L,)
    return SAEFit(recon_error=err.astype(np.float32),
                  active_frac=active.astype(np.float32),
                  best_layer=int(np.argmin(err)))


def demo_sae(dim: int, n_features: int = 256, seed: int = 0, bias: float = 0.05) -> SAE:
    """Untrained random dictionary: tied weights, unit decoder rows.

    A negative encoder bias keeps activations sparse.  Deterministic for a
    (dim, n_features, seed) triple.  For development and tests only — the
    "features" are arbitrary directions, not learned structure.
    """
    rng = np.random.default_rng(seed)
    w_dec = rng.normal(size=(n_features, dim)).astype(np.float32)
    w_dec /= np.linalg.norm(w_dec, axis=1, keepdims=True)
    return SAE(
        w_enc=w_dec.T.copy(),
        b_enc=np.full(n_features, -float(bias), dtype=np.float32),
        w_dec=w_dec,
        b_dec=np.zeros(dim, dtype=np.float32),
    )


# ---------------------------------------------------------------- features
def feature_trajectory(traj: StateTrajectory, sae: SAE) -> np.ndarray:
    """Feature activations for every captured state: (L, T, F).

    The SAE dictionary must match the trajectory's hidden dimension —
    features are directions in that state space.
    """
    if sae.dim != traj.dim:
        raise ValueError(f"SAE dim {sae.dim} != trajectory dim {traj.dim}")
    return sae.encode(traj.hidden)


def top_features(acts: np.ndarray, layer: int, token: int, k: int = 5) -> list[tuple[int, float]]:
    """Strongest-activating features at one state: [(feature_id, activation)].

    Only features that actually fire (activation > 0) are returned, so the
    list may be shorter than k.
    """
    a = np.asarray(acts)[layer, token]
    order = np.argsort(-a)[:k]
    return [(int(i), float(a[i])) for i in order if a[i] > 0]


def active_features(acts: np.ndarray, k: int = 20) -> np.ndarray:
    """Feature ids ranked by peak activation anywhere in the trajectory."""
    peak = np.asarray(acts).max(axis=(0, 1))
    order = np.argsort(-peak)[:k]
    return order[peak[order] > 0]


# ------------------------------------------------------------ feature field
@dataclass
class DomainSummary:
    """One feature's territory on the projection plane."""

    feature: int
    area: float        # fraction of the firing plane this feature dominates
    x: float           # centroid, in plane coordinates
    y: float


@dataclass
class FeatureField:
    """The SAE evaluated over the projection plane itself.

    The domain-coloring analogue: where a complex-plane plot evaluates f(z)
    at every point of the plane and shows arg(f) as hue and |f| as
    brightness, this evaluates the SAE at every grid point of the projected
    manifold — `dominant` (which feature decodes strongest) plays the role
    of the phase, `magnitude` (its activation) the modulus.
    """

    grid_x: np.ndarray      # (W,)
    grid_y: np.ndarray      # (H,)
    magnitude: np.ndarray   # (H, W) strongest activation at each plane point
    dominant: np.ndarray    # (H, W) id of the strongest feature; -1 where none fires
    n_active: np.ndarray    # (H, W) how many features fire at each point

    @property
    def features(self) -> np.ndarray:
        """Sorted ids of every feature that dominates somewhere."""
        ids = np.unique(self.dominant)
        return ids[ids >= 0]

    def domains(self, k: int = 6) -> list["DomainSummary"]:
        """The k largest territories, biggest first.

        Where the field is worth labelling: a domain-colored plane has one
        dominant feature per point, and only the few that own real area are
        readable. Centroids are the mean of a feature's cells, so a
        disconnected territory reports the middle of its scatter — fine for
        placing a name, not a claim about shape.
        """
        out = []
        rows, cols = np.indices(self.dominant.shape)
        total = float((self.dominant >= 0).sum())
        if total <= 0:
            return out
        for fid in self.features:
            mask = self.dominant == fid
            n = int(mask.sum())
            yi, xi = float(rows[mask].mean()), float(cols[mask].mean())
            out.append(DomainSummary(
                feature=int(fid), area=n / total,
                x=float(np.interp(xi, np.arange(len(self.grid_x)), self.grid_x)),
                y=float(np.interp(yi, np.arange(len(self.grid_y)), self.grid_y))))
        out.sort(key=lambda d: -d.area)
        return out[:k]


def feature_field(sae: SAE, projector, grid_x: np.ndarray, grid_y: np.ndarray) -> FeatureField:
    """Evaluate `sae` over a grid on the projection plane: the domain coloring.

    Each (x, y) grid point is inverse-projected back to hidden space and
    encoded; the field records, per point, the strongest feature and its
    activation.  Requires an invertible projector (PCA is exact — grid
    points map onto the fitted affine 2-plane in hidden space, so the field
    shows what the SAE dictionary sees *along the plane you are looking
    at*; UMAP's inverse is approximate).
    """
    if not hasattr(projector, "inverse_transform"):
        raise ValueError(
            f"projection {type(projector).__name__} has no inverse_transform; "
            "the feature field needs plane -> hidden-space inversion (use pca)")
    gx, gy = np.meshgrid(np.asarray(grid_x), np.asarray(grid_y))
    plane = np.column_stack([gx.ravel(), gy.ravel()])
    hidden = np.asarray(projector.inverse_transform(plane))
    if hidden.shape[-1] != sae.dim:
        raise ValueError(f"projector inverts to dim {hidden.shape[-1]}, SAE dim is {sae.dim}")

    acts = sae.encode(hidden)                      # (H*W, F)
    magnitude = acts.max(axis=-1)
    dominant = acts.argmax(axis=-1).astype(np.int32)
    dominant[magnitude <= 0] = -1
    shape = gx.shape
    return FeatureField(
        grid_x=np.asarray(grid_x, dtype=np.float32),
        grid_y=np.asarray(grid_y, dtype=np.float32),
        magnitude=magnitude.reshape(shape).astype(np.float32),
        dominant=dominant.reshape(shape),
        n_active=(acts > 0).sum(axis=-1).reshape(shape).astype(np.int32),
    )


# ------------------------------------------------------------ feature labels
# Neuronpedia hosts auto-interpretability explanations for the public SAEs.
# A dictionary's features are just indices until something names them, and an
# unnamed feature overlay is a colour with no meaning.
_NEURONPEDIA = "https://www.neuronpedia.org/api/feature/{model}/{sae}/{index}"

# Mottled's SAE sources -> the (model, sae) pair Neuronpedia knows them by.
NEURONPEDIA_SOURCES = {
    ("jbloom/GPT2-Small-SAEs-Reformatted", "blocks.8.hook_resid_pre"):
        ("gpt2-small", "8-res-jb"),
}


@dataclass
class FeatureLabel:
    """An auto-generated explanation of one SAE feature.

    These are **not** ground truth. Each is written by a language model
    reading the feature's top activations, so `explained_by` and `method`
    travel with the text and surfaces are expected to say so. `score` is the
    auto-interp score when the source published one, and is often absent.
    """

    index: int
    description: str
    explained_by: str | None = None
    method: str | None = None
    score: float | None = None

    def __str__(self) -> str:
        return f"f{self.index} · {self.description}"


def fetch_labels(
    indices,
    source: tuple[str, str] | None = None,
    model_id: str | None = None,
    sae_id: str | None = None,
    cache_dir: str | Path | None = None,
    timeout: float = 10.0,
    fetch=None,
) -> dict[int, FeatureLabel]:
    """Look up auto-interp explanations for specific features.

    Deliberately lazy and bounded: pass only the features you are about to
    show (a scene has a handful active out of ~24k), because there is no bulk
    endpoint without an API key. Results are cached on disk, so a second look
    costs nothing and an offline run still works.

    Never raises for network reasons — an unreachable source yields no labels,
    and the caller falls back to bare indices. It does not fail *silently*
    either: lookups that failed are counted and reported in one line on
    stderr (with the first error), so a scene missing its labels says why.
    `fetch` is injectable for tests.
    """
    import json
    import urllib.request

    if source is not None:
        model_id, sae_id = NEURONPEDIA_SOURCES.get(tuple(source), (None, None))
    if not model_id or not sae_id:
        return {}

    import cache as cache_mod

    root = Path(cache_dir) if cache_dir else cache_mod.default_dir("labels")
    cache_file = root / f"{model_id}__{sae_id}.json"
    cached: dict = {}
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
        except (OSError, ValueError):
            cached = {}

    def _default_fetch(url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": "mottled"})
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return json.loads(res.read())

    fetch = fetch or _default_fetch
    out: dict[int, FeatureLabel] = {}
    fresh = False
    failed: list[int] = []
    first_error: Exception | None = None
    wanted = sorted({int(i) for i in indices})
    for i in wanted:
        key = str(i)
        if key not in cached:
            try:
                payload = fetch(_NEURONPEDIA.format(model=model_id, sae=sae_id, index=i))
                exps = (payload or {}).get("explanations") or []
                cached[key] = _first_explanation(exps)
                fresh = True
            except Exception as exc:  # offline, rate-limited, changed shape …
                failed.append(i)
                first_error = first_error or exc
                continue
        entry = cached.get(key)
        if entry:
            out[i] = FeatureLabel(index=i, description=str(entry.get("description", "")),
                                  explained_by=entry.get("explained_by"),
                                  method=entry.get("method"),
                                  score=entry.get("score"))
    if failed:
        import sys

        shown = ", ".join(str(i) for i in failed[:5]) + (" …" if len(failed) > 5 else "")
        print(f"mottled: warning: {len(failed)} of {len(wanted)} Neuronpedia label "
              f"lookups failed for {model_id}/{sae_id} (features {shown}): "
              f"{type(first_error).__name__}: {first_error}; those features "
              f"are shown by index only", file=sys.stderr)
    if fresh:
        try:
            root.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cached))
        except OSError as exc:
            import sys

            print(f"mottled: warning: could not write the label cache "
                  f"{cache_file}: {exc}", file=sys.stderr)
    return out


def _first_explanation(explanations: list) -> dict | None:
    """The first usable explanation record, flattened to what we keep."""
    for e in explanations:
        desc = (e or {}).get("description")
        if desc:
            scores = e.get("scores") or []
            score = None
            for s in scores:
                if isinstance(s, dict) and isinstance(s.get("value"), (int, float)):
                    score = float(s["value"])
                    break
            return {"description": str(desc),
                    "explained_by": e.get("explanationModelName"),
                    "method": e.get("typeName"),
                    "score": score}
    return None


def apply_labels(sae: SAE, labels: dict) -> SAE:
    """Write fetched labels onto a dictionary's `labels` list, in place.

    Features without an explanation keep their bare `fN` name, so a partial
    lookup degrades to exactly what the dictionary showed before.
    """
    if sae.labels is None:
        sae.labels = [f"f{i}" for i in range(sae.n_features)]
    for i, label in labels.items():
        if 0 <= int(i) < sae.n_features and label.description:
            sae.labels[int(i)] = f"f{i} · {label.description}"
    return sae
