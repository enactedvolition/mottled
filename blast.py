"""The shotgun blast: one pellet per prompt, followed through every layer.

The token-trajectory scene draws every (layer, token) state through one global
projection, and on real captures that projection spends its axes on what the
reader already knows: in the bundled GPT-2 capitals scene one attention-sink
token's path is ~15x the others' and which prompt a state came from explains
none of the on-screen variance. Here each prompt is one pellet, read at one
position (by default the last token, the state a deployed monitor would read
before the model writes), and every layer gets its own frame around the
family's centroid. Where every prompt ends on the same template token and the
model adds position only inside attention (rotary, as Qwen and Llama do), the
layer-0 pellets are one point: the muzzle is in the data, and the spread with
depth is the content. With learned absolute positions (GPT-2) layer 0 already
differs by prompt length, and length is a channel every layer can carry.

A pellet family is an ordinary StateTrajectory whose T axis indexes pellets
(`pellets`), so nothing here reaches into a model. Two layouts:

  monitor  x = the score a deployed linear monitor would read: each pellet's
           dot product with the diff-of-means direction for one label,
           K-fold cross-fitted, so a labelled pellet is scored on a direction
           fitted without it: in-sample, that direction ranks even labels
           with no held-out signal highly, because 48 points in 896-d leave
           it room to. y = the direction most pellets share in what x leaves
           out, carrying no label. Supervised, so it needs labels and says
           which one drives x.
  open     unsupervised. One camera per layer, built from the pellet patterns
           the layers agree on (a multi-table PCA of unit bearings), each
           pellet drawn with a camera that leaves out its own bearing at
           that depth. Screen radius never exceeds the pellet's full-space
           distance from the origin.

Both were chosen against criteria fixed in advance on 192 deployment prompts
(refusal, sycophancy, prompt injection, truthfulness) on Qwen2.5-0.5B-Instruct.
A split on screen is a readout of these prompts under this view, to be
confirmed on held-out prompts, not evidence of mechanism (docs/validity.md).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from projection import neighborhood_preservation
from trajectory import StateTrajectory

# A label drives a monitor only with at least this many labelled pellets per
# class: fewer and its held-out AUROC can only take a handful of values.
MIN_PER_CLASS = 3
FOLDS = 5
NULL_DRAWS = 200
# Pellets whose flights stay within this fraction of the family's RMS spread
# of each other at every depth are copies of one flight and share one vote in
# choosing the open camera, so one reply repeated four times is not four
# votes for its own direction. Declared, not tuned per scene, and it is a
# cut, not a gap: on the truth sample it merges identical replies and a
# true/false pair one word apart. The merged pairs and those near the cut are
# reported in the layout's params.
COPY_TOL = 0.5
# relative: a layer whose deviations are all below this is one point (layer 0
# of a generation-prompt family is bit-identical) and gets no direction
_POINT_TOL = 1e-9
_NBHD_K = 5
_LABELS = ("monitor", "open")


@dataclass
class BlastConfig:
    """What a blast export ran under: the analysis record's `config` block
    (provenance.record), in place of a MarbleConfig whose terrain knobs a
    blast does not use."""

    model: str
    chat: bool = False
    position: int = -1
    methods: tuple[str, ...] = ("monitor", "open")
    seed: int = 0
    folds: int = FOLDS
    null_draws: int = NULL_DRAWS
    copy_tol: float = COPY_TOL
    top_k: int = 5
    device: str = "auto"
    dtype: str = "float32"

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return {"kind": "blast", **asdict(self), "methods": list(self.methods)}


@dataclass
class BlastLayout:
    """One way to draw a pellet family. `positions` (N, L, 2) are in units of
    each layer's mean state norm, around that layer's centroid."""

    name: str
    method: str
    positions: np.ndarray                  # (N, L, 2)
    quality: np.ndarray                    # (L, N) neighbourhood preservation
    exact: list[str]
    projected: list[str]
    fitted: list[str] = field(default_factory=list)
    driver: str | None = None
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------- pellets
def pellets(trajs: list[StateTrajectory], ids: list[str],
            labels: list[dict] | None = None, texts: list[str] | None = None,
            position: int = -1, groups: list[str | None] | None = None) -> StateTrajectory:
    """Stack one position of each trajectory into a pellet family.

    Returns a StateTrajectory with hidden (L, N, D): the T axis indexes
    pellets, `tokens` are the pellet ids, entropy and top-k are read at the
    same position, and meta["pellets"] keeps each pellet's id, text and
    labels. Every trajectory must come from the same model (same depth and
    width): a family shares one frame per layer. `groups` names pellets that
    must be held out together by the monitor's cross-fitting (a contrast
    pair: two answers to one question).
    """
    if not trajs:
        raise ValueError("pellets needs at least one trajectory")
    if len(ids) != len(trajs) or len(set(ids)) != len(ids):
        raise ValueError("pellet ids must be unique, one per trajectory")
    labels = labels or [{} for _ in trajs]
    texts = texts or [t.meta.get("prompt", "") for t in trajs]
    L, D = trajs[0].n_layers, trajs[0].dim
    for t in trajs:
        if (t.n_layers, t.dim) != (L, D):
            raise ValueError("a pellet family needs one depth and width; got "
                             f"{(t.n_layers, t.dim)} beside {(L, D)}")
    hidden = np.stack([t.hidden[:, position] for t in trajs], axis=1)
    entropy = (np.stack([t.entropy[:, position] for t in trajs], axis=1)
               if all(t.entropy is not None for t in trajs) else None)
    topk = ([[t.topk[l][position] for t in trajs] for l in range(L)]
            if all(t.topk is not None for t in trajs) else None)
    meta = dict(trajs[0].meta)
    # the first trajectory's own prompt and token positions describe it, not
    # the family; each pellet's read position is in its own entry below
    meta.pop("prompt", None)
    meta.pop("positions", None)
    meta["axis"] = "pellets"
    meta["position"] = position
    meta["pellets"] = [{"id": str(i), "text": str(x), "labels": dict(lab)}
                       for i, x, lab in zip(ids, texts, labels)]
    for p, g in zip(meta["pellets"], groups or []):
        if g is not None:
            p["group"] = str(g)
    for p, t in zip(meta["pellets"], trajs):
        if "positions" in t.meta:
            p["read_at"] = int(t.meta["positions"][position])
        p["read_token"] = str(t.tokens[position])
    return StateTrajectory(hidden=hidden.astype(np.float32), tokens=[str(i) for i in ids],
                           entropy=entropy, topk=topk, meta=meta)


def label_vector(traj: StateTrajectory, name: str) -> np.ndarray:
    """(N,) int: 1 / 0 where a pellet carries `name`, -1 where it does not."""
    out = []
    for p in traj.meta.get("pellets", []):
        v = p.get("labels", {}).get(name)
        # strictly 0/1: bool("0") is True, and a label read the wrong way
        # round is a monitor pointed the wrong way
        if v is not None and v not in (0, 1):
            raise ValueError(f"label {name!r} of pellet {p.get('id')!r} must be "
                             f"0, 1, true, false or null; got {v!r}")
        out.append(-1 if v is None else int(v))
    return np.asarray(out, dtype=np.int64)


# ------------------------------------------------------------------ frame
def frame(hidden: np.ndarray) -> dict:
    """Per-layer frame of a pellet family, (L, N, D) in.

    dev (L, N, D): each pellet's deviation from the layer centroid, in units
    of the layer's mean state norm (so the residual stream's growth in norm
    with depth does not pass for a blast); range (N, L) its exact length;
    spread (L,) the family's RMS range; norm (L,) the unit.
    """
    X = np.asarray(hidden, dtype=np.float64)
    if not np.isfinite(X).all():
        l, i = np.argwhere(~np.isfinite(X).all(axis=2))[0]
        raise ValueError(f"pellet {i} has a non-finite state at layer {l}")
    norm = np.linalg.norm(X, axis=2).mean(axis=1)
    dev = (X - X.mean(axis=1, keepdims=True)) / np.maximum(norm, 1e-30)[:, None, None]
    rng = np.linalg.norm(dev, axis=2).T
    return {"dev": dev, "range": rng, "spread": np.sqrt((rng ** 2).mean(axis=0)),
            "norm": norm}


def _is_point(dev_l: np.ndarray) -> bool:
    return float(np.sqrt((dev_l ** 2).sum(axis=1).max())) <= _POINT_TOL


def _quality(dev: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """(L, N) k-NN overlap between each layer's full-space pellets and their
    screen positions. A layer that is one point preserves everything. k is at
    most half the family: with k = N - 1 every pellet's neighbours are all
    the others, and noise scores a perfect 1."""
    L, N, _ = dev.shape
    k = min(_NBHD_K, max(1, (N - 1) // 2))
    q = np.ones((L, N), dtype=np.float32)
    for l in range(L):
        if not _is_point(dev[l]):
            q[l] = neighborhood_preservation(dev[l], pos[:, l], k=k)
    return q


# ---------------------------------------------------------------- monitor
def _auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC with ties at half credit."""
    from scipy.stats import rankdata

    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _hash(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()


def _folds(ids: list[str], y: np.ndarray, k: int, groups: list[str] | None = None) -> np.ndarray:
    """Fold per labelled pellet, -1 elsewhere. Members of a class are dealt
    round-robin in the order of a hash of their id, so a pellet's fold
    follows the pellet, not its position in the file: reordering the items
    cannot move the picture.

    With `groups`, whole groups are dealt instead: a pellet whose twin in the
    other class is still in training gets scored on a direction its twin
    pulled its way. On TruthfulQA true/false answer pairs that put 9 of 25
    layers below the shuffle null's 5th percentile; holding pairs out
    together removes the pull."""
    fold = np.full(len(y), -1)
    if groups is not None:
        names = sorted({groups[i] for i in np.flatnonzero(y >= 0)}, key=_hash)
        rank = {g: r % k for r, g in enumerate(names)}
        for i in np.flatnonzero(y >= 0):
            fold[i] = rank[groups[i]]
        return fold
    for cls in (0, 1):
        order = sorted(np.flatnonzero(y == cls), key=lambda i: _hash(ids[i]))
        for rank, i in enumerate(order):
            fold[i] = rank % k
    return fold


def _crossfit_x(dev: np.ndarray, y: np.ndarray, fold: np.ndarray, k: int) -> np.ndarray:
    """(L, N) score of each labelled pellet on the unit diff-of-means
    direction fitted without its fold; 0 elsewhere."""
    L, N, _ = dev.shape
    x = np.zeros((L, N))
    for f in range(k):
        te = fold == f
        if not te.any():
            continue
        tr1 = (y == 1) & ~te & (fold >= 0)
        tr0 = (y == 0) & ~te & (fold >= 0)
        if not tr1.any() or not tr0.any():
            continue                              # no direction to score on
        d = dev[:, tr1].mean(axis=1) - dev[:, tr0].mean(axis=1)          # (L, D)
        dn = np.linalg.norm(d, axis=1, keepdims=True)
        dh = np.where(dn > 1e-30, d / np.maximum(dn, 1e-300), 0.0)
        x[:, te] = np.einsum("lnd,ld->ln", dev[:, te], dh)
    return x


def _shuffle(y: np.ndarray, ids: list[str], groups: list[str] | None,
             rng: np.random.Generator) -> np.ndarray:
    """One draw of the label-shuffle null, in an order fixed by the pellets'
    ids, so reordering the items cannot move the band.

    Without groups, labels are shuffled across the labelled pellets. With
    groups they are shuffled the way the design could have assigned them:
    whole label sets move between groups of one size, then labels move within
    a group. Shuffling freely across pellets makes patterns a contrast-pair
    design cannot (both answers to one question false), and on no-signal
    pairs it put 0% of layers outside a band meant to hold 5% on each side."""
    yp = y.copy()
    lab = np.flatnonzero(y >= 0)
    if groups is None:
        canon = sorted(lab, key=lambda i: _hash(ids[i]))
        yp[canon] = y[canon][rng.permutation(len(canon))]
        return yp
    members: dict[str, list[int]] = {}
    for i in sorted(lab, key=lambda i: _hash(ids[i])):
        members.setdefault(groups[i], []).append(i)
    by_size: dict[int, list[str]] = {}
    for g in sorted(members, key=_hash):
        by_size.setdefault(len(members[g]), []).append(g)
    for size in sorted(by_size):
        names = by_size[size]
        sets = [y[members[g]] for g in names]
        for g, src in zip(names, rng.permutation(len(names))):
            yp[members[g]] = sets[src][rng.permutation(size)]
    return yp


def _sign_direction(R: np.ndarray) -> np.ndarray:
    """Unit direction of the top spatial-sign principal component of rows R:
    rows are unit-normalised before the Gram eigenproblem, so the direction
    is the one most pellets share rather than the one a single far pellet
    owns (with raw rows one off-axis prompt-injection pellet took most of y)."""
    rn = np.linalg.norm(R, axis=1)
    U = R / np.maximum(rn, 1e-300)[:, None]
    w, V = np.linalg.eigh(U @ U.T)
    v = U.T @ V[:, -1]
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def monitor(traj: StateTrajectory, driver: str, folds: int = FOLDS,
            null_draws: int = NULL_DRAWS, seed: int = 0) -> BlastLayout:
    """The monitor's-eye blast for one label (see the module docstring).

    Raises if `driver` has fewer than MIN_PER_CLASS labelled pellets in
    either class: without labels there is no monitor to look through, and
    the open layout is the unsupervised view.
    """
    y = label_vector(traj, driver)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if min(n1, n0) < MIN_PER_CLASS:
        raise ValueError(f"monitor needs >= {MIN_PER_CLASS} labelled pellets per class "
                         f"for {driver!r}; got {n1} / {n0}. Use the open layout.")
    ids = list(traj.tokens)
    pel = traj.meta.get("pellets", [])
    # an ungrouped pellet is a group of one
    groups = ([str(p.get("group", f"\0{p.get('id')}")) for p in pel]
              if any("group" in p for p in pel) else None)
    fr = frame(traj.hidden)
    dev = fr["dev"]
    L, N, _ = dev.shape
    n_units = len({groups[i] for i in np.flatnonzero(y >= 0)}) if groups else min(n1, n0)
    k = int(min(folds, n_units))
    fold = _folds(ids, y, k, groups)
    lab = y >= 0
    # with groups, the unit held out is a group, so a fold can take a whole
    # class out of training; its pellets would sit at x = 0 under a caption
    # saying they were held out
    for f in range(k):
        tr = lab & (fold != f)
        if k < 2 or not (y[tr] == 1).any() or not (y[tr] == 0).any():
            raise ValueError(f"monitor for {driver!r}: with these groups fold {f} of {k} "
                             "trains on one class or none; split the groups, or use the "
                             "open layout")

    point = np.array([_is_point(dev[l]) for l in range(L)])
    x = _crossfit_x(dev, y, fold, k)
    d = dev[:, y == 1].mean(axis=1) - dev[:, y == 0].mean(axis=1)
    dn = np.linalg.norm(d, axis=1, keepdims=True)
    dh = np.where(dn > 1e-30, d / np.maximum(dn, 1e-300), 0.0)
    full = np.einsum("lnd,ld->ln", dev, dh)
    x[:, ~lab] = full[:, ~lab]        # unlabelled pellets: the full-data direction

    pos = np.zeros((N, L, 2))
    for l in range(L):
        if point[l]:
            continue                                   # the muzzle: one point
        R = dev[l] - np.outer(full[l], dh[l])
        v = _sign_direction(R)
        # pinned as projection.PCAProjection pins PCA: largest-|.| entry positive
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        pos[:, l, 0] = x[l]
        pos[:, l, 1] = R @ v
    # y has no sign of its own: each layer's is flipped to agree with the next
    # deeper one, so a trail is a pellet moving rather than an axis flipping
    for l in range(L - 2, -1, -1):
        if pos[:, l, 1] @ pos[:, l + 1, 1] < 0:
            pos[:, l, 1] *= -1

    auroc = np.array([np.nan if point[l] else _auroc(x[l, lab], y[lab]) for l in range(L)])
    rng = np.random.default_rng(seed)
    null = np.full((null_draws, L), np.nan)
    for b in range(null_draws):
        yp = _shuffle(y, ids, groups, rng)
        xp = _crossfit_x(dev, yp, _folds(ids, yp, k, groups), k)
        null[b] = [np.nan if point[l] else _auroc(xp[l, lab], yp[lab]) for l in range(L)]
    # both tails: with few pellets, or pellets that come in near-twin pairs
    # across classes, a held-out pellet can land on the other class's side,
    # and an AUROC far below chance is a finding about the prompts too
    with np.errstate(all="ignore"):
        null05, null95 = (np.where(point, np.nan,
                                   np.nanpercentile(np.where(point, 0, null), q, axis=0))
                          for q in (5, 95))

    return BlastLayout(
        name=f"monitor · {driver}", method="monitor", driver=driver,
        positions=pos.astype(np.float32), quality=_quality(dev, pos),
        exact=["origin: the family's centroid at each layer; unit: that layer's "
               "mean state norm"],
        fitted=[f"x: a full-space linear readout score, each pellet's dot product with "
                f"the diff-of-means direction for {driver!r} ({n1} vs {n0} labelled), "
                f"{k}-fold cross-fitted" + (" by group" if groups else "")
                + ": a labelled pellet is scored on a direction fitted without it"
                + (" or its group" if groups else "")
                + ", so a split on x is held out, not memorised"],
        projected=["y: the direction most pellets share in what x leaves out (top "
                   "spatial-sign principal direction of the residual); it carries no "
                   "label, and screen distance is not full-space distance"],
        arrays={"auroc": auroc.astype(np.float32), "null05": null05.astype(np.float32),
                "null95": null95.astype(np.float32), "labelled": lab.astype(np.int32)},
        params={"folds": k, "grouped": groups is not None, "null_draws": null_draws,
                "seed": seed, "min_per_class": MIN_PER_CLASS},
    )


# ------------------------------------------------------------------- open
def _inv_sqrt_2x2(M: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root of a 2x2 PSD matrix; a direction with
    no length gets none (pseudo-inverse) rather than an infinity."""
    w, Q = np.linalg.eigh(M)
    tol = max(float(w.max()), 0.0) * 1e-12
    s = np.where(w > tol, 1.0 / np.sqrt(np.maximum(w, 1e-300)), 0.0)
    return (Q * s) @ Q.T


def open_layout(traj: StateTrajectory, copy_tol: float = COPY_TOL) -> BlastLayout:
    """The unsupervised compromise blast (see the module docstring)."""
    fr = frame(traj.hidden)
    dev = fr["dev"]
    L, N, _ = dev.shape
    if N < 3:
        raise ValueError(f"the open layout needs at least 3 pellets; got {N}")
    rng = fr["range"]                                      # (N, L)
    live = rng > _POINT_TOL
    U = np.where(live.T[..., None], dev / np.maximum(rng.T, 1e-300)[..., None], 0.0)
    G = np.einsum("lid,ljd->ij", U, U)                     # sum over layers of U_l U_l^T

    # copies: within copy_tol x spread of each other at every depth
    far = np.zeros((N, N))
    for l in range(L):
        if fr["spread"][l] <= 0:
            continue
        Gl = dev[l] @ dev[l].T
        n = np.diag(Gl)
        far = np.maximum(far, np.sqrt(np.maximum(n[:, None] + n[None, :] - 2 * Gl, 0.0))
                         / fr["spread"][l])
    K = far < copy_tol
    h = 1.0 / np.sqrt(K.sum(axis=1))                       # mass = 1 / number of copies
    w, E = np.linalg.eigh(h[:, None] * G * h[None, :])
    V = h[:, None] * E[:, np.argsort(w)[::-1][:2]]         # (N, 2) pellet patterns
    sign = np.sign(V[np.argmax(np.abs(V), axis=0), [0, 1]])
    sign[sign == 0] = 1
    V = V * sign

    pos = np.zeros((N, L, 2))
    for l in range(L):
        if not live[:, l].any():
            continue                                       # the muzzle: one point
        B_all = U[l].T @ V                                 # (D, 2)
        for i in range(N):
            # its own bearing at this depth is left out of its camera: otherwise a
            # depth with no structure of its own inherits the pellet's pattern
            # value from the other depths and still looks split
            B = B_all - np.outer(U[l, i], V[i])
            pos[i, l] = dev[l, i] @ (B @ _inv_sqrt_2x2(B.T @ B))
    tot = (rng ** 2).sum(axis=0)
    shown = np.where(tot > 0, (pos ** 2).sum(axis=(0, 2)) / np.maximum(tot, 1e-300), 1.0)

    iu = np.triu_indices(N, 1)
    copies = [[int(i), int(j)] for i, j in zip(*iu) if K[i, j]]
    near = [[int(i), int(j), round(float(far[i, j]), 3)] for i, j in zip(*iu)
            if abs(far[i, j] / copy_tol - 1) < 0.1]
    n_groups = int(round(float((1.0 / K.sum(axis=1)).sum())))
    return BlastLayout(
        name="open", method="open",
        positions=pos.astype(np.float32), quality=_quality(dev, pos),
        exact=["origin: the family's centroid at each layer; unit: that layer's "
               "mean state norm",
               "a pellet's distance from the origin on screen is at most its "
               "full-space distance; the readout gives the share of each layer's spread "
               "the camera keeps"],
        projected=["positions: one camera per layer, built from the pellet patterns "
                   "the layers agree on, without labels. It is built from every depth, "
                   "so structure from deeper layers can show at shallower ones; each "
                   "pellet's own bearing is left out of its camera, which reduces "
                   "that, not removes it",
                   f"pellets within {copy_tol:g}x the family spread of each other at "
                   f"every depth count once when choosing the camera ({N} pellets, "
                   f"{n_groups} groups); adding a pellet redraws every layer"],
        arrays={"shown": shown.astype(np.float32), "mass": (h ** 2).astype(np.float32)},
        params={"copy_tol": copy_tol, "copies": copies, "near_copies": near},
    )


# ------------------------------------------------------------------ scene
def layouts(traj: StateTrajectory, drivers: list[str] | None = None,
            methods: tuple[str, ...] = ("monitor", "open"), seed: int = 0,
            folds: int = FOLDS, null_draws: int = NULL_DRAWS,
            copy_tol: float = COPY_TOL):
    """Every layout a family supports: a monitor per usable label, then open.
    Returns (layouts, skipped) where skipped explains each label that could
    not drive a monitor."""
    unknown = set(methods) - set(_LABELS)
    if unknown:
        raise ValueError(f"unknown layout {sorted(unknown)}; choose from {list(_LABELS)}")
    names = drivers if drivers is not None else sorted(
        {k for p in traj.meta.get("pellets", []) for k in p.get("labels", {})})
    out, skipped = [], []
    if "monitor" in methods:
        for name in names:
            try:
                out.append(monitor(traj, name, folds=folds, null_draws=null_draws,
                                   seed=seed))
            except ValueError as e:
                skipped.append(f"{name}: {e}")
    if "open" in methods:
        out.append(open_layout(traj, copy_tol=copy_tol))
    if not out:
        raise ValueError("no layout could be built: " + "; ".join(skipped))
    return out, skipped


def scene(traj: StateTrajectory, built: list[BlastLayout], skipped=()) -> dict:
    """A save_scene-ready result: a flat terrain, one run whose trajectories
    are the pellets (drawn at the first layout), and the `blast` record.

    The terrain and run keep the scene readable by any viewer that predates
    the blast record: it draws the pellets flying out of the muzzle on flat
    ground."""
    from terrain import TerrainMesh
    from trajectory import Trajectory

    fr = frame(traj.hidden)
    every = np.concatenate([b.positions.reshape(-1, 2) for b in built])
    lo, hi = every.min(axis=0), every.max(axis=0)
    pad = np.maximum((hi - lo) * 0.1, 1e-3)
    mesh = TerrainMesh(x=np.array([lo[0] - pad[0], hi[0] + pad[0]], np.float32),
                       y=np.array([lo[1] - pad[1], hi[1] + pad[1]], np.float32),
                       z=np.zeros((2, 2), np.float32))
    lift = float(max(hi - lo)) * 0.005
    first = built[0].positions
    trajectories = [Trajectory(token=i, label=traj.tokens[i],
                               points=np.column_stack([first[i], np.full(len(first[i]), lift)]))
                    for i in range(traj.n_tokens)]

    class _Quality:              # save_scene reads .preservation only
        preservation = built[0].quality

    return {
        "traj": traj, "mesh": mesh, "landscape": None,
        "trajectories": trajectories, "quality": _Quality(),
        "prompt": f"{traj.n_tokens} pellets",
        "blast": {"layouts": built, "skipped": list(skipped),
                  "range": fr["range"].astype(np.float32),
                  "spread": fr["spread"].astype(np.float32),
                  "norm": fr["norm"].astype(np.float32)},
    }
