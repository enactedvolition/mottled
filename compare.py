"""Phase 2 — trajectory comparison.

Compare two StateTrajectories — prompt A vs prompt B, or a baseline vs a
counterfactual branch: geometric distance between their paths (symmetric
Hausdorff, dynamic time warping), shared-prefix alignment, and layerwise
divergence profiles (`metrics.branch_divergence` is the pointwise seed).

Everything here is a *measurement* over trajectories — nothing touches
transformer internals, so any backend that produces a StateTrajectory
(transformers, closed-model logprobs, future substrates) is comparable, and every
distance works in any coordinate space: full hidden space or projected
coordinates.  The two runs must come from the same substrate (same layer
count and hidden dimension) for states to be commensurable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics import branch_divergence
from trajectory import StateTrajectory

_EPS = 1e-12


# --------------------------------------------------------------- geometry
def hausdorff(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two paths (as point sets).

    The largest distance from any point of one path to the nearest point of
    the other — how far apart the paths get, ignoring traversal order.
    """
    from scipy.spatial.distance import directed_hausdorff

    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or not len(a) or not len(b):
        raise ValueError("hausdorff expects two non-empty (N, D) point arrays")
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


@dataclass
class DTWResult:
    """Optimal monotone alignment of two paths."""

    distance: float      # accumulated Euclidean cost along the optimal path
    normalized: float    # distance / alignment length (comparable across sizes)
    path: np.ndarray     # (K, 2) aligned index pairs (index in A, index in B)


def dtw(points_a: np.ndarray, points_b: np.ndarray) -> DTWResult:
    """Dynamic time warping between two paths (Euclidean local cost).

    Unlike the pointwise `metrics.branch_divergence`, DTW aligns paths that
    move through the same regions at different speeds — two trajectories that
    trace the same route but settle at different layers still score low.
    """
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or not len(a) or not len(b):
        raise ValueError("dtw expects two non-empty (N, D) point arrays")

    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        acc[i, 1:] = cost[i - 1]
        for j in range(1, m + 1):
            acc[i, j] += min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])

    path = [(n - 1, m - 1)]
    i, j = n, m
    while (i, j) != (1, 1):
        steps = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
        i, j = min((ij for ij in steps if min(ij) >= 1), key=lambda ij: acc[ij])
        path.append((i - 1, j - 1))
    path.reverse()

    total = float(acc[n, m])
    return DTWResult(distance=total, normalized=total / len(path),
                     path=np.asarray(path, dtype=np.int64))


# ------------------------------------------------------------- alignment
def shared_prefix(tokens_a: list[str], tokens_b: list[str]) -> int:
    """Length of the common token prefix of two prompts."""
    n = 0
    for x, y in zip(tokens_a, tokens_b):
        if x != y:
            break
        n += 1
    return n


# ------------------------------------------------------------- comparison
@dataclass
class TrajectoryComparison:
    """How two forward passes relate — geometry, alignment, and divergence.

    All fields are measurements; any reading of them ("the prompts commit at
    layer k") is a separate, defeasible narrative.
    """

    shared_tokens: int            # common token prefix length of the prompts
    hausdorff: float              # symmetric Hausdorff between compared paths
    dtw: DTWResult                # time-warped alignment of compared paths
    profile: np.ndarray           # (L,) pointwise distance at the compared token
    positionwise: np.ndarray      # (L, min(Ta, Tb)) hidden-space distance per state
    relative: np.ndarray          # (L,) |h_b - h_a| / |h_a| at the compared token
    cosine: np.ndarray            # (L,) cosine of h_a and h_b at the compared token
    onset_token: int | None       # first aligned position whose states separate
    onset_layer: int              # first layer the compared token's distance exceeds the band
    readout_changed: int | None   # first layer the top-1 logit-lens prediction differs
    token: int                    # compared token index (of trajectory A)


def compare(
    traj_a: StateTrajectory,
    traj_b: StateTrajectory,
    coords_a: np.ndarray | None = None,
    coords_b: np.ndarray | None = None,
    token: int = -1,
    rel: float = 0.05,
) -> TrajectoryComparison:
    """Compare two forward passes at one token position (default: final).

    Geometry (Hausdorff / DTW / profile) is measured over the compared
    token's path — in the shared projected space when `coords_a`/`coords_b`
    (both (L, T, C), e.g. from `projection.project_joint`) are given,
    otherwise in full hidden space.  Alignment, `positionwise`, `relative`
    and `cosine` always use hidden space.  `onset_*` use the same relative band as
    `intervene.divergence`: the first index whose distance exceeds `rel`
    times the maximum separation.
    """
    if traj_a.n_layers != traj_b.n_layers or traj_a.dim != traj_b.dim:
        raise ValueError("trajectories must share layer count and hidden dimension "
                         "(same model / backend) to be comparable")
    if (coords_a is None) != (coords_b is None):
        raise ValueError("pass both coords_a and coords_b (a shared projection) or neither")

    t_a = int(token) % traj_a.n_tokens
    t_b = int(token) % traj_b.n_tokens
    path_a = (coords_a if coords_a is not None else traj_a.hidden)[:, t_a, :]
    path_b = (coords_b if coords_b is not None else traj_b.hidden)[:, t_b, :]

    profile = branch_divergence(path_a, path_b)

    n_shared = min(traj_a.n_tokens, traj_b.n_tokens)
    positionwise = np.linalg.norm(
        traj_a.hidden[:, :n_shared].astype(np.float64)
        - traj_b.hidden[:, :n_shared].astype(np.float64),
        axis=-1,
    )

    def _onset(dist: np.ndarray) -> int | None:
        peak = float(dist.max())
        if peak <= 0:
            return None
        exceed = np.flatnonzero(dist > rel * peak)
        return int(exceed[0]) if exceed.size else None

    # The compared token's divergence in hidden space, in units no projection
    # stands between: the norm grows with depth, so an absolute distance
    # mostly says which layer it was taken at; relative to A's state and as
    # a cosine it says how far B has moved from A at that layer.
    h_a = traj_a.hidden[:, t_a].astype(np.float64)
    h_b = traj_b.hidden[:, t_b].astype(np.float64)
    n_a, n_b = np.linalg.norm(h_a, axis=-1), np.linalg.norm(h_b, axis=-1)
    relative = np.linalg.norm(h_b - h_a, axis=-1) / np.maximum(n_a, _EPS)
    cosine = (h_a * h_b).sum(axis=-1) / np.maximum(n_a * n_b, _EPS)

    onset_token = _onset(positionwise.max(axis=0))
    onset_layer = _onset(profile)
    onset_layer = len(profile) - 1 if onset_layer is None else onset_layer

    readout_changed = None
    if (traj_a.logits is not None and traj_b.logits is not None
            and traj_a.logits.shape[-1] == traj_b.logits.shape[-1]):
        top_a = traj_a.logits[:, t_a].astype(np.float32).argmax(axis=-1)
        top_b = traj_b.logits[:, t_b].astype(np.float32).argmax(axis=-1)
        diff = np.flatnonzero(top_a != top_b)
        readout_changed = int(diff[0]) if diff.size else None

    return TrajectoryComparison(
        shared_tokens=shared_prefix(traj_a.tokens, traj_b.tokens),
        hausdorff=hausdorff(path_a, path_b),
        dtw=dtw(path_a, path_b),
        profile=profile.astype(np.float32),
        positionwise=positionwise.astype(np.float32),
        relative=relative.astype(np.float32),
        cosine=cosine.astype(np.float32),
        onset_token=onset_token,
        onset_layer=onset_layer,
        readout_changed=readout_changed,
        token=t_a,
    )
