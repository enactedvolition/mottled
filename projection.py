"""Projection of hidden vectors to low-dimensional coordinates.

Plugin registry: any object with fit_transform / transform can be registered
as a projection.  PCA is the deterministic default; UMAP is available when
umap-learn is installed.  `transform` enables incremental projection of new
states into an already-fitted space.

Every projection distorts: `projection_quality` measures how much, per state
(k-NN neighborhood preservation for any projection, reconstruction residual
and explained variance for linear ones), so viewers can show where the 2-D
picture is trustworthy and where it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PROJECTIONS: dict[str, type] = {}


def register_projection(name: str):
    def deco(cls):
        PROJECTIONS[name] = cls
        return cls

    return deco


def get_projection(name: str, n_components: int = 2, seed: int = 0):
    try:
        cls = PROJECTIONS[name]
    except KeyError:
        raise ValueError(f"unknown projection {name!r}; available: {sorted(PROJECTIONS)}") from None
    return cls(n_components=n_components, seed=seed)


@register_projection("pca")
class PCAProjection:
    """Deterministic linear projection (exact full SVD)."""

    def __init__(self, n_components: int = 2, seed: int = 0):
        from sklearn.decomposition import PCA

        self._pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)
        self.fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        out = self._pca.fit_transform(np.asarray(X, dtype=np.float64))
        # Pin the component signs instead of inheriting scikit-learn's, which
        # changed in 1.5 (from U-based to V-based `svd_flip`): each
        # component's largest-|.| entry is positive. A no-op on >= 1.5; it
        # keeps the reference — and viewer/scene.js, which mirrors it — from
        # flipping with the installed version.
        comps = self._pca.components_
        signs = np.sign(comps[np.arange(len(comps)), np.abs(comps).argmax(axis=1)])
        signs[signs == 0] = 1
        comps *= signs[:, None]
        out *= signs
        self.fitted = True
        return out.astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._pca.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """(N, n_components) plane coordinates -> (N, D) hidden vectors.

        Exact for PCA: the returned vectors lie on the fitted affine
        subspace, so `transform(inverse_transform(Y)) == Y`.  This is what
        lets a field be evaluated over the projection plane itself (see
        `sae.feature_field`).
        """
        return self._pca.inverse_transform(
            np.asarray(Y, dtype=np.float64)).astype(np.float32)

    @property
    def explained_variance(self) -> float:
        """Fraction of total variance the kept components explain."""
        return float(self._pca.explained_variance_ratio_.sum())

    def reconstruction_residual(self, X: np.ndarray) -> np.ndarray:
        """Per-point relative projection loss in [0, 1].

        ||x - reconstruct(project(x))|| / ||x - center||: the fraction of a
        point's (centered) length that falls outside the fitted plane — 0
        means the state lies exactly on the plane you are looking at.
        """
        X = np.asarray(X, dtype=np.float64)
        recon = self._pca.inverse_transform(self._pca.transform(X))
        lost = np.linalg.norm(X - recon, axis=1)
        total = np.linalg.norm(X - self._pca.mean_, axis=1)
        return (lost / np.maximum(total, 1e-12)).astype(np.float32)


@register_projection("umap")
class UMAPProjection:
    """Nonlinear manifold projection; deterministic for a fixed seed."""

    def __init__(self, n_components: int = 2, seed: int = 0):
        import umap  # optional dependency

        self._umap = umap.UMAP(n_components=n_components, random_state=seed, n_jobs=1)
        self.fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        # UMAP needs more samples than neighbors; shrink for tiny inputs.
        self._umap.n_neighbors = int(min(self._umap.n_neighbors, max(2, len(X) - 1)))
        out = self._umap.fit_transform(X)
        self.fitted = True
        return np.asarray(out, dtype=np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._umap.transform(np.asarray(X, dtype=np.float32)), dtype=np.float32)

    def inverse_transform(self, Y: np.ndarray) -> np.ndarray:
        """Approximate embedding-space -> data-space inverse (umap's own)."""
        return np.asarray(
            self._umap.inverse_transform(np.asarray(Y, dtype=np.float32)),
            dtype=np.float32)


def project(
    hidden: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    seed: int = 0,
):
    """Project (L, T, D) hidden states to (L, T, n_components) coordinates.

    All states across layers and tokens are embedded in one shared space so
    that distances between layers are meaningful.  Returns (coords, projector);
    the fitted projector supports incremental `transform` for new states.
    """
    hidden = np.asarray(hidden)
    L, T, D = hidden.shape
    proj = get_projection(method, n_components=n_components, seed=seed)
    coords = proj.fit_transform(hidden.reshape(L * T, D))
    return coords.reshape(L, T, n_components), proj


def project_joint(
    hiddens: list[np.ndarray],
    method: str = "pca",
    n_components: int = 2,
    seed: int = 0,
):
    """Project several (L_i, T_i, D) hidden arrays into ONE shared space.

    The projection is fitted on the union of every run's states, so
    coordinates — and distances — are comparable *across* runs.  This is what
    the prompt A/B overlay needs: two forward passes drawn on the same
    manifold.  Returns (coords_list, projector); each coords_list[i] has
    shape (L_i, T_i, n_components).
    """
    arrays = [np.asarray(h) for h in hiddens]
    if not arrays:
        raise ValueError("project_joint needs at least one hidden array")
    D = arrays[0].shape[-1]
    if any(a.ndim != 3 or a.shape[-1] != D for a in arrays):
        raise ValueError("all hidden arrays must be (L, T, D) with a shared D")

    proj = get_projection(method, n_components=n_components, seed=seed)
    flat = proj.fit_transform(np.concatenate([a.reshape(-1, D) for a in arrays]))

    coords, offset = [], 0
    for a in arrays:
        n = a.shape[0] * a.shape[1]
        coords.append(flat[offset : offset + n].reshape(a.shape[0], a.shape[1], n_components))
        offset += n
    return coords, proj


# ------------------------------------------------------------ distortion
@dataclass
class ProjectionQuality:
    """How faithfully the projection preserved each state.

    preservation: (L, T) fraction of each state's k hidden-space nearest
        neighbors that are still among its k nearest neighbors in the
        projected space — 1.0 means the local structure survived intact.
        Defined for every projection.
    residual: (L, T) relative reconstruction loss (see
        `PCAProjection.reconstruction_residual`), or None when the
        projection has no exact inverse.
    explained_variance: global fraction of variance kept, or None for
        nonlinear projections.
    k: neighborhood size the preservation was measured at.
    """

    preservation: np.ndarray
    residual: np.ndarray | None
    explained_variance: float | None
    k: int
    # Within-layer fidelity (added): the same k-NN overlap, but each layer's
    # neighbors are searched among that layer's states only, in both spaces.
    # The pooled `preservation` above mixes in cross-layer neighbors (the
    # same token at adjacent layers, and layer separation), so a projection
    # can score well on it while scrambling every layer internally; see
    # `per_layer_preservation`. None when a layer has < 2 states.
    per_layer: np.ndarray | None = None          # (L, T)
    per_layer_cosine: np.ndarray | None = None   # (L, T), cosine neighbors in hidden space
    k_layer: int | None = None
    chance_layer: float | None = None            # expected per_layer under a random placement


def neighborhood_preservation(X: np.ndarray, Y: np.ndarray, k: int = 10,
                              metric: str = "euclidean") -> np.ndarray:
    """Per-point k-NN overlap between a high-dim cloud and its projection.

    X: (N, D) original points, Y: (N, C) projected points.  Returns (N,)
    values in [0, 1]: the fraction of each point's k nearest neighbors in X
    that remain among its k nearest neighbors in Y.  `metric="cosine"`
    ranks the hidden-space neighbors by angle (Euclidean on unit vectors,
    which gives the same ranking); the projected space stays Euclidean.
    """
    from sklearn.neighbors import NearestNeighbors

    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"unknown metric {metric!r}; use 'euclidean' or 'cosine'")
    X = np.asarray(X, dtype=np.float64)
    if metric == "cosine":
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Y = np.asarray(Y, dtype=np.float64)
    n = len(X)
    if n != len(Y):
        raise ValueError("X and Y must contain the same number of points")
    k = int(min(k, n - 1))
    if k < 1:
        return np.ones(n, dtype=np.float32)
    # +1 for the point itself, dropped below
    nn_x = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)
    nn_y = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        a = set(nn_x[i].tolist()) - {i}
        b = set(nn_y[i].tolist()) - {i}
        out[i] = len(a & b) / max(len(a), 1)
    return out


# Below this neighborhood-preservation score a state is drawn where the
# projection *could* put it rather than where it is. One home for the
# threshold: it drives the explorer's prose, the ✕ markers on the scene, and
# the viewer's scene panel, which would otherwise disagree about the same run.
LOW_FIDELITY = 0.5


def fidelity_summary(preservation, low_fidelity: float = LOW_FIDELITY) -> dict:
    """How much of the neighborhood structure survived the projection.

    One computation, two surfaces: the Streamlit explorer's inline note and
    the web viewer's scene panel (`viewer/reading.js`, pinned by
    `tests/test_reading_conformance.py`). A scene that reports one fidelity in
    the app and another in the browser is two tools, and a screenshot from
    either says nothing about the other.
    """
    arr = np.asarray(preservation, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("no preservation values to summarize")
    return {
        "mean": float(arr.mean()),
        "low_fraction": float((arr < low_fidelity).mean()),
        "low_fidelity": float(low_fidelity),
        "n": int(arr.size),
    }


def per_layer_preservation(hidden: np.ndarray, coords: np.ndarray, k: int = 5,
                           metric: str = "euclidean") -> np.ndarray:
    """(L, T) within-layer k-NN preservation: for each layer, neighbors are
    searched among that layer's T states only, in hidden and projected space.

    This is the reading the pooled score is usually taken to give ("is the
    local structure *at this depth* intact?"). On GPT-2 small the pooled score
    cannot tell PCA from a random projection while this one can
    (fidelity_experiment.py). k is clipped to T-1 per layer.
    """
    hidden = np.asarray(hidden)
    coords = np.asarray(coords)
    L, T = hidden.shape[:2]
    out = np.empty((L, T), dtype=np.float32)
    for layer in range(L):
        out[layer] = neighborhood_preservation(hidden[layer], coords[layer], k=k,
                                               metric=metric)
    return out


def preservation_chance(n: int, k: int) -> float:
    """Expected k-NN overlap when the projected neighbors are a random k of
    the other n-1 points: k / (n - 1)."""
    k = int(min(k, n - 1))
    return 1.0 if k < 1 else k / (n - 1)


def projection_quality(
    hidden: np.ndarray,
    coords: np.ndarray,
    projector=None,
    k: int = 10,
    k_layer: int = 5,
) -> ProjectionQuality:
    """Measure the distortion of a fitted projection, per state.

    hidden: (L, T, D) states, coords: (L, T, C) their projections (from
    `project` / `project_joint`).  Preservation is always computed;
    residual / explained variance come from the projector when it exposes
    them (PCA does, UMAP does not).
    """
    hidden = np.asarray(hidden)
    coords = np.asarray(coords)
    L, T, D = hidden.shape
    flat_x = hidden.reshape(L * T, D)
    flat_y = coords.reshape(L * T, coords.shape[-1])

    preservation = neighborhood_preservation(flat_x, flat_y, k=k).reshape(L, T)

    residual = None
    if projector is not None and hasattr(projector, "reconstruction_residual"):
        residual = projector.reconstruction_residual(flat_x).reshape(L, T)
    explained = None
    if projector is not None and hasattr(projector, "explained_variance"):
        explained = float(projector.explained_variance)

    per_layer = per_layer_cos = chance = kl = None
    if T >= 2:
        kl = int(min(k_layer, T - 1))
        per_layer = per_layer_preservation(hidden, coords, k=kl)
        per_layer_cos = per_layer_preservation(hidden, coords, k=kl, metric="cosine")
        chance = preservation_chance(T, kl)

    return ProjectionQuality(preservation=preservation, residual=residual,
                             explained_variance=explained, k=int(min(k, L * T - 1)),
                             per_layer=per_layer, per_layer_cosine=per_layer_cos,
                             k_layer=kl, chance_layer=chance)
