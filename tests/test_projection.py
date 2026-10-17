"""Projection deterministic (spec test 3) + incremental support."""

import numpy as np
import pytest

from projection import (
    PROJECTIONS,
    get_projection,
    neighborhood_preservation,
    project,
    projection_quality,
)


@pytest.fixture(scope="module")
def hidden():
    rng = np.random.default_rng(7)
    return rng.normal(size=(13, 6, 64)).astype(np.float32)


def test_pca_shape_and_determinism(hidden):
    coords1, _ = project(hidden, method="pca")
    coords2, _ = project(hidden, method="pca")
    assert coords1.shape == (13, 6, 2)
    assert np.array_equal(coords1, coords2)
    assert np.isfinite(coords1).all()


def test_pca_incremental_transform(hidden):
    coords, proj = project(hidden, method="pca")
    new_states = hidden[0] + 0.01  # "new" states projected into fitted space
    out = proj.transform(new_states)
    assert out.shape == (6, 2)
    # transform of the training data reproduces the fitted coordinates
    flat = hidden.reshape(-1, 64)
    assert np.allclose(proj.transform(flat), coords.reshape(-1, 2), atol=1e-4)


@pytest.mark.skipif("umap" not in PROJECTIONS, reason="umap not registered")
def test_umap_shape(hidden):
    pytest.importorskip("umap")
    coords, _ = project(hidden[:4, :3], method="umap")  # small input: fast
    assert coords.shape == (4, 3, 2)
    assert np.isfinite(coords).all()


def test_unknown_projection_rejected(hidden):
    with pytest.raises(ValueError, match="unknown projection"):
        project(hidden, method="tsne-magic")


def test_neighborhood_preservation_perfect_when_structure_kept():
    # A projection that keeps the first two (already dominant) coordinates
    # preserves neighborhoods exactly.
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 2))
    X = np.column_stack([X, 0.001 * rng.normal(size=(40, 3))])  # near-2D cloud
    Y = X[:, :2]
    pres = neighborhood_preservation(X, Y, k=5)
    assert pres.shape == (40,)
    assert (pres >= 0).all() and (pres <= 1).all()
    assert pres.mean() > 0.9


def test_projection_quality_reports_pca_residual_and_variance(hidden):
    coords, proj = project(hidden, method="pca")
    q = projection_quality(hidden, coords, proj, k=8)
    assert q.preservation.shape == hidden.shape[:2]
    assert (q.preservation >= 0).all() and (q.preservation <= 1).all()
    # PCA exposes both residual and explained variance
    assert q.residual is not None and q.residual.shape == hidden.shape[:2]
    assert (q.residual >= 0).all()
    assert q.explained_variance is not None and 0.0 <= q.explained_variance <= 1.0


def test_projection_quality_residual_matches_variance_direction():
    # A cloud that lives almost entirely in a 2-plane: low residual, high var.
    rng = np.random.default_rng(9)
    flat = np.column_stack([rng.normal(size=(30, 2)) * 10,
                            rng.normal(size=(30, 6)) * 0.01])
    hidden = flat.reshape(5, 6, 8)
    coords, proj = project(hidden, method="pca")
    q = projection_quality(hidden, coords, proj)
    assert q.explained_variance > 0.98
    assert q.residual.mean() < 0.1


def test_registry_plugin():
    class Identity2D:
        def __init__(self, n_components=2, seed=0):
            pass

        def fit_transform(self, X):
            return np.asarray(X)[:, :2]

    PROJECTIONS["identity"] = Identity2D
    try:
        p = get_projection("identity")
        X = np.arange(12.0).reshape(3, 4)
        assert np.array_equal(p.fit_transform(X), X[:, :2])
    finally:
        del PROJECTIONS["identity"]


# ---- within-layer preservation (per_layer_preservation) ----------------

def _layered(seed=0, L=6, T=12, D=32):
    """States whose layers sit far apart (big per-layer offset) but which
    carry real within-layer structure (a low-rank token signal)."""
    rng = np.random.default_rng(seed)
    tok = rng.normal(size=(T, 2)) @ rng.normal(size=(2, D))
    offs = rng.normal(size=(L, 1, D)) * 50.0
    return offs + tok[None] + 0.01 * rng.normal(size=(L, T, D))


def test_per_layer_preservation_is_perfect_for_an_exact_projection():
    from projection import per_layer_preservation
    H = _layered()
    Y = (H - H.mean(axis=1, keepdims=True)) @ np.linalg.svd(
        (H - H.mean(axis=1, keepdims=True)).reshape(-1, H.shape[-1]),
        full_matrices=False)[2][:2].T
    assert per_layer_preservation(H, Y, k=4).mean() > 0.95


def test_pooled_score_survives_within_layer_scramble_but_per_layer_does_not():
    """The failure the per-layer score exists to catch: keep each layer's 2-D
    point set, reassign it to random tokens. Layer separation is untouched,
    within-layer fidelity is destroyed."""
    from projection import (neighborhood_preservation, per_layer_preservation,
                            preservation_chance, project)
    H = _layered()
    L, T, D = H.shape
    Y, _ = project(H, method="pca")
    rng = np.random.default_rng(1)
    Z = np.stack([Y[l][rng.permutation(T)] for l in range(L)])
    pooled = lambda C: neighborhood_preservation(H.reshape(-1, D), C.reshape(-1, 2), k=10).mean()
    assert pooled(Z) > 0.6 * pooled(Y)          # pooled barely notices
    good = per_layer_preservation(H, Y, k=4).mean()
    bad = per_layer_preservation(H, Z, k=4).mean()
    assert bad < preservation_chance(T, 4) + 0.15 < good


def test_cosine_metric_ignores_scale():
    from projection import neighborhood_preservation
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 16))
    Y = X[:, :2]
    scaled = X * rng.uniform(0.1, 10, size=(40, 1))
    a = neighborhood_preservation(X, Y, k=5, metric="cosine")
    b = neighborhood_preservation(scaled, Y, k=5, metric="cosine")
    np.testing.assert_allclose(a, b)
    with pytest.raises(ValueError):
        neighborhood_preservation(X, Y, metric="manhattan")


def test_projection_quality_carries_per_layer_fields(hidden):
    from projection import project, projection_quality
    coords, proj = project(hidden, method="pca")
    q = projection_quality(hidden, coords, proj)
    L, T = hidden.shape[:2]
    assert q.per_layer.shape == (L, T) and q.per_layer_cosine.shape == (L, T)
    assert q.k_layer == min(5, T - 1)
    assert 0 < q.chance_layer <= 1


# ---- the low-fidelity flag (fidelity_basis + low_fidelity_cut) ----------

def test_the_cut_is_halfway_from_chance_to_every_neighbor_kept():
    from projection import LOW_FIDELITY, low_fidelity_cut
    assert low_fidelity_cut(0.0) == LOW_FIDELITY == 0.5     # the plain cut, as before
    assert low_fidelity_cut(0.5) == 0.75
    assert low_fidelity_cut(1.0) == 1.0


def test_flag_reads_within_layer_and_falls_back_only_where_it_cannot_discriminate():
    from projection import ProjectionQuality, fidelity_basis
    rng = np.random.default_rng(0)
    long = rng.normal(size=(5, 12, 16))
    coords, proj = project(long)
    q = projection_quality(long, coords, proj)
    b = fidelity_basis(q)
    assert b.metric == "within-layer" and b.k == 5 and b.chance == pytest.approx(5 / 11)
    np.testing.assert_array_equal(b.values, np.minimum(q.per_layer, q.per_layer_cosine))

    # Six tokens: k clips to T - 1, every other state is a neighbor, and the
    # within-layer score is 1.0 whatever the projection did. It cannot be
    # what the flag reads.
    short = long[:, :6]
    scrambled = rng.normal(size=(5, 6, 2))
    q = projection_quality(short, scrambled)
    assert (q.per_layer == 1).all() and q.chance_layer == 1
    b = fidelity_basis(q)
    assert b.metric == "pooled" and b.chance == 0.0
    np.testing.assert_array_equal(b.values, q.preservation)

    # a quality without within-layer fields (an older caller) reads pooled
    bare = ProjectionQuality(preservation=np.ones((5, 6)), residual=None,
                             explained_variance=None, k=10)
    assert fidelity_basis(bare).metric == "pooled"


def _flag_rate(hidden, coords, projector=None, chance_relative=True):
    from projection import fidelity_basis, fidelity_summary
    b = fidelity_basis(projection_quality(hidden, coords, projector))
    return fidelity_summary(b.values, chance=b.chance if chance_relative else 0.0)["low_fraction"]


def _three_ways(prompt):
    """Flag rates for the tiny model's states under PCA, random Gaussian maps
    to 2-D, and PCA's own points shuffled among the tokens of each layer
    (layer separation intact, within-layer structure gone), over three sets
    of weights."""
    tiny = pytest.importorskip("tiny")
    rates = {"pca": [], "random": [], "shuffled": [], "shuffled_plain": []}
    for seed in (0, 1, 2):
        H = np.asarray(tiny.capture(prompt, seed=seed).hidden, np.float64)
        L, T, D = H.shape
        Y, proj = project(H)
        rates["pca"].append(_flag_rate(H, Y, proj))
        for s in range(5):
            R = np.random.default_rng(s).normal(size=(D, 2))
            rates["random"].append(_flag_rate(H, (H.reshape(-1, D) @ R).reshape(L, T, 2)))
        rng = np.random.default_rng(seed)
        Z = np.stack([Y[l][rng.permutation(T)] for l in range(L)])
        rates["shuffled"].append(_flag_rate(H, Z))
        rates["shuffled_plain"].append(_flag_rate(H, Z, chance_relative=False))
    return {k: np.asarray(v) for k, v in rates.items()}


def test_flag_marks_a_random_projection_far_more_than_pca():
    """The trust signal has to tell a projection that kept structure from one
    that did not. On a real forward pass (the tiny model, 13 tokens, so
    k = 5 and chance = 5/12 = 0.42, cut 0.71), measured: mean within-layer
    preservation PCA 0.64, random 0.48, shuffled 0.37; flagged PCA 57%
    (53-62%), random 88%, shuffled 96% (93-98%)."""
    r = _three_ways("the capital of france is paris and the capital of germany is berlin")
    assert r["random"].mean() > r["pca"].mean() + 0.2
    assert r["shuffled"].min() > 0.85


def test_a_plain_cut_cannot_flag_a_short_prompt_and_the_chance_relative_one_can():
    """Seven tokens: a random placement keeps 5/6 of its within-layer
    neighbors on average, so a plain 0.5 cut flagged nothing at all, not even
    PCA's points shuffled within every layer. Measured with the cut halfway
    from chance: PCA 47%, random 77%, shuffled 89% (78-97%)."""
    r = _three_ways("the capital of france is paris and")
    assert r["shuffled_plain"].max() == 0.0
    assert r["shuffled"].min() > 0.7
    assert r["random"].mean() > r["pca"].mean() + 0.2
