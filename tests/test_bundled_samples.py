"""The analysis pipeline, run on the real captures bundled in viewer/samples/.

The tiny randomly initialised Llamas in tests/tiny.py exercise the capture
code paths. These tests do the other half: they load the shipped scenes
(real GPT-2, DistilGPT-2, Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct
captures) and run the downstream steps on them, offline, with no model.

Scenes store projected coordinates, entropies and per-state preservation,
not raw hidden states, so the steps here start from the projected space:
density and terrain are recomputed from the stored points and must match
the stored terrain; geometry, comparison and fidelity are pinned to values
computed from these files on 2026-10-11. If a sample is regenerated, update
the pinned values in the same commit and say why.

Numerical pins. The values were computed with numpy 2.5.3, scipy 1.18.1 and
scikit-learn 1.9.1; requirements.lock pins numpy 2.4.6, scipy 1.17.1 and
scikit-learn 1.9.0, and CI resolves requirements.txt's floors to whatever is
current. The stored terrain was written at capture time, by whatever versions
were installed then (the oldest in July 2026), and it still recomputes to
within 1e-16, which is the evidence that KDE here is stable across those
versions. The tolerances (1e-3) are there for float noise, not version
drift. If an upgrade moves a pin past them, find out what changed before widening anything: a moved density or
geometry number is the regression these tests exist to catch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import compare as C
import metrics
from density import compute_density
from projection import fidelity_summary, neighborhood_preservation, project
from statefile import load_scene
from terrain import mesh

SAMPLES = Path(__file__).resolve().parent.parent / "viewer" / "samples"

# name: (model, path_length of run 0's final token, entropy_collapse of run 0's
#        final token, mean preservation, low fraction)
PINNED = {
    "gpt2-capitals.mtj": ("gpt2", 423.9340, -4.7537, 0.6438, 0.2692),
    "gpt2-decode.mtj": ("gpt2", 481.4730, -2.7910, 0.3149, 0.7149),
    "gpt2-features.mtj": ("gpt2", 452.5531, -5.3769, 0.5462, 0.4231),
    "models-gpt2-distilgpt2.mtj": ("gpt2", 0.4166, -4.7527, 0.6490, 0.2400),
    "models-qwen-gpt2.mtj": ("Qwen/Qwen2.5-1.5B-Instruct", 3.3027, -3.5324, 0.4710, 0.5095),
    "qwen-capitals.mtj": ("Qwen/Qwen2.5-1.5B-Instruct", 246.6330, -3.7676, 0.4466, 0.6448),
    "scene-abc.mtj": ("gpt2", 423.3516, -4.7537, 0.6415, 0.2769),
    "self-portrait.mtj": ("gpt2", 485.2397, -3.4456, 0.3659, 0.7055),
    "single.mtj": ("gpt2", 424.6904, -4.7537, 0.6431, 0.2769),
}

# name: (Hausdorff, normalised DTW) between run 0 and run 1 final-token paths
PAIRS = {
    "gpt2-capitals.mtj": (4.1187, 1.0588),
    "gpt2-features.mtj": (4.6923, 1.1650),
    "models-gpt2-distilgpt2.mtj": (0.1261, 0.0313),
    "models-qwen-gpt2.mtj": (0.7870, 0.2610),
    "qwen-capitals.mtj": (10.6164, 1.1155),
    "scene-abc.mtj": (4.0720, 1.0476),
    "self-portrait.mtj": (41.0881, 12.0005),
}

REL = 1e-3

# The music-qa-honesty scenes (Qwen2.5-0.5B-Instruct) get the structural and
# terrain checks; their findings are pinned in that folder's own README.
MUSIC = sorted(f"music-qa-honesty/{p.name}" for p in (SAMPLES / "music-qa-honesty").glob("*.mtj"))
ALL = sorted(PINNED) + MUSIC


def _scene(name):
    return load_scene(SAMPLES / name)


def _points(scene):
    return [np.asarray(r["points"], dtype=np.float64) for r in scene["runs"]]


def test_every_bundled_scene_is_covered():
    shipped = {p.name for p in SAMPLES.glob("*.mtj")}
    assert shipped == set(PINNED), "a sample was added or removed: pin its values here"
    assert len(MUSIC) == 5


@pytest.mark.parametrize("name", ALL)
def test_scene_shapes_are_consistent(name):
    s = _scene(name)
    want = PINNED[name][0] if name in PINNED else "Qwen/Qwen2.5-0.5B-Instruct"
    assert s["meta"]["model"] == want
    for run in s["runs"]:
        pts, ent, q = (np.asarray(run[k]) for k in ("points", "entropy", "quality"))
        T, L, three = pts.shape
        assert three == 3 and ent.shape == (L, T) and q.shape == (L, T)
        assert np.isfinite(pts).all() and np.isfinite(ent).all()
        assert (ent >= 0).all()
        assert ((q >= 0) & (q <= 1)).all()


@pytest.mark.parametrize("name", ALL)
def test_density_and_terrain_recompute_from_stored_points(name):
    s = _scene(name)
    stored = s["terrain"]
    xy = np.concatenate([p.reshape(-1, 3)[:, :2] for p in _points(s)])
    land = compute_density(xy, grid_size=stored["density"].shape[0])
    np.testing.assert_allclose(land.grid_x, stored["x"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(land.grid_y, stored["y"], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(land.density, stored["density"], atol=1e-3)
    assert 0.0 <= land.density.min() and land.density.max() == pytest.approx(1.0)
    t = mesh(land)
    assert t.z.shape == stored["density"].shape
    assert np.isfinite(t.z).all() and t.z.min() >= 0 and t.z.max() <= 1 + 1e-6
    # mesh() defaults (smooth_sigma=1.5, height_scale=1.0, invert=False) are
    # what the scenes were written with, so the heights must match too.
    np.testing.assert_allclose(t.z, stored["z"], atol=1e-3)


@pytest.mark.parametrize("name", sorted(PINNED))
def test_geometry_metrics_are_pinned(name):
    s = _scene(name)
    _, pl, ec, _, _ = PINNED[name]
    final = _points(s)[0][-1]
    assert metrics.path_length(final) == pytest.approx(pl, rel=REL)
    ent = np.asarray(s["runs"][0]["entropy"], dtype=np.float64)[:, -1]
    assert metrics.entropy_collapse(ent) == pytest.approx(ec, abs=1e-3)
    assert metrics.path_length(final) >= np.linalg.norm(final[-1] - final[0]) - 1e-9


@pytest.mark.parametrize("name", sorted(PINNED))
def test_fidelity_summary_is_pinned(name):
    s = _scene(name)
    _, _, _, mean, low = PINNED[name]
    q = np.concatenate([np.asarray(r["quality"]).ravel() for r in s["runs"]])
    fs = fidelity_summary(q)
    assert fs["mean"] == pytest.approx(mean, abs=1e-3)
    assert fs["low_fraction"] == pytest.approx(low, abs=1e-3)


def test_pairs_lists_exactly_the_multi_run_scenes():
    multi = {n for n in PINNED if len(_scene(n)["runs"]) >= 2}
    assert set(PAIRS) == multi


@pytest.mark.parametrize("name", sorted(PAIRS))
def test_run_comparison_is_pinned(name):
    a, b = (p[-1] for p in _points(_scene(name))[:2])
    haus, dtw_n = PAIRS[name]
    assert C.hausdorff(a, b) == pytest.approx(haus, rel=REL, abs=1e-4)
    assert C.hausdorff(a, b) == pytest.approx(C.hausdorff(b, a))
    assert C.dtw(a, b).normalized == pytest.approx(dtw_n, rel=REL, abs=1e-4)
    assert C.hausdorff(a, a) == 0.0 and C.dtw(a, a).distance == 0.0


@pytest.mark.parametrize("name", sorted(PINNED))
def test_pca_of_stored_plane_keeps_every_neighbourhood(name):
    # A 2-D PCA of 2-D points is a rigid motion, so k-NN sets are unchanged.
    # Only where the k-th neighbour is unambiguous: music-qa-honesty item-1, -2
    # and -6 have duplicate points and distance ties at k=5, which
    # NearestNeighbors breaks differently after the rotation (item-1 and -6
    # drop to 0.8), so they are left out on purpose.
    pts = _points(_scene(name))[0]
    xy = np.transpose(pts[:, :, :2], (1, 0, 2))  # (L, T, 2)
    coords, _ = project(xy, method="pca", n_components=2)
    pres = neighborhood_preservation(xy.reshape(-1, 2), coords.reshape(-1, 2), k=5)
    assert pres.min() == pytest.approx(1.0)


# Blast scenes: Qwen2.5-0.5B-Instruct, 48 pellets x 25 layers each.
# (layout, AUROC at layer 0 or None where undefined, AUROC at layer 1,
#  final-layer AUROC, peak AUROC, peak layer)
BLAST = {
    "truth.mtj": {
        "monitor · false_claim": (0.537, 0.637, 0.731, 0.821, 9),
        "monitor · negation": (0.389, 0.500, 0.929, 0.929, 21),
    },
    "refusal.mtj": {
        "monitor · harmful": (None, 0.771, 1.000, 1.000, 15),
        "monitor · refused": (None, 0.774, 0.995, 0.995, 17),
    },
}


def _layouts(name):
    b = load_scene(SAMPLES / "blast" / name)["blast"]
    return {(lay.get("name") or lay.get("label")): lay["arrays"] for lay in b["layouts"]}


@pytest.mark.parametrize("name", sorted(BLAST))
def test_blast_monitor_readouts_are_pinned(name):
    lays = _layouts(name)
    for label, (l0, l1, last, peak, at) in BLAST[name].items():
        a = np.asarray(lays[label]["auroc"], dtype=np.float64)
        assert a.shape == (25,)
        assert len(lays[label]["labelled"]) == 48
        if l0 is not None:  # the README quotes false_claim at layer 0 (0.54)
            assert a[0] == pytest.approx(l0, abs=2e-3)
        assert a[1] == pytest.approx(l1, abs=2e-3)
        assert a[-1] == pytest.approx(last, abs=2e-3)
        assert np.nanmax(a) == pytest.approx(peak, abs=2e-3)
        assert int(np.nanargmax(a)) == at
        valid = ~np.isnan(a)
        assert ((a[valid] >= 0) & (a[valid] <= 1)).all()
        lo, hi = (np.asarray(lays[label][k], dtype=np.float64) for k in ("null05", "null95"))
        assert (lo[valid] <= hi[valid]).all()


def test_refusal_layer_zero_is_one_point():
    # Every refusal pellet is read at the same template token, so layer 0 has
    # nothing to separate and its AUROC is undefined, as the README says.
    for label in BLAST["refusal.mtj"]:
        a = np.asarray(_layouts("refusal.mtj")[label]["auroc"])
        assert np.isnan(a[0]) and not np.isnan(a[1:]).any()
