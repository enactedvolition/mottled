"""The analysis pipeline, run on the real captures bundled in viewer/samples/.

The tiny randomly initialised Llamas in tests/tiny.py exercise the capture
code paths. These tests do the other half: they load the shipped scenes
(real GPT-2, DistilGPT-2, Qwen2.5-1.5B-Instruct and Qwen2.5-0.5B-Instruct
captures) and run the downstream steps on them, offline, with no model.

Scenes store projected coordinates, entropies and per-state preservation,
not raw hidden states, so the steps here start from the projected space:
density and terrain are recomputed from the stored points and must match
the stored terrain; geometry, comparison and fidelity are checked as
invariants on every scene. Only two things are pinned: one canary scene
(self-portrait, whose fidelity the docs quote) and the blast values its
README states in prose. If either moves, update the docs in the same commit.

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

MODELS = {
    "gpt2-capitals.mtj": "gpt2",
    "gpt2-decode.mtj": "gpt2",
    "gpt2-features.mtj": "gpt2",
    "models-gpt2-distilgpt2.mtj": "gpt2",
    "models-qwen-gpt2.mtj": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen-capitals.mtj": "Qwen/Qwen2.5-1.5B-Instruct",
    "scene-abc.mtj": "gpt2",
    "self-portrait.mtj": "gpt2",
    "single.mtj": "gpt2",
}
MUSIC = sorted(f"music-qa-honesty/{p.name}" for p in (SAMPLES / "music-qa-honesty").glob("*.mtj"))
ALL = sorted(MODELS) + MUSIC
# Scenes with two or more runs: the ones a run comparison applies to.
PAIRS = sorted(n for n in MODELS if len(load_scene(SAMPLES / n)["runs"]) > 1)


def _scene(name):
    return load_scene(SAMPLES / name)


def _points(scene):
    return [np.asarray(r["points"], dtype=np.float64) for r in scene["runs"]]


def _each(names, check):
    """Run `check` on every scene and report all failures at once, by name."""
    failed = []
    for name in names:
        try:
            check(name, _scene(name))
        except AssertionError as e:
            failed.append(f"{name}: {e}")
    assert not failed, "\n".join(failed)


def test_every_bundled_scene_is_covered():
    assert {p.name for p in SAMPLES.glob("*.mtj")} == set(MODELS), \
        "a sample was added or removed: list its model here"
    assert len(MUSIC) == 5


def test_scene_structure_and_ranges():
    def check(name, s):
        assert s["meta"]["model"] == MODELS.get(name, "Qwen/Qwen2.5-0.5B-Instruct")
        for run in s["runs"]:
            pts, ent, q = (np.asarray(run[k]) for k in ("points", "entropy", "quality"))
            T, L, three = pts.shape
            assert three == 3 and ent.shape == (L, T) and q.shape == (L, T)
            assert np.isfinite(pts).all() and np.isfinite(ent).all() and (ent >= 0).all()
            assert ((q >= 0) & (q <= 1)).all()
    _each(ALL, check)


def test_density_and_terrain_recompute_from_stored_points():
    # Not a snapshot: the stored terrain is what the viewer draws, so the
    # pipeline must reproduce it from the stored points, heights included.
    def check(name, s):
        stored = s["terrain"]
        xy = np.concatenate([p.reshape(-1, 3)[:, :2] for p in _points(s)])
        land = compute_density(xy, grid_size=stored["density"].shape[0])
        np.testing.assert_allclose(land.grid_x, stored["x"], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(land.grid_y, stored["y"], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(land.density, stored["density"], atol=1e-3)
        assert land.density.min() >= 0 and land.density.max() == pytest.approx(1.0)
        # mesh() defaults (smooth_sigma=1.5, height_scale=1.0, invert=False)
        # are what the scenes were written with, so the heights must match.
        t = mesh(land)
        assert np.isfinite(t.z).all() and t.z.min() >= 0 and t.z.max() <= 1 + 1e-6
        np.testing.assert_allclose(t.z, stored["z"], atol=1e-3)
    _each(ALL, check)


def test_geometry_and_fidelity_invariants():
    def check(name, s):
        for run, pts in zip(s["runs"], _points(s)):
            final = pts[-1]
            assert metrics.path_length(final) >= np.linalg.norm(final[-1] - final[0]) - 1e-9
            assert np.isfinite(metrics.entropy_collapse(np.asarray(run["entropy"])[:, -1]))
        q = np.concatenate([np.asarray(r["quality"]).ravel() for r in s["runs"]])
        fs = fidelity_summary(q)
        assert fs["n"] == q.size and 0 <= fs["mean"] <= 1 and 0 <= fs["low_fraction"] <= 1
        assert fs["low_fraction"] == pytest.approx((q < fs["low_fidelity"]).mean())
    _each(ALL, check)


def test_pairs_lists_exactly_the_multi_run_scenes():
    assert PAIRS == ["gpt2-capitals.mtj", "gpt2-features.mtj", "models-gpt2-distilgpt2.mtj",
                     "models-qwen-gpt2.mtj", "qwen-capitals.mtj", "scene-abc.mtj",
                     "self-portrait.mtj"]


def test_run_comparison_invariants():
    def check(name, s):
        a, b = (p[-1] for p in _points(s)[:2])
        h = C.hausdorff(a, b)
        assert h > 0 and h == pytest.approx(C.hausdorff(b, a))
        assert C.dtw(a, b).normalized == pytest.approx(C.dtw(b, a).normalized)
        assert C.hausdorff(a, a) == 0.0 and C.dtw(a, a).distance == 0.0
    _each(PAIRS, check)


def test_self_portrait_canary():
    # The one pinned scene. Its fidelity is the figure the README and the
    # project site quote (mean 0.37, 71% of states under the 0.5 flag); if it
    # moves, those need updating in the same commit.
    s = _scene("self-portrait.mtj")
    q = np.concatenate([np.asarray(r["quality"]).ravel() for r in s["runs"]])
    fs = fidelity_summary(q)
    assert fs["mean"] == pytest.approx(0.3659, abs=1e-3)
    assert fs["low_fraction"] == pytest.approx(0.7055, abs=1e-3)
    a, b = (p[-1] for p in _points(s)[:2])
    assert metrics.path_length(a) == pytest.approx(485.2397, rel=1e-3)
    assert C.hausdorff(a, b) == pytest.approx(41.0881, rel=1e-3)
    assert C.dtw(a, b).normalized == pytest.approx(12.0005, rel=1e-3)


def test_pca_of_stored_plane_keeps_every_neighbourhood():
    # A 2-D PCA of 2-D points is a rigid motion, so k-NN sets are unchanged.
    # Only where the k-th neighbour is unambiguous: music-qa-honesty item-1, -2
    # and -6 have duplicate points and distance ties at k=5, which
    # NearestNeighbors breaks differently after the rotation, so the music
    # scenes are left out on purpose.
    def check(name, s):
        xy = np.transpose(_points(s)[0][:, :, :2], (1, 0, 2))  # (L, T, 2)
        coords, _ = project(xy, method="pca", n_components=2)
        pres = neighborhood_preservation(xy.reshape(-1, 2), coords.reshape(-1, 2), k=5)
        assert pres.min() == pytest.approx(1.0)
    _each(sorted(MODELS), check)


# Blast scenes: Qwen2.5-0.5B-Instruct, 48 pellets x 25 layers each. Only the
# values blast/README.md states in prose are pinned, so the docs can't drift.
# README ranges are the measured min/max rounded outward to two decimals.
def _monitor(name, label):
    b = load_scene(SAMPLES / "blast" / name)["blast"]
    lay = {(x.get("name") or x.get("label")): x["arrays"] for x in b["layouts"]}[label]
    return {k: np.asarray(v, dtype=np.float64) for k, v in lay.items()}


MONITORS = [("truth.mtj", "monitor · false_claim"), ("truth.mtj", "monitor · negation"),
            ("refusal.mtj", "monitor · harmful"), ("refusal.mtj", "monitor · refused")]


def test_blast_readouts_are_well_formed():
    for name, label in MONITORS:
        m = _monitor(name, label)
        a = m["auroc"]
        assert a.shape == (25,) and len(m["labelled"]) == 48, label
        ok = ~np.isnan(a)
        assert ((a[ok] >= 0) & (a[ok] <= 1)).all(), label
        assert (m["null05"][ok] <= m["null95"][ok]).all(), label


def test_blast_matches_what_the_readme_says():
    harm = _monitor("refusal.mtj", "monitor · harmful")["auroc"]
    assert harm[1] == pytest.approx(0.77, abs=5e-3)            # "0.77 at layer 1"
    assert (harm[13:] >= 0.99).all()                            # "0.99 or more from layer 13"
    fc = _monitor("truth.mtj", "monitor · false_claim")
    assert fc["auroc"][0] == pytest.approx(0.54, abs=5e-3)     # "0.54 at layer 0"
    assert fc["auroc"][1:].min() == pytest.approx(0.637, abs=2e-3)  # "0.63 to 0.83"
    assert fc["auroc"][1:].max() == pytest.approx(0.821, abs=2e-3)
    assert int((fc["auroc"] > fc["null95"]).sum()) == 24       # "24 of 25 layers"
    neg = _monitor("truth.mtj", "monitor · negation")["auroc"]
    assert neg[15:].min() == pytest.approx(0.876, abs=2e-3)  # "0.87 to 0.93 from layer 15"
    assert neg[15:].max() == pytest.approx(0.929, abs=2e-3)


def test_refusal_layer_zero_is_one_point():
    # Every refusal pellet is read at the same template token, so layer 0 has
    # nothing to separate and its AUROC is undefined, as the README says.
    for label in ("monitor · harmful", "monitor · refused"):
        a = _monitor("refusal.mtj", label)["auroc"]
        assert np.isnan(a[0]) and not np.isnan(a[1:]).any()
