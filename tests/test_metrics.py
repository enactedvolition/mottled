"""Metric correctness on analytic cases + research summary."""

import numpy as np
import pytest

import metrics as M
import tiny as synthetic


def test_entropy_uniform_and_peaked():
    v = 32
    uniform = np.zeros(v)
    assert M.entropy(uniform) == pytest.approx(np.log(v))
    peaked = np.full(v, -1e9)
    peaked[3] = 0.0
    assert M.entropy(peaked) == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence():
    rng = np.random.default_rng(1)
    p = rng.normal(size=64)
    assert M.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    assert M.kl_divergence(p, rng.normal(size=64)) > 0


def test_straight_line_kinematics():
    line = np.column_stack([np.linspace(0, 9, 10), np.zeros(10)])
    assert M.path_length(line) == pytest.approx(9.0)
    assert M.integrated_curvature(line) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(M.velocity(line)[:, 0], 1.0)
    assert np.allclose(M.acceleration(line), 0.0)


def test_right_angle_turn():
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert M.integrated_curvature(path) == pytest.approx(np.pi / 2)
    assert M.path_length(path) == pytest.approx(2.0)


def test_layer_displacement_and_drift():
    h = np.zeros((3, 2, 4))
    h[1] = 1.0  # move every token by ||(1,1,1,1)|| = 2
    disp = M.layer_displacement(h)
    assert disp.shape == (2, 2)
    assert disp[0] == pytest.approx(2.0)
    drift = M.layer_cosine_drift(np.array([[[1.0, 0.0]], [[0.0, 1.0]]]))
    assert drift[0, 0] == pytest.approx(1.0)  # orthogonal vectors: drift 1


def test_entropy_collapse_and_stability():
    assert M.entropy_collapse(np.array([3.0, 2.0, 0.5])) == pytest.approx(2.5)
    ids = [[1, 2, 3], [1, 2, 3], [4, 5, 6]]
    stab = M.neighbor_stability(ids)
    assert stab[0] == pytest.approx(1.0) and stab[1] == pytest.approx(0.0)


def test_branch_divergence():
    a = np.zeros((5, 2))
    b = np.column_stack([np.arange(5.0), np.zeros(5)])
    assert np.allclose(M.branch_divergence(a, b), np.arange(5.0))


def test_summary_on_tiny_model():
    from projection import project

    traj = synthetic.capture("the capital of france is paris")
    coords, _ = project(traj.hidden)
    out = M.summarize(traj, coords, token=-1)
    expected = {"trajectory_length", "integrated_curvature", "avg_layer_cosine_drift",
                "avg_layer_displacement", "final_vector_norm", "entropy_collapse",
                "final_entropy", "nn_stability"}
    assert expected <= set(out)
    assert "avg_semantic_drift" not in out    # validity.md: not a semantic measure
    assert all(np.isfinite(v) for v in out.values())
    # `entropy_collapse` is first-layer minus last-layer entropy. The old
    # version of this test asserted it was positive, which held only because
    # the synthetic backend sharpened its logits with depth *by construction* —
    # it was checking the fixture, not the metric. A randomly-initialised model
    # has no reason to become more confident with depth, so what is pinned here
    # is the definition: the sign is whatever the two entropies say it is.
    ent = traj.entropy[:, -1]
    assert out["entropy_collapse"] == pytest.approx(
        float(ent[0] - ent[-1]), abs=1e-5)
    assert out["final_entropy"] == pytest.approx(float(ent[-1]), abs=1e-5)
