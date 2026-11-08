"""Phase 2 — trajectory comparison: geometry, alignment, A/B pipeline."""

from dataclasses import replace

import numpy as np
import pytest

import compare as C
from config import MarbleConfig
import tiny as synthetic
from projection import project, project_joint
from ui import render, run_compare

PROMPT_A = "the capital of france is"
PROMPT_B = "the capital of germany is"


# ------------------------------------------------------------------ geometry
def test_hausdorff_identity_and_known_offset():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert C.hausdorff(a, a) == pytest.approx(0.0)
    b = a + np.array([3.0, 4.0])  # every point shifted by a 3-4-5 triangle
    assert C.hausdorff(a, b) == pytest.approx(5.0)
    assert C.hausdorff(b, a) == pytest.approx(5.0)  # symmetric


def test_hausdorff_ignores_traversal_order():
    a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert C.hausdorff(a, a[::-1]) == pytest.approx(0.0)


def test_dtw_identity_is_zero():
    path = np.random.default_rng(0).normal(size=(7, 3))
    res = C.dtw(path, path)
    assert res.distance == pytest.approx(0.0)
    assert res.normalized == pytest.approx(0.0)


def test_dtw_known_small_case():
    # 1-D sequences [0, 1, 2] vs [0, 2]: optimal alignment pairs
    # (0,0), (1,0)|(1,1), (2,1) -> total cost 1.0
    a = np.array([[0.0], [1.0], [2.0]])
    b = np.array([[0.0], [2.0]])
    res = C.dtw(a, b)
    assert res.distance == pytest.approx(1.0)


def test_dtw_path_is_valid_monotone_alignment():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(6, 2)), rng.normal(size=(9, 2))
    res = C.dtw(a, b)
    path = res.path
    assert tuple(path[0]) == (0, 0)
    assert tuple(path[-1]) == (len(a) - 1, len(b) - 1)
    steps = np.diff(path, axis=0)
    assert (steps >= 0).all() and (steps <= 1).all()  # monotone, unit moves
    assert (steps.sum(axis=1) >= 1).all()             # always advances


def test_dtw_aligns_time_warped_paths():
    # The same route traversed at different speeds: DTW stays near zero
    # while the pointwise (branch_divergence-style) distance does not.
    t_fast = np.linspace(0, 1, 40)
    t_slow = np.linspace(0, 1, 100) ** 2
    a = np.column_stack([t_fast, np.sin(2 * np.pi * t_fast)])
    b = np.column_stack([t_slow, np.sin(2 * np.pi * t_slow)])
    res = C.dtw(a, b)
    pointwise = np.linalg.norm(a - b[: len(a)], axis=1).mean()
    assert res.normalized < 0.1 * pointwise
    assert res.normalized < 0.05


def test_geometry_rejects_empty():
    a = np.zeros((3, 2))
    with pytest.raises(ValueError):
        C.hausdorff(a, np.zeros((0, 2)))
    with pytest.raises(ValueError):
        C.dtw(np.zeros((0, 2)), a)


# ----------------------------------------------------------------- alignment
def test_shared_prefix():
    assert C.shared_prefix(["the", "capital", "of", "france"],
                           ["the", "capital", "of", "germany"]) == 3
    assert C.shared_prefix(["a"], ["b"]) == 0
    assert C.shared_prefix(["a", "b"], ["a", "b"]) == 2
    assert C.shared_prefix([], ["a"]) == 0


# ------------------------------------------------------------ joint projection
def test_project_joint_shared_space():
    ta = synthetic.capture(PROMPT_A)
    tb = synthetic.capture(PROMPT_B)
    (ca, cb), proj = project_joint([ta.hidden, tb.hidden])
    assert ca.shape == (ta.n_layers, ta.n_tokens, 2)
    assert cb.shape == (tb.n_layers, tb.n_tokens, 2)
    assert proj.fitted
    # The shared fit must equal projecting the concatenated states directly.
    both = np.concatenate([ta.hidden, tb.hidden], axis=1)
    coords, _ = project(both)
    assert np.allclose(np.concatenate([ca, cb], axis=1), coords, atol=1e-4)


def test_project_joint_rejects_mismatched_dims():
    with pytest.raises(ValueError):
        project_joint([np.zeros((2, 3, 8)), np.zeros((2, 3, 16))])
    with pytest.raises(ValueError):
        project_joint([])


# ------------------------------------------------------------------- compare
def test_compare_self_is_null():
    traj = synthetic.capture(PROMPT_A)
    cmp = C.compare(traj, traj)
    assert cmp.shared_tokens == traj.n_tokens
    assert cmp.hausdorff == pytest.approx(0.0)
    assert cmp.dtw.distance == pytest.approx(0.0)
    assert np.allclose(cmp.profile, 0.0)
    assert cmp.onset_token is None
    assert cmp.readout_changed is None


def test_compare_ab_prompts():
    ta = synthetic.capture(PROMPT_A)
    tb = synthetic.capture(PROMPT_B)
    (ca, cb), _ = project_joint([ta.hidden, tb.hidden])
    cmp = C.compare(ta, tb, ca, cb)
    assert cmp.shared_tokens == 3          # "the capital of"
    assert cmp.token == ta.n_tokens - 1
    assert cmp.hausdorff > 0
    assert cmp.dtw.distance > 0
    assert cmp.profile.shape == (ta.n_layers,)
    assert cmp.positionwise.shape == (ta.n_layers, min(ta.n_tokens, tb.n_tokens))
    assert np.isfinite(cmp.positionwise).all()
    assert cmp.onset_token is not None
    assert 0 <= cmp.onset_layer < ta.n_layers


def test_compare_divergence_curves_are_hidden_space_and_scale_free():
    ta = synthetic.capture(PROMPT_A)
    tb = synthetic.capture(PROMPT_B)
    (ca, cb), _ = project_joint([ta.hidden, tb.hidden])
    cmp = C.compare(ta, tb, ca, cb)
    h_a = ta.hidden[:, -1].astype(np.float64)
    h_b = tb.hidden[:, -1].astype(np.float64)
    np.testing.assert_allclose(
        cmp.relative, np.linalg.norm(h_b - h_a, axis=1) / np.linalg.norm(h_a, axis=1), rtol=1e-5)
    np.testing.assert_allclose(
        cmp.cosine, (h_a * h_b).sum(1) / np.linalg.norm(h_a, axis=1) / np.linalg.norm(h_b, axis=1),
        rtol=1e-5)
    # the projection does not enter them, and neither does a common scale
    same = C.compare(ta, tb)
    assert np.array_equal(same.relative, cmp.relative) and np.array_equal(same.cosine, cmp.cosine)
    scaled = C.compare(replace(ta, hidden=ta.hidden * 7), replace(tb, hidden=tb.hidden * 7))
    np.testing.assert_allclose(scaled.relative, cmp.relative, rtol=1e-5)
    np.testing.assert_allclose(scaled.cosine, cmp.cosine, rtol=1e-5)
    self_cmp = C.compare(ta, ta)
    assert np.allclose(self_cmp.relative, 0.0) and np.allclose(self_cmp.cosine, 1.0)


def test_compare_requires_commensurable_runs():
    ta = synthetic.capture(PROMPT_A)
    tb = synthetic.capture(PROMPT_B, n_layers=5)
    with pytest.raises(ValueError):
        C.compare(ta, tb)
    with pytest.raises(ValueError):  # coords must come as a pair
        C.compare(ta, synthetic.capture(PROMPT_B), coords_a=np.zeros((13, 5, 2)))


# ------------------------------------------------------------------ pipeline
@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    return MarbleConfig(model="tiny", use_cache=True,
                        cache_dir=str(tmp_path_factory.mktemp("cache")))


@pytest.fixture(scope="module")
def result(cfg):
    return run_compare(cfg, PROMPT_A, PROMPT_B, **synthetic.mt())


def test_compare_pipeline_artifacts(cfg, result):
    ta, tb = result["traj"], result["traj_b"]
    assert result["coords"].shape == (ta.n_layers, ta.n_tokens, 2)
    assert result["coords_b"].shape == (tb.n_layers, tb.n_tokens, 2)
    assert result["mesh"].z.shape == (cfg.grid_size, cfg.grid_size)
    assert len(result["trajectories"]) == ta.n_tokens
    assert len(result["trajectories_b"]) == tb.n_tokens
    assert all(t.points.shape == (ta.n_layers, 3) for t in result["trajectories"])
    assert all(t.points.shape == (tb.n_layers, 3) for t in result["trajectories_b"])
    cmp = result["comparison"]
    assert isinstance(cmp, C.TrajectoryComparison)
    assert cmp.shared_tokens == 3


def test_compare_pipeline_cache_roundtrip(cfg, result):
    again = run_compare(cfg, PROMPT_A, PROMPT_B, **synthetic.mt())
    assert np.array_equal(again["coords"], result["coords"])
    assert np.array_equal(again["coords_b"], result["coords_b"])


def test_render_overlay(cfg, result):
    fig = render(result["traj"], result["mesh"],
                 result["trajectories"], result["fine_paths"],
                 frames_per_layer=cfg.frames_per_layer,
                 traj_b=result["traj_b"],
                 trajectories_b=result["trajectories_b"],
                 fine_paths_b=result["fine_paths_b"])
    n_a, n_b = result["traj"].n_tokens, result["traj_b"].n_tokens
    lines = [tr for tr in fig.data if tr.type == "scatter3d" and tr.mode == "lines+markers"]
    assert len(lines) == n_a + n_b
    a_lines = [tr for tr in lines if tr.name.startswith("A · ")]
    b_lines = [tr for tr in lines if tr.name.startswith("B · ")]
    assert len(a_lines) == n_a and len(b_lines) == n_b
    assert all(tr.line.dash == "dash" for tr in b_lines)
    assert all(tr.line.dash is None for tr in a_lines)
    assert len(fig.frames) == min(len(p) for p in
                                  result["fine_paths"] + result["fine_paths_b"])


def test_streamlit_app_compare_mode():
    """Drive the real app headlessly in A/B mode."""
    pytest.importorskip("streamlit.testing.v1")
    from tests.apptest import app_test

    at = app_test(default_timeout=120)
    at.run()
    at.text_area(key="prompt").set_value(PROMPT_A)
    at.text_area(key="prompt_b").set_value(PROMPT_B)
    at.selectbox(key="model").select("gpt2")
    at.button(key="run").click()
    at.run()
    assert not at.exception
    result = at.session_state["result"]
    assert result["traj_b"] is not None
    assert result["comparison"].shared_tokens == 3
