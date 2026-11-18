"""Animation continuous + UI interactive (spec tests 7-8)."""

import numpy as np
import pytest

from config import MarbleConfig
from ui import render, run_pipeline
import tiny as synthetic


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    return MarbleConfig(model="tiny", use_cache=True,
                        cache_dir=str(tmp_path_factory.mktemp("cache")))


@pytest.fixture(scope="module")
def result(cfg):
    return run_pipeline(cfg, "The capital of France is", **synthetic.mt())


def test_pipeline_artifacts(cfg, result):
    traj = result["traj"]
    L, T = traj.n_layers, traj.n_tokens
    assert result["coords"].shape == (L, T, 2)
    assert result["mesh"].z.shape == (cfg.grid_size, cfg.grid_size)
    assert len(result["trajectories"]) == T  # all_tokens mode
    assert all(t.points.shape == (L, 3) for t in result["trajectories"])


def test_pipeline_reports_uncertainty(cfg, result):
    traj = result["traj"]
    q = result["quality"]
    assert q.preservation.shape == (traj.n_layers, traj.n_tokens)
    assert (q.preservation >= 0).all() and (q.preservation <= 1).all()
    # synthetic default config runs the density bootstrap
    if cfg.density_bootstrap >= 2:
        assert result["landscape"].density_se is not None
        assert result["landscape"].density_se.shape == result["mesh"].z.shape


def test_pipeline_cache_roundtrip(cfg, result):
    again = run_pipeline(cfg, "The capital of France is", **synthetic.mt())
    assert np.array_equal(again["coords"], result["coords"])


def test_render_figure(cfg, result):
    fig = render(result["traj"], result["mesh"], result["trajectories"],
                 result["fine_paths"], frames_per_layer=cfg.frames_per_layer)
    types = [tr.type for tr in fig.data]
    assert types[0] == "surface"
    assert types.count("scatter3d") == result["traj"].n_tokens + 1  # lines + marbles
    assert len(fig.frames) == min(len(p) for p in result["fine_paths"])
    assert fig.layout.sliders and fig.layout.updatemenus  # scrubber + play/pause


def test_marble_animation_is_continuous(result):
    """Marble positions across frames form a gap-free path on the terrain."""
    for path in result["fine_paths"]:
        assert np.isfinite(path).all()
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        span = np.linalg.norm(path.max(axis=0) - path.min(axis=0))
        assert steps.max() < 0.5 * span  # no teleporting between frames
        z = result["mesh"].z
        assert path[:, 2].min() >= z.min() - 0.1  # stays on/above the terrain


def test_streamlit_app_interactive():
    """Drive the real Streamlit app headlessly: run capture, scrub, inspect."""
    pytest.importorskip("streamlit.testing.v1")
    from tests.apptest import app_test

    at = app_test(default_timeout=120)
    at.run()
    assert not at.exception

    at.text_area(key="prompt").set_value("The capital of France is")
    at.selectbox(key="model").select("gpt2")
    at.button(key="run").click()
    at.run()
    assert not at.exception
    assert at.session_state["result"] is not None

    traj = at.session_state["result"]["traj"]
    at.slider(key="layer").set_value(0)
    at.selectbox(key="token").select(traj.n_tokens - 1)
    at.run()
    assert not at.exception


def test_streamlit_app_surfaces_projection_fidelity_inline():
    """Fidelity is stated in the main view, not buried in a collapsed panel."""
    pytest.importorskip("streamlit.testing.v1")
    from tests.apptest import app_test

    at = app_test(default_timeout=120)
    at.run()
    at.text_area(key="prompt").set_value("The capital of France is")
    at.selectbox(key="model").select("gpt2")
    at.button(key="run").click()
    at.run()
    assert not at.exception
    texts = [m.value for m in at.markdown]
    assert any("Projection fidelity" in t for t in texts)


def test_streamlit_app_tests_the_basin_on_demand():
    """The null runs only when asked (200 refits are too slow for every
    rerun), and its answer lands in the explorer. Raw GPT-2 states carry the
    position-0 sink, so the answer here is the refusal that names it."""
    pytest.importorskip("streamlit.testing.v1")
    from tests.apptest import app_test

    at = app_test(default_timeout=120)
    at.run()
    at.text_area(key="prompt").set_value("The capital of France is")
    at.selectbox(key="model").select("gpt2")
    at.button(key="run").click()
    at.run()
    assert not any("Is it more than slowing down?" in m.value for m in at.markdown)
    at.button(key="basin_null").click()
    at.run()
    assert not at.exception
    answer = [m.value for m in at.markdown if "Is it more than slowing down?" in m.value]
    assert len(answer) == 1 and "not tested" in answer[0] and "`rms`" in answer[0]


def test_flat_api_survives_the_module_split():
    """`ui.py` was split into pipeline.py + render.py; every name the README
    documents must still import from `ui`, and be the same object."""
    import pipeline
    import render
    import ui

    for name in ("run_pipeline", "run_scene", "run_compare", "run_intervention",
                 "attach_features", "degraded_note"):
        assert getattr(ui, name) is getattr(pipeline, name), name
    for name in ("render", "render_feature_field", "field_rgb"):
        assert getattr(ui, name) is getattr(render, name), name
    assert callable(ui.main)


def test_pipeline_and_render_are_importable_without_streamlit(monkeypatch):
    """The split exists so a scene can be built and drawn headlessly: neither
    module may pull Streamlit in at import time."""
    import subprocess
    import sys

    code = ("import sys;"
            "sys.modules['streamlit'] = None;"   # any import would raise
            "import pipeline, render;"
            "print('ok')")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(__import__('pathlib').Path(__file__).parent.parent))
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
