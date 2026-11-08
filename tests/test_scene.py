"""Phase 4 — multi-prompt scenes, attention flow, interactive patching."""

import numpy as np
import pytest

from config import MarbleConfig
import tiny as synthetic
from ui import render, run_compare, run_intervention, run_pipeline, run_scene

PROMPTS = ["the capital of france is",
           "the capital of germany is",
           "the king of spain is"]


# ------------------------------------------------------------ attention
def test_tiny_attention_is_causal_stochastic():
    traj = synthetic.capture(PROMPTS[0], capture_attention=True)
    traj.validate()
    L, T, _ = traj.hidden.shape
    att = traj.attention
    assert att.shape == (L - 1, T, T)
    assert np.allclose(att.sum(axis=-1), 1.0, atol=1e-5)     # rows are distributions
    assert np.allclose(att, np.tril(att))                     # causal mask
    assert (att >= 0).all()
    assert synthetic.capture(PROMPTS[0]).attention is None    # off by default
    again = synthetic.capture(PROMPTS[0], capture_attention=True)
    assert np.array_equal(att, again.attention)               # deterministic


def test_validate_rejects_bad_attention_shape():
    traj = synthetic.capture(PROMPTS[0], capture_attention=True)
    traj.attention = traj.attention[:, :2, :2]
    with pytest.raises(ValueError):
        traj.validate()


def test_torch_attention_capture_matches_reference():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tests.test_capture import VOCAB_SIZE, DummyTokenizer

    from capture import capture

    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attn_implementation="eager",
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    tok = DummyTokenizer()
    traj = capture(model, PROMPTS[0], tokenizer=tok, capture_attention=True)
    traj.validate()
    T = traj.n_tokens
    assert traj.attention.shape == (3, T, T)
    assert np.allclose(traj.attention.sum(axis=-1), 1.0, atol=1e-4)
    assert np.allclose(traj.attention, np.tril(traj.attention), atol=1e-6)

    with torch.no_grad():
        out = model(tok(PROMPTS[0])["input_ids"], output_attentions=True)
    ref = torch.stack([a.float().mean(dim=1)[0] for a in out.attentions]).numpy()
    assert np.allclose(traj.attention, ref, atol=1e-5)


# ------------------------------------------------------------------ scenes
@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    return MarbleConfig(model="tiny", use_cache=True,
                        cache_dir=str(tmp_path_factory.mktemp("cache")))


@pytest.fixture(scope="module")
def scene(cfg):
    return run_scene(cfg, PROMPTS, **synthetic.mt())


def test_scene_artifacts(cfg, scene):
    assert scene["prompts"] == PROMPTS
    assert len(scene["trajs"]) == len(scene["coords_list"]) == 3
    assert len(scene["trajectories_list"]) == len(scene["fine_paths_list"]) == 3
    assert len(scene["comparisons"]) == 2
    for traj, coords, trajs in zip(scene["trajs"], scene["coords_list"],
                                   scene["trajectories_list"]):
        assert coords.shape == (traj.n_layers, traj.n_tokens, 2)
        assert len(trajs) == traj.n_tokens
        assert all(t.points.shape == (traj.n_layers, 3) for t in trajs)
    # run-0 view mirrors run_pipeline keys; pairwise aliases only for 2 runs
    assert scene["traj"] is scene["trajs"][0]
    assert scene["trajectories"] is scene["trajectories_list"][0]
    assert "traj_b" not in scene
    assert scene["mesh"].z.shape == (cfg.grid_size, cfg.grid_size)


def test_scene_cache_roundtrip_and_compare_alias(cfg, scene):
    again = run_scene(cfg, PROMPTS, **synthetic.mt())
    assert np.array_equal(again["coords_list"][2], scene["coords_list"][2])
    pair = run_compare(cfg, PROMPTS[0], PROMPTS[1], **synthetic.mt())
    assert pair["traj_b"] is pair["trajs"][1]
    assert pair["comparison"] is pair["comparisons"][0]
    assert pair["prompt_b"] == PROMPTS[1]


def test_gauge_is_what_the_scene_projects_and_measures():
    from dataclasses import replace

    from sklearn.decomposition import PCA

    import projection
    cfg = MarbleConfig(model="tiny", use_cache=False, gauge="rms+center", density_bootstrap=0)
    gauged = run_scene(cfg, PROMPTS, **synthetic.mt())
    raw = run_scene(replace(cfg, gauge="none"), PROMPTS, **synthetic.mt())
    spaces = projection.gauge([t.hidden for t in gauged["trajs"]], "rms+center")
    union = np.concatenate([g.reshape(-1, g.shape[-1]) for g in spaces]).astype(np.float64)
    # fidelity is measured on the states that were projected, not the raw ones
    ev = PCA(2, svd_solver="full").fit(union).explained_variance_ratio_.sum()
    assert gauged["quality_list"][0].explained_variance == pytest.approx(ev, rel=1e-6)
    assert not np.allclose(gauged["coords_list"][1], raw["coords_list"][1])
    # the divergence curves are read in hidden space, so no gauge moves them
    assert np.array_equal(gauged["comparisons"][0].relative, raw["comparisons"][0].relative)
    one = run_pipeline(cfg, PROMPTS[0], **synthetic.mt())
    expect, _ = projection.project(projection.gauge([one["traj"].hidden], "rms+center")[0])
    assert np.array_equal(one["coords"], expect)


def test_scene_requires_prompts(cfg):
    with pytest.raises(ValueError):
        run_scene(cfg, [], **synthetic.mt())


def test_render_scene_runs(cfg, scene):
    extra = [(scene["trajs"][2], scene["trajectories_list"][2], scene["fine_paths_list"][2])]
    fig = render(scene["trajs"][0], scene["mesh"],
                 scene["trajectories_list"][0], scene["fine_paths_list"][0],
                 frames_per_layer=cfg.frames_per_layer,
                 traj_b=scene["trajs"][1],
                 trajectories_b=scene["trajectories_list"][1],
                 fine_paths_b=scene["fine_paths_list"][1],
                 extra_runs=extra)
    lines = [tr for tr in fig.data if tr.type == "scatter3d" and tr.mode == "lines+markers"]
    n_expected = sum(t.n_tokens for t in scene["trajs"])
    assert len(lines) == n_expected
    assert {tr.name[:4] for tr in lines} == {"A · ", "B · ", "C · "}
    dashes = {tr.name[:1]: tr.line.dash for tr in lines}
    assert dashes["A"] is None and dashes["B"] == "dash" and dashes["C"] == "dot"
    n_frames = min(len(p) for paths in scene["fine_paths_list"] for p in paths)
    assert len(fig.frames) == n_frames


def test_render_attention_flow(cfg):
    result = run_scene(cfg, [PROMPTS[0]], **synthetic.mt())
    traj = result["traj"]
    assert traj.attention is not None  # cfg.capture_attention defaults on

    fig = render(traj, result["mesh"], result["trajectories"], result["fine_paths"],
                 current_layer=traj.n_layers - 1, show_attention=True)
    names = [tr.name for tr in fig.data]
    assert "attention" in names
    att = fig.data[names.index("attention")]
    assert len(att.x) % 3 == 0 and len(att.x) > 0  # segments with None separators

    fig0 = render(traj, result["mesh"], result["trajectories"], result["fine_paths"],
                  current_layer=0, show_attention=True)
    assert "attention" not in [tr.name for tr in fig0.data]  # no attention into layer 0


# ----------------------------------------------------------- interactive patching
def test_run_intervention_pipeline():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from tests.test_capture import VOCAB_SIZE, DummyTokenizer

    from intervene import Perturb

    torch.manual_seed(0)
    mcfg = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attn_implementation="eager",
    )
    model = transformers.LlamaForCausalLM(mcfg).eval()
    cfg = MarbleConfig(model="tiny", use_cache=False)

    rng = np.random.default_rng(0)
    edits = [Perturb(2, 40.0 * rng.normal(size=32).astype(np.float32), token=-1)]
    result = run_intervention(cfg, PROMPTS[0], edits, model, DummyTokenizer())

    assert result["traj_b"].meta.get("counterfactual") is True
    assert "patched" in result["prompt_b"]
    assert result["comparison"].hausdorff > 0
    div = result["divergence"]
    assert div.profile.shape == (result["traj"].n_layers,)
    assert div.profile[:2].max() < 1e-6  # nothing moves before the edited layer
    assert div.profile[2:].min() > 0     # everything after it does
    assert 2 <= div.onset


def test_streamlit_app_scene_mode():
    """Drive the real app with two overlay prompts and attention flow on."""
    pytest.importorskip("streamlit.testing.v1")
    from tests.apptest import app_test

    at = app_test(default_timeout=120)
    at.run()
    at.text_area(key="prompt").set_value(PROMPTS[0])
    at.text_area(key="prompt_b").set_value(PROMPTS[1] + "\n" + PROMPTS[2])
    at.selectbox(key="model").select("gpt2")
    at.checkbox(key="attention_flow").check()
    at.button(key="run").click()
    at.run()
    assert not at.exception
    result = at.session_state["result"]
    assert len(result["trajs"]) == 3
    assert len(result["comparisons"]) == 2
    assert result["traj"].attention is not None
