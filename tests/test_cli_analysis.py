"""The analysis commands (compare, sae, intervene, dose, arrays), bare
`mottled`, and the two measurement fixes (all-position steering, interleaved
MoE layer indices)."""
from __future__ import annotations

import io
import json
import types

import numpy as np
import pytest

import cli
import statefile
import tiny

PROMPTS = ["the capital of france is", "the capital of germany is"]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    return tiny.save_to(tmp_path_factory.mktemp("tiny") / "model")


@pytest.fixture(scope="module")
def runs(tmp_path_factory, model_dir):
    out = tmp_path_factory.mktemp("runs") / "t.mtj"
    assert cli.main(["capture", *PROMPTS, "--model", model_dir, "-o", str(out)]) == 0
    return out


def _json(capsys):
    return json.loads(capsys.readouterr().out)


def test_bare_mottled_prints_help_not_a_ui(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "usage: mottled" in out and "ui " in out


def test_compare_prints_json(runs, capsys):
    assert cli.main(["compare", str(runs)]) == 0
    d = _json(capsys)
    assert d["prompt_a"] == PROMPTS[0] and d["prompt_b"] == PROMPTS[1]
    assert d["hausdorff"] > 0 and len(d["profile"]) > 1


def test_compare_needs_two(tmp_path, model_dir, capsys):
    one = tmp_path / "one.mtj"
    cli.main(["capture", PROMPTS[0], "--model", model_dir, "-o", str(one)])
    assert cli.main(["compare", str(one)]) == 1


def test_sae_overlay_json(runs, tmp_path, capsys):
    import sae

    traj = statefile.load_stream(runs)[0][0]
    path = tmp_path / "d.npz"
    sae.save_npz(sae.demo_sae(traj.dim, n_features=32), path)
    assert cli.main(["sae", str(runs), "--sae", str(path), "-k", "3"]) == 0
    d = _json(capsys)
    assert len(d) == 2 and len(d[0]["top_features"]) <= 3
    assert 0 <= d[0]["layer"] < traj.n_layers


def test_sae_width_mismatch_is_one_line(runs, tmp_path, capsys):
    import sae

    path = tmp_path / "bad.npz"
    sae.save_npz(sae.demo_sae(7, n_features=8), path)
    assert cli.main(["sae", str(runs), "--sae", str(path)]) == 1
    assert "width" in capsys.readouterr().err


def test_intervene_json_and_scene(model_dir, tmp_path, capsys):
    scene = tmp_path / "s.mtj"
    assert cli.main(["intervene", PROMPTS[0], "--model", model_dir, "--layer", "-1",
                     "--direction-token", "paris", "--scale", "20",
                     "--target", "paris", "--all-positions", "-o", str(scene)]) == 0
    d = _json(capsys)
    assert d["positions"] == "all"
    assert "faithfulness" in d and "effect" in d["faithfulness"]
    assert scene.exists()
    assert cli.main(["validate", str(scene)]) == 0


def test_intervene_needs_a_direction(model_dir, capsys):
    assert cli.main(["intervene", PROMPTS[0], "--model", model_dir, "--layer", "1"]) == 2


def test_dose_json(model_dir, capsys):
    assert cli.main(["dose", PROMPTS[0], "--model", model_dir, "--layer", "-1",
                     "--direction-token", "paris", "--target", "paris",
                     "--grid=-1,0,1", "--n-random", "1"]) == 0
    d = _json(capsys)
    assert d["points"] and "spec" in d


@pytest.mark.parametrize("fmt", ["npz", "safetensors"])
def test_arrays_roundtrip(runs, tmp_path, fmt):
    out = tmp_path / f"a.{fmt}"
    assert cli.main(["arrays", str(runs), "--format", fmt, "-o", str(out)]) == 0
    arrays = statefile.read_stream(runs)
    if fmt == "npz":
        z = np.load(out)
        manifest = json.loads(str(z["__manifest__"]))
        got = {k: z[k] for k in z.files if k != "__manifest__"}
    else:
        from safetensors import safe_open
        from safetensors.numpy import load_file
        got = load_file(str(out))
        with safe_open(str(out), "np") as f:
            manifest = json.loads(f.metadata()["mtj_manifest"])
    assert len(manifest) == 2
    for k, v in arrays[1][1].items():
        np.testing.assert_array_equal(got[f"1/{k}"], v)


# ------------------------------------------------ measurement fixes
def test_all_position_steer_is_measured_at_every_position():
    """A steer pushed at every position is scored as the mean shift over all
    positions, not the last one alone."""
    import intervene as iv

    torch = pytest.importorskip("torch")
    model, tok = tiny.model(n_layers=4), tiny.tokenizer()
    from capture import capture

    base = capture(model, PROMPTS[0], tokenizer=tok, top_k=3)
    d = 20.0 * iv.direction_from_token(base, 5)
    branch = iv.intervene(model, PROMPTS[0], [iv.Perturb(base.n_layers - 1, d, token=None)],
                          tokenizer=tok)
    f = iv.score_against_control(model, PROMPTS[0], base, branch, d, base.n_layers - 1, 5,
                                 token=None, tokenizer=tok)
    every = np.mean([iv.target_logit_shift(base, branch, 5, t) for t in range(base.n_tokens)])
    assert f.steer_shift == pytest.approx(every)
    assert f.token == base.n_tokens - 1


@pytest.mark.parametrize("cfg,n_blocks,returned,want", [
    # one entry per block, dense ones None: the position is the block
    ({}, 4, [None, 1, None, 1], [1, 3]),
    # only MoE blocks returned, interleaved every 2nd block (Qwen-MoE)
    ({"decoder_sparse_step": 2}, 4, [1, 1], [1, 3]),
    # DeepSeek: first k dense, then every block
    ({"first_k_dense_replace": 1, "moe_layer_freq": 1}, 4, [1, 1, 1], [1, 2, 3]),
    # mlp_only_layers removes blocks from a step-1 stack
    ({"decoder_sparse_step": 1, "mlp_only_layers": [0, 2]}, 4, [1, 1], [1, 3]),
    # nothing in config: the tail, as before
    ({}, 4, [1, 1], [2, 3]),
])
def test_moe_block_indices(cfg, n_blocks, returned, want):
    from capture import moe_block_indices

    assert moe_block_indices(types.SimpleNamespace(**cfg), n_blocks, returned) == want
