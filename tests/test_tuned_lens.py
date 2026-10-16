"""The tuned lens: applied exactly, refused when it does not fit, recorded
wherever a readout travels, and named wherever one is shown.

Offline against tiny models, except the one test that loads the published
GPT-2 lens: that is the check that the translator indexing matches the one
the lens was trained with, which no locally-built lens can establish.
"""
import json

import numpy as np
import pytest

import tiny
import tuned_lens

torch = pytest.importorskip("torch")

PROMPT = "the capital of france is"
N_LAYERS = 4          # captured: embedding + 3 blocks
DIM = 32


def _lens(seed=0, scale=0.1, n=N_LAYERS - 1, dim=DIM, bias=True):
    rng = np.random.default_rng(seed)
    return tuned_lens.TunedLens(
        weight=(scale * rng.standard_normal((n, dim, dim))).astype(np.float32),
        bias=((scale * rng.standard_normal((n, dim))) if bias
              else np.zeros((n, dim))).astype(np.float32),
        source="test", revision=None, base_model_revision=None)


def _write(path, lens, bias=True):
    """A lens directory in the tuned-lens package's own format."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({
        "base_model_name_or_path": "tiny", "d_model": DIM, "bias": bias,
        "num_hidden_layers": lens.n_layers, "base_model_revision": "abc123",
        "lens_type": "linear_tuned_lens"}))
    state = {f"{i}.weight": torch.from_numpy(lens.weight[i]) for i in range(lens.n_layers)}
    if bias:
        state.update({f"{i}.bias": torch.from_numpy(lens.bias[i])
                      for i in range(lens.n_layers)})
    torch.save(state, path / "params.pt")
    return path


def test_identity_lens_is_the_logit_lens():
    zero = _lens(scale=0.0)
    raw = tiny.capture(PROMPT, n_layers=N_LAYERS)
    tuned = tiny.capture(PROMPT, n_layers=N_LAYERS, lens=zero)
    np.testing.assert_array_equal(raw.logits, tuned.logits)
    assert raw.meta["lens"] == {"kind": "logit"}
    assert tuned.meta["lens"]["kind"] == "tuned"


def test_translator_is_residual_affine_and_skips_the_last_layer():
    from capture import HookCapture, logit_lens

    lens = _lens()
    raw = tiny.capture(PROMPT, n_layers=N_LAYERS, keep_logits=True)
    tuned = tiny.capture(PROMPT, n_layers=N_LAYERS, lens=lens)
    model = tiny.model(n_layers=N_LAYERS - 1, dim=DIM)
    with HookCapture(model) as cap:
        adapter = cap.adapter
    h = torch.from_numpy(raw.hidden)
    l = 1
    by_hand = h[l] + h[l] @ torch.from_numpy(lens.weight[l]).T + torch.from_numpy(lens.bias[l])
    expected = logit_lens(by_hand[None], adapter)[0].numpy()
    np.testing.assert_allclose(tuned.logits[l].astype(np.float32), expected,
                               rtol=1e-2, atol=1e-2)       # stored as float16
    # the last layer is the head's own input: no translator, same readout
    np.testing.assert_array_equal(tuned.logits[-1], raw.logits[-1])
    assert not np.allclose(tuned.logits[l], raw.logits[l])


@pytest.mark.parametrize("n, dim", [(N_LAYERS, DIM), (N_LAYERS - 2, DIM),
                                    (N_LAYERS - 1, DIM * 2)])
def test_a_lens_for_another_model_is_refused(n, dim):
    with pytest.raises(ValueError, match="translators"):
        tiny.capture(PROMPT, n_layers=N_LAYERS, lens=_lens(n=n, dim=dim))


@pytest.mark.parametrize("bias", [True, False])
def test_reads_the_tuned_lens_directory_format(tmp_path, bias):
    lens = _lens(bias=bias)
    loaded = tuned_lens.from_dir(_write(tmp_path / "lens", lens, bias=bias))
    np.testing.assert_array_equal(loaded.weight, lens.weight)
    np.testing.assert_array_equal(loaded.bias, lens.bias)
    assert loaded.base_model_revision == "abc123"
    assert tuned_lens.resolve(str(tmp_path / "lens"), "tiny").n_layers == lens.n_layers


def test_resolve():
    assert tuned_lens.resolve("logit", "gpt2") is None
    with pytest.raises(ValueError, match="'logit', 'tuned' or a tuned-lens directory"):
        tuned_lens.resolve("/no/such/lens", "gpt2")


def test_the_config_lens_reaches_the_record(tmp_path):
    import pipeline
    import provenance
    from config import MarbleConfig

    path = _write(tmp_path / "lens", _lens())
    cfg = MarbleConfig(model="tiny", lens=str(path), use_cache=False,
                       capture_components=False, capture_attention=False)
    traj = pipeline._capture_with(cfg, PROMPT, model=tiny.model(n_layers=N_LAYERS - 1),
                                  tokenizer=tiny.tokenizer())
    rec = provenance.record(cfg, trajs=[traj])
    assert rec["config"]["lens"] == str(path)
    assert rec["models"][0]["lens"]["kind"] == "tuned"
    assert rec["models"][0]["lens"]["source"] == str(path)


def test_cli_capture_records_the_lens(tmp_path):
    import cli
    import statefile

    model_dir = tiny.save_to(tmp_path / "model", n_layers=N_LAYERS)
    lens = _write(tmp_path / "lens", _lens())
    out, scene = tmp_path / "t.mtj", tmp_path / "s.mtj"
    assert cli.main(["capture", PROMPT, "--model", model_dir, "--lens", str(lens),
                     "-o", str(out)]) == 0
    assert cli.main(["project", str(out), "-o", str(scene)]) == 0
    (traj, _, _), = statefile.load_stream(out)
    assert traj.meta["lens"]["kind"] == "tuned"
    # `project` carries the capture stage's lens into the scene's record
    manifest, _ = statefile.read_container(scene)
    assert manifest["analysis"]["config"]["lens"] == str(lens)


def test_explain_names_the_readout_and_cautions_the_raw_one():
    import attractor
    import density
    import projection

    texts = {}
    for name, lens in (("raw", None), ("tuned", _lens())):
        traj = tiny.capture(PROMPT, n_layers=N_LAYERS, lens=lens)
        coords, _ = projection.project(traj.hidden)
        report = attractor.analyze(traj, coords, density.compute_density(coords))
        texts[name] = attractor.explain(report, traj)
    assert "logit lens top-1" in texts["raw"]
    assert tuned_lens.EARLY_LAYER_CAVEAT in texts["raw"]
    assert "tuned lens top-1" in texts["tuned"]
    assert tuned_lens.EARLY_LAYER_CAVEAT not in texts["tuned"]


@pytest.mark.network
def test_published_gpt2_lens_reads_closer_to_the_output_than_the_raw_lens():
    """The check that the indexing is the one the lens was trained with: a
    translator applied one layer off makes the readout worse, not better."""
    from capture import capture, load_model

    model, tok = load_model("gpt2")
    lens = tuned_lens.from_hub("gpt2")
    assert lens.revision                       # the record names a commit

    def kl_to_output(traj):
        lg = traj.logits.astype(np.float64)
        lp = lg - lg.max(-1, keepdims=True)
        lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
        return (np.exp(lp[-1]) * (lp[-1] - lp)).sum(-1).mean(-1)

    prompt = "The quick brown fox jumps over the lazy dog because it was"
    raw = kl_to_output(capture(model, prompt, tokenizer=tok))
    tuned = kl_to_output(capture(model, prompt, tokenizer=tok, lens=lens))
    assert (tuned[:-1] < raw[:-1]).all()
    assert tuned[-1] == raw[-1] == 0.0
