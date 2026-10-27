"""Activation patching (experiments/patching.py): what a patch writes, where,
and that the logits it reports are the model's own.

Offline on two tiny in-process models: the suite's Llama, and a GPT-2 built
here, because GPT-2 is what the experiment patches and its block signature,
learned positions and final LayerNorm are a different hook path from Llama's.
The one network test repeats the identity and the positive control on real
GPT-2 at the experiment's pinned revision.
"""
import functools
import json

import numpy as np
import pytest

import tiny

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from experiments import patching  # noqa: E402
from intervene import SetState  # noqa: E402
from tuned_lens import TunedLens  # noqa: E402

PROMPT = "the capital of france is paris and the residual stream"
OTHER = "hello world the quick brown fox jumps over the dog"
N_BLOCKS = 3


def _ids(text):
    return tiny.tokenizer()(text)["input_ids"]


IDS, OTHER_IDS = _ids(PROMPT), _ids(OTHER)
assert len(IDS) == len(OTHER_IDS)


@functools.lru_cache(maxsize=1)
def _gpt2():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=tiny.vocab_size(), n_positions=64,
                                  n_embd=32, n_layer=N_BLOCKS, n_head=4)
    return transformers.GPT2LMHeadModel(cfg).eval()


@pytest.fixture(params=["llama", "gpt2"])
def model(request):
    return tiny.model(n_layers=N_BLOCKS) if request.param == "llama" else _gpt2()


def _lens(model, seed=0):
    """Random translators: exercises the tuned path, means nothing."""
    dim = model.config.hidden_size
    rng = np.random.default_rng(seed)
    return TunedLens(weight=(0.1 * rng.standard_normal((N_BLOCKS, dim, dim))).astype(np.float32),
                     bias=(0.1 * rng.standard_normal((N_BLOCKS, dim))).astype(np.float32),
                     source="synthetic (test only)", revision=None, base_model_revision=None)


def test_self_patch_is_bit_identical(model):
    base = patching.clean(model, IDS)
    for layer in range(N_BLOCKS + 1):
        for position in (0, len(IDS) // 2, len(IDS) - 1):
            p = patching.patch(model, IDS, layer, position, base.hidden[layer, position],
                               clean_run=base)
            np.testing.assert_array_equal(p.logits, base.logits)
            np.testing.assert_array_equal(p.state_after, p.state_before)


def test_clean_logits_are_the_models_own_output(model):
    readout = (2, len(IDS) - 1)
    base = patching.clean(model, IDS, readout)
    x = torch.tensor([IDS])
    with torch.no_grad():
        same_call = model(x, logits_to_keep=torch.tensor(readout), use_cache=False).logits[0]
        full = model(x).logits[0, list(readout)]
    assert base.logits.dtype == np.float32 and base.logits.shape == (2, tiny.vocab_size())
    # the hooks change nothing: bit-identical to the same call without them
    np.testing.assert_array_equal(base.logits, same_call.numpy())
    # and the kept rows are those positions of the full output; the head ran
    # on a (2, D) slice instead of (T, D), so agreement is to rounding
    np.testing.assert_allclose(base.logits, full.numpy(), rtol=0, atol=1e-5)
    assert base.hidden.shape == (N_BLOCKS + 1, len(IDS), model.config.hidden_size)


def test_a_patch_reaches_its_own_position_and_later_ones_only(model):
    every = tuple(range(len(IDS)))
    t = len(IDS) // 2
    base = patching.clean(model, IDS, every)
    donor = patching.clean(model, OTHER_IDS, every)
    for layer in range(N_BLOCKS + 1):
        p = patching.patch(model, IDS, layer, t, donor.hidden[layer, t], readout=every,
                           clean_run=base)
        np.testing.assert_array_equal(p.logits[:t], base.logits[:t])
        assert not np.array_equal(p.logits[t], base.logits[t])
    # after the last block nothing mixes positions: the write reaches only its own
    np.testing.assert_array_equal(p.logits[t + 1:], base.logits[t + 1:])


def test_last_layer_lens_prediction_is_the_observed_change(model):
    """The positive control the experiment runs: at the last layer a lens is
    the head itself, so predicted and observed changes must coincide. A
    misaligned layer, position or target shows up here as an effect-sized gap."""
    last, site = N_BLOCKS, len(IDS) - 1
    base = patching.clean(model, IDS)
    value = patching.clean(model, OTHER_IDS).hidden[last, site]
    p = patching.patch(model, IDS, last, site, value, clean_run=base)
    states = np.stack([base.hidden[:, site], base.hidden[:, site]], axis=1)
    states[last, 1] = value
    for lens in (None, _lens(model)):
        z = patching.lens_logits(model, states, lens)[last]
        for y in (int(base.logits[0].argmax()), int(p.logits[0].argmax())):
            observed = patching.log_odds(p.logits[0], y) - patching.log_odds(base.logits[0], y)
            predicted = patching.log_odds(z[1], y) - patching.log_odds(z[0], y)
            assert abs(observed) > 1e-3          # not passing on a zero change
            assert abs(predicted - observed) < 1e-5


def test_provenance_names_the_ids_the_site_and_the_value(model):
    value = patching.clean(model, OTHER_IDS).hidden[1, 2]
    p = patching.patch(model, IDS, 1, -1, value)
    rec = p.provenance
    assert rec["input_ids"] == list(IDS)
    assert rec["site"] == {"layer": 1, "position": len(IDS) - 1}
    assert rec["readout"] == [len(IDS) - 1]
    assert rec["edit"] == SetState(1, value, token=len(IDS) - 1).record()
    assert rec["revision"] is None                 # built in-process: no hub commit
    assert (rec["dtype"], rec["device"], rec["family"]) == ("float32", "cpu",
                                                            type(model).__name__)
    assert rec["torch"] == torch.__version__
    json.dumps(rec)
    np.testing.assert_array_equal(p.state_after, value)


def test_refuses_what_would_compare_two_different_inputs(model):
    base = patching.clean(model, IDS)
    value = base.hidden[1, 0]
    with pytest.raises(ValueError, match="token ids"):
        patching.clean(model, PROMPT)
    with pytest.raises(ValueError, match="clean_run"):
        patching.patch(model, OTHER_IDS, 1, 0, value, clean_run=base)
    with pytest.raises(ValueError, match="clean_run"):
        patching.patch(model, IDS, 1, 0, value, readout=0, clean_run=base)
    with pytest.raises(ValueError, match="layer"):
        patching.patch(model, IDS, N_BLOCKS + 1, 0, value, clean_run=base)
    with pytest.raises(ValueError, match="positions"):
        patching.patch(model, IDS, 1, len(IDS), value, clean_run=base)
    with pytest.raises(ValueError, match="shape"):
        patching.patch(model, IDS, 1, 0, value[:-1], clean_run=base)


def test_log_odds_is_stable_where_a_top_token_is_confident():
    z = np.array([2.0, 1.0, 0.5, -1.0])
    p = np.exp(z) / np.exp(z).sum()
    for y in range(4):
        assert patching.log_odds(z, y) == pytest.approx(np.log(p[y]) - np.log1p(-p[y]))
    # p_0 = 1 - 2e-26 rounds to 1 in float64, so log(1 - p) would be -inf
    assert patching.log_odds(np.array([60.0, 0.0, 0.0]), 0) == pytest.approx(60 - np.log(2))


def test_load_model_refuses_a_revision_it_did_not_resolve(tmp_path):
    path = tiny.save_to(tmp_path / "m", n_layers=N_BLOCKS + 1)
    model, tokenizer = patching.load_model(path, None)
    assert not model.training and next(model.parameters()).dtype == torch.float32
    with pytest.raises(ValueError, match="resolved to commit"):
        patching.load_model(path, "607a30d783dfa663caf39e06633721c8d4cfcd7e")


@pytest.mark.network
def test_gpt2_self_patch_identity_and_last_layer_positive_control():
    """Real GPT-2 at the experiment's pinned revision, on two of its corpus
    windows: the deterministic identity at every layer, and the last-layer
    control for both the raw and the published tuned lens."""
    import tuned_lens
    from experiments import lens_patch_predictiveness as exp

    model, tokenizer = patching.load_model(exp.MODEL, exp.MODEL_REVISION)
    assert model.config._commit_hash == exp.MODEL_REVISION
    docs = [json.loads(line) for line in exp.CORPUS.read_text().splitlines()[:2]]
    a, b = (tokenizer(d["text"], add_special_tokens=False)["input_ids"][:exp.WINDOW_TOKENS]
            for d in docs)
    base, donor = patching.clean(model, a), patching.clean(model, b)
    n_layers, site = base.hidden.shape[0], len(a) - 1
    assert n_layers == 13
    for layer in range(n_layers):
        p = patching.patch(model, a, layer, site, base.hidden[layer, site], clean_run=base)
        np.testing.assert_array_equal(p.logits, base.logits)

    last = n_layers - 1
    p = patching.patch(model, a, last, site, donor.hidden[last, site], clean_run=base)
    states = np.stack([base.hidden[:, site], base.hidden[:, site]], axis=1)
    states[last, 1] = donor.hidden[last, site]
    for lens in (None, tuned_lens.from_hub(exp.MODEL)):
        z = patching.lens_logits(model, states, lens)[last]
        y = int(base.logits[0].argmax())
        observed = patching.log_odds(p.logits[0], y) - patching.log_odds(base.logits[0], y)
        predicted = patching.log_odds(z[1], y) - patching.log_odds(z[0], y)
        assert abs(predicted - observed) <= exp.POSITIVE_CONTROL_TOL
