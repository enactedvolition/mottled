"""The Jacobian-lens estimator (experiments/jacobian_lens.py) against the
derivative itself, and the lens it writes read back as head(norm(J h)).

Offline on a tiny Llama. Its J means nothing about any real model; what is
pinned is that the estimator computes the registered quantity and that the
tuned-lens format carries it without changing it.
"""
import hashlib
import json

import numpy as np
import pytest

import tiny
import tuned_lens

torch = pytest.importorskip("torch")

from capture import HookCapture, logit_lens  # noqa: E402
from experiments import jacobian_lens as jl  # noqa: E402

BLOCKS = 2
DIM = 16
SKIP = 1
PROMPT = "the quick brown fox jumps over the lazy dog"


def _model():
    return tiny.model(n_layers=BLOCKS, dim=DIM)


def _ids(text=PROMPT):
    return torch.tensor(tiny.tokenizer()(text, add_special_tokens=False)["input_ids"])


def _brute_force(model, ids, layer, skip):
    """mean over valid source p of sum over valid target p' of
    d h_final[p'] / d h_layer[p], from the full autograd Jacobian."""
    with torch.no_grad(), HookCapture(model) as cap:
        model(input_ids=ids[None])
    x0 = cap.states[layer].clone()

    def final(x):
        out = {}

        def grab(h):
            out["h"] = h
            return h
        with HookCapture(model, state_edits={layer: lambda h: x, BLOCKS: grab}):
            model(input_ids=ids[None], logits_to_keep=1)
        return out["h"]

    full = torch.autograd.functional.jacobian(final, x0)[0, :, :, 0]   # (T, D, T, D)
    pos = jl.valid_positions(len(ids), skip)
    return full[pos][:, :, pos].sum(0).mean(1)                        # (D, D)


@pytest.mark.parametrize("dim_batch", [1, 5])      # 5 leaves a remainder pass
def test_estimator_is_the_registered_derivative(dim_batch):
    model, ids = _model(), _ids()
    J = jl.jacobians(model, ids, skip_first=SKIP, dim_batch=dim_batch)
    assert J.shape == (BLOCKS, DIM, DIM)
    for layer in range(BLOCKS):
        torch.testing.assert_close(J[layer], _brute_force(model, ids, layer, SKIP),
                                   rtol=1e-4, atol=1e-5)


def test_fit_is_the_mean_over_sequences():
    model = _model()
    a, b = _ids(), _ids("the cat sat on the mat and the dog sat on the mat")
    torch.testing.assert_close(jl.fit(model, [a, b], skip_first=SKIP),
                               (jl.jacobians(model, a, SKIP) + jl.jacobians(model, b, SKIP)) / 2)


def test_fitting_one_layer_gives_that_layers_full_fit_J():
    model, ids = _model(), _ids()
    full = jl.jacobians(model, ids, skip_first=SKIP)
    torch.testing.assert_close(jl.jacobians(model, ids, skip_first=SKIP, layers=[1])[0], full[1],
                               rtol=1e-5, atol=1e-6)


def test_a_partial_lens_is_refused_as_a_readout(tmp_path):
    model, ids = _model(), _ids()
    jl.save(tmp_path / "lens", jl.jacobians(model, ids, SKIP, layers=[1]), {},
            layers=[1], n_blocks=BLOCKS)
    assert json.loads((tmp_path / "lens" / "config.json").read_text())["layers"] == [1]
    with pytest.raises(ValueError, match="partial lens"):
        tuned_lens.from_dir(tmp_path / "lens")


def test_too_short_a_sequence_is_refused():
    with pytest.raises(ValueError, match="no valid position"):
        jl.valid_positions(3, skip_first=2)


def test_the_written_lens_reads_as_head_of_J_h(tmp_path):
    model, ids = _model(), _ids()
    J = jl.jacobians(model, ids, skip_first=SKIP)
    jl.save(tmp_path / "lens", J, {"base_model_revision": "abc123"})
    lens = tuned_lens.from_dir(tmp_path / "lens")
    assert lens.record()["kind"] == "jacobian"

    prompt = "the capital of france is"
    raw = tiny.capture(prompt, n_layers=BLOCKS + 1, dim=DIM, keep_logits=True)
    read = tiny.capture(prompt, n_layers=BLOCKS + 1, dim=DIM, lens=lens)
    assert tuned_lens.readout_name(read.meta) == "Jacobian lens"
    with HookCapture(model) as cap:
        adapter = cap.adapter
    h = torch.from_numpy(raw.hidden)
    for layer in range(BLOCKS):
        by_hand = logit_lens((h[layer] @ J[layer].T)[None], adapter)[0].numpy()
        np.testing.assert_allclose(read.logits[layer].astype(np.float32), by_hand,
                                   rtol=1e-2, atol=1e-2)       # stored as float16
    # the target is the head's own input: J_final = I, no translator
    np.testing.assert_array_equal(read.logits[-1], raw.logits[-1])


def test_main_writes_a_lens_from_the_registered_train_documents(tmp_path, capsys):
    from experiments import introspection as ix

    out = tmp_path / "lens"
    assert jl.main(["--out", str(out), "--docs", "2", "--max-tokens", "12",
                    "--skip-first", str(SKIP)], loaded=(_model(), tiny.tokenizer())) == 0
    lens = tuned_lens.from_dir(out)
    assert lens.kind == "jacobian" and lens.n_layers == BLOCKS
    config = json.loads((out / "config.json").read_text())
    train, _ = ix.doc_split(ix.load_corpus())
    assert config["docs"] == [d["id"] for d in train[:2]]
    assert config["corpus_sha256"] == ix.CORPUS_SHA256
    assert capsys.readouterr().out.strip() == hashlib.sha256(
        (out / "params.pt").read_bytes()).hexdigest()

