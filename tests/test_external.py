"""Ingesting states captured outside Mottled (models/external.py, and
models/hooked.from_cache for a TransformerLens cache).

The contract worth pinning is not that these produce *a* trajectory but that
they produce the *same* one a native capture does — an adapter that near-misses
the readout gives every downstream basin and top-k a different question to
answer, which is the failure mode docs/validity.md exists to prevent.
"""
import numpy as np
import pytest

import tiny as synthetic
from models.external import from_hidden_states, stack_layers

torch = pytest.importorskip("torch")

PROMPT = "the capital of france is"


@pytest.fixture(scope="module")
def pair():
    """A tiny model + tokenizer and the native capture of one prompt."""
    model, tokenizer = synthetic.model(n_layers=3), synthetic.tokenizer()
    traj = synthetic.capture(PROMPT, n_layers=4)
    return model, tokenizer, traj


# ------------------------------------------------------------------ the shim
def test_stack_accepts_what_a_tracing_library_hands_back():
    rng = np.random.default_rng(0)
    want = rng.normal(size=(4, 6, 8)).astype(np.float32)

    assert np.array_equal(stack_layers(want), want)
    assert np.array_equal(stack_layers(torch.from_numpy(want)), want)
    assert np.array_equal(stack_layers(want[:, None]), want)       # (L, 1, T, D)
    assert np.array_equal(stack_layers(list(want)), want)          # per-layer (T, D)
    assert np.array_equal(                                          # per-layer (1, T, D)
        stack_layers([torch.from_numpy(l)[None] for l in want]), want)


def test_stack_unwraps_a_saved_proxy():
    """NNsight's .save() hands back a proxy whose .value is the tensor."""
    class Saved:
        def __init__(self, value):
            self.value = value

    rng = np.random.default_rng(1)
    want = rng.normal(size=(3, 5, 8)).astype(np.float32)
    assert np.array_equal(stack_layers([Saved(torch.from_numpy(l)) for l in want]),
                          want)


def test_stack_converts_bfloat16():
    """bfloat16 has no numpy dtype, and a bf16 capture is the common case."""
    t = torch.randn(3, 5, 8).to(torch.bfloat16)
    out = stack_layers(t)
    assert out.dtype == np.float32
    assert np.allclose(out, t.float().numpy())


def test_stack_rejects_ragged_and_misshapen_input():
    with pytest.raises(ValueError, match="differing shapes"):
        stack_layers([np.zeros((5, 8), np.float32), np.zeros((4, 8), np.float32)])
    with pytest.raises(ValueError, match="expected"):
        stack_layers(np.zeros((5, 8), np.float32))
    with pytest.raises(ValueError, match="no layers"):
        stack_layers([])


# --------------------------------------------------------------- conformance
def test_external_states_reproduce_the_native_capture(pair):
    """The whole point: hand back the states capture() recorded and every
    readout — logits, entropy, top-k, neighbors — comes out identical."""
    model, tokenizer, native = pair

    traj = from_hidden_states(native.hidden, native.tokens, model, tokenizer,
                              prompt=PROMPT)
    assert traj.hidden.shape == native.hidden.shape
    assert traj.tokens == native.tokens
    assert traj.vocab == native.vocab
    assert np.allclose(traj.logits.astype(np.float32),
                       native.logits.astype(np.float32), atol=1e-3)
    assert np.allclose(traj.entropy, native.entropy, atol=1e-4)
    assert [t for t, _ in traj.topk[-1][-1]] == [t for t, _ in native.topk[-1][-1]]
    assert np.array_equal(traj.embedding_matrix, native.embedding_matrix)


def test_external_trajectory_flows_through_the_stack(pair):
    """It is a first-class StateTrajectory, not a lookalike."""
    from projection import project
    from statefile import save, load

    model, tokenizer, native = pair
    traj = from_hidden_states(list(native.hidden), native.tokens, model, tokenizer)
    coords, _ = project(traj.hidden)
    assert coords.shape == (traj.n_layers, traj.n_tokens, 2)

    import io
    buf = io.BytesIO()
    save(traj, buf)
    buf.seek(0)
    assert load(buf).n_layers == traj.n_layers


def test_token_ids_are_accepted(pair):
    model, tokenizer, native = pair
    ids = tokenizer(PROMPT)["input_ids"]
    traj = from_hidden_states(native.hidden, ids, model, tokenizer)
    assert traj.tokens == native.tokens


def test_token_count_mismatch_is_named(pair):
    """The real failure: a BOS in one list and not the other."""
    model, tokenizer, native = pair
    with pytest.raises(ValueError, match="captured positions"):
        from_hidden_states(native.hidden, native.tokens[:-1], model, tokenizer)


def test_meta_carries_what_provenance_records(pair):
    """An external capture must fill the same meta keys the native one does,
    or a scene built from it records nulls where the record wants facts."""
    import provenance as P
    from config import MarbleConfig

    model, tokenizer, native = pair
    traj = from_hidden_states(native.hidden, native.tokens, model, tokenizer,
                              backend="nnsight")
    assert traj.meta["backend"] == "nnsight"
    assert traj.meta["device"] == native.meta["device"]
    assert traj.meta["dtype"] == native.meta["dtype"]
    assert traj.meta["family"] == "LlamaForCausalLM"

    recorded, = P.record(MarbleConfig(model="tiny"), trajs=[traj])["models"]
    assert recorded["backend"] == "nnsight"
    assert recorded["dtype"] == native.meta["dtype"]


# ------------------------------------------------- TransformerLens cache path
def test_residual_stack_follows_mottleds_layer_convention():
    """`ActivationCache` only has to index like a dict, so the extraction is
    testable without the library: layer 0 = resid_pre 0, then each resid_post."""
    from models.hooked import residual_stack

    rng = np.random.default_rng(2)
    blocks = [torch.from_numpy(rng.normal(size=(1, 5, 8)).astype(np.float32))
              for _ in range(4)]
    pre = torch.from_numpy(rng.normal(size=(1, 5, 8)).astype(np.float32))
    cache = {("resid_pre", 0): pre,
             **{("resid_post", i): b for i, b in enumerate(blocks)}}

    stack = residual_stack(cache, 4)
    assert stack.shape == (5, 5, 8)                      # n_layers + 1
    assert torch.equal(stack[0], pre[0])
    assert torch.equal(stack[-1], blocks[-1][0])


@pytest.mark.network
def test_from_cache_matches_from_hooked_transformer():
    bridge = pytest.importorskip("transformer_lens.model_bridge")
    from models.hooked import from_cache, from_hooked_transformer

    model = bridge.TransformerBridge.boot_transformers("gpt2")
    prompt = "The capital of France is"
    _, cache = model.run_with_cache(model.to_tokens(prompt))

    reused = from_cache(cache, model, prompt)
    fresh = from_hooked_transformer(model, prompt)
    assert np.allclose(reused.hidden, fresh.hidden)
    assert reused.tokens == fresh.tokens
    assert reused.n_layers == model.cfg.n_layers + 1
