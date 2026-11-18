"""Optional TransformerLens producer (models/hooked.py).

The StateTrajectory assembly is tested offline with numpy stand-ins; the real
TransformerLens path (TransformerBridge, TL >= 4) is a network test, skipped
unless transformer-lens is installed (loading a model downloads weights).
"""
import numpy as np
import pytest

from models.hooked import _to_trajectory


def test_to_trajectory_assembles_and_flows_through_the_stack():
    import tiny as synthetic
    from projection import project

    base = synthetic.capture("the capital of france is")
    L, T, D = base.hidden.shape
    V = 40
    rng = np.random.default_rng(0)
    lens = rng.normal(size=(L, T, V)).astype(np.float32)
    vocab = [f"tok{i}" for i in range(V)]
    emb = rng.normal(size=(V, D)).astype(np.float32)

    traj = _to_trajectory(base.hidden, base.tokens, lens, vocab, emb, top_k=3)
    traj.validate()
    assert traj.meta["backend"] == "transformer_lens"
    assert (traj.n_layers, traj.n_tokens, traj.dim) == (L, T, D)
    assert traj.entropy.shape == (L, T)
    assert len(traj.topk[0][0]) == 3
    assert traj.embedding_matrix.shape == (V, D)

    # the whole point: it is a first-class StateTrajectory, so it projects
    coords, _ = project(traj.hidden)
    assert coords.shape == (L, T, 2)


@pytest.mark.network
def test_from_hooked_transformer_gpt2():
    bridge = pytest.importorskip("transformer_lens.model_bridge")
    from models.hooked import from_hooked_transformer

    model = bridge.TransformerBridge.boot_transformers("gpt2")
    traj = from_hooked_transformer(model, "The capital of France is")
    traj.validate()
    assert traj.n_layers == model.cfg.n_layers + 1
    assert traj.logits is not None
    assert traj.meta["family"] == "TransformerBridge"
