"""Hidden tensors captured correctly + shape consistency (spec tests 1-2)."""

import zlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from capture import HookCapture, capture, logit_lens  # noqa: E402
from models.families import resolve_family  # noqa: E402
import tiny as synthetic

VOCAB_SIZE = 128
PROMPT = "the capital of france is"


class DummyTokenizer:
    """Deterministic word-level tokenizer for locally-built test models."""

    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self._names = {}

    def _id(self, word):
        i = zlib.crc32(word.encode()) % (self.vocab_size - 1) + 1
        self._names[i] = word
        return i

    def __call__(self, text, return_tensors="pt"):
        ids = [self._id(w) for w in text.split()]
        return {"input_ids": torch.tensor([ids])}

    def convert_ids_to_tokens(self, ids):
        return [self._names.get(int(i), f"<{int(i)}>") for i in ids]


@pytest.fixture(scope="module")
def tiny_llama():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return transformers.LlamaForCausalLM(cfg).eval()


def test_hooks_match_output_hidden_states(tiny_llama):
    """Hook-captured residual stream equals HF's own hidden_states."""
    tok = DummyTokenizer()
    ids = tok(PROMPT)["input_ids"]
    with HookCapture(tiny_llama) as cap, torch.no_grad():
        out = tiny_llama(ids, output_hidden_states=True)
    hidden = cap.stacked()  # (L+1, T, D)

    ref = torch.stack([h[0] for h in out.hidden_states]).float()
    n_blocks = len(cap.adapter.blocks)
    # HF's tuple is (embeddings, block1..block_{N-1}, final_norm(block_N)):
    # every pre-norm entry must match our capture exactly...
    assert torch.allclose(hidden[:n_blocks], ref[:n_blocks], atol=1e-5)
    # ...and our raw final-block output must match after applying final norm.
    with torch.no_grad():
        final = cap.adapter.final_norm(hidden[-1])
    assert torch.allclose(final, ref[-1], atol=1e-5)


def test_capture_shapes_and_stats(tiny_llama):
    traj = capture(tiny_llama, PROMPT, tokenizer=DummyTokenizer(), top_k=3)
    traj.validate()
    L, T, D = traj.hidden.shape
    assert (L, D) == (4, 32) and T == len(PROMPT.split())
    assert traj.logits.shape == (L, T, VOCAB_SIZE)
    assert traj.entropy.shape == (L, T)
    assert np.isfinite(traj.entropy).all()
    assert len(traj.topk) == L and len(traj.topk[0]) == T and len(traj.topk[0][0]) == 3
    assert all(0.0 <= p <= 1.0 for _, p in traj.topk[0][0])
    assert traj.embedding_matrix.shape == (VOCAB_SIZE, D)
    assert traj.tokens[1] == "capital"


def test_logit_lens_final_layer_matches_model(tiny_llama):
    """Logit lens at the last layer must reproduce the model's real logits."""
    tok = DummyTokenizer()
    ids = tok(PROMPT)["input_ids"]
    with HookCapture(tiny_llama) as cap, torch.no_grad():
        out = tiny_llama(ids)
    lens = logit_lens(cap.stacked(), cap.adapter)
    assert torch.allclose(lens[-1], out.logits[0].float(), atol=1e-4)


def test_entropy_topk_is_the_whole_block_softmax_and_a_stable_sort():
    """`_entropy_topk` works a layer at a time and partitions rather than
    sorting the vocabulary. It has to return exactly what the whole-block
    float64 softmax and a full stable sort return, ties included."""
    from capture import _entropy_topk

    rng = np.random.default_rng(0)
    logits = rng.standard_normal((3, 7, 50)).astype(np.float32) * 4
    logits[1, 2, [3, 9, 17, 30]] = 20.0               # ties inside the top-k
    logits[2, 4] = np.round(logits[2, 4])             # ties all along a row
    vocab = [f"t{i}" for i in range(50)]

    x = logits.astype(np.float64)
    x -= x.max(axis=-1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=-1, keepdims=True)
    want = (-(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=-1)).astype(np.float32)
    order = np.argsort(-p, axis=-1, kind="stable")

    for k in (0, 1, 3, 5, 50, 80):
        entropy, topk = _entropy_topk(logits, vocab, k)
        np.testing.assert_array_equal(entropy, want)
        assert topk == [[[(vocab[j], float(p[l, t, j])) for j in order[l, t, :k]]
                         for t in range(7)] for l in range(3)]


def test_tiny_backend_shapes():
    traj = synthetic.capture(PROMPT, top_k=5)
    traj.validate()
    assert traj.n_tokens == len(PROMPT.split())
    assert traj.n_layers > 1
    assert traj.entropy is not None and np.isfinite(traj.entropy).all()
    # deterministic per prompt
    again = synthetic.capture(PROMPT, top_k=5)
    assert np.array_equal(traj.hidden, again.hidden)


def test_gpt2_style_layout_resolves():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=VOCAB_SIZE, n_embd=32, n_layer=2, n_head=4, n_positions=64)
    model = transformers.GPT2LMHeadModel(cfg).eval()
    adapter = resolve_family(model)
    assert adapter.n_layers == 2
    traj = capture(model, PROMPT, tokenizer=DummyTokenizer(), top_k=2)
    assert traj.hidden.shape[0] == 3


def test_mamba_layout_resolves():
    """Mamba is a state-space model, not a transformer: the producer
    abstraction must hold anyway (block capture + logit lens unchanged)."""
    torch.manual_seed(0)
    cfg = transformers.MambaConfig(vocab_size=VOCAB_SIZE, hidden_size=32,
                                   state_size=8, num_hidden_layers=2, expand=2)
    model = transformers.MambaForCausalLM(cfg).eval()
    adapter = resolve_family(model)
    assert adapter.n_layers == 2
    traj = capture(model, PROMPT, tokenizer=DummyTokenizer(), top_k=3)
    traj.validate()
    assert traj.hidden.shape == (3, len(PROMPT.split()), 32)
    assert np.isfinite(traj.entropy).all()
    # no attention, no attn/mlp split: those captures must refuse, not lie
    with pytest.raises(ValueError):
        capture(model, PROMPT, tokenizer=DummyTokenizer(), capture_components=True)
    with pytest.raises(ValueError):
        capture(model, PROMPT, tokenizer=DummyTokenizer(), capture_attention=True)


def test_load_by_repo_id_records_the_commit_it_loaded(tmp_path, monkeypatch):
    # transformers >= 5.20 no longer sets config._commit_hash, so every scene
    # recorded revision: null until load_model read it from the hub cache. A
    # fake cache holding one snapshot under refs/main stands in for the hub.
    import huggingface_hub.constants as hub_constants

    from capture import load_model

    sha = "0123456789abcdef0123456789abcdef01234567"
    repo = tmp_path / "models--mottled-test--tiny"
    synthetic.save_to(repo / "snapshots" / sha)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text(sha)
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)

    model, _ = load_model("mottled-test/tiny")
    assert model.config._commit_hash == sha

    # a local checkpoint under the same name is what loads; it is not that commit
    local = tmp_path / "cwd"
    synthetic.save_to(local / "mottled-test" / "tiny")
    monkeypatch.chdir(local)
    model, _ = load_model("mottled-test/tiny")
    assert getattr(model.config, "_commit_hash", None) is None
