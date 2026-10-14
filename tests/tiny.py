"""Tiny locally-built models: the test suite's stand-in for a real capture.

This replaces the old `models/synthetic` backend. That module generated
plausible-looking trajectories analytically, which made it fast and
torch-free but also meant most of the suite was testing the pipeline against
numbers no transformer produced. A two-layer Llama built here exercises the
same code paths the real backend does — hooks, the logit lens, the residual
decomposition, attention capture — and is still fast enough to run offline in
CI, because the cost of a transformer is its width and depth, not its
existence.

The tokenizer is word-level on purpose. Prompts in these tests are ordinary
sentences and many assertions are about *which token* is at a position, so
splitting on whitespace keeps a token meaning what a reader expects while
still being a real `PreTrainedTokenizerFast` that `capture()` drives exactly
as it drives GPT-2's.

Models are cached per shape: building one is cheap, but the suite asks for
the same handful of shapes hundreds of times.
"""
from __future__ import annotations

import functools

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

# The words every prompt in the suite is built from, plus the handful of
# continuations the generation tests expect to be able to produce.
CORPUS_WORDS = [
    "the", "capital", "of", "france", "is", "paris", "germany", "berlin",
    "italy", "rome", "spain", "madrid", "england", "london", "residual",
    "stream", "moves", "turns", "and", "settles", "into", "ground", "a",
    "new", "state", "equilibrium", "hidden", "states", "travel", "through",
    "layers", "pile", "up", "in", "basin", "mottled", "visualizes",
    "trajectories", "over", "density", "terrain", "model", "token", "layer",
    "prompt", "quick", "brown", "fox", "jumps", "lazy", "dog", "cat", "sat",
    "on", "mat", "hello", "world", "test", "one", "two", "three", "four",
]


@functools.lru_cache(maxsize=4)
def tokenizer():
    """A real word-level tokenizer over the suite's vocabulary."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"<unk>": 0}
    for word in CORPUS_WORDS:
        vocab.setdefault(word, len(vocab))

    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<unk>")
    return hf


def vocab_size() -> int:
    return len(tokenizer().get_vocab())


@functools.lru_cache(maxsize=8)
def model(n_layers: int = 12, dim: int = 32, n_heads: int = 4,
          n_kv_heads: int = 2, seed: int = 0):
    """A deterministic tiny Llama.

    `n_layers` is the number of blocks, so a capture of it has `n_layers + 1`
    entries — layer 0 being the embedding stream, as everywhere else.
    """
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(
        vocab_size=vocab_size(), hidden_size=dim, intermediate_size=2 * dim,
        num_hidden_layers=n_layers, num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads, max_position_embeddings=128,
        rms_norm_eps=1e-6, tie_word_embeddings=False,
    )
    m = transformers.LlamaForCausalLM(cfg).eval()
    # Random init leaves the norm weights near-identical, which can hide a bug
    # in the norm path; and it leaves the readout nearly uniform, which makes
    # entropy-collapse assertions meaningless. Both are nudged.
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "layernorm" in name or name.endswith("norm.weight"):
                p.add_(torch.randn_like(p) * 0.1)
        m.lm_head.weight.mul_(3.0)
    return m


def capture(prompt: str, n_layers: int = 13, dim: int = 32, seed: int = 0, **kw):
    """Drop-in replacement for the old `synthetic.capture`.

    `n_layers` counts captured layers (embeddings + blocks), matching what the
    synthetic backend reported, so call sites keep their expectations.
    """
    from capture import capture as _capture

    if n_layers < 2:
        # There is no such thing as a transformer with no blocks; a one-layer
        # run is the embedding stream on its own, so it is produced by
        # truncating a real capture. The analysis code must handle it (an
        # empty block range is where nan leaks in), which is the point.
        return _truncate(_capture(model(n_layers=1, dim=dim, seed=seed), prompt,
                                  tokenizer=tokenizer(), **kw), 1)

    return _capture(model(n_layers=n_layers - 1, dim=dim, seed=seed), prompt,
                    tokenizer=tokenizer(), **kw)


def _truncate(traj, n_layers: int):
    """Keep only the first `n_layers` captured layers."""
    import dataclasses

    fields = {"hidden": traj.hidden[:n_layers]}
    if traj.logits is not None:
        fields["logits"] = traj.logits[:n_layers]
    if traj.entropy is not None:
        fields["entropy"] = traj.entropy[:n_layers]
    if traj.topk is not None:
        fields["topk"] = traj.topk[:n_layers]
    if traj.components is not None:
        fields["components"] = {k: v[:n_layers - 1] for k, v in traj.components.items()}
    if traj.attention is not None:
        fields["attention"] = traj.attention[:n_layers - 1]
    return dataclasses.replace(traj, **fields)


def generate_and_capture(prompt: str, max_new_tokens: int = 4,
                         n_layers: int = 13, dim: int = 32,
                         model_seed: int = 0, **kw):
    """`seed` is deliberately NOT consumed here.

    It belongs to the decode loop's sampling, and swallowing it would make
    `seed=` silently mean "which weights" instead of "which sample" — the
    reproducibility tests would then pass while testing nothing.
    """
    from capture import generate_and_capture as _gen

    return _gen(model(n_layers=n_layers - 1, dim=dim, seed=model_seed), prompt,
                max_new_tokens=max_new_tokens, tokenizer=tokenizer(), **kw)


def config(**kw):
    """A MarbleConfig wired to a tiny model, for pipeline-level tests."""
    from config import MarbleConfig

    kw.setdefault("use_cache", False)
    return MarbleConfig(model="tiny", **kw)


def loaded(n_layers: int = 13, dim: int = 32, seed: int = 0) -> dict:
    """The `loaded` mapping pipeline entry points take, keyed as "tiny"."""
    return {"tiny": (model(n_layers=n_layers - 1, dim=dim, seed=seed), tokenizer())}


def mt(n_layers: int = 13, dim: int = 32, seed: int = 0) -> dict:
    """`model=`/`tokenizer=` kwargs for the pipeline entry points.

    Those resolve `cfg.model` through the Hub when no instance is passed, and
    "tiny" is not a hub name — so pipeline-level tests hand the objects over
    directly, exactly as the explorer does with its cached model.
    """
    return {"model": model(n_layers=n_layers - 1, dim=dim, seed=seed),
            "tokenizer": tokenizer()}


def save_to(path, n_layers: int = 4, dim: int = 32, seed: int = 0) -> str:
    """Write a tiny model + tokenizer to `path` as a HF checkpoint directory,
    so code that loads by name (the CLI's `--model DIR`) runs it offline.
    `n_layers` counts captured layers, as in `capture`."""
    from pathlib import Path

    Path(path).mkdir(parents=True, exist_ok=True)
    model(n_layers=n_layers - 1, dim=dim, seed=seed).save_pretrained(path)
    tokenizer().save_pretrained(path)
    return str(path)
