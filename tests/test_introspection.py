"""Introspection experiment (experiments/introspection.py): where a concept is
written, that an unedited pass is the model's own, and that the scoring and
statistics say what the registered plan says.

Offline on a tiny Llama with a word-level tokenizer that knows the concept
words and answer labels. Its numbers mean nothing; these tests pin the
machinery.
"""
import json

import numpy as np
import pytest

import tiny

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from capture import HookCapture  # noqa: E402
from experiments import introspection as ix  # noqa: E402

FILLER = [w for w in tiny.CORPUS_WORDS if w not in ix.CONCEPTS]


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    words = ["<unk>", *tiny.CORPUS_WORDS, *ix.CONCEPTS, *ix.LETTERS, *ix.NUMBERS,
             "[", "]", ")", ".", "?", "Tell", "me", "about", "Say", "word"]
    vocab = {}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>",
                                                pad_token="<unk>")


def _model(n_layers=3, seed=0):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(vocab_size=len(_tokenizer().get_vocab()), hidden_size=32,
                                   intermediate_size=64, num_hidden_layers=n_layers,
                                   num_attention_heads=4, num_key_value_heads=2,
                                   max_position_embeddings=512, tie_word_embeddings=False)
    return transformers.LlamaForCausalLM(cfg).eval()


def _doc(i):
    rng = np.random.default_rng(i)
    return {"id": f"d{i:02d}", "text": " ".join(rng.choice(FILLER, 110))}


def _corpus(tmp_path, n=6):
    p = tmp_path / "corpus.jsonl"
    p.write_text("\n".join(json.dumps(_doc(i)) for i in range(n)))
    return p


def test_concept_split_is_fixed_and_disjoint():
    train, test = ix.concept_split()
    assert (len(train), len(test)) == (30, 20)
    assert not set(train) & set(test)
    assert ix.concept_split() == (train, test)


def test_items_never_list_a_word_the_passage_contains():
    doc = {"id": "x", "text": " ".join(["the"] * 30 + ["apple"] + ["dog"] * 80)}
    items, skipped = ix.make_items("identify", [doc], ix.CONCEPTS)
    assert skipped == 1 and all(i.concept != "apple" for i in items)
    for i in items:
        assert "apple" not in i.options
        assert i.options[i.target] == i.concept
        assert len(set(i.options)) == 4
    loc, _ = ix.make_items("localize", [doc], ix.CONCEPTS)
    assert all(i.inject_parts == (i.target,) and 0 <= i.target < ix.N_PARTS for i in loc)


def test_part_masks_cover_exactly_each_part():
    tok = _tokenizer()
    item = ix.make_items("identify", [_doc(1)], ix.CONCEPTS)[0][0]
    ids, masks = ix.render(tok, item, ix.LETTERS)
    assert not (masks.sum(0) > 1).any()
    for i, part in enumerate(item.parts):
        assert tok.decode(ids[masks[i]]).split() == part.split()


def test_unedited_pass_is_the_models_own():
    tok, model = _tokenizer(), _model()
    item = ix.make_items("identify", [_doc(2)], ix.CONCEPTS)[0][0]
    ids, masks = ix.render(tok, item, ix.LETTERS)
    answer = ix.label_ids(tok, ix.LETTERS)
    with torch.no_grad():
        # same head call shape as answer_logits: a (1, D) and a (1, T, D) head
        # call round differently in float32, which is not an edit
        plain = model(input_ids=ids[None], logits_to_keep=1).logits[0, -1, answer]
        none = ix.answer_logits(model, ids, masks.any(0), None, 2, answer, "cpu")
        zero = ix.answer_logits(model, ids, masks.any(0), torch.zeros(32), 2, answer, "cpu")
    assert torch.equal(plain, none) and torch.equal(plain, zero)


def test_write_lands_only_from_the_injected_part_on():
    tok, model = _tokenizer(), _model()
    item = ix.make_items("localize", [_doc(3)], ix.CONCEPTS)[0][0]
    ids, masks = ix.render(tok, item, ix.NUMBERS)
    mask = masks[3]
    first = int(mask.nonzero()[0])
    v = torch.randn(32) * 5
    edit = {2: lambda h: h + mask[None, :, None].to(h.dtype) * v}
    with torch.no_grad(), HookCapture(model) as clean:
        model(input_ids=ids[None])
    with torch.no_grad(), HookCapture(model, state_edits=edit) as hit:
        model(input_ids=ids[None])
    for layer in range(len(clean.states)):
        assert torch.equal(clean.states[layer][0, :first], hit.states[layer][0, :first])
    assert not torch.equal(clean.states[3][0, -1], hit.states[3][0, -1])


def test_random_control_matches_the_concept_norm():
    d = {"apple": torch.nn.functional.normalize(torch.randn(32), dim=0)}
    item = ix.Item("localize", "d", "apple", ("a",) * 4, (), 0, (0,))
    r = ix.condition_vector("random", item, d, 7.0)
    assert torch.allclose(r.norm(), torch.tensor(7.0))
    assert torch.equal(r, ix.condition_vector("random", item, d, 7.0))
    assert ix.condition_vector("none", item, d, 7.0) is None


def test_lora_training_moves_only_adapters():
    pytest.importorskip("peft")
    tok, model = _tokenizer(), _model()
    base_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    train_c, _ = ix.concept_split()
    items = ix.make_items("identify", [_doc(4)], train_c[:4])[0]
    dirs = {c: torch.nn.functional.normalize(torch.randn(32), dim=0) for c in ix.CONCEPTS}
    trained, losses = ix.train_lora(model, tok, items, dirs, 3.0, 2, "cpu", steps_per_update=2)
    assert losses and all(np.isfinite(losses))
    named = dict(trained.named_parameters())
    lora = [n for n in named if "lora_" in n]
    assert lora
    for n, before in base_before.items():
        match = [k for k in named if k.endswith(n.replace("weight", "base_layer.weight")) or k == n]
        assert torch.equal(named[match[0]], before), n


def test_contrasts_and_verdict():
    concepts, docs = ["a", "b"], ["x", "y"]
    recs = []
    for stage in ("untrained", "trained"):
        for task, cond, right in (("identify", "concept", stage == "trained"),
                                  ("identify", "none", False), ("prime", "concept", False),
                                  ("prime", "none", False)):
            for c in concepts:
                for d in docs:
                    recs.append({"stage": stage, "task": task, "cond": cond, "concept": c,
                                 "doc": d, "target": 0, "pred": 0 if right else 1})
    s = ix.summarize(recs, concepts, docs, B=50)
    assert s["accuracy"]["trained/identify/concept"]["point"] == 1.0
    assert s["contrasts"]["trained: identify concept - prime concept"]["point"] == 1.0
    assert s["contrasts"]["training gain beyond priming"]["point"] == 1.0
    assert s["H1"] == "supported"


def test_end_to_end_on_a_tiny_model(tmp_path):
    pytest.importorskip("peft")
    tok, model = _tokenizer(), _model()
    out, summ = tmp_path / "r.json", tmp_path / "s.json"
    s = ix.main(["--corpus", str(_corpus(tmp_path)), "--docs", "2", "--bootstrap", "20",
                 "--out", str(out), "--summary", str(summ)],
                loaded=(model, tok, ix.LETTERS, ix.NUMBERS))
    r = json.loads(out.read_text())
    stages = {x["stage"] for x in r["records"]}
    assert stages == {"untrained", "trained"}
    assert s["H1"] in {"supported", "not supported"}
    assert json.loads(summ.read_text())["config"]["strength"] in ix.STRENGTHS
