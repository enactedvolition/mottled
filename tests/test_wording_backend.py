"""Offline mechanics tests; random tiny weights establish no semantic claim."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from models.wording import WordingBackend  # noqa: E402
import tiny  # noqa: E402


@pytest.fixture(scope="module")
def backend():
    return WordingBackend(tiny.model(n_layers=2, seed=81), tiny.tokenizer())


PROMPT = "the capital of france is"
CANDIDATES = {"a": " paris and rome", "b": " berlin"}


def hook_counts(model):
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]


def test_multitoken_scores_equal_final_forward_probabilities(backend):
    records = backend.score(PROMPT, CANDIDATES)
    for label, candidate in CANDIDATES.items():
        record = records[label]
        full = backend.tokenizer(PROMPT + candidate, add_special_tokens=False)["input_ids"]
        prefix = backend.tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            logits = backend.model(input_ids=torch.tensor([full]), use_cache=False).logits
        expected = [float(logits[0, i - 1].double().log_softmax(0)[full[i]])
                    for i in range(len(prefix), len(full))]
        assert record["logprob"] == pytest.approx(sum(expected), abs=1e-10)
        assert record["token_logprobs"] == pytest.approx(expected, abs=1e-10)
        assert record["token_ids"] == full[len(prefix):]
        assert record["input_ids"] == full
        assert record["prompt_token_ids"] == prefix


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_self_patch_is_noop_and_releases_hooks(backend, layer):
    before = hook_counts(backend.model)
    baseline = backend.score(PROMPT, CANDIDATES)
    state = backend.state(PROMPT, layer)
    assert state.shape == (32,)
    patched = backend.score(PROMPT, CANDIDATES, {"layer": layer, "value": state})
    for label in CANDIDATES:
        assert patched[label]["token_logprobs"] == pytest.approx(
            baseline[label]["token_logprobs"], abs=1e-7)
    assert hook_counts(backend.model) == before


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_matched_self_control_is_exact_and_uses_full_input(backend, layer):
    baseline = backend.score(PROMPT, CANDIDATES)
    before = hook_counts(backend.model)
    observed_ids = []

    def observe(module, args, kwargs):
        observed_ids.append(kwargs["input_ids"][0].tolist())

    handle = backend.model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        patched = backend.self_score(PROMPT, CANDIDATES, layer)
    finally:
        handle.remove()
    assert observed_ids == [baseline[label]["input_ids"]
                            for label in CANDIDATES for _ in range(2)]
    for label in CANDIDATES:
        assert patched[label]["logprob"] == baseline[label]["logprob"]
        assert patched[label]["token_logprobs"] == baseline[label]["token_logprobs"]
        assert patched[label]["self_patch_capture_policy"] == "matched_full_input_ids"
    assert hook_counts(backend.model) == before


def test_transplant_changes_downstream_scores(backend):
    # Change the donor's last token so the layer-0 state differs as well.
    donor = backend.state("the capital of germany", 0)
    baseline = backend.score(PROMPT, CANDIDATES)
    patched = backend.score(PROMPT, CANDIDATES, {"layer": 0, "value": donor})
    assert abs(patched["a"]["logprob"] - baseline["a"]["logprob"]) > 1e-5


def test_final_layer_patch_hits_last_prompt_not_last_completion(backend):
    donor_prompt = "the capital of germany"
    donor = backend.state(donor_prompt, 2)
    baseline = backend.score(PROMPT, CANDIDATES)
    patched = backend.score(PROMPT, CANDIDATES, {"layer": 2, "value": donor})
    donor_scores = backend.score(donor_prompt, CANDIDATES)
    for label in CANDIDATES:
        # A final-block patch changes the logits predicting the first
        # continuation token. Later forced-token positions cannot receive
        # this edit: there are no downstream blocks left to transmit it.
        assert patched[label]["token_logprobs"][0] == pytest.approx(
            donor_scores[label]["token_logprobs"][0], abs=1e-7)
        assert abs(patched[label]["token_logprobs"][0]
                   - baseline[label]["token_logprobs"][0]) > 1e-5
        assert patched[label]["token_logprobs"][1:] == baseline[label]["token_logprobs"][1:]


@pytest.mark.parametrize("prompt,candidates,match", [
    ("", CANDIDATES, "prompt"),
    (PROMPT, {}, "nonempty"),
    (PROMPT, {"a": ""}, "nonempty"),
    (PROMPT, {"a": " "}, "no additional tokens"),
    ("par", {"a": "is"}, "retokenizes"),
    (PROMPT + " ", {"a": "paris", "b": " paris"}, "identical token ids"),
    (PROMPT, {"": " paris"}, "labels"),
])
def test_invalid_candidates_fail_before_inference(backend, monkeypatch, prompt, candidates, match):
    def should_not_run(*args, **kwargs):
        pytest.fail("invalid candidates should be rejected before model inference")

    monkeypatch.setattr(backend.model, "forward", should_not_run)
    with pytest.raises(ValueError, match=match):
        backend.score(prompt, candidates)


@pytest.mark.parametrize("patch", [
    {"layer": -1, "value": np.zeros(32)},
    {"layer": 3, "value": np.zeros(32)},
    {"layer": True, "value": np.zeros(32)},
    {"layer": 1.0, "value": np.zeros(32)},
    {"layer": 1, "value": np.zeros((1, 32))},
    {"layer": 1, "value": np.zeros(31)},
    {"layer": 1, "value": np.full(32, np.nan)},
    {"layer": 1, "value": np.full(32, 1e300)},
    {"layer": 1, "value": np.full(32, 1j)},
    {"layer": 1, "value": np.zeros(32), "position": -1},
])
def test_invalid_patches_fail_and_release_hooks(backend, patch):
    before = hook_counts(backend.model)
    with pytest.raises(ValueError):
        backend.score(PROMPT, CANDIDATES, patch)
    assert hook_counts(backend.model) == before


def test_hook_cleanup_when_forward_fails(backend, monkeypatch):
    before = hook_counts(backend.model)

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate model failure")

    monkeypatch.setattr(backend.model, "forward", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        backend.score(PROMPT, CANDIDATES, {"layer": 1, "value": np.zeros(32)})
    with pytest.raises(RuntimeError, match="deliberate"):
        backend.state(PROMPT, 1)
    with pytest.raises(RuntimeError, match="deliberate"):
        backend.self_score(PROMPT, CANDIDATES, 1)
    assert hook_counts(backend.model) == before


def test_nonfinite_logits_rejected(backend):
    handle = backend.model.lm_head.register_forward_hook(
        lambda module, args, output: output * float("nan"))
    try:
        with pytest.raises(ValueError, match="nonfinite"):
            backend.score(PROMPT, CANDIDATES)
    finally:
        handle.remove()


def test_inference_contract_and_metadata(backend):
    observed = []

    def observe(module, args, kwargs):
        observed.append((module.training, torch.is_grad_enabled(), kwargs["use_cache"]))

    backend.model.train()
    handle = backend.model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        backend.score(PROMPT, CANDIDATES)
        backend.state(PROMPT, 1)
    finally:
        handle.remove()
    assert observed == [(False, False, False)] * 3
    meta = backend.metadata()
    assert meta["n_residual_sites"] == 3
    assert meta["model_revision"] is None
    assert meta["model_weights_sha256"] is None
    assert len(meta["tokenizer_sha256"]) == 64
    assert meta["inference"]["patch_position"] == "last_prompt_token"
    assert meta["inference"]["append_eos"] is False
    assert meta["tokenizer_sha256"] == backend.metadata()["tokenizer_sha256"]
