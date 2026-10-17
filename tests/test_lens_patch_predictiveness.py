"""The lens patch-predictiveness experiment, end to end on a tiny model.

Machinery only: a tiny Llama, a word-level corpus and a tuned lens with
random translators. None of the numbers produced here say anything about
lenses or GPT-2. What is tested is that the registered design is what runs:
the split and pairing, the controls and the exit code they drive, the
summary's schema and determinism, and the intervals' ordering. The real run
is `main()` without the `loaded` hook used here.
"""
import hashlib
import json
import re

import numpy as np
import pytest

import tiny

torch = pytest.importorskip("torch")

from experiments import lens_patch_predictiveness as exp  # noqa: E402
from tuned_lens import TunedLens  # noqa: E402

N_BLOCKS = 3
N_DOCS = 8
VERDICTS = {"tuned better", "raw better", "not distinguishable"}


def _corpus(path):
    rng = np.random.default_rng(0)
    n_words = exp.WINDOWS_PER_DOC * exp.WINDOW_TOKENS + 5
    path.write_text("".join(
        json.dumps({"id": f"doc{d:02d}", "text": " ".join(rng.choice(tiny.CORPUS_WORDS, n_words))})
        + "\n" for d in range(N_DOCS)))
    return path


def _loaded():
    model = tiny.model(n_layers=N_BLOCKS)
    dim = model.config.hidden_size
    rng = np.random.default_rng(1)
    lens = TunedLens(weight=(0.1 * rng.standard_normal((N_BLOCKS, dim, dim))).astype(np.float32),
                     bias=(0.1 * rng.standard_normal((N_BLOCKS, dim))).astype(np.float32),
                     source="synthetic: random translators, machinery test only",
                     revision=None, base_model_revision=None)
    return model, tiny.tokenizer(), lens


def _main(out_dir, corpus, *extra):
    return exp.main(["--out", str(out_dir / "results.json"),
                     "--summary", str(out_dir / "summary.json"), "--corpus", str(corpus),
                     # one thread: a tiny model's ops are too small to split, and
                     # on a busy machine the extra threads only wait
                     "--threads", "1", *extra], loaded=_loaded())


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Two runs of the same experiment into two directories."""
    root = tmp_path_factory.mktemp("lens-patch")
    corpus = _corpus(root / "corpus.jsonl")
    out = []
    for name in ("a", "b"):
        (root / name).mkdir()
        out.append((_main(root / name, corpus), root / name))
    return corpus, out


def _json(path):
    return json.loads(path.read_text())


def test_controls_pass_and_the_run_exits_zero(runs):
    _, out = runs
    for code, d in out:
        assert code == 0
        ctl = _json(d / "summary.json")["controls"]
        assert ctl["passed"]
        assert ctl["self_max_abs_logit_diff"] == 0.0
        assert ctl["self_max_abs_delta_obs"] == 0.0
        assert ctl["positive_control_layer"] == N_BLOCKS
        assert ctl["positive_control_max_abs_err"] <= exp.POSITIVE_CONTROL_TOL


def test_two_runs_write_the_same_summary_bytes(runs):
    _, ((_, a), (_, b)) = runs
    assert (a / "summary.json").read_bytes() == (b / "summary.json").read_bytes()
    ra, rb = _json(a / "results.json"), _json(b / "results.json")
    ra.pop("timing"), rb.pop("timing")
    assert ra == rb


def test_summary_schema(runs):
    corpus, ((_, d), _) = runs
    s = _json(d / "summary.json")
    assert set(s) == {"schema", "config", "provenance", "controls", "counts", "primary",
                      "cells"}
    assert s["schema"] == exp.SUMMARY_SCHEMA
    n_layers = N_BLOCKS + 1
    assert s["config"]["primary_layers"] == list(range(1, n_layers - 1))
    assert s["counts"] == {"documents": N_DOCS, "items": 2 * N_DOCS, "train_items": N_DOCS,
                           "test_items": N_DOCS, "test_documents": N_DOCS // 2,
                           "items_with_equal_targets": s["counts"]["items_with_equal_targets"]}
    prov = s["provenance"]
    assert prov["corpus"]["sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert prov["lens"]["source"].startswith("synthetic")
    assert prov["model"]["revision"] is None and prov["model"]["dtype"] == "float32"
    assert prov["environment"]["deterministic_algorithms"] is True
    assert set(s["cells"]) == set(exp.SCORED)
    for by_target in s["cells"].values():
        assert set(by_target) == set(exp.TARGETS)
        for block in by_target.values():
            assert block["layers"] == list(range(n_layers))
            for lens in exp.LENSES:
                assert all(len(v) == n_layers for v in block[lens].values())
            assert all(len(v) == n_layers for v in block["paired"].values())
    primary = s["primary"]
    assert primary["layers"] == list(range(1, n_layers - 1))
    assert set(primary["verdict"]) <= VERDICTS and primary["pooled"]["verdict"] in VERDICTS
    # the last layer is the positive control: both lenses are the head there
    for block in (b for t in s["cells"].values() for b in t.values()):
        assert block["raw"]["spearman"][-1] == pytest.approx(1.0)
        assert block["tuned"]["spearman"][-1] == pytest.approx(1.0)


def _intervals(obj, key=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _intervals(v, k)
    elif key == "ci":                     # the pooled difference: one interval
        yield obj
    elif key.endswith("_ci"):             # one interval per layer
        yield from obj


def test_intervals_are_ordered_and_in_range(runs):
    _, ((_, d), _) = runs
    s = _json(d / "summary.json")
    found = list(_intervals(s))
    assert len(found) > 50
    for ci in found:
        assert ci is not None and ci[0] <= ci[1]
    for t in s["cells"].values():
        for block in t.values():
            for lens in exp.LENSES:
                assert all(-1 <= lo <= hi <= 1 for lo, hi in block[lens]["spearman_ci"])
            assert all(-2 <= lo <= hi <= 2 for lo, hi in block["paired"]["spearman_diff_ci"])


def test_items_follow_the_registered_split_and_pairing(runs):
    corpus, ((_, d), _) = runs
    items = _json(d / "results.json")["items"]
    docs = sorted(json.loads(line)["id"] for line in corpus.read_text().splitlines())
    texts = {json.loads(line)["id"]: json.loads(line)["text"]
             for line in corpus.read_text().splitlines()}
    tok = tiny.tokenizer()
    for it in items:
        assert it["split"] == ("train" if docs.index(it["doc"]) % 2 == 0 else "test")
        partner = items[it["partner"]]
        assert partner["doc"] != it["doc"] and partner["split"] == it["split"]
        assert partner["window"] == it["window"]
        ids = tok(texts[it["doc"]], add_special_tokens=False)["input_ids"]
        assert it["input_ids"] == ids[it["start_token"]:it["start_token"] + exp.WINDOW_TOKENS]
        assert it["site_position"] == it["readout_position"] == exp.WINDOW_TOKENS - 1
        assert it["targets"]["donor_top1"] == partner["targets"]["clean_top1"]
        for t in exp.TARGETS:
            assert it["observed"]["self"][t] == [0.0] * (N_BLOCKS + 1)
    # within a split every document donates once and receives once, so no
    # document dominates the donor side
    for split in ("train", "test"):
        got = sorted(items[it["partner"]]["doc"] for it in items if it["split"] == split)
        assert got == sorted(it["doc"] for it in items if it["split"] == split)


def test_summary_is_rounded_sorted_and_small(runs):
    _, ((_, d), _) = runs
    data = (d / "summary.json").read_bytes()
    s = json.loads(data)
    assert json.dumps(s, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode() + b"\n" == data

    def floats(obj):
        if isinstance(obj, float):
            yield obj
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from floats(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from floats(v)

    assert all(float(f"{x:.6g}") == x for x in floats(s))
    assert len(data) < 25_000


def test_hashes_go_to_stderr_and_outputs_are_not_clobbered(tmp_path, capsys):
    corpus = _corpus(tmp_path / "corpus.jsonl")
    assert _main(tmp_path, corpus) == 0
    err = capsys.readouterr().err
    assert re.search(r"^controls: .*: PASS$", err, re.M)
    for name in ("results.json", "summary.json"):
        digest = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        assert f"sha256 {digest}  {tmp_path / name}" in err
    before = (tmp_path / "summary.json").read_bytes()
    assert _main(tmp_path, corpus) == 1
    assert "pass -f/--force" in capsys.readouterr().err
    assert (tmp_path / "summary.json").read_bytes() == before


def test_a_failed_control_exits_nonzero_after_writing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(exp, "POSITIVE_CONTROL_TOL", -1.0)
    assert _main(tmp_path, _corpus(tmp_path / "corpus.jsonl")) == 1
    assert "FAIL" in capsys.readouterr().err
    assert _json(tmp_path / "summary.json")["controls"]["passed"] is False


def test_rowwise_spearman_is_scipys():
    """The bootstrap ranks 2000 resamples at once; each row must be the same
    statistic scipy computes, ties (which resampling creates) included."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    x = np.round(rng.standard_normal((5, 30)), 1)
    y = x + rng.standard_normal((5, 30))
    got = exp._spearman(x, y)
    for row in range(5):
        assert got[row] == pytest.approx(spearmanr(x[row], y[row]).statistic, abs=1e-12)


def test_the_registered_corpus_is_the_committed_file():
    docs, sha = exp.load_corpus(exp.CORPUS)
    assert sha == exp.CORPUS_SHA256
    assert len(docs) == 80 and len({d["id"] for d in docs}) == 80
    assert all(len(d["text"]) > 900 for d in docs)


def test_a_changed_registered_corpus_is_refused_before_any_download(tmp_path, monkeypatch,
                                                                     capsys):
    monkeypatch.setattr(exp, "CORPUS_SHA256", "0" * 64)
    code = exp.main(["--out", str(tmp_path / "r.json"), "--summary", str(tmp_path / "s.json")])
    assert code == 1
    assert "the plan registered" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()
