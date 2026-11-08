"""introspection_workspace (experiments/introspection_workspace.py): the
decomposition is the registered NN-OMP, lens vectors are the lens logit's
direction, a lens that is not the registered one is refused before any
evaluation, and the decision rule computes as the plan states it.

Offline on the tiny Llama and word-level tokenizer of test_introspection.
Its numbers mean nothing; these tests pin the machinery.
"""
import json

import numpy as np
import pytest

import test_introspection as base

torch = pytest.importorskip("torch")

from experiments import introspection as ix  # noqa: E402
from experiments import introspection_workspace as ws  # noqa: E402
from experiments import jacobian_lens as jl  # noqa: E402


def test_nn_omp_recovers_a_nonnegative_combination_and_stops_when_nothing_helps():
    rng = np.random.default_rng(0)
    atoms = rng.standard_normal((40, 12))
    u = 2.0 * atoms[3] + 0.5 * atoms[17]
    u /= np.linalg.norm(u)
    sel, coef, fit = ws.nn_omp(u, atoms, k=16)
    assert set(np.array(sel)[coef > 1e-9]) == {3, 17}
    np.testing.assert_allclose(fit, u, atol=1e-10)
    # a direction no atom points into gets no component
    sel, coef, fit = ws.nn_omp(-np.abs(rng.standard_normal(12)), np.abs(atoms), k=16)
    assert sel == [] and not fit.any()


def test_nn_omp_residual_is_orthogonal_to_its_positive_atoms():
    rng = np.random.default_rng(1)
    atoms = rng.standard_normal((200, 16))
    u = rng.standard_normal(16)
    sel, coef, fit = ws.nn_omp(u / np.linalg.norm(u), atoms, k=16)
    chosen = ws.unit_rows(atoms[sel])
    assert np.abs(chosen[coef > 0] @ (u / np.linalg.norm(u) - fit)).max() <= ws.ORTH_TOL


def test_decompose_splits_each_direction_into_orthogonal_parts():
    tok = base._tokenizer()
    rng = np.random.default_rng(2)
    atoms = torch.from_numpy(rng.standard_normal((len(tok), 32)).astype(np.float32))
    dirs = {c: torch.nn.functional.normalize(torch.randn(32), dim=0) for c in ("apple", "tiger")}
    parts = ws.decompose(dirs, list(dirs), atoms, tok)
    arms = ws.arm_directions(dirs, parts, list(dirs))
    for c, p in parts.items():
        # the residual is orthogonal to every atom the fit uses, so to the fit
        assert abs(float(p["u_J"] @ p["u_perp"])) < 1e-8
        np.testing.assert_allclose(p["variance_share"] + float(p["u_perp"] @ p["u_perp"]), 1.0,
                                   atol=1e-6)
        assert len(p["tokens"]) == len(p["coef"]) <= ws.K
        for arm in ("full", "J", "perp"):
            assert abs(float(arms[arm][c].norm()) - 1.0) < 1e-6
    assert torch.equal(arms["full"]["apple"], dirs["apple"])


def test_lens_vectors_give_the_lens_logit_up_to_the_norm_scale():
    from models.families import resolve_family

    model = base._model()
    ad = resolve_family(model)
    J = torch.randn(32, 32, dtype=torch.float64)
    h = torch.randn(32)
    a = ws.lens_vectors(model, J, 10)
    g = ad.final_norm.weight.detach().float()
    expected = ad.lm_head.weight.detach().float()[:10] @ (g * (J.float() @ h))
    torch.testing.assert_close(a @ h, expected, rtol=1e-4, atol=1e-4)


def test_lens_vectors_refuse_a_layernorm():
    model = base._model()
    model.model.norm = torch.nn.LayerNorm(32)
    with pytest.raises(ValueError, match="RMSNorm"):
        ws.lens_vectors(model, torch.eye(32, dtype=torch.float64), 10)


def _lens(tmp_path, model, tok, docs, layers=None):
    seqs = [torch.tensor(tok(d["text"], add_special_tokens=False)["input_ids"][:jl.MAX_TOKENS])
            for d in docs]
    path = tmp_path / "lens"
    jl.save(path, jl.fit(model, seqs, layers=layers), {"docs": [d["id"] for d in docs]},
            layers=layers, n_blocks=3)
    return path


def test_a_lens_fit_on_other_documents_is_refused(tmp_path):
    tok, model = base._tokenizer(), base._model()
    train, test = ix.doc_split(ix.load_corpus(base._corpus(tmp_path), check=False))
    path = _lens(tmp_path, model, tok, test[:2])
    with pytest.raises(ValueError, match="train documents"):
        ws.load_lens(path, train[:2], 3, None, 2)


def test_a_lens_without_J_at_the_injection_state_is_refused(tmp_path):
    tok, model = base._tokenizer(), base._model()
    train, _ = ix.doc_split(ix.load_corpus(base._corpus(tmp_path), check=False))
    path = _lens(tmp_path, model, tok, train[:2], layers=[1])
    with pytest.raises(ValueError, match="no J at state 2"):
        ws.load_lens(path, train[:2], 3, None, 2)


def test_decision_rule():
    concepts, docs = ["a", "b"], ["x", "y"]
    recs = [{"task": t, "cond": a, "concept": c, "doc": d, "target": 0,
             "pred": 0 if (t, a) == ("identify", "J") else 1}
            for t, a in ws.PLAN for c in concepts for d in docs]
    s = ws.summarize(recs, concepts, docs, B=50)
    assert s["accuracy"]["identify/J"]["point"] == 1.0
    assert s["contrasts"]["identify J - identify none"]["point"] == 1.0
    assert s["H1"] == "supported"
    assert s["answers"]["identify/none"] == [0, 4, 0, 0]      # 2 concepts x 2 docs
    for r in recs:
        r["pred"] = 1
    assert ws.summarize(recs, concepts, docs, B=50)["H1"] == "not supported"


def test_end_to_end_on_a_tiny_model(tmp_path):
    tok, model = base._tokenizer(), base._model()
    corpus = base._corpus(tmp_path)
    train, _ = ix.doc_split(ix.load_corpus(corpus, check=False))
    lens = _lens(tmp_path, model, tok, train[:2], layers=[2])     # the injection state only
    out, summ = tmp_path / "r.json", tmp_path / "s.json"
    s = ws.main(["--corpus", str(corpus), "--docs", "2", "--bootstrap", "20", "--lens", str(lens),
                 "--out", str(out), "--summary", str(summ)],
                loaded=(model, tok, ix.LETTERS, ix.NUMBERS))
    r = json.loads(out.read_text())
    assert {(x["task"], x["cond"]) for x in r["records"]} == set(ws.PLAN)
    assert s["controls"]["passed"], s["controls"]
    assert s["controls"]["unedited_max_abs_diff"] == 0.0
    assert s["provenance"]["lens_config"]["layers"] == [2]
    assert s["H1"] in {"supported", "not supported"}
    assert set(s["manipulation_check"]) == {"full", "J", "perp"}
    assert all(len(d["tokens"]) <= ws.K for d in s["decomposition"].values())
    assert json.loads(summ.read_text())["schema"] == ws.SUMMARY_SCHEMA
    assert len(s["provenance"]["lens_params_sha256"]) == 64
