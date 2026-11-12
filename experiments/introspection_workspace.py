"""Does the Jacobian-lens component of the introspection concept directions
reach report?

The design is pre-registered in experiments/README.md
(`introspection_workspace`), which is the authority on it; this file
implements that plan. Each held-out concept direction from the introspection
run is split, at the injection layer, into its component in the span of the
Jacobian-lens vectors (non-negative OMP, k = 16) and the remainder. Each
part, and the whole, is written at the same norm over held-out passages, and
the untrained model is asked which concept was injected.

    python -m experiments.jacobian_lens --device xpu --dtype bfloat16 --layers 19 --out lens/
    python -m experiments.introspection_workspace --device xpu --dtype bfloat16 \
        --lens lens/ --out results.json --summary summary.json

Everything the plan does not restate — model, concepts, passages, items,
rendering, the write and the scoring — is `experiments.introspection`'s,
called rather than copied. A written vector shows sufficiency under these
conditions (docs/validity.md), not mechanism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import nnls

from capture import load_model
from experiments import introspection as ix

SCHEMA = "mottled-introspection-workspace/1"
SUMMARY_SCHEMA = "mottled-introspection-workspace-summary/1"

K = 16              # the paper's k for concept vectors
STRENGTH = 4.0      # the registered introspection run's strength, not re-calibrated
ORTH_TOL = 1e-4
NORM_TOL = 1e-3
# items per task for the unedited-pass control: the plan requires a difference
# of exactly 0, not a census, and each check is two forward passes
CONTROL_ITEMS = 50
# (task, arm); "none" writes nothing, the others write that arm's direction
PLAN = (("identify", "none"), ("identify", "full"), ("identify", "J"),
        ("identify", "perp"), ("prime", "none"), ("prime", "J"))


def lens_vectors(model, J: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """(n_tokens, D) float32: a_t = J^T (gamma * W_U[t]), so <a_t, h> is the
    lens logit of t up to the 1/rms factor the final RMSNorm applies."""
    from models.families import resolve_family

    ad = resolve_family(model)
    if getattr(ad.final_norm, "bias", None) is not None:
        # a LayerNorm also centres, so <a_t, h> would not be the lens logit
        # up to a scale; the plan defines lens vectors for RMSNorm only
        raise ValueError(f"{type(ad.final_norm).__name__} has a bias: lens vectors "
                         "are defined here for an RMSNorm final norm")
    w = ad.lm_head.weight.detach().float().cpu()[:n_tokens]
    g = ad.final_norm.weight.detach().float().cpu()
    return (w * g) @ J.float()


def unit_rows(atoms: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(atoms, axis=1)
    return atoms / np.where(norms > 0, norms, 1)[:, None]


def nn_omp(u: np.ndarray, atoms: np.ndarray, k: int = K, unit: np.ndarray | None = None):
    """Non-negative orthogonal matching pursuit, as registered: select by the
    largest positive inner product of a unit-normalised atom with the
    residual, refit NNLS on every selected atom. Returns (selected ids,
    coefficients, fit), float64. `unit` is `unit_rows(atoms)`, passed in so a
    vocabulary-sized copy is made once rather than per concept."""
    u = np.asarray(u, np.float64)
    unit = unit_rows(atoms) if unit is None else unit
    selected, coef, fit = [], np.zeros(0), np.zeros_like(u)
    for _ in range(k):
        scores = unit @ (u - fit).astype(unit.dtype)
        scores[selected] = -np.inf
        best = int(np.argmax(scores))
        if not scores[best] > 0:
            break
        selected.append(best)
        basis = atoms[selected].astype(np.float64)
        coef, _ = nnls(basis.T, u)
        fit = basis.T @ coef
    return selected, coef, fit


def decompose(directions, concepts, atoms, tokenizer):
    """Per concept: u_J, u_perp and what the plan reports about the split."""
    a = atoms.numpy()
    unit = unit_rows(a)
    out = {}
    for c in concepts:
        u = directions[c].double().numpy()
        sel, coef, fit = nn_omp(u, a, unit=unit)
        resid = u - fit
        chosen = unit_rows(a[sel].astype(np.float64))
        orth = float(np.abs(chosen[coef > 0] @ resid).max()) if (coef > 0).any() else 0.0
        own = ix._word_token_ids(tokenizer, c)
        out[c] = {"u_J": torch.from_numpy(fit), "u_perp": torch.from_numpy(resid),
                  "selected": sel, "tokens": [tokenizer.decode([i]) for i in sel],
                  "coef": [float(x) for x in coef], "variance_share": float(fit @ fit),
                  "own_token_selected": bool(own & set(sel)), "orthogonality": orth}
    return out


def arm_directions(directions, parts, concepts):
    """Unit directions per arm: the registered vector, its J part, the rest."""
    unit = lambda v: (v / v.norm()).float()
    return {"full": {c: directions[c] for c in concepts},
            "J": {c: unit(parts[c]["u_J"]) for c in concepts},
            "perp": {c: unit(parts[c]["u_perp"]) for c in concepts}}


@torch.no_grad()
def unedited_control(model, tokenizer, items, layer, device, letters):
    """Max |logit difference| between the hooked pass with nothing written
    and an unhooked pass, over `items`. The plan requires exactly 0."""
    answer = ix.label_ids(tokenizer, letters)
    worst = 0.0
    for item in items:
        ids, masks = ix.render(tokenizer, item, letters)
        plain = model(input_ids=ids.to(device)[None], logits_to_keep=1).logits[0, -1, answer].float()
        none = ix.answer_logits(model, ids, masks.any(0), None, layer, answer, device)
        worst = max(worst, float((plain - none).abs().max()))
    return worst


def summarize(records, concepts, docs, B=2000):
    correct = lambda r: float(r["pred"] == r["target"])
    mats = {f"{t}/{a}": ix.cell_matrix([r for r in records if r["task"] == t and r["cond"] == a],
                                       concepts, docs, correct)
            for t, a in PLAN}
    boot = ix.crossed_bootstrap(mats, B=B)
    out = {"accuracy": {k: {"point": ix._point(m), "ci95": ix._ci(boot[k]),
                            "n": int((~np.isnan(m)).sum())} for k, m in mats.items()},
           "contrasts": {}}
    for name, a, b in (("identify J - identify none", "identify/J", "identify/none"),
                       ("identify J - identify full", "identify/J", "identify/full"),
                       ("identify J - identify perp", "identify/J", "identify/perp"),
                       ("prime J - prime none", "prime/J", "prime/none"),
                       ("identify J - prime J", "identify/J", "prime/J")):
        ci = ix._ci(boot[a] - boot[b])
        out["contrasts"][name] = {"point": ix._point(mats[a]) - ix._point(mats[b]),
                                  "ci95": ci, "above_zero": bool(ci[0] > 0)}
    out["answers"] = {f"{t}/{a}": np.bincount([r["pred"] for r in records
                                               if r["task"] == t and r["cond"] == a],
                                              minlength=len(ix.LETTERS)).tolist()
                      for t, a in PLAN}
    out["H1"] = ("supported" if out["contrasts"]["identify J - identify none"]["above_zero"]
                 else "not supported")
    return out


def load_lens(path, train_docs, n_blocks, revision, layer):
    """J at `layer` from the directory jacobian_lens.py wrote (a full lens or
    one fit only there), refused before any evaluation unless it is a
    Jacobian lens fit on these train documents for these weights.
    Returns (J float64 (D, D), config, sha256 of params.pt)."""
    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    state = torch.load(path / "params.pt", map_location="cpu", weights_only=True)
    problems = []
    if config.get("lens_kind") != "jacobian":
        problems.append(f"lens_kind is {config.get('lens_kind')!r}")
    if config.get("docs") != [d["id"] for d in train_docs]:
        problems.append("it was not fit on the plan's train documents")
    if config.get("num_hidden_layers") != n_blocks:
        problems.append(f"it is for {config.get('num_hidden_layers')} blocks, not {n_blocks}")
    if f"{layer}.weight" not in state:
        problems.append(f"it has no J at state {layer}")
    if revision is not None and config.get("base_model_revision") not in (None, revision):
        problems.append(f"it was fit on revision {config['base_model_revision']}")
    if problems:
        raise ValueError(f"{path} is not the registered lens: " + "; ".join(problems))
    w = state[f"{layer}.weight"].double()
    sha = hashlib.sha256((path / "params.pt").read_bytes()).hexdigest()
    return w + torch.eye(w.shape[0], dtype=torch.float64), config, sha


def main(argv=None, loaded=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=ix.MODEL)
    ap.add_argument("--revision", default=ix.MODEL_REVISION)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--corpus", default=str(ix.CORPUS))
    ap.add_argument("--docs", type=int, default=None, help="cap documents per split (tests)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--lens", required=True, help="directory written by experiments.jacobian_lens")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    args = ap.parse_args(argv)
    log = lambda s: print(s, file=sys.stderr, flush=True)
    t0 = time.time()
    torch.manual_seed(0)

    letters, numbers = ix.LETTERS, ix.NUMBERS
    if loaded is None:
        model, tok = load_model(args.model, device=args.device, dtype=args.dtype,
                                revision=args.revision)
    else:
        model, tok, letters, numbers = loaded
        model.to(args.device)
    device = args.device

    docs = ix.load_corpus(args.corpus, check=args.corpus == str(ix.CORPUS))
    train_docs, test_docs = ix.doc_split(docs)
    if args.docs:
        train_docs, test_docs = train_docs[:args.docs], test_docs[:args.docs]
    train_c, test_c = ix.concept_split()
    layer = ix.injection_layer(model)
    from models.families import resolve_family
    revision = getattr(model.config, "_commit_hash", None)
    J, lens_config, lens_sha = load_lens(args.lens, train_docs,
                                         resolve_family(model).n_layers, revision, layer)
    atoms = lens_vectors(model, J, len(tok))

    directions = ix.concept_directions(model, tok, ix.CONCEPTS, train_c, layer, device)
    scale_items, _ = ix.make_items("localize", train_docs[:10], train_c[:1])
    scale = ix.median_norm(model, tok, scale_items, layer, numbers, device)
    magnitude = STRENGTH * scale
    parts = decompose(directions, test_c, atoms, tok)
    arms = arm_directions(directions, parts, test_c)
    log(f"layer {layer}, median norm {scale:.2f} -> |v| = {magnitude:.2f}; "
        f"median J variance share {np.median([p['variance_share'] for p in parts.values()]):.3f}")

    norm_err = max(abs(float(d[c].norm()) * magnitude - magnitude) / magnitude
                   for d in arms.values() for c in test_c)
    controls = {
        "orthogonality_max": max(p["orthogonality"] for p in parts.values()),
        "orthogonality_tol": ORTH_TOL,
        "zero_J_concepts": [c for c in test_c if not parts[c]["variance_share"] > 0],
        "norm_max_rel_err": norm_err, "norm_tol": NORM_TOL,
    }

    test_items, skipped = {}, {}
    for task in ("identify", "prime"):
        test_items[task], skipped[f"test/{task}"] = ix.make_items(task, test_docs, test_c)
    controls["unedited_items"] = 2 * CONTROL_ITEMS
    controls["unedited_max_abs_diff"] = unedited_control(
        model, tok, test_items["identify"][:CONTROL_ITEMS] + test_items["prime"][:CONTROL_ITEMS],
        layer, device, letters)

    records = []
    for task, arm in PLAN:
        cond, dirs = ("none", directions) if arm == "none" else ("concept", arms[arm])
        rs = ix.evaluate(model, tok, test_items[task], cond, dirs, magnitude, layer,
                         device, letters, numbers)
        for r in rs:
            r["cond"] = arm
        records.extend(rs)
        log(f"  {task}/{arm}: {np.mean([r['pred'] == r['target'] for r in rs]):.3f} over {len(rs)}")

    manipulation = {}
    for arm in ("full", "J", "perp"):
        _, table = ix.calibrate_strength(model, tok, arms[arm], test_c, layer, scale, device)
        manipulation[arm] = {str(s): v for s, v in table.items()}
        log(f"  say-one-word top-{ix.CALIBRATION_TOPK} hit rate, {arm}: {manipulation[arm]}")

    controls["passed"] = bool(controls["orthogonality_max"] <= ORTH_TOL
                              and not controls["zero_J_concepts"]
                              and norm_err <= NORM_TOL
                              and controls["unedited_max_abs_diff"] == 0.0)

    summary = summarize(records, list(test_c), [d["id"] for d in test_docs], B=args.bootstrap)
    import transformers
    provenance = {
        "git_commit": ix._git_commit(), "model": args.model,
        "model_revision_requested": args.revision, "model_revision_resolved": revision,
        "dtype": args.dtype, "device": device,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "corpus_sha256": hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(),
        "lens_params_sha256": lens_sha,
        "lens_config": {k: lens_config.get(k) for k in
                        ("base_model_revision", "layers", "skip_first", "max_tokens",
                         "dtype", "device", "git_commit")},
    }
    decomposition = {c: {k: v for k, v in p.items() if k not in ("u_J", "u_perp")}
                     for c, p in parts.items()}
    config = {"layer": layer, "median_norm": scale, "strength": STRENGTH,
              "magnitude": magnitude, "k": K, "n_atoms": int(atoms.shape[0]),
              "test_concepts": list(test_c), "n_test_docs": len(test_docs),
              "skipped": skipped}
    summary.update({
        "schema": SUMMARY_SCHEMA, "config": config, "provenance": provenance,
        "controls": controls, "manipulation_check": manipulation,
        "decomposition": {c: {k: d[k] for k in ("tokens", "coef", "variance_share",
                                                 "own_token_selected")}
                          for c, d in decomposition.items()},
    })
    results = {"schema": SCHEMA, "config": config, "provenance": provenance,
               "controls": controls, "manipulation_check": manipulation,
               "decomposition": decomposition, "records": records,
               "timing": {"total_s": time.time() - t0}}
    Path(args.out).write_text(json.dumps(ix._round(results)))
    Path(args.summary).write_text(json.dumps(ix._round(summary), sort_keys=True, indent=1))
    log(f"controls {'passed' if controls['passed'] else 'FAILED'}: {controls}")
    log(f"H1: {summary['H1']}")
    for name, c in summary["contrasts"].items():
        log(f"  {name}: {c['point']:+.3f} {c['ci95']}")
    for p in (args.out, args.summary):
        log(f"{p} sha256 {hashlib.sha256(Path(p).read_bytes()).hexdigest()}")
    return summary


if __name__ == "__main__":
    raise SystemExit(0 if main()["controls"]["passed"] else 1)
