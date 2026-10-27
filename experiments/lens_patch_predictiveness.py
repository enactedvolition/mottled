"""Do lens readouts anticipate what a residual-state patch does to the output?

The design is pre-registered in experiments/README.md, which is the authority
on it; this file implements that plan. For each 32-token window of the State
of the Union corpus and each GPT-2 layer, the state at the window's last
position is replaced (by a partner window's state from another address, by a
norm-matched random push, or by itself), and the change in the model's own
final log-odds of a target token is set against the change the raw and the
tuned lens read off the patched state at that layer.

    python -m experiments.lens_patch_predictiveness --out results.json --summary summary.json

A lens is a readout diagnostic and a patch effect is sufficiency under the
tested conditions (docs/validity.md): a lens that predicts patch effects
better is a better readout here, not the model's belief at that layer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from experiments import patching
from intervene import _norm_matched_random

SCHEMA = "mottled-lens-patch-predictiveness/1"
SUMMARY_SCHEMA = "mottled-lens-patch-predictiveness-summary/1"

MODEL = "gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
HERE = Path(__file__).resolve().parent
CORPUS = HERE / "sotu_corpus.jsonl"
CORPUS_SHA256 = "0e00dc3a326d362094e8c142418edbf6c1a5a42421ed9ecb6b3a108605b6d9a2"

WINDOWS_PER_DOC = 2
WINDOW_TOKENS = 32
BOOTSTRAP = 2000
SEED = 0
THREADS = 4
# 2.8e-5 was measured on a GPT-2-small-shaped model with logits scaled to
# +-350: float32 rounding between the head on (1, D) and on (1, T, D). A
# misaligned layer, position or target would miss by an effect size instead.
POSITIVE_CONTROL_TOL = 1e-3

CONDITIONS = ("donor", "random", "self")
SCORED = ("donor", "random")     # self is the identity control, not a predictor test
TARGETS = ("clean_top1", "donor_top1")
LENSES = ("raw", "tuned")


def load_corpus(path) -> tuple[list[dict], str]:
    data = Path(path).read_bytes()
    docs = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    return sorted(docs, key=lambda d: d["id"]), hashlib.sha256(data).hexdigest()


def make_items(docs: list[dict], tokenizer) -> list[dict]:
    """Windows, split and partners exactly as the plan states them.

    Documents sorted by id: even rank train, odd rank test. A window's
    partner is the same window of the next document in its own split, so a
    donor always comes from another address and never crosses the split.
    """
    docs = sorted(docs, key=lambda d: d["id"])
    split = {d["id"]: "train" if r % 2 == 0 else "test" for r, d in enumerate(docs)}
    ranked = [[d["id"] for d in docs if split[d["id"]] == s] for s in ("train", "test")]
    if min(len(ids) for ids in ranked) < 2:
        raise ValueError("each split needs at least two documents to pair donors")
    partner = {ids[r]: ids[(r + 1) % len(ids)] for ids in ranked for r in range(len(ids))}
    need = WINDOWS_PER_DOC * WINDOW_TOKENS
    items, index = [], {}
    for d in docs:
        # no BOS: HF GPT-2 prepends none, and asking for none makes that true
        # of any tokenizer this runs with
        ids = tokenizer(d["text"], add_special_tokens=False)["input_ids"]
        if len(ids) < need:
            raise ValueError(f"{d['id']} has {len(ids)} tokens; its windows need {need}")
        for m in range(WINDOWS_PER_DOC):
            index[d["id"], m] = len(items)
            start = m * WINDOW_TOKENS
            items.append({"item": len(items), "doc": d["id"], "split": split[d["id"]],
                          "window": m, "start_token": start,
                          "input_ids": [int(i) for i in ids[start:start + WINDOW_TOKENS]],
                          "site_position": WINDOW_TOKENS - 1,
                          "readout_position": WINDOW_TOKENS - 1})
    for it in items:
        it["partner"] = index[partner[it["doc"]], it["window"]]
    return items


def measure(model, lens, items: list[dict], log) -> None:
    """Add targets, observed and predicted log-odds changes to every item."""
    site = WINDOW_TOKENS - 1
    t0 = time.perf_counter()
    cleans = [patching.clean(model, it["input_ids"], readout=site) for it in items]
    states = [c.hidden[:, site] for c in cleans]                 # (L, D) per item
    top1 = [int(c.logits[0].argmax()) for c in cleans]
    n_layers = states[0].shape[0]
    for i, it in enumerate(items):
        j = it["partner"]
        base, h, h_donor = cleans[i], states[i], states[j]
        targets = {"clean_top1": top1[i], "donor_top1": top1[j]}
        values = {
            "donor": h_donor,
            "random": np.stack([h[k] + _norm_matched_random(h_donor[k] - h[k],
                                                            seed=SEED + i * n_layers + k)
                                for k in range(n_layers)]),
            "self": h,
        }
        clean_lo = {t: float(patching.log_odds(base.logits[0], y)) for t, y in targets.items()}
        observed = {c: {t: [] for t in TARGETS} for c in CONDITIONS}
        self_diff = 0.0
        for k in range(n_layers):
            for c in CONDITIONS:
                p = patching.patch(model, it["input_ids"], k, site, values[c][k],
                                   readout=site, clean_run=base)
                for t, y in targets.items():
                    observed[c][t].append(float(patching.log_odds(p.logits[0], y)) - clean_lo[t])
                if c == "self":
                    self_diff = max(self_diff, float(np.abs(p.logits - base.logits).max()))
        # the lens sees only the state at the site, clean and patched, read
        # as the layer it sits at: (L, 3, D) = clean, donor, random per layer
        read = np.stack([h, values["donor"], values["random"]], axis=1)
        predicted = {}
        for name, lz in zip(LENSES, (None, lens)):
            z = patching.lens_logits(model, read, lz)
            lo = {t: patching.log_odds(z, y) for t, y in targets.items()}      # (L, 3)
            predicted[name] = {c: {t: (lo[t][:, col] - lo[t][:, 0]).tolist() for t in TARGETS}
                               for col, c in ((1, "donor"), (2, "random"))}
        it.update(targets=targets, clean_logodds=clean_lo, observed=observed,
                  predicted=predicted, self_max_abs_logit_diff=self_diff,
                  delta_norm=np.linalg.norm((h_donor - h).astype(np.float64), axis=-1).tolist())
        if (i + 1) % 10 == 0 or i + 1 == len(items):
            log(f"  {i + 1}/{len(items)} items, {time.perf_counter() - t0:.0f} s")


def controls(items: list[dict], n_layers: int) -> dict:
    last = n_layers - 1
    self_logit = max(it["self_max_abs_logit_diff"] for it in items)
    self_obs = max(abs(v) for it in items for t in TARGETS for v in it["observed"]["self"][t])
    positive = max(abs(it["predicted"][lens][c][t][last] - it["observed"][c][t][last])
                   for it in items for lens in LENSES for c in SCORED for t in TARGETS)
    return {"self_max_abs_logit_diff": self_logit, "self_max_abs_delta_obs": self_obs,
            "positive_control_layer": last, "positive_control_max_abs_err": positive,
            "positive_control_tol": POSITIVE_CONTROL_TOL,
            "passed": self_logit == 0.0 and self_obs == 0.0 and positive <= POSITIVE_CONTROL_TOL}


def analyse(items: list[dict], n_layers: int, bootstrap: int = BOOTSTRAP,
            seed: int = SEED) -> dict:
    """Held-out metrics per cell, paired cluster-bootstrap intervals, H1."""
    train = [i for i, it in enumerate(items) if it["split"] == "train"]
    test = [i for i, it in enumerate(items) if it["split"] == "test"]
    test_docs = sorted({items[i]["doc"] for i in test})
    at = {i: n for n, i in enumerate(test)}
    clusters = np.array([[at[i] for i in test if items[i]["doc"] == d] for d in test_docs])
    # one draw of documents for every statistic and every cell: that is what
    # makes the lens differences paired and the pooled layer mean coherent
    draw = np.random.default_rng(seed).integers(0, len(test_docs),
                                                size=(bootstrap, len(test_docs)))
    boot = clusters[draw].reshape(bootstrap, -1)

    cells, rho_diff = {}, {}
    for c in SCORED:
        cells[c] = {}
        for t in TARGETS:
            obs = np.array([it["observed"][c][t] for it in items])
            fits = {}
            for lens in LENSES:
                pred = np.array([it["predicted"][lens][c][t] for it in items])
                fits[lens] = [_cell(pred[train, k], obs[train, k], pred[test, k], obs[test, k],
                                    boot) for k in range(n_layers)]
            block = {"layers": list(range(n_layers)),
                     "mae_null": [f["mae_null"] for f in fits["raw"]]}
            for lens, fs in fits.items():
                block[lens] = {key: [f[key] for f in fs] for key in
                               ("spearman", "pearson", "mae_cal", "rmse_cal", "mae_uncal",
                                "calib_a", "calib_b")}
                block[lens]["spearman_ci"] = [_ci(f["boot_rho"]) for f in fs]
                cis = [_ci(f["boot_mae_cal"] - f["boot_mae_null"]) for f in fs]
                block[lens]["mae_cal_minus_null_ci"] = cis
                block[lens]["beats_null"] = [ci is not None and ci[1] < 0 for ci in cis]
            raw, tuned = fits["raw"], fits["tuned"]
            rho_diff[c, t] = [b["boot_rho"] - a["boot_rho"] for a, b in zip(raw, tuned)]
            block["paired"] = {
                "spearman_diff": [b["spearman"] - a["spearman"] for a, b in zip(raw, tuned)],
                "spearman_diff_ci": [_ci(d) for d in rho_diff[c, t]],
                "mae_cal_diff": [b["mae_cal"] - a["mae_cal"] for a, b in zip(raw, tuned)],
                "mae_cal_diff_ci": [_ci(b["boot_mae_cal"] - a["boot_mae_cal"])
                                    for a, b in zip(raw, tuned)]}
            cells[c][t] = block

    layers = list(range(1, n_layers - 1))
    h1 = cells["donor"]["clean_top1"]
    diff_ci = [h1["paired"]["spearman_diff_ci"][k] for k in layers]
    pooled_ci = _ci(np.mean([rho_diff["donor", "clean_top1"][k] for k in layers], axis=0))
    primary = {
        "hypothesis": "donor condition, clean_top1 target, layers 1..L-2: "
                      "Spearman rho(tuned) - rho(raw) > 0 on test documents",
        "layers": layers,
        "spearman_raw": [h1["raw"]["spearman"][k] for k in layers],
        "spearman_tuned": [h1["tuned"]["spearman"][k] for k in layers],
        "spearman_diff": [h1["paired"]["spearman_diff"][k] for k in layers],
        "spearman_diff_ci": diff_ci,
        "verdict": [_verdict(ci) for ci in diff_ci],
        "pooled": {"spearman_diff": float(np.mean([h1["paired"]["spearman_diff"][k]
                                                   for k in layers])),
                   "ci": pooled_ci, "verdict": _verdict(pooled_ci)},
    }
    return {"cells": cells, "primary": primary,
            "counts": {"documents": len({it["doc"] for it in items}), "items": len(items),
                       "train_items": len(train), "test_items": len(test),
                       "test_documents": len(test_docs),
                       "items_with_equal_targets": sum(it["targets"]["clean_top1"]
                                                       == it["targets"]["donor_top1"]
                                                       for it in items)}}


def _cell(x_tr, y_tr, x_te, y_te, boot) -> dict:
    """One (lens, layer, condition, target): calibration fitted on train, every
    error measured on test, and the bootstrap replicates the intervals need."""
    var = float(x_tr.var())
    # a predictor with no spread on train cannot be calibrated; it falls back
    # to the train mean, i.e. to the null predictor
    b = float(((x_tr - x_tr.mean()) * (y_tr - y_tr.mean())).mean() / var) if var > 0 else 0.0
    a = float(y_tr.mean() - b * x_tr.mean())
    resid = y_te - (a + b * x_te)
    err_cal, err_null = np.abs(resid), np.abs(y_te - y_tr.mean())
    return {"calib_a": a, "calib_b": b,
            "spearman": float(_spearman(x_te[None], y_te[None])[0]),
            "pearson": float(_pearson(x_te[None], y_te[None])[0]),
            "mae_cal": float(err_cal.mean()), "rmse_cal": float(np.sqrt((resid ** 2).mean())),
            "mae_uncal": float(np.abs(y_te - x_te).mean()), "mae_null": float(err_null.mean()),
            "boot_rho": _spearman(x_te[boot], y_te[boot]),
            "boot_mae_cal": err_cal[boot].mean(axis=1),
            "boot_mae_null": err_null[boot].mean(axis=1)}


def _spearman(x, y):
    """Row-wise Spearman with average ranks for ties, as scipy.stats.spearmanr."""
    return _pearson(rankdata(x, axis=1), rankdata(y, axis=1))


def _pearson(x, y):
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    den = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (x * y).sum(axis=1) / den, np.nan)


def _ci(replicates):
    # an undefined replicate (a resample with no spread) is not dropped: an
    # interval over the rest would describe a different resampling scheme
    reps = np.asarray(replicates, dtype=np.float64)
    if np.isnan(reps).any():
        return None
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def _verdict(ci) -> str:
    if ci is None:
        return "undefined"
    if ci[0] > 0:
        return "tuned better"
    if ci[1] < 0:
        return "raw better"
    return "not distinguishable"


def describe(model, tokenizer, lens, corpus: str, corpus_sha: str, items: list[dict],
             requested_revision: str | None, lens_revision: str | None) -> dict:
    import torch

    import provenance

    ids = json.dumps([it["input_ids"] for it in items]).encode()
    weights = hashlib.sha256()
    for arr in (lens.weight, lens.bias):
        arr = np.ascontiguousarray(arr, dtype="<f4")
        weights.update(repr(arr.shape).encode() + arr.tobytes())
    return {
        "git": _git(),
        "model": {**patching.model_record(model), "requested_revision": requested_revision},
        "tokenizer": {"class": type(tokenizer).__name__,
                      "name_or_path": getattr(tokenizer, "name_or_path", None),
                      "requested_revision": requested_revision,
                      "vocab_size": len(tokenizer), "add_special_tokens": False},
        "lens": {**lens.record(), "requested_revision": lens_revision,
                 "n_translators": lens.n_layers, "sha256": weights.hexdigest()},
        "corpus": {"file": Path(corpus).name, "sha256": corpus_sha,
                   "registered_sha256": CORPUS_SHA256},
        "input_ids_sha256": hashlib.sha256(ids).hexdigest(),
        "environment": {**provenance.environment(), "torch_threads": torch.get_num_threads(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
    }


def config(n_layers: int) -> dict:
    # which model ran, and at which commit, is the provenance block's to say
    return {
        "windows_per_doc": WINDOWS_PER_DOC, "window_tokens": WINDOW_TOKENS,
        "site_position": WINDOW_TOKENS - 1, "readout_position": WINDOW_TOKENS - 1,
        "n_layers": n_layers, "conditions": list(CONDITIONS), "targets": list(TARGETS),
        "lenses": list(LENSES), "primary_layers": list(range(1, n_layers - 1)),
        "split_rule": "documents sorted by id; even rank train, odd rank test",
        "partner_rule": "same window of the next document in the same split, cyclically",
        "random_rule": "intervene._norm_matched_random(h_donor - h, seed=SEED + item*L + layer)",
        "log_odds": "z_y - logsumexp_{j != y} z_j, float64",
        "bootstrap": BOOTSTRAP, "seed": SEED, "interval": "percentile 95%, cluster over test documents",
        "positive_control_tol": POSITIVE_CONTROL_TOL,
    }


def summarize(results: dict) -> dict:
    """The compact, deterministic part: no items, no timing, 6 significant digits."""
    keep = ("config", "provenance", "controls", "counts", "primary", "cells")
    return _round({"schema": SUMMARY_SCHEMA, **{k: results[k] for k in keep}})


def _round(obj):
    if isinstance(obj, float):
        return float(f"{obj:.6g}") if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v) for v in obj]
    return obj


def _finite(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def _dump(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False,
                       ensure_ascii=False) + "\n").encode("utf-8")


def _git() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(HERE), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    # untracked files count: the code that ran may be one of them
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status)}


def report(results: dict, log) -> None:
    ctl = results["controls"]
    log(f"controls: self-patch max |d logit| {ctl['self_max_abs_logit_diff']:g} and max "
        f"|d obs| {ctl['self_max_abs_delta_obs']:g} (both must be exactly 0); positive "
        f"control at layer {ctl['positive_control_layer']} max |d pred - d obs| "
        f"{ctl['positive_control_max_abs_err']:.3g} (tol {ctl['positive_control_tol']:g}): "
        f"{'PASS' if ctl['passed'] else 'FAIL'}")
    p = results["primary"]
    log("H1, donor / clean_top1, Spearman rho on test documents (95% paired cluster bootstrap):")
    for k, r, tu, d, ci, v in zip(p["layers"], p["spearman_raw"], p["spearman_tuned"],
                                  p["spearman_diff"], p["spearman_diff_ci"], p["verdict"]):
        span = "undefined" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        log(f"  layer {k:2d}  raw {r:+.3f}  tuned {tu:+.3f}  diff {d:+.3f} {span}  {v}")
    pooled = p["pooled"]
    span = "undefined" if pooled["ci"] is None else \
        f"[{pooled['ci'][0]:+.3f}, {pooled['ci'][1]:+.3f}]"
    log(f"  pooled layers {p['layers'][0]}-{p['layers'][-1]}  diff "
        f"{pooled['spearman_diff']:+.3f} {span}  {pooled['verdict']}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m experiments.lens_patch_predictiveness",
        description="Raw vs tuned lens predictions of residual-patch effects on GPT-2 "
                    "(pre-registered in experiments/README.md).")
    p.add_argument("--out", required=True, help="full results JSON")
    p.add_argument("--summary", required=True, help="compact, deterministic summary JSON")
    p.add_argument("--lens-revision", default=None,
                   help="tuned-lens Space commit (default: its main at run time, recorded)")
    p.add_argument("--corpus", default=str(CORPUS),
                   help="corpus JSONL (default: the registered one, checked by SHA-256)")
    p.add_argument("--threads", type=int, default=THREADS, help="torch CPU threads")
    p.add_argument("-f", "--force", action="store_true", help="overwrite the outputs")
    return p


def main(argv=None, loaded=None) -> int:
    """`loaded` = (model, tokenizer, lens) stands in for the hub download: the
    hook that lets a test run everything else here on a tiny model."""
    import torch

    args = _parser().parse_args(argv)
    for path in (args.out, args.summary):
        if Path(path).exists() and not args.force:
            print(f"error: {path} exists; pass -f/--force to overwrite", file=sys.stderr)
            return 1
    threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    try:
        return _run(args, loaded)
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


def _run(args, loaded) -> int:
    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    t0 = time.perf_counter()
    docs, corpus_sha = load_corpus(args.corpus)
    if Path(args.corpus).resolve() == CORPUS and corpus_sha != CORPUS_SHA256:
        log(f"error: {args.corpus} has sha256 {corpus_sha}; the plan registered {CORPUS_SHA256}")
        return 1
    if loaded is None:
        import tuned_lens

        log(f"loading {MODEL}@{MODEL_REVISION} and its tuned lens")
        model, tokenizer = patching.load_model(MODEL, MODEL_REVISION)
        lens = tuned_lens.from_hub(MODEL, revision=args.lens_revision)
        requested = MODEL_REVISION
    else:
        model, tokenizer, lens = loaded
        requested = None
    items = make_items(docs, tokenizer)
    described = describe(model, tokenizer, lens, args.corpus, corpus_sha, items,
                         requested, args.lens_revision)
    log(f"{len(items)} items from {len(docs)} documents")
    measure(model, lens, items, log)
    n_layers = len(items[0]["observed"]["self"]["clean_top1"])
    results = {"schema": SCHEMA, "config": config(n_layers), "provenance": described,
               "controls": controls(items, n_layers), **analyse(items, n_layers),
               "items": items, "timing": {"seconds": time.perf_counter() - t0}}
    hashes = []
    for path, data in ((args.out, _dump(_finite(results))),
                       (args.summary, _dump(summarize(results)))):
        Path(path).write_bytes(data)
        hashes.append(f"sha256 {hashlib.sha256(data).hexdigest()}  {path} ({len(data)} bytes)")
    report(results, log)
    for line in hashes:
        log(line)
    return 0 if results["controls"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
