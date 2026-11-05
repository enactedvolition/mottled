"""Capture-to-projection reconstruction: re-run a scene from its own record.

A scene stores projected, draped coordinates, entropies and readouts, not
hidden states, so `test_bundled_samples.py` can only start from the projected
plane. This module starts from the other end. It reads a scene's analysis
record (`provenance.record`: config, prompts, and per run the model id, hub
revision, dtype, lens and backend), loads those weights, and runs the prompts
through the same functions `mottled export` uses
(`pipeline.capture_trajectories` -> `pipeline.project_trajectories` ->
`statefile.save_scene`). Then it compares the rebuilt scene with the stored one,
quantity by quantity: tokens and decoded ids exactly; entropies, attention,
top-k readouts, projected coordinates and terrain to a measured tolerance.

Nothing is guessed. If a record lacks a field the re-run needs, the scene is
reported with the dotted path of every missing field and is not re-run on a
default. A default that happened to match would look like evidence, and one
that didn't would look like a regression.

Tolerances are measured at test time. The same weights are cast to float64
and run through the same pipeline. For each quantity, the largest
float32-vs-float64 difference D is the footprint of float32 rounding for this
model, these prompts and this config. Two correct float32 runs that differ only
in summation order (another torch or BLAS build, another thread count) each
sit within about D of the float64 result, so they can be up to 2D apart. The
tolerance is SAFETY * D (twice that bound), plus two float32 ulps of the
quantity's scale for the storage cast. If D exceeds ILL_CONDITIONED of a
quantity's scale, the test fails and names that quantity as ill-conditioned
instead of comparing it, because a tolerance derived from it would accept
almost anything. The re-run is on CPU. A scene captured on a GPU also carries
that kernel's rounding, which this measurement does not model; such a scene
failing here is not by itself a regression.

PCA axes are defined only up to sign. `projection.py` pins each component's
sign (largest-|loading| entry positive), but a near-tie between two loadings,
or a scene projected before the pin existed, can still flip an axis. A flip
changes every x (or y) and mirrors the terrain grid without changing anything
the scene means. So the comparison first checks pairwise distances between all
projected states, which are rotation- and reflection-invariant and need no
alignment. Then it aligns each rebuilt axis to the stored one by the sign of
their correlation, mirrors the rebuilt terrain with it, and compares
coordinates and terrain directly. If an axis correlates with its stored
counterpart below MIN_AXIS_CORRELATION in absolute value, the planes differ by
more than a flip (a rotation, i.e. near-equal leading eigenvalues). That is
reported, not aligned away.

The offline tests below prove the logic on the tiny model from tests/tiny.py.
The `network` test runs it on real GPT-2:

    python -m pytest -m network tests/test_real_capture_e2e.py -v -rsP

The bundled samples predate the analysis record, so none of them can be
re-run this way. Covering one means re-exporting it with a record.
"""
from __future__ import annotations

import copy
import io
from dataclasses import fields, replace
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist

import pipeline
import statefile
from config import MarbleConfig

SAMPLES = Path(__file__).resolve().parent.parent / "viewer" / "samples"

SAFETY = 4.0
ILL_CONDITIONED = 1e-3
# Gates the sign alignment only. Matching planes correlate at 1 - O((D/spread)^2).
# The coordinate comparison that follows is what runs at tolerance.
MIN_AXIS_CORRELATION = 0.999
F32_EPS = float(np.finfo(np.float32).eps)

# Every MarbleConfig field a stored number can depend on: the cache key's set
# (pipeline._UNKEYED_FIELDS are where the cache lives and a display rate).
CONFIG_FIELDS = sorted(f.name for f in fields(MarbleConfig)
                       if f.name not in pipeline._UNKEYED_FIELDS)
MODEL_FIELDS = ("id", "revision", "dtype", "lens", "backend")
# Quantities whose values depend on the plane's orientation. They are compared
# only after a successful sign alignment.
FRAME_DEPENDENT = ("xy", "terrain x", "terrain y", "terrain z", "terrain density",
                   "terrain se")


# ------------------------------------------------------------ the record
def _is_local_dir(model_id) -> bool:
    # A local checkpoint has no hub commit (provenance._model records None).
    # Its identity is its path, which the re-run loads as given.
    return isinstance(model_id, str) and Path(model_id).expanduser().is_dir()


def missing_fields(scene: dict) -> list[str]:
    """Dotted path of every record field a re-run of `scene` needs and lacks."""
    n = len(scene["runs"])
    rec = scene.get("analysis")
    if not isinstance(rec, dict):
        return (["analysis.config", "analysis.prompts"]
                + [f"analysis.models[{i}].{k}" for i in range(n) for k in MODEL_FIELDS])
    out = []
    cfg = rec.get("config")
    if not isinstance(cfg, dict):
        out.append("analysis.config")
    else:
        out += [f"analysis.config.{k}" for k in CONFIG_FIELDS if k not in cfg]
    prompts = rec.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != n:
        out.append("analysis.prompts")
    models = rec.get("models") if isinstance(rec.get("models"), list) else []
    for i in range(n):
        m = models[i] if i < len(models) and isinstance(models[i], dict) else {}
        for k in MODEL_FIELDS:
            if m.get(k) is None and not (k == "revision" and _is_local_dir(m.get("id"))):
                out.append(f"analysis.models[{i}].{k}")
    return out


# ------------------------------------------------------------- the re-run
def _rebuild(cfg: MarbleConfig, prompts: list[str], model, tokenizer) -> dict:
    """The `mottled export` path on pre-loaded weights, read back from bytes
    so the rebuild goes through the same float32 storage as the stored scene."""
    trajs = pipeline.capture_trajectories(cfg, prompts, model=model, tokenizer=tokenizer)
    buf = io.BytesIO()
    statefile.save_scene(pipeline.project_trajectories(cfg, trajs), buf)
    buf.seek(0)
    return statefile.load_scene(buf)


def reconstruct(scene: dict, load) -> tuple[dict, dict]:
    """Re-run `scene` from its record: (rebuild in the recorded dtype,
    the same rebuild in float64). `load(id, revision, dtype)` returns
    (model, tokenizer)."""
    import torch

    rec = scene["analysis"]
    cfg = replace(MarbleConfig(**{k: rec["config"][k] for k in CONFIG_FIELDS}),
                  use_cache=False, cache_dir=None)
    m = rec["models"][0]
    model, tokenizer = load(m["id"], m["revision"], m["dtype"])
    rebuilt = _rebuild(cfg, rec["prompts"], model, tokenizer)
    # The capture passes dtype only to a by-name load, so a float64 copy of
    # the instance runs the whole forward pass in float64.
    wide = _rebuild(cfg, rec["prompts"], copy.deepcopy(model).to(torch.float64), tokenizer)
    return rebuilt, wide


# ---------------------------------------------------------- the comparison
def _plane(scene: dict) -> np.ndarray:
    return np.concatenate([np.asarray(r["points"], np.float64)[..., :2].reshape(-1, 2)
                           for r in scene["runs"]])


def axis_signs(ref: dict, other: dict):
    """Per-axis signs mapping `other`'s plane onto `ref`'s, or (None, why)."""
    a, b = _plane(ref), _plane(other)
    if a.shape != b.shape:
        return None, f"projected states: {a.shape[0]} vs {b.shape[0]}"
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    signs = np.ones(2)
    for ax in range(2):
        r = float(a[:, ax] @ b[:, ax]) / max(
            float(np.linalg.norm(a[:, ax]) * np.linalg.norm(b[:, ax])), 1e-300)
        if abs(r) < MIN_AXIS_CORRELATION:
            return None, (f"projection axis {'xy'[ax]} correlates at r={r:.6f}: the planes "
                          "differ by more than a sign flip")
        signs[ax] = np.sign(r)
    return signs, None


def flipped(scene: dict, signs) -> dict:
    """`scene` with its plane axes multiplied by `signs` and its terrain
    mirrored to match. grid_x is linspace(lo, hi) of the points, so x -> -x
    turns it into -grid_x[::-1] and reverses each row of z (z[i, j] sits at
    (x[j], y[i])). y and the rows work the same way."""
    out = {**scene, "terrain": dict(scene["terrain"]), "runs": []}
    for run in scene["runs"]:
        p = np.array(run["points"], dtype=np.float64)
        p[..., :2] *= signs
        out["runs"].append({**run, "points": p})
    t = out["terrain"]
    for ax, grid, axis in ((0, "x", 1), (1, "y", 0)):
        if signs[ax] < 0:
            t[grid] = -np.asarray(t[grid])[::-1]
            for k in ("z", "density", "se"):
                if k in t:
                    t[k] = np.flip(np.asarray(t[k]), axis=axis)
    return out


def _numbers(scene: dict) -> dict[str, np.ndarray]:
    out = {"pairwise distances": pdist(_plane(scene))}
    xy, z = [], []
    for run in scene["runs"]:
        lab = run["label"]
        out[f"run {lab} entropy"] = np.asarray(run["entropy"], np.float64)
        if "attention" in run:
            out[f"run {lab} attention"] = np.asarray(run["attention"], np.float64)
        if run.get("topk") is not None:
            out[f"run {lab} top-k p"] = np.array(
                [[sorted((p for _, p in state), reverse=True) for state in layer]
                 for layer in run["topk"]], dtype=np.float64)
        p = np.asarray(run["points"], np.float64)
        xy.append(p[..., :2].ravel())
        z.append(p[..., 2].ravel())
    out["xy"], out["points z"] = np.concatenate(xy), np.concatenate(z)
    for k, arr in scene["terrain"].items():
        out[f"terrain {k}"] = np.asarray(arr, np.float64)
    return out


def _gaps(ref: dict, other: dict):
    """{name: (max |ref - other|, scale of ref, where)}, plus shape problems."""
    gaps, problems = {}, []
    for name in sorted(set(ref) | set(other)):
        if name not in ref or name not in other:
            problems.append(f"{name}: present in only one scene")
        elif ref[name].shape != other[name].shape:
            problems.append(f"{name}: shape {ref[name].shape} vs {other[name].shape}")
        elif ref[name].size:
            d = np.abs(ref[name] - other[name])
            where = np.unravel_index(int(np.argmax(d)), d.shape)
            gaps[name] = (float(d.max()), float(np.abs(ref[name]).max()),
                          tuple(int(i) for i in where))
    return gaps, problems


def tolerances(rebuilt: dict, wide: dict) -> dict[str, float]:
    """Per-quantity tolerance from the float32-vs-float64 gap of one re-run."""
    signs, why = axis_signs(rebuilt, wide)
    assert why is None, f"float32 and float64 re-runs disagree: {why}"
    # Tokens can differ only through a decode (generate_tokens > 0) whose
    # greedy choice is a near-tie that float32 rounding decides.
    assert [r["tokens"] for r in rebuilt["runs"]] == [r["tokens"] for r in wide["runs"]], \
        "float32 and float64 re-runs decoded different tokens: a near-tie decided by rounding"
    gaps, problems = _gaps(_numbers(rebuilt), _numbers(flipped(wide, signs)))
    assert not problems, problems
    ill = [f"{n}: float32 rounding alone moves it by {d:.3g} of {s:.3g}"
           for n, (d, s, _) in gaps.items() if d > ILL_CONDITIONED * s]
    assert not ill, "ill-conditioned here, not comparable:\n" + "\n".join(ill)
    return {n: SAFETY * d + 2 * F32_EPS * s for n, (d, s, _) in gaps.items()}


def _topk_misses(lab: str, stored, rebuilt, tol: float) -> list[str]:
    # A token missing from the other list is a mismatch only if it beats that
    # list's k-th probability by more than rounding could explain (2 * tol).
    # Anything closer is a near-tie at the cut, and either order is correct.
    out = []
    for layer, (sl, rl) in enumerate(zip(stored, rebuilt)):
        for t, (ss, rs) in enumerate(zip(sl, rl)):
            for mine, other, who in ((ss, rs, "stored"), (rs, ss, "rebuilt")):
                floor = min((p for _, p in other), default=0.0)
                theirs = {tok for tok, _ in other}
                out += [f"run {lab} top-k [{layer}, {t}]: {who} has {tok!r} at p={p:.6g}; "
                        f"the other's k-th p is {floor:.6g}"
                        for tok, p in mine if p > floor + 2 * tol and tok not in theirs]
    return out


def compare(stored: dict, rebuilt: dict, tol: dict[str, float]) -> list[str]:
    """Every way `rebuilt` differs from `stored` beyond `tol`, as messages."""
    if len(stored["runs"]) != len(rebuilt["runs"]):
        return [f"runs: stored {len(stored['runs'])}, rebuilt {len(rebuilt['runs'])}"]
    bad = []
    for s, r in zip(stored["runs"], rebuilt["runs"]):
        lab = s["label"]
        diff = [f"{i}: {a!r} vs {b!r}" for i, (a, b)
                in enumerate(zip_longest(s["tokens"], r["tokens"])) if a != b]
        if diff:
            bad.append(f"run {lab} tokens (stored vs rebuilt) differ at " + "; ".join(diff[:8]))
        ids = [[st.get("id") for st in (x.get("generation") or {}).get("steps", [])]
               for x in (s, r)]
        if ids[0] != ids[1]:
            bad.append(f"run {lab} decoded token ids: stored {ids[0]}, rebuilt {ids[1]}")
        if s.get("topk") is not None and r.get("topk") is not None:
            misses = _topk_misses(lab, s["topk"], r["topk"], tol.get(f"run {lab} top-k p", 0.0))
            # one wrong weight moves every readout; three say which way
            bad += misses[:3] + ([f"run {lab} top-k: {len(misses) - 3} more like these"]
                                 if len(misses) > 3 else [])
    signs, why = axis_signs(stored, rebuilt)
    if why:
        bad.append(why)
        signs = np.ones(2)
    gaps, problems = _gaps(_numbers(stored), _numbers(flipped(rebuilt, signs)))
    bad += problems
    for name, (d, _, where) in gaps.items():
        if why and name in FRAME_DEPENDENT:
            continue
        if name not in tol:
            bad.append(f"{name}: no measured tolerance")
        elif d > tol[name]:
            bad.append(f"{name}: max |stored - rebuilt| = {d:.3g} at {list(where)} "
                       f"exceeds the tolerance {tol[name]:.3g}")
    return bad


def check_reconstructs(scene: dict, load) -> str:
    """Re-run `scene`, compare, fail with every mismatch; returns a report."""
    rebuilt, wide = reconstruct(scene, load)
    tol = tolerances(rebuilt, wide)
    bad = compare(scene, rebuilt, tol)
    signs, _ = axis_signs(scene, rebuilt)
    gaps, _ = _gaps(_numbers(scene), _numbers(flipped(rebuilt, signs if signs is not None
                                                      else np.ones(2))))
    m = scene["analysis"]["models"][0]
    lines = [f"model {m['id']} @ {m['revision']} ({m['dtype']}, lens {m['lens']}); "
             f"axis signs {signs}",
             f"{'quantity':<24} {'|stored-rebuilt|':>17} {'tolerance':>11}"]
    lines += [f"{n:<24} {gaps[n][0]:>17.3g} {tol[n]:>11.3g}" for n in sorted(gaps) if n in tol]
    report = "\n".join(lines)
    assert not bad, "scene does not reconstruct:\n" + "\n".join(bad) + "\n\n" + report
    return report


# ---------------------------------------------------- offline: tiny model
TINY_PROMPTS = ["the capital of france is", "the capital of germany is"]


def _load_local(model_id, revision, dtype):
    from capture import load_model

    assert revision is None and _is_local_dir(model_id), (model_id, revision)
    return load_model(model_id, device="cpu", dtype=dtype)


@pytest.fixture(scope="module")
def tiny_scene(tmp_path_factory):
    import cli
    import tiny

    model_dir = tiny.save_to(tmp_path_factory.mktemp("tiny") / "model")
    out = tmp_path_factory.mktemp("scene") / "tiny.mtj"
    assert cli.main(["export", *TINY_PROMPTS, "--model", model_dir, "-o", str(out)]) == 0
    return statefile.load_scene(out)


@pytest.fixture(scope="module")
def tiny_rebuild(tiny_scene):
    rebuilt, wide = reconstruct(tiny_scene, _load_local)
    return rebuilt, tolerances(rebuilt, wide)


def test_an_exported_scene_reconstructs_from_its_own_record(tiny_scene):
    assert missing_fields(tiny_scene) == []
    report = check_reconstructs(tiny_scene, _load_local)
    assert "pairwise distances" in report and "terrain z" in report


def test_a_perturbed_scene_fails_and_names_what_moved(tiny_scene, tiny_rebuild):
    rebuilt, tol = tiny_rebuild
    assert compare(tiny_scene, rebuilt, tol) == []

    def perturbed(amount):
        # Moving one point's x moves its distances too, so that edit is sized
        # against both tolerances it touches.
        s = copy.deepcopy(tiny_scene)
        s["runs"][0]["points"] = np.array(s["runs"][0]["points"], np.float64)
        s["runs"][0]["points"][2, 3, 0] += amount("xy", "pairwise distances")
        s["runs"][1]["entropy"] = np.array(s["runs"][1]["entropy"], np.float64)
        s["runs"][1]["entropy"][1, 2] += amount("run B entropy")
        s["terrain"]["z"] = np.array(s["terrain"]["z"], np.float64)
        s["terrain"]["z"][10, 20] += amount("terrain z")
        return s

    def under(*names):
        return 0.5 * min(tol[n] for n in names)

    def over(*names):
        return 10.0 * max(tol[n] for n in names)

    # Below the tolerance, a change is indistinguishable from rounding.
    assert compare(perturbed(under), rebuilt, tol) == []
    # Ten times past it, each moved quantity is named, with where it moved.
    bad = "\n".join(compare(perturbed(over), rebuilt, tol))
    assert "xy: max |stored - rebuilt|" in bad and "pairwise distances" in bad
    assert "run B entropy: max |stored - rebuilt|" in bad and "at [1, 2]" in bad
    assert "terrain z: max |stored - rebuilt|" in bad and "at [10, 20]" in bad
    assert "run A entropy" not in bad and "tokens" not in bad

    wrong = copy.deepcopy(tiny_scene)
    wrong["runs"][1]["tokens"][3] = "italy"
    assert compare(wrong, rebuilt, tol) == [
        "run B tokens (stored vs rebuilt) differ at 3: 'italy' vs 'germany'"]


def test_an_axis_flip_is_aligned_not_reported(tiny_scene, tiny_rebuild):
    rebuilt, tol = tiny_rebuild
    mirrored = flipped(tiny_scene, np.array([-1.0, 1.0]))
    assert axis_signs(mirrored, rebuilt)[0].tolist() == [-1.0, 1.0]
    assert compare(mirrored, rebuilt, tol) == []
    # A rotation is not a flip: refused rather than aligned away.
    rot = copy.deepcopy(tiny_scene)
    c, s = np.cos(0.3), np.sin(0.3)
    for run in rot["runs"]:
        p = np.array(run["points"], np.float64)
        p[..., :2] = p[..., :2] @ np.array([[c, -s], [s, c]])
        run["points"] = p
    assert any("more than a sign flip" in m for m in compare(rot, rebuilt, tol))


def test_a_missing_record_field_is_named_not_defaulted(tiny_scene):
    partial = copy.deepcopy(tiny_scene)
    del partial["analysis"]["config"]["seed"]
    partial["analysis"]["models"][1]["dtype"] = None
    assert missing_fields(partial) == ["analysis.config.seed", "analysis.models[1].dtype"]
    # The bundled samples predate the record: none carries one, so none can
    # be re-run, and their fidelity to a real capture stays unchecked.
    single = statefile.load_scene(SAMPLES / "single.mtj")
    assert missing_fields(single) == [
        "analysis.config", "analysis.prompts", "analysis.models[0].id",
        "analysis.models[0].revision", "analysis.models[0].dtype",
        "analysis.models[0].lens", "analysis.models[0].backend"]


# ----------------------------------------------------- network: real GPT-2
GPT2_PROMPTS = ["The capital of France is", "The capital of Germany is"]


def _load_hub(model_id, revision, dtype):
    """`capture.load_model`, pinned to the recorded revision (it takes none):
    the same Auto classes and dtype argument, eval mode, CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=getattr(torch, dtype)).eval()
    served = getattr(model.config, "_commit_hash", None)
    if served is None:
        # transformers >= 5.20 keeps no commit on the config; what it fetched
        # for `revision` sits in the cache under the snapshot it resolved to
        from pathlib import Path

        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(model_id, "config.json", revision=revision)
        if isinstance(hit, str):
            served = Path(hit).parent.name
    assert served == revision, f"asked the hub for {model_id}@{revision}, got {served}"
    return model, tokenizer


@pytest.mark.network
def test_fresh_gpt2_export_reconstructs_from_its_own_record(tmp_path):
    # The scene `mottled export` writes today, re-run from nothing but its
    # record. This checks that the record is enough to rebuild the scene, and
    # that the hub revision it names serves the same weights.
    import cli

    out = tmp_path / "gpt2.mtj"
    assert cli.main(["export", *GPT2_PROMPTS, "--model", "gpt2", "-o", str(out)]) == 0
    scene = statefile.load_scene(out)
    assert missing_fields(scene) == []
    print(check_reconstructs(scene, _load_hub))
