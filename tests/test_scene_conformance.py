"""Scene conformance: the Python pipeline and the JS viewer build one scene.

`projection.py` / `density.py` / `terrain.py` are the reference
implementations; the browser builds a scene with `viewer/scene.js` when the
model runs in the page and there is no server to ask. A scene assembled in the
browser has to *be* the scene Python would have assembled — otherwise the two
surfaces are two tools, and a screenshot from one says nothing about the other.

Pinned the same way `test_bvh_conformance.py` pins picking: run both
implementations over identical data and compare numerically.

The tolerances are stated per-stage rather than globally, because the stages
lose precision differently: PCA is exact up to component sign (fixed to
scikit-learn's `svd_flip` convention on both sides), while the terrain is a
smoothed field whose absolute scale is normalised away.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from density import compute_density
from projection import project_joint
from terrain import drape, mesh

ROOT = Path(__file__).resolve().parent.parent
SCENE_JS = ROOT / "viewer" / "scene.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not available")


def _run_js(script: str, payload: dict) -> dict:
    proc = subprocess.run(["node", "-e", script, str(SCENE_JS)],
                          input=json.dumps(payload), capture_output=True,
                          text=True, check=True)
    return json.loads(proc.stdout)


def _hidden(n_layers=6, n_tokens=5, dim=24, seed=0):
    """A smooth, non-degenerate state cloud: momentum walk per token."""
    rng = np.random.default_rng(seed)
    h = np.zeros((n_layers, n_tokens, dim), dtype=np.float32)
    h[0] = rng.normal(size=(n_tokens, dim))
    vel = np.zeros((n_tokens, dim), dtype=np.float32)
    for l in range(1, n_layers):
        vel = 0.7 * vel + 0.2 * rng.normal(size=(n_tokens, dim))
        h[l] = h[l - 1] + vel
    return h


# Several seeds, because a sign rule is a per-input coin flip: with one
# fixture the two sides once disagreed on 12 of 20 inputs while this test
# passed on the one where they happened to agree.
@pytest.mark.parametrize("seed", range(8))
def test_pca_matches_python(seed):
    """Projected coordinates agree, including component signs."""
    h = _hidden(seed=seed)
    coords, projector = project_joint([h])
    coords = coords[0].reshape(-1, 2)

    # The reference's own rule, pinned rather than inherited from whichever
    # scikit-learn is installed: each component's largest-|.| entry positive.
    comps = projector._pca.components_
    assert (comps[np.arange(len(comps)), np.abs(comps).argmax(axis=1)] > 0).all()

    flat = h.reshape(-1, h.shape[-1])
    js = _run_js(
        """
        const S = require(process.argv[1]);
        let raw = ""; process.stdin.on("data", d => raw += d).on("end", () => {
          const p = JSON.parse(raw);
          const out = S.pca(Float64Array.from(p.hidden), p.n, p.d, 2);
          process.stdout.write(JSON.stringify({
            coords: Array.from(out.coords),
            explainedVariance: out.explainedVariance,
          }));
        });
        """,
        {"hidden": flat.astype(np.float64).ravel().tolist(),
         "n": int(flat.shape[0]), "d": int(flat.shape[1])},
    )

    got = np.asarray(js["coords"], dtype=np.float64).reshape(-1, 2)
    # Sign convention is pinned on both sides, so this is a direct comparison
    # rather than a compare-up-to-sign: a mirrored scene is a failure.
    assert np.allclose(got, coords, atol=1e-6), \
        f"max |Δ| = {np.abs(got - coords).max():.3e}"
    assert js["explainedVariance"] == pytest.approx(
        projector.explained_variance, abs=1e-9)


def test_density_and_terrain_match_python():
    """The density field and the terrain height map agree cell for cell."""
    h = _hidden(seed=3)
    coords, _ = project_joint([h])
    coords = coords[0].reshape(-1, 2).astype(np.float64)

    land = compute_density(coords, method="kde", grid_size=24)
    surface = mesh(land)

    js = _run_js(
        """
        const S = require(process.argv[1]);
        let raw = ""; process.stdin.on("data", d => raw += d).on("end", () => {
          const p = JSON.parse(raw);
          const c = Float64Array.from(p.coords);
          const land = S.computeDensity(c, p.n, { gridSize: p.gridSize });
          const z = S.heightMap(land.density, land.gridSize, {});
          process.stdout.write(JSON.stringify({
            gridX: Array.from(land.gridX), gridY: Array.from(land.gridY),
            density: Array.from(land.density), z: Array.from(z),
          }));
        });
        """,
        {"coords": coords.ravel().tolist(), "n": int(len(coords)), "gridSize": 24},
    )

    assert np.allclose(js["gridX"], land.grid_x, atol=1e-12)
    assert np.allclose(js["gridY"], land.grid_y, atol=1e-12)

    got_d = np.asarray(js["density"]).reshape(land.density.shape)
    assert np.allclose(got_d, land.density, atol=1e-9), \
        f"density max |Δ| = {np.abs(got_d - land.density).max():.3e}"

    got_z = np.asarray(js["z"]).reshape(surface.z.shape)
    assert np.allclose(got_z, surface.z, atol=1e-6), \
        f"terrain max |Δ| = {np.abs(got_z - surface.z).max():.3e}"


def test_drape_matches_python():
    """Trajectories land at the same height on the same terrain."""
    h = _hidden(seed=5)
    coords, _ = project_joint([h])
    coords = coords[0].reshape(-1, 2).astype(np.float64)
    land = compute_density(coords, method="kde", grid_size=24)
    surface = mesh(land)
    expected = drape(surface, coords)

    js = _run_js(
        """
        const S = require(process.argv[1]);
        let raw = ""; process.stdin.on("data", d => raw += d).on("end", () => {
          const p = JSON.parse(raw);
          const c = Float64Array.from(p.coords);
          const land = S.computeDensity(c, p.n, { gridSize: p.gridSize });
          const z = S.heightMap(land.density, land.gridSize, {});
          const terrain = { gridX: land.gridX, gridY: land.gridY, z,
                            gridSize: land.gridSize };
          process.stdout.write(JSON.stringify({
            draped: Array.from(S.drape(terrain, c, p.n)),
          }));
        });
        """,
        {"coords": coords.ravel().tolist(), "n": int(len(coords)), "gridSize": 24},
    )

    got = np.asarray(js["draped"]).reshape(-1, 3)
    assert np.allclose(got, expected, atol=1e-6), \
        f"drape max |Δ| = {np.abs(got - expected).max():.3e}"


def test_entropy_and_topk_match_python():
    """The inspector's readout is the same softmax on both sides."""
    rng = np.random.default_rng(7)
    logits = rng.normal(size=32) * 3.0

    x = logits - logits.max()
    p = np.exp(x)
    p /= p.sum()
    entropy = float(-(p * np.log(np.where(p > 0, p, 1.0))).sum())
    order = np.argsort(-p)[:5]

    js = _run_js(
        """
        const S = require(process.argv[1]);
        let raw = ""; process.stdin.on("data", d => raw += d).on("end", () => {
          const p = JSON.parse(raw);
          const r = S.entropyAndTopK(Float64Array.from(p.logits), 5, null);
          process.stdout.write(JSON.stringify(
            { entropy: r.entropy, topk: r.topk }));
        });
        """,
        {"logits": logits.tolist()},
    )

    assert js["entropy"] == pytest.approx(entropy, abs=1e-12)
    assert [i for i, _ in js["topk"]] == order.tolist()
    assert np.allclose([v for _, v in js["topk"]], p[order], atol=1e-12)


def test_scene_assembly_shares_one_projection():
    """Two runs assembled in JS share the space Python's project_joint fits."""
    a, b = _hidden(seed=11), _hidden(seed=12)
    (ca, cb), _ = project_joint([a, b])

    js = _run_js(
        """
        const S = require(process.argv[1]);
        let raw = ""; process.stdin.on("data", d => raw += d).on("end", () => {
          const p = JSON.parse(raw);
          const mk = (r) => ({ hidden: Float64Array.from(r.hidden),
                               nLayers: r.nLayers, nTokens: r.nTokens,
                               dim: r.dim, tokens: r.tokens, model: "test" });
          const scene = S.buildScene(p.runs.map(mk), { gridSize: 24 });
          process.stdout.write(JSON.stringify({
            coords: scene.runs.map(r => Array.from(r.coords)),
            nRuns: scene.runs.length,
          }));
        });
        """,
        {"runs": [
            {"hidden": r.astype(np.float64).ravel().tolist(),
             "nLayers": int(r.shape[0]), "nTokens": int(r.shape[1]),
             "dim": int(r.shape[2]), "tokens": ["t"] * int(r.shape[1])}
            for r in (a, b)
        ]},
    )

    assert js["nRuns"] == 2
    got_a = np.asarray(js["coords"][0]).reshape(ca.reshape(-1, 2).shape)
    got_b = np.asarray(js["coords"][1]).reshape(cb.reshape(-1, 2).shape)
    assert np.allclose(got_a, ca.reshape(-1, 2), atol=1e-6)
    assert np.allclose(got_b, cb.reshape(-1, 2), atol=1e-6)
