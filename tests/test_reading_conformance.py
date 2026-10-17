"""Fidelity conformance: the explorer and the viewer report one number.

`projection.fidelity_summary` is the reference; `viewer/reading.js` ports it
so a shared scene can state its own trustworthiness in a browser with no
Python. The two must agree, or the same run has two fidelities depending on
which surface you opened it in — and a screenshot from either says nothing
about the other.

Pinned the way `test_scene_conformance.py` pins the scene pipeline: run both
implementations over identical data and compare numerically.
"""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import projection as P

ROOT = Path(__file__).resolve().parent.parent
READING_JS = ROOT / "viewer" / "reading.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not available")

_SCRIPT = """
const R = require(process.argv[1]);
let raw = "";
process.stdin.on("data", (d) => (raw += d));
process.stdin.on("end", () => {
  const input = JSON.parse(raw);
  process.stdout.write(JSON.stringify(
    R.fidelitySummary(input.preservation, undefined, input.chance)));
});
"""


def _js_summary(preservation, chance=None) -> dict:
    proc = subprocess.run(["node", "-e", _SCRIPT, str(READING_JS)],
                          input=json.dumps({"preservation": list(preservation),
                                            "chance": chance}),
                          capture_output=True, text=True, check=True)
    return json.loads(proc.stdout)


@pytest.mark.parametrize("chance", [None, 5 / 11, 5 / 6])
@pytest.mark.parametrize("values", [
    [0.9, 0.2, 0.8, 0.1],                       # a plain mix
    [1.0, 1.0, 1.0],                            # nothing lost
    [0.0, 0.0],                                 # everything lost
    [0.5, 0.4999999, 0.5000001],                # right at the threshold
    list(np.linspace(0.0, 1.0, 101)),           # a spread across the range
])
def test_fidelity_summary_matches_the_python_reference(values, chance):
    want = P.fidelity_summary(np.asarray(values), chance=chance or 0.0)
    got = _js_summary(values, chance)
    assert got["n"] == want["n"]
    assert got["low_fidelity"] == want["low_fidelity"]      # the same double
    assert got["chance"] == want["chance"]
    assert got["mean"] == pytest.approx(want["mean"], abs=1e-12)
    assert got["low_fraction"] == pytest.approx(want["low_fraction"], abs=1e-12)


def test_values_on_the_chance_cut_fall_the_same_way():
    """A state exactly at chance is not flagged on either side (the flag is
    strictly below it), and one ulp under it is, on both."""
    chance = 5 / 11
    cut = P.low_fidelity_cut(chance)
    values = [np.nextafter(cut, 0), cut, np.nextafter(cut, 1)]
    want = P.fidelity_summary(np.asarray(values), chance=chance)
    got = _js_summary(values, chance)
    assert want["low_fraction"] == got["low_fraction"] == pytest.approx(1 / 3)


def test_the_threshold_has_one_home():
    """`LOW_FIDELITY` is mirrored in the port; a drift would silently change
    which states the two surfaces call untrustworthy."""
    script = "const R = require(process.argv[1]); process.stdout.write(String(R.LOW_FIDELITY));"
    proc = subprocess.run(["node", "-e", script, str(READING_JS)],
                          capture_output=True, text=True, check=True)
    assert float(proc.stdout) == P.LOW_FIDELITY


def test_both_sides_refuse_an_empty_summary():
    """Reporting 'fidelity 0.00' for a scene that carries no quality layer
    would be a measurement nobody made."""
    with pytest.raises(ValueError, match="no preservation values"):
        P.fidelity_summary(np.asarray([]))
    proc = subprocess.run(["node", "-e", _SCRIPT, str(READING_JS)],
                          input=json.dumps({"preservation": []}),
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "no preservation values" in proc.stderr


def test_a_real_scenes_quality_layer_summarizes_the_same_on_both_sides():
    """End to end on a scene the pipeline actually produced, not a fixture."""
    import tiny as synthetic
    from config import MarbleConfig
    from ui import run_scene

    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    result = run_scene(cfg, ["the capital of france is"], **synthetic.mt())
    preservation = result["quality_list"][0].preservation

    want = P.fidelity_summary(preservation)
    got = _js_summary(np.asarray(preservation, dtype=float).ravel())
    assert got["mean"] == pytest.approx(want["mean"], abs=1e-9)
    assert got["low_fraction"] == pytest.approx(want["low_fraction"], abs=1e-12)


_SCENE_SCRIPT = """
const M = require(process.argv[1]), R = require(process.argv[2]);
const scene = M.loadScene(require("fs").readFileSync(0));
process.stdout.write(JSON.stringify(scene.runs.map((r) => R.sceneFidelity([r]))));
"""


def test_a_saved_scene_reads_the_explorers_flag_measure_in_the_viewer(tmp_path):
    """End to end: the explorer flags on `projection.fidelity_basis`; the
    viewer, reading the saved scene, must summarize the same values against
    the same cut and name the same measure — within-layer for a run long
    enough to have it, pooled for one that is not."""
    import io

    import statefile
    import tiny as synthetic
    from config import MarbleConfig
    from ui import run_scene

    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    result = run_scene(cfg, ["the capital of france is paris and the capital of germany",
                             "the capital of france is"], **synthetic.mt())
    buf = io.BytesIO()
    statefile.save_scene(result, buf)
    proc = subprocess.run(["node", "-e", _SCENE_SCRIPT, str(ROOT / "viewer" / "mtj.js"),
                           str(READING_JS)], input=buf.getvalue(),
                          capture_output=True, check=True)
    got = json.loads(proc.stdout)

    metrics = []
    for js, q in zip(got, result["quality_list"]):
        basis = P.fidelity_basis(q)
        want = P.fidelity_summary(basis.values, chance=basis.chance)
        assert js["metric"] == basis.metric
        assert js["n"] == want["n"]
        assert js["mean"] == pytest.approx(want["mean"], abs=1e-6)   # float32 on disk
        assert js["low_fraction"] == pytest.approx(want["low_fraction"], abs=1e-12)
        assert js["low_fidelity"] == pytest.approx([want["low_fidelity"]] * 2, abs=1e-12)
        metrics.append(basis.metric)
    assert metrics == ["within-layer", "pooled"]
