"""`.mtj` conformance: the Python writer and the JavaScript viewer agree.

test_statefile.py already covers the Python round-trip; this pins the
cross-language contract the format exists for — a scene written by
statefile.save_scene must decode identically in viewer/mtj.js — plus the
little-endian header the spec (docs/mtj-format.md) promises.
"""
import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

import statefile
from config import MarbleConfig
from ui import run_scene
import tiny as synthetic

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ["the capital of france is", "the capital of germany is"]


def _scene(tmp_path) -> Path:
    cfg = MarbleConfig(model="tiny", use_cache=False)
    result = run_scene(cfg, PROMPTS, **synthetic.mt())
    path = tmp_path / "scene.mtj"
    statefile.save_scene(result, path)
    return path


def test_header_is_little_endian_and_versioned(tmp_path):
    raw = _scene(tmp_path).read_bytes()
    assert raw[:4] == statefile.MAGIC == b"MTRJ"
    version, mlen = struct.unpack_from("<II", raw, 4)
    assert version == statefile.VERSION == 1
    assert 12 + mlen <= len(raw)
    # the same header read big-endian is a different number -> it really is LE
    assert struct.unpack_from(">II", raw, 4)[0] != version


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_python_writer_decodes_in_js_viewer(tmp_path):
    path = _scene(tmp_path)

    # Read the Python-written scene with the actual viewer parser (viewer/mtj.js).
    script = r"""
      const MTJ = require(process.argv[1]);
      const fs = require("fs");
      const scene = MTJ.loadScene(fs.readFileSync(process.argv[2]));
      const r0 = scene.runs[0];
      let sum = 0; for (const v of r0.points.data) sum += v;
      process.stdout.write(JSON.stringify({
        kind: scene.manifest.kind,
        labels: scene.runs.map((r) => r.label),
        zshape: scene.terrain.z.shape,
        p0shape: r0.points.shape,
        p0sum: sum,
        entshape: r0.entropy ? r0.entropy.shape : null,
      }));
    """
    proc = subprocess.run(
        ["node", "-e", script, str(ROOT / "viewer" / "mtj.js"), str(path)],
        capture_output=True, text=True, check=True)
    js = json.loads(proc.stdout)

    py = statefile.load_scene(path)   # Python's view of the same bytes
    assert js["kind"] == "scene"
    assert js["labels"] == [r["label"] for r in py["runs"]]
    assert list(js["zshape"]) == list(py["terrain"]["z"].shape)

    pts0 = py["runs"][0]["points"]
    assert list(js["p0shape"]) == list(pts0.shape)          # same geometry
    assert js["p0sum"] == pytest.approx(float(pts0.sum()), rel=1e-3)  # same bytes
    assert list(js["entshape"]) == list(py["runs"][0]["entropy"].shape)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_blast_record_decodes_in_js_viewer(tmp_path):
    """The blast record (blast.py) reads the same in the viewer as in Python:
    every layout, its positions and arrays, and the pellets."""
    import numpy as np

    import blast
    from trajectory import StateTrajectory

    rng = np.random.default_rng(0)
    h = np.cumsum(rng.normal(size=(5, 12, 16)), axis=0).astype(np.float32)
    h[0] = h[0, 0]                                       # the muzzle
    pel = [{"id": f"p{i}", "text": f"prompt {i}", "labels": {"c": i % 2}} for i in range(12)]
    t = StateTrajectory(hidden=h, tokens=[p["id"] for p in pel],
                        meta={"axis": "pellets", "pellets": pel})
    built, skipped = blast.layouts(t, null_draws=10)
    path = tmp_path / "blast.mtj"
    statefile.save_scene(blast.scene(t, built, skipped), path)

    script = r"""
      const MTJ = require(process.argv[1]);
      const b = MTJ.loadScene(require("fs").readFileSync(process.argv[2])).blast;
      const sum = (a) => { let s = 0; for (const v of a.data) if (!Number.isNaN(v)) s += v; return s; };
      process.stdout.write(JSON.stringify({
        names: b.layouts.map((l) => l.name), drivers: b.layouts.map((l) => l.driver),
        psum: b.layouts.map((l) => sum(l.positions)), pshape: b.layouts[0].positions.shape,
        auroc: sum(b.layouts[0].arrays.auroc), qshape: b.layouts[0].quality.shape,
        fitted: b.layouts[0].fitted, ids: b.pellets.map((p) => p.id),
        labels: b.pellets[3].labels, spread: sum(b.spread), rshape: b.range.shape,
      }));
    """
    js = json.loads(subprocess.run(
        ["node", "-e", script, str(ROOT / "viewer" / "mtj.js"), str(path)],
        capture_output=True, text=True, check=True).stdout)

    py = statefile.load_scene(path)["blast"]
    assert js["names"] == [b.name for b in built] and js["drivers"] == ["c", None]
    assert list(js["pshape"]) == [12, 5, 2] and list(js["qshape"]) == [5, 12]
    assert js["psum"] == pytest.approx([float(x["positions"].sum()) for x in py["layouts"]],
                                       rel=1e-5, abs=1e-5)
    assert js["auroc"] == pytest.approx(float(np.nansum(py["layouts"][0]["arrays"]["auroc"])),
                                        rel=1e-5)
    assert js["fitted"] == built[0].fitted
    assert js["ids"] == [p["id"] for p in pel] and js["labels"] == {"c": 1}
    assert js["spread"] == pytest.approx(float(py["spread"].sum()), rel=1e-5)
    assert list(js["rshape"]) == [12, 5]
