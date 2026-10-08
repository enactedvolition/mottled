"""`.mtj` — the StateTrajectory interchange format (see docs/mtj-format.md).

StateTrajectory is the center of Mottled: producers emit one, viewers and
analyses consume one.  This module is the stable boundary between the two —
a versioned binary container (JSON manifest + raw little-endian buffers,
glb-style) that any language can parse with nothing but its standard
library.

    save(traj, "run.mtj")            # full-fidelity StateTrajectory
    traj = load("run.mtj")

    save_scene(result, "scene.mtj")  # viewer-ready bundle from run_pipeline /
    scene = load_scene("scene.mtj")  # run_scene / run_compare / run_intervention

Python stays responsible for capture and analysis; scene files carry the
finished artifacts (projected + draped trajectories, terrain, inspector
stats) so a viewer — the web viewer in viewer/, a notebook, a future desktop
app — only has to draw.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from trajectory import StateTrajectory

MAGIC = b"MTRJ"
VERSION = 1
_ALIGN = 16
_DTYPES = {"float16": np.float16, "float32": np.float32, "int32": np.int32}


# --------------------------------------------------------------- container
class _Writer:
    """Accumulates named arrays into an aligned blob + manifest references."""

    def __init__(self):
        self.chunks: list[bytes] = []
        self.refs: dict[str, dict] = {}
        self._offset = 0

    def add(self, name: str, array: np.ndarray) -> str:
        arr = np.ascontiguousarray(array)
        if arr.dtype.name not in _DTYPES:
            arr = arr.astype(np.float32)
        data = arr.astype(arr.dtype.newbyteorder("<")).tobytes()
        pad = (-self._offset) % _ALIGN
        if pad:
            self.chunks.append(b"\x00" * pad)
            self._offset += pad
        self.refs[name] = {"dtype": arr.dtype.name, "shape": list(arr.shape),
                           "offset": self._offset, "length": len(data)}
        self.chunks.append(data)
        self._offset += len(data)
        return name

    def write(self, fh: BinaryIO, manifest: dict) -> None:
        manifest = dict(manifest)
        manifest["arrays"] = self.refs
        blob = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
        pad = (-(12 + len(blob))) % _ALIGN
        blob += b" " * pad
        fh.write(MAGIC)
        fh.write(struct.pack("<II", VERSION, len(blob)))
        fh.write(blob)
        for chunk in self.chunks:
            fh.write(chunk)


def _write(path_or_fh, manifest: dict, writer: _Writer) -> None:
    if hasattr(path_or_fh, "write"):
        writer.write(path_or_fh, manifest)
    else:
        with Path(path_or_fh).open("wb") as fh:
            writer.write(fh, manifest)


def read_container(path_or_fh) -> tuple[dict, dict[str, np.ndarray]]:
    """Parse any .mtj file: (manifest, {array name -> numpy array})."""
    if hasattr(path_or_fh, "read"):
        raw = path_or_fh.read()
    else:
        raw = Path(path_or_fh).read_bytes()
    if raw[:4] != MAGIC:
        raise ValueError("not a .mtj file (bad magic)")
    version, mlen = struct.unpack_from("<II", raw, 4)
    if version != VERSION:
        raise ValueError(f"unsupported .mtj version {version} (reader supports {VERSION})")
    manifest = json.loads(raw[12 : 12 + mlen].decode("utf-8"))
    blob = raw[12 + mlen :]
    arrays = {}
    for name, ref in manifest.get("arrays", {}).items():
        dtype = _DTYPES.get(ref["dtype"])
        if dtype is None:  # unknown dtype from a newer writer: skip, stay stable
            continue
        buf = blob[ref["offset"] : ref["offset"] + ref["length"]]
        arrays[name] = np.frombuffer(buf, dtype=np.dtype(dtype).newbyteorder("<")) \
            .reshape(ref["shape"]).astype(dtype)
    return manifest, arrays


# ------------------------------------------------------------- trajectory IO
def save(traj: StateTrajectory, path_or_fh, include_logits: bool = True,
         include_embeddings: bool = True, analysis: dict | None = None) -> None:
    """Serialize one StateTrajectory at full fidelity (kind: "trajectory").

    `include_logits` / `include_embeddings` drop the two largest optional
    arrays for smaller files; everything else always round-trips. `analysis`
    is an analysis record (`provenance.record`) to carry with the file — the
    archival counterpart of what `save_scene` writes for a shared scene.
    """
    w = _Writer()
    w.add("hidden", traj.hidden.astype(np.float32))
    if traj.entropy is not None:
        w.add("entropy", traj.entropy.astype(np.float32))
    if traj.logits is not None and include_logits:
        w.add("logits", traj.logits.astype(np.float16))
    if traj.attention is not None:
        w.add("attention", traj.attention.astype(np.float32))
    if traj.components is not None:
        for name, arr in traj.components.items():
            w.add(f"components.{name}", arr.astype(np.float32))
    if traj.embedding_matrix is not None and include_embeddings:
        w.add("embedding_matrix", traj.embedding_matrix.astype(np.float32))

    manifest: dict[str, Any] = {
        "format": "mottled-trajectory",
        "version": VERSION,
        "kind": "trajectory",
        "meta": _jsonable(traj.meta),
        "tokens": list(traj.tokens),
    }
    if traj.vocab is not None:
        manifest["vocab"] = list(traj.vocab)
    if traj.topk is not None:
        manifest["topk"] = [[[[tok, float(p)] for tok, p in state] for state in layer]
                            for layer in traj.topk]
    if analysis:
        manifest["analysis"] = _jsonable(analysis)
    _write(path_or_fh, manifest, w)


def load(path_or_fh) -> StateTrajectory:
    """Read a kind:"trajectory" .mtj back into a StateTrajectory."""
    manifest, arrays = read_container(path_or_fh)
    if manifest.get("kind") != "trajectory":
        raise ValueError(f"expected kind 'trajectory', got {manifest.get('kind')!r}")
    components = {name.split(".", 1)[1]: arr for name, arr in arrays.items()
                  if name.startswith("components.")} or None
    topk = manifest.get("topk")
    if topk is not None:
        topk = [[[(tok, float(p)) for tok, p in state] for state in layer]
                for layer in topk]
    traj = StateTrajectory(
        hidden=arrays["hidden"],
        tokens=list(manifest["tokens"]),
        logits=arrays.get("logits"),
        entropy=arrays.get("entropy"),
        topk=topk,
        vocab=manifest.get("vocab"),
        embedding_matrix=arrays.get("embedding_matrix"),
        components=components,
        attention=arrays.get("attention"),
        meta=manifest.get("meta", {}),
    )
    traj.validate()
    return traj


# ---------------------------------------------------------------- scene IO
def save_scene(result: dict, path_or_fh) -> None:
    """Serialize a pipeline result as a viewer-ready bundle (kind: "scene").

    `result` is a dict from `ui.run_pipeline`, `ui.run_scene`,
    `ui.run_compare`, or `ui.run_intervention`.  The scene carries draped
    trajectory points, the terrain, and per-state inspector data — no hidden
    states — so files stay small enough for the web viewer.
    """
    trajs = result.get("trajs") or [result["traj"]]
    trajectories_list = result.get("trajectories_list") or [result["trajectories"]]
    prompts = result.get("prompts") or [result.get("prompt", "")]
    qualities = result.get("quality_list")
    if qualities is None:
        qualities = [result["quality"]] if result.get("quality") is not None else []
    mesh = result["mesh"]
    landscape = result.get("landscape")

    w = _Writer()
    terrain_refs = {"x": w.add("terrain.x", mesh.x),
                    "y": w.add("terrain.y", mesh.y),
                    "z": w.add("terrain.z", mesh.z)}
    if landscape is not None:
        terrain_refs["density"] = w.add("terrain.density", landscape.density)
        if landscape.density_se is not None:
            terrain_refs["se"] = w.add("terrain.se", landscape.density_se)

    runs = []
    for i, (traj, trajectories) in enumerate(zip(trajs, trajectories_list)):
        run: dict[str, Any] = {
            "label": chr(65 + i),
            "prompt": prompts[i] if i < len(prompts) else "",
            # per-run model identity: the scene-level `meta` describes run 0,
            # which is wrong for a scene whose runs are *different models*
            "model": traj.meta.get("model"),
            "tokens": list(traj.tokens),
            "trajectory_labels": [str(t.label or t.token) for t in trajectories],
            "points": w.add(f"run{i}.points",
                            np.stack([t.points for t in trajectories]).astype(np.float32)),
        }
        if traj.entropy is not None:
            run["entropy"] = w.add(f"run{i}.entropy", traj.entropy.astype(np.float32))
        if i < len(qualities) and qualities[i] is not None:
            run["quality"] = w.add(f"run{i}.quality",
                                   qualities[i].preservation.astype(np.float32))
        if traj.attention is not None:
            run["attention"] = w.add(f"run{i}.attention", traj.attention.astype(np.float32))
        if traj.topk is not None:
            run["topk"] = [[[[tok, float(p)] for tok, p in state] for state in layer]
                           for layer in traj.topk]
        if isinstance(traj.meta.get("generation"), dict):
            # the decode record (prompt/continuation boundary + per-step
            # token, id, p, entropy) — additive, so old readers ignore it
            run["generation"] = _jsonable(traj.meta["generation"])
        insp = (result.get("inspector_list") or [])
        if i < len(insp) and insp[i]:
            # inspector layers the viewer cannot recompute (pipeline.
            # attach_inspector): nearest vocabulary tokens per state, and the
            # attn/MLP share of each block's write — additive
            e, block = insp[i], {}
            if "neighbor_idx" in e:
                block["tokens"] = list(e["neighbor_tokens"])
                block["idx"] = w.add(f"run{i}.inspector.nidx",
                                     np.asarray(e["neighbor_idx"], np.int32))
                block["sim"] = w.add(f"run{i}.inspector.nsim",
                                     np.asarray(e["neighbor_sim"], np.float32))
            if "component_shares" in e:
                block["component_shares"] = w.add(
                    f"run{i}.inspector.shares",
                    np.asarray(e["component_shares"], np.float32))
            if block:
                run["inspector"] = block

        feats = (result.get("features_list") or [])
        if i < len(feats) and isinstance(feats[i], dict):
            # SAE feature layer (ui.attach_features): dominant feature per
            # state plus the dictionary's measured fit — additive
            f = feats[i]
            run["features"] = {
                "source": f.get("source"), "hook": f.get("hook"),
                "best_layer": int(f["best_layer"]),
                "recon_error": w.add(f"run{i}.features.recon_error",
                                     np.asarray(f["recon_error"], np.float32)),
                "top_id": w.add(f"run{i}.features.top_id",
                                np.asarray(f["top_id"], np.int32)),
                "top_act": w.add(f"run{i}.features.top_act",
                                 np.asarray(f["top_act"], np.float32)),
            }
        runs.append(run)

    manifest: dict[str, Any] = {
        "format": "mottled-trajectory",
        "version": VERSION,
        "kind": "scene",
        "meta": _jsonable(trajs[0].meta),
        "terrain": terrain_refs,
        "runs": runs,
    }
    if result.get("analysis"):
        # the analysis record (pipeline.attach_manifest): config, environment,
        # model and dictionary identity, so a scene that outlives the session
        # that made it can still say what it is — additive, old readers ignore it
        manifest["analysis"] = _jsonable(result["analysis"])
    if result.get("blast"):
        # one pellet per prompt (blast.py): the run above is the pellets drawn
        # at the first layout, so a viewer that predates this record still
        # shows them flying out of the muzzle; this record holds every layout
        # and what each one's axes are. Pellet texts and labels are in
        # meta["pellets"]. Additive, old readers ignore it.
        manifest["blast"] = _blast_manifest(result["blast"], w)
    if result.get("comparisons"):
        manifest["comparisons"] = [
            {"label": chr(65 + i), "hausdorff": float(c.hausdorff),
             "dtw_normalized": float(c.dtw.normalized),
             "shared_tokens": int(c.shared_tokens),
             "onset_layer": int(c.onset_layer),
             "readout_changed": None if c.readout_changed is None else int(c.readout_changed)}
            for i, c in enumerate(result["comparisons"], start=1)
        ]
    _write(path_or_fh, manifest, w)


def _blast_manifest(record: dict, w: _Writer) -> dict:
    layouts = []
    for i, b in enumerate(record["layouts"]):
        layouts.append({
            "name": b.name, "method": b.method, "driver": b.driver,
            "positions": w.add(f"blast.{i}.positions", b.positions.astype(np.float32)),
            "quality": w.add(f"blast.{i}.quality", b.quality.astype(np.float32)),
            "exact": list(b.exact), "fitted": list(b.fitted),
            "projected": list(b.projected),
            "arrays": {k: w.add(f"blast.{i}.{k}", v) for k, v in b.arrays.items()},
            "params": _jsonable(b.params),
        })
    return {"schema": "mottled-blast/1", "layouts": layouts,
            "skipped": list(record.get("skipped", [])),
            **{k: w.add(f"blast.{k}", record[k]) for k in ("range", "spread", "norm")}}


def load_scene(path_or_fh) -> dict:
    """Read a kind:"scene" .mtj: the manifest with array references resolved
    to numpy arrays in place ("points", "entropy", "attention", terrain)."""
    manifest, arrays = read_container(path_or_fh)
    if manifest.get("kind") != "scene":
        raise ValueError(f"expected kind 'scene', got {manifest.get('kind')!r}")
    scene = dict(manifest)
    scene["terrain"] = {axis: arrays[name] for axis, name in manifest["terrain"].items()}
    scene["runs"] = [
        {**run, **{key: arrays[run[key]]
                   for key in ("points", "entropy", "attention", "quality")
                   if key in run}}
        for run in manifest["runs"]
    ]
    for run in scene["runs"]:
        if isinstance(run.get("inspector"), dict):
            run["inspector"] = {
                **run["inspector"],
                **{k: arrays[run["inspector"][k]]
                   for k in ("idx", "sim", "component_shares")
                   if run["inspector"].get(k) in arrays},
            }
        if isinstance(run.get("features"), dict):
            run["features"] = {
                **run["features"],
                **{k: arrays[run["features"][k]]
                   for k in ("recon_error", "top_id", "top_act")
                   if run["features"].get(k) in arrays},
            }
    if isinstance(scene.get("blast"), dict):
        b = scene["blast"]
        scene["blast"] = {
            **b, **{k: arrays[b[k]] for k in ("range", "spread", "norm") if b.get(k) in arrays},
            "layouts": [{**lay,
                         **{k: arrays[lay[k]] for k in ("positions", "quality")
                            if lay.get(k) in arrays},
                         "arrays": {k: arrays[v] for k, v in lay.get("arrays", {}).items()
                                    if v in arrays}}
                        for lay in b.get("layouts", [])],
        }
    return scene


def _jsonable(meta: dict) -> dict:
    """Best-effort JSON-safe copy of a meta dict (drop what can't serialize)."""
    out = {}
    for key, value in meta.items():
        try:
            json.dumps(value)
            out[key] = value
        except (TypeError, ValueError):
            out[key] = str(value)
    return out
