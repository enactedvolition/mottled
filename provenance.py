"""The analysis record: what a scene was made with, machine-readable.

`docs/validity.md` asks anyone publishing on Mottled output to version-lock
model weights, tokenizer, library versions, precision, seeds and SAE artifact
hashes. That was a norm without an affordance — the knobs lived in a
`MarbleConfig` the user had to transcribe by hand, and the environment facts
lived nowhere at all, so a shared `.mtj` could not say what produced it.

`record()` collects the lot in one call and `statefile` carries the result in
the `.mtj` manifest under the additive key `"analysis"`, so a scene arrives
with its own methods section:

    rec = provenance.record(cfg, prompts=prompts, trajs=[traj])
    json.dumps(rec)                       # citable, timestamped, diffable

What this is not: evidence that the run reproduces. It is the
parameterization a reproduction *attempt* needs — the same distinction
`sae.fit_report` draws between provenance and calibration.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "mottled-analysis/1"

# Packages whose version can move a number in a scene: the forward pass, the
# projection, the density estimate, the neighbor search, the weight load.
# Deliberately not a `pip freeze` — an environment dump buries the handful of
# libraries that actually decide what the picture shows.
_PACKAGES = ("numpy", "scipy", "scikit-learn", "torch", "transformers",
             "tokenizers", "umap-learn", "faiss-cpu", "safetensors")


def record(cfg, prompts=None, trajs=None, sae=None,
           sae_source: str | None = None, sae_hook: str | None = None) -> dict:
    """The full analysis parameterization of one run or scene, JSON-safe.

    `cfg` is the `MarbleConfig` that drove the pipeline; `trajs` the captured
    trajectories (one per run), whose `meta` supplies the facts only the
    capture knows — the resolved device and dtype, the hub commit the
    weights came from, and the edits a counterfactual run was made under.
    `sae` attaches the dictionary's content hash.
    """
    metas = [dict(getattr(t, "meta", None) or {}) for t in (trajs or [])]
    if prompts is None:
        prompts = [m["prompt"] for m in metas if "prompt" in m]
    rec = {
        "schema": SCHEMA,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds")
                                             .replace("+00:00", "Z"),
        "mottled": _version(),
        "config": cfg.to_dict(),
        "prompts": list(prompts),
        # a list, not a field: a cross-model scene (pipeline.run_model_scene)
        # has one identity per run, and recording only run 0's would name the
        # wrong weights for every other one
        "models": [_model(m, cfg) for m in metas] or [_model({}, cfg)],
        "environment": environment(),
    }
    if sae is not None or sae_source is not None:
        rec["sae"] = {
            "source": sae_source,
            "hook": sae_hook,
            "sha256": None if sae is None else sae_digest(sae),
            "n_features": None if sae is None else int(sae.n_features),
        }
    # per run, in run order, as `models` is: a counterfactual run
    # (intervene.intervene) is the prompt under these edits, and a record
    # without them describes an untouched run instead of the one drawn
    edits = [list(m.get("interventions") or []) for m in metas]
    if any(edits):
        rec["interventions"] = edits
    return rec


def sae_digest(sae) -> str:
    """Content hash of a dictionary's weights.

    Hashes the arrays, not the file they arrived in: the same dictionary
    reaches Mottled as safetensors from the hub, an `.npz`, or a SAELens
    object, and only the numbers decide what the feature layer shows. Shapes
    go into the hash too, so a reshape cannot collide with the original.
    """
    h = hashlib.sha256()
    for name in ("w_enc", "b_enc", "w_dec", "b_dec"):
        arr = np.ascontiguousarray(getattr(sae, name), dtype="<f4")
        h.update(name.encode("ascii"))
        h.update(repr(arr.shape).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _model(meta: dict, cfg) -> dict:
    """One run's model identity, from the capture's own report of it."""
    return {
        # `revision` is the hub commit the weights resolved to; None when they
        # did not come from a snapshot (a local path, a model built in-process)
        "id": meta.get("model") or cfg.model,
        "revision": meta.get("revision"),
        "backend": meta.get("backend"),
        "family": meta.get("family"),
        "device": meta.get("device"),
        "dtype": meta.get("dtype"),
        # the readout the scene's logits, entropy and top-k came from; absent
        # for a backend that does not report it
        "lens": meta.get("lens"),
    }


def environment() -> dict:
    """Python, platform, and the versions that can move a number in a scene.

    Public because `parity.py` reports the same block: a parity number is only
    checkable against the environment that produced it.
    """
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: found for name in _PACKAGES
                     if (found := _package_version(name)) is not None},
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version() -> str | None:
    """Mottled's own version — installed metadata first, then pyproject.

    The reproducibility case is usually a clone on `sys.path` rather than an
    install, and a record that omits the tool's version there is exactly the
    gap this module exists to close.
    """
    if (found := _package_version("mottled")) is not None:
        return found
    try:
        import tomllib

        pyproject = Path(__file__).resolve().parent / "pyproject.toml"
        return tomllib.loads(pyproject.read_text())["project"]["version"]
    except (OSError, KeyError, ValueError):
        return None
