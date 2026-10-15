"""`.mtj` schemas and `mottled validate`: every file a writer produces is
strict JSON and passes the schema; every way a file can be broken is named.

The schemas are defined once in `mtjschema.py`; docs/schema/*.json is
generated from it (tests/test_codegen.py checks they agree).
"""
import io
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

import mtjschema
import statefile
import tiny
from config import MarbleConfig
from pipeline import attach_inspector, attach_manifest, project_trajectories

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = sorted((ROOT / "viewer" / "samples").rglob("*.mtj"))


def _errors(raw: bytes, strict=False):
    return [p for p in mtjschema.validate_bytes(raw, strict=strict) if p.level == "error"]


@pytest.fixture(scope="module")
def scene_bytes():
    cfg = MarbleConfig(model="tiny", use_cache=False, density_bootstrap=0)
    trajs = [tiny.capture(p) for p in ("the cat sat on the mat", "hello world")]
    buf = io.BytesIO()
    statefile.save_scene(project_trajectories(cfg, trajs), buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def traj_bytes():
    buf = io.BytesIO()
    statefile.save(tiny.capture("the cat sat"), buf)
    return buf.getvalue()


def _manifest(raw: bytes) -> dict:
    mlen = struct.unpack_from("<I", raw, 8)[0]
    return json.loads(raw[12:12 + mlen])


def _rewrite(raw: bytes, edit) -> bytes:
    """Same blob, edited manifest (re-padded so the blob stays aligned)."""
    mlen = struct.unpack_from("<I", raw, 8)[0]
    manifest = json.loads(raw[12:12 + mlen])
    edit(manifest)
    blob = json.dumps(manifest, allow_nan=True).encode()
    blob += b" " * ((-(12 + len(blob))) % 16)
    return raw[:4] + struct.pack("<II", 1, len(blob)) + blob + raw[12 + mlen:]


# ------------------------------------------------------------ valid files
@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_every_bundled_sample_is_valid(path):
    assert _errors(path.read_bytes()) == []


def test_current_writers_pass_even_strict(scene_bytes, traj_bytes):
    assert mtjschema.validate_bytes(scene_bytes, strict=True) == []
    assert mtjschema.validate_bytes(traj_bytes, strict=True) == []
    assert _manifest(scene_bytes)["schema"] == "mottled-scene/1"
    assert _manifest(traj_bytes)["schema"] == "mottled-trajectory/1"


def test_a_stream_of_containers_validates(traj_bytes):
    assert mtjschema.validate_bytes(traj_bytes * 3, strict=True) == []
    assert len(statefile.read_stream(io.BytesIO(traj_bytes * 3))) == 3


def test_published_schemas_agree_with_a_standard_validator(scene_bytes, traj_bytes):
    """The generated files are real JSON Schema: a third-party validator
    accepts what ours accepts."""
    jsonschema = pytest.importorskip("jsonschema")
    for name, raw in (("mottled-scene-1", scene_bytes), ("mottled-trajectory-1", traj_bytes)):
        schema = json.loads((ROOT / "docs" / "schema" / f"{name}.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(_manifest(raw))
    for path in SAMPLES:
        m = _manifest(path.read_bytes())
        schema = mtjschema.SCHEMAS[mtjschema.KIND_SCHEMA[m["kind"]]]()
        jsonschema.Draft202012Validator(schema).validate(m)


# ------------------------------------------------------- NaN-free writing
def test_non_finite_manifest_numbers_are_written_as_null():
    traj = tiny.capture("the cat sat")
    traj.meta.update(temperature=float("nan"), bound=float("inf"),
                     nested={"x": [1.0, float("-inf")]}, np_value=np.float32("nan"))
    buf = io.BytesIO()
    statefile.save(traj, buf)
    raw = buf.getvalue()
    mlen = struct.unpack_from("<I", raw, 8)[0]
    text = raw[12:12 + mlen].decode()
    assert "NaN" not in text and "Infinity" not in text
    meta = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-JSON {c}"))["meta"]
    assert meta["temperature"] is None and meta["bound"] is None
    assert meta["nested"] == {"x": [1.0, None]} and meta["np_value"] is None
    assert _errors(raw) == []


def test_logprobs_scenes_are_strict_json():
    """The producer that used to put a bare NaN in every scene it made."""
    from models.logprobs import from_logprobs

    steps = [{"token": t, "logprobs": {t: math.log(0.6), " x": math.log(0.3)}}
             for t in (" Paris", ",", " France", " is", " big")]
    traj = from_logprobs(steps, model="api-model", prompt="The capital of France is")
    cfg = MarbleConfig(model="api-model", use_cache=False, density_bootstrap=0, grid_size=16)
    buf = io.BytesIO()
    statefile.save_scene(project_trajectories(cfg, [traj]), buf)
    assert _errors(buf.getvalue(), strict=True) == []


# ------------------------------------------------------------ broken files
def test_nan_in_a_manifest_is_an_error(scene_bytes):
    bad = _rewrite(scene_bytes, lambda m: m["meta"].update(t=float("nan")))
    (err,) = _errors(bad)
    assert "not strict JSON" in err.message and "null" in err.message


@pytest.mark.parametrize("edit, where, needle", [
    (lambda m: m["runs"][0].update(points="run0.nope"), "runs[0].points", "not in `arrays`"),
    (lambda m: m["arrays"]["terrain.x"].update(offset=3), "arrays['terrain.x']", "multiple of 16"),
    (lambda m: m["arrays"]["terrain.x"].update(length=4), "arrays['terrain.x']", "length 4"),
    (lambda m: m["runs"][0].update(tokens=["only-one"]), "runs[0].entropy", "tokens"),
    (lambda m: m.pop("runs"), "manifest", "missing required field 'runs'"),
    (lambda m: m.update(version=2), "version", "must be 1"),
    (lambda m: m.update(kind="mystery"), "kind", "unknown kind"),
    (lambda m: m["runs"][0].update(label=7), "runs[0].label", "must be string"),
    (lambda m: m.update(schema="mottled-scene/9"), "schema", "must be"),
])
def test_each_way_to_break_a_scene_is_named(scene_bytes, edit, where, needle):
    errs = _errors(_rewrite(scene_bytes, edit))
    assert any(where in e.where and needle in e.message for e in errs), [str(e) for e in errs]


def test_trajectory_checks(traj_bytes):
    errs = _errors(_rewrite(traj_bytes, lambda m: m["tokens"].append("extra")))
    assert any("tokens but hidden has" in e.message for e in errs)
    # non-finite hidden states: the one array whose values are checked
    m = _manifest(traj_bytes)
    ref = m["arrays"]["hidden"]
    mlen = struct.unpack_from("<I", traj_bytes, 8)[0]
    lo = 12 + mlen + ref["offset"]
    broken = traj_bytes[:lo] + struct.pack("<f", float("nan")) + traj_bytes[lo + 4:]
    assert any("NaN" in e.message for e in _errors(broken))


def test_container_level_damage(scene_bytes):
    assert "bad magic" in _errors(b"hello world!")[0].message
    assert "runs past the end" in _errors(scene_bytes[:40])[0].message
    assert "past the end of the blob" in " ".join(e.message for e in _errors(scene_bytes[:-8]))
    trailing = _errors(scene_bytes + b"garbage!")
    assert "trailing byte" in trailing[0].message
    assert _errors(scene_bytes + b"\x00" * 7) == []        # padding is fine


def test_old_files_without_a_schema_id_warn_not_fail(scene_bytes):
    old = _rewrite(scene_bytes, lambda m: m.pop("schema"))
    problems = mtjschema.validate_bytes(old)
    assert [p.level for p in problems] == ["warning"]
    assert _errors(old, strict=True)


def test_readers_still_load_old_and_new_files(scene_bytes):
    old = _rewrite(scene_bytes, lambda m: m.pop("schema"))
    assert statefile.load_scene(io.BytesIO(old))["runs"]
    assert statefile.load_scene(io.BytesIO(scene_bytes))["schema"] == "mottled-scene/1"


def test_truncated_header_is_named_as_truncated():
    import mtjschema
    problems = mtjschema.validate_bytes(b"MTRJjunk", strict=False)
    assert any("truncated" in p.message for p in problems)
